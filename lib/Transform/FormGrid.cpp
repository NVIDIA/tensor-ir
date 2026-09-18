// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Layout propagation describes every tensor source as a logical strided view
// in a normalized iteration-space carrier. This pass consumes a function whose
// buffer boundary is already explicit (produced by tir-bufferize) and tiles its
// tensor computation onto a 1D GPU block grid.
//
// Key responsibilities of this pass:
// 1. Grid creation: Establishes an scf.forall loop mapped to #gpu.block<x>
//    covering the total number of tiles in the flattened iteration space.
// 2. Delinearization: Converts the 1D block index into multi-dimensional tile
//    coordinates using column-major ordering.
// 3. Tile materialization: Recursively traces backwards from each boundary StoreOp
//    to produce fixed-size tile values:
//    - Retargets untiled whole-tensor LoadOps to tiled LoadOps with concrete
//      offsets, sizes, strides, alignments, and tile slice coordinates.
//    - Folds layout-modifying operations (ReshapeOp, TransposeOp, BroadcastOp,
//      SliceOp) directly into access descriptors, eliminating them from the
//      tiled computation DAG.
//    - Evaluates ConcatenateOps into nested scf.if branches partitioned along
//      the concatenation dimension.
//    - Materializes IotaOps by composing local tile coordinates with global
//      block offsets.
//    - Tiles PointwiseOps by cloning them with concrete tile result types.
//    - Prepares coarse tiles for ReduceOps and MatmulOps, attaching formation
//      metadata for the downstream TileReductionsPass.
// 4. Cleanup: Erases the original untiled operations and strips analysis-only
//    attributes (layouts, iteration spaces, tile sizes) so downstream passes
//    operate on explicit SSA values and types.

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"
#include "tensor_ir/Transform/Passes.h" // IWYU pragma: keep
#include "tensor_ir/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DEF_FORMGRIDPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace {

namespace tcg = mlir::nv_tensor_ir::tcutegen;

/// Physical buffer pointer and descriptor metadata extracted from an untiled
/// boundary LoadOp or StoreOp.
///
/// Holds both the mixed static/dynamic attributes and SSA values from the
/// boundary operation, along with a flattened list of dynamic descriptor values
/// used by mapDynamicValues to resolve dynamic layout extents and strides.
struct BoundaryAccess {
  /// Base pointer to the underlying memory allocation.
  Value base;
  /// Mixed static (IntegerAttr) and dynamic (SSA Value) buffer dimension sizes.
  SmallVector<OpFoldResult> sizes;
  /// Mixed static (IntegerAttr) and dynamic (SSA Value) buffer dimension strides.
  SmallVector<OpFoldResult> strides;
  /// Base offset into the buffer, in elements.
  OpFoldResult offset;
  /// Known memory alignment in bytes.
  IntegerAttr alignment;
  /// Flattened sequence of dynamic sizes followed by dynamic strides, indexed
  /// by the dynamic-value mapping of a TensorSourceAttr.
  SmallVector<Value> dynamicValues;
  /// Element type of the stored or loaded tensor.
  Type elementType;
};

/// Synthesize a default TensorSourceAttr from a boundary access descriptor.
///
/// Used when an explicit output view attribute (result_views) is omitted on the
/// function. Encodes the boundary access's static and dynamic sizes/strides into
/// a CuTe Layout and derives the corresponding dynamic value mapping.
static TensorSourceAttr buildTensorSource(MLIRContext *context,
                                          const BoundaryAccess &access) {
  tcg::Shape shape;
  for (OpFoldResult size : access.sizes) {
    if (auto attr = size.dyn_cast<Attribute>()) {
      shape.append(cast<IntegerAttr>(attr).getInt());
    } else {
      shape.appendDynamic();
    }
  }
  tcg::Stride stride;
  for (OpFoldResult value : access.strides) {
    if (auto attr = value.dyn_cast<Attribute>()) {
      stride.append(cast<IntegerAttr>(attr).getInt());
    } else {
      stride.appendDynamic();
    }
  }
  tcg::Layout layout(shape, stride);
  int64_t offset =
      isa<Attribute>(access.offset)
          ? cast<IntegerAttr>(cast<Attribute>(access.offset)).getInt()
          : ShapedType::kDynamic;
  return TensorSourceAttr::get(context, /*tensorId=*/-1, offset,
                               layout.toString(),
                               getDynamicValueMapping(layout));
}

/// Unpack the layout provenance for each operand of an operation.
///
/// Layout propagation annotates each operation with a layout source describing
/// how its results are derived from input tensors. Different operation kinds
/// represent operand provenance differently:
/// - Pointwise ops with >1 operands: CompositeSourceAttr containing one layout
///   per operand.
/// - ConcatenateOp: ConcatSourceAttr containing the layouts of the concatenated
///   sub-tensors.
/// - ReduceOp / ReduceUDOp: ReductionSourceAttr containing the operand source
///   layout (or a composite if multiple inputs).
/// - MatmulOp: MatmulSourceAttr containing LHS, RHS, and optional accumulator.
/// - Single-operand / uniform ops: The operation's own layout is shared across
///   operands.
static FailureOr<SmallVector<LayoutSourceAttrInterface>>
getOperandLayouts(Operation *op) {
  auto layout = op->getAttrOfType<LayoutSourceAttrInterface>(
      TensorIRDialect::getLayoutAttrName());
  if (!layout) {
    return op->emitError("missing layout attribute; run graph-splitting "
                         "before tir-form-grid");
  }

  if (isa<PointwiseOpInterface>(op) && op->getNumOperands() > 1) {
    auto composite = dyn_cast<CompositeSourceAttr>(layout);
    if (!composite || composite.size() != op->getNumOperands()) {
      return op->emitError("expected one composite layout source per operand");
    }
    return SmallVector<LayoutSourceAttrInterface>(composite.getSources());
  }
  if (isa<ConcatenateOp>(op)) {
    auto concat = dyn_cast<ConcatSourceAttr>(layout);
    if (!concat) {
      return op->emitError("expected concatenation layout");
    }
    return SmallVector<LayoutSourceAttrInterface>(concat.getSources());
  }
  if (auto reduction = dyn_cast<ReductionSourceAttr>(layout);
      reduction && isa<ReduceOp>(op)) {
    return SmallVector<LayoutSourceAttrInterface>{reduction.getSource()};
  }
  if (auto reduction = dyn_cast<ReductionSourceAttr>(layout);
      reduction && isa<ReduceUDOp>(op)) {
    if (op->getNumOperands() > 1) {
      auto composite = dyn_cast<CompositeSourceAttr>(reduction.getSource());
      if (!composite || composite.size() != op->getNumOperands()) {
        return op->emitError(
            "expected one reduction source layout per operand");
      }
      return SmallVector<LayoutSourceAttrInterface>(composite.getSources());
    }
    return SmallVector<LayoutSourceAttrInterface>{reduction.getSource()};
  }
  if (auto matmul = dyn_cast<MatmulSourceAttr>(layout);
      matmul && isa<MatmulOp>(op)) {
    SmallVector<LayoutSourceAttrInterface> result{matmul.getLhs(),
                                                  matmul.getRhs()};
    if (op->getNumOperands() == 3) {
      result.push_back(layout);
    }
    return result;
  }
  return SmallVector<LayoutSourceAttrInterface>(op->getNumOperands(), layout);
}

/// Extract a single concrete TensorSourceAttr from a layout source attribute.
///
/// Physical memory loads can only target a single underlying tensor source. If
/// the layout is a single-element composite, unwraps the nested source.
static TensorSourceAttr
getSingleTensorSource(LayoutSourceAttrInterface layout) {
  if (auto source = dyn_cast<TensorSourceAttr>(layout)) {
    return source;
  }
  if (auto composite = dyn_cast<CompositeSourceAttr>(layout);
      composite && composite.size() == 1) {
    return getSingleTensorSource(composite.getSource(0));
  }
  return {};
}

/// Strip analysis-only attributes from an operation.
///
/// Attributes describing layout propagation, iteration spaces, tile sizes, and
/// candidate sets are compiler annotations consumed during grid formation.
/// They must not survive into downstream lowering passes, which expect purely
/// concrete SSA operations with explicit types.
static void removeAnalysisAttrs(Operation *op) {
  op->removeAttr(TensorIRDialect::getLayoutAttrName());
  op->removeAttr(TensorIRDialect::getIterSpaceMapAttrName());
  op->removeAttr(TensorIRDialect::getIterSpaceIdAttrName());
  op->removeAttr(TensorIRDialect::getIterSpaceIdsAttrName());
  op->removeAttr(TensorIRDialect::getIterSpaceDimDomainsAttrName());
  op->removeAttr(TensorIRDialect::getIterationSpaceAttrName());
  op->removeAttr(TensorIRDialect::getResultLayoutsAttrName());
  op->removeAttr(TensorIRDialect::getResultViewsAttrName());
  op->removeAttr(TensorIRDialect::getTileSizeAttrName());
  op->removeAttr(TensorIRDialect::getTileCandidatesAttrName());
}

/// Resolve the runtime SSA values required by a layout's dynamic dimensions.
///
/// A TensorSourceAttr stores a dynamic-value mapping array whose entries index
/// into the boundary access's flattened dynamicValues list (sizes then strides).
/// This function validates those indices and returns the corresponding SSA values
/// in layout order.
static FailureOr<SmallVector<Value>>
mapDynamicValues(TensorSourceAttr source, const BoundaryAccess &access,
                 Operation *diagnosticOp) {
  SmallVector<Value> values;
  ArrayRef<int32_t> mapping = source.getDynamicValueMapping();
  values.reserve(mapping.size());
  for (int32_t index : mapping) {
    if (index < 0 ||
        static_cast<size_t>(index) >= access.dynamicValues.size()) {
      return diagnosticOp->emitError()
             << "layout dynamic-value mapping index " << index
             << " is out of range for tensor #" << source.getTensorId()
             << " with " << access.dynamicValues.size()
             << " dynamic descriptor value(s)";
    }
    values.push_back(access.dynamicValues[index]);
  }
  return values;
}

/// Finalized memory access parameters ready to be attached to a tiled LoadOp or
/// StoreOp.
struct AccessMetadata {
  /// Base offset in elements (including any logical source offset).
  OpFoldResult offset;
  /// Tensor extents in elements for each dimension.
  SmallVector<OpFoldResult> sizes;
  /// Tensor strides in elements for each dimension.
  SmallVector<OpFoldResult> strides;
  /// Memory alignment in bytes, adjusted for any non-zero byte offset.
  IntegerAttr alignment;
};

/// Extract descriptor metadata from a boundary LoadOp or StoreOp.
///
/// Flattens the operation's dynamic sizes and strides into a single dynamicValues
/// vector so they can be indexed by TensorSourceAttr's dynamic-value mapping.
template <typename OpTy>
static BoundaryAccess getBoundaryAccess(OpTy op) {
  SmallVector<Value> dynamicValues(op.getSizes());
  llvm::append_range(dynamicValues, op.getStrides());
  return {op.getBase(),
          op.getMixedSizes(),
          op.getMixedStrides(),
          op.getMixedOffset(),
          op.getAlignmentAttr(),
          std::move(dynamicValues),
          op.getTile().getType().getElementType()};
}

/// Decoded logical view of a tensor after resolving its CuTe layout against the
/// boundary access's dynamic runtime values.
struct LogicalDescriptor {
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;

  /// Extract the dynamic SSA values from the decoded sizes, returning null
  /// Value for static dimensions.
  SmallVector<Value> getDynamicExtents() const {
    return llvm::map_to_vector(
        sizes, [](OpFoldResult size) { return size.dyn_cast<Value>(); });
  }
};

/// Decode a TensorSourceAttr's CuTe layout into concrete OpFoldResult sizes and
/// strides.
///
/// Translates static extents and strides into IndexAttr, sets broadcast
/// dimensions (static stride == 0) to extent 1, and matches dynamic layout
/// positions against runtime SSA values extracted by mapDynamicValues.
/// Verifies that all expected dynamic values are consumed without extras.
static FailureOr<LogicalDescriptor>
decodeLogicalDescriptor(TensorSourceAttr source, const BoundaryAccess &access,
                        Operation *diagnosticOp) {
  auto logical = source.getCuteLayout();
  size_t rank = tcg::rank(logical);

  MLIR_ASSIGN_OR_RETURN(SmallVector<Value> dynamicValues,
                        mapDynamicValues(source, access, diagnosticOp));
  auto dynamic = dynamicValues.begin();
  Builder builder(diagnosticOp->getContext());
  SmallVector<OpFoldResult> sizes;
  sizes.reserve(rank);
  for (size_t dimension = 0; dimension < rank; ++dimension) {
    auto extent = tcg::get(logical.shape(), dimension);
    auto stride = tcg::get(logical.stride(), dimension);
    if (tcg::is_static(extent)) {
      sizes.push_back(builder.getIndexAttr(tcg::static_size(extent)));
    } else if (tcg::is_static(stride) && stride.as_int() == 0) {
      sizes.push_back(builder.getIndexAttr(1));
    } else {
      if (dynamic == dynamicValues.end()) {
        return diagnosticOp->emitError()
               << "missing dynamic shape value for layout " << source;
      }
      sizes.push_back(*dynamic++);
    }
  }

  SmallVector<OpFoldResult> strides;
  strides.reserve(rank);
  for (size_t dimension = 0; dimension < rank; ++dimension) {
    auto stride = tcg::get(logical.stride(), dimension);
    if (tcg::is_static(stride)) {
      strides.push_back(builder.getIndexAttr(tcg::static_size(stride)));
    } else {
      if (dynamic == dynamicValues.end()) {
        return diagnosticOp->emitError()
               << "missing dynamic stride value for layout " << source;
      }
      strides.push_back(*dynamic++);
    }
  }
  if (dynamic != dynamicValues.end()) {
    return diagnosticOp->emitError()
           << "layout " << source << " has "
           << std::distance(dynamic, dynamicValues.end())
           << " unused dynamic descriptor value(s)";
  }

  return LogicalDescriptor{std::move(sizes), std::move(strides)};
}

/// Compute concrete access parameters (offset, sizes, strides, alignment) for a
/// tiled memory operation.
///
/// Combines the physical buffer offset with the logical source offset, checking
/// for static overflow or emitting dynamic index addition. Adjusts the base
/// buffer alignment using std::gcd with the byte offset.
static FailureOr<AccessMetadata>
createAccessMetadata(OpBuilder &builder, Operation *diagnosticOp,
                     TensorSourceAttr source, const BoundaryAccess &access,
                     const LogicalDescriptor &descriptor) {
  Location loc = diagnosticOp->getLoc();
  int64_t alignment = access.alignment.getInt();
  unsigned elementBytes =
      std::max(1u, access.elementType.getIntOrFloatBitWidth() / 8);
  int64_t offsetBytes = source.getOffset() * elementBytes;
  if (offsetBytes != 0) {
    alignment = std::gcd(alignment, std::abs(offsetBytes));
  }
  OpFoldResult offset = access.offset;
  if (auto attr = offset.dyn_cast<Attribute>()) {
    int64_t combinedOffset;
    if (llvm::AddOverflow(cast<IntegerAttr>(attr).getInt(), source.getOffset(),
                          combinedOffset)) {
      return emitError(loc, "logical buffer offset overflows index range");
    }
    offset = builder.getIndexAttr(combinedOffset);
  } else if (source.getOffset() != 0) {
    Value increment =
        arith::ConstantIndexOp::create(builder, loc, source.getOffset());
    offset = arith::AddIOp::create(builder, loc, cast<Value>(offset), increment)
                 .getResult();
  }
  return AccessMetadata{offset, descriptor.sizes, descriptor.strides,
                        builder.getI64IntegerAttr(alignment)};
}

/// Convert an OpFoldResult into an SSA Value, materializing an arith.constant
/// when the value is a static attribute.
static Value materializeIndex(OpBuilder &builder, Location loc,
                              OpFoldResult value) {
  if (auto dynamic = value.dyn_cast<Value>()) {
    return dynamic;
  }
  return arith::ConstantIndexOp::create(
      builder, loc, cast<IntegerAttr>(cast<Attribute>(value)).getInt());
}

/// Delinearize a 1D linear block index into multi-dimensional tile coordinates.
///
/// Uses column-major ordering (dimension 0 is fastest-varying):
///   coord[0] = linear % tileCounts[0]; linear /= tileCounts[0];
///   coord[1] = linear % tileCounts[1]; linear /= tileCounts[1];
///   ...
///   coord[N-1] = linear;
///
/// Dimensions with a tile count of 1 are optimized to constant 0 without
/// emitting modulo or division operations.
static SmallVector<Value>
delinearize(OpBuilder &builder, Location loc, Value linear,
            ArrayRef<OpFoldResult> dimensionTileCounts) {
  SmallVector<Value> result;
  result.reserve(dimensionTileCounts.size());
  Value zero;
  for (size_t dimension = 0; dimension < dimensionTileCounts.size();
       ++dimension) {
    if (dimension + 1 == dimensionTileCounts.size()) {
      result.push_back(linear);
      break;
    }
    if (getConstantIntValue(dimensionTileCounts[dimension]) == 1) {
      if (!zero) {
        zero = arith::ConstantIndexOp::create(builder, loc, 0);
      }
      result.push_back(zero);
      continue;
    }
    Value count =
        materializeIndex(builder, loc, dimensionTileCounts[dimension]);
    Value coordinate = arith::RemUIOp::create(builder, loc, linear, count);
    result.push_back(coordinate);
    linear = arith::DivUIOp::create(builder, loc, linear, count);
  }
  return result;
}

/// Determine the logical output view for a result store.
///
/// Returns the view from resultViews if provided by pass options or metadata.
/// Otherwise, synthesizes a layout from the physical output buffer and reshapes
/// it into the normalized iteration-space carrier shape.
static FailureOr<TensorSourceAttr>
getOutputView(Operation *diagnosticOp, const BoundaryAccess &output,
              size_t resultIndex, ArrayRef<int64_t> rootShape,
              ArrayRef<TensorSourceAttr> resultViews) {
  if (!resultViews.empty()) {
    return resultViews[resultIndex];
  }
  TensorSourceAttr physical =
      buildTensorSource(diagnosticOp->getContext(), output);
  auto reshaped =
      dyn_cast_if_present<TensorSourceAttr>(physical.reshape(rootShape));
  if (!reshaped) {
    return diagnosticOp->emitError(
        "cannot reshape the output descriptor into the normalized carrier");
  }
  return reshaped;
}

/// Geometric plan for materializing a tiled reduction operation.
///
/// Reduction operations in TensorIR combine non-reduced parallel output
/// dimensions with trailing contracting (reduction) dimensions.
struct ReductionTilePlan {
  /// Input tile shape: non-broadcast output dimensions followed by the
  /// full reduction shape.
  SmallVector<int64_t> inputShape;
  /// Indices of the output tile dimensions that supply coordinates to the input.
  SmallVector<size_t> inputDimensions;
  /// Layout source attributes for each reduction operand.
  SmallVector<LayoutSourceAttrInterface> operandLayouts;
  /// Zero-based indices of the contracting dimensions within inputShape.
  SmallVector<int32_t> reductionDimensions;
  /// Product of all contracting dimension extents.
  int64_t reductionExtent;
  /// Intermediate shape produced by the reduction op before broadcast/reshape
  /// (contracting dimensions collapsed to extent 1).
  SmallVector<int64_t> rawResultShape;
  /// Intermediate shape before restoring broadcast dimensions.
  SmallVector<int64_t> unbroadcastShape;
};

/// Validate and derive all tile shapes and coordinate mappings for a reduction.
///
/// Inspects the ReductionSourceAttr view to determine which dimensions are
/// preserved, broadcasted, or reduced. Derives inputTileShape, rawResultShape,
/// and unbroadcastShape without emitting any IR.
static FailureOr<ReductionTilePlan>
planReductionTile(Operation *op, ArrayRef<int64_t> outputTileShape) {
  auto layout = op->getAttrOfType<ReductionSourceAttr>(
      TensorIRDialect::getLayoutAttrName());
  if (!layout) {
    return op->emitError("expected reduction layout");
  }

  auto view = layout.getCuteLayout();
  size_t viewRank = tcg::rank(view);
  if (viewRank == 0 || outputTileShape.size() + 1 < viewRank) {
    return op->emitError("reduction layout does not match the output tile");
  }

  SmallVector<int64_t> inputTileShape;
  SmallVector<size_t> inputDimensions;
  for (size_t dimension = 0; dimension + 1 < viewRank; ++dimension) {
    auto stride = tcg::get(view, dimension).stride();
    if (!tcg::is_static(stride)) {
      return op->emitError("reduction view requires static dimension strides");
    }
    if (stride.as_int() != 0) {
      inputTileShape.push_back(outputTileShape[dimension]);
      inputDimensions.push_back(dimension);
    }
  }
  SmallVector<int64_t> reductionShape = layout.getReductionShape();
  inputTileShape.append(reductionShape);
  MLIR_ASSIGN_OR_RETURN(SmallVector<LayoutSourceAttrInterface> operandLayouts,
                        getOperandLayouts(op));
  if (operandLayouts.size() != op->getNumOperands()) {
    return op->emitError() << "expected " << op->getNumOperands()
                           << " reduction operand layouts, got "
                           << operandLayouts.size();
  }
  SmallVector<int32_t> reductionDimensions;
  for (size_t dimension = inputTileShape.size() - reductionShape.size();
       dimension < inputTileShape.size(); ++dimension) {
    reductionDimensions.push_back(static_cast<int32_t>(dimension));
  }
  int64_t reductionExtent = 1;
  for (int64_t extent : reductionShape) {
    if (llvm::MulOverflow(reductionExtent, extent, reductionExtent)) {
      return op->emitError("reduction extent overflows the index range");
    }
  }
  SmallVector<int64_t> rawResultShape(inputTileShape);
  std::fill(rawResultShape.end() - reductionShape.size(), rawResultShape.end(),
            1);
  SmallVector<int64_t> unbroadcastShape;
  size_t inputDimension = 0;
  for (size_t dimension = 0; dimension + 1 < viewRank; ++dimension) {
    bool broadcast = tcg::get(view, dimension).stride().as_int() == 0;
    unbroadcastShape.push_back(broadcast ? 1
                                         : inputTileShape[inputDimension++]);
  }

  return ReductionTilePlan{std::move(inputTileShape),
                           std::move(inputDimensions),
                           std::move(operandLayouts),
                           std::move(reductionDimensions),
                           reductionExtent,
                           std::move(rawResultShape),
                           std::move(unbroadcastShape)};
}

/// Tile geometry and coordinate mapping for one matrix-multiply operand (LHS or RHS).
struct MatmulOperandTilePlan {
  /// Operand tile shape before input reshaping.
  SmallVector<int64_t> shape;
  /// Zero-based indices of contracting dimensions within `shape`.
  SmallVector<int64_t> contractingDimensions;
  /// Output tile coordinate dimension feeding each operand axis, or -1 for
  /// coordinate 0 (contracting or full-span dimensions).
  SmallVector<int64_t> coordinateDimensions;
};

/// Complete geometric plan for materializing a tiled matrix multiplication.
///
/// Matmuls are planned in a canonical MMA shape [B, M, K] x [B, K, N] -> [B, M, N].
/// Input operands may be reshaped, and the MMA result is reshaped, permuted, and
/// broadcast back to the output tile shape.
struct MatmulTilePlan {
  MatmulSourceAttr layout;
  MatmulOperandTilePlan lhs;
  MatmulOperandTilePlan rhs;
  SmallVector<int64_t> contractionShape;
  SmallVector<int64_t> mmaLhsShape;
  SmallVector<int64_t> mmaRhsShape;
  SmallVector<int64_t> mmaOutputShape;
  SmallVector<int64_t> resultTileShape;
  SmallVector<int64_t> normalizedShape;
  SmallVector<int64_t> permutation;
};

/// Plan operand coordinates, canonical MMA shapes, and result reconstruction
/// without emitting IR.
///
/// Analyzes the MatmulSourceAttr layout to classify each dimension of the
/// iteration space into Broadcast, Batch, Left (M), or Right (N) based on stride
/// thresholds. Computes operand tile shapes and the permutation needed to
/// restore the canonical MMA output to the logical result tile shape.
static FailureOr<MatmulTilePlan>
planMatmulTile(MatmulOp matmul, ArrayRef<int64_t> outputTileShape) {
  auto layout = matmul->getAttrOfType<MatmulSourceAttr>(
      TensorIRDialect::getLayoutAttrName());
  if (!layout) {
    return matmul.emitError("expected matmul layout");
  }
  if (matmul.getAcc()) {
    return matmul.emitError(
        "custom accumulator is not supported by grid formation");
  }
  size_t iterationRank = outputTileShape.size();
  tcg::Layout view = layout.getCuteLayout();
  if (tcg::rank(view) != iterationRank + 1) {
    return matmul.emitError("matmul view rank does not match its tile");
  }
  FailureOr<SmallVector<size_t>> lhsDimensionMap = layout.getLhsDimensionMap();
  FailureOr<SmallVector<size_t>> rhsDimensionMap = layout.getRhsDimensionMap();
  if (failed(lhsDimensionMap) || failed(rhsDimensionMap)) {
    return matmul.emitError("cannot derive matmul operand dimension maps");
  }

  SmallVector<int64_t> contractionShape = layout.getContractingShape();
  if (contractionShape.empty() ||
      llvm::any_of(contractionShape,
                   [](int64_t extent) { return extent <= 0; })) {
    return matmul.emitError(
        "matmul requires positive static contracting extents");
  }
  auto planOperandTile =
      [&](ArrayRef<size_t> dimensionMap) -> FailureOr<MatmulOperandTilePlan> {
    MatmulOperandTilePlan plan;
    auto &sourceShape = plan.shape;
    auto &contractingDimensions = plan.contractingDimensions;
    auto &operandCoordinates = plan.coordinateDimensions;
    for (size_t mappedDimension : dimensionMap) {
      if (mappedDimension < iterationRank) {
        sourceShape.push_back(outputTileShape[mappedDimension]);
        int64_t extent = layout.getShape()[mappedDimension];
        if (!ShapedType::isDynamic(extent) &&
            extent <= outputTileShape[mappedDimension]) {
          operandCoordinates.push_back(-1);
        } else {
          operandCoordinates.push_back(mappedDimension);
        }
        continue;
      }
      size_t contractionDimension = mappedDimension - iterationRank;
      if (contractionDimension >= contractionShape.size()) {
        return matmul.emitError(
            "matmul operand dimension map exceeds the contraction rank");
      }
      sourceShape.push_back(contractionShape[contractionDimension]);
      contractingDimensions.push_back(sourceShape.size() - 1);
      operandCoordinates.push_back(-1);
    }
    return plan;
  };

  MLIR_ASSIGN_OR_RETURN(MatmulOperandTilePlan lhs,
                        planOperandTile(*lhsDimensionMap));
  MLIR_ASSIGN_OR_RETURN(MatmulOperandTilePlan rhs,
                        planOperandTile(*rhsDimensionMap));
  if (lhs.contractingDimensions.size() != contractionShape.size() ||
      rhs.contractingDimensions.size() != contractionShape.size()) {
    return matmul.emitError(
        "matmul operand maps do not cover every contracting dimension");
  }

  enum class DimensionKind { Broadcast, Batch, Left, Right };
  struct DimensionInfo {
    DimensionKind kind;
    size_t index;
    int64_t stride;
  };
  SmallVector<DimensionInfo> dimensions;
  int64_t tileB = 1;
  int64_t tileM = 1;
  int64_t tileN = 1;
  int64_t batchStride = layout.getM() * layout.getN() * layout.getK();
  int64_t lhsStride = layout.getN() * layout.getK();
  for (size_t dimension = 0; dimension < iterationRank; ++dimension) {
    auto stride = tcg::get(view, dimension).stride();
    if (!tcg::is_static(stride)) {
      return matmul.emitError("matmul view requires static strides");
    }
    int64_t value = stride.as_int();
    if (value == 0) {
      dimensions.push_back({DimensionKind::Broadcast, dimension, value});
    } else if (value >= batchStride) {
      dimensions.push_back({DimensionKind::Batch, dimension, value});
      tileB *= outputTileShape[dimension];
    } else if (value >= lhsStride) {
      dimensions.push_back({DimensionKind::Left, dimension, value});
      tileM *= outputTileShape[dimension];
    } else {
      dimensions.push_back({DimensionKind::Right, dimension, value});
      tileN *= outputTileShape[dimension];
    }
  }
  int64_t tileK = llvm::product_of(contractionShape);
  SmallVector<int64_t> mmaLhsShape{tileM, tileK};
  SmallVector<int64_t> mmaRhsShape{tileK, tileN};
  SmallVector<int64_t> mmaOutputShape{tileM, tileN};
  if (tileB > 1) {
    mmaLhsShape.insert(mmaLhsShape.begin(), tileB);
    mmaRhsShape.insert(mmaRhsShape.begin(), tileB);
    mmaOutputShape.insert(mmaOutputShape.begin(), tileB);
  }

  SmallVector<int64_t> resultTileShape(outputTileShape);
  for (const DimensionInfo &dimension : dimensions) {
    if (dimension.kind == DimensionKind::Broadcast) {
      resultTileShape[dimension.index] = 1;
    }
  }
  llvm::sort(dimensions,
             [](const DimensionInfo &lhs, const DimensionInfo &rhs) {
               if (lhs.kind != rhs.kind) {
                 return lhs.kind < rhs.kind;
               }
               return lhs.stride > rhs.stride;
             });
  SmallVector<int64_t> normalizedShape;
  SmallVector<int64_t> permutation(dimensions.size());
  for (auto [position, dimension] : llvm::enumerate(dimensions)) {
    normalizedShape.push_back(resultTileShape[dimension.index]);
    permutation[dimension.index] = static_cast<int64_t>(position);
  }

  return MatmulTilePlan{layout,
                        std::move(lhs),
                        std::move(rhs),
                        std::move(contractionShape),
                        std::move(mmaLhsShape),
                        std::move(mmaRhsShape),
                        std::move(mmaOutputShape),
                        std::move(resultTileShape),
                        std::move(normalizedShape),
                        std::move(permutation)};
}

/// Memoization cache for recursively materialized tile values.
///
/// TensorIR graphs frequently exhibit reconvergent dataflow (e.g. an input or
/// intermediate value consumed by multiple operations). The cache ensures each
/// tile is materialized only once per (source Value, layout, coordinates, tileShape)
/// tuple.
///
/// Scope provides RAII checkpointing for control-flow regions (such as the then/else
/// branches generated for ConcatenateOp). When exiting a branch, any tiles
/// materialized within that branch are popped from the cache so they cannot be
/// referenced outside their defining region, preventing SSA dominance violations.
class TileCache {
public:
  /// Values in an enclosing region remain visible, but values created in this
  /// scope must not escape into a sibling region or its parent.
  class Scope {
  public:
    explicit Scope(TileCache &cache)
        : cache(cache), checkpoint(cache.entries.size()) {}
    ~Scope() { cache.entries.resize(checkpoint); }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

  private:
    TileCache &cache;
    size_t checkpoint;
  };

  Value lookup(Value source, Attribute layout, ArrayRef<Value> coordinates,
               ArrayRef<int64_t> tileShape) const {
    for (const Entry &entry : entries) {
      if (entry.source == source && entry.layout == layout &&
          llvm::equal(entry.coordinates, coordinates) &&
          llvm::equal(entry.tileShape, tileShape)) {
        return entry.tile;
      }
    }
    return {};
  }

  void insert(Value source, Attribute layout, ArrayRef<Value> coordinates,
              ArrayRef<int64_t> tileShape, Value tile) {
    entries.push_back({source, layout, SmallVector<Value>(coordinates),
                       SmallVector<int64_t>(tileShape), tile});
  }

private:
  struct Entry {
    Value source;
    Attribute layout;
    SmallVector<Value> coordinates;
    SmallVector<int64_t> tileShape;
    Value tile;
  };
  SmallVector<Entry> entries;
};

/// Recursively traces the untiled TensorIR computation backwards from a result
/// store and materializes fixed-size tile operations inside the GPU block grid.
///
/// Traversal is backward-recursive: starting from StoreOp tile operands, each
/// operation requests its inputs at specific tile coordinates and shapes.
/// This ensures operations inside control-flow branches (e.g. ConcatenateOp)
/// are materialized strictly inside their respective branches rather than
/// speculated before them.
class TileMaterializer {
public:
  /// Recursively materialize a tile of `value` conforming to `expectedLayout`
  /// at the given `coordinates` with shape `tileShape`.
  FailureOr<Value> materialize(Value value,
                               LayoutSourceAttrInterface expectedLayout,
                               ArrayRef<Value> coordinates,
                               ArrayRef<int64_t> tileShape,
                               OpBuilder &builder) {
    if (!isa<TensorType>(value.getType())) {
      return materializeScalar(value, builder);
    }

    if (auto load = value.getDefiningOp<LoadOp>()) {
      return loadSource(load, expectedLayout, coordinates, tileShape, builder);
    }
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      return emitError(argument.getLoc())
             << "bufferized TensorIR functions cannot have tensor arguments";
    }

    if (Value found = computedValues.lookup(value, /*layout=*/{}, coordinates,
                                            tileShape)) {
      return found;
    }

    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != value.getParentBlock()) {
      return emitError(value.getLoc())
             << "tiled value is not defined in the TensorIR graph body";
    }

    if (auto concat = dyn_cast<ConcatenateOp>(op)) {
      MLIR_ASSIGN_OR_RETURN(
          Value result,
          materializeConcatenate(concat, coordinates, tileShape, builder));
      computedValues.insert(value, /*layout=*/{}, coordinates, tileShape,
                            result);
      return result;
    }
    if (auto matmul = dyn_cast<MatmulOp>(op)) {
      return materializeMatmul(matmul, coordinates, tileShape, builder);
    }
    if (isa<ReduceOp, ReduceUDOp>(op)) {
      return materializeReduction(value, coordinates, tileShape, builder);
    }
    if (auto iota = dyn_cast<IotaOp>(op)) {
      MLIR_ASSIGN_OR_RETURN(
          Value result, materializeIota(iota, coordinates, tileShape, builder));
      computedValues.insert(value, /*layout=*/{}, coordinates, tileShape,
                            result);
      return result;
    }

    MLIR_ASSIGN_OR_RETURN(SmallVector<LayoutSourceAttrInterface> operandLayouts,
                          getOperandLayouts(op));
    if (operandLayouts.size() != op->getNumOperands()) {
      return op->emitError()
             << "expected " << op->getNumOperands() << " operand layouts, got "
             << operandLayouts.size();
    }

    // These operations only modify logical addressing. Their analyzed layout
    // is passed to the underlying source load, so no tile compute remains.
    if (isa<ReshapeOp, BroadcastOp, TransposeOp, SliceOp>(op)) {
      MLIR_ASSIGN_OR_RETURN(
          Value result, materialize(op->getOperand(0), operandLayouts.front(),
                                    coordinates, tileShape, builder));
      computedValues.insert(value, /*layout=*/{}, coordinates, tileShape,
                            result);
      return result;
    }

    IRMapping mapping;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!isa<TensorType>(operand.getType())) {
        MLIR_ASSIGN_OR_RETURN(Value scalar,
                              materializeScalar(operand, builder));
        mapping.map(operand, scalar);
        continue;
      }
      if (!operandLayouts[index]) {
        return op->emitError() << "tensor operand #" << index
                               << " has no iteration-space layout";
      }
      MLIR_ASSIGN_OR_RETURN(Value tiledOperand,
                            materialize(operand, operandLayouts[index],
                                        coordinates, tileShape, builder));
      mapping.map(operand, tiledOperand);
    }

    if (auto constant = dyn_cast<ConstantOp>(op)) {
      auto original = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!original || !original.isSplat()) {
        return constant.emitError(
            "grid formation requires tensor constants to be splats");
      }
      Type elementType = cast<TensorType>(constant.getType()).getElementType();
      auto resultType = RankedTensorType::get(tileShape, elementType);
      DenseElementsAttr constantValue =
          elementType.isFloat()
              ? DenseElementsAttr::get(resultType,
                                       original.getSplatValue<APFloat>())
              : DenseElementsAttr::get(resultType,
                                       original.getSplatValue<APInt>());
      Value result = ConstantOp::create(builder, constant.getLoc(), resultType,
                                        cast<TypedAttr>(constantValue));
      computedValues.insert(value, /*layout=*/{}, coordinates, tileShape,
                            result);
      return result;
    }

    Operation *clone = builder.clone(*op, mapping);
    removeAnalysisAttrs(clone);
    for (auto [original, result] :
         llvm::zip(op->getResults(), clone->getResults())) {
      if (auto tensorType = dyn_cast<TensorType>(original.getType())) {
        result.setType(
            RankedTensorType::get(tileShape, tensorType.getElementType()));
      }
    }
    auto result = dyn_cast<OpResult>(value);
    if (!result || result.getResultNumber() >= clone->getNumResults()) {
      clone->erase();
      return op->emitError("expected a value-producing TensorIR compute op");
    }
    for (auto [original, cloned] :
         llvm::zip(op->getResults(), clone->getResults())) {
      computedValues.insert(original, /*layout=*/{}, coordinates, tileShape,
                            cloned);
    }
    return computedValues.lookup(value, /*layout=*/{}, coordinates, tileShape);
  }

private:
  /// Materialize an IotaOp tile.
  ///
  /// Computes both the intra-tile index variations and the inter-tile global
  /// block offsets. Intra-tile values are generated via IotaOp along each
  /// dimension and scaled by the dimension's stride. Inter-tile offsets are
  /// computed as `coordinates[dim] * tileShape[dim] * stride`, summed into
  /// a scalar, splatted across the tile, and added to the local indices.
  /// Floating-point iotas are generated as unsigned integer and converted.
  FailureOr<Value> materializeIota(IotaOp iota, ArrayRef<Value> coordinates,
                                   ArrayRef<int64_t> tileShape,
                                   OpBuilder &builder) {
    auto layout = iota->getAttrOfType<TensorSourceAttr>(
        TensorIRDialect::getLayoutAttrName());
    if (!layout || layout.getShape().size() != tileShape.size() ||
        coordinates.size() != tileShape.size()) {
      return iota.emitError(
          "iota layout does not match the normalized tile rank");
    }

    Location loc = iota.getLoc();
    Type elementType = iota.getType().getElementType();
    Type iotaElementType = elementType;
    if (isa<FloatType>(elementType)) {
      iotaElementType =
          IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned);
    }
    auto iotaType = RankedTensorType::get(tileShape, iotaElementType);
    Value local;
    Value globalOffset = arith::ConstantIndexOp::create(builder, loc, 0);
    auto strides = layout.getCuteLayout().stride();
    for (size_t dimension = 0; dimension < tileShape.size(); ++dimension) {
      auto stride = tcg::get(strides, dimension);
      if (!tcg::is_static(stride)) {
        return iota.emitError("iota layout requires static value strides");
      }
      int64_t scale = stride.as_int();
      if (scale == 0) {
        continue;
      }

      Value contribution = IotaOp::create(builder, loc, iotaType, dimension,
                                          /*dynamic_sizes=*/ValueRange{});
      if (scale != 1) {
        Attribute scalar = builder.getIntegerAttr(iotaElementType, scale);
        auto dense = cast<TypedAttr>(DenseElementsAttr::get(iotaType, scalar));
        Value factor =
            ConstantOp::create(builder, loc, iotaType, dense).getResult();
        contribution =
            MulOp::create(builder, loc, contribution, factor).getResult();
      }
      local = local
                  ? AddOp::create(builder, loc, local, contribution).getResult()
                  : contribution;

      int64_t tileScale;
      if (llvm::MulOverflow(tileShape[dimension], scale, tileScale)) {
        return iota.emitError("iota tile offset overflows index range");
      }
      Value scaleValue =
          arith::ConstantIndexOp::create(builder, loc, tileScale);
      Value offset = arith::MulIOp::create(builder, loc, coordinates[dimension],
                                           scaleValue);
      globalOffset = arith::AddIOp::create(builder, loc, globalOffset, offset);
    }
    if (!local) {
      return iota.emitError("iota layout has no varying dimension");
    }

    auto integer = dyn_cast<IntegerType>(iotaElementType);
    if (!integer) {
      return iota.emitError("iota requires an integer or floating-point type");
    }
    Type signless = builder.getIntegerType(integer.getWidth());
    Value scalarOffset =
        arith::IndexCastOp::create(builder, loc, signless, globalOffset);
    if (signless != iotaElementType) {
      scalarOffset = UnrealizedConversionCastOp::create(
                         builder, loc, TypeRange{iotaElementType}, scalarOffset)
                         .getResult(0);
    }
    Value offsetTile =
        SplatOp::create(builder, loc, iotaType, scalarOffset, ValueRange{});
    Value result = AddOp::create(builder, loc, local, offsetTile).getResult();
    if (isa<FloatType>(elementType)) {
      auto signlessType =
          RankedTensorType::get(tileShape, builder.getI32Type());
      result =
          UnrealizedConversionCastOp::create(builder, loc, signlessType, result)
              .getResult(0);
      return arith::UIToFPOp::create(
                 builder, loc, RankedTensorType::get(tileShape, elementType),
                 result)
          .getResult();
    }
    return result;
  }

  /// Clone pure, memory-effect-free scalar setup operations into the current
  /// insertion block.
  ///
  /// Function arguments and scalar constants or computations (e.g. dynamic
  /// shape calculations) are cloned on-demand and memoized in computedScalars.
  FailureOr<Value> materializeScalar(Value value, OpBuilder &builder) {
    if (isa<BlockArgument>(value)) {
      return value;
    }
    if (Value found = computedScalars.lookup(value)) {
      return found;
    }

    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != value.getParentBlock() ||
        !isMemoryEffectFree(op) || op->getNumRegions() != 0) {
      return emitError(value.getLoc())
             << "scalar value is neither a function argument nor pure setup";
    }

    IRMapping mapping;
    for (Value operand : op->getOperands()) {
      if (isa<TensorType>(operand.getType())) {
        return op->emitError("scalar setup cannot depend on a tensor operand");
      }
      MLIR_ASSIGN_OR_RETURN(Value scalar, materializeScalar(operand, builder));
      mapping.map(operand, scalar);
    }
    if (llvm::any_of(op->getResultTypes(),
                     [](Type type) { return isa<TensorType>(type); })) {
      return op->emitError("scalar setup produced a tensor result");
    }
    Operation *clone = builder.clone(*op, mapping);
    removeAnalysisAttrs(clone);
    for (auto [original, cloned] :
         llvm::zip(op->getResults(), clone->getResults())) {
      computedScalars.insert(original, cloned);
    }
    Value result = computedScalars.lookup(value);
    if (!result) {
      clone->erase();
      return op->emitError(
          "scalar setup did not reproduce the requested value");
    }
    return result;
  }

  /// Retarget an untiled boundary LoadOp to a tiled LoadOp.
  ///
  /// Decodes the logical layout descriptor, constructs AccessMetadata with
  /// adjusted offset and alignment, clamps coordinates to 0 for broadcast
  /// dimensions (stride == 0), and emits a fixed-shape tiled LoadOp.
  FailureOr<Value> loadSource(LoadOp load,
                              LayoutSourceAttrInterface expectedLayout,
                              ArrayRef<Value> coordinates,
                              ArrayRef<int64_t> tileShape, OpBuilder &builder) {
    TensorSourceAttr source = getSingleTensorSource(expectedLayout);
    if (!source) {
      return load.emitError("input access does not have one concrete source "
                            "layout");
    }
    if (Value found = loadedValues.lookup(load.getResult(), source, coordinates,
                                          tileShape)) {
      return found;
    }
    if (source.getShape().size() != tileShape.size()) {
      return load.emitError()
             << "iteration-space source rank " << source.getShape().size()
             << " does not match tile rank " << tileShape.size();
    }

    Location loc = load.getLoc();
    BoundaryAccess access = getBoundaryAccess(load);
    MLIR_ASSIGN_OR_RETURN(LogicalDescriptor descriptor,
                          decodeLogicalDescriptor(source, access, load));
    MLIR_ASSIGN_OR_RETURN(
        AccessMetadata metadata,
        createAccessMetadata(builder, load, source, access, descriptor));
    SmallVector<OpFoldResult> sourceCoordinates =
        getAsOpFoldResult(coordinates);
    auto sourceLayout = source.getCuteLayout();
    for (size_t dimension = 0; dimension < sourceCoordinates.size();
         ++dimension) {
      auto stride = tcg::get(sourceLayout.stride(), dimension);
      if (tcg::is_static(stride) && stride.as_int() == 0) {
        sourceCoordinates[dimension] = builder.getIndexAttr(0);
      }
    }
    auto resultType =
        RankedTensorType::get(tileShape, load.getType().getElementType());
    Value result = LoadOp::create(
        builder, loc, resultType, access.base, metadata.offset, metadata.sizes,
        metadata.strides, sourceCoordinates, metadata.alignment,
        /*padding=*/TilePaddingAttr{});
    loadedValues.insert(load.getResult(), source, coordinates, tileShape,
                        result);
    return result;
  }

  /// Materialize a ConcatenateOp by generating a cascade of scf.if operations.
  ///
  /// Concatenation partitions the iteration space along the concatenation
  /// dimension. For each source boundary, emits an `scf.if (coord < threshold)`.
  /// The `then` branch materializes the current source operand, while the `else`
  /// branch subtracts `threshold` from the coordinate and recurses on remaining
  /// sources. Uses CacheScope to isolate tiles materialized within each branch.
  FailureOr<Value> materializeConcatenate(ConcatenateOp concat,
                                          ArrayRef<Value> coordinates,
                                          ArrayRef<int64_t> tileShape,
                                          OpBuilder &builder) {
    auto layout = concat->getAttrOfType<ConcatSourceAttr>(
        TensorIRDialect::getLayoutAttrName());
    if (!layout) {
      return concat.emitError("expected concatenation layout");
    }
    if (layout.size() == 0) {
      return concat.emitError("concatenation layout has no sources");
    }
    int64_t dimension = layout.getDimension();
    if (dimension < 0 || static_cast<size_t>(dimension) >= tileShape.size()) {
      return concat.emitError()
             << "concatenation layout dimension " << dimension
             << " is outside tile rank " << tileShape.size();
    }
    if (tileShape[dimension] != 1) {
      return concat.emitError()
             << "concatenation dimension " << dimension
             << " must have tile size 1, but got " << tileShape[dimension];
    }

    ArrayRef<int32_t> argumentIndex = layout.getArgumentIndex();
    if (argumentIndex.empty() && layout.size() != concat.getNumOperands()) {
      return concat.emitError()
             << "concatenation layout has " << layout.size() << " sources for "
             << concat.getNumOperands() << " operands";
    }
    if (!argumentIndex.empty() && argumentIndex.size() != layout.size()) {
      return concat.emitError(
          "concatenation argument mapping does not match source count");
    }

    std::function<FailureOr<Value>(size_t, SmallVector<Value>)> emitBranch =
        [&](size_t sourceIndex,
            SmallVector<Value> branchCoordinates) -> FailureOr<Value> {
      size_t operandIndex =
          argumentIndex.empty()
              ? sourceIndex
              : static_cast<size_t>(argumentIndex[sourceIndex]);
      if (operandIndex >= concat.getNumOperands()) {
        return concat.emitError()
               << "concatenation argument index " << operandIndex
               << " refers to a non-existent operand";
      }
      LayoutSourceAttrInterface sourceLayout = layout.getSource(sourceIndex);
      if (sourceIndex + 1 == layout.size()) {
        return materialize(concat.getOperand(operandIndex), sourceLayout,
                           branchCoordinates, tileShape, builder);
      }

      int64_t threshold = sourceLayout.getShape()[dimension];
      if (ShapedType::isDynamic(threshold) || threshold <= 0) {
        return concat.emitError()
               << "concatenation source " << sourceIndex
               << " requires a positive static transition extent";
      }
      Location loc = concat.getLoc();
      Value bound = arith::ConstantIndexOp::create(builder, loc, threshold);
      Value beforeBoundary =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                branchCoordinates[dimension], bound);
      auto resultType = RankedTensorType::get(
          tileShape, cast<TensorType>(concat.getType()).getElementType());
      auto ifOp = scf::IfOp::create(builder, loc, TypeRange{resultType},
                                    beforeBoundary, /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard guard(builder);
        CacheScope scope(*this);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        MLIR_ASSIGN_OR_RETURN(Value thenValue,
                              materialize(concat.getOperand(operandIndex),
                                          sourceLayout, branchCoordinates,
                                          tileShape, builder));
        scf::YieldOp::create(builder, loc, thenValue);
      }
      {
        OpBuilder::InsertionGuard guard(builder);
        CacheScope scope(*this);
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        SmallVector<Value> nextCoordinates(branchCoordinates);
        nextCoordinates[dimension] = arith::SubIOp::create(
            builder, loc, branchCoordinates[dimension], bound);
        MLIR_ASSIGN_OR_RETURN(
            Value elseValue,
            emitBranch(sourceIndex + 1, std::move(nextCoordinates)));
        scf::YieldOp::create(builder, loc, elseValue);
      }
      return ifOp.getResult(0);
    };

    return emitBranch(/*sourceIndex=*/0, SmallVector<Value>(coordinates));
  }

  /// Materialize a tiled reduction (ReduceOp or ReduceUDOp).
  ///
  /// Materializes operand tiles with trailing contracting dimensions at the
  /// full reduction extent. Clones the reduction op with updated reduction
  /// dimensions and records `nv_tensor_ir.reduction_extent`. Reshapes and
  /// broadcasts the result tile if needed to match the requested outputTileShape.
  ///
  /// Note: Contracting dimensions remain coarse here; strip-mining them into
  /// accumulation loops is handled by the subsequent TileReductionsPass.
  FailureOr<Value> materializeReduction(Value requestedResult,
                                        ArrayRef<Value> coordinates,
                                        ArrayRef<int64_t> outputTileShape,
                                        OpBuilder &builder) {
    Operation *op = requestedResult.getDefiningOp();
    if (coordinates.size() != outputTileShape.size()) {
      return op->emitError("reduction layout does not match the output tile");
    }
    MLIR_ASSIGN_OR_RETURN(ReductionTilePlan plan,
                          planReductionTile(op, outputTileShape));
    SmallVector<Value> inputCoordinates;
    for (size_t dimension : plan.inputDimensions) {
      inputCoordinates.push_back(coordinates[dimension]);
    }
    for (size_t dimension = plan.inputDimensions.size();
         dimension < plan.inputShape.size(); ++dimension) {
      Value zero = arith::ConstantIndexOp::create(builder, op->getLoc(), 0);
      inputCoordinates.push_back(zero);
    }

    IRMapping mapping;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      MLIR_ASSIGN_OR_RETURN(Value tiledOperand,
                            materialize(operand, plan.operandLayouts[index],
                                        inputCoordinates, plan.inputShape,
                                        builder));
      mapping.map(operand, tiledOperand);
    }

    Operation *clone = builder.clone(*op, mapping);
    removeAnalysisAttrs(clone);
    // Reduction axes follow the non-reduced output axes in the input tile.
    if (auto reduce = dyn_cast<ReduceOp>(clone)) {
      reduce.setDimensions(plan.reductionDimensions);
    } else {
      cast<ReduceUDOp>(clone).setDimensions(plan.reductionDimensions);
    }
    clone->setAttr("nv_tensor_ir.reduction_extent",
                   builder.getI64IntegerAttr(plan.reductionExtent));

    for (auto [original, cloned] :
         llvm::zip(op->getResults(), clone->getResults())) {
      Type elementType = cast<TensorType>(original.getType()).getElementType();
      cloned.setType(RankedTensorType::get(plan.rawResultShape, elementType));
      Value result = cloned;
      if (plan.rawResultShape != plan.unbroadcastShape) {
        result = ReshapeOp::create(
            builder, op->getLoc(),
            RankedTensorType::get(plan.unbroadcastShape, elementType), result);
      }
      if (plan.unbroadcastShape != outputTileShape) {
        result = BroadcastOp::create(
            builder, op->getLoc(),
            RankedTensorType::get(outputTileShape, elementType), result);
      }
      computedValues.insert(original, /*layout=*/{}, coordinates,
                            outputTileShape, result);
    }
    return computedValues.lookup(requestedResult, /*layout=*/{}, coordinates,
                                 outputTileShape);
  }

  /// Materialize a tiled matrix multiplication (MatmulOp).
  ///
  /// Materializes LHS and RHS operand tiles, reshapes them to canonical MMA
  /// layout ([B, M, K] and [B, K, N]), and emits a tiled MatmulOp annotated
  /// with formation metadata attributes (contraction shape, source shapes,
  /// contracting dimensions). Applies reshape, transposition (permutation),
  /// and broadcasting to reconstruct the expected output tile shape.
  FailureOr<Value> materializeMatmul(MatmulOp matmul,
                                     ArrayRef<Value> coordinates,
                                     ArrayRef<int64_t> outputTileShape,
                                     OpBuilder &builder) {
    if (coordinates.size() != outputTileShape.size()) {
      return matmul.emitError(
          "matmul coordinates do not match the output tile rank");
    }
    MLIR_ASSIGN_OR_RETURN(MatmulTilePlan plan,
                          planMatmulTile(matmul, outputTileShape));
    auto buildOperandTile =
        [&](Value operand, LayoutSourceAttrInterface layout,
            const MatmulOperandTilePlan &operandPlan) -> FailureOr<Value> {
      SmallVector<Value> operandCoordinates;
      for (int64_t dimension : operandPlan.coordinateDimensions) {
        if (dimension < 0) {
          Value zero =
              arith::ConstantIndexOp::create(builder, matmul.getLoc(), 0);
          operandCoordinates.push_back(zero);
        } else {
          operandCoordinates.push_back(coordinates[dimension]);
        }
      }
      return materialize(operand, layout, operandCoordinates, operandPlan.shape,
                         builder);
    };
    MLIR_ASSIGN_OR_RETURN(
        Value lhs,
        buildOperandTile(matmul.getA(), plan.layout.getLhs(), plan.lhs));
    MLIR_ASSIGN_OR_RETURN(
        Value rhs,
        buildOperandTile(matmul.getB(), plan.layout.getRhs(), plan.rhs));

    auto reshapeInput = [&](Value input,
                            ArrayRef<int64_t> targetShape) -> Value {
      auto inputType = cast<RankedTensorType>(input.getType());
      if (inputType.getShape() == targetShape) {
        return input;
      }
      auto reshape = ReshapeOp::create(
          builder, matmul.getLoc(),
          RankedTensorType::get(targetShape, inputType.getElementType()),
          input);
      reshape->setAttr("nv_tensor_ir.matmul_input_reshape",
                       builder.getUnitAttr());
      return reshape.getResult();
    };
    lhs = reshapeInput(lhs, plan.mmaLhsShape);
    rhs = reshapeInput(rhs, plan.mmaRhsShape);

    Type elementType = matmul.getType().getElementType();
    auto mmaResultType =
        RankedTensorType::get(plan.mmaOutputShape, elementType);
    auto tiledMatmul =
        MatmulOp::create(builder, matmul.getLoc(), mmaResultType, lhs, rhs);
    tiledMatmul->setAttr("nv_tensor_ir.contraction_shape",
                         builder.getDenseI64ArrayAttr(plan.contractionShape));
    tiledMatmul->setAttr("nv_tensor_ir.lhs_source_shape",
                         builder.getDenseI64ArrayAttr(plan.lhs.shape));
    tiledMatmul->setAttr("nv_tensor_ir.rhs_source_shape",
                         builder.getDenseI64ArrayAttr(plan.rhs.shape));
    tiledMatmul->setAttr(
        "nv_tensor_ir.lhs_contracting_dimensions",
        builder.getDenseI64ArrayAttr(plan.lhs.contractingDimensions));
    tiledMatmul->setAttr(
        "nv_tensor_ir.rhs_contracting_dimensions",
        builder.getDenseI64ArrayAttr(plan.rhs.contractingDimensions));

    Value result = tiledMatmul;
    if (plan.normalizedShape != plan.mmaOutputShape) {
      result = ReshapeOp::create(
          builder, matmul.getLoc(),
          RankedTensorType::get(plan.normalizedShape, elementType), result);
    }
    bool transpose =
        llvm::any_of(llvm::enumerate(plan.permutation), [](auto entry) {
          return static_cast<int64_t>(entry.index()) != entry.value();
        });
    if (transpose) {
      result = TransposeOp::create(
          builder, matmul.getLoc(),
          RankedTensorType::get(plan.resultTileShape, elementType), result,
          plan.permutation);
    }
    if (plan.resultTileShape != outputTileShape) {
      result = BroadcastOp::create(
          builder, matmul.getLoc(),
          RankedTensorType::get(outputTileShape, elementType), result);
    }
    computedValues.insert(matmul.getResult(), /*layout=*/{}, coordinates,
                          outputTileShape, result);
    return result;
  }

  struct CacheScope {
    explicit CacheScope(TileMaterializer &materializer)
        : computed(materializer.computedValues),
          loaded(materializer.loadedValues),
          scalars(materializer.computedScalars) {}

    TileCache::Scope computed;
    TileCache::Scope loaded;
    llvm::ScopedHashTableScope<Value, Value> scalars;
  };

  llvm::ScopedHashTable<Value, Value> computedScalars;
  llvm::ScopedHashTableScope<Value, Value> scalarScope{computedScalars};
  TileCache computedValues;
  TileCache loadedValues;
};

/// Main driver for FormGridPass.
///
/// Execution proceeds in six phases:
/// 1. Validation: Verify the function has bufferized_program attribute, a single
///    entry block, valid outputs, iteration space, and tile size.
/// 2. Output view decoding: Resolve logical views and decode descriptors for
///    all boundary StoreOps. Collect dynamic extents if any dimension is dynamic.
/// 3. Grid sizing: Compute tile counts per dimension (ceil(extent / tile)) and
///    flatten into a 1D totalTiles count with static overflow checking.
/// 4. Launch construction: Emit a 1D scf.forall mapped to #gpu.block<x> at the
///    end of the entry block.
/// 5. Body materialization: Inside the forall body, delinearize the linear block
///    ID into multi-dimensional coordinates, run TileMaterializer backwards
///    from each StoreOp, and emit tiled StoreOps.
/// 6. Cleanup: Erase original untiled operations in reverse topological order
///    and remove compiler analysis attributes.
static LogicalResult formGrid(func::FuncOp function) {
  if (!function->hasAttr(TensorIRDialect::getBufferizedProgramAttrName())) {
    return success();
  }

  // === Phase 1: Validate function structure, boundary stores, and layout attributes ===
  if (function.isExternal() || !function.getBody().hasOneBlock()) {
    return function.emitError(
        "bufferized TensorIR program must define one entry block");
  }
  Block &entry = function.getBody().front();
  SmallVector<StoreOp> outputs;
  SmallVector<Operation *> untiledOps;
  for (Operation &op : entry) {
    if (auto store = dyn_cast<StoreOp>(op)) {
      outputs.push_back(store);
    }
    if (op.getName().getDialectNamespace() ==
        TensorIRDialect::getDialectNamespace()) {
      untiledOps.push_back(&op);
    }
  }
  if (outputs.empty()) {
    return function.emitError(
        "grid formation requires at least one TensorIR store");
  }

  auto rootLayout = function->getAttrOfType<LayoutSourceAttrInterface>(
      TensorIRDialect::getIterationSpaceAttrName());
  if (!rootLayout) {
    return function.emitError(
        "missing iteration_space attribute on bufferized program");
  }
  auto tileAttr = function->getAttrOfType<DenseI32ArrayAttr>(
      TensorIRDialect::getTileSizeAttrName());
  if (!tileAttr) {
    return function.emitError(
        "missing tile_size attribute on bufferized program");
  }
  SmallVector<int64_t> tileShape(tileAttr.asArrayRef().begin(),
                                 tileAttr.asArrayRef().end());
  SmallVector<int64_t> rootShape = rootLayout.getShape();
  if (tileShape.size() != rootShape.size() ||
      llvm::any_of(tileShape, [](int64_t size) { return size <= 0; })) {
    return function.emitError() << "tile_size must contain " << rootShape.size()
                                << " positive dimensions";
  }

  MLIR_ASSIGN_OR_RETURN(auto resultViews,
                        getResultViews(function.getOperation()));
  MLIR_ASSIGN_OR_RETURN(auto resultLayouts,
                        getResultLayouts(function.getOperation()));
  if (outputs.size() > 1) {
    if (resultLayouts.size() != outputs.size() ||
        resultViews.size() != outputs.size()) {
      return function.emitError()
             << "multi-output layout metadata must have " << outputs.size()
             << " entries, but result_layouts has " << resultLayouts.size()
             << " and result_views has " << resultViews.size();
    }
    for (auto [index, pair] :
         llvm::enumerate(llvm::zip_equal(resultLayouts, resultViews))) {
      auto [layout, view] = pair;
      if (layout.getShape() != rootShape || view.getShape() != rootShape) {
        return function.emitError()
               << "multi-output result " << index
               << " is not expressed in normalized carrier shape "
               << vectorToString(rootShape);
      }
    }
  } else if (!resultLayouts.empty() || !resultViews.empty()) {
    if (resultLayouts.size() != 1 || resultViews.size() != 1) {
      return function.emitError(
          "single-output result metadata must be absent or have one entry");
    }
    if (resultLayouts.front().getShape() != rootShape ||
        resultViews.front().getShape() != rootShape) {
      return function.emitError()
             << "single-output result metadata is not expressed in "
                "normalized carrier shape "
             << vectorToString(rootShape);
    }
  } else {
    resultLayouts.push_back(rootLayout);
  }

  // === Phase 2: Resolve logical views and decode descriptors for output stores ===
  SmallVector<BoundaryAccess> outputAccesses;
  SmallVector<TensorSourceAttr> outputViews;
  SmallVector<LogicalDescriptor> outputDescriptors;
  outputAccesses.reserve(outputs.size());
  outputViews.reserve(outputs.size());
  for (auto [index, output] : llvm::enumerate(outputs)) {
    outputAccesses.push_back(getBoundaryAccess(output));
    MLIR_ASSIGN_OR_RETURN(TensorSourceAttr view,
                          getOutputView(function, outputAccesses.back(), index,
                                        rootShape, resultViews));
    outputViews.push_back(view);
    MLIR_ASSIGN_OR_RETURN(
        LogicalDescriptor descriptor,
        decodeLogicalDescriptor(view, outputAccesses.back(), output));
    outputDescriptors.push_back(std::move(descriptor));
  }

  SmallVector<Value> dynamicRootExtents(rootShape.size());
  for (const LogicalDescriptor &descriptor : outputDescriptors) {
    SmallVector<Value> outputRootExtents = descriptor.getDynamicExtents();
    for (auto [rootExtent, outputExtent] :
         llvm::zip_equal(dynamicRootExtents, outputRootExtents)) {
      if (!rootExtent && outputExtent) {
        rootExtent = outputExtent;
      }
    }
  }

  // === Phase 3: Compute tile counts per dimension and flattened grid size ===
  // Buffer metadata for later results may be materialized after an earlier
  // store. Build the grid at the terminator so every boundary pointer and
  // dynamic descriptor value dominates the tiled body. The untiled TensorIR
  // computation is erased once materialization succeeds.
  OpBuilder entryBuilder(function.getBody().front().getTerminator());
  SmallVector<OpFoldResult> tileCounts;
  tileCounts.reserve(rootShape.size());
  for (auto [dimension, pair] :
       llvm::enumerate(llvm::zip_equal(rootShape, tileShape))) {
    auto [extent, tile] = pair;
    if (!ShapedType::isDynamic(extent)) {
      tileCounts.push_back(
          entryBuilder.getIndexAttr(llvm::divideCeil(extent, tile)));
      continue;
    }
    Value dynamicExtent = dynamicRootExtents[dimension];
    if (!dynamicExtent) {
      return function.emitError() << "normalized dimension " << dimension
                                  << " is dynamic but has no runtime extent";
    }
    Value adjusted = dynamicExtent;
    if (tile != 1) {
      Value increment = arith::ConstantIndexOp::create(
          entryBuilder, function.getLoc(), tile - 1);
      adjusted = arith::AddIOp::create(entryBuilder, function.getLoc(),
                                       adjusted, increment);
      Value divisor =
          arith::ConstantIndexOp::create(entryBuilder, function.getLoc(), tile);
      adjusted = arith::DivUIOp::create(entryBuilder, function.getLoc(),
                                        adjusted, divisor);
    }
    tileCounts.push_back(adjusted);
  }

  OpFoldResult totalTiles = entryBuilder.getIndexAttr(1);
  for (OpFoldResult count : tileCounts) {
    if (auto lhs = totalTiles.dyn_cast<Attribute>()) {
      if (auto rhs = count.dyn_cast<Attribute>()) {
        int64_t product;
        if (llvm::MulOverflow(cast<IntegerAttr>(lhs).getInt(),
                              cast<IntegerAttr>(rhs).getInt(), product)) {
          return function.emitError(
              "flattened static grid tile count overflows index range");
        }
        totalTiles = entryBuilder.getIndexAttr(product);
        continue;
      }
    }
    Value lhs = materializeIndex(entryBuilder, function.getLoc(), totalTiles);
    Value rhs = materializeIndex(entryBuilder, function.getLoc(), count);
    totalTiles =
        arith::MulIOp::create(entryBuilder, function.getLoc(), lhs, rhs)
            .getResult();
  }

  // === Phase 4: Construct 1D scf.forall grid mapped to #gpu.block<x> ===
  ArrayAttr mapping = entryBuilder.getArrayAttr({gpu::GPUBlockMappingAttr::get(
      function.getContext(), gpu::MappingId::DimX)});
  LogicalResult bodyStatus = success();
  scf::ForallOp::create(
      entryBuilder, function.getLoc(), ArrayRef<OpFoldResult>{totalTiles},
      /*outputs=*/ValueRange{}, mapping,
      [&](OpBuilder &builder, Location loc, ValueRange bodyArguments) {
        // Delinearize 1D block ID to multi-dimensional tile coordinates.
        SmallVector<Value> coordinates =
            delinearize(builder, loc, bodyArguments.front(), tileCounts);
        TileMaterializer materializer;

        // Recursively materialize tiled compute graph backwards from each store.
        for (auto [index, output] : llvm::enumerate(outputs)) {
          FailureOr<Value> tiledResult =
              materializer.materialize(output.getTile(), resultLayouts[index],
                                       coordinates, tileShape, builder);
          if (failed(tiledResult)) {
            bodyStatus = failure();
            return;
          }
          FailureOr<AccessMetadata> metadata = createAccessMetadata(
              builder, output, outputViews[index], outputAccesses[index],
              outputDescriptors[index]);
          if (failed(metadata)) {
            bodyStatus = failure();
            return;
          }
          StoreOp::create(builder, output.getLoc(), *tiledResult,
                          outputAccesses[index].base, metadata->offset,
                          metadata->sizes, metadata->strides,
                          getAsOpFoldResult(coordinates), metadata->alignment);
        }
        scf::InParallelOp::create(builder, loc);
      });
  if (failed(bodyStatus)) {
    return failure();
  }

  // === Phase 5: Clean up untiled operations and strip analysis attributes ===
  for (Operation *op : llvm::reverse(untiledOps)) {
    op->erase();
  }
  removeAnalysisAttrs(function);
  function->removeAttr(TensorIRDialect::getBufferizedProgramAttrName());
  return success();
}

struct FormGridPass : public impl::FormGridPassBase<FormGridPass> {
  void runOnOperation() override {
    if (failed(formGrid(getOperation()))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace
} // namespace mlir::nv_tensor_ir
