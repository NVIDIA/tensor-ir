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
// 3. Tile materialization: Recursively traces backward from each boundary
//    bufferization.materialize_in_destination to produce fixed-size tile
//    values:
//    - Replaces bufferization.to_tensor inputs with tiled LoadOps having
//      concrete offsets, sizes, strides, alignments, and tile slice
//      coordinates.
//    - Folds layout-modifying operations into access descriptors, then restores
//      broadcast dimensions with explicit tile broadcasts.
//    - Evaluates ConcatenateOps into nested scf.if branches partitioned along
//      the concatenation dimension.
//    - Materializes IotaOps by composing local tile coordinates with global
//      block offsets.
//    - Tiles PointwiseOps by cloning them with concrete tile result types.
//    - Prepares coarse tiles for ReduceOps and MatmulOps, attaching formation
//      metadata for the downstream TileReductionsPass.
// 4. Cleanup: Erases the bufferization bridges and dead untiled computation,
//    and strips analysis-only attributes (tile sizes) so downstream passes
//    operate on explicit SSA values and types.

#include "tensor_ir/Analysis/LayoutPropagation.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"
#include "tensor_ir/Transform/Passes.h" // IWYU pragma: keep
#include "tensor_ir/Utils/LayoutPermutation.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DEF_FORMGRIDPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace {

namespace tcg = mlir::nv_tensor_ir::tcutegen;

/// Physical buffer and descriptor metadata for a boundary memref argument.
///
/// Holds mixed static/dynamic buffer metadata and a flattened list of dynamic
/// descriptor values used by mapDynamicValues to resolve dynamic layout extents
/// and strides.
struct BoundaryAccess {
  /// Memref function argument bound to the boundary tensor.
  Value buffer;
  /// Mixed static (IntegerAttr) and dynamic (SSA Value) buffer dimension sizes.
  SmallVector<OpFoldResult> sizes;
  /// Mixed static (IntegerAttr) and dynamic (SSA Value) buffer dimension strides.
  SmallVector<OpFoldResult> strides;
  /// Known memory alignment in bytes.
  IntegerAttr alignment;
  /// Whether the backend may use TMA for accesses to the buffer.
  bool allowTma = true;
  /// Flattened sequence of dynamic sizes followed by dynamic strides, indexed
  /// by the dynamic-value mapping of a TensorSourceAttr.
  SmallVector<Value> dynamicValues;
  /// Element type of the buffer.
  Type elementType;
};

/// Synthesize a default TensorSourceAttr from a boundary access descriptor.
///
/// Used when layout normalization records no explicit output views. Encodes
/// the boundary access's static and dynamic sizes/strides into a CuTe Layout
/// and derives the corresponding dynamic value mapping.
static TensorSourceAttr buildTensorSource(MLIRContext *context,
                                          const BoundaryAccess &access) {
  auto type = cast<MemRefType>(access.buffer.getType());
  auto [strides, offset] = type.getStridesAndOffset();
  tcg::Layout layout(tcg::Shape(type.getShape()), tcg::Stride(strides));
  // Boundary buffers have statically zero offsets.
  return TensorSourceAttr::get(context, /*tensorId=*/-1, /*offset=*/0,
                               layout.toString(),
                               getDynamicValueMapping(layout));
}

/// Remap the operand layouts of a multi-source pointwise operation into the
/// requested iteration space.
///
/// Layout propagation assigns the operation the deduplicated union of its
/// operand sources. The requested layout lists the same sources, in the same
/// order, expressed in the iteration space. Each operand receives the
/// requested counterparts of its own propagated sources.
static FailureOr<SmallVector<LayoutSourceAttrInterface>>
remapPointwiseOperandLayouts(Operation *op, CompositeSourceAttr annotated,
                             LayoutSourceAttrInterface requested,
                             const LayoutPropagationAnalysis &propagation) {
  auto target = dyn_cast<CompositeSourceAttr>(requested);
  if (!target || target.size() != annotated.size()) {
    return op->emitError("expected matching composite layout");
  }

  ArrayRef<LayoutSourceAttrInterface> annotatedSources = annotated.getSources();
  SmallVector<LayoutSourceAttrInterface> operandLayouts;
  for (Value operand : op->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType())) {
      operandLayouts.push_back(nullptr);
      continue;
    }
    LayoutSourceAttrInterface operandLayout = propagation.getLayout(operand);
    if (!operandLayout) {
      return op->emitError("missing operand layout");
    }
    ArrayRef<LayoutSourceAttrInterface> operandSources(operandLayout);
    if (auto composite = dyn_cast<CompositeSourceAttr>(operandLayout)) {
      operandSources = composite.getSources();
    }

    SmallVector<LayoutSourceAttrInterface> remappedSources;
    for (LayoutSourceAttrInterface source : operandSources) {
      auto it = llvm::find(annotatedSources, source);
      if (it == annotatedSources.end()) {
        return op->emitError("composite layout mapping error");
      }
      remappedSources.push_back(
          target.getSource(std::distance(annotatedSources.begin(), it)));
    }
    operandLayouts.push_back(
        remappedSources.size() == 1
            ? remappedSources.front()
            : CompositeSourceAttr::get(op->getContext(), remappedSources));
  }
  return operandLayouts;
}

/// Derive the layout of each operand from the layout requested of `op`.
///
/// Requested layouts are expressed in the normalized iteration space, and a
/// value with several users may be requested in several layouts. The forward
/// layout of `op` only supplies operand provenance for multi-source pointwise
/// operations. Operands that need no tile (scalars, pruned concatenation
/// sources, and streaming scan_slice inputs) receive a null layout.
static FailureOr<SmallVector<LayoutSourceAttrInterface>>
inferOperandLayouts(Operation *op, LayoutSourceAttrInterface layout,
                    const LayoutPropagationAnalysis &propagation) {
  SmallVector<LayoutSourceAttrInterface> operandLayouts(op->getNumOperands());
  if (op->hasTrait<OpTrait::Elementwise>()) {
    auto annotated = dyn_cast_or_null<CompositeSourceAttr>(
        propagation.getLayout(op->getResult(0)));
    if (annotated && op->getNumOperands() > 1) {
      return remapPointwiseOperandLayouts(op, annotated, layout, propagation);
    }
    llvm::fill(operandLayouts, layout);
    return operandLayouts;
  }
  if (isa<ReshapeOp, TransposeOp, BroadcastOp, SliceOp>(op)) {
    operandLayouts.front() = layout;
    return operandLayouts;
  }
  if (isa<ConcatenateOp>(op)) {
    auto concat = dyn_cast<ConcatSourceAttr>(layout);
    if (!concat) {
      return op->emitError("expected concatenation layout");
    }
    ArrayRef<int32_t> argumentIndex = concat.getArgumentIndex();
    if (argumentIndex.empty()) {
      if (concat.size() != op->getNumOperands()) {
        return op->emitError("operand layouts cannot be inferred");
      }
      return SmallVector<LayoutSourceAttrInterface>(concat.getSources());
    }
    if (argumentIndex.size() != concat.size()) {
      return op->emitError("invalid argument index for concatenation layout");
    }
    for (auto [sourceIndex, operandIndex] : llvm::enumerate(argumentIndex)) {
      if (operandIndex < 0 ||
          static_cast<size_t>(operandIndex) >= op->getNumOperands()) {
        return op->emitError("invalid argument index for concatenation layout");
      }
      operandLayouts[operandIndex] = concat.getSource(sourceIndex);
    }
    return operandLayouts;
  }
  if (isa<ReduceOp, ReduceUDOp>(op)) {
    auto reduction = dyn_cast<ReductionSourceAttr>(layout);
    if (!reduction) {
      return op->emitError("expected reduction layout");
    }
    if (isa<ReduceUDOp>(op) && op->getNumOperands() > 1) {
      auto composite = dyn_cast<CompositeSourceAttr>(reduction.getSource());
      if (!composite || composite.size() != op->getNumOperands()) {
        return op->emitError(
            "expected one reduction source layout per operand");
      }
      return SmallVector<LayoutSourceAttrInterface>(composite.getSources());
    }
    operandLayouts.front() = reduction.getSource();
    return operandLayouts;
  }
  if (isa<MatmulOp>(op)) {
    auto matmul = dyn_cast<MatmulSourceAttr>(layout);
    if (!matmul) {
      return op->emitError("expected matmul layout");
    }
    operandLayouts[0] = matmul.getLhs();
    operandLayouts[1] = matmul.getRhs();
    return operandLayouts;
  }
  if (isa<ScanSliceOp>(op)) {
    auto scanSlice = dyn_cast<ScanSliceSourceAttr>(layout);
    if (!scanSlice) {
      return op->emitError("expected scan_slice source layout");
    }
    operandLayouts.front() = scanSlice.getCarry();
    return operandLayouts;
  }
  if (llvm::any_of(op->getOperandTypes(), llvm::IsaPred<RankedTensorType>)) {
    return op->emitError("operand layouts cannot be inferred");
  }
  return operandLayouts;
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
  op->removeAttr(TensorIRDialect::getIterSpaceMapAttrName());
  op->removeAttr(TensorIRDialect::getIterSpaceIdAttrName());
  op->removeAttr(TensorIRDialect::getIterSpaceIdsAttrName());
  op->removeAttr(TensorIRDialect::getIterSpaceDimDomainsAttrName());
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

/// Extracts descriptor metadata from a boundary memref function argument.
///
/// Static metadata is derived from the memref type. Emits
/// `memref.dim` for dynamic sizes and `memref.extract_strided_metadata` for
/// dynamic strides. Dynamic sizes and strides are flattened
/// into `dynamicValues` for indexing by `TensorSourceAttr`'s dynamic-value
/// mapping.
static FailureOr<BoundaryAccess> getBoundaryAccess(OpBuilder &builder,
                                                   func::FuncOp function,
                                                   Value buffer,
                                                   Operation *diagnosticOp) {
  // getLayoutBoundary already validated the strided memref argument.
  auto argument = cast<BlockArgument>(buffer);
  auto type = cast<MemRefType>(buffer.getType());
  auto [strides, offset] = type.getStridesAndOffset();
  if (offset != 0) {
    return diagnosticOp->emitError()
           << "bufferized TensorIR boundary must have a statically zero "
              "offset, but got "
           << type;
  }
  if (type.getMemorySpace()) {
    return diagnosticOp->emitError() << "bufferized TensorIR boundary must be "
                                        "in the default memory space, but got "
                                     << type;
  }

  memref::ExtractStridedMetadataOp metadata;
  if (llvm::any_of(strides, ShapedType::isDynamic)) {
    metadata = memref::ExtractStridedMetadataOp::create(
        builder, diagnosticOp->getLoc(), buffer);
  }
  BoundaryAccess access;
  access.buffer = buffer;
  access.elementType = type.getElementType();
  access.sizes = memref::getMixedSizes(builder, diagnosticOp->getLoc(), buffer);
  access.dynamicValues = decomposeMixedValues(access.sizes).second;
  access.strides = metadata
                       ? metadata.getConstifiedMixedStrides()
                       : getAsIndexOpFoldResult(builder.getContext(), strides);
  llvm::append_range(access.dynamicValues,
                     decomposeMixedValues(access.strides).second);
  auto alignment = function.getArgAttrOfType<IntegerAttr>(
      argument.getArgNumber(), TensorIRDialect::getAlignmentAttrName());
  int64_t elementBytes =
      std::max(1u, type.getElementType().getIntOrFloatBitWidth() / 8);
  access.alignment =
      builder.getI64IntegerAttr(alignment ? alignment.getInt() : elementBytes);
  auto allowTma = function.getArgAttrOfType<BoolAttr>(
      argument.getArgNumber(), TensorIRDialect::getAllowTMAAttrName());
  access.allowTma = !allowTma || allowTma.getValue();
  return access;
}

/// Reuse the boundary memref when its logical view is unchanged. Otherwise
/// describe the complete view with a reinterpret cast; offset is relative to
/// the aligned base, as required by memref.reinterpret_cast.
static Value createAccessView(OpBuilder &builder, Location loc,
                              const BoundaryAccess &access, OpFoldResult offset,
                              ArrayRef<OpFoldResult> sizes,
                              ArrayRef<OpFoldResult> strides) {
  auto staticOffset = getConstantIntValue(offset);
  if (staticOffset == 0 && sizes == ArrayRef(access.sizes) &&
      strides == ArrayRef(access.strides)) {
    return access.buffer;
  }
  if (staticOffset) {
    offset = builder.getIndexAttr(*staticOffset);
  }
  return memref::ReinterpretCastOp::create(builder, loc, access.buffer, offset,
                                           sizes, strides);
}

/// Decoded logical view of a tensor after resolving its CuTe layout against the
/// boundary access's dynamic runtime values.
struct LogicalDescriptor {
  /// Logical source offset relative to the zero-offset boundary buffer.
  OpFoldResult offset;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  /// Byte alignment of the view origin, adjusted for the logical offset.
  IntegerAttr alignment;
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

  int64_t alignment = access.alignment.getInt();
  unsigned elementBytes =
      std::max(1u, access.elementType.getIntOrFloatBitWidth() / 8);
  int64_t offsetBytes = source.getOffset() * elementBytes;
  if (offsetBytes != 0) {
    alignment = std::gcd(alignment, std::abs(offsetBytes));
  }
  return LogicalDescriptor{builder.getIndexAttr(source.getOffset()),
                           std::move(sizes), std::move(strides),
                           builder.getI64IntegerAttr(alignment)};
}

/// Load broadcast axes at extent one with non-unit strides, then broadcast
/// to the requested tile shape.
static Value createTiledLoad(OpBuilder &builder, Location loc,
                             const BoundaryAccess &access,
                             const LogicalDescriptor &metadata,
                             ArrayRef<OpFoldResult> coordinates,
                             ArrayRef<int64_t> tileShape) {
  SmallVector<OpFoldResult> sizes, strides, indices;
  SmallVector<int64_t> loadShape;
  for (auto [size, stride, coordinate, tile] : llvm::zip_equal(
           metadata.sizes, metadata.strides, coordinates, tileShape)) {
    if (getConstantIntValue(stride) == 0) {
      sizes.push_back(builder.getIndexAttr(1));
      strides.push_back(builder.getIndexAttr(2));
      indices.push_back(builder.getIndexAttr(0));
      loadShape.push_back(1);
      continue;
    }
    sizes.push_back(size);
    strides.push_back(stride);
    indices.push_back(coordinate);
    loadShape.push_back(tile);
  }
  Value memory =
      createAccessView(builder, loc, access, metadata.offset, sizes, strides);
  auto loadType = RankedTensorType::get(loadShape, access.elementType);
  Value result =
      LoadOp::create(builder, loc, loadType, memory, indices,
                     metadata.alignment, TilePaddingAttr{}, access.allowTma);
  if (loadShape != tileShape) {
    auto resultType = RankedTensorType::get(tileShape, access.elementType);
    result = BroadcastOp::create(builder, loc, resultType, result);
  }
  return result;
}

/// Select one representative slice for zero-stride output dimensions and
/// retain those axes at extent one with non-unit strides.
static void createTiledStore(OpBuilder &builder, Location loc, Value tile,
                             const BoundaryAccess &access,
                             const LogicalDescriptor &metadata,
                             ArrayRef<OpFoldResult> coordinates) {
  auto tileType = cast<RankedTensorType>(tile.getType());
  SmallVector<int64_t> sliceShape(tileType.getShape());
  SmallVector<OpFoldResult> sizes, strides, indices;
  for (auto [dimension, values] : llvm::enumerate(
           llvm::zip_equal(metadata.sizes, metadata.strides, coordinates))) {
    auto [size, stride, coordinate] = values;
    if (getConstantIntValue(stride) == 0) {
      sliceShape[dimension] = 1;
      sizes.push_back(builder.getIndexAttr(1));
      strides.push_back(builder.getIndexAttr(2));
      indices.push_back(builder.getIndexAttr(0));
      continue;
    }
    sizes.push_back(size);
    strides.push_back(stride);
    indices.push_back(coordinate);
  }
  if (!llvm::equal(sliceShape, tileType.getShape())) {
    SmallVector<int64_t> starts(sliceShape.size(), 0);
    SmallVector<int64_t> sliceStrides(sliceShape.size(), 1);
    auto sliceType = RankedTensorType::get(sliceShape, access.elementType);
    tile = SliceOp::create(builder, loc, sliceType, tile,
                           builder.getDenseI64ArrayAttr(starts),
                           builder.getDenseI64ArrayAttr(sliceShape),
                           builder.getDenseI64ArrayAttr(sliceStrides));
  }
  Value memory =
      createAccessView(builder, loc, access, metadata.offset, sizes, strides);
  StoreOp::create(builder, loc, tile, memory, indices, metadata.alignment,
                  access.allowTma);
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
/// emitting modulo or division operations. The last dimension with another
/// tile count takes the remaining linear index.
static SmallVector<Value>
delinearize(OpBuilder &builder, Location loc, Value linear,
            ArrayRef<OpFoldResult> dimensionTileCounts) {
  auto isSingleTile = [](OpFoldResult count) {
    return getConstantIntValue(count) == 1;
  };
  size_t lastTiled = dimensionTileCounts.size();
  for (size_t dimension = dimensionTileCounts.size(); dimension > 0;
       --dimension) {
    if (!isSingleTile(dimensionTileCounts[dimension - 1])) {
      lastTiled = dimension - 1;
      break;
    }
  }
  SmallVector<Value> result;
  result.reserve(dimensionTileCounts.size());
  Value zero;
  for (size_t dimension = 0; dimension < dimensionTileCounts.size();
       ++dimension) {
    if (dimension == lastTiled) {
      result.push_back(linear);
      continue;
    }
    if (dimension > lastTiled || isSingleTile(dimensionTileCounts[dimension])) {
      if (!zero) {
        zero = arith::ConstantIndexOp::create(builder, loc, 0);
      }
      result.push_back(zero);
      continue;
    }
    Value count = getValueOrCreateConstantIndexOp(
        builder, loc, dimensionTileCounts[dimension]);
    Value coordinate = arith::RemUIOp::create(builder, loc, linear, count);
    result.push_back(coordinate);
    linear = arith::DivUIOp::create(builder, loc, linear, count);
  }
  return result;
}

/// Determine the logical output view for a result store.
///
/// Returns the view from resultViews if layout normalization provides one.
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
planReductionTile(Operation *op, LayoutSourceAttrInterface requested,
                  ArrayRef<int64_t> outputTileShape,
                  const LayoutPropagationAnalysis &propagation) {
  auto layout = dyn_cast<ReductionSourceAttr>(requested);
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
                        inferOperandLayouts(op, layout, propagation));
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
planMatmulTile(MatmulOp matmul, LayoutSourceAttrInterface requested,
               ArrayRef<int64_t> outputTileShape) {
  auto layout = dyn_cast<MatmulSourceAttr>(requested);
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

/// Map per-dimension values (tile extents or carry axes) through unit-only reshapes.
/// Match logical extents, not tile extents: a tiled non-unit dimension may
/// itself have tile size one and must not be mistaken for an inserted axis.
static FailureOr<SmallVector<int64_t>>
mapScanSliceDimensions(ArrayRef<int64_t> inputShape,
                       ArrayRef<int64_t> inputValues,
                       ArrayRef<int64_t> outputShape, int64_t unitValue = 1) {
  SmallVector<int64_t> outputValues;
  size_t inputDim = 0;
  for (int64_t extent : outputShape) {
    if (extent == 1) {
      outputValues.push_back(unitValue);
      continue;
    }
    while (inputDim < inputShape.size() && inputShape[inputDim] == 1) {
      ++inputDim;
    }
    if (inputDim == inputShape.size() || inputShape[inputDim] != extent) {
      return failure();
    }
    outputValues.push_back(inputValues[inputDim++]);
  }
  while (inputDim < inputShape.size() && inputShape[inputDim] == 1) {
    ++inputDim;
  }
  if (inputDim != inputShape.size()) {
    return failure();
  }
  return outputValues;
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

  /// Find a tile of `source` that holds the requested tile with permuted
  /// dimensions. Returns the tile and the `nv_tensor_ir.transpose` permutation
  /// that produces the requested tile from it.
  std::optional<std::pair<Value, SmallVector<int64_t>>>
  lookupPermuted(Value source, LayoutSourceAttrInterface layout,
                 ArrayRef<Value> coordinates,
                 ArrayRef<int64_t> tileShape) const {
    size_t rank = tileShape.size();
    for (const Entry &entry : entries) {
      auto entryLayout = dyn_cast<LayoutSourceAttrInterface>(entry.layout);
      if (entry.source != source || !entryLayout ||
          entry.tileShape.size() != rank) {
        continue;
      }
      auto permutation = findLayoutPermutation(
          entryLayout, layout, [&](unsigned candidate, unsigned dimension) {
            return entry.tileShape[candidate] == tileShape[dimension] &&
                   isEqualConstantIntOrValue(entry.coordinates[candidate],
                                             coordinates[dimension]);
          });
      if (succeeded(permutation)) {
        return std::make_pair(entry.tile, std::move(*permutation));
      }
    }
    return std::nullopt;
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

/// Recursively traces the untiled TensorIR computation backward from results
/// and materializes fixed-size tile operations inside the GPU block grid.
///
/// Traversal is backward-recursive: starting from the output sources, each
/// operation requests its inputs at specific tile coordinates and shapes.
/// This ensures operations inside control-flow branches (e.g. ConcatenateOp)
/// are materialized strictly inside their respective branches rather than
/// speculated before them.
class TileMaterializer {
public:
  /// Maps boundary tensors to their input or output buffer metadata. ScanSlice
  /// histories use the output metadata to emit stores inside the carry loop.
  explicit TileMaterializer(
      const LayoutPropagationAnalysis &propagation,
      const llvm::DenseMap<Value, BoundaryAccess> &inputAccesses,
      const llvm::DenseMap<Value, BoundaryAccess> &outputAccesses)
      : propagation(propagation), inputAccesses(inputAccesses),
        outputAccesses(outputAccesses) {}

  /// Recursively materialize a tile of `value` conforming to `expectedLayout`
  /// at the given `coordinates` with shape `tileShape`.
  FailureOr<Value> materialize(Value value,
                               LayoutSourceAttrInterface expectedLayout,
                               ArrayRef<Value> coordinates,
                               ArrayRef<int64_t> tileShape,
                               OpBuilder &builder) {
    if (!isa<RankedTensorType>(value.getType())) {
      return materializeScalar(value, builder);
    }

    if (auto input = value.getDefiningOp<bufferization::ToTensorOp>()) {
      return loadSource(input, expectedLayout, coordinates, tileShape, builder);
    }
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      return emitError(argument.getLoc())
             << "bufferized TensorIR functions cannot have tensor arguments";
    }

    if (!expectedLayout) {
      return emitError(value.getLoc())
             << "tiled value has no iteration-space layout";
    }
    if (Value found = computedValues.lookup(value, expectedLayout, coordinates,
                                            tileShape)) {
      return found;
    }
    if (Value found = reusePermutedTile(computedValues, value, expectedLayout,
                                        coordinates, tileShape, builder)) {
      return found;
    }

    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != value.getParentBlock()) {
      return emitError(value.getLoc())
             << "tiled value is not defined in the TensorIR graph body";
    }

    if (auto scanSlice = dyn_cast<ScanSliceOp>(op)) {
      if (value != scanSlice.getResults().front()) {
        return scanSlice.emitError(
            "streaming scan_slice outputs cannot be materialized as tiles");
      }
      MLIR_ASSIGN_OR_RETURN(
          Value result, materializeScanSlice(scanSlice, expectedLayout,
                                             coordinates, tileShape, builder));
      computedValues.insert(value, expectedLayout, coordinates, tileShape,
                            result);
      return result;
    }
    if (auto concat = dyn_cast<ConcatenateOp>(op)) {
      MLIR_ASSIGN_OR_RETURN(Value result, materializeConcatenate(
                                              concat, expectedLayout,
                                              coordinates, tileShape, builder));
      computedValues.insert(value, expectedLayout, coordinates, tileShape,
                            result);
      return result;
    }
    if (auto matmul = dyn_cast<MatmulOp>(op)) {
      return materializeMatmul(matmul, expectedLayout, coordinates, tileShape,
                               builder);
    }
    if (isa<ReduceOp, ReduceUDOp>(op)) {
      return materializeReduction(value, expectedLayout, coordinates, tileShape,
                                  builder);
    }
    if (auto iota = dyn_cast<IotaOp>(op)) {
      MLIR_ASSIGN_OR_RETURN(Value result,
                            materializeIota(iota, expectedLayout, coordinates,
                                            tileShape, builder));
      computedValues.insert(value, expectedLayout, coordinates, tileShape,
                            result);
      return result;
    }

    MLIR_ASSIGN_OR_RETURN(SmallVector<LayoutSourceAttrInterface> operandLayouts,
                          inferOperandLayouts(op, expectedLayout, propagation));

    // These operations only modify logical addressing. Their analyzed layout
    // is passed to the underlying source load, so no tile compute remains.
    if (isa<ReshapeOp, BroadcastOp, TransposeOp, SliceOp>(op)) {
      MLIR_ASSIGN_OR_RETURN(
          Value result, materialize(op->getOperand(0), operandLayouts.front(),
                                    coordinates, tileShape, builder));
      computedValues.insert(value, expectedLayout, coordinates, tileShape,
                            result);
      return result;
    }

    IRMapping mapping;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!isa<RankedTensorType>(operand.getType())) {
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
      Type elementType =
          cast<RankedTensorType>(constant.getType()).getElementType();
      auto resultType = RankedTensorType::get(tileShape, elementType);
      DenseElementsAttr constantValue =
          elementType.isFloat()
              ? DenseElementsAttr::get(resultType,
                                       original.getSplatValue<APFloat>())
              : DenseElementsAttr::get(resultType,
                                       original.getSplatValue<APInt>());
      Value result = ConstantOp::create(builder, constant.getLoc(), resultType,
                                        cast<TypedAttr>(constantValue));
      computedValues.insert(value, expectedLayout, coordinates, tileShape,
                            result);
      return result;
    }

    Operation *clone = builder.clone(*op, mapping);
    removeAnalysisAttrs(clone);
    for (auto [original, result] :
         llvm::zip(op->getResults(), clone->getResults())) {
      if (auto tensorType = dyn_cast<RankedTensorType>(original.getType())) {
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
      computedValues.insert(original, expectedLayout, coordinates, tileShape,
                            cloned);
    }
    return computedValues.lookup(value, expectedLayout, coordinates, tileShape);
  }

private:
  /// Derive a requested tile from a cached tile of the same value whose
  /// dimensions are permuted, instead of loading or computing it again.
  Value reusePermutedTile(TileCache &cache, Value value,
                          LayoutSourceAttrInterface layout,
                          ArrayRef<Value> coordinates,
                          ArrayRef<int64_t> tileShape, OpBuilder &builder) {
    auto permuted = cache.lookupPermuted(value, layout, coordinates, tileShape);
    if (!permuted) {
      return {};
    }
    auto [tile, permutation] = *permuted;
    Value result = tile;
    if (!isIdentityPermutation(permutation)) {
      auto type = RankedTensorType::get(
          tileShape, cast<RankedTensorType>(tile.getType()).getElementType());
      result =
          TransposeOp::create(builder, value.getLoc(), type, tile, permutation);
    }
    cache.insert(value, layout, coordinates, tileShape, result);
    return result;
  }

  /// Lower a recurrence to explicit tiled TensorIR inside an scf.for. Streaming
  /// accesses are emitted here, before the ordinary outlined-kernel conversion.
  FailureOr<Value> materializeScanSlice(ScanSliceOp scanSlice,
                                        LayoutSourceAttrInterface layout,
                                        ArrayRef<Value> coordinates,
                                        ArrayRef<int64_t> tileShape,
                                        OpBuilder &builder) {
    auto source = dyn_cast<ScanSliceSourceAttr>(layout);
    auto carryType = cast<TensorType>(scanSlice.getInputs().front().getType());
    ArrayRef<int64_t> carryShape = carryType.getShape();
    if (!source || tileShape.size() != carryShape.size()) {
      return scanSlice.emitError(
          "scan_slice requires a carry-ranked source layout and tile");
    }
    Block &body = scanSlice.getBody().front();
    for (Operation &operation : body.without_terminator()) {
      if (!isa<AddOp, MulOp, ExpOp, SubOp, DivOp, MinOp, MaxOp, NegOp, AbsOp,
               LogOp, SqrtOp, RsqrtOp, ReciprocalOp, SinOp, CosOp, TanhFwdOp,
               CeilOp, FloorOp, ConvertOp, ReshapeOp, BroadcastOp, ReduceOp>(
              operation)) {
        return operation.emitOpError("is not supported inside scan_slice");
      }
      if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
        auto input = reshape.getInput().getType().getShape();
        auto output = reshape.getOutput().getType().getShape();
        if (failed(mapScanSliceDimensions(input, input, output))) {
          return reshape.emitError(
              "scan_slice body reshape only supports inserting "
              "or removing unit dimensions");
        }
      }
    }
    SmallVector<BoundaryAccess> stores;
    for (Value output : scanSlice.getResults().drop_front()) {
      if (!output.hasOneUse()) {
        return scanSlice.emitError(
            "scan_slice side results must have one boundary store");
      }
      auto store = dyn_cast<bufferization::MaterializeInDestinationOp>(
          *output.user_begin());
      auto access = outputAccesses.find(output);
      if (!store || store.getSource() != output ||
          access == outputAccesses.end()) {
        return scanSlice.emitError(
            "scan_slice side results must have one boundary store");
      }
      auto outputType = cast<TensorType>(output.getType());
      int64_t stride = 1;
      for (int64_t dim = outputType.getRank() - 1; dim >= 0; --dim) {
        if (outputType.getDimSize(dim) != 1 &&
            getConstantIntValue(access->second.strides[dim]) != stride) {
          return scanSlice.emitError(
              "scan_slice side outputs require contiguous strides");
        }
        stride *= outputType.getDimSize(dim);
      }
      stores.push_back(access->second);
    }
    MLIR_ASSIGN_OR_RETURN(Value carry,
                          materialize(scanSlice.getInputs().front(),
                                      source.getCarry(), coordinates, tileShape,
                                      builder));
    Location loc = scanSlice.getLoc();
    int64_t extent = scanSlice.getScanExtent();
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value upper = arith::ConstantIndexOp::create(builder, loc, extent);
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    auto loop =
        scf::ForOp::create(builder, loc, zero, upper, one, ValueRange{carry});
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(loop.getBody());
    Value step = loop.getInductionVar();
    if (scanSlice.getReverse()) {
      Value last = arith::ConstantIndexOp::create(builder, loc, extent - 1);
      step = arith::SubIOp::create(builder, loc, last, step);
    }
    auto offsetAtStep = [&](OpFoldResult base, int64_t stride) -> Value {
      Value strideValue = arith::ConstantIndexOp::create(builder, loc, stride);
      Value offset = arith::MulIOp::create(builder, loc, step, strideValue);
      Value baseValue = getValueOrCreateConstantIndexOp(builder, loc, base);
      return arith::AddIOp::create(builder, loc, baseValue, offset);
    };
    IRMapping mapping;
    mapping.map(body.getArgument(0), loop.getRegionIterArgs().front());
    // Unit-only reshapes preserve axis identities. Track those identities to
    // reject broadcasts/uses that would require data from a different tile.
    SmallVector<int64_t> carryAxes;
    for (auto [dim, extent] : llvm::enumerate(carryShape)) {
      carryAxes.push_back(extent == 1 ? -1 : static_cast<int64_t>(dim));
    }
    DenseMap<Value, SmallVector<int64_t>> axes;
    axes[body.getArgument(0)] = carryAxes;
    for (auto [index, attribute] : llvm::enumerate(source.getInputs())) {
      auto input = cast<ScanSliceInputAttr>(attribute);
      auto view = cast<TensorSourceAttr>(input.getSource());
      OpOperand &operand = scanSlice->getOpOperand(index + 1);
      auto accessIt = inputAccesses.find(operand.get());
      if (accessIt == inputAccesses.end()) {
        return scanSlice.emitError(
            "scan_slice inputs must be boundary tensors during formation");
      }
      const BoundaryAccess &access = accessIt->second;
      MLIR_ASSIGN_OR_RETURN(auto descriptor,
                            decodeLogicalDescriptor(view, access, scanSlice));
      Value offset = offsetAtStep(descriptor.offset, input.getScanStride());
      BlockArgument argument = scanSlice.getBlockArgumentForOperand(operand);
      auto argumentType = cast<TensorType>(argument.getType());
      if (!llvm::equal(view.getShape(), argumentType.getShape())) {
        return scanSlice.emitError(
            "scan_slice input layout must retain the native slice shape");
      }
      auto inputAxes =
          getScanSliceInputAxes(carryShape, argumentType.getShape());
      if (failed(inputAxes)) {
        return scanSlice.emitError(
            "scan_slice input axes do not match the carry dimensions");
      }
      SmallVector<int64_t> inputShape;
      SmallVector<OpFoldResult> indices;
      for (int64_t axis : *inputAxes) {
        inputShape.push_back(axis < 0 ? 1 : tileShape[axis]);
        indices.push_back(axis < 0 ? OpFoldResult(builder.getIndexAttr(0))
                                   : getAsOpFoldResult(coordinates[axis]));
      }
      // descriptor.alignment already accounts for the static source offset.
      // Every step adds scanStride elements (also for reverse scans), so the
      // view origin retains the gcd of that byte stride and its alignment.
      // Factor out the element-byte gcd to avoid overflowing stride * bytes.
      int64_t alignment = descriptor.alignment.getInt();
      int64_t elementBytes =
          std::max(1u, access.elementType.getIntOrFloatBitWidth() / 8);
      int64_t byteAlignment = std::gcd(alignment, elementBytes);
      alignment = byteAlignment *
                  std::gcd(alignment / byteAlignment, input.getScanStride());
      descriptor.offset = offset;
      descriptor.alignment = builder.getI64IntegerAttr(alignment);
      Value tile = createTiledLoad(builder, loc, access, descriptor, indices,
                                   inputShape);
      mapping.map(argument, tile);
      axes[argument] = std::move(*inputAxes);
    }
    for (Operation &operation : body.without_terminator()) {
      if (auto convert = dyn_cast<ConvertOp>(operation);
          convert &&
          convert.getInput().getType() == convert.getOutput().getType()) {
        mapping.map(convert.getOutput(), mapping.lookup(convert.getInput()));
        axes[convert.getOutput()] = axes.lookup(convert.getInput());
        continue;
      }
      auto type = cast<TensorType>(operation.getResult(0).getType());
      auto inputType = cast<TensorType>(operation.getOperand(0).getType());
      auto tiledInput =
          cast<TensorType>(mapping.lookup(operation.getOperand(0)).getType());
      SmallVector<int64_t> resultShape(tiledInput.getShape());
      auto resultAxes = axes.lookup(operation.getOperand(0));
      if (isa<ReshapeOp>(operation)) {
        auto reshaped = mapScanSliceDimensions(inputType.getShape(),
                                               resultShape, type.getShape());
        if (failed(reshaped)) {
          return operation.emitError("cannot reshape scan_slice body tile");
        }
        resultShape = std::move(*reshaped);
        resultAxes = *mapScanSliceDimensions(inputType.getShape(), resultAxes,
                                             type.getShape(), -1);
      } else if (isa<BroadcastOp>(operation)) {
        // Broadcast uses positional carry axes only. A non-unit input axis
        // must already be in its carry position; expanding a unit dimension
        // binds that position to the corresponding carry axis.
        for (auto [dim, axis] : llvm::enumerate(resultAxes)) {
          if (axis != -1 && axis != static_cast<int64_t>(dim)) {
            return operation.emitError(
                "scan_slice body broadcast input axis must be "
                "-1 or match its dimension position");
          }
        }
        for (auto [dim, extent] : llvm::enumerate(type.getShape())) {
          if (extent == 1) {
            resultShape[dim] = 1;
            resultAxes[dim] = -1;
            continue;
          }
          if (dim >= carryShape.size() || extent != carryShape[dim]) {
            return operation.emitError(
                "scan_slice body broadcast dimensions must be one or match the "
                "corresponding carry dimension");
          }
          resultShape[dim] = tileShape[dim];
          resultAxes[dim] = dim;
        }
      } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
        if (reduce.getReductionMode() != ReductionMode::add ||
            reduce.getDimensions().size() != 1 ||
            reduce.getDimensions().front() != inputType.getRank() - 1 ||
            resultShape.back() != inputType.getShape().back()) {
          return reduce.emitError(
              "scan_slice supports only a fully tiled innermost add reduction");
        }
        resultShape.back() = 1;
        resultAxes.back() = -1;
      } else {
        for (Value operand : operation.getOperands().drop_front()) {
          if (axes.lookup(operand) != resultAxes) {
            return operation.emitError(
                "scan_slice body operands must use the same carry axes");
          }
        }
      }
      axes[operation.getResult(0)] = std::move(resultAxes);
      Operation *clone = builder.clone(operation, mapping);
      removeAnalysisAttrs(clone);
      clone->getResult(0).setType(
          RankedTensorType::get(resultShape, type.getElementType()));
      if (isa<ReduceOp>(clone)) {
        // The recurrence already supplies a complete reduction tile. Splitting
        // its producer would incorrectly try to retile the loop-carried state.
        clone->setAttr("nv_tensor_ir.scan_slice_reduction",
                       builder.getUnitAttr());
      }
    }
    auto yield = scanSlice.getBodyTerminator();
    for (Value value : yield.getOperands()) {
      auto type = cast<TensorType>(value.getType());
      auto valueAxes = axes.lookup(value);
      for (auto [dim, extent] : llvm::enumerate(type.getShape())) {
        if (extent != 1 && valueAxes[dim] != carryAxes[dim]) {
          return scanSlice.emitError(
              "scan_slice body yield must preserve the carry axes");
        }
      }
    }
    for (auto [index, store] : llvm::enumerate(stores)) {
      auto type = cast<TensorType>(scanSlice.getResults()[index + 1].getType());
      auto slice =
          getScanSliceOutputSlice(type, scanSlice.getOutputDimensions()[index]);
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strides;
      SmallVector<OpFoldResult> indices = getAsOpFoldResult(coordinates);
      for (auto [dim, size] : llvm::enumerate(slice.shape)) {
        sizes.push_back(builder.getIndexAttr(size));
        strides.push_back(builder.getIndexAttr(slice.strides[dim]));
        if (size == 1) {
          indices[dim] = builder.getIndexAttr(0);
        }
      }
      Value offset = offsetAtStep(builder.getIndexAttr(0), slice.scanStride);
      Value tile = mapping.lookup(yield.getOperand(index + 1));
      LogicalDescriptor metadata{offset, sizes, strides,
                                 builder.getI64IntegerAttr(1)};
      createTiledStore(builder, loc, tile, store, metadata, indices);
    }
    Value nextCarry = mapping.lookup(yield.getOperand(0));
    scf::YieldOp::create(builder, loc, nextCarry);
    return loop.getResult(0);
  }

  /// Materialize an IotaOp tile.
  ///
  /// Computes both the intra-tile index variations and the inter-tile global
  /// block offsets. Intra-tile values are generated via IotaOp along each
  /// dimension and scaled by the dimension's stride. Inter-tile offsets are
  /// computed as `coordinates[dim] * tileShape[dim] * stride`, summed into
  /// a scalar, splatted across the tile, and added to the local indices.
  /// Floating-point iotas are generated as unsigned integer and converted.
  FailureOr<Value> materializeIota(IotaOp iota,
                                   LayoutSourceAttrInterface requested,
                                   ArrayRef<Value> coordinates,
                                   ArrayRef<int64_t> tileShape,
                                   OpBuilder &builder) {
    auto layout = dyn_cast<TensorSourceAttr>(requested);
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
      if (isa<RankedTensorType>(operand.getType())) {
        return op->emitError("scalar setup cannot depend on a tensor operand");
      }
      MLIR_ASSIGN_OR_RETURN(Value scalar, materializeScalar(operand, builder));
      mapping.map(operand, scalar);
    }
    if (llvm::any_of(op->getResultTypes(),
                     [](Type type) { return isa<RankedTensorType>(type); })) {
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

  /// Replaces an untiled boundary input with a tiled LoadOp.
  ///
  /// Decodes the logical layout descriptor and alignment, then loads the
  /// varying dimensions. Explicit broadcasts restore the requested tile shape.
  FailureOr<Value> loadSource(bufferization::ToTensorOp input,
                              LayoutSourceAttrInterface expectedLayout,
                              ArrayRef<Value> coordinates,
                              ArrayRef<int64_t> tileShape, OpBuilder &builder) {
    TensorSourceAttr source = getSingleTensorSource(expectedLayout);
    if (!source) {
      return input.emitError("input access does not have one concrete source "
                             "layout");
    }
    if (Value found = loadedValues.lookup(input.getResult(), source,
                                          coordinates, tileShape)) {
      return found;
    }
    if (Value found = reusePermutedTile(loadedValues, input.getResult(), source,
                                        coordinates, tileShape, builder)) {
      return found;
    }
    if (source.getShape().size() != tileShape.size()) {
      return input.emitError()
             << "iteration-space source rank " << source.getShape().size()
             << " does not match tile rank " << tileShape.size();
    }

    Location loc = input.getLoc();
    auto it = inputAccesses.find(input.getResult());
    if (it == inputAccesses.end()) {
      return input.emitError("input is not a boundary of the bufferized "
                             "TensorIR function");
    }
    const BoundaryAccess &access = it->second;
    MLIR_ASSIGN_OR_RETURN(LogicalDescriptor descriptor,
                          decodeLogicalDescriptor(source, access, input));
    Value result = createTiledLoad(builder, loc, access, descriptor,
                                   getAsOpFoldResult(coordinates), tileShape);
    loadedValues.insert(input.getResult(), source, coordinates, tileShape,
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
                                          LayoutSourceAttrInterface requested,
                                          ArrayRef<Value> coordinates,
                                          ArrayRef<int64_t> tileShape,
                                          OpBuilder &builder) {
    auto layout = dyn_cast<ConcatSourceAttr>(requested);
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
          tileShape, cast<RankedTensorType>(concat.getType()).getElementType());
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
                                        LayoutSourceAttrInterface layout,
                                        ArrayRef<Value> coordinates,
                                        ArrayRef<int64_t> outputTileShape,
                                        OpBuilder &builder) {
    Operation *op = requestedResult.getDefiningOp();
    if (coordinates.size() != outputTileShape.size()) {
      return op->emitError("reduction layout does not match the output tile");
    }
    MLIR_ASSIGN_OR_RETURN(
        ReductionTilePlan plan,
        planReductionTile(op, layout, outputTileShape, propagation));
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
      Type elementType =
          cast<RankedTensorType>(original.getType()).getElementType();
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
      computedValues.insert(original, layout, coordinates, outputTileShape,
                            result);
    }
    return computedValues.lookup(requestedResult, layout, coordinates,
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
                                     LayoutSourceAttrInterface layout,
                                     ArrayRef<Value> coordinates,
                                     ArrayRef<int64_t> outputTileShape,
                                     OpBuilder &builder) {
    if (coordinates.size() != outputTileShape.size()) {
      return matmul.emitError(
          "matmul coordinates do not match the output tile rank");
    }
    MLIR_ASSIGN_OR_RETURN(MatmulTilePlan plan,
                          planMatmulTile(matmul, layout, outputTileShape));
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
    computedValues.insert(matmul.getResult(), layout, coordinates,
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

  const LayoutPropagationAnalysis &propagation;
  const llvm::DenseMap<Value, BoundaryAccess> &inputAccesses;
  const llvm::DenseMap<Value, BoundaryAccess> &outputAccesses;
  llvm::ScopedHashTable<Value, Value> computedScalars;
  llvm::ScopedHashTableScope<Value, Value> scalarScope{computedScalars};
  TileCache computedValues;
  TileCache loadedValues;
};

/// Rebuild the boundary layout of `tensor`, bridged from `buffer`. Tensor IDs
/// are function argument numbers, which follow the graph signature.
static FailureOr<TensorSourceAttr> getBoundarySource(func::FuncOp function,
                                                     Value buffer, Value tensor,
                                                     Operation *diagnosticOp) {
  auto argument = dyn_cast<BlockArgument>(buffer);
  auto type = dyn_cast<MemRefType>(buffer.getType());
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!argument || argument.getOwner() != &function.getBody().front() ||
      !type || !tensorType) {
    return diagnosticOp->emitError(
        "bufferized TensorIR boundary must be a memref function argument");
  }
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset))) {
    return diagnosticOp->emitError()
           << "bufferized TensorIR boundary must be strided, but got " << type;
  }
  tcg::Layout layout(getShapeRef(tensorType), tcg::Stride(strides));
  return TensorSourceAttr::get(function.getContext(), argument.getArgNumber(),
                               /*offset=*/0, layout.toString(),
                               getDynamicValueMapping(layout));
}

/// Build the layout-propagation boundary of a bufferized function from its
/// bufferization bridges.
static FailureOr<LayoutBoundary>
getLayoutBoundary(func::FuncOp function,
                  ArrayRef<bufferization::ToTensorOp> inputs,
                  ArrayRef<bufferization::MaterializeInDestinationOp> outputs) {
  LayoutBoundary boundary;
  boundary.op = function;
  boundary.body = &function.getBody().front();
  auto isDynamic = [](Value value) {
    return !cast<ShapedType>(value.getType()).hasStaticShape();
  };
  for (bufferization::ToTensorOp input : inputs) {
    MLIR_ASSIGN_OR_RETURN(TensorSourceAttr source,
                          getBoundarySource(function, input.getBuffer(),
                                            input.getResult(), input));
    boundary.inputs[input.getResult()] = source;
    boundary.boundaryOps.insert(input);
    boundary.hasDynamicShapes |= isDynamic(input.getResult());
  }
  for (bufferization::MaterializeInDestinationOp output : outputs) {
    MLIR_ASSIGN_OR_RETURN(TensorSourceAttr source,
                          getBoundarySource(function, output.getDest(),
                                            output.getSource(), output));
    boundary.results.push_back(output.getSource());
    boundary.outputs.push_back(source);
    boundary.boundaryOps.insert(output);
    boundary.hasDynamicShapes |= isDynamic(output.getSource());
  }
  return boundary;
}

/// Main driver for FormGridPass.
///
/// Execution proceeds in six phases:
/// 1. Validation: Verify the function has bufferized_program attribute, a single
///    entry block, valid outputs, and tile size, and compute its layout
///    propagation and normalized iteration space.
/// 2. Boundary decoding: Read metadata for each boundary buffer, resolve
///    logical output views, and decode their descriptors. Collect dynamic
///    extents when any dimension is dynamic.
/// 3. Grid sizing: Compute tile counts per dimension (ceil(extent / tile)) and
///    flatten into a 1D totalTiles count with static overflow checking.
/// 4. Launch construction: Emit a 1D scf.forall mapped to #gpu.block<x> at the
///    end of the entry block.
/// 5. Body materialization: Inside the forall body, delinearize the linear block
///    ID into multi-dimensional coordinates, run TileMaterializer backward
///    from each output, and emit tiled StoreOps.
/// 6. Cleanup: Erase the bufferization bridges and dead untiled computation,
///    and remove compiler analysis attributes.
static LogicalResult formGrid(func::FuncOp function) {
  if (!function->hasAttr(TensorIRDialect::getBufferizedProgramAttrName())) {
    return success();
  }

  // === Phase 1: Validate function structure, boundary outputs, and layouts ===
  if (function.isExternal() || !function.getBody().hasOneBlock()) {
    return function.emitError(
        "bufferized TensorIR program must define one entry block");
  }
  Block &entry = function.getBody().front();
  SmallVector<bufferization::ToTensorOp> inputs(
      entry.getOps<bufferization::ToTensorOp>());
  SmallVector<bufferization::MaterializeInDestinationOp> outputs(
      entry.getOps<bufferization::MaterializeInDestinationOp>());
  if (outputs.empty()) {
    return function.emitError("grid formation requires at least one "
                              "bufferization.materialize_in_destination");
  }

  MLIR_ASSIGN_OR_RETURN(LayoutBoundary boundary,
                        getLayoutBoundary(function, inputs, outputs));
  FailureOr<LayoutPropagationAnalysis> propagation =
      LayoutPropagationAnalysis::compute(boundary);
  if (failed(propagation)) {
    return failure();
  }
  MLIR_ASSIGN_OR_RETURN(LayoutNormalization normalization,
                        LayoutNormalization::compute(*propagation));
  LayoutSourceAttrInterface rootLayout = normalization.iterationSpace;
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

  SmallVector<TensorSourceAttr> resultViews = normalization.resultViews;
  SmallVector<LayoutSourceAttrInterface> resultLayouts =
      normalization.resultLayouts;
  for (auto [index, pair] :
       llvm::enumerate(llvm::zip_equal(resultLayouts, resultViews))) {
    auto [layout, view] = pair;
    if (layout.getShape() != rootShape || view.getShape() != rootShape) {
      return function.emitError()
             << "result " << index
             << " is not expressed in normalized carrier shape "
             << vectorToString(rootShape);
    }
  }
  if (resultLayouts.empty()) {
    resultLayouts.push_back(rootLayout);
  }

  // === Phase 2: Read boundary metadata and decode output descriptors ===
  llvm::DenseMap<Value, BoundaryAccess> inputAccesses;
  for (bufferization::ToTensorOp input : inputs) {
    OpBuilder builder(input);
    MLIR_ASSIGN_OR_RETURN(
        BoundaryAccess access,
        getBoundaryAccess(builder, function, input.getBuffer(), input));
    inputAccesses.try_emplace(input.getResult(), std::move(access));
  }

  SmallVector<BoundaryAccess> outputAccesses;
  llvm::DenseMap<Value, BoundaryAccess> outputAccessByValue;
  SmallVector<LogicalDescriptor> outputDescriptors;
  outputAccesses.reserve(outputs.size());
  for (auto [index, output] : llvm::enumerate(outputs)) {
    if (output->getNumResults() != 0) {
      return output.emitError(
          "bufferized TensorIR outputs must materialize into a memref");
    }
    OpBuilder builder(output);
    MLIR_ASSIGN_OR_RETURN(
        BoundaryAccess access,
        getBoundaryAccess(builder, function, output.getDest(), output));
    outputAccessByValue.try_emplace(output.getSource(), access);
    outputAccesses.push_back(std::move(access));
    MLIR_ASSIGN_OR_RETURN(TensorSourceAttr view,
                          getOutputView(function, outputAccesses.back(), index,
                                        rootShape, resultViews));
    MLIR_ASSIGN_OR_RETURN(
        LogicalDescriptor descriptor,
        decodeLogicalDescriptor(view, outputAccesses.back(), output));
    outputDescriptors.push_back(std::move(descriptor));
  }

  SmallVector<Value> dynamicRootExtents(rootShape.size());
  for (const LogicalDescriptor &descriptor : outputDescriptors) {
    for (auto [rootExtent, outputExtent] :
         llvm::zip_equal(dynamicRootExtents, descriptor.sizes)) {
      if (!rootExtent) {
        rootExtent = dyn_cast<Value>(outputExtent);
      }
    }
  }

  // === Phase 3: Compute tile counts per dimension and flattened grid size ===
  // Build the grid before the terminator so that metadata extracted at every
  // bridge dominates the tiled body. The untiled TensorIR computation is
  // erased once materialization succeeds.
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
    Value tileCount = dynamicExtent;
    if (tile != 1) {
      Value divisor =
          arith::ConstantIndexOp::create(entryBuilder, function.getLoc(), tile);
      tileCount = arith::CeilDivUIOp::create(entryBuilder, function.getLoc(),
                                             dynamicExtent, divisor);
    }
    tileCounts.push_back(tileCount);
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
    Value lhs = getValueOrCreateConstantIndexOp(entryBuilder, function.getLoc(),
                                                totalTiles);
    Value rhs =
        getValueOrCreateConstantIndexOp(entryBuilder, function.getLoc(), count);
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
        TileMaterializer materializer(*propagation, inputAccesses,
                                      outputAccessByValue);

        // Recursively materialize the tiled compute graph backward from each
        // output.
        for (auto [index, output] : llvm::enumerate(outputs)) {
          if (isStreamingScanSliceResult(output.getSource())) {
            continue;
          }
          FailureOr<Value> tiledResult =
              materializer.materialize(output.getSource(), resultLayouts[index],
                                       coordinates, tileShape, builder);
          if (failed(tiledResult)) {
            bodyStatus = failure();
            return;
          }
          createTiledStore(builder, output.getLoc(), *tiledResult,
                           outputAccesses[index], outputDescriptors[index],
                           getAsOpFoldResult(coordinates));
        }
        scf::InParallelOp::create(builder, loc);
      });
  if (failed(bodyStatus)) {
    return failure();
  }

  // === Phase 5: Clean up untiled operations and strip analysis attributes ===
  for (bufferization::MaterializeInDestinationOp output : outputs) {
    output->erase();
  }
  IRRewriter rewriter(function.getContext());
  eliminateTriviallyDeadOps(rewriter, function.getBody(),
                            /*includeNestedRegions=*/false);
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
