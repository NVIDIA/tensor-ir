// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/LayoutPropagation.h"

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/IR/AttrTypeSubElements.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "layout-propagation"

namespace mlir::nv_tensor_ir {

namespace {

namespace tcg = mlir::nv_tensor_ir::tcutegen;

/// Return the forward layout of a result value. Copy/passthrough results are
/// boundary inputs and obtain their layout from the boundary instead of a
/// defining operation.
static FailureOr<LayoutSourceAttrInterface>
getResultLayout(const LayoutPropagationAnalysis &propagation, Value value) {
  LayoutSourceAttrInterface layout = propagation.getLayout(value);
  if (!layout) {
    if (Operation *definingOp = value.getDefiningOp()) {
      definingOp->emitError("missing layout data for graph result");
    }
    return failure();
  }
  return layout;
}

/// Return the physical output layout for result `resultIndex`. Output tensor
/// IDs follow the inputs in the converted kernel signature. These sources are
/// kept in the result views and are not added to the root input-source union.
static FailureOr<TensorSourceAttr>
getPhysicalResultSource(const LayoutBoundary &boundary, size_t resultIndex) {
  TensorSourceAttr source = boundary.outputs[resultIndex];
  if (!source) {
    return boundary.op->emitError() << "multi-output layout propagation "
                                       "requires tensor result "
                                    << resultIndex;
  }
  return source;
}

/// Try to express `layout` in `carrierShape`. Besides ordinary reshape and
/// broadcast, this supports rank expansion by inserting unit dimensions before
/// broadcasting them. The latter is what lifts a compact projection such as
/// [P,D] into carrier [P,G,D]. Ambiguous unit insertion is rejected rather than
/// matching axes by equal numeric extent.
static FailureOr<LayoutSourceAttrInterface>
liftLayoutToCarrier(LayoutSourceAttrInterface layout,
                    ArrayRef<int64_t> carrierShape) {
  if (layout.getShape() == carrierShape) {
    return layout;
  }
  if (auto broadcast = layout.broadcast(carrierShape)) {
    return broadcast;
  }
  if (auto reshape = layout.reshape(carrierShape)) {
    return reshape;
  }

  SmallVector<int64_t> sourceShape = layout.getShape();
  if (sourceShape.size() >= carrierShape.size()) {
    return failure();
  }

  SmallVector<LayoutSourceAttrInterface> candidates;
  SmallVector<int64_t> expanded(carrierShape.size(), 1);
  std::function<void(size_t, size_t)> enumerate = [&](size_t sourceDim,
                                                      size_t carrierDim) {
    if (sourceDim == sourceShape.size()) {
      auto reshaped = layout.reshape(expanded);
      auto lifted = reshaped ? reshaped.broadcast(carrierShape) : nullptr;
      if (lifted && !llvm::is_contained(candidates, lifted)) {
        candidates.push_back(lifted);
      }
      return;
    }
    if (carrierDim == carrierShape.size()) {
      return;
    }

    size_t remainingSource = sourceShape.size() - sourceDim;
    size_t remainingCarrier = carrierShape.size() - carrierDim;
    if (remainingCarrier < remainingSource) {
      return;
    }

    int64_t sourceSize = sourceShape[sourceDim];
    int64_t carrierSize = carrierShape[carrierDim];
    bool compatible = sourceSize == carrierSize || sourceSize == 1 ||
                      (ShapedType::isDynamic(sourceSize) &&
                       ShapedType::isDynamic(carrierSize));
    if (compatible) {
      expanded[carrierDim] = sourceSize;
      enumerate(sourceDim + 1, carrierDim + 1);
      expanded[carrierDim] = 1;
    }

    if (remainingCarrier > remainingSource) {
      expanded[carrierDim] = 1;
      enumerate(sourceDim, carrierDim + 1);
    }
  };
  enumerate(/*sourceDim=*/0, /*carrierDim=*/0);

  if (candidates.size() != 1) {
    return failure();
  }
  return candidates.front();
}

/// Collect the backward SSA ancestry of a terminal value. TensorIR operations
/// are pure in the layout-propagation subset, so every outer operand is a
/// semantic dependency of each result. Region block arguments of ReduceUDOp
/// are deliberately not traversed; the enclosing op operands are.
static void collectAncestors(Value value, llvm::DenseSet<Value> &ancestors) {
  if (!ancestors.insert(value).second) {
    return;
  }
  if (Operation *definingOp = value.getDefiningOp()) {
    if (auto scanSlice = dyn_cast<ScanSliceOp>(definingOp)) {
      collectAncestors(scanSlice.getInputs().front(), ancestors);
      return;
    }
    for (Value operand : definingOp->getOperands()) {
      collectAncestors(operand, ancestors);
    }
  }
}

static bool isCarrierAnchor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() == 0) {
    return false;
  }
  if (Operation *definingOp = value.getDefiningOp()) {
    return !isa<ConstantOp, SplatOp>(definingOp);
  }
  return true;
}

struct LiftedResult {
  LayoutSourceAttrInterface layout;
  TensorSourceAttr view;
};

struct CarrierCandidate {
  Value anchor;
  SmallVector<int64_t> shape;
};

/// Try to lift a terminal layout and its physical output view into the same
/// carrier shape. Both must succeed because joint normalization must preserve
/// the computation layout and the ABI store mapping together.
static FailureOr<LiftedResult>
liftResultPairToCarrier(LayoutSourceAttrInterface layout, TensorSourceAttr view,
                        ArrayRef<int64_t> carrierShape) {
  auto liftedLayout = liftLayoutToCarrier(layout, carrierShape);
  auto liftedView = liftLayoutToCarrier(view, carrierShape);
  if (failed(liftedLayout) || failed(liftedView) ||
      !isa<TensorSourceAttr>(*liftedView)) {
    return failure();
  }
  return LiftedResult{*liftedLayout, cast<TensorSourceAttr>(*liftedView)};
}

static SmallVector<int64_t>
getInversePermutation(ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> inverse(permutation.size());
  for (auto [resultDim, operandDim] : llvm::enumerate(permutation)) {
    inverse[operandDim] = resultDim;
  }
  return inverse;
}

static bool hasAncestor(Value value, Value ancestor) {
  llvm::DenseSet<Value> ancestors;
  collectAncestors(value, ancestors);
  return ancestors.contains(ancestor);
}

/// Express one terminal result in `carrierShape`, using its SSA lineage to
/// recover coordinate semantics that shape-only lifting cannot infer. The walk
/// always reaches `carrierAnchor`; shape lifting is only attempted at that base
/// case. This prevents a shape-compatible terminal layout from bypassing a
/// coordinate transform on the result-to-carrier path.
///
/// Transpose and reshape are transported backward while the walk remains in the
/// terminal coordinate-preserving suffix. A structured operation such as a
/// reduction or broadcast ends that suffix: its forward-propagated layout is a
/// semantic summary of the preceding path. The walk still follows SSA ancestry
/// to the carrier, but does not apply transforms already represented inside
/// that summary.
static FailureOr<LiftedResult>
liftResultToCarrier(Value result, Value carrierAnchor,
                    LayoutSourceAttrInterface layout, TensorSourceAttr view,
                    ArrayRef<int64_t> carrierShape) {
  using LiftedResultKey =
      std::pair<LayoutSourceAttrInterface, TensorSourceAttr>;
  llvm::SmallSetVector<LiftedResultKey, 4> candidates;
  // Shared SSA paths may reach a value with different coordinate mappings.
  // Only identical traversal states can reuse the candidates already found.
  using VisitKey =
      std::tuple<Value, LayoutSourceAttrInterface, TensorSourceAttr, unsigned>;
  llvm::DenseSet<VisitKey> visited;

  // The carrier is fixed for this traversal. Cache reachability instead of
  // collecting the same operand's full ancestry at every incoming edge.
  llvm::DenseMap<Value, bool> reachesCarrier;
  std::function<bool(Value)> hasCarrierAncestor = [&](Value value) {
    if (value == carrierAnchor) {
      return true;
    }
    auto cached = reachesCarrier.find(value);
    if (cached != reachesCarrier.end()) {
      return cached->second;
    }
    bool reaches = false;
    if (Operation *definingOp = value.getDefiningOp()) {
      if (auto scanSlice = dyn_cast<ScanSliceOp>(definingOp)) {
        reaches = hasCarrierAncestor(scanSlice.getInputs().front());
      } else {
        reaches = llvm::any_of(definingOp->getOperands(), hasCarrierAncestor);
      }
    }
    reachesCarrier.try_emplace(value, reaches);
    return reaches;
  };

  std::function<void(Value, LayoutSourceAttrInterface, TensorSourceAttr, bool)>
      visit = [&](Value value, LayoutSourceAttrInterface currentLayout,
                  TensorSourceAttr currentView, bool transportCoordinates) {
        if (!visited
                 .insert(
                     {value, currentLayout, currentView, transportCoordinates})
                 .second) {
          return;
        }

        if (value == carrierAnchor) {
          auto lifted =
              liftResultPairToCarrier(currentLayout, currentView, carrierShape);
          if (succeeded(lifted)) {
            candidates.insert({lifted->layout, lifted->view});
          }
          return;
        }

        Operation *definingOp = value.getDefiningOp();
        if (!definingOp) {
          return;
        }

        // A scan_slice is transparent to surrounding layout propagation except for
        // its carry edge. Streaming side inputs have their mappings embedded
        // in ScanSliceSourceAttr and must not become graph carrier candidates.
        if (auto scanSlice = dyn_cast<ScanSliceOp>(definingOp)) {
          Value carry = scanSlice.getInputs().front();
          if (hasCarrierAncestor(carry)) {
            visit(carry, currentLayout, currentView, transportCoordinates);
          }
          return;
        }

        if (auto transpose = transportCoordinates
                                 ? dyn_cast<TransposeOp>(definingOp)
                                 : nullptr) {
          Value operand = transpose.getInput();
          if (!hasCarrierAncestor(operand)) {
            return;
          }
          SmallVector<int64_t> inverse =
              getInversePermutation(transpose.getPermutation());
          auto operandLayout = currentLayout.transpose(inverse);
          auto operandView = currentView.transpose(inverse);
          if (!operandLayout ||
              !isa_and_nonnull<TensorSourceAttr>(operandView)) {
            return;
          }
          visit(operand, operandLayout, cast<TensorSourceAttr>(operandView),
                /*transportCoordinates=*/true);
          return;
        }

        if (auto reshape = transportCoordinates
                               ? dyn_cast<ReshapeOp>(definingOp)
                               : nullptr) {
          Value operand = reshape.getInput();
          if (!hasCarrierAncestor(operand)) {
            return;
          }
          ArrayRef<int64_t> operandShape =
              cast<RankedTensorType>(operand.getType()).getShape();
          auto operandLayout = currentLayout.reshape(operandShape);
          auto operandView = currentView.reshape(operandShape);
          if (!operandLayout ||
              !isa_and_nonnull<TensorSourceAttr>(operandView)) {
            return;
          }
          visit(operand, operandLayout, cast<TensorSourceAttr>(operandView),
                /*transportCoordinates=*/true);
          return;
        }

        bool transparent = definingOp->hasTrait<OpTrait::Elementwise>();
        bool transportOperands = transportCoordinates && transparent;
        for (Value operand : definingOp->getOperands()) {
          if (!isa<RankedTensorType>(operand.getType()) ||
              !hasCarrierAncestor(operand)) {
            continue;
          }
          visit(operand, currentLayout, currentView, transportOperands);
        }
      };

  visit(result, layout, view, /*transportCoordinates=*/true);
  if (candidates.size() != 1) {
    return failure();
  }
  auto [candidateLayout, candidateView] = candidates.front();
  return LiftedResult{candidateLayout, candidateView};
}

/// Select the downstream-most candidate in one shape group. It minimizes the
/// result-to-carrier paths while remaining independent of ABI result order.
/// When candidates are incomparable, their shape is already known to be
/// compatible; use their stable graph order established by the caller.
static Value selectCarrierAnchor(ArrayRef<Value> candidates) {
  assert(!candidates.empty() && "expected at least one carrier candidate");
  for (Value candidate : candidates) {
    llvm::DenseSet<Value> ancestors;
    collectAncestors(candidate, ancestors);
    if (llvm::all_of(candidates,
                     [&](Value value) { return ancestors.contains(value); })) {
      return candidate;
    }
  }
  return candidates.front();
}

/// Select a result-order-independent common carrier SSA anchor. Static
/// candidates with the greatest element count are maximal. Distinct shapes at
/// that size are accepted only when one comes from a common value that is
/// downstream of a candidate in every other shape group; otherwise the
/// semantic carrier is ambiguous and the graph is rejected.
static FailureOr<CarrierCandidate>
deriveCommonCarrier(const LayoutBoundary &boundary,
                    ArrayRef<Value> graphResults,
                    ArrayRef<LayoutSourceAttrInterface> resultLayouts,
                    ArrayRef<TensorSourceAttr> resultSources) {
  assert(graphResults.size() > 1 && "multi-result helper requires N > 1");
  Operation *graphOp = boundary.op;

  if (boundary.hasDynamicShapes) {
    return graphOp->emitError(
        "dynamic multi-output layout propagation is not yet supported");
  }

  SmallVector<llvm::DenseSet<Value>> ancestry(graphResults.size());
  for (auto [index, result] : llvm::enumerate(graphResults)) {
    collectAncestors(result, ancestry[index]);
  }

  SmallVector<Value> orderedValues(boundary.body->getArguments());
  for (Operation &op : boundary.body->getOperations()) {
    llvm::append_range(orderedValues, op.getResults());
  }

  SmallVector<Value> commonValues;
  for (Value candidate : orderedValues) {
    if (!isCarrierAnchor(candidate)) {
      continue;
    }
    bool common = llvm::all_of(
        ancestry, [&](const auto &set) { return set.contains(candidate); });
    if (!common) {
      continue;
    }

    ArrayRef<int64_t> candidateShape =
        cast<RankedTensorType>(candidate.getType()).getShape();
    bool compatible = true;
    for (auto [result, layout, source] :
         llvm::zip_equal(graphResults, resultLayouts, resultSources)) {
      if (failed(liftResultToCarrier(result, candidate, layout, source,
                                     candidateShape))) {
        compatible = false;
        break;
      }
    }
    if (compatible) {
      commonValues.push_back(candidate);
    }
  }

  if (commonValues.empty()) {
    return graphOp->emitError(
        "multi-output results have no compatible nonconstant common SSA "
        "ancestor; split the results into multiple TensorIR graphs");
  }

  int64_t maxElements = 0;
  for (Value value : commonValues) {
    int64_t elements = cast<RankedTensorType>(value.getType()).getNumElements();
    maxElements = std::max(maxElements, elements);
  }

  std::map<SmallVector<int64_t>, SmallVector<Value>> maximalByShape;
  for (Value value : commonValues) {
    auto type = cast<RankedTensorType>(value.getType());
    if (type.getNumElements() == maxElements) {
      maximalByShape[SmallVector<int64_t>(type.getShape())].push_back(value);
    }
  }
  if (maximalByShape.size() == 1) {
    auto &[shape, values] = *maximalByShape.begin();
    return CarrierCandidate{selectCarrierAnchor(values), shape};
  }

  std::optional<CarrierCandidate> selected;
  for (Value candidate : commonValues) {
    auto candidateType = cast<RankedTensorType>(candidate.getType());
    if (candidateType.getNumElements() != maxElements) {
      continue;
    }
    llvm::DenseSet<Value> candidateAncestors;
    collectAncestors(candidate, candidateAncestors);
    bool downstreamOfEveryShape =
        llvm::all_of(maximalByShape, [&](const auto &entry) {
          return llvm::any_of(entry.second, [&](Value value) {
            return candidateAncestors.contains(value);
          });
        });
    if (!downstreamOfEveryShape) {
      continue;
    }

    SmallVector<int64_t> shape(candidateType.getShape());
    if (selected && selected->shape != shape) {
      return graphOp->emitError(
          "multi-output results have multiple incomparable maximal carrier "
          "shapes; split the results into multiple TensorIR graphs");
    }
    if (!selected || hasAncestor(candidate, selected->anchor)) {
      selected = CarrierCandidate{candidate, std::move(shape)};
    }
  }

  if (!selected) {
    return graphOp->emitError(
        "multi-output results have multiple incomparable maximal carrier "
        "shapes; split the results into multiple TensorIR graphs");
  }
  return *selected;
}

using LayoutMap = llvm::DenseMap<Value, LayoutSourceAttrInterface>;

/// Record `layout` for every result of `op`.
static void setLayout(LayoutMap &layouts, Operation *op,
                      LayoutSourceAttrInterface layout) {
  for (Value result : op->getResults()) {
    layouts[result] = layout;
  }
}

/// Process a single operation based on its type.
static LogicalResult processOperation(Operation *op, LayoutMap &layouts) {
  // Skip terminator operations, as they have no results.
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return success();
  }
  // ScanSlice body operations are lowered against loop-carried tiles by the scan_slice
  // conversion itself; they intentionally do not participate in graph-level
  // layout propagation.
  if (op->getParentOfType<ScanSliceOp>()) {
    return success();
  }

  // Operations with multiple or no results are not supported.
  if (op->getNumResults() != 1 && !isa<ReduceUDOp, ScanSliceOp>(op)) {
    op->emitError("unsupported operation");
    return failure();
  }

  // Only tensor results have layout propagation data.
  auto tensorTy = dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
  if (!tensorTy) {
    return success();
  }

  MLIRContext *ctx = op->getContext();

  // Operand-free tensor producers (constants, splats, iota).
  if (isa<ConstantOp, SplatOp>(op)) {
    tcg::Layout layout(getShapeRef(tensorTy),
                       tcg::repeat<tcg::Stride>(tensorTy.getRank(), 0));
    auto result = TensorSourceAttr::get(ctx, /*tensorId=*/-1, /*offset=*/0,
                                        layout.toString(),
                                        getDynamicValueMapping(layout));
    setLayout(layouts, op, result);
    return success();
  }

  if (auto iotaOp = dyn_cast<IotaOp>(op)) {
    int64_t dim = iotaOp.getDimension();
    tcg::Stride strides;
    for (unsigned i = 0; i < tensorTy.getRank(); ++i) {
      strides.append(i == static_cast<unsigned>(dim) ? 1 : 0);
    }
    tcg::Layout layout(getShapeRef(tensorTy), strides);
    auto result = TensorSourceAttr::get(ctx, /*tensorId=*/-1, /*offset=*/0,
                                        layout.toString(), {});
    setLayout(layouts, op, result);
    return success();
  }

  // Get the layouts of the operands.
  SmallVector<LayoutSourceAttrInterface> sources;
  for (Value operand : op->getOperands()) {
    LayoutSourceAttrInterface src = layouts.lookup(operand);
    if (!src) {
      if (auto ref = operand.getDefiningOp()) {
        ref->emitError("missing layout data");
      } else {
        op->emitError("unsupported argument");
      }
      return failure();
    }
    sources.push_back(src);
  }

  // Compute the resulting layout attribute.
  LayoutSourceAttrInterface result;
  bool combinesSources = op->hasTrait<OpTrait::Elementwise>();
  if (sources.size() > 1 && combinesSources) {
    // Handle operations that combine multiple source layouts.
    llvm::SmallSetVector<LayoutSourceAttrInterface, 4> srcSet;
    for (LayoutSourceAttrInterface src : sources) {
      if (auto composite = dyn_cast<CompositeSourceAttr>(src)) {
        for (size_t i = 0; i < composite.size(); ++i) {
          srcSet.insert(composite.getSource(i));
        }
      } else {
        srcSet.insert(src);
      }
    }

    SmallVector<LayoutSourceAttrInterface> srcList(srcSet.begin(),
                                                   srcSet.end());
    if (srcList.size() == 1) {
      result = srcList[0];
    } else {
      result = CompositeSourceAttr::get(ctx, srcList);
    }
  } else if (auto layoutProp = dyn_cast<LayoutPropInterface>(op)) {
    // Infer the result layout from operand layouts in the forward SSA
    // direction. The result retains source provenance for later passes that
    // materialize operand layouts while traversing the graph backwards.
    result = dyn_cast_or_null<LayoutSourceAttrInterface>(
        layoutProp.inferResultLayout(sources));
  } else {
    // Other operations are not supported.
    op->emitError("unsupported operation");
    return failure();
  }

  // Result may be invalid, e.g. if layout composition failed.
  if (!result) {
    op->emitError("failed to compute layout");
    return failure();
  }

  // Store the result, if successful.
  LLVM_DEBUG(llvm::dbgs() << op->getResult(0) << " layout: " << result << "\n");
  setLayout(layouts, op, result);
  return success();
}

/// Fill streaming-result metadata after the ordinary results have been
/// normalized. These views constrain tiling, while the scan_slice owns their stores.
static LogicalResult populateStreamingResultMetadata(
    const LayoutBoundary &boundary, ScanSliceOp scanSlice,
    ArrayRef<size_t> activeIndices,
    MutableArrayRef<LayoutSourceAttrInterface> resultLayouts,
    MutableArrayRef<TensorSourceAttr> resultViews) {
  Operation *graphOp = boundary.op;
  ArrayRef<Value> graphResults = boundary.results;
  MLIRContext *ctx = graphOp->getContext();
  if (scanSlice && scanSlice.getResults().size() > 1) {
    ScanSliceSourceAttr normalizedScanSliceSource;
    bool conflictingMappings = false;
    for (size_t resultIndex : activeIndices) {
      // Pointwise users can wrap the scan_slice in a composite source. Inspect the
      // complete rebuilt layout tree, not just its outermost attribute.
      resultLayouts[resultIndex].walk([&](ScanSliceSourceAttr candidate) {
        if (normalizedScanSliceSource &&
            normalizedScanSliceSource != candidate) {
          conflictingMappings = true;
        }
        normalizedScanSliceSource = candidate;
      });
    }
    if (conflictingMappings) {
      return graphOp->emitError(
          "scan_slice carry has multiple normalized layout mappings");
    }
    if (!normalizedScanSliceSource) {
      return graphOp->emitError(
          "cannot recover normalized scan_slice layout from graph results");
    }
    SmallVector<int64_t> carrierShape = normalizedScanSliceSource.getShape();
    for (auto [graphResultIndex, graphResult] : llvm::enumerate(graphResults)) {
      auto scanSliceResult = dyn_cast<OpResult>(graphResult);
      if (!scanSliceResult ||
          scanSliceResult.getOwner() != scanSlice.getOperation() ||
          scanSliceResult.getResultNumber() == 0) {
        continue;
      }
      // Stores remain inside the scan_slice loop, but their physical views still
      // constrain tiling: a projected output axis requires a full tile
      // along the corresponding carry axis (for a region reduction).
      size_t resultIndex = scanSliceResult.getResultNumber();
      TensorType resultType =
          cast<TensorType>(scanSlice.getResult(resultIndex).getType());
      int64_t scanDim = scanSlice.getOutputDimensions()[resultIndex - 1];
      auto slice = getScanSliceOutputSlice(resultType, scanDim);
      tcg::Stride stride;
      for (auto [index, pair] :
           llvm::enumerate(llvm::zip_equal(slice.shape, slice.strides))) {
        auto [dim, value] = pair;
        stride.append(dim == 1 && carrierShape[index] != 1 ? 0 : value);
      }
      tcg::Layout viewLayout(tcg::Shape(carrierShape), stride);
      resultLayouts[graphResultIndex] = normalizedScanSliceSource;
      resultViews[graphResultIndex] = TensorSourceAttr::get(
          ctx, /*tensorId=*/static_cast<int32_t>(graphResultIndex),
          /*offset=*/0, viewLayout.toString(),
          getDynamicValueMapping(viewLayout));
    }
  }

  return success();
}

/// Calculate the reduction underlying shape from the view layout.
/// Apply the reshape if the expected shape is different from the current.
static FailureOr<LayoutSourceAttrInterface>
maybeReshapeUnderlying(ReductionSourceAttr source) {
  SmallVector<int64_t> expectedShape;
  auto view = source.getCuteLayout();
  for (size_t i = 0, n = tcg::rank(view) - 1; i < n; i++) {
    auto part = tcg::get(view, i);
    if (part.stride().as_int() != 0) {
      expectedShape.push_back(tcg::is_static(part.shape())
                                  ? tcg::static_size(part.shape())
                                  : ShapedType::kDynamic);
    }
  }
  expectedShape.append(source.getReductionShape());

  auto underlying = source.getSource();
  if (underlying.getShape() != expectedShape) {
    underlying = underlying.reshape(expectedShape);
    if (!underlying) {
      return llvm::failure();
    }
  }
  return underlying;
}

/// Calculate the matmul underlying shape from the view layout.
/// Apply the reshape if the expected shape is different from the current.
static FailureOr<
    std::pair<LayoutSourceAttrInterface, LayoutSourceAttrInterface>>
maybeReshapeUnderlying(MatmulSourceAttr source) {
  SmallVector<int64_t> batchParts, lhsParts, rhsParts, contractingParts;
  auto view = tcg::flatten(source.getCuteLayout());
  // A reduction may permute the output view without permuting either matmul
  // operand. Reconstruct operand factors in physical, not view, order: each
  // B/M/N/K group is last-dimension-fastest, just as in MatmulSourceAttr.
  SmallVector<tcg::Layout> physicalParts;
  for (size_t i = 0, n = tcg::rank(view); i < n; i++) {
    physicalParts.push_back(tcg::get(view, i));
  }
  llvm::sort(physicalParts, [](const auto &lhs, const auto &rhs) {
    return lhs.stride().as_int() > rhs.stride().as_int();
  });
  for (const auto &part : physicalParts) {
    int64_t shape = part.shape().as_int();
    int64_t stride = part.stride().as_int();
    if (stride != 0) {
      if (stride < source.getK()) {
        contractingParts.push_back(shape);
      } else if (stride < source.getK() * source.getN()) {
        rhsParts.push_back(shape);
      } else if (stride < source.getK() * source.getN() * source.getM()) {
        lhsParts.push_back(shape);
      } else {
        batchParts.push_back(shape);
      }
    }
  }

  auto lhs = source.getLhs();
  SmallVector<int64_t> expectedLhsShape(batchParts);
  expectedLhsShape.append(lhsParts);
  expectedLhsShape.append(contractingParts);
  if (expectedLhsShape.empty()) {
    expectedLhsShape.push_back(1);
  }
  if (lhs.getShape() != expectedLhsShape) {
    lhs = lhs.reshape(expectedLhsShape);
  }

  auto rhs = source.getRhs();
  SmallVector<int64_t> expectedRhsShape(batchParts);
  expectedRhsShape.append(contractingParts);
  expectedRhsShape.append(rhsParts);
  if (expectedRhsShape.empty()) {
    expectedRhsShape.push_back(1);
  }
  if (rhs.getShape() != expectedRhsShape) {
    rhs = rhs.reshape(expectedRhsShape);
  }

  if (!lhs || !rhs) {
    return llvm::failure();
  }
  return std::make_pair(lhs, rhs);
}

/// Reduction layout reshape doesn't affect the underlying source (it only
/// updates the view), so the normalization could leave some layouts with
/// incorrect shapes. Rebuild the layout to fix this.
static FailureOr<LayoutSourceAttrInterface>
rebuildLayoutFixReductions(LayoutSourceAttrInterface source) {
  return llvm::TypeSwitch<LayoutSourceAttrInterface,
                          FailureOr<LayoutSourceAttrInterface>>(source)
      .Case<TensorSourceAttr>([&](auto attr) { return attr; })
      .Case<CompositeSourceAttr>(
          [&](auto attr) -> FailureOr<LayoutSourceAttrInterface> {
            SmallVector<LayoutSourceAttrInterface> sources;
            for (auto source : attr.getSources()) {
              MLIR_ASSIGN_OR_RETURN(auto res,
                                    rebuildLayoutFixReductions(source));
              sources.push_back(res);
            }
            return static_cast<LayoutSourceAttrInterface>(
                CompositeSourceAttr::get(attr.getContext(), sources));
          })
      .Case<ConcatSourceAttr>(
          [&](auto attr) -> FailureOr<LayoutSourceAttrInterface> {
            SmallVector<LayoutSourceAttrInterface> sources;
            for (auto source : attr.getSources()) {
              MLIR_ASSIGN_OR_RETURN(auto res,
                                    rebuildLayoutFixReductions(source));
              sources.push_back(res);
            }
            return static_cast<LayoutSourceAttrInterface>(
                ConcatSourceAttr::get(attr.getContext(), attr.getDimension(),
                                      sources, attr.getArgumentIndex()));
          })
      .Case<ReductionSourceAttr>(
          [&](auto attr) -> FailureOr<LayoutSourceAttrInterface> {
            MLIR_ASSIGN_OR_RETURN(auto src, maybeReshapeUnderlying(attr));
            MLIR_ASSIGN_OR_RETURN(auto res, rebuildLayoutFixReductions(src));
            return static_cast<LayoutSourceAttrInterface>(
                ReductionSourceAttr::get(attr.getContext(), attr.getView(),
                                         res));
          })
      .Case<MatmulSourceAttr>([&](auto attr)
                                  -> FailureOr<LayoutSourceAttrInterface> {
        MLIR_ASSIGN_OR_RETURN(auto src, maybeReshapeUnderlying(attr));
        MLIR_ASSIGN_OR_RETURN(auto lhs, rebuildLayoutFixReductions(src.first));
        MLIR_ASSIGN_OR_RETURN(auto rhs, rebuildLayoutFixReductions(src.second));
        return static_cast<LayoutSourceAttrInterface>(MatmulSourceAttr::get(
            attr.getContext(), attr.getView(), attr.getB(), attr.getM(),
            attr.getN(), attr.getK(), lhs, rhs));
      })
      .Case<ScanSliceSourceAttr>(
          [&](auto attr) -> FailureOr<LayoutSourceAttrInterface> {
            MLIR_ASSIGN_OR_RETURN(auto carry,
                                  rebuildLayoutFixReductions(attr.getCarry()));
            return static_cast<LayoutSourceAttrInterface>(
                ScanSliceSourceAttr::get(attr.getContext(), carry,
                                         attr.getOutputDimensions(),
                                         attr.getInputs()));
          })
      .Default([&](auto attr) { return attr; });
}

/// Normalize a single non-scan_slice result together with its physical output
/// layout. Only the input layouts are kept: the output is excluded from the
/// iteration space and handled separately by its consumers.
static FailureOr<LayoutNormalization>
normalizeSingleResult(const LayoutPropagationAnalysis &propagation) {
  const LayoutBoundary &boundary = propagation.getBoundary();
  Operation *graphOp = boundary.op;
  MLIRContext *ctx = graphOp->getContext();
  Value graphResult = boundary.results.front();

  // For copy/passthrough graphs the result is a boundary input, so its layout
  // is the input layout.
  LayoutSourceAttrInterface layoutAttr = propagation.getLayout(graphResult);
  if (!layoutAttr) {
    if (Operation *resultOp = graphResult.getDefiningOp()) {
      return resultOp->emitError("missing layout data");
    }
    return graphOp->emitError("missing layout data for copy fusion input");
  }

  TensorSourceAttr output = boundary.outputs.front();
  if (!output) {
    return graphOp->emitError(
        "layout propagation requires a ranked tensor result");
  }
  auto resultSource = TensorSourceAttr::get(ctx, /*tensorId=*/0, /*offset=*/0,
                                            output.getLayout(),
                                            output.getDynamicValueMapping());
  auto compositeSource =
      CompositeSourceAttr::get(ctx, {layoutAttr, resultSource});

  // Run the normalization on the composite source.
  auto normalizedSource = compositeSource.normalize();
  if (!normalizedSource) {
    return graphOp->emitError("failed to normalize layout");
  }

  // Rebuild the input layout to fix reductions.
  auto normalizedResult = rebuildLayoutFixReductions(
      cast<CompositeSourceAttr>(normalizedSource).getSource(0));
  if (failed(normalizedResult)) {
    return graphOp->emitError("failed to reshape reduction layout");
  }

  LayoutNormalization normalization;
  normalization.iterationSpace = *normalizedResult;
  return normalization;
}

/// Normalize graph results in a common carrier. Streaming scan_slice outputs do
/// not participate in carrier selection because scan_slice conversion stores them
/// inside the loop; their metadata remains ABI ordered for descriptor lookup.
static FailureOr<LayoutNormalization>
normalizeMultipleResults(const LayoutPropagationAnalysis &propagation) {
  const LayoutBoundary &boundary = propagation.getBoundary();
  Operation *graphOp = boundary.op;
  MLIRContext *ctx = graphOp->getContext();
  ArrayRef<Value> graphResults = boundary.results;

  ScanSliceOp scanSlice;
  for (ScanSliceOp candidate : boundary.body->getOps<ScanSliceOp>()) {
    if (scanSlice) {
      return graphOp->emitError(
          "layout propagation supports at most one top-level scan_slice");
    }
    scanSlice = candidate;
  }

  if (scanSlice && !isa_and_nonnull<ScanSliceSourceAttr>(
                       propagation.getLayout(scanSlice.getResult(0)))) {
    return scanSlice.emitError("missing scan_slice source layout");
  }

  SmallVector<size_t> activeIndices;
  SmallVector<Value> activeResults;
  SmallVector<LayoutSourceAttrInterface> terminalLayouts;
  SmallVector<TensorSourceAttr> physicalSources;
  terminalLayouts.reserve(graphResults.size());
  physicalSources.reserve(graphResults.size());
  for (auto [index, result] : llvm::enumerate(graphResults)) {
    if (isStreamingScanSliceResult(result)) {
      continue;
    }
    activeIndices.push_back(index);
    activeResults.push_back(result);
    MLIR_ASSIGN_OR_RETURN(auto layout, getResultLayout(propagation, result));
    terminalLayouts.push_back(layout);
    MLIR_ASSIGN_OR_RETURN(auto source,
                          getPhysicalResultSource(boundary, index));
    physicalSources.push_back(source);
  }
  if (activeResults.empty()) {
    return graphOp->emitError(
        "scan_slice graph requires a non-streaming result carrying its layout");
  }

  SmallVector<LayoutSourceAttrInterface> liftedLayouts;
  SmallVector<TensorSourceAttr> liftedViews;
  liftedLayouts.reserve(activeResults.size());
  liftedViews.reserve(activeResults.size());
  CarrierCandidate carrier;
  if (activeResults.size() == 1) {
    carrier.anchor = activeResults.front();
    carrier.shape =
        llvm::to_vector(cast<TensorType>(carrier.anchor.getType()).getShape());
  } else {
    MLIR_ASSIGN_OR_RETURN(carrier, deriveCommonCarrier(boundary, activeResults,
                                                       terminalLayouts,
                                                       physicalSources));
  }
  for (auto [activeIndex, tuple] : llvm::enumerate(
           llvm::zip_equal(activeResults, terminalLayouts, physicalSources))) {
    auto [result, layout, source] = tuple;
    auto lifted = liftResultToCarrier(result, carrier.anchor, layout, source,
                                      carrier.shape);
    if (failed(lifted)) {
      return graphOp->emitError()
             << "failed to lift result " << activeIndices[activeIndex]
             << " into carrier " << vectorToString(carrier.shape)
             << " through its SSA layout path";
    }
    liftedLayouts.push_back(lifted->layout);
    liftedViews.push_back(lifted->view);
  }

  SmallVector<LayoutSourceAttrInterface> combinedSources(liftedLayouts);
  combinedSources.append(liftedViews.begin(), liftedViews.end());
  auto combined = CompositeSourceAttr::get(ctx, combinedSources);
  auto normalized = dyn_cast_or_null<CompositeSourceAttr>(combined.normalize());
  if (!normalized) {
    return graphOp->emitError(
        "failed to jointly normalize graph result layouts");
  }
  LayoutNormalization normalization;
  SmallVector<LayoutSourceAttrInterface> &resultLayouts =
      normalization.resultLayouts;
  SmallVector<TensorSourceAttr> &resultViews = normalization.resultViews;
  resultLayouts.resize(graphResults.size());
  resultViews.resize(graphResults.size());
  for (auto [index, graphResultIndex] : llvm::enumerate(activeIndices)) {
    MLIR_ASSIGN_OR_RETURN(
        auto rebuilt, rebuildLayoutFixReductions(normalized.getSource(index)));
    resultLayouts[graphResultIndex] = rebuilt;
    auto view = dyn_cast<TensorSourceAttr>(
        normalized.getSource(activeResults.size() + index));
    if (!view) {
      return graphOp->emitError()
             << "normalized physical result view " << activeIndices[index]
             << " is not a TensorSourceAttr";
    }
    resultViews[graphResultIndex] = view;
  }

  MLIR_RETURN_IF_ERROR(populateStreamingResultMetadata(
      boundary, scanSlice, activeIndices, resultLayouts, resultViews));

  // Build the root input-source union in stable semantic order. Exact
  // duplicates are kept only once, which also ensures a multi-result
  // ReduceUDOp is loaded/materialized once for identical result layouts.
  SmallVector<std::pair<std::string, LayoutSourceAttrInterface>> keyed;
  keyed.reserve(activeIndices.size());
  for (size_t graphResultIndex : activeIndices) {
    LayoutSourceAttrInterface layout = resultLayouts[graphResultIndex];
    std::string spelling;
    llvm::raw_string_ostream os(spelling);
    layout.print(os);
    keyed.emplace_back(os.str(), layout);
  }
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  SmallVector<LayoutSourceAttrInterface> rootSources;
  for (const auto &[spelling, layout] : keyed) {
    if (rootSources.empty() || rootSources.back() != layout) {
      rootSources.push_back(layout);
    }
  }
  normalization.iterationSpace =
      rootSources.size() == 1 ? rootSources.front()
                              : static_cast<LayoutSourceAttrInterface>(
                                    CompositeSourceAttr::get(ctx, rootSources));
  return normalization;
}

} // namespace

FailureOr<LayoutBoundary> LayoutBoundary::get(GraphOp graph) {
  LayoutBoundary boundary;
  boundary.op = graph;
  boundary.body = graph.getBody();
  boundary.hasDynamicShapes = hasDynamicInputOrOutputTensor(graph);

  for (BlockArgument argument : graph.getArguments()) {
    auto tensorType = dyn_cast<RankedTensorType>(argument.getType());
    if (!tensorType || !isTensorType(tensorType)) {
      continue;
    }
    MLIR_ASSIGN_OR_RETURN(tcg::Stride stride, getStrideFromGraph(argument));
    tcg::Layout layout(getShapeRef(tensorType), stride);
    boundary.inputs[argument] = TensorSourceAttr::get(
        graph.getContext(), argument.getArgNumber(), /*offset=*/0,
        layout.toString(), getDynamicValueMapping(layout));
  }

  llvm::append_range(boundary.results,
                     graph.getBody()->getTerminator()->getOperands());
  ArrayAttr resultAttrs = graph.getAllResultAttrs();
  for (auto [resultIndex, type] : llvm::enumerate(graph.getResultTypes())) {
    auto resultType = dyn_cast<RankedTensorType>(type);
    if (!resultType) {
      boundary.outputs.push_back(nullptr);
      continue;
    }
    std::optional<tcg::Stride> outputStride;
    if (auto dict = resultAttrs
                        ? dyn_cast<DictionaryAttr>(resultAttrs[resultIndex])
                        : DictionaryAttr()) {
      if (auto strideAttr =
              dict.getAs<StringAttr>(TensorIRDialect::getStrideAttrName())) {
        outputStride = tcg::from_string<tcg::Stride>(strideAttr.str());
        if (!outputStride) {
          return graph.emitError() << "invalid output stride for result "
                                   << resultIndex << ": " << strideAttr;
        }
      }
    }
    tcg::Layout outputLayout =
        outputStride ? tcg::Layout(getShapeRef(resultType), *outputStride)
                     : tcg::Layout(getShapeRef(resultType));
    int32_t tensorId =
        static_cast<int32_t>(graph.getNumArguments() + resultIndex);
    boundary.outputs.push_back(TensorSourceAttr::get(
        graph.getContext(), tensorId, /*offset=*/0, outputLayout.toString(),
        getDynamicValueMapping(outputLayout)));
  }
  return boundary;
}

FailureOr<LayoutPropagationAnalysis>
LayoutPropagationAnalysis::compute(const LayoutBoundary &boundary) {
  for (MatmulOp matmulOp : boundary.body->getOps<MatmulOp>()) {
    if (matmulOp.getAcc()) {
      return matmulOp.emitError(
          "custom accumulator is not supported by layout propagation");
    }
  }

  LayoutPropagationAnalysis analysis(boundary);
  for (auto [value, layout] : boundary.inputs) {
    analysis.layouts[value] = layout;
  }
  for (Operation &op : *boundary.body) {
    if (boundary.boundaryOps.contains(&op)) {
      continue;
    }
    WalkResult result = op.walk([&](Operation *nested) {
      return succeeded(processOperation(nested, analysis.layouts))
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      return boundary.op->emitError("layout propagation failed");
    }
  }
  return analysis;
}

FailureOr<LayoutNormalization>
LayoutNormalization::compute(const LayoutPropagationAnalysis &propagation) {
  const LayoutBoundary &boundary = propagation.getBoundary();
  if (boundary.results.empty()) {
    return boundary.op->emitError(
        "layout propagation requires at least one output");
  }
  bool hasScanSlice = !boundary.body->getOps<ScanSliceOp>().empty();
  if (boundary.results.size() > 1 || hasScanSlice) {
    return normalizeMultipleResults(propagation);
  }
  return normalizeSingleResult(propagation);
}

FailureOr<LayoutNormalization> LayoutNormalization::compute(GraphOp graph) {
  MLIR_ASSIGN_OR_RETURN(LayoutBoundary boundary, LayoutBoundary::get(graph));
  MLIR_ASSIGN_OR_RETURN(LayoutPropagationAnalysis propagation,
                        LayoutPropagationAnalysis::compute(boundary));
  return compute(propagation);
}

} // namespace mlir::nv_tensor_ir
