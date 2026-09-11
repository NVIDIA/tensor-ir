// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"
#include "tensor_ir/Transform/Passes.h"
#include "tensor_ir/Utils/Utils.h"

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
#include <utility>

#define DEBUG_TYPE "layout-propagation-annotation"

namespace mlir::nv_tensor_ir {

// Include the generated pass definitions
#define GEN_PASS_DEF_LAYOUTPROPAGATIONANNOTATIONPASS
#define GEN_PASS_DEF_LAYOUTPROPAGATIONNORMALIZATIONPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace {

namespace tcg = mlir::nv_tensor_ir::tcutegen;

/// Return the layout attached to a graph result value. Copy/passthrough
/// results are block arguments and obtain their layout from the graph
/// signature instead of a defining operation.
static FailureOr<LayoutSourceAttrInterface> getGraphResultLayout(Value value) {
  auto layout =
      dyn_cast_or_null<LayoutSourceAttrInterface>(getLayoutSourceAttr(value));
  if (!layout) {
    if (Operation *definingOp = value.getDefiningOp()) {
      definingOp->emitError("missing layout data for graph result");
    }
    return failure();
  }
  return layout;
}

/// Build the physical output layout for result `resultIndex`. Output tensor
/// IDs follow graph inputs in the converted kernel signature. These sources
/// are kept in result_views and are not added to the root input-source union.
static FailureOr<TensorSourceAttr>
buildPhysicalResultSource(GraphOp graphOp, size_t resultIndex) {
  auto resultType = dyn_cast<TensorType>(graphOp.getResultTypes()[resultIndex]);
  if (!resultType) {
    return graphOp.emitError() << "multi-output layout propagation requires "
                                  "tensor result "
                               << resultIndex;
  }

  std::optional<tcg::Stride> outputStride;
  if (auto resultAttrs = graphOp.getAllResultAttrs()) {
    if (auto dict = dyn_cast<DictionaryAttr>(resultAttrs[resultIndex])) {
      if (auto strideAttr =
              dict.getAs<StringAttr>(TensorIRDialect::getStrideAttrName())) {
        outputStride = tcg::from_string<tcg::Stride>(strideAttr.str());
        if (!outputStride) {
          return graphOp.emitError() << "invalid output stride for result "
                                     << resultIndex << ": " << strideAttr;
        }
      }
    }
  }

  tcg::Layout outputLayout =
      outputStride ? tcg::Layout(getShapeRef(resultType), *outputStride)
                   : tcg::Layout(getShapeRef(resultType));
  int32_t tensorId =
      static_cast<int32_t>(graphOp.getArgumentTypes().size() + resultIndex);
  return TensorSourceAttr::get(graphOp.getContext(), tensorId, /*offset=*/0,
                               outputLayout.toString(),
                               getDynamicValueMapping(outputLayout));
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
    for (Value operand : definingOp->getOperands()) {
      collectAncestors(operand, ancestors);
    }
  }
}

static bool isCarrierAnchor(Value value) {
  auto type = dyn_cast<TensorType>(value.getType());
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
  std::function<void(Value, LayoutSourceAttrInterface, TensorSourceAttr, bool)>
      visit = [&](Value value, LayoutSourceAttrInterface currentLayout,
                  TensorSourceAttr currentView, bool transportCoordinates) {
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

        if (auto transpose = transportCoordinates
                                 ? dyn_cast<TransposeOp>(definingOp)
                                 : nullptr) {
          Value operand = transpose.getInput();
          if (!hasAncestor(operand, carrierAnchor)) {
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
          if (!hasAncestor(operand, carrierAnchor)) {
            return;
          }
          ArrayRef<int64_t> operandShape =
              cast<TensorType>(operand.getType()).getShape();
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

        bool transparent = isa<PointwiseOpInterface>(definingOp) &&
                           !isa<BroadcastOp>(definingOp);
        bool transportOperands = transportCoordinates && transparent;
        for (Value operand : definingOp->getOperands()) {
          if (!isa<TensorType>(operand.getType()) ||
              !hasAncestor(operand, carrierAnchor)) {
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
deriveCommonCarrier(GraphOp graphOp,
                    ArrayRef<LayoutSourceAttrInterface> resultLayouts,
                    ArrayRef<TensorSourceAttr> resultSources) {
  auto graphResults = graphOp.getResults();
  assert(graphResults.size() > 1 && "multi-result helper requires N > 1");

  if (hasDynamicInputOrOutputTensor(graphOp)) {
    return graphOp.emitError(
        "dynamic multi-output layout propagation is not yet supported");
  }

  SmallVector<llvm::DenseSet<Value>> ancestry(graphResults.size());
  for (auto [index, result] : llvm::enumerate(graphResults)) {
    collectAncestors(result, ancestry[index]);
  }

  SmallVector<Value> orderedValues(graphOp.getArguments());
  for (Operation &op : graphOp.getBody()->getOperations()) {
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
        cast<TensorType>(candidate.getType()).getShape();
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
    return graphOp.emitError(
        "multi-output results have no compatible nonconstant common SSA "
        "ancestor; split the results into multiple TensorIR graphs");
  }

  int64_t maxElements = 0;
  for (Value value : commonValues) {
    int64_t elements = cast<TensorType>(value.getType()).getNumElements();
    maxElements = std::max(maxElements, elements);
  }

  std::map<SmallVector<int64_t>, SmallVector<Value>> maximalByShape;
  for (Value value : commonValues) {
    auto type = cast<TensorType>(value.getType());
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
    auto candidateType = cast<TensorType>(candidate.getType());
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
      return graphOp.emitError(
          "multi-output results have multiple incomparable maximal carrier "
          "shapes; split the results into multiple TensorIR graphs");
    }
    if (!selected || hasAncestor(candidate, selected->anchor)) {
      selected = CarrierCandidate{candidate, std::move(shape)};
    }
  }

  if (!selected) {
    return graphOp.emitError(
        "multi-output results have multiple incomparable maximal carrier "
        "shapes; split the results into multiple TensorIR graphs");
  }
  return *selected;
}

/// Process a single operation based on its type.
LogicalResult processOperation(Operation *op) {
  // Skip terminator operations, as they have no results.
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return success();
  }

  // Operations with multiple or no results are not supported.
  if (op->getNumResults() != 1 && !isa<ReduceUDOp>(op)) {
    op->emitError("unsupported operation");
    return failure();
  }

  // Only tensor results have layout propagation data.
  auto tensorTy = dyn_cast<TensorType>(op->getResultTypes()[0]);
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
    op->setAttr(TensorIRDialect::getLayoutAttrName(), result);
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
    op->setAttr(TensorIRDialect::getLayoutAttrName(), result);
    return success();
  }

  // Get layout attributes for the operands.
  SmallVector<LayoutSourceAttrInterface> sources;
  for (Value operand : op->getOperands()) {
    auto src = dyn_cast_or_null<LayoutSourceAttrInterface>(
        getLayoutSourceAttr(operand));
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
  Attribute result;
  bool combinesSources = isa<PointwiseOpInterface>(op);
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
    result = layoutProp.inferResultLayout(sources);
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
  op->setAttr(TensorIRDialect::getLayoutAttrName(), result);
  return success();
}

} // namespace

struct LayoutPropagationAnnotationPass
    : public impl::LayoutPropagationAnnotationPassBase<
          LayoutPropagationAnnotationPass> {

  void runOnOperation() override {
    // Run layout propagation for each graph operation.
    GraphOp graphOp = getOperation();
    for (MatmulOp matmulOp : graphOp.getOps<MatmulOp>()) {
      if (matmulOp.getAcc()) {
        matmulOp.emitError(
            "custom accumulator is not supported by layout propagation");
        return signalPassFailure();
      }
    }

    auto result = graphOp.walk([&](Operation *op) {
      return op == graphOp || succeeded(processOperation(op))
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      graphOp.emitError("layout propagation failed");
      signalPassFailure();
    }
  }
};

struct LayoutPropagationNormalizationPass
    : public impl::LayoutPropagationNormalizationPassBase<
          LayoutPropagationNormalizationPass> {

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    MLIRContext *ctx = graphOp.getContext();

    // Get the graph output value.
    auto graphResults = graphOp.getResults();
    if (graphResults.empty()) {
      graphOp.emitError("layout propagation requires at least one output");
      return signalPassFailure();
    }
    if (graphResults.size() > 1) {
      if (failed(normalizeMultipleResults(graphOp))) {
        return signalPassFailure();
      }
      return;
    }

    // Get the layout attribute for the graph result.  For copy/passthrough
    // graphs the result is a block argument, so derive layout from the input.
    LayoutSourceAttrInterface layoutAttr;
    if (auto resultOp = graphResults.front().getDefiningOp()) {
      layoutAttr = resultOp->getAttrOfType<LayoutSourceAttrInterface>(
          TensorIRDialect::getLayoutAttrName());
      if (!layoutAttr) {
        resultOp->emitError("missing layout data");
        return signalPassFailure();
      }
    } else {
      layoutAttr = dyn_cast_or_null<LayoutSourceAttrInterface>(
          getLayoutSourceAttr(graphResults.front()));
      if (!layoutAttr) {
        graphOp.emitError("missing layout data for copy fusion input");
        return signalPassFailure();
      }
    }

    // Build the output layout from res_attrs directly.  For copy/passthrough
    // graphs the result value is the block argument, so getStrideFromGraph
    // would return the input stride (value identity matches arg_attrs first).
    auto resultTy = cast<TensorType>(graphResults.front().getType());
    std::optional<tcg::Stride> outputStride;
    if (auto resAttrs = graphOp.getAllResultAttrs()) {
      if (auto dictAttr = dyn_cast<DictionaryAttr>(resAttrs[0])) {
        if (auto strideAttr = dictAttr.getAs<StringAttr>(
                TensorIRDialect::getStrideAttrName())) {
          outputStride = tcg::from_string<tcg::Stride>(strideAttr.str());
          if (!outputStride) {
            graphOp.emitError("invalid output stride: ") << strideAttr;
            return signalPassFailure();
          }
        }
      }
    }
    auto resultLayout = outputStride
                            ? tcg::Layout(getShapeRef(resultTy), *outputStride)
                            : tcg::Layout(getShapeRef(resultTy));
    auto resultSource = TensorSourceAttr::get(
        ctx, /*tensorId=*/0, /*offset=*/0, resultLayout.toString(),
        getDynamicValueMapping(resultLayout));
    auto compositeSource =
        CompositeSourceAttr::get(ctx, {layoutAttr, resultSource});

    // Run the normalization on the composite source.
    auto normalizedSource = compositeSource.normalize();
    if (!normalizedSource) {
      graphOp.emitError("failed to normalize layout");
      return signalPassFailure();
    }

    // Rebuild the input layout to fix reductions.
    auto normalizedResult = rebuildLayoutFixReductions(
        cast<CompositeSourceAttr>(normalizedSource).getSource(0));
    if (failed(normalizedResult)) {
      graphOp.emitError("failed to reshape reduction layout");
      return signalPassFailure();
    }

    // Store only the input layouts (first child of the normalized composite).
    // The output is excluded: codegen's extractTensorSources expects only
    // input TensorSourceAttr children and handles the output separately
    // via computeColMajorStrides.  TileAnalyzerPass derives output strides
    // from the graph's result stride attributes (getResultStrides).
    auto graphResultsOp = graphOp.getBody()->getTerminator();
    graphResultsOp->setAttr(TensorIRDialect::getIterationSpaceAttrName(),
                            *normalizedResult);
  }

private:
  /// Jointly normalize every terminal result and physical output view in a
  /// common root carrier. Result-specific layouts remain ABI ordered, while
  /// the root input-source union is sorted by attribute spelling so changing
  /// result order cannot change load/materialization order.
  static LogicalResult normalizeMultipleResults(GraphOp graphOp) {
    MLIRContext *ctx = graphOp.getContext();
    auto graphResults = graphOp.getResults();

    SmallVector<LayoutSourceAttrInterface> terminalLayouts;
    SmallVector<TensorSourceAttr> physicalSources;
    terminalLayouts.reserve(graphResults.size());
    physicalSources.reserve(graphResults.size());
    for (auto [index, result] : llvm::enumerate(graphResults)) {
      MLIR_ASSIGN_OR_RETURN(auto layout, getGraphResultLayout(result));
      terminalLayouts.push_back(layout);
      MLIR_ASSIGN_OR_RETURN(auto source,
                            buildPhysicalResultSource(graphOp, index));
      physicalSources.push_back(source);
    }

    MLIR_ASSIGN_OR_RETURN(
        CarrierCandidate carrier,
        deriveCommonCarrier(graphOp, terminalLayouts, physicalSources));

    SmallVector<LayoutSourceAttrInterface> liftedLayouts;
    SmallVector<TensorSourceAttr> liftedViews;
    liftedLayouts.reserve(graphResults.size());
    liftedViews.reserve(graphResults.size());
    for (auto [index, tuple] : llvm::enumerate(
             llvm::zip_equal(graphResults, terminalLayouts, physicalSources))) {
      auto [result, layout, source] = tuple;
      auto lifted = liftResultToCarrier(result, carrier.anchor, layout, source,
                                        carrier.shape);
      if (failed(lifted)) {
        return graphOp.emitError()
               << "failed to lift result " << index << " into carrier "
               << vectorToString(carrier.shape)
               << " through its SSA layout path";
      }
      liftedLayouts.push_back(lifted->layout);
      liftedViews.push_back(lifted->view);
    }

    SmallVector<LayoutSourceAttrInterface> combinedSources(liftedLayouts);
    combinedSources.append(liftedViews.begin(), liftedViews.end());
    auto combined = CompositeSourceAttr::get(ctx, combinedSources);
    auto normalized =
        dyn_cast_or_null<CompositeSourceAttr>(combined.normalize());
    if (!normalized) {
      return graphOp.emitError(
          "failed to jointly normalize multi-output result layouts");
    }

    SmallVector<LayoutSourceAttrInterface> resultLayouts;
    SmallVector<TensorSourceAttr> resultViews;
    resultLayouts.reserve(graphResults.size());
    resultViews.reserve(graphResults.size());
    for (size_t index = 0; index < graphResults.size(); ++index) {
      MLIR_ASSIGN_OR_RETURN(auto rebuilt, rebuildLayoutFixReductions(
                                              normalized.getSource(index)));
      resultLayouts.push_back(rebuilt);

      auto view = dyn_cast<TensorSourceAttr>(
          normalized.getSource(graphResults.size() + index));
      if (!view) {
        return graphOp.emitError() << "normalized physical result view "
                                   << index << " is not a TensorSourceAttr";
      }
      resultViews.push_back(view);
    }

    // Build the root input-source union in stable semantic order. Exact
    // duplicates are kept only once, which also ensures a multi-result
    // ReduceUDOp is loaded/materialized once for identical result layouts.
    SmallVector<std::pair<std::string, LayoutSourceAttrInterface>> keyed;
    keyed.reserve(resultLayouts.size());
    for (LayoutSourceAttrInterface layout : resultLayouts) {
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
    LayoutSourceAttrInterface rootLayout =
        rootSources.size() == 1
            ? rootSources.front()
            : static_cast<LayoutSourceAttrInterface>(
                  CompositeSourceAttr::get(ctx, rootSources));

    auto resultsOp = graphOp.getBody()->getTerminator();
    resultsOp->setAttr(TensorIRDialect::getIterationSpaceAttrName(),
                       rootLayout);

    SmallVector<Attribute> layoutAttrs(resultLayouts.begin(),
                                       resultLayouts.end());
    SmallVector<Attribute> viewAttrs(resultViews.begin(), resultViews.end());
    resultsOp->setAttr(TensorIRDialect::getResultLayoutsAttrName(),
                       ArrayAttr::get(ctx, layoutAttrs));
    resultsOp->setAttr(TensorIRDialect::getResultViewsAttrName(),
                       ArrayAttr::get(ctx, viewAttrs));
    return success();
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
        .Case<MatmulSourceAttr>(
            [&](auto attr) -> FailureOr<LayoutSourceAttrInterface> {
              MLIR_ASSIGN_OR_RETURN(auto src, maybeReshapeUnderlying(attr));
              MLIR_ASSIGN_OR_RETURN(auto lhs,
                                    rebuildLayoutFixReductions(src.first));
              MLIR_ASSIGN_OR_RETURN(auto rhs,
                                    rebuildLayoutFixReductions(src.second));
              return static_cast<LayoutSourceAttrInterface>(
                  MatmulSourceAttr::get(attr.getContext(), attr.getView(),
                                        attr.getB(), attr.getM(), attr.getN(),
                                        attr.getK(), lhs, rhs));
            })
        .Default([&](auto attr) { return attr; });
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
    for (size_t i = 0, n = tcg::rank(view); i < n; i++) {
      auto part = tcg::get(view, i);
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
};

} // namespace mlir::nv_tensor_ir
