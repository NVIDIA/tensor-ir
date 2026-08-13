// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"
#include "tensor_ir/Transform/Passes.h"
#include "tensor_ir/Utils/Utils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "layout-propagation-annotation"

namespace mlir::nv_tensor_ir {

// Include the generated pass definitions
#define GEN_PASS_DEF_LAYOUTPROPAGATIONANNOTATIONPASS
#define GEN_PASS_DEF_LAYOUTPROPAGATIONNORMALIZATIONPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace {

namespace tcg = mlir::nv_tensor_ir::tcutegen;

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
    if (graphResults.size() != 1) {
      graphOp.emitError("unsupported multi-output fusion");
      return signalPassFailure();
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
              MLIR_ASSIGN_OR_RETURN(auto lhs,
                                    rebuildLayoutFixReductions(attr.getLhs()));
              MLIR_ASSIGN_OR_RETURN(auto rhs,
                                    rebuildLayoutFixReductions(attr.getRhs()));
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
        expectedShape.push_back(part.shape().as_int());
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
};

} // namespace mlir::nv_tensor_ir
