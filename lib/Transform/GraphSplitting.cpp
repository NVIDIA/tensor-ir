// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Dialect/TensorIRAttrs.h"
#include "tensor_ir/Transform/Passes.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tensor-graph-splitting"

//===----------------------------------------------------------------------===//
// Graph Splitting Pass — Algorithm
//===----------------------------------------------------------------------===//
// Converts reconvergent DAGs into trees. When the same SSA value
// is consumed by multiple paths that require different layouts (different
// strides or offsets), the pass clones the value's defining operation — one
// copy per unique layout.
//===----------------------------------------------------------------------===//

namespace mlir::nv_tensor_ir {

// Include the generated pass definitions.
#define GEN_PASS_DEF_GRAPHSPLITTINGPASS
#include "tensor_ir/Transform/Passes.h.inc"

struct CacheKey {
  Operation *op = nullptr;
  int32_t iterSpaceId;
  Attribute layout;

  bool operator==(const CacheKey &other) const {
    return op == other.op && iterSpaceId == other.iterSpaceId &&
           layout == other.layout;
  }
};

} // namespace mlir::nv_tensor_ir

namespace llvm {
template <>
struct DenseMapInfo<mlir::nv_tensor_ir::CacheKey> {
  static inline mlir::nv_tensor_ir::CacheKey getEmptyKey() {
    return {nullptr, -1, mlir::Attribute()};
  }
  static inline mlir::nv_tensor_ir::CacheKey getTombstoneKey() {
    return {nullptr, -2, mlir::Attribute()};
  }
  static inline unsigned getHashValue(const mlir::nv_tensor_ir::CacheKey &val) {
    return llvm::hash_combine(val.op, val.iterSpaceId, val.layout);
  }
  static bool isEqual(const mlir::nv_tensor_ir::CacheKey &lhs,
                      const mlir::nv_tensor_ir::CacheKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace mlir::nv_tensor_ir {
namespace {

struct GraphSplittingPass
    : public impl::GraphSplittingPassBase<GraphSplittingPass> {

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    auto resultsOp = cast<ResultsOp>(graphOp.getBody()->getTerminator());
    OpBuilder builder(resultsOp);

    // For each (op, layout), store the resulting values.
    llvm::DenseMap<CacheKey, SmallVector<Value>> cache;

    // Number of iteration spaces created (in addition to the default).
    // Operations that create a new iteration space (concat, reduce, matmul)
    // increment this counter.
    int32_t iterSpaceCounter = 0;

    // Recursive materialization: clone operations per (op, layout).
    std::function<FailureOr<Value>(Value, int32_t, LayoutSourceAttrInterface)>
        materialize =
            [&](Value value, int32_t iterSpaceId,
                LayoutSourceAttrInterface layout) -> FailureOr<Value> {
      // If the value is a block argument, return it as-is.
      Operation *op = value.getDefiningOp();
      if (op == nullptr) {
        auto blockArg = cast<BlockArgument>(value);
        if (!isa<TensorSourceAttr>(layout)) {
          graphOp.emitError() << "expected tensor layout for argument #"
                              << blockArg.getArgNumber();
          return failure();
        }
        return value;
      }

      // If the operation has multiple results, find the value index.
      size_t resultIndex = 0;
      if (op->getNumResults() > 1) {
        for (auto [idx, result] : llvm::enumerate(op->getResults())) {
          if (result == value) {
            resultIndex = idx;
            break;
          }
        }
      }

      // If the result is already in the cache, return it.
      CacheKey key{op, iterSpaceId, layout};
      if (auto it = cache.find(key); it != cache.end()) {
        return it->second[resultIndex];
      }

      // Collect operand layouts (infer from the current layout).
      SmallVector<LayoutSourceAttrInterface> operandLayouts;
      // Number of iteration spaces added by the operation.
      int newIterationSpaces = 0;

      // Case #1: pointwise operations.
      if (isa<PointwiseOpInterface>(op) && !isa<BroadcastOp>(op)) {
        auto composite =
            dyn_cast_or_null<CompositeSourceAttr>(getLayoutSourceAttr(value));
        if (!composite || op->getNumOperands() == 1) {
          // Case #1a: unary or binary pointwise with the same layout.
          operandLayouts = SmallVector<LayoutSourceAttrInterface>(
              op->getNumOperands(), layout);
        } else {
          // Case #1b: remap operand layouts to the iteration space.
          auto target = dyn_cast<CompositeSourceAttr>(layout);
          if (!target || target.size() != composite.size()) {
            op->emitError("expected matching composite layout");
            return failure();
          }

          // Both the annotated (forward-propagated) and the argument
          // (backward-propagated) layouts are composites of the same size.
          auto originalSources = composite.getSources();
          for (auto operand : op->getOperands()) {
            // Get the operand's layout(s).
            auto operandLayout =
                cast<LayoutSourceAttrInterface>(getLayoutSourceAttr(operand));
            ArrayRef<LayoutSourceAttrInterface> operandSources(operandLayout);
            if (auto operandComposite =
                    dyn_cast<CompositeSourceAttr>(operandLayout)) {
              operandSources = operandComposite.getSources();
            }

            // Re-map the original sources to the iteration space.
            SmallVector<LayoutSourceAttrInterface> remappedSources;
            for (auto source : operandSources) {
              auto it = llvm::find(originalSources, source);
              if (it == originalSources.end()) {
                op->emitError("composite layout mapping error");
                return failure();
              }
              size_t index = std::distance(originalSources.begin(), it);
              remappedSources.push_back(target.getSource(index));
            }

            // Re-build the operand layout in the iteration space.
            operandLayouts.push_back(
                remappedSources.size() == 1
                    ? remappedSources.front()
                    : CompositeSourceAttr::get(op->getContext(),
                                               remappedSources));
          }
        }
      }

      // Case #2: layout modification operations.
      else if (isa<ReshapeOp, TransposeOp, BroadcastOp, SliceOp>(op)) {
        operandLayouts.push_back(layout);
      }

      // Case #3: concatenate operation.
      else if (isa<ConcatenateOp>(op)) {
        auto concatLayout = dyn_cast<ConcatSourceAttr>(layout);
        if (!concatLayout) {
          op->emitError("expected concatenation layout");
          return failure();
        }
        if (concatLayout.getArgumentIndex().empty()) {
          // All concat operands have layout sources.
          operandLayouts =
              SmallVector<LayoutSourceAttrInterface>(concatLayout.getSources());
        } else {
          // Some concat operands are unused (pruned).
          auto current = concatLayout.getArgumentIndex().begin();
          auto end = concatLayout.getArgumentIndex().end();
          size_t sourceIndex = 0;
          for (int32_t operandIndex : llvm::seq(op->getNumOperands())) {
            if (current != end && *current == operandIndex &&
                sourceIndex < concatLayout.size()) {
              operandLayouts.push_back(concatLayout.getSource(sourceIndex++));
              ++current;
            } else {
              operandLayouts.push_back(nullptr);
            }
          }
          if (current != end || sourceIndex != concatLayout.size()) {
            op->emitError("invalid argument index for concatenation layout");
            return failure();
          }
        }
        newIterationSpaces = concatLayout.size();
      }

      // Case #4: reduction operation.
      else if (isa<ReduceOp, ReduceUDOp>(op)) {
        auto reduceLayout = dyn_cast<ReductionSourceAttr>(layout);
        if (!reduceLayout) {
          op->emitError("expected reduction layout");
          return failure();
        }
        if (isa<ReduceUDOp>(op) && op->getNumOperands() > 1) {
          // Reduction layout with multiple inputs must be a composite.
          auto composite =
              dyn_cast<CompositeSourceAttr>(reduceLayout.getSource());
          if (composite) {
            operandLayouts =
                SmallVector<LayoutSourceAttrInterface>(composite.getSources());
          }
        } else {
          operandLayouts.push_back(reduceLayout.getSource());
        }
        newIterationSpaces = 1;
      }

      // Case #5: matmul operation.
      else if (isa<MatmulOp>(op)) {
        auto matmulLayout = dyn_cast<MatmulSourceAttr>(layout);
        if (!matmulLayout) {
          op->emitError("expected matmul layout");
          return failure();
        }
        operandLayouts.push_back(matmulLayout.getLhs());
        operandLayouts.push_back(matmulLayout.getRhs());
        newIterationSpaces = 2;
      }

      // Verify that all operand layouts are collected.
      if (operandLayouts.size() != op->getNumOperands()) {
        op->emitError("operand layouts cannot be inferred");
        return failure();
      }

      // Recursively materialize each operand.
      IRMapping mapping;
      int32_t operandIterSpaceId = iterSpaceId;
      for (auto [operand, operandLayout] :
           llvm::zip_equal(op->getOperands(), operandLayouts)) {
        if (operandLayout == nullptr) {
          continue;
        }
        if (newIterationSpaces > 0) {
          operandIterSpaceId = ++iterSpaceCounter;
          --newIterationSpaces;
        }
        FailureOr<Value> newOperand =
            materialize(operand, operandIterSpaceId, operandLayout);
        if (failed(newOperand)) {
          return failure();
        }
        mapping.map(operand, *newOperand);
      }

      // Clone op with remapped operands and new layout.
      // For non-unary pointwise ops, keep the layouts for the operands.
      LayoutSourceAttrInterface newLayout = layout;
      if (isa<PointwiseOpInterface>(op) && op->getNumOperands() > 1) {
        newLayout = CompositeSourceAttr::get(op->getContext(), operandLayouts);
      }

      Operation *cloned = builder.clone(*op, mapping);
      cloned->setAttr(TensorIRDialect::getLayoutAttrName(), newLayout);

      // Set the iteration space ID for the cloned operation.
      auto i32Ty = IntegerType::get(op->getContext(), 32);
      cloned->setAttr(TensorIRDialect::getIterSpaceIdAttrName(),
                      IntegerAttr::get(i32Ty, iterSpaceId));

      // Store the cloned operation results in the cache.
      SmallVector<Value> results(cloned->getResults());
      cache.insert({key, results});
      return results[resultIndex];
    };

    // Get iteration space for the graph result.
    auto rootLayout = resultsOp->getAttrOfType<LayoutSourceAttrInterface>(
        TensorIRDialect::getIterationSpaceAttrName());
    if (!rootLayout) {
      graphOp.emitError("missing iteration space");
      return signalPassFailure();
    }
    if (resultsOp->getNumOperands() != 1) {
      graphOp.emitError("expected single result");
      return signalPassFailure();
    }

    // Start recursive materialization from the graph result.
    Value oldRoot = resultsOp->getOperand(0);
    FailureOr<Value> newRoot =
        materialize(oldRoot, /*iterSpaceId=*/0, rootLayout);
    if (failed(newRoot)) {
      return signalPassFailure();
    }
    resultsOp.setOperand(0, *newRoot);

    // Run simple DCE (dead code elimination).
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation &op :
           llvm::make_early_inc_range(graphOp.getBody()->getOperations())) {
        if (llvm::all_of(op.getResults(),
                         [](Value r) { return r.use_empty(); }) &&
            &op != resultsOp.getOperation()) {
          op.erase();
          changed = true;
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After graph splitting:\n";
      graphOp.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
  }
};

} // namespace
} // namespace mlir::nv_tensor_ir
