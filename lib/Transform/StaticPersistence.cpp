// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Transform/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <limits>

namespace mlir::nv_tensor_ir {
#define GEN_PASS_DEF_STATICPERSISTENCEPASS
#include "tensor_ir/Transform/Passes.h.inc"
} // namespace mlir::nv_tensor_ir

using namespace mlir;
using namespace mlir::nv_tensor_ir;

/// Limit the hardware grid and let each block visit the remaining flattened
/// tile IDs serially.
///
/// Static persistence caps the launch grid at `sm_count * occupancy` blocks.
/// If the total number of tiles exceeds this limit, the scf.forall upper bound
/// is clamped to the persistent limit, and the entire per-tile computation in the
/// forall body is wrapped in a serial grid-stride loop:
///   scf.for %tileId = %blockId to %totalTiles step %gridSize
/// Uses of the original block ID induction variable inside the body are rewired
/// to the serial loop's induction variable.
///
/// Static persistence also supports logical grid extents known only at runtime.
static LogicalResult formStaticPersistence(RewriterBase &builder,
                                           func::FuncOp function,
                                           int64_t smCount, int64_t occupancy) {
  // 1. Verify static persistent grid limit does not overflow 32-bit index.
  int64_t persistentGridLimit;
  constexpr int64_t maxI32 = std::numeric_limits<int32_t>::max();
  if (llvm::MulOverflow(smCount, occupancy, persistentGridLimit) ||
      persistentGridLimit > maxI32) {
    return function.emitError(
        "static persistent grid size overflows 32-bit index range");
  }

  // 2. For each rank-one destination-free scf.forall, clamp the grid bound.
  SmallVector<scf::ForallOp> foralls;
  function.walk([&](scf::ForallOp forallOp) { foralls.push_back(forallOp); });
  for (scf::ForallOp forallOp : foralls) {
    if (forallOp.getRank() != 1 || !forallOp.getOutputs().empty()) {
      return forallOp.emitError(
          "static persistence requires a rank-one destination-free forall");
    }

    builder.setInsertionPoint(forallOp);
    Location loc = forallOp.getLoc();
    OpFoldResult mixedTotalTiles = forallOp.getMixedUpperBound().front();
    Value totalTiles;
    Value gridSize;
    if (auto staticTotal = dyn_cast<Attribute>(mixedTotalTiles)) {
      int64_t total = *getConstantIntValue(staticTotal);
      if (total <= persistentGridLimit) {
        continue;
      }
      int64_t grid = std::max<int64_t>(1, persistentGridLimit);
      totalTiles =
          getValueOrCreateConstantIndexOp(builder, loc, mixedTotalTiles);
      gridSize = arith::ConstantIndexOp::create(builder, loc, grid).getResult();
      builder.modifyOpInPlace(forallOp,
                              [&] { forallOp.setStaticUpperBound({grid}); });
    } else {
      totalTiles = cast<Value>(mixedTotalTiles);
      Value limit =
          arith::ConstantIndexOp::create(builder, loc, persistentGridLimit);
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      Value bounded = arith::MinUIOp::create(builder, loc, totalTiles, limit);
      gridSize = arith::MaxUIOp::create(builder, loc, bounded, one);
      builder.modifyOpInPlace(forallOp, [&] {
        forallOp.getDynamicUpperBoundMutable().assign(gridSize);
      });
    }

    // 3. Wrap the per-tile computation in a grid-stride scf.for loop.
    Block *body = forallOp.getBody();
    SmallVector<Operation *> bodyOps;
    for (Operation &op : body->without_terminator()) {
      bodyOps.push_back(&op);
    }
    builder.setInsertionPointToStart(body);
    auto serialLoop = scf::ForOp::create(
        builder, loc, forallOp.getInductionVar(0), totalTiles, gridSize);
    Operation *serialTerminator = serialLoop.getBody()->getTerminator();
    for (Operation *op : bodyOps) {
      builder.moveOpBefore(op, serialTerminator);
    }
    builder.replaceUsesWithIf(
        forallOp.getInductionVar(0), serialLoop.getInductionVar(),
        [&](OpOperand &use) {
          return serialLoop->isProperAncestor(use.getOwner());
        });
  }
  return success();
}

namespace {
struct StaticPersistencePass
    : public nv_tensor_ir::impl::StaticPersistencePassBase<
          StaticPersistencePass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void StaticPersistencePass::runOnOperation() {
  if (sm_count < 1 || occupancy < 1) {
    getOperation().emitError(
        "static persistence requires sm-count >= 1 and occupancy >= 1");
    signalPassFailure();
    return;
  }

  IRRewriter rewriter(&getContext());
  if (failed(formStaticPersistence(rewriter, getOperation(), sm_count,
                                   occupancy))) {
    signalPassFailure();
  }
}
