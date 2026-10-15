// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- CudaTileInsertGdcWait.cpp - Conservative PDL wait insertion --------===//
//
// Makes a CUDA Tile kernel safe to launch with programmatic dependent launch
// (PDL) without any user annotation. A single `gdc_wait_tko` is inserted
// before the first global-memory access of the entry, and its token is
// threaded into (or joined with the existing token of) every global-memory
// operation the wait dominates. No `gdc_launch_dependents_tko` is emitted:
// dependents are triggered implicitly when the grid exits.
//
// Entries that already contain an explicit `gdc_wait_tko`, or that perform no
// global-memory access, are left unchanged. An explicit
// `gdc_launch_dependents_tko` does not count: it releases dependent grids but
// does not order this kernel's own memory accesses after the preceding grid.
//
//===----------------------------------------------------------------------===//

#include "tensor_ir/Transform/Passes.h"

#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DEF_CUDATILEINSERTGDCWAITPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace {

bool isGlobalMemoryAccess(Operation *op) {
  return isa<cuda_tile::AtomicCASTkoOp, cuda_tile::AtomicRMWTkoOp,
             cuda_tile::AtomicRedViewTkoOp, cuda_tile::LoadPtrTkoOp,
             cuda_tile::LoadViewTkoOp, cuda_tile::StorePtrTkoOp,
             cuda_tile::StoreViewTkoOp>(op);
}

Operation *findFirstGlobalMemoryAccess(Operation *op) {
  Operation *first = nullptr;
  op->walk<WalkOrder::PreOrder>([&](Operation *nested) {
    if (!isGlobalMemoryAccess(nested)) {
      return WalkResult::advance();
    }
    first = nested;
    return WalkResult::interrupt();
  });
  return first;
}

/// Make `op` depend on `waitToken`, joining it with the token `op` already
/// consumes, if any.
template <typename OpTy>
LogicalResult setOrJoinToken(OpBuilder &builder, OpTy op, Value waitToken) {
  if (Value existingToken = op.getToken()) {
    builder.setInsertionPoint(op);
    SmallVector<Value, 2> tokens{waitToken, existingToken};
    auto join = cuda_tile::JoinTokensOp::create(
        builder, op.getLoc(), cuda_tile::TokenType::get(op->getContext()),
        tokens);
    waitToken = join.getResult();
  }
  op.getTokenMutable().assign(waitToken);
  return success();
}

LogicalResult threadWaitTokenToGlobalMemoryAccess(OpBuilder &builder,
                                                  Operation *op,
                                                  Value waitToken) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<cuda_tile::AtomicCASTkoOp, cuda_tile::AtomicRMWTkoOp,
            cuda_tile::AtomicRedViewTkoOp, cuda_tile::LoadPtrTkoOp,
            cuda_tile::LoadViewTkoOp, cuda_tile::StorePtrTkoOp,
            cuda_tile::StoreViewTkoOp>([&](auto memoryOp) {
        return setOrJoinToken(builder, memoryOp, waitToken);
      })
      .Default([](Operation *unknownOp) {
        return unknownOp->emitOpError()
               << "is classified as a global memory access but automatic PDL "
                  "insertion does not know how to attach a wait token";
      });
}

struct CudaTileInsertGdcWaitPass
    : public impl::CudaTileInsertGdcWaitPassBase<CudaTileInsertGdcWaitPass> {
  void runOnOperation() override {
    cuda_tile::EntryOp entryOp = getOperation();

    bool hasExplicitWait = false;
    bool hasGlobalMemoryAccess = false;
    entryOp.walk([&](Operation *op) {
      hasExplicitWait |= isa<cuda_tile::GdcWaitTkoOp>(op);
      hasGlobalMemoryAccess |= isGlobalMemoryAccess(op);
      return hasExplicitWait && hasGlobalMemoryAccess ? WalkResult::interrupt()
                                                      : WalkResult::advance();
    });
    if (hasExplicitWait || !hasGlobalMemoryAccess) {
      return;
    }

    Block &entryBlock = entryOp.getBody().front();
    Operation *waitInsertionPoint = entryBlock.getTerminator();
    for (Operation &op : entryBlock) {
      if (findFirstGlobalMemoryAccess(&op)) {
        waitInsertionPoint = &op;
        break;
      }
    }

    OpBuilder builder(entryOp.getContext());
    builder.setInsertionPoint(waitInsertionPoint);
    auto waitOp = cuda_tile::GdcWaitTkoOp::create(
        builder, waitInsertionPoint->getLoc(),
        cuda_tile::TokenType::get(builder.getContext()),
        /*token=*/Value());

    DominanceInfo dominanceInfo(entryOp);
    SmallVector<Operation *> memoryOps;
    entryOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op != waitOp && isGlobalMemoryAccess(op) &&
          dominanceInfo.dominates(waitOp.getOperation(), op)) {
        memoryOps.push_back(op);
      }
    });

    if (memoryOps.empty()) {
      waitOp.emitOpError()
          << "dominates no global memory access despite one being present";
      return signalPassFailure();
    }

    for (Operation *memoryOp : memoryOps) {
      Operation *isolatedAncestor =
          memoryOp->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
      if (isolatedAncestor && isolatedAncestor != entryOp.getOperation()) {
        memoryOp->emitOpError()
            << "is nested under isolated op '" << isolatedAncestor->getName()
            << "' and cannot depend on the automatic PDL wait";
        return signalPassFailure();
      }
      if (failed(threadWaitTokenToGlobalMemoryAccess(
              builder, memoryOp, waitOp.getResultToken()))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::nv_tensor_ir
