// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::nv_tensor_ir {
#define GEN_PASS_DEF_ERASESHAPEASSUMPTIONSPASS
#include "tensor_ir/Transform/Passes.h.inc"
} // namespace mlir::nv_tensor_ir

using namespace mlir;
using namespace mlir::nv_tensor_ir;

namespace {
struct EraseShapeAssumptionsPass
    : nv_tensor_ir::impl::EraseShapeAssumptionsPassBase<
          EraseShapeAssumptionsPass> {
  void runOnOperation() override;
};
} // namespace

void EraseShapeAssumptionsPass::runOnOperation() {
  bool erasedAssumptions = false;
  getOperation()->walk([&](AssumeShapeOp assume) {
    assume.erase();
    erasedAssumptions = true;
  });
  if (!erasedAssumptions) {
    markAllAnalysesPreserved();
    return;
  }

  IRRewriter rewriter(&getContext());
  for (Region &region : getOperation()->getRegions()) {
    eliminateTriviallyDeadOps(rewriter, region);
  }
}
