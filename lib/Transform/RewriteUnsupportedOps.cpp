// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/IR/Value.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "rewrite-unsupported-ops"

//===----------------------------------------------------------------------===//
// Rewrite Unsupported Ops
//===----------------------------------------------------------------------===//
// This is the place to fix up ops that TensorIR needs but the cutile dialect
// cannot represent, before the graph is converted to CudaTile. Everything
// after this pass can assume a CudaTile-compatible graph. New rewrites for
// unsupported ops should be added here, not scattered across later passes.
//===----------------------------------------------------------------------===//

#include "tensor_ir/Transform/Passes.h"

namespace mlir::nv_tensor_ir {

// Include the generated pass definitions.
#define GEN_PASS_DEF_REWRITEUNSUPPORTEDOPSPASS
#include "tensor_ir/Transform/Passes.h.inc"

} // namespace mlir::nv_tensor_ir

namespace mlir::nv_tensor_ir {
namespace {

/// Rewrites a matmul whose result type CudaTile does not support for its input
/// types. Tensor IR must follow the CudaTile rules, but we want to accept
/// more result types from the user. The pattern does the matmul in a supported
/// type and converts the result to the type that the user requested.
class RewriteMatmulOpPattern : public OpRewritePattern<MatmulOp> {
public:
  RewriteMatmulOpPattern(MLIRContext *ctx, bool &errorFlag)
      : OpRewritePattern<MatmulOp>(ctx, /*benefit=*/1), errorFlag(errorFlag) {}

  /// Does nothing if the result element type is already supported. Otherwise,
  /// it converts the accumulator to a supported type, creates a new matmul
  /// with that type, and converts the new result back to the original type.
  llvm::LogicalResult
  matchAndRewrite(MatmulOp op, mlir::PatternRewriter &rewriter) const override {
    auto aElemTy = op.getA().getType().getElementType();
    auto bElemTy = op.getB().getType().getElementType();

    auto resTy = op.getResult().getType();
    Type resElemTy = resTy.getElementType();

    auto maybeExpectedResultTypes =
        MatmulOp::deduceCudaTileStagedMatmulTypes(aElemTy, bElemTy);
    if (!maybeExpectedResultTypes) {
      op.emitError("unsupported input element types");
      errorFlag = true;
      return failure();
    }
    auto expectedResultTypes = maybeExpectedResultTypes.value();
    if (llvm::is_contained(expectedResultTypes, resElemTy)) {
      return success();
    }

    auto cvtResType = [&expectedResultTypes](Type origType) -> Type {
      Type destType = nullptr;
      for (auto type : expectedResultTypes) {
        if (type.getIntOrFloatBitWidth() >= origType.getIntOrFloatBitWidth()) {
          destType = type;
        }
      }
      if (!destType) {
        destType = expectedResultTypes.back();
      }
      return destType;
    };

    auto destType = cvtResType(resElemTy);

    Location loc = op->getLoc();
    Value acc = op.getAcc();
    if (acc) {
      acc = ConvertOp::create(rewriter, loc, acc, destType);
    }

    auto newResTy = RankedTensorType::get(resTy.getShape(), destType);
    auto newOp =
        MatmulOp::create(rewriter, loc, newResTy, op.getA(), op.getB(), acc);
    auto newCvtOp = ConvertOp::create(rewriter, loc, newOp, resElemTy);

    rewriter.replaceAllOpUsesWith(op, newCvtOp);
    rewriter.eraseOp(op);
    return success();
  }

private:
  bool &errorFlag;
};

struct RewriteUnsupportedOpsPass
    : public mlir::nv_tensor_ir::impl::RewriteUnsupportedOpsPassBase<
          RewriteUnsupportedOpsPass> {
  using RewriteUnsupportedOpsPassBase::RewriteUnsupportedOpsPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    bool rewriteFalilureFlag = false;

    RewritePatternSet patterns(ctx);
    patterns.add<RewriteMatmulOpPattern>(ctx, rewriteFalilureFlag);
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    if (rewriteFalilureFlag) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace
} // namespace mlir::nv_tensor_ir
