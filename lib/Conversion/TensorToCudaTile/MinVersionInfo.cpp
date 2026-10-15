// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MinVersionInfo.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::nv_tensor_ir;

LogicalResult
mlir::nv_tensor_ir::validateBytecodeVersion(ModuleOp parentModule,
                                            cuda_tile::BytecodeVersion target) {
  for (auto module : parentModule.getOps<cuda_tile::ModuleOp>()) {
    auto minimum = cuda_tile::BytecodeVersion::kMinSupportedVersion;
    AttrTypeWalker typeWalker;
    typeWalker.addWalk([&](Type type) {
      if (isa<Float8E8M0FNUType>(type)) {
        minimum = std::max(minimum,
                           getMinCudaTileBytecodeVersion<Float8E8M0FNUType>());
      } else if (isa<Float4E2M1FNType>(type)) {
        minimum = std::max(minimum,
                           getMinCudaTileBytecodeVersion<Float4E2M1FNType>());
      } else if (isa<Float8E5M3FNUType>(type)) {
        minimum = std::max(minimum,
                           getMinCudaTileBytecodeVersion<Float8E5M3FNUType>());
      }
    });
    module.walk([&](Operation *op) {
      if (auto exp = dyn_cast<cuda_tile::ExpOp>(op)) {
        /// Note: cuda_tile::RoundingMode::FULL is the default for ExpOp,
        /// matching semantics for version <13.3. BytecodeWriter will handle
        /// targets <13.3 as long as attribute has the default value.
        if (exp.getRoundingMode() != cuda_tile::RoundingMode::FULL) {
          minimum = std::max(
              minimum, getMinCudaTileBytecodeVersion<ExpApproxRounding>());
        }
      }
      for (Type type : op->getOperandTypes()) {
        typeWalker.walk(type);
      }
      for (Type type : op->getResultTypes()) {
        typeWalker.walk(type);
      }
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument argument : block.getArguments()) {
            typeWalker.walk(argument.getType());
          }
        }
      }
    });
    if (minimum > target) {
      return module.emitError("requires CUDA Tile bytecode version ")
             << minimum.toString()
             << " or newer, but target-bytecode-version is "
             << target.toString();
    }
  }
  return success();
}
