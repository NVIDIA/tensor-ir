// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CudaTileCompiler - Compiler backend for CudaTile

#ifndef TENSOR_IR_COMPILER_CUDATILECOMPILER_H
#define TENSOR_IR_COMPILER_CUDATILECOMPILER_H

#include "tensor_ir/Compiler/Compiler.h"
#include "tensor_ir/Options/BytecodeVersionOptions.h"
#include "tensor_ir/Options/Options.h"
#include "tensor_ir/Support/Status.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tensor_ir {

/// A tile shape candidate. Analysis returns candidates in best-first order
/// according to the estimated cost.
struct TileCandidate {
  std::vector<int32_t> shape;
};

} // namespace tensor_ir

namespace mlir::nv_tensor_ir::backend::cuda_tile {

//===----------------------------------------------------------------------===//
// CudaTileCompiler - Compiles TensorIR MLIR to CudaTile artifacts
//===----------------------------------------------------------------------===//

class CudaTileCompiler : public ICompiler {
public:
  CudaTileCompiler() = default;

  StatusOr<std::unique_ptr<::tensor_ir::rt::TensorIRArtifact>>
  compile(mlir::ModuleOp module, const CompilerInvocation &options) override;

  // XLA integration contract: keep this lightweight and coordinate before
  // changing the signature or semantics.
  bool canCompile(mlir::ModuleOp module,
                  const CompilerInvocation &options) const override;

  /// Analyze a graph in place and return tile-size candidates ordered best-first.
  /// Requires exactly one graph and a positive maxTileCandidates or a tile hint.
  /// Returns no candidates when options supplies an explicit tile size.
  /// The module may also be modified on failure.
  StatusOr<std::vector<::tensor_ir::TileCandidate>>
  analyzeModule(mlir::ModuleOp module, const CompilerInvocation &options);
};

} // namespace mlir::nv_tensor_ir::backend::cuda_tile

#endif // TENSOR_IR_COMPILER_CUDATILECOMPILER_H
