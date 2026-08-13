// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ICompiler - Interface for compiling MLIR to IRuntimeKernel

#ifndef TENSOR_IR_COMPILER_COMPILER_H
#define TENSOR_IR_COMPILER_COMPILER_H

#include "tensor_ir/Compiler/CompileOptions.h"
#include "tensor_ir/Runtime/IRuntimeKernel.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir::nv_tensor_ir {

//===----------------------------------------------------------------------===//
// ICompiler - Interface for compiling MLIR to IRuntimeKernel
//===----------------------------------------------------------------------===//

class ICompiler {
public:
  virtual ~ICompiler() = default;

  virtual StatusOr<::tensor_ir::rt::IRuntimeKernelPtr>
  compile(mlir::ModuleOp module, const CompileOptions &options) = 0;

  virtual StatusOr<::tensor_ir::rt::IRuntimeKernelPtr>
  compile(llvm::StringRef source, const CompileOptions &options);

  // XLA integration contract: keep this lightweight and coordinate before
  // changing the signature or semantics.
  virtual bool canCompile(mlir::ModuleOp module,
                          const CompileOptions &options) const = 0;

  static std::unique_ptr<ICompiler>
  create(CompilerBackend backend = CompilerBackend::Auto);
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_COMPILER_COMPILER_H
