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
#include <string>

namespace mlir::nv_tensor_ir {

/// Builds the unhashed cache key for compiling an MLIR module into a kernel.
///
/// Graph symbols and SSA names are normalized on a clone before printing. The
/// original module remains unchanged, and names that do not affect generated
/// code do not prevent otherwise equivalent modules from sharing a kernel.
std::string calculateCacheKey(mlir::ModuleOp module,
                              const CompileOptions &options);

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
