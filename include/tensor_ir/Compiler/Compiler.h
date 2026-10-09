// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ICompiler - Interface for compiling MLIR to TensorIRArtifact

#ifndef TENSOR_IR_COMPILER_COMPILER_H
#define TENSOR_IR_COMPILER_COMPILER_H

#include "tensor_ir/Artifact/TensorIRArtifact.h"
#include "tensor_ir/Options/Options.h"

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
                              const CompilerInvocation &options);

//===----------------------------------------------------------------------===//
// ICompiler - Interface for compiling MLIR to TensorIRArtifact
//===----------------------------------------------------------------------===//

class ICompiler {
public:
  virtual ~ICompiler() = default;

  /// Compile to an owning artifact without creating runtime state.
  virtual StatusOr<std::unique_ptr<::tensor_ir::rt::TensorIRArtifact>>
  compile(mlir::ModuleOp module, const CompilerInvocation &options) = 0;

  virtual StatusOr<std::unique_ptr<::tensor_ir::rt::TensorIRArtifact>>
  compile(llvm::StringRef source, const CompilerInvocation &options);

  // XLA integration contract: keep this lightweight and coordinate before
  // changing the signature or semantics.
  virtual bool canCompile(mlir::ModuleOp module,
                          const CompilerInvocation &options) const = 0;

  static std::unique_ptr<ICompiler>
  create(CompilerBackend backend = CompilerBackend::Auto);
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_COMPILER_COMPILER_H
