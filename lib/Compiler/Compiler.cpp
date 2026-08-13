// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/Compiler.h"

#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"
#include "tensor_ir/Registration/Registration.h"

#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringExtras.h"

#include <memory>

namespace mlir::nv_tensor_ir {

llvm::StringRef stringifyCompilerBackend(CompilerBackend backend) {
  switch (backend) {
  case CompilerBackend::Auto:
    return "auto";
  case CompilerBackend::CudaTile:
    return "cuda_tile";
  default:
    return "";
  }
}

Status CompileOptions::validate() const { return validateDerived(); }

std::string CompileOptions::toString() const {
  std::string s = "Compilation options:";
  s += "\nBackend: " + stringifyCompilerBackend(backend).str();
  s += "\nCompute capability: " + computeCapability.toString();
  std::string derived = toStringDerived();
  if (!derived.empty()) {
    s += "\nBackend options: " + derived;
  }
  return s;
}

std::string CompileOptions::toUniqueString() const {
  return llvm::join_items("_", stringifyCompilerBackend(backend).str(),
                          computeCapability.toString(),
                          toUniqueStringDerived());
}

StatusOr<::tensor_ir::rt::IRuntimeKernelPtr>
ICompiler::compile(llvm::StringRef source, const CompileOptions &options) {
  // Create MLIR context with TensorIR dialects registered
  mlir::DialectRegistry registry;
  mlir::nv_tensor_ir::registerDialects(registry);
  mlir::MLIRContext mlirContext(registry);
  mlirContext.loadAllAvailableDialects();

  // Parse the source string to a ModuleOp
  mlir::ParserConfig parserConfig(&mlirContext);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, parserConfig);
  if (!module) {
    return Status::CompilationError("Failed to parse MLIR source string");
  }

  // Delegate to the ModuleOp compile method
  return compile(*module, options);
}

std::unique_ptr<ICompiler> ICompiler::create(CompilerBackend backend) {
  if (backend == CompilerBackend::CudaTile) {
    return std::make_unique<backend::cuda_tile::CudaTileCompiler>();
  }

  if (backend == CompilerBackend::Auto) {
    return std::make_unique<backend::cuda_tile::CudaTileCompiler>();
  }

  return nullptr;
}

} // namespace mlir::nv_tensor_ir
