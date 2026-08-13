// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_COMPILEOPTIONS_H
#define TENSOR_IR_COMPILER_COMPILEOPTIONS_H

#include "tensor_ir/Support/Status.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::nv_tensor_ir {

//===----------------------------------------------------------------------===//
// Backend Selection
//===----------------------------------------------------------------------===//

enum class CompilerBackend {
  Auto, // Auto-select based on graph analysis
  CudaTile,
};

llvm::StringRef stringifyCompilerBackend(CompilerBackend backend);

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

/// Base compilation options shared by public TensorIR compiler backends.
class CompileOptions {
public:
  SmTarget computeCapability;

  virtual ~CompileOptions() = default;

  virtual Status validate() const;
  virtual std::string toString() const;
  virtual std::string toUniqueString() const;

  CompilerBackend getBackend() const { return backend; }

protected:
  virtual Status validateDerived() const = 0;
  virtual std::string toStringDerived() const = 0;
  virtual std::string toUniqueStringDerived() const = 0;

  CompileOptions(SmTarget computeCapability_, CompilerBackend backend_)
      : computeCapability(computeCapability_), backend(backend_) {}

private:
  CompilerBackend backend = CompilerBackend::Auto;
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_COMPILER_COMPILEOPTIONS_H
