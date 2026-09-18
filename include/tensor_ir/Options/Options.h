// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_OPTIONS_OPTIONS_H
#define TENSOR_IR_OPTIONS_OPTIONS_H

#include "tensor_ir/Options/OptionsDetail.h"
#include "tensor_ir/Options/OptionsEnums.h"
#include "tensor_ir/Support/Status.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "llvm/ADT/StringRef.h" // IWYU pragma: keep
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h" // IWYU pragma: keep
#include "llvm/Support/raw_ostream.h"   // IWYU pragma: keep

#include <optional> // IWYU pragma: keep
#include <string>   // IWYU pragma: keep
#include <vector>   // IWYU pragma: keep

namespace llvm::cl {
class OptionCategory;
}

#define GEN_OPTIONS_DECLS
#include "tensor_ir/Options/CompilerOptions.h.inc"

namespace mlir::nv_tensor_ir {

struct CompilerInvocation : CompilerInvocationComponents {
  /// Parses the target stored in `deviceTarget.kernelTarget`.
  [[nodiscard]] FailureOr<SmTarget> getKernelSmTarget() const {
    return SmTarget::fromString(deviceTarget.kernelTarget);
  }

  /// Parses the stored target or reports a fatal error.
  [[nodiscard]] SmTarget getKernelSmTargetOrFatal() const {
    FailureOr<SmTarget> target = getKernelSmTarget();
    if (failed(target)) {
      llvm::report_fatal_error(llvm::Twine("invalid target SM '") +
                               deviceTarget.kernelTarget + "'");
    }
    return *target;
  }

  /// Stores \p target in its canonical spelling.
  void setKernelSmTarget(const SmTarget &target) {
    deviceTarget.kernelTarget = target.toString();
  }

  /// Resolves the concrete compute capability used for code generation.
  /// A zero `codegenTarget` selects the kernel target's baseline
  /// compute capability. Explicit values must satisfy the kernel target's
  /// portability contract.
  [[nodiscard]] FailureOr<ComputeCapability>
  getCodegenComputeCapability() const;

  /// Returns a compact, filesystem-safe description for shader names.
  [[nodiscard]] std::string toShaderNameSuffix() const;

  /// Validates cross-component option combinations.
  [[nodiscard]] Status validate() const;
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_OPTIONS_OPTIONS_H
