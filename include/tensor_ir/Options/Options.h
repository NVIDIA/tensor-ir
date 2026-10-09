// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_OPTIONS_OPTIONS_H
#define TENSOR_IR_OPTIONS_OPTIONS_H

#include "tensor_ir/Options/OptionsEnums.h" // IWYU pragma: keep
#include "tensor_ir/Support/Status.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "llvm/ADT/ArrayRef.h"  // IWYU pragma: keep
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

  /// Returns the CUDA toolkit path to be used during compilation when
  /// it is necessary to lookup dynamic libraries or tools supplied
  /// via CUDA Toolkit installation (e.g. `tileiras`).
  ///
  /// It is a convention in CUDA that tools/libraries will look for
  /// other required tools or dynamic libraries typically shipped with
  /// a CTK installation by searching (1) relative to a supplied
  /// 'cuda-toolkit-path' flag, (2) relative to a suppied override
  /// given through the 'CUDA_ROOT' environment variable, (3) relative to
  /// their installation path.
  ///
  /// TensorIR also tries to follow this convention. This function
  /// will return in priority order:
  /// (1) `codegen.cudaToolkitPath` if set,
  /// (2) the value of the `CUDA_ROOT` environment if set,
  /// (3) empty string indicating that the caller should look for the tool
  ///     on the PATH or relative to their installation path.
  [[nodiscard]] std::string getCudaToolkitPath() const;
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_OPTIONS_OPTIONS_H
