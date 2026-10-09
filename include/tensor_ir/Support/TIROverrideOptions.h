// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSORIR_SUPPORT_TIROVERRIDEOPTIONS_H
#define TENSORIR_SUPPORT_TIROVERRIDEOPTIONS_H

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::nv_tensor_ir {

/// Owns an argument vector rewritten according to `TIR_OVERRIDE_OPTIONS`.
///
/// This interface is inspired by Clang's `CCC_OVERRIDE_OPTIONS`, but the edit
/// grammar and semantics are intentionally different and are not compatible.
///
/// Supported edits are applied in order, from left to right within the spec:
///   #flag          drop `--flag` and its registered required value
///   ^flag          move `--flag` and its registered required value to the front
///   +flag[=val]    append `--flag` or `--flag=val` if absent
///   s/old/new/     replace `--old` prefix with `--new` on matching arguments
///   x              drop the next positional argument once
///   X              drop all positional arguments
///
/// Positional arguments are those not starting with `-`. The program name
/// (`argv[0]`) is never modified.
class TIRCommandLineOverride {
public:
  TIRCommandLineOverride(int argc, char **argv);
  TIRCommandLineOverride(const TIRCommandLineOverride &) = delete;
  TIRCommandLineOverride &operator=(const TIRCommandLineOverride &) = delete;
  TIRCommandLineOverride(TIRCommandLineOverride &&) = delete;
  TIRCommandLineOverride &operator=(TIRCommandLineOverride &&) = delete;

  [[nodiscard]] int argc() const { return static_cast<int>(argv_.size()) - 1; }
  [[nodiscard]] char **argv() { return argv_.data(); }

private:
  llvm::SmallVector<std::string, 32> args_;
  llvm::SmallVector<char *, 32> argv_;
};

} // namespace mlir::nv_tensor_ir

#endif // TENSORIR_SUPPORT_TIROVERRIDEOPTIONS_H
