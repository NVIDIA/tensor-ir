// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_OPTIONS_BYTECODEVERSIONOPTIONS_H
#define TENSOR_IR_OPTIONS_BYTECODEVERSIONOPTIONS_H

#include "tensor_ir/Options/Options.h"

#include "llvm/ADT/StringRef.h"

#include "cuda_tile/Bytecode/Common/Version.h"
#include <algorithm>
#include <cassert>
#include <optional>
#include <string>

namespace mlir::nv_tensor_ir::backend::cuda_tile {

/// Resolves `default`, `current`, `compatibility`, or `major.minor[.tag]`.
StatusOr<mlir::cuda_tile::BytecodeVersion>
resolveBytecodeVersion(llvm::StringRef spelling);

/// Returns CUDA Tile's compatibility version, with a 13.3 minimum.
inline mlir::cuda_tile::BytecodeVersion getDefaultBytecodeVersion() {
  using mlir::cuda_tile::BytecodeVersion;
  const auto &compatibility = BytecodeVersion::kCurrentCompatibilityVersion;
  auto minimum = BytecodeVersion::fromVersion(/*verMajor=*/13,
                                              /*verMinor=*/3,
                                              /*verTag=*/0);
  assert(minimum && "bytecode version 13.3 must be supported");
  return std::max(compatibility, *minimum);
}

Status verifyCudaTileOptions(const CudaTileOptions &opts);

} // namespace mlir::nv_tensor_ir::backend::cuda_tile

#endif // TENSOR_IR_OPTIONS_BYTECODEVERSIONOPTIONS_H
