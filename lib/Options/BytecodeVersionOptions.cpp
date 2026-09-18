// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Options/BytecodeVersionOptions.h"

namespace mlir::nv_tensor_ir::backend::cuda_tile {

StatusOr<mlir::cuda_tile::BytecodeVersion>
resolveBytecodeVersion(llvm::StringRef spelling) {
  using mlir::cuda_tile::BytecodeVersion;

  auto invalidVersion = [&]() {
    return Status::InvalidArgument(
        "bytecode-version: unsupported CUDA Tile bytecode version (got '" +
        spelling.str() + "')");
  };

  if (spelling == "default") {
    return getDefaultBytecodeVersion();
  }
  if (spelling == "current") {
    return BytecodeVersion::kCurrentVersion;
  }
  if (spelling == "compatibility") {
    return BytecodeVersion::kCurrentCompatibilityVersion;
  }

  llvm::StringRef rest = spelling;
  uint8_t verMajor = 0;
  uint8_t verMinor = 0;
  uint16_t verTag = 0;
  if (rest.consumeInteger(/*Radix=*/10, verMajor) || !rest.consume_front(".") ||
      rest.consumeInteger(/*Radix=*/10, verMinor)) {
    return invalidVersion();
  }
  if (rest.consume_front(".")) {
    if (rest.consumeInteger(/*Radix=*/10, verTag) || !rest.empty()) {
      return invalidVersion();
    }
  } else if (!rest.empty()) {
    return invalidVersion();
  }
  std::optional<BytecodeVersion> version =
      BytecodeVersion::fromVersion(verMajor, verMinor, verTag);
  if (!version) {
    return invalidVersion();
  }
  return *version;
}

Status verifyCudaTileOptions(const CudaTileOptions &opts) {
  StatusOr<mlir::cuda_tile::BytecodeVersion> version =
      resolveBytecodeVersion(opts.bytecodeVersion);
  return version.ok() ? Status::Ok() : version.status();
}

} // namespace mlir::nv_tensor_ir::backend::cuda_tile
