// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Artifact/TensorIRArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>

using mlir::nv_tensor_ir::SmTarget;
using mlir::nv_tensor_ir::Status;
using mlir::nv_tensor_ir::StatusOr;

namespace tensor_ir::rt {

Status TensorIRArtifact::validateFields(
    llvm::StringRef kernelName, llvm::StringRef funcName, const Binary &binary,
    const std::optional<std::array<int32_t, 3>> &staticGrid) {
  bool binaryEmpty =
      std::visit([](const auto &b) { return b.bytes.empty(); }, binary);
  if (binaryEmpty) {
    return Status::InvalidArgument("TensorIRArtifact: binary is empty");
  }
  if (kernelName.empty() || funcName.empty()) {
    return Status::InvalidArgument(
        "TensorIRArtifact: kernelName/funcName must not be empty");
  }

  if (staticGrid &&
      llvm::any_of(*staticGrid, [](int32_t dim) { return dim < 1; })) {
    return Status::InvalidArgument(
        "TensorIRArtifact: staticGrid dimensions must be at least 1");
  }

  return Status::Ok();
}

TensorIRArtifact::TensorIRArtifact(
    std::string kernelName, std::string funcName, Binary &&binary,
    SmTarget arch, KernelMetadata argLayout,
    const std::optional<std::array<int32_t, 3>> &staticGrid)
    : kernelName(std::move(kernelName)), funcName(std::move(funcName)),
      binary(std::move(binary)), arch(arch), argLayout(std::move(argLayout)),
      staticGrid(staticGrid) {}

StatusOr<std::unique_ptr<TensorIRArtifact>> TensorIRArtifact::create(
    std::string kernelName, std::string funcName, Binary &&binary,
    SmTarget arch, KernelMetadata argLayout,
    const std::optional<std::array<int32_t, 3>> &staticGrid) {
  if (Status validation =
          validateFields(kernelName, funcName, binary, staticGrid);
      !validation.ok()) {
    return validation;
  }
  return std::unique_ptr<TensorIRArtifact>(new TensorIRArtifact(
      std::move(kernelName), std::move(funcName), std::move(binary), arch,
      std::move(argLayout), staticGrid));
}

} // namespace tensor_ir::rt
