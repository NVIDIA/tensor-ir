// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Artifact/TensorIRArtifact.h"

#include "tensor_ir/Runtime/HostRuntimeAbi.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>

using mlir::nv_tensor_ir::SmTarget;
using mlir::nv_tensor_ir::Status;
using mlir::nv_tensor_ir::StatusOr;

namespace tensor_ir::rt {

Status TensorIRArtifact::validateFields(
    llvm::StringRef kernelName, llvm::StringRef funcName, const Binary &binary,
    const std::optional<std::array<int32_t, 3>> &staticGrid,
    const std::optional<HostModule> &hostModule,
    llvm::StringRef entrypointName) {
  bool binaryEmpty =
      std::visit([](const auto &b) { return b.bytes.empty(); }, binary);
  if (binaryEmpty) {
    return Status::InvalidArgument("TensorIRArtifact: binary is empty");
  }
  if (kernelName.empty() || funcName.empty()) {
    return Status::InvalidArgument(
        "TensorIRArtifact: kernelName/funcName must not be empty");
  }

  if (entrypointName.empty() || entrypointName.contains('\0')) {
    return Status::InvalidArgument("TensorIRArtifact: invalid entrypoint name");
  }

  if (staticGrid &&
      llvm::any_of(*staticGrid, [](int32_t dim) { return dim < 1; })) {
    return Status::InvalidArgument(
        "TensorIRArtifact: staticGrid dimensions must be at least 1");
  }

  if (hostModule) {
    if (hostModule->bytes.empty() || hostModule->triple.empty() ||
        hostModule->triple.find('\0') != std::string::npos) {
      return Status::InvalidArgument("TensorIRArtifact: invalid host module "
                                     "bytes or triple");
    }
    if (hostModule->abiVersion != TENSOR_IR_HOST_ABI_VERSION_1) {
      return Status::NotSupported(
          "TensorIRArtifact: unsupported host ABI version");
    }
  }
  return Status::Ok();
}

TensorIRArtifact::TensorIRArtifact(
    std::string kernelName, std::string funcName, Binary &&binary,
    SmTarget arch, KernelMetadata argLayout,
    const std::optional<std::array<int32_t, 3>> &staticGrid, bool usesPDL,
    std::optional<HostModule> hostModule, std::string entrypointName)
    : kernelName(std::move(kernelName)), funcName(std::move(funcName)),
      entrypointName(std::move(entrypointName)), binary(std::move(binary)),
      hostModule(std::move(hostModule)), arch(arch),
      argLayout(std::move(argLayout)), staticGrid(staticGrid),
      usesPDL_(usesPDL) {}

StatusOr<std::unique_ptr<TensorIRArtifact>> TensorIRArtifact::create(
    std::string kernelName, std::string funcName, Binary &&binary,
    SmTarget arch, KernelMetadata argLayout,
    const std::optional<std::array<int32_t, 3>> &staticGrid, bool usesPDL,
    std::optional<HostModule> hostModule, std::string entrypointName) {
  // Interpreter artifacts historically use the display name as their entrypoint.
  // Native artifacts must supply the actual symbol exported by their host object.
  if (entrypointName.empty() && !hostModule) {
    entrypointName = kernelName;
  }
  if (Status validation = validateFields(
          kernelName, funcName, binary, staticGrid, hostModule, entrypointName);
      !validation.ok()) {
    return validation;
  }
  return std::unique_ptr<TensorIRArtifact>(new TensorIRArtifact(
      std::move(kernelName), std::move(funcName), std::move(binary), arch,
      std::move(argLayout), staticGrid, usesPDL, std::move(hostModule),
      std::move(entrypointName)));
}

} // namespace tensor_ir::rt
