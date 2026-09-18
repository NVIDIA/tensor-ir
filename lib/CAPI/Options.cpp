// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Options/Options.h" // IWYU pragma: keep

#include "tensor_ir/Options/BytecodeVersionOptions.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "mlir-c/Support.h" // IWYU pragma: keep

#include "llvm/ADT/StringRef.h"

#include "cuda_tile/Bytecode/Common/Version.h"
#include "tensor_ir-c/TensorIR.h" // IWYU pragma: keep
#include <cstdint>
#include <optional>
#include <string>

#define GEN_COMPILERINVOCATION_CAPI_IMPL
#include "tensor_ir/Options/CompilerOptions.h.inc"

using mlir::nv_tensor_ir::ArchPortability;
using mlir::nv_tensor_ir::CompilerInvocation;
using mlir::nv_tensor_ir::SmTarget;

namespace {

// The C mirror of ArchPortability is hand-written rather than generated, since
// the target is not stored as an option of its own. Casting is still the
// conversion, as it is for the generated enums.
static_assert(static_cast<int>(ArchPortability::portable) ==
                  MlirTensorIRArchPortabilityPortable &&
              static_cast<int>(ArchPortability::family_portable) ==
                  MlirTensorIRArchPortabilityFamilyPortable &&
              static_cast<int>(ArchPortability::arch_conditional) ==
                  MlirTensorIRArchPortabilityArchConditional);

std::string
formatBytecodeVersionTuple(MlirTensorIRBytecodeVersion bytecodeVersion) {
  return std::to_string(static_cast<unsigned>(bytecodeVersion.major)) + "." +
         std::to_string(static_cast<unsigned>(bytecodeVersion.minor)) + "." +
         std::to_string(bytecodeVersion.tag);
}

std::optional<MlirTensorIRBytecodeVersion>
parseBytecodeVersionTuple(llvm::StringRef spelling) {
  MlirTensorIRBytecodeVersion version{};
  llvm::StringRef rest = spelling;
  if (rest.consumeInteger(/*Radix=*/10, version.major) ||
      !rest.consume_front(".") ||
      rest.consumeInteger(/*Radix=*/10, version.minor) ||
      !rest.consume_front(".") ||
      rest.consumeInteger(/*Radix=*/10, version.tag) || !rest.empty()) {
    return std::nullopt;
  }
  return version;
}

} // namespace

MlirLogicalResult mlirTensorIRCompilerInvocationValidate(
    MlirTensorIRCompilerInvocation invocation, MlirStringCallback errorCallback,
    void *errorUserData) {
  auto *opts = unwrap(invocation);
  if (!opts) {
    return mlirLogicalResultFailure();
  }
  mlir::nv_tensor_ir::Status status = opts->validate();
  if (status.ok()) {
    return mlirLogicalResultSuccess();
  }
  if (errorCallback) {
    llvm::StringRef message = status.message();
    errorCallback(MlirStringRef{message.data(), message.size()}, errorUserData);
  }
  return mlirLogicalResultFailure();
}

MlirLogicalResult mlirTensorIRCompilerInvocationSetKernelTarget(
    MlirTensorIRCompilerInvocation invocation, int32_t computeCapability,
    MlirTensorIRArchPortability portability) {
  auto *opts = unwrap(invocation);
  if (!opts) {
    return mlirLogicalResultFailure();
  }
  mlir::FailureOr<SmTarget> target = SmTarget::fromCc(
      computeCapability, static_cast<ArchPortability>(portability));
  if (mlir::failed(target)) {
    return mlirLogicalResultFailure();
  }
  opts->setKernelSmTarget(*target);
  return mlirLogicalResultSuccess();
}

int32_t mlirTensorIRCompilerInvocationGetKernelComputeCapability(
    MlirTensorIRCompilerInvocation invocation) {
  auto *opts = unwrap(invocation);
  if (!opts) {
    return 0;
  }
  mlir::FailureOr<SmTarget> target = opts->getKernelSmTarget();
  return mlir::succeeded(target) ? target->getComputeCapabilityVersion() : 0;
}

MlirTensorIRArchPortability
mlirTensorIRCompilerInvocationGetKernelArchPortability(
    MlirTensorIRCompilerInvocation invocation) {
  auto *opts = unwrap(invocation);
  if (!opts) {
    return MlirTensorIRArchPortabilityFamilyPortable;
  }
  mlir::FailureOr<SmTarget> target = opts->getKernelSmTarget();
  if (mlir::failed(target)) {
    return MlirTensorIRArchPortabilityFamilyPortable;
  }
  return static_cast<MlirTensorIRArchPortability>(target->getPortability());
}

void mlirTensorIRCompilerInvocationSetBytecodeVersion(
    MlirTensorIRCompilerInvocation invocation,
    MlirTensorIRBytecodeVersion version) {
  if (auto *opts = unwrap(invocation)) {
    auto bytecodeVersion = mlir::cuda_tile::BytecodeVersion::fromVersion(
        version.major, version.minor, version.tag);
    opts->cudaTile.bytecodeVersion = bytecodeVersion
                                         ? bytecodeVersion->toString()
                                         : formatBytecodeVersionTuple(version);
  }
}

MlirTensorIRBytecodeVersion mlirTensorIRCompilerInvocationGetBytecodeVersion(
    MlirTensorIRCompilerInvocation invocation) {
  auto *opts = unwrap(invocation);
  if (!opts) {
    return MlirTensorIRBytecodeVersion{0, 0, 0};
  }
  auto version = mlir::nv_tensor_ir::backend::cuda_tile::resolveBytecodeVersion(
      opts->cudaTile.bytecodeVersion);
  if (!version.ok()) {
    return parseBytecodeVersionTuple(opts->cudaTile.bytecodeVersion)
        .value_or(MlirTensorIRBytecodeVersion{0, 0, 0});
  }
  return MlirTensorIRBytecodeVersion{static_cast<uint8_t>(version->getMajor()),
                                     static_cast<uint8_t>(version->getMinor()),
                                     static_cast<uint16_t>(version->getTag())};
}

void mlirTensorIRCompilerInvocationSetTileSizes(
    MlirTensorIRCompilerInvocation invocation, const int32_t *sizes,
    size_t numSizes) {
  if (auto *opts = unwrap(invocation)) {
    if (numSizes == 0) {
      opts->codegen.tileSize.clear();
      return;
    }
    opts->codegen.tileSize.assign(sizes, sizes + numSizes);
  }
}

const int32_t *mlirTensorIRCompilerInvocationGetTileSizes(
    MlirTensorIRCompilerInvocation invocation, size_t *numSizes) {
  auto *opts = unwrap(invocation);
  *numSizes = opts ? opts->codegen.tileSize.size() : 0;
  return opts && !opts->codegen.tileSize.empty() ? opts->codegen.tileSize.data()
                                                 : nullptr;
}
