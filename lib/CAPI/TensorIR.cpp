// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir-c/TensorIR.h"

#include "tensor_ir/Artifact/TensorIRArtifactCodec.h"
#include "tensor_ir/Compiler/Compiler.h"
#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Options/BytecodeVersionOptions.h"
#include "tensor_ir/Registration/Registration.h"
#include "tensor_ir/Runtime/CudaTile/CudaTileRuntimeKernel.h"
#include "tensor_ir/Runtime/Types.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(nv_tensor_ir, nv_tensor_ir,
                                      mlir::nv_tensor_ir::TensorIRDialect)

namespace {

using mlir::cuda_tile::BytecodeVersion;
using mlir::nv_tensor_ir::CompilerInvocation;
using mlir::nv_tensor_ir::ICompiler;
using mlir::nv_tensor_ir::Status;
using mlir::nv_tensor_ir::StatusCode;
using mlir::nv_tensor_ir::StatusOr;
using tensor_ir::rt::Any;
using tensor_ir::rt::CudaTileRuntimeKernel;
using tensor_ir::rt::PackedArgs;
using tensor_ir::rt::Stream;
using tensor_ir::rt::TensorView;
using tensor_ir::rt::Workspace;

MlirTensorIRBytecodeVersion
wrapBytecodeVersion(const BytecodeVersion &version) {
  return {version.getMajor(), version.getMinor(), version.getTag()};
}

StatusOr<BytecodeVersion>
unwrapBytecodeVersion(MlirTensorIRBytecodeVersion version) {
  std::optional<BytecodeVersion> cppVersion =
      BytecodeVersion::fromVersion(version.major, version.minor, version.tag);
  if (!cppVersion) {
    return Status::InvalidArgument(
        "Unsupported TensorIR bytecode version: " +
        std::to_string(static_cast<unsigned>(version.major)) + "." +
        std::to_string(static_cast<unsigned>(version.minor)) + "." +
        std::to_string(version.tag));
  }
  return *cppVersion;
}

llvm::SmallVector<Any, 8> unwrapPackedArgs(MlirTensorIRPackedArgs args) {
  llvm::SmallVector<Any, 8> unpacked;
  unpacked.reserve(args.numArgs);
  for (size_t index = 0; index < args.numArgs; ++index) {
    const MlirTensorIRArgument &argument = args.args[index];
    switch (argument.kind) {
    case MlirTensorIRArgumentKindNone:
      unpacked.emplace_back();
      break;
    case MlirTensorIRArgumentKindInt64:
      unpacked.emplace_back(argument.value.int64Value);
      break;
    case MlirTensorIRArgumentKindFloat64:
      unpacked.emplace_back(argument.value.float64Value);
      break;
    case MlirTensorIRArgumentKindPointer:
      unpacked.emplace_back(argument.value.pointerValue);
      break;
    case MlirTensorIRArgumentKindTensor:
      unpacked.emplace_back(TensorView{
          argument.value.tensor.data, argument.value.tensor.rank,
          argument.value.tensor.shape, argument.value.tensor.strides});
      break;
    default:
      llvm_unreachable("unsupported TensorIR runtime argument kind");
    }
  }
  return unpacked;
}

PackedArgs makePackedArgs(llvm::ArrayRef<Any> args) {
  return PackedArgs(args.data(), static_cast<int32_t>(args.size()));
}

void emitError(llvm::StringRef message, MlirStringCallback errorCallback,
               void *errorUserData) {
  if (errorCallback) {
    errorCallback(mlirStringRefCreate(message.data(), message.size()),
                  errorUserData);
  }
}

MlirLogicalResult reportStatus(const Status &status,
                               MlirStringCallback errorCallback,
                               void *errorUserData) {
  if (status.ok()) {
    return mlirLogicalResultSuccess();
  }
  llvm::StringRef message = status.message();
  if (message.empty()) {
    message = "TensorIR runtime call failed";
  }
  emitError(message, errorCallback, errorUserData);
  return mlirLogicalResultFailure();
}

/// The C++ invocation behind `invocation`, validated and paired with a compiler
/// for its backend. Returns nullopt after reporting through `errorCallback`.
std::optional<std::pair<const CompilerInvocation *, std::unique_ptr<ICompiler>>>
prepareCompile(MlirTensorIRCompilerInvocation invocation,
               MlirStringCallback errorCallback, void *errorUserData) {
  const auto *cppInvocation =
      static_cast<const CompilerInvocation *>(invocation.ptr);
  if (!cppInvocation) {
    emitError("TensorIR compiler invocation is null", errorCallback,
              errorUserData);
    return std::nullopt;
  }
  Status validation = cppInvocation->validate();
  if (!validation.ok()) {
    reportStatus(validation, errorCallback, errorUserData);
    return std::nullopt;
  }
  std::unique_ptr<ICompiler> compiler =
      ICompiler::create(cppInvocation->codegen.compilerBackend);
  if (!compiler) {
    emitError(
        "TensorIR backend '" +
            stringifyCompilerBackend(cppInvocation->codegen.compilerBackend)
                .str() +
            "' is not supported by this entry point",
        errorCallback, errorUserData);
    return std::nullopt;
  }
  return std::make_pair(cppInvocation, std::move(compiler));
}

} // namespace

void mlirTensorIRRegisterAllDialects(MlirDialectRegistry registry) {
  mlir::nv_tensor_ir::registerDialects(*unwrap(registry));
}

// BytecodeVersion C API

MlirTensorIRBytecodeVersion mlirTensorIRGetCurrentBytecodeVersion(void) {
  return wrapBytecodeVersion(BytecodeVersion::kCurrentVersion);
}

MlirTensorIRBytecodeVersion mlirTensorIRGetCompatibilityBytecodeVersion(void) {
  return wrapBytecodeVersion(BytecodeVersion::kCurrentCompatibilityVersion);
}

MlirTensorIRBytecodeVersion mlirTensorIRGetDefaultBytecodeVersion(void) {
  auto version =
      mlir::nv_tensor_ir::backend::cuda_tile::getDefaultBytecodeVersion();
  return wrapBytecodeVersion(version);
}

void mlirTensorIRFormatBytecodeVersion(MlirTensorIRBytecodeVersion version,
                                       MlirStringCallback callback,
                                       void *userData) {
  if (!callback) {
    return;
  }
  StatusOr<BytecodeVersion> bytecodeVersion = unwrapBytecodeVersion(version);
  if (!bytecodeVersion.ok()) {
    return;
  }
  std::string formatted = bytecodeVersion->toString();
  callback(mlirStringRefCreate(formatted.data(), formatted.size()), userData);
}

bool mlirTensorIRBytecodeVersionEqual(MlirTensorIRBytecodeVersion lhs,
                                      MlirTensorIRBytecodeVersion rhs) {
  return lhs.major == rhs.major && lhs.minor == rhs.minor && lhs.tag == rhs.tag;
}

uint32_t mlirTensorIRBytecodeVersionHash(MlirTensorIRBytecodeVersion version) {
  return (static_cast<uint32_t>(version.major) << 24) |
         (static_cast<uint32_t>(version.minor) << 16) | version.tag;
}

// Program C API

MlirLogicalResult mlirTensorIRCalculateCacheKey(
    MlirModule module, MlirTensorIRCompilerInvocation invocation,
    MlirStringCallback keyCallback, void *keyUserData,
    MlirStringCallback errorCallback, void *errorUserData) {
  if (mlirModuleIsNull(module)) {
    return reportStatus(Status::InvalidArgument("module must not be null"),
                        errorCallback, errorUserData);
  }
  if (!keyCallback) {
    return reportStatus(Status::InvalidArgument("keyCallback must not be null"),
                        errorCallback, errorUserData);
  }
  const auto *options = static_cast<const CompilerInvocation *>(invocation.ptr);
  if (!options) {
    return reportStatus(Status::InvalidArgument("invocation must not be null"),
                        errorCallback, errorUserData);
  }
  Status validation = options->validate();
  if (!validation.ok()) {
    return reportStatus(validation, errorCallback, errorUserData);
  }

  std::string key =
      mlir::nv_tensor_ir::calculateCacheKey(unwrap(module), *options);
  keyCallback(mlirStringRefCreate(key.data(), key.size()), keyUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult mlirTensorIRCudaTileAnalyzeModule(
    MlirModule module, MlirTensorIRCompilerInvocation invocation,
    MlirTensorIRTileCandidateCallback candidateCallback,
    void *candidateUserData, MlirStringCallback errorCallback,
    void *errorUserData) {
  if (mlirModuleIsNull(module)) {
    return reportStatus(Status::InvalidArgument("module must not be null"),
                        errorCallback, errorUserData);
  }
  if (!invocation.ptr) {
    return reportStatus(Status::InvalidArgument("invocation must not be null"),
                        errorCallback, errorUserData);
  }
  if (!candidateCallback) {
    return reportStatus(
        Status::InvalidArgument("candidateCallback must not be null"),
        errorCallback, errorUserData);
  }
  const auto &options =
      *static_cast<const CompilerInvocation *>(invocation.ptr);
  mlir::nv_tensor_ir::backend::cuda_tile::CudaTileCompiler compiler;
  auto candidates = compiler.analyzeModule(unwrap(module), options);
  if (!candidates.ok()) {
    return reportStatus(candidates.status(), errorCallback, errorUserData);
  }
  for (const auto &candidate : *candidates) {
    candidateCallback(candidate.shape.data(), candidate.shape.size(),
                      candidateUserData);
  }
  return mlirLogicalResultSuccess();
}

MlirLogicalResult mlirTensorIRCanCompile(
    MlirModule module, MlirTensorIRCompilerInvocation invocation,
    bool *canCompile, MlirStringCallback errorCallback, void *errorUserData) {
  if (mlirModuleIsNull(module)) {
    return reportStatus(Status::InvalidArgument("module must not be null"),
                        errorCallback, errorUserData);
  }
  auto prepared = prepareCompile(invocation, errorCallback, errorUserData);
  if (!prepared) {
    return mlirLogicalResultFailure();
  }
  auto &[cppInvocation, compiler] = *prepared;
  *canCompile = compiler->canCompile(unwrap(module), *cppInvocation);
  return mlirLogicalResultSuccess();
}

MlirTensorIRArtifact mlirTensorIRArtifactCompile(
    MlirModule module, MlirTensorIRCompilerInvocation invocation,
    MlirStringCallback errorCallback, void *errorUserData) {
  if (mlirModuleIsNull(module)) {
    reportStatus(Status::InvalidArgument("module must not be null"),
                 errorCallback, errorUserData);
    return {nullptr};
  }
  auto prepared = prepareCompile(invocation, errorCallback, errorUserData);
  if (!prepared) {
    return {nullptr};
  }
  auto &[options, compiler] = *prepared;
  auto artifact = compiler->compile(unwrap(module), *options);
  if (!artifact.ok()) {
    reportStatus(artifact.status(), errorCallback, errorUserData);
    return {nullptr};
  }
  return {artifact->release()};
}

void mlirTensorIRArtifactDestroy(MlirTensorIRArtifact artifact) {
  delete static_cast<tensor_ir::rt::TensorIRArtifact *>(artifact.ptr);
}

MlirLogicalResult mlirTensorIRArtifactGetBytecode(
    MlirTensorIRArtifact artifact, MlirStringCallback callback, void *userData,
    MlirStringCallback errorCallback, void *errorUserData) {
  if (!artifact.ptr) {
    return reportStatus(Status::InvalidArgument("TensorIR artifact is null"),
                        errorCallback, errorUserData);
  }
  const auto &binary =
      static_cast<tensor_ir::rt::TensorIRArtifact *>(artifact.ptr)->getBinary();
  std::visit(
      [&](const auto &code) {
        callback(mlirStringRefCreate(code.bytes.data(), code.bytes.size()),
                 userData);
      },
      binary);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult mlirTensorIRArtifactSerialize(
    MlirTensorIRArtifact artifact, MlirStringCallback callback, void *userData,
    MlirStringCallback errorCallback, void *errorUserData) {
  if (!artifact.ptr) {
    return reportStatus(Status::InvalidArgument("TensorIR artifact is null"),
                        errorCallback, errorUserData);
  }
  auto bytes = tensor_ir::rt::serializeArtifact(
      *static_cast<tensor_ir::rt::TensorIRArtifact *>(artifact.ptr));
  if (!bytes.ok()) {
    return reportStatus(bytes.status(), errorCallback, errorUserData);
  }
  callback(mlirStringRefCreate((*bytes)->getBufferStart(),
                               (*bytes)->getBufferSize()),
           userData);
  return mlirLogicalResultSuccess();
}

MlirTensorIRArtifact
mlirTensorIRArtifactDeserialize(MlirStringRef bytes,
                                MlirStringCallback errorCallback,
                                void *errorUserData) {
  auto artifact = tensor_ir::rt::deserializeArtifact(
      llvm::MemoryBufferRef(llvm::StringRef(bytes.data, bytes.length), ""));
  if (!artifact.ok()) {
    reportStatus(artifact.status(), errorCallback, errorUserData);
    return {nullptr};
  }
  return {artifact->release()};
}

MlirTensorIRRuntimeKernel
mlirTensorIRArtifactLoad(MlirTensorIRArtifact artifact,
                         MlirStringCallback errorCallback,
                         void *errorUserData) {
  if (!artifact.ptr) {
    reportStatus(Status::InvalidArgument("TensorIR artifact is null"),
                 errorCallback, errorUserData);
    return {nullptr};
  }
  auto copy = std::make_unique<tensor_ir::rt::TensorIRArtifact>(
      *static_cast<tensor_ir::rt::TensorIRArtifact *>(artifact.ptr));
  auto kernel = CudaTileRuntimeKernel::create(std::move(copy));
  if (!kernel.ok()) {
    reportStatus(kernel.status(), errorCallback, errorUserData);
    return {nullptr};
  }
  Status status = (*kernel)->initializeRuntimeState();
  if (!status.ok()) {
    reportStatus(status, errorCallback, errorUserData);
    return {nullptr};
  }
  return {kernel->release()};
}

void mlirTensorIRRuntimeKernelDestroy(MlirTensorIRRuntimeKernel kernel) {
  delete static_cast<tensor_ir::rt::IRuntimeKernel *>(kernel.ptr);
}

MlirLogicalResult mlirTensorIRRuntimeKernelCheckSupport(
    MlirTensorIRRuntimeKernel kernel, MlirTensorIRPackedArgs args,
    bool *supported, MlirStringCallback errorCallback, void *errorUserData) {
  if (!kernel.ptr) {
    return reportStatus(
        Status::InvalidArgument("TensorIR runtime kernel is null"),
        errorCallback, errorUserData);
  }
  auto unpacked = unwrapPackedArgs(args);
  Status status = static_cast<tensor_ir::rt::IRuntimeKernel *>(kernel.ptr)
                      ->checkSupport(makePackedArgs(unpacked));
  *supported = status.ok();
  if (status.code() == StatusCode::kConstraintNotSatisfied) {
    return mlirLogicalResultSuccess();
  }
  return reportStatus(status, errorCallback, errorUserData);
}

MlirLogicalResult mlirTensorIRRuntimeKernelQueryWorkspaceSize(
    MlirTensorIRRuntimeKernel kernel, MlirTensorIRPackedArgs args,
    size_t *workspaceSize, MlirStringCallback errorCallback,
    void *errorUserData) {
  if (!kernel.ptr) {
    return reportStatus(
        Status::InvalidArgument("TensorIR runtime kernel is null"),
        errorCallback, errorUserData);
  }
  auto unpacked = unwrapPackedArgs(args);
  *workspaceSize = static_cast<tensor_ir::rt::IRuntimeKernel *>(kernel.ptr)
                       ->queryWorkspaceSize(makePackedArgs(unpacked));
  return mlirLogicalResultSuccess();
}

MlirLogicalResult mlirTensorIRRuntimeKernelLaunch(
    MlirTensorIRRuntimeKernel kernel, MlirTensorIRPackedArgs args,
    void *workspaceData, size_t workspaceSize, void *stream,
    MlirStringCallback errorCallback, void *errorUserData) {
  if (!kernel.ptr) {
    return reportStatus(
        Status::InvalidArgument("TensorIR runtime kernel is null"),
        errorCallback, errorUserData);
  }
  auto unpacked = unwrapPackedArgs(args);
  return reportStatus(static_cast<tensor_ir::rt::IRuntimeKernel *>(kernel.ptr)
                          ->launch(makePackedArgs(unpacked),
                                   Workspace{workspaceData, workspaceSize},
                                   reinterpret_cast<Stream>(stream)),
                      errorCallback, errorUserData);
}
