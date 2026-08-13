// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir-c/TensorIR.h"

#include "tensor_ir/Compiler/Compiler.h"
#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Registration/Registration.h"
#include "tensor_ir/Runtime/CudaTile/CudaTileRuntimeKernel.h"
#include "tensor_ir/Runtime/Types.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(nv_tensor_ir, nv_tensor_ir,
                                      mlir::nv_tensor_ir::TensorIRDialect)

namespace {

using mlir::cuda_tile::BytecodeVersion;
using mlir::nv_tensor_ir::ArchPortability;
using mlir::nv_tensor_ir::CompilerBackend;
using mlir::nv_tensor_ir::CudaTileCodegenStrategy;
using mlir::nv_tensor_ir::ICompiler;
using mlir::nv_tensor_ir::SmTarget;
using mlir::nv_tensor_ir::Status;
using mlir::nv_tensor_ir::StatusCode;
using mlir::nv_tensor_ir::StatusOr;
using mlir::nv_tensor_ir::backend::cuda_tile::CudaTileArtifactKind;
using mlir::nv_tensor_ir::backend::cuda_tile::CudaTileCompileOptions;
using tensor_ir::rt::Any;
using tensor_ir::rt::CudaTileRuntimeKernel;
using tensor_ir::rt::IRuntimeKernelPtr;
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

ArchPortability unwrapArchPortability(MlirTensorIRArchPortability portability) {
  return static_cast<ArchPortability>(portability);
}

CudaTileCodegenStrategy
unwrapCodegenStrategy(MlirTensorIRCudaTileCodegenStrategy codegenStrategy) {
  return static_cast<CudaTileCodegenStrategy>(codegenStrategy);
}

StatusOr<CudaTileArtifactKind>
unwrapCudaTileArtifactKind(MlirTensorIRCudaTileArtifactKind kind) {
  switch (kind) {
  case MlirTensorIRCudaTileArtifactKindTileIR:
    return CudaTileArtifactKind::TileIR;
  case MlirTensorIRCudaTileArtifactKindCubin:
    return CudaTileArtifactKind::Cubin;
  }
  return Status::InvalidArgument("Unrecognized CUDA Tile artifact kind: " +
                                 std::to_string(static_cast<int>(kind)));
}

std::string copyString(MlirStringRef value) {
  return std::string(value.data, value.length);
}

StatusOr<CudaTileCompileOptions>
makeCudaTileCompileOptions(MlirTensorIRCudaTileCompileOptions options) {
  auto smTarget =
      SmTarget::fromCc(options.computeCapability,
                       unwrapArchPortability(options.archPortability));
  if (mlir::failed(smTarget)) {
    return Status::InvalidArgument("Unsupported TensorIR compute capability: " +
                                   std::to_string(options.computeCapability));
  }

  CudaTileCompileOptions cudaTileOptions{*smTarget};
  cudaTileOptions.ctaCount = options.ctaCount;
  cudaTileOptions.warpCount = options.warpCount;
  if (options.numTileSizes != 0) {
    if (!options.tileSizes) {
      return Status::InvalidArgument(
          "TensorIR tileSizes must not be null when numTileSizes is nonzero");
    }
    cudaTileOptions.tileSize.assign(options.tileSizes,
                                    options.tileSizes + options.numTileSizes);
  }
  cudaTileOptions.uniformSignature = options.uniformSignature;
  cudaTileOptions.codegenStrategy =
      unwrapCodegenStrategy(options.codegenStrategy);
  cudaTileOptions.maxTileCandidates = options.maxTileCandidates;
  cudaTileOptions.irDebug.dumpCudaTileIRPath =
      copyString(options.dumpCudaTileIRPath);
  cudaTileOptions.irDebug.dumpTileIRBCPath =
      copyString(options.dumpTileIRBCPath);
  cudaTileOptions.irDebug.loadTileIRBCPath =
      copyString(options.loadTileIRBCPath);
  cudaTileOptions.irDebug.printIRTreeDir = copyString(options.printIRTreeDir);
  cudaTileOptions.irDebug.printIRAfterAll = options.printIRAfterAll;
  cudaTileOptions.irDebug.enableTiming = options.enableTiming;
  StatusOr<BytecodeVersion> bytecodeVersion =
      unwrapBytecodeVersion(options.bytecodeVersion);
  if (!bytecodeVersion.ok()) {
    return bytecodeVersion.status();
  }
  cudaTileOptions.bytecodeVersion = *bytecodeVersion;
  StatusOr<CudaTileArtifactKind> artifactKind =
      unwrapCudaTileArtifactKind(options.artifactKind);
  if (!artifactKind.ok()) {
    return artifactKind.status();
  }
  cudaTileOptions.artifactKind = *artifactKind;

  Status status = cudaTileOptions.validate();
  if (!status.ok()) {
    return status;
  }
  return cudaTileOptions;
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

MlirLogicalResult reportStatus(Status status, MlirStringCallback errorCallback,
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

const CudaTileRuntimeKernel *
getCudaTileRuntimeKernel(const IRuntimeKernelPtr &rtk) {
  if (!rtk) {
    return nullptr;
  }
  // The public compiler currently constructs only CudaTile backend runtime
  // kernels.
  return static_cast<const CudaTileRuntimeKernel *>(rtk.get());
}

class Program {
public:
  Program(std::unique_ptr<ICompiler> compiler, IRuntimeKernelPtr rtk)
      : compiler_(std::move(compiler)), rtk_(std::move(rtk)) {}

  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;
  Program(Program &&) = delete;
  Program &operator=(Program &&) = delete;
  ~Program() = default;

  void destroy() {
    rtk_.reset();
    compiler_.reset();
    initialized_ = false;
  }

  bool isDestroyed() const { return !rtk_; }

  bool isInitialized() const { return rtk_ && initialized_; }

  Status initialize() {
    Status status = validate();
    if (!status.ok()) {
      return status;
    }
    if (initialized_) {
      return Status::Ok();
    }
    status = rtk_->initializeRuntimeState();
    if (status.ok()) {
      initialized_ = true;
    }
    return status;
  }

  StatusOr<bool> checkSupport(PackedArgs args) {
    Status validation = validate();
    if (!validation.ok()) {
      return validation;
    }
    Status status = rtk_->checkSupport(args);
    if (status.ok()) {
      return true;
    }
    if (status.code() == StatusCode::kConstraintNotSatisfied) {
      return false;
    }
    return status;
  }

  StatusOr<size_t> queryWorkspaceSize(PackedArgs args) const {
    Status status = validate();
    if (!status.ok()) {
      return status;
    }
    return rtk_->queryWorkspaceSize(args);
  }

  Status launch(PackedArgs args, Workspace workspace, Stream stream) {
    Status status = initialize();
    if (!status.ok()) {
      return status;
    }
    return rtk_->launch(args, workspace, stream);
  }

  StatusOr<llvm::ArrayRef<char>> getBytecode() const {
    const CudaTileRuntimeKernel *rtk = getCudaTileRuntimeKernel(rtk_);
    if (!rtk) {
      return Status::NotSupported(
          "TensorIR compiled program is not a CudaTile runtime kernel");
    }
    return rtk->deviceCode();
  }

private:
  Status validate() const {
    if (isDestroyed()) {
      return Status::InvalidArgument(
          "TensorIR program has already been destroyed");
    }
    return Status::Ok();
  }

  std::unique_ptr<ICompiler> compiler_;
  IRuntimeKernelPtr rtk_;
  bool initialized_ = false;
};

MlirTensorIRProgram wrapProgram(Program *program) { return {program}; }

Program *unwrapProgram(MlirTensorIRProgram program) {
  return static_cast<Program *>(program.ptr);
}

MlirLogicalResult reportNullProgram(MlirStringCallback errorCallback,
                                    void *errorUserData) {
  return reportStatus(Status::InvalidArgument("TensorIR program is null"),
                      errorCallback, errorUserData);
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

MlirTensorIRProgram mlirTensorIRProgramCompile(
    MlirModule module, MlirTensorIRCudaTileCompileOptions options,
    MlirStringCallback errorCallback, void *errorUserData) {
  StatusOr<CudaTileCompileOptions> cudaTileOptions =
      makeCudaTileCompileOptions(options);
  if (!cudaTileOptions.ok()) {
    reportStatus(cudaTileOptions.status(), errorCallback, errorUserData);
    return wrapProgram(nullptr);
  }

  std::unique_ptr<ICompiler> compiler =
      ICompiler::create(CompilerBackend::CudaTile);
  if (!compiler) {
    emitError("TensorIR CudaTile compiler is unavailable", errorCallback,
              errorUserData);
    return wrapProgram(nullptr);
  }

  StatusOr<IRuntimeKernelPtr> rtk =
      compiler->compile(unwrap(module), *cudaTileOptions);
  if (!rtk.ok()) {
    reportStatus(rtk.status(), errorCallback, errorUserData);
    return wrapProgram(nullptr);
  }

  return wrapProgram(new Program(std::move(compiler), std::move(rtk).value()));
}

MlirLogicalResult mlirTensorIRProgramCanCompile(
    MlirModule module, MlirTensorIRCudaTileCompileOptions options,
    bool *canCompile, MlirStringCallback errorCallback, void *errorUserData) {
  StatusOr<CudaTileCompileOptions> cudaTileOptions =
      makeCudaTileCompileOptions(options);
  if (!cudaTileOptions.ok()) {
    return reportStatus(cudaTileOptions.status(), errorCallback, errorUserData);
  }

  std::unique_ptr<ICompiler> compiler =
      ICompiler::create(CompilerBackend::CudaTile);
  if (!compiler) {
    emitError("TensorIR CudaTile compiler is unavailable", errorCallback,
              errorUserData);
    return mlirLogicalResultFailure();
  }

  *canCompile = compiler->canCompile(unwrap(module), *cudaTileOptions);
  return mlirLogicalResultSuccess();
}

void mlirTensorIRProgramDelete(MlirTensorIRProgram program) {
  Program *cppProgram = unwrapProgram(program);
  if (cppProgram) {
    delete cppProgram;
  }
}

void mlirTensorIRProgramDestroy(MlirTensorIRProgram program) {
  Program *cppProgram = unwrapProgram(program);
  if (cppProgram) {
    cppProgram->destroy();
  }
}

bool mlirTensorIRProgramIsDestroyed(MlirTensorIRProgram program) {
  Program *cppProgram = unwrapProgram(program);
  if (cppProgram) {
    return cppProgram->isDestroyed();
  } else {
    return true;
  }
}

bool mlirTensorIRProgramIsInitialized(MlirTensorIRProgram program) {
  Program *cppProgram = unwrapProgram(program);
  if (cppProgram) {
    return cppProgram->isInitialized();
  } else {
    return false;
  }
}

MlirLogicalResult
mlirTensorIRProgramInitialize(MlirTensorIRProgram program,
                              MlirStringCallback errorCallback,
                              void *errorUserData) {
  Program *cppProgram = unwrapProgram(program);
  if (!cppProgram) {
    return reportNullProgram(errorCallback, errorUserData);
  }
  return reportStatus(cppProgram->initialize(), errorCallback, errorUserData);
}

MlirLogicalResult mlirTensorIRProgramCheckSupport(
    MlirTensorIRProgram program, MlirTensorIRPackedArgs args, bool *supported,
    MlirStringCallback errorCallback, void *errorUserData) {
  Program *cppProgram = unwrapProgram(program);
  if (!cppProgram) {
    return reportNullProgram(errorCallback, errorUserData);
  }
  llvm::SmallVector<Any, 8> unpackedArgs = unwrapPackedArgs(args);
  StatusOr<bool> result =
      cppProgram->checkSupport(makePackedArgs(unpackedArgs));
  if (!result.ok()) {
    return reportStatus(result.status(), errorCallback, errorUserData);
  }
  *supported = *result;
  return mlirLogicalResultSuccess();
}

MlirLogicalResult mlirTensorIRProgramQueryWorkspaceSize(
    MlirTensorIRProgram program, MlirTensorIRPackedArgs args,
    size_t *workspaceSize, MlirStringCallback errorCallback,
    void *errorUserData) {
  Program *cppProgram = unwrapProgram(program);
  if (!cppProgram) {
    return reportNullProgram(errorCallback, errorUserData);
  }
  llvm::SmallVector<Any, 8> unpackedArgs = unwrapPackedArgs(args);
  StatusOr<size_t> result =
      cppProgram->queryWorkspaceSize(makePackedArgs(unpackedArgs));
  if (!result.ok()) {
    return reportStatus(result.status(), errorCallback, errorUserData);
  }
  *workspaceSize = *result;
  return mlirLogicalResultSuccess();
}

MlirLogicalResult mlirTensorIRProgramLaunch(MlirTensorIRProgram program,
                                            MlirTensorIRPackedArgs args,
                                            void *workspaceData,
                                            size_t workspaceSize, void *stream,
                                            MlirStringCallback errorCallback,
                                            void *errorUserData) {
  Program *cppProgram = unwrapProgram(program);
  if (!cppProgram) {
    return reportNullProgram(errorCallback, errorUserData);
  }
  llvm::SmallVector<Any, 8> unpackedArgs = unwrapPackedArgs(args);
  Workspace workspace{workspaceData, workspaceSize};
  Stream cudaStream = reinterpret_cast<Stream>(stream);
  return reportStatus(
      cppProgram->launch(makePackedArgs(unpackedArgs), workspace, cudaStream),
      errorCallback, errorUserData);
}

MlirLogicalResult mlirTensorIRProgramGetBytecode(
    MlirTensorIRProgram program, MlirStringCallback bytecodeCallback,
    void *bytecodeUserData, MlirStringCallback errorCallback,
    void *errorUserData) {
  Program *cppProgram = unwrapProgram(program);
  if (!cppProgram) {
    return reportNullProgram(errorCallback, errorUserData);
  }
  StatusOr<llvm::ArrayRef<char>> bytecode = cppProgram->getBytecode();
  if (!bytecode.ok()) {
    return reportStatus(bytecode.status(), errorCallback, errorUserData);
  }
  bytecodeCallback(mlirStringRefCreate(bytecode->data(), bytecode->size()),
                   bytecodeUserData);
  return mlirLogicalResultSuccess();
}
