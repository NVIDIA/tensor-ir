// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Runtime/CudaTile/CudaTileRuntimeKernel.h"

#include "tensor_ir/Artifact/TensorIRArtifactCodec.h"
#include "tensor_ir/Compiler/CudaTile/TileIRAssembly.h"
#include "tensor_ir/Support/CudaApi.h"

#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace tensor_ir::rt {
namespace cuda = ::mlir::nv_tensor_ir::cuda;
using mlir::nv_tensor_ir::SmTarget;

/// Inline buffer size for SmallVectors holding kernel arguments.
static constexpr unsigned kMaxInlineKernelArgs = 128;

StatusOr<IRuntimeKernelPtr>
CudaTileRuntimeKernel::create(std::unique_ptr<TensorIRArtifact> artifact) {
  std::unique_ptr<const TensorIRArtifact> owned = std::move(artifact);
  const KernelArgLayout &layout = owned->argLayout;
  std::optional<std::array<int32_t, 3>> staticGrid = owned->staticGrid;
  std::optional<::mlir::cuda_tile::BytecodeVersion> bytecodeVersion;
  if (const auto *tileir = std::get_if<TileIRBytecode>(&owned->binary)) {
    bytecodeVersion = tileir->version;
  }
  SmTarget compiledTarget = owned->arch;

  std::unique_ptr<CudaTileRuntimeKernel> kernel(
      new CudaTileRuntimeKernel(std::move(owned)));

  using Acc = RuntimeOperandAccessor;
  if (layout.uniformSignature || layout.hasDynamicShapes()) {
    kernel->setArgPacker(std::make_unique<FlatArgPacker<Acc>>(layout));
  } else {
    kernel->setArgPacker(std::make_unique<PointerOnlyArgPacker<Acc>>());
  }
  if (staticGrid) {
    kernel->setGridComputer(
        std::make_unique<StaticGridComputer<Acc>>(*staticGrid));
  } else {
    kernel->setGridComputer(
        std::make_unique<TileBasedGridComputer<Acc>>(layout));
  }

  if (std::holds_alternative<TileIRBytecode>(kernel->artifact_->binary)) {
    kernel->setTileIRFallbackAssembler(
        [compiledTarget, bytecodeVersion](
            llvm::ArrayRef<char> bytecode,
            int deviceCc) -> StatusOr<llvm::SmallVector<char, 0>> {
          using ::mlir::nv_tensor_ir::ArchPortability;
          using ::mlir::nv_tensor_ir::backend::cuda_tile::assembleTileIRToCubin;
          if (!compiledTarget.validateCodegenTargetCompatibility(deviceCc)) {
            return Status::NotSupported(
                "TileIR target is incompatible with the current CUDA "
                "device");
          }
          ::mlir::FailureOr<SmTarget> deviceTarget =
              SmTarget::fromCc(deviceCc, ArchPortability::arch_conditional);
          if (::mlir::failed(deviceTarget)) {
            return Status::NotSupported(
                "unrecognized CUDA device compute capability");
          }
          TIR_ASSIGN_OR_RETURN(
              auto cubin,
              assembleTileIRToCubin(bytecode, *deviceTarget, bytecodeVersion));
          if (!cubin) {
            return Status::NotSupported("no compatible TileIR assembler");
          }
          return std::move(*cubin);
        });
  }

  return IRuntimeKernelPtr(std::move(kernel));
}

StatusOr<IRuntimeKernelPtr>
CudaTileRuntimeKernel::deserialize(llvm::MemoryBufferRef bytes) {
  TIR_ASSIGN_OR_RETURN(std::unique_ptr<TensorIRArtifact> artifact,
                       deserializeArtifact(bytes));
  return CudaTileRuntimeKernel::create(std::move(artifact));
}

static StatusOr<int> getCurrentDeviceComputeCapability() {
  StatusOr<int> device = cuda::runtime::getDevice();
  if (!device.ok()) {
    return device.status();
  }
  StatusOr<cuda::runtime::DeviceProperties> properties =
      cuda::runtime::getDeviceProperties(*device);
  if (!properties.ok()) {
    return properties.status();
  }
  return properties->major * 10 + properties->minor;
}

Status CudaTileRuntimeKernel::initializeRuntimeState() const {
  llvm::ArrayRef<char> code = activeDeviceCode();
  if (code.empty()) {
    return Status::NotSupported("No device code available");
  }

  auto loadKernel = [&](llvm::ArrayRef<char> deviceCode) -> Status {
    StatusOr<CUlibrary> library = cuda::driver::loadLibrary(deviceCode.data());
    if (!library.ok()) {
      return library.status();
    }
    lib_ = *library;
    StatusOr<CUkernel> kernel =
        cuda::driver::getKernel(lib_, artifact_->funcName.c_str());
    if (!kernel.ok()) {
      (void)cuda::driver::unloadLibrary(lib_);
      lib_ = nullptr;
      return kernel.status();
    }
    kernel_ = *kernel;
    return Status::Ok();
  };

  const auto *tileir = std::get_if<TileIRBytecode>(&artifact_->binary);

  Status driverStatus = loadKernel(code);
  bool alreadyFellBack = !fallbackCubin_.empty();
  if (driverStatus.ok() || alreadyFellBack || !tileir) {
    return driverStatus;
  }

  if (tileIRFallbackAssembler_) {
    StatusOr<int> deviceComputeCapability = getCurrentDeviceComputeCapability();
    if (!deviceComputeCapability.ok()) {
      return deviceComputeCapability.status();
    }
    StatusOr<llvm::SmallVector<char, 0>> fallback =
        tileIRFallbackAssembler_(code, *deviceComputeCapability);
    if (!fallback.ok()) {
      return fallback.status();
    }
    Status fallbackStatus = loadKernel(*fallback);
    if (!fallbackStatus.ok()) {
      return fallbackStatus;
    }
    fallbackCubin_ = std::move(*fallback);
    return Status::Ok();
  }

  StatusOr<int> driverVersion = cuda::driver::getVersion();
  if (driverVersion.ok() && tileir->version) {
    auto message =
        getBytecodeDriverCompatibilityMessage(*tileir->version, *driverVersion);
    if (message) {
      return Status::NotSupported(driverStatus.message() + ". " + *message);
    }
  }
  return driverStatus;
}

Status CudaTileRuntimeKernel::unloadRuntimeState() const {
  if (lib_) {
    Status status = ::mlir::nv_tensor_ir::cuda::driver::unloadLibrary(lib_);
    if (!status.ok()) {
      return status;
    }
    lib_ = nullptr;
    kernel_ = nullptr;
  }

  return Status::Ok();
}

Status CudaTileRuntimeKernel::checkSupport(PackedArgs /*args*/) const {
  // TODO: Validate arguments match kernel signature
  return Status::Ok();
}

size_t CudaTileRuntimeKernel::queryWorkspaceSize(PackedArgs /*args*/) const {
  // TODO: Calculate workspace size from kernel metadata
  return 0;
}

Status CudaTileRuntimeKernel::launch(PackedArgs args, Workspace workspace,
                                     Stream stream) const {
  (void)workspace;

  if (activeDeviceCode().empty()) {
    return Status::NotSupported("No device code available");
  }

  if (!argPacker_ || !gridComputer_) {
    return Status::InvalidArgument(
        "CudaTileRuntimeKernel: argPacker or gridComputer not set");
  }

  // Wrap OSS PackedArgs in the generic accessor.
  RuntimeOperandAccessor acc{args};

  // Pack arguments via strategy.
  int32_t numArgs = argPacker_->numKernelArgs(acc.size());
  llvm::SmallVector<int64_t, kMaxInlineKernelArgs> argValues(numArgs);
  int32_t packed = argPacker_->packArgs(acc, argValues);
  if (packed < 0) {
    return Status::InvalidArgument(
        "Failed to pack kernel arguments: unsupported operand kind");
  }

  // Compute grid via strategy.
  ::tensor_ir::rt::GridDims grid = gridComputer_->computeGrid(acc);

  // Configure launch parameters.
  CUlaunchConfig config = {};
  config.gridDimX = grid.x;
  config.gridDimY = grid.y;
  config.gridDimZ = grid.z;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = stream;

  CUlaunchAttribute launchAttr[1];
  launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
  launchAttr[0].value.clusterSchedulingPolicyPreference =
      CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
  config.attrs = launchAttr;
  config.numAttrs = 1;

  // Build the pointer-to-arg array required by cuLaunchKernelEx.
  llvm::SmallVector<void *, kMaxInlineKernelArgs> kernelParams(numArgs);
  for (int32_t i = 0; i < numArgs; ++i) {
    kernelParams[i] = &argValues[i];
  }

  return ::mlir::nv_tensor_ir::cuda::driver::launchKernel(config, kernel_,
                                                          kernelParams.data());
}

} // namespace tensor_ir::rt
