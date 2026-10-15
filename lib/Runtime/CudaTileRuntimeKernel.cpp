// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Runtime/CudaTile/CudaTileRuntimeKernel.h"

#include "tensor_ir/Artifact/TensorIRArtifactCodec.h"
#include "tensor_ir/Compiler/CudaTile/TileIRAssembly.h"
#include "tensor_ir/Support/CudaApi.h"

#include "llvm/ADT/SmallVector.h"

#include "NativeHostModule.h"
#include <array>
#include <mutex>
#include <string>
#include <utility>

namespace tensor_ir::rt {
namespace cuda = ::mlir::nv_tensor_ir::cuda;
using mlir::nv_tensor_ir::SmTarget;

/// Inline buffer size for SmallVectors holding kernel arguments.
static constexpr unsigned kMaxInlineKernelArgs = 128;

CudaTileRuntimeKernel::CudaTileRuntimeKernel(
    std::unique_ptr<const TensorIRArtifact> artifact)
    : artifact_(std::move(artifact)),
      argPacker_(artifact_->getLaunchMetadata()) {}

CudaTileRuntimeKernel::~CudaTileRuntimeKernel() { (void)unloadRuntimeState(); }

StatusOr<IRuntimeKernelPtr>
CudaTileRuntimeKernel::create(std::unique_ptr<TensorIRArtifact> artifact) {
  std::unique_ptr<const TensorIRArtifact> owned = std::move(artifact);
  const KernelMetadata &layout = owned->getLaunchMetadata();
  const std::optional<std::array<int32_t, 3>> &staticGrid =
      owned->getStaticGrid();
  std::optional<::mlir::cuda_tile::BytecodeVersion> bytecodeVersion;
  if (const auto *tileir = std::get_if<TileIRBytecode>(&owned->getBinary())) {
    bytecodeVersion = tileir->version;
  }
  SmTarget compiledTarget = owned->getArch();

  std::unique_ptr<CudaTileRuntimeKernel> kernel(
      new CudaTileRuntimeKernel(std::move(owned)));

  using Acc = RuntimeOperandAccessor;
  if (staticGrid) {
    kernel->setGridComputer(
        std::make_unique<StaticGridComputer<Acc>>(*staticGrid));
  } else {
    kernel->setGridComputer(
        std::make_unique<TileBasedGridComputer<Acc>>(layout));
  }

  if (std::holds_alternative<TileIRBytecode>(kernel->artifact_->getBinary())) {
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
  std::unique_lock lock(stateMutex_);
  if (kernel_) {
    return Status::Ok();
  }
  std::unique_ptr<NativeHostModule> host;
  if (artifact_->getHostModule()) {
    TIR_ASSIGN_OR_RETURN(
        host, NativeHostModule::load(*artifact_->getHostModule(),
                                     artifact_->getEntrypointName()));
  }
  TIR_RETURN_IF_ERROR(initializeDeviceState());
  hostModule_ = std::move(host);
  return Status::Ok();
}

Status CudaTileRuntimeKernel::initializeDeviceState() const {
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
        cuda::driver::getKernel(lib_, artifact_->getFuncName().c_str());
    if (!kernel.ok()) {
      (void)cuda::driver::unloadLibrary(lib_);
      lib_ = nullptr;
      return kernel.status();
    }
    kernel_ = *kernel;
    return Status::Ok();
  };

  const auto *tileir = std::get_if<TileIRBytecode>(&artifact_->getBinary());

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
  std::unique_lock lock(stateMutex_);
  if (lib_) {
    Status status = ::mlir::nv_tensor_ir::cuda::driver::unloadLibrary(lib_);
    if (!status.ok()) {
      return status;
    }
    lib_ = nullptr;
    kernel_ = nullptr;
  }
  hostModule_.reset();

  return Status::Ok();
}

Status CudaTileRuntimeKernel::checkSupport(PackedArgs args) const {
  if (artifact_->getHostModule()) {
    return packNativeHostArgs(artifact_->getLaunchMetadata(), args).status();
  }
  // TODO: Validate interpreter arguments match kernel signature
  return Status::Ok();
}

size_t CudaTileRuntimeKernel::queryWorkspaceSize(PackedArgs /*args*/) const {
  // TODO: Calculate workspace size from kernel metadata
  return 0;
}

namespace {
Status launchCudaKernel(CUkernel kernel, Stream stream,
                        const TensorIRLaunchConfigV1 &launch, void **params,
                        bool usesPDL) {
  CUlaunchConfig config = {};
  config.gridDimX = launch.grid[0];
  config.gridDimY = launch.grid[1];
  config.gridDimZ = launch.grid[2];
  config.blockDimX = launch.block[0];
  config.blockDimY = launch.block[1];
  config.blockDimZ = launch.block[2];
  config.sharedMemBytes = launch.dynamicSharedMemoryBytes;
  config.hStream = stream;
  std::array<CUlaunchAttribute, 2> launchAttrs{};
  unsigned numAttrs = 0;
  launchAttrs[numAttrs].id =
      CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
  launchAttrs[numAttrs].value.clusterSchedulingPolicyPreference =
      CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
  ++numAttrs;
  // Programmatic stream serialization lets this grid start while the previous
  // grid in the stream is still finishing. It is only requested for kernels
  // that wait on the preceding grid (griddepcontrol.wait) before touching
  // global memory, which is what keeps the early start free of races.
  if (usesPDL) {
    launchAttrs[numAttrs].id =
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    launchAttrs[numAttrs].value.programmaticStreamSerializationAllowed = 1;
    ++numAttrs;
  }
  config.attrs = launchAttrs.data();
  config.numAttrs = numAttrs;
  return cuda::driver::launchKernel(config, kernel, params);
}

struct NativeLaunchContext {
  CUkernel kernel;
  Stream stream;
  bool usesPDL;
  Status status;
};

int32_t launchNative(void *userData, uint32_t ordinal,
                     const TensorIRLaunchConfigV1 *config,
                     void *const *params) noexcept {
  auto &context = *static_cast<NativeLaunchContext *>(userData);
  if (ordinal != 0 || !config) {
    context.status = Status::InvalidArgument(
        "Invalid native host kernel ordinal or launch configuration");
  } else {
    context.status =
        launchCudaKernel(context.kernel, context.stream, *config,
                         const_cast<void **>(params), context.usesPDL);
  }
  return context.status.ok() ? 0 : -3;
}
} // namespace

Status CudaTileRuntimeKernel::launch(PackedArgs args, Workspace workspace,
                                     Stream stream) const {
  (void)workspace;
  std::shared_lock lock(stateMutex_);
  if (!kernel_) {
    return Status::NotInitialized("CudaTileRuntimeKernel is not initialized");
  }
  if (hostModule_) {
    TIR_ASSIGN_OR_RETURN(
        auto packed, packNativeHostArgs(artifact_->getLaunchMetadata(), args));
    NativeLaunchContext context{kernel_, stream, artifact_->usesPDL(),
                                Status::Ok()};
    TensorIRHostRuntimeV1 runtime{sizeof(TensorIRHostRuntimeV1),
                                  TENSOR_IR_HOST_ABI_VERSION_1, &context,
                                  launchNative};
    int32_t result = hostModule_->invoke(runtime, packed);
    TIR_RETURN_IF_ERROR(context.status);
    if (result != 0) {
      return Status::InvalidArgument("Native host entry failed with status " +
                                     std::to_string(result));
    }
    return Status::Ok();
  }

  if (activeDeviceCode().empty()) {
    return Status::NotSupported("No device code available");
  }

  if (!gridComputer_) {
    return Status::InvalidArgument(
        "CudaTileRuntimeKernel: gridComputer not set");
  }

  // Wrap OSS PackedArgs in the generic accessor.
  RuntimeOperandAccessor acc{args};

  // Pack arguments according to the compiled signature.
  int32_t numArgs = argPacker_.numKernelArgs();
  llvm::SmallVector<int64_t, kMaxInlineKernelArgs> argValues(numArgs);
  int32_t packed = argPacker_.packArgs(acc, argValues);
  if (packed < 0) {
    return Status::InvalidArgument("Failed to pack kernel arguments");
  }

  // Compute grid via strategy.
  ::tensor_ir::rt::GridDims grid = gridComputer_->computeGrid(acc);

  // Build the pointer-to-arg array required by cuLaunchKernelEx.
  llvm::SmallVector<void *, kMaxInlineKernelArgs> kernelParams(numArgs);
  for (int32_t i = 0; i < numArgs; ++i) {
    kernelParams[i] = &argValues[i];
  }

  TensorIRLaunchConfigV1 config{{static_cast<uint32_t>(grid.x),
                                 static_cast<uint32_t>(grid.y),
                                 static_cast<uint32_t>(grid.z)},
                                {1, 1, 1},
                                0};
  return launchCudaKernel(kernel_, stream, config, kernelParams.data(),
                          artifact_->usesPDL());
}

} // namespace tensor_ir::rt
