// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CudaTileRuntimeKernel - Compiled runtime kernel for CudaTile backend

#ifndef TENSOR_IR_RUNTIME_CUDATILE_CUDATILERUNTIMEKERNEL_H
#define TENSOR_IR_RUNTIME_CUDATILE_CUDATILERUNTIMEKERNEL_H

#include "tensor_ir/Runtime/CudaTile/KernelLaunchHelpers.h"
#include "tensor_ir/Runtime/CudaTile/RuntimeOperandAccessor.h"
#include "tensor_ir/Runtime/IRuntimeKernel.h"
#include "tensor_ir/Support/Status.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "cuda.h"
#include "cuda_tile/Bytecode/Common/Version.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace tensor_ir::rt {

inline int requiredCudaDriverApiVersionForBytecode(
    ::mlir::cuda_tile::BytecodeVersion version) {
  // cuDriverGetVersion encodes CUDA x.y as (x * 1000) + (y * 10). CUDA Tile
  // bytecode major/minor versions track the CUDA Toolkit version they target.
  return static_cast<int>(version.getMajor()) * 1000 +
         static_cast<int>(version.getMinor()) * 10;
}

inline std::string formatCudaDriverApiVersion(int driverVersion) {
  int major = driverVersion / 1000;
  int minor = (driverVersion % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor);
}

inline std::optional<std::string> getBytecodeDriverCompatibilityMessage(
    ::mlir::cuda_tile::BytecodeVersion bytecodeVersion, int driverVersion) {
  int requiredDriverVersion =
      requiredCudaDriverApiVersionForBytecode(bytecodeVersion);
  if (driverVersion >= requiredDriverVersion) {
    return std::nullopt;
  }

  return "CUDA driver API version " +
         formatCudaDriverApiVersion(driverVersion) +
         " is too old for TileIR bytecode version " +
         bytecodeVersion.toString() +
         ". Upgrade to a driver compatible with CUDA Toolkit " +
         formatCudaDriverApiVersion(requiredDriverVersion) +
         " or lower the bytecode target by setting "
         "CompileOptions.bytecode_version to BytecodeVersion.compatibility().";
}

/// Type aliases for runtime-specific template instantiations.
using RuntimeArgPacker =
    ::tensor_ir::rt::KernelArgPacker<RuntimeOperandAccessor>;
using RuntimeGridComputer =
    ::tensor_ir::rt::GridSizeComputer<RuntimeOperandAccessor>;

class CudaTileRuntimeKernel : public IRuntimeKernel {
public:
  using TileIRFallbackAssembler =
      std::function<StatusOr<llvm::SmallVector<char, 0>>(
          llvm::ArrayRef<char>, int deviceComputeCapability)>;

  CudaTileRuntimeKernel(
      std::string name, std::string funcName, llvm::SmallVector<char, 0> cubin,
      std::optional<::mlir::cuda_tile::BytecodeVersion> bytecodeVersion)
      : name_(std::move(name)), funcName_(std::move(funcName)),
        cubin_(std::move(cubin)), bytecodeVersion_(bytecodeVersion) {}

  explicit CudaTileRuntimeKernel(std::string name) : name_(std::move(name)) {}

  Status initializeRuntimeState() const override;

  Status unloadRuntimeState() const override;

  Status checkSupport(PackedArgs /*args*/) const override;

  size_t queryWorkspaceSize(PackedArgs /*args*/) const override;

  Status launch(PackedArgs args, Workspace workspace,
                Stream stream) const override;

  bool hasDeviceCode() const { return !cubin_.empty(); }

  // XLA integration contract: exposes the selected device artifact.
  llvm::ArrayRef<char> deviceCode() const { return cubin_; }

  bool hasTileIRBytecode() const {
    static constexpr char kTileIRMagic[] = "\x7fTileIR\x00";
    return llvm::StringRef(cubin_.data(), cubin_.size())
        .starts_with(llvm::StringRef(kTileIRMagic, sizeof(kTileIRMagic) - 1));
  }

  const std::string &name() const override { return name_; }

  const std::string &funcName() const { return funcName_; }

  /// Set the argument packer strategy (static or dynamic shapes).
  /// Must be called after construction and before first launch.
  void setArgPacker(std::unique_ptr<RuntimeArgPacker> packer) {
    argPacker_ = std::move(packer);
  }

  /// Set the grid-size computation strategy.
  /// Must be called after construction and before first launch.
  void setGridComputer(std::unique_ptr<RuntimeGridComputer> computer) {
    gridComputer_ = std::move(computer);
  }

  /// Set the fallback used when the CUDA driver cannot JIT TileIR.
  void setTileIRFallbackAssembler(TileIRFallbackAssembler assembler) {
    tileIRFallbackAssembler_ = std::move(assembler);
  }

private:
  std::string name_;
  std::string funcName_;
  mutable llvm::SmallVector<char, 0> cubin_;
  mutable std::optional<::mlir::cuda_tile::BytecodeVersion> bytecodeVersion_;
  TileIRFallbackAssembler tileIRFallbackAssembler_;

  mutable CUlibrary lib_ = nullptr;
  mutable CUkernel kernel_ = nullptr;

  /// Strategy for packing tensor operands into the flat kernel arg list.
  std::unique_ptr<RuntimeArgPacker> argPacker_;

  /// Strategy for computing grid dimensions at launch time.
  std::unique_ptr<RuntimeGridComputer> gridComputer_;
};
} // namespace tensor_ir::rt

#endif // TENSOR_IR_RUNTIME_CUDATILE_CUDATILERUNTIMEKERNEL_H
