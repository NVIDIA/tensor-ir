// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CudaTileRuntimeKernel - Compiled runtime kernel for CudaTile backend

#ifndef TENSOR_IR_RUNTIME_CUDATILE_CUDATILERUNTIMEKERNEL_H
#define TENSOR_IR_RUNTIME_CUDATILE_CUDATILERUNTIMEKERNEL_H

#include "tensor_ir/Artifact/TensorIRArtifact.h"
#include "tensor_ir/Runtime/CudaTile/KernelLaunchHelpers.h"
#include "tensor_ir/Runtime/CudaTile/RuntimeOperandAccessor.h"
#include "tensor_ir/Runtime/IRuntimeKernel.h"
#include "tensor_ir/Support/Status.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

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

  /// Builds a ready-to-launch kernel from an artifact: wires the arg packer
  /// and grid computer strategies from the artifact's layout, so the
  /// result needs no further setup before initializeRuntimeState()/launch().
  static StatusOr<IRuntimeKernelPtr>
  create(std::unique_ptr<TensorIRArtifact> artifact);

  /// Rebuilds a kernel from bytes produced by serializeArtifact(), restoring
  /// the same launch-readiness as create(), plus the TileIR JIT fallback
  /// when the artifact carries bytecode.
  static StatusOr<IRuntimeKernelPtr> deserialize(llvm::MemoryBufferRef bytes);

  Kind getKind() const override { return Kind::CudaTile; }

  // To implement LLVM-style polymorphism
  static bool classof(const IRuntimeKernel *kernel) {
    return kernel->getKind() == Kind::CudaTile;
  }

  Status initializeRuntimeState() const override;

  Status unloadRuntimeState() const override;

  Status checkSupport(PackedArgs /*args*/) const override;

  size_t queryWorkspaceSize(PackedArgs /*args*/) const override;

  Status launch(PackedArgs args, Workspace workspace,
                Stream stream) const override;

  bool hasDeviceCode() const { return !activeDeviceCode().empty(); }

  llvm::ArrayRef<char> deviceCode() const { return activeDeviceCode(); }

  bool hasTileIRBytecode() const {
    static constexpr char kTileIRMagic[] = "\x7fTileIR\x00";
    llvm::ArrayRef<char> code = activeDeviceCode();
    return llvm::StringRef(code.data(), code.size())
        .starts_with(llvm::StringRef(kTileIRMagic, sizeof(kTileIRMagic) - 1));
  }

  const std::string &name() const override { return artifact_->kernelName; }

  const std::string &funcName() const { return artifact_->funcName; }

  const TensorIRArtifact &getArtifact() const { return *artifact_; }

private:
  explicit CudaTileRuntimeKernel(
      std::unique_ptr<const TensorIRArtifact> artifact)
      : artifact_(std::move(artifact)) {}

  /// Set by create(); must be non-null before first launch (checked there).
  void setArgPacker(std::unique_ptr<RuntimeArgPacker> packer) {
    argPacker_ = std::move(packer);
  }

  void setGridComputer(std::unique_ptr<RuntimeGridComputer> computer) {
    gridComputer_ = std::move(computer);
  }

  /// Set when the artifact carries TileIR bytecode, for the case the CUDA
  /// driver can't JIT it directly.
  void setTileIRFallbackAssembler(TileIRFallbackAssembler assembler) {
    tileIRFallbackAssembler_ = std::move(assembler);
  }

  /// The original binary, unless a TileIR fallback JIT has since replaced
  /// it (fallbackCubin_ non-empty), in which case that takes precedence.
  /// The artifact's own binary is never mutated -- it stays the immutable
  /// compile product.
  llvm::ArrayRef<char> activeDeviceCode() const {
    if (!fallbackCubin_.empty()) {
      return llvm::ArrayRef<char>(fallbackCubin_);
    }
    return std::visit(
        [](const auto &b) { return llvm::ArrayRef<char>(b.bytes); },
        artifact_->binary);
  }

  std::unique_ptr<const TensorIRArtifact> artifact_;
  mutable llvm::SmallVector<char, 0> fallbackCubin_;
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
