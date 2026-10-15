// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TensorIRArtifact - the persistent compile product for TensorIR's CudaTile
// backend.

#ifndef TENSOR_IR_ARTIFACT_TENSORIRARTIFACT_H
#define TENSOR_IR_ARTIFACT_TENSORIRARTIFACT_H

#include "tensor_ir/Artifact/KernelMetadata.h"
#include "tensor_ir/Support/Status.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "cuda_tile/Bytecode/Common/Version.h"
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace tensor_ir::rt {

/// A real cubin -- no bytecode version, since it's not TileIR bytecode.
struct Cubin {
  llvm::SmallVector<char, 0> bytes;

  bool operator==(const Cubin &other) const { return bytes == other.bytes; }
};

/// TileIR bytecode. `version` is absent when it's genuinely unknown to us
/// (e.g. loaded from a file outside the normal compile path) -- the
/// bytecode itself still encodes its own version internally.
struct TileIRBytecode {
  llvm::SmallVector<char, 0> bytes;
  std::optional<::mlir::cuda_tile::BytecodeVersion> version;

  bool operator==(const TileIRBytecode &other) const {
    return bytes == other.bytes && version == other.version;
  }
};

/// The kernel binary itself: either a cubin, or TileIR bytecode (whose
/// version may or may not be known).
using Binary = std::variant<Cubin, TileIRBytecode>;

/// Relocatable native host code implementing HostRuntimeAbi.h.
struct HostModule {
  llvm::SmallVector<char, 0> bytes;
  uint32_t abiVersion;
  std::string triple;

  bool operator==(const HostModule &other) const {
    return bytes == other.bytes && abiVersion == other.abiVersion &&
           triple == other.triple;
  }
};

/// TensorIR compiler output: cubin or TileIR bytecode, paired with the
/// launch metadata produced by the same compilation. Only constructible via
/// create(), which validates internally: there is no way to hold an
/// invalid TensorIRArtifact.
class TensorIRArtifact final {
public:
  friend class TensorIRArtifactEncoder;

  /// `usesPDL` records that the device code waits on the preceding grid, which
  /// makes programmatic stream serialization safe.
  static ::mlir::nv_tensor_ir::StatusOr<std::unique_ptr<TensorIRArtifact>>
  create(std::string kernelName, std::string funcName, Binary &&binary,
         ::mlir::nv_tensor_ir::SmTarget arch, KernelMetadata argLayout,
         const std::optional<std::array<int32_t, 3>> &staticGrid,
         bool usesPDL = false,
         std::optional<HostModule> hostModule = std::nullopt,
         std::string entrypointName = {});

  TensorIRArtifact(const TensorIRArtifact &) = default;
  TensorIRArtifact &operator=(const TensorIRArtifact &) = delete;
  TensorIRArtifact(TensorIRArtifact &&) = default;
  TensorIRArtifact &operator=(TensorIRArtifact &&) = delete;
  ~TensorIRArtifact() = default;

  bool operator==(const TensorIRArtifact &other) const {
    return kernelName == other.kernelName && funcName == other.funcName &&
           binary == other.binary && arch == other.arch &&
           argLayout == other.argLayout && staticGrid == other.staticGrid &&
           usesPDL_ == other.usesPDL_ && hostModule == other.hostModule &&
           entrypointName == other.entrypointName;
  }

  const std::string &getKernelName() const { return kernelName; }
  const std::string &getFuncName() const { return funcName; }
  /// Name of the sole invocable entrypoint; an exported host symbol for native
  /// artifacts. Independent of the display name and device kernel symbol.
  const std::string &getEntrypointName() const { return entrypointName; }
  const std::optional<HostModule> &getHostModule() const { return hostModule; }
  const Binary &getBinary() const { return binary; }
  ::mlir::nv_tensor_ir::SmTarget getArch() const { return arch; }
  const KernelMetadata &getLaunchMetadata() const { return argLayout; }
  const std::optional<std::array<int32_t, 3>> &getStaticGrid() const {
    return staticGrid;
  }
  /// True when the kernel waits on the preceding grid (griddepcontrol), so the
  /// runtime may launch it with programmatic stream serialization.
  bool usesPDL() const { return usesPDL_; }

private:
  TensorIRArtifact(std::string kernelName, std::string funcName,
                   Binary &&binary, ::mlir::nv_tensor_ir::SmTarget arch,
                   KernelMetadata argLayout,
                   const std::optional<std::array<int32_t, 3>> &staticGrid,
                   bool usesPDL, std::optional<HostModule> hostModule,
                   std::string entrypointName);

  static ::mlir::nv_tensor_ir::Status
  validateFields(llvm::StringRef kernelName, llvm::StringRef funcName,
                 const Binary &binary,
                 const std::optional<std::array<int32_t, 3>> &staticGrid,
                 const std::optional<HostModule> &hostModule,
                 llvm::StringRef entrypointName);

  std::string kernelName;
  std::string funcName;
  std::string entrypointName;
  Binary binary;
  std::optional<HostModule> hostModule;
  ::mlir::nv_tensor_ir::SmTarget arch;
  KernelMetadata argLayout;
  /// Absent selects the runtime-computed grid.
  std::optional<std::array<int32_t, 3>> staticGrid;
  bool usesPDL_ = false;
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_ARTIFACT_TENSORIRARTIFACT_H
