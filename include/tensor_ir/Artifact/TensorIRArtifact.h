// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TensorIRArtifact - the persistent compile product for TensorIR's CudaTile
// backend.

#ifndef TENSOR_IR_ARTIFACT_TENSORIRARTIFACT_H
#define TENSOR_IR_ARTIFACT_TENSORIRARTIFACT_H

#include "tensor_ir/Runtime/CudaTile/KernelArgLayout.h"
#include "tensor_ir/Support/Status.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include "cuda_tile/Bytecode/Common/Version.h"
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace tensor_ir::rt {

class CudaTileRuntimeKernel;

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

/// TensorIR compiler output: cubin or TileIR bytecode, paired with the
/// launch metadata produced by the same compilation. Only constructible via
/// create(), which validates internally: there is no way to hold an
/// invalid TensorIRArtifact.
class TensorIRArtifact final {
public:
  friend class CudaTileRuntimeKernel;
  friend class TensorIRArtifactEncoder;

  static ::mlir::nv_tensor_ir::StatusOr<std::unique_ptr<TensorIRArtifact>>
  create(std::string kernelName, std::string funcName, Binary &&binary,
         ::mlir::nv_tensor_ir::SmTarget arch, KernelArgLayout argLayout,
         const std::optional<std::array<int32_t, 3>> &staticGrid);

  TensorIRArtifact(const TensorIRArtifact &) = default;
  TensorIRArtifact &operator=(const TensorIRArtifact &) = default;
  TensorIRArtifact(TensorIRArtifact &&) = default;
  TensorIRArtifact &operator=(TensorIRArtifact &&) = default;
  ~TensorIRArtifact() = default;

  bool operator==(const TensorIRArtifact &other) const {
    return kernelName == other.kernelName && funcName == other.funcName &&
           binary == other.binary && arch == other.arch &&
           argLayout == other.argLayout && staticGrid == other.staticGrid;
  }

private:
  TensorIRArtifact(std::string kernelName, std::string funcName,
                   Binary &&binary, ::mlir::nv_tensor_ir::SmTarget arch,
                   KernelArgLayout argLayout,
                   const std::optional<std::array<int32_t, 3>> &staticGrid);

  static ::mlir::nv_tensor_ir::Status
  validateFields(llvm::StringRef kernelName, llvm::StringRef funcName,
                 const Binary &binary, const KernelArgLayout &argLayout,
                 const std::optional<std::array<int32_t, 3>> &staticGrid);

  std::string kernelName;
  std::string funcName;
  Binary binary;
  ::mlir::nv_tensor_ir::SmTarget arch;
  KernelArgLayout argLayout;
  /// Absent selects the runtime-computed grid.
  std::optional<std::array<int32_t, 3>> staticGrid;
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_ARTIFACT_TENSORIRARTIFACT_H
