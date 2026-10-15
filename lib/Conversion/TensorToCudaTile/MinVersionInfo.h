// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_CONVERSION_TENSORTOCUDATILE_MINVERSIONINFO_H
#define TENSOR_IR_CONVERSION_TENSORTOCUDATILE_MINVERSIONINFO_H

#include "mlir/Support/LogicalResult.h"

#include "cuda_tile/Bytecode/Common/Version.h"

namespace mlir {
class ModuleOp;
class Float8E8M0FNUType;
class Float4E2M1FNType;
class Float8E5M3FNUType;
} // namespace mlir

namespace mlir::cuda_tile {
class Atan2Op;
class FPowFOp;
class FPowIOp;
} // namespace mlir::cuda_tile

namespace mlir::nv_tensor_ir {

/// Check the versioned attributes and element types emitted by TensorToCudaTile.
/// Keep this explicit until CUDA Tile's minimum-version analyzer is reliable.
LogicalResult validateBytecodeVersion(ModuleOp module,
                                      cuda_tile::BytecodeVersion target);

/// Non-default rounding attribute on cuda_tile.exp.
struct ExpApproxRounding {};

/// Returns the introduction version of a CUDA Tile operation, attribute, type, or feature.
/// Keep entries in sync with CUDA Tile's TableGen version metadata. Add explicit
/// specializations as lowering needs them; unlisted classes have no default.
/// Individual instances may require newer versions because of their operands,
/// types, or attributes; introduction versions alone do not validate an instance.
template <typename OpOrAttr>
cuda_tile::BytecodeVersion getMinCudaTileBytecodeVersion() = delete;

template <>
inline cuda_tile::BytecodeVersion
getMinCudaTileBytecodeVersion<cuda_tile::Atan2Op>() {
  return *cuda_tile::BytecodeVersion::fromVersion(13, 2);
}

template <>
inline cuda_tile::BytecodeVersion
getMinCudaTileBytecodeVersion<cuda_tile::FPowFOp>() {
  return *cuda_tile::BytecodeVersion::fromVersion(13, 1);
}

template <>
inline cuda_tile::BytecodeVersion
getMinCudaTileBytecodeVersion<cuda_tile::FPowIOp>() {
  return *cuda_tile::BytecodeVersion::fromVersion(13, 4);
}

template <>
inline cuda_tile::BytecodeVersion
getMinCudaTileBytecodeVersion<ExpApproxRounding>() {
  return *cuda_tile::BytecodeVersion::fromVersion(13, 3);
}

template <>
inline cuda_tile::BytecodeVersion
getMinCudaTileBytecodeVersion<Float8E8M0FNUType>() {
  return *cuda_tile::BytecodeVersion::fromVersion(13, 2);
}

template <>
inline cuda_tile::BytecodeVersion
getMinCudaTileBytecodeVersion<Float4E2M1FNType>() {
  return *cuda_tile::BytecodeVersion::fromVersion(13, 3);
}

template <>
inline cuda_tile::BytecodeVersion
getMinCudaTileBytecodeVersion<Float8E5M3FNUType>() {
  return *cuda_tile::BytecodeVersion::fromVersion(13, 4);
}

/// Checks the class's introduction version, not instance-specific requirements.
template <typename OpOrAttr>
bool isSupported(cuda_tile::BytecodeVersion targetVersion) {
  return targetVersion >= getMinCudaTileBytecodeVersion<OpOrAttr>();
}

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_CONVERSION_TENSORTOCUDATILE_MINVERSIONINFO_H
