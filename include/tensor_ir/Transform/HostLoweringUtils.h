// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_TRANSFORM_HOST_LOWERING_H
#define TENSOR_IR_TRANSFORM_HOST_LOWERING_H

#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir::nv_tensor_ir {

/// LLVM type converter for host lowering. Converts MLIR types (including
/// memrefs and unrealized casts) to LLVM dialect types, mapping generic-space
/// pointer attributes to address space 0 (default host flat address space).
class HostLLVMTypeConverter : public LLVMTypeConverter {
public:
  HostLLVMTypeConverter(MLIRContext *context,
                        const DataLayoutAnalysis &dataLayoutAnalysis);
};

/// Struct builder for the `TensorIRLaunchConfigV1` LLVM struct:
///   struct TensorIRLaunchConfigV1 {
///     uint32_t grid[3];
///     uint32_t block[3];
///     uint32_t dynamicSharedMemoryBytes;
///   };
class LaunchConfigV1Builder : public mlir::StructBuilder {
public:
  static constexpr int64_t kGridPosInDescriptor = 0;
  static constexpr int64_t kBlockPosInDescriptor = 1;
  static constexpr int64_t kDynamicSharedMemoryBytesPosInDescriptor = 2;

  /// Return the LLVM struct type: `!llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>`.
  static LLVM::LLVMStructType getType(OpBuilder &b);

  /// Construct an uninitialized (poison) launch config v1 struct value.
  static LaunchConfigV1Builder poison(OpBuilder &b, Location loc);

  /// Construct a launch config v1 struct from an existing struct value.
  LaunchConfigV1Builder(Value structValue);

  /// Set grid dimensions (x, y, z). Each index, i32, or i64 value is converted
  /// to i32 and must fit in uint32_t.
  void setGrid(OpBuilder &b, Location loc, Value x, Value y, Value z);

  /// Set block dimensions (x, y, z). Each index, i32, or i64 value is converted
  /// to i32 and must fit in uint32_t.
  void setBlock(OpBuilder &b, Location loc, Value x, Value y, Value z);

  /// Set dynamic shared memory in bytes. The index, i32, or i64 value is
  /// converted to i32 and must fit in uint32_t.
  void setDynamicSharedMemoryBytes(OpBuilder &b, Location loc, Value smemBytes);
};

/// Struct builder for extracting fields from the `TensorIRHostRuntimeV1` LLVM struct:
///   struct TensorIRHostRuntimeV1 {
///     uint32_t structSize;
///     uint32_t abiVersion;
///     void *userData;
///     TensorIRLaunchFnV1 launch;
///   };
class RuntimeV1Builder : public mlir::StructBuilder {
public:
  static constexpr int64_t kStructSizePosInDescriptor = 0;
  static constexpr int64_t kAbiVersionPosInDescriptor = 1;
  static constexpr int64_t kUserDataPosInDescriptor = 2;
  static constexpr int64_t kLaunchPosInDescriptor = 3;

  /// Return the LLVM struct type: `!llvm.struct<(i32, i32, ptr, ptr)>`.
  static LLVM::LLVMStructType getType(OpBuilder &b);

  /// Compute the runtime struct size in bytes using its LLVM type.
  static Value getSizeInBytes(OpBuilder &b, Location loc);

  /// Load individual fields without reading the rest of an unvalidated runtime.
  static Value loadStructSize(OpBuilder &b, Location loc, Value runtime);
  static Value loadAbiVersion(OpBuilder &b, Location loc, Value runtime);
  static Value loadLaunchCallbackFn(OpBuilder &b, Location loc, Value runtime);

  /// Return the indirect callback function type: `i32 (ptr, ptr, i32, ptr, ptr)`.
  static LLVM::LLVMFunctionType getLaunchCallbackFnType(OpBuilder &b);

  /// Construct an uninitialized (poison) runtime v1 struct value.
  static RuntimeV1Builder poison(OpBuilder &b, Location loc);

  /// Extract the kernel launch callback function pointer (`launch`).
  Value getLaunchCallbackFn(OpBuilder &b, Location loc) const;

  /// Extract the opaque runtime user data pointer (`userData`).
  Value getUserData(OpBuilder &b, Location loc) const;

  /// Construct a runtime v1 struct wrapper from an existing struct value.
  RuntimeV1Builder(Value structValue);
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_TRANSFORM_HOST_LOWERING_H
