// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Transform/HostLoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

/// Convert an index, i32, or i64 value to the i32 width of the uint32_t launch
/// config ABI fields. Callback lowering checks that the value fits uint32_t
/// before invoking this helper.
static Value castToI32(OpBuilder &b, Location loc, Value value) {
  if (value.getType().isIndex()) {
    value = arith::IndexCastOp::create(b, loc, b.getI64Type(), value);
  }
  assert(isa<IntegerType>(value.getType()) &&
         "expected an index or integer launch value");
  if (value.getType().getIntOrFloatBitWidth() == 32) {
    return value;
  }
  return LLVM::TruncOp::create(b, loc, b.getI32Type(), value);
}

/// Initialize the host LLVM type converter, mapping ptr.generic_space memrefs
/// to address space 0 (default host flat address space).
HostLLVMTypeConverter::HostLLVMTypeConverter(
    MLIRContext *context, const DataLayoutAnalysis &dataLayoutAnalysis)
    : LLVMTypeConverter(context) {
  addTypeAttributeConversion(
      [](BaseMemRefType type,
         ptr::GenericSpaceAttr) -> TypeConverter::AttributeConversionResult {
        return IntegerAttr::get(IntegerType::get(type.getContext(), 64), 0);
      });
}

/// Returns `!llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>` matching
/// `TensorIRLaunchConfigV1`.
LLVM::LLVMStructType
nv_tensor_ir::LaunchConfigV1Builder::getType(OpBuilder &b) {
  auto i32ArrayType = LLVM::LLVMArrayType::get(b.getI32Type(), 3);
  return LLVM::LLVMStructType::getLiteral(
      b.getContext(), {
                          i32ArrayType,   // grid[3]
                          i32ArrayType,   // block[3]
                          b.getI32Type(), // dynamicSharedMemoryBytes
                      });
}

/// Construct a poison launch config struct.
LaunchConfigV1Builder
nv_tensor_ir::LaunchConfigV1Builder::poison(OpBuilder &b, Location loc) {
  Value value = LLVM::PoisonOp::create(b, loc, getType(b));
  return LaunchConfigV1Builder(value);
}

LaunchConfigV1Builder::LaunchConfigV1Builder(Value structValue)
    : StructBuilder(structValue) {
  OpBuilder b(structValue.getContext());
  assert(structValue.getType() == getType(b) &&
         "Expected launch config v1 type");
}

/// Populate grid dimensions (x, y, z) into the config struct.
void nv_tensor_ir::LaunchConfigV1Builder::setGrid(OpBuilder &b, Location loc,
                                                  Value x, Value y, Value z) {
  for (auto [idx, element] : llvm::enumerate(SmallVector<Value>{x, y, z})) {
    Value dim = castToI32(b, loc, element);
    value = LLVM::InsertValueOp::create(
        b, loc, value, dim,
        ArrayRef<int64_t>{kGridPosInDescriptor, static_cast<int64_t>(idx)});
  }
}

/// Populate block dimensions (x, y, z) into the config struct.
void nv_tensor_ir::LaunchConfigV1Builder::setBlock(OpBuilder &b, Location loc,
                                                   Value x, Value y, Value z) {
  for (auto [idx, element] : llvm::enumerate(SmallVector<Value>{x, y, z})) {
    Value dim = castToI32(b, loc, element);
    value = LLVM::InsertValueOp::create(
        b, loc, value, dim,
        ArrayRef<int64_t>{kBlockPosInDescriptor, static_cast<int64_t>(idx)});
  }
}

/// Populate dynamic shared memory in bytes into the config struct.
void nv_tensor_ir::LaunchConfigV1Builder::setDynamicSharedMemoryBytes(
    OpBuilder &b, Location loc, Value smemBytes) {
  Value bytes = castToI32(b, loc, smemBytes);
  value = LLVM::InsertValueOp::create(
      b, loc, value, bytes,
      ArrayRef<int64_t>{kDynamicSharedMemoryBytesPosInDescriptor});
}

/// Returns `!llvm.struct<(i32, i32, ptr, ptr)>` matching `TensorIRHostRuntimeV1`.
LLVM::LLVMStructType nv_tensor_ir::RuntimeV1Builder::getType(OpBuilder &b) {
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());
  return LLVM::LLVMStructType::getLiteral(
      b.getContext(), {b.getI32Type(), b.getI32Type(), ptrType, ptrType});
}

Value RuntimeV1Builder::getSizeInBytes(OpBuilder &b, Location loc) {
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  Value null = LLVM::ZeroOp::create(b, loc, ptr);
  Value end = LLVM::GEPOp::create(b, loc, ptr, getType(b), null,
                                  ArrayRef<LLVM::GEPArg>{1});
  return LLVM::PtrToIntOp::create(b, loc, b.getI32Type(), end);
}

static Value loadRuntimeField(OpBuilder &b, Location loc, Value runtime,
                              int32_t field) {
  auto type = RuntimeV1Builder::getType(b);
  Value ptr = LLVM::GEPOp::create(b, loc, runtime.getType(), type, runtime,
                                  ArrayRef<LLVM::GEPArg>{0, field});
  return LLVM::LoadOp::create(b, loc, type.getBody()[field], ptr);
}

Value RuntimeV1Builder::loadStructSize(OpBuilder &b, Location loc,
                                       Value runtime) {
  return loadRuntimeField(b, loc, runtime, kStructSizePosInDescriptor);
}

Value RuntimeV1Builder::loadAbiVersion(OpBuilder &b, Location loc,
                                       Value runtime) {
  return loadRuntimeField(b, loc, runtime, kAbiVersionPosInDescriptor);
}

Value RuntimeV1Builder::loadLaunchCallbackFn(OpBuilder &b, Location loc,
                                             Value runtime) {
  return loadRuntimeField(b, loc, runtime, kLaunchPosInDescriptor);
}

/// Returns indirect callback signature:
///   int32_t (*)(void *userData, uint32_t kernelOrdinal,
///               const TensorIRLaunchConfigV1 *config, void *const *kernelParams)
/// LLVM indirect call passes callee ptr as operand 0, followed by the 4 args.
LLVM::LLVMFunctionType
nv_tensor_ir::RuntimeV1Builder::getLaunchCallbackFnType(OpBuilder &b) {
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());
  return LLVM::LLVMFunctionType::get(
      b.getI32Type(), {ptrType, ptrType, b.getI32Type(), ptrType, ptrType});
}

/// Construct a poison runtime struct.
RuntimeV1Builder nv_tensor_ir::RuntimeV1Builder::poison(OpBuilder &b,
                                                        Location loc) {
  Value value = LLVM::PoisonOp::create(b, loc, getType(b));
  return RuntimeV1Builder(value);
}

/// Extract the `launch` function pointer from the runtime struct.
Value nv_tensor_ir::RuntimeV1Builder::getLaunchCallbackFn(OpBuilder &b,
                                                          Location loc) const {
  return LLVM::ExtractValueOp::create(
      b, loc, value, ArrayRef<int64_t>{kLaunchPosInDescriptor});
}

/// Extract the `userData` pointer from the runtime struct.
Value nv_tensor_ir::RuntimeV1Builder::getUserData(OpBuilder &b,
                                                  Location loc) const {
  return LLVM::ExtractValueOp::create(
      b, loc, value, ArrayRef<int64_t>{kUserDataPosInDescriptor});
}

RuntimeV1Builder::RuntimeV1Builder(Value structValue)
    : StructBuilder(structValue) {
  OpBuilder b(structValue.getContext());
  assert(structValue.getType() == getType(b) && "Expected runtime v1 type");
}
