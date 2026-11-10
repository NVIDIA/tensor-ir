// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Runtime/CudaTile/RuntimeOperandAccessor.h"
#include "tensor_ir/Runtime/KernelLaunchHelpers.h"

#include "gtest/gtest.h"
#include <cstring>

using tensor_ir::rt::Any;
using tensor_ir::rt::ArgumentPacker;
using tensor_ir::rt::ElementType;
using tensor_ir::rt::KernelMetadata;
using tensor_ir::rt::KernelMetadataBuilder;
using tensor_ir::rt::PackedArgs;
using tensor_ir::rt::RuntimeOperandAccessor;
using tensor_ir::rt::ScalarArgDesc;
using tensor_ir::rt::TensorArgDesc;
using tensor_ir::rt::TensorView;

namespace {

TEST(RuntimeOperandAccessor, RecognizesScalarTagAndNotOthers) {
  float f32Value = 1.5f;
  Any scalarArg = Any::makeScalarAny(ElementType::F32, &f32Value);
  Any tensorArg = Any(TensorView{nullptr, 0, nullptr, nullptr});
  Any ptrArg = Any(static_cast<void *>(&scalarArg));
  Any args[] = {scalarArg, tensorArg, ptrArg};
  PackedArgs packed(args, 3);
  RuntimeOperandAccessor acc{packed};

  EXPECT_TRUE(acc.isScalar(0));
  EXPECT_FALSE(acc.isScalar(1));
  EXPECT_FALSE(acc.isScalar(2));
}

TEST(RuntimeOperandAccessor, ScalarSizeInBytesMatchesElementType) {
  uint8_t u8Value = 0x12;
  uint16_t u16Value = 0x1234;
  float f32Value = 1.0f;
  double f64Value = 1.0;
  Any args[] = {
      Any::makeScalarAny(ElementType::UI8, &u8Value),
      Any::makeScalarAny(ElementType::UI16, &u16Value),
      Any::makeScalarAny(ElementType::F32, &f32Value),
      Any::makeScalarAny(ElementType::F64, &f64Value),
  };
  PackedArgs packed(args, 4);
  RuntimeOperandAccessor acc{packed};

  EXPECT_EQ(acc.scalarSizeInBytes(0), 1);
  EXPECT_EQ(acc.scalarSizeInBytes(1), 2);
  EXPECT_EQ(acc.scalarSizeInBytes(2), 4);
  EXPECT_EQ(acc.scalarSizeInBytes(3), 8);
}

TEST(RuntimeOperandAccessor, NonScalarScalarSizeInBytesIsZero) {
  Any args[] = {Any(TensorView{nullptr, 0, nullptr, nullptr})};
  PackedArgs packed(args, 1);
  RuntimeOperandAccessor acc{packed};

  EXPECT_FALSE(acc.isScalar(0));
  EXPECT_EQ(acc.scalarSizeInBytes(0), 0);
}

/// End-to-end: ArgumentPacker (the strategy MemBound's static-shape kernels
/// use, with a layout that has no dynamic sizes/strides to pack) must place
/// the scalar's true bit pattern in the packed slot, not an 8-byte-widened
/// copy of unrelated bits.
TEST(RuntimeOperandAccessor, ArgumentPackerPacksTrueScalarWidth) {
  float f32Value = -2.5f;
  Any args[] = {Any::makeScalarAny(ElementType::F32, &f32Value)};
  PackedArgs packed(args, 1);
  RuntimeOperandAccessor acc{packed};

  KernelMetadataBuilder builder;
  builder.args = std::vector<tensor_ir::rt::ArgDesc>{
      ScalarArgDesc::create(ElementType::F32).value()};
  ArgumentPacker<RuntimeOperandAccessor> packer(
      KernelMetadata::create(std::move(builder)).value());
  int64_t outArgs[1] = {0};
  int32_t packedCount = packer.packArgs(acc, outArgs);

  ASSERT_EQ(packedCount, 1);
  float repacked;
  std::memcpy(&repacked, &outArgs[0], sizeof(float));
  EXPECT_EQ(repacked, f32Value);
  // High-order bytes must be zeroed, not left as whatever garbage happened
  // to be in the Any's 8-byte bit pattern above the true 4-byte value.
  EXPECT_EQ(outArgs[0] >> 32, 0);
}

TEST(RuntimeOperandAccessor, PacksDynamicTensorSizesAndStrides) {
  constexpr int64_t dynamic = TensorArgDesc::kDynamic;
  int64_t compiledStrides[] = {dynamic, 1};
  KernelMetadataBuilder builder;
  builder.args.push_back(
      TensorArgDesc::create(ElementType::F32, {dynamic, 3}, compiledStrides)
          .value());
  auto metadata = KernelMetadata::create(std::move(builder));
  ASSERT_TRUE(metadata.ok());
  ArgumentPacker<RuntimeOperandAccessor> packer(std::move(*metadata));
  float storage[10] = {};
  int64_t shape[] = {2, 3}, strides[] = {7, 1};
  Any value(TensorView{storage, 2, shape, strides});
  PackedArgs args(&value, 1);
  int64_t slots[3] = {};
  ASSERT_EQ(packer.packArgs(RuntimeOperandAccessor{args}, slots), 3);
  // Only dynamic sizes and strides follow the pointer in the device signature.
  EXPECT_EQ(
      llvm::ArrayRef<int64_t>(slots),
      (llvm::ArrayRef<int64_t>{reinterpret_cast<intptr_t>(storage), 2, 7}));
}

TEST(RuntimeOperandAccessor, PacksEmptyDynamicTensorWithoutData) {
  KernelMetadataBuilder builder;
  builder.args.push_back(
      TensorArgDesc::create(ElementType::F32, {TensorArgDesc::kDynamic})
          .value());
  auto metadata = KernelMetadata::create(std::move(builder));
  ASSERT_TRUE(metadata.ok());
  ArgumentPacker<RuntimeOperandAccessor> packer(std::move(*metadata));
  int64_t extent = 0;
  Any value(TensorView{nullptr, 1, &extent, nullptr});
  PackedArgs args(&value, 1);
  int64_t slots[] = {-1, -1};
  ASSERT_EQ(packer.packArgs(RuntimeOperandAccessor{args}, slots), 2);
  EXPECT_EQ(llvm::ArrayRef<int64_t>(slots), (llvm::ArrayRef<int64_t>{0, 0}));
}

} // namespace
