// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Kernel launch metadata — compiled arguments, grid, and scheduling
//
// The compiler produces this metadata and the runtime uses it to pack the flat
// kernel argument list and compute grid dimensions dynamically.
//===----------------------------------------------------------------------===//

#ifndef TENSOR_IR_ARTIFACT_KERNEL_METADATA_H_
#define TENSOR_IR_ARTIFACT_KERNEL_METADATA_H_

#include "tensor_ir/Options/OptionsEnums.h"
#include "tensor_ir/Support/Status.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

namespace tensor_ir::rt {

/// Element type of a tensor or scalar kernel argument.
enum class ElementType : uint8_t {
  Bool = 0,
  F16,
  BF16,
  F32,
  F64,
  F8E4M3FN,
  F8E5M2,
  SI8,
  SI16,
  SI32,
  SI64,
  UI8,
  UI16,
  UI32,
  UI64,
  // The largest defined value -- the enum is dense, so [Bool,
  // kMaxElementType] is exactly its valid range.
  kMaxElementType = UI64,
};

int32_t getElementByteWidth(ElementType type);

bool isValidElementType(ElementType type);

/// A by-value scalar kernel argument. All state is immutable, including through
/// assignment; construction rejects unknown element types.
struct ScalarArgDesc {
  static ::mlir::nv_tensor_ir::StatusOr<ScalarArgDesc>
  create(ElementType elementType);

  const ElementType elementType;

  int32_t totalArgs() const;

  bool operator==(const ScalarArgDesc &rhs) const;

private:
  explicit ScalarArgDesc(ElementType elementType);
};

/// A pointer argument with a shape and optional explicit strides. Factories
/// copy input storage; callers cannot mutate a descriptor through input aliases.
/// Const storage also leaves moved-from descriptors valid and unchanged.
struct TensorArgDesc {
  static constexpr int64_t kDynamic = std::numeric_limits<int64_t>::min();

  static ::mlir::nv_tensor_ir::StatusOr<TensorArgDesc>
  create(ElementType elementType, llvm::ArrayRef<int64_t> shape,
         std::optional<llvm::ArrayRef<int64_t>> strides = std::nullopt);

  const ElementType elementType;

  int32_t rank() const;
  llvm::ArrayRef<int64_t> shape() const;
  bool hasExplicitStrides() const;
  llvm::ArrayRef<int64_t> strides() const;
  int32_t numDynSizes() const;
  int32_t numDynStrides() const;
  int32_t totalArgs() const;
  bool operator==(const TensorArgDesc &other) const;

private:
  TensorArgDesc(ElementType elementType, llvm::ArrayRef<int64_t> shape,
                std::optional<llvm::ArrayRef<int64_t>> strides);

  const llvm::SmallVector<int64_t> shape_;
  const std::optional<llvm::SmallVector<int64_t>> strides_;
};

using ArgDesc = std::variant<ScalarArgDesc, TensorArgDesc>;

ElementType getElementType(const ArgDesc &arg);

int32_t getTotalArgs(const ArgDesc &arg);

/// Mutable inputs used while assembling kernel launch metadata. A builder may
/// be temporarily incomplete; KernelMetadata::create() validates all
/// cross-field invariants before producing complete metadata.
struct KernelMetadataBuilder {
  std::vector<ArgDesc> args;
  llvm::SmallVector<int32_t> tileSizes;
  int32_t numInputs = 0;
  int32_t gridShapeTensorIdx = 0;
  llvm::SmallVector<int64_t> gridShape;
  llvm::SmallVector<int32_t> gridShapeDimMapping;
  mlir::nv_tensor_ir::PersistenceMode persistence =
      mlir::nv_tensor_ir::PersistenceMode::None;
  int32_t smCount = 0;
  int32_t occupancy = 1;
};

/// Validated argument signature, grid configuration, and scheduling metadata.
class KernelMetadata final {
public:
  KernelMetadata(const KernelMetadata &) = default;
  KernelMetadata(KernelMetadata &&) = default;
  KernelMetadata &operator=(KernelMetadata &&) = delete;
  KernelMetadata &operator=(const KernelMetadata &) = delete;

  static ::mlir::nv_tensor_ir::StatusOr<KernelMetadata>
  create(KernelMetadataBuilder builder);

  const std::vector<ArgDesc> args;
  const llvm::SmallVector<int32_t> tileSizes;
  const int32_t numInputs;
  const int32_t totalKernelArgs;
  const int32_t gridShapeTensorIdx;
  const llvm::SmallVector<int64_t> gridShape;
  const llvm::SmallVector<int32_t> gridShapeDimMapping;
  const mlir::nv_tensor_ir::PersistenceMode persistence;
  const int32_t smCount;
  const int32_t occupancy;

  KernelMetadataBuilder toBuilder() const;

  int32_t outputTensorStartIdx() const;

  int32_t getArgOffset(int32_t tensorIdx) const;

  bool operator==(const KernelMetadata &other) const;

private:
  KernelMetadata(KernelMetadataBuilder builder, int32_t totalKernelArgs);
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_ARTIFACT_KERNEL_METADATA_H_
