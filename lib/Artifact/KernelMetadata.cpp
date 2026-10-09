// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Artifact/KernelMetadata.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <utility>

namespace tensor_ir::rt {

int32_t getElementByteWidth(ElementType type) {
  switch (type) {
  case ElementType::Bool:
  case ElementType::SI8:
  case ElementType::UI8:
  case ElementType::F8E4M3FN:
  case ElementType::F8E5M2:
    return 1;
  case ElementType::F16:
  case ElementType::BF16:
  case ElementType::SI16:
  case ElementType::UI16:
    return 2;
  case ElementType::F32:
  case ElementType::SI32:
  case ElementType::UI32:
    return 4;
  case ElementType::F64:
  case ElementType::SI64:
  case ElementType::UI64:
    return 8;
  }
  llvm_unreachable("unhandled ElementType");
}

bool isValidElementType(ElementType type) {
  return static_cast<uint8_t>(type) <=
         static_cast<uint8_t>(ElementType::kMaxElementType);
}

::mlir::nv_tensor_ir::StatusOr<ScalarArgDesc>
ScalarArgDesc::create(ElementType elementType) {
  if (!isValidElementType(elementType)) {
    return ::mlir::nv_tensor_ir::Status::InvalidArgument(
        "scalar argument has an invalid element type");
  }
  return ScalarArgDesc(elementType);
}

int32_t ScalarArgDesc::totalArgs() const { return 1; }

bool ScalarArgDesc::operator==(const ScalarArgDesc &rhs) const {
  return elementType == rhs.elementType;
}

ScalarArgDesc::ScalarArgDesc(ElementType elementType)
    : elementType(elementType) {}

::mlir::nv_tensor_ir::StatusOr<TensorArgDesc>
TensorArgDesc::create(ElementType elementType, llvm::ArrayRef<int64_t> shape,
                      std::optional<llvm::ArrayRef<int64_t>> strides) {
  using ::mlir::nv_tensor_ir::Status;
  if (!isValidElementType(elementType)) {
    return Status::InvalidArgument(
        "tensor argument has an invalid element type");
  }
  // A pointer plus dynamic sizes and strides must fit an int32 count.
  if (shape.size() > (std::numeric_limits<int32_t>::max() - 1) / 2) {
    return Status::InvalidArgument("tensor argument rank is too large");
  }
  if (strides && strides->size() != shape.size()) {
    return Status::InvalidArgument(
        "tensor argument strides length must match its rank");
  }
  auto validDim = [](int64_t dim) { return dim >= 0 || dim == kDynamic; };
  if (!llvm::all_of(shape, validDim) ||
      (strides && !llvm::all_of(*strides, validDim))) {
    return Status::InvalidArgument(
        "tensor argument shape and strides must be non-negative or kDynamic");
  }
  return TensorArgDesc(elementType, shape, strides);
}

int32_t TensorArgDesc::rank() const {
  return static_cast<int32_t>(shape_.size());
}

llvm::ArrayRef<int64_t> TensorArgDesc::shape() const { return shape_; }

bool TensorArgDesc::hasExplicitStrides() const { return strides_.has_value(); }

llvm::ArrayRef<int64_t> TensorArgDesc::strides() const {
  return strides_ ? llvm::ArrayRef<int64_t>(*strides_)
                  : llvm::ArrayRef<int64_t>();
}

int32_t TensorArgDesc::numDynSizes() const {
  return llvm::count(shape_, kDynamic);
}

int32_t TensorArgDesc::numDynStrides() const {
  return strides_ ? llvm::count(*strides_, kDynamic) : 0;
}

int32_t TensorArgDesc::totalArgs() const {
  return 1 + numDynSizes() + numDynStrides();
}

bool TensorArgDesc::operator==(const TensorArgDesc &other) const {
  return elementType == other.elementType && shape_ == other.shape_ &&
         strides_ == other.strides_;
}

TensorArgDesc::TensorArgDesc(ElementType elementType,
                             llvm::ArrayRef<int64_t> shape,
                             std::optional<llvm::ArrayRef<int64_t>> strides)
    : elementType(elementType), shape_(shape.begin(), shape.end()),
      strides_(strides ? std::optional<llvm::SmallVector<int64_t>>(
                             std::in_place, strides->begin(), strides->end())
                       : std::nullopt) {}

ElementType getElementType(const ArgDesc &arg) {
  return std::visit([](const auto &desc) { return desc.elementType; }, arg);
}

int32_t getTotalArgs(const ArgDesc &arg) {
  return std::visit([&](const auto &desc) { return desc.totalArgs(); }, arg);
}

::mlir::nv_tensor_ir::StatusOr<KernelMetadata>
KernelMetadata::create(KernelMetadataBuilder builder) {
  using ::mlir::nv_tensor_ir::PersistenceMode;
  using ::mlir::nv_tensor_ir::Status;
  if (builder.numInputs < 0 ||
      builder.numInputs > static_cast<int32_t>(builder.args.size())) {
    return Status::InvalidArgument("numInputs is out of bounds");
  }
  bool validPersistence = builder.persistence == PersistenceMode::None ||
                          builder.persistence == PersistenceMode::Static;
  if (!validPersistence) {
    return Status::InvalidArgument("persistence mode is invalid");
  }
  if (builder.smCount < 0) {
    return Status::InvalidArgument("smCount must not be negative");
  }
  if (builder.occupancy < 1) {
    return Status::InvalidArgument("occupancy must be at least 1");
  }
  if (builder.gridShape.size() != builder.gridShapeDimMapping.size()) {
    return Status::InvalidArgument(
        "gridShape and gridShapeDimMapping must have the same length");
  }
  if (!builder.gridShape.empty() &&
      builder.gridShape.size() != builder.tileSizes.size()) {
    return Status::InvalidArgument(
        "gridShape length must match tileSizes when gridShape is present");
  }
  if (builder.args.empty()) {
    if (builder.gridShapeTensorIdx != 0) {
      return Status::InvalidArgument(
          "gridShapeTensorIdx must be 0 when there are no arguments");
    }
    if (!builder.gridShape.empty()) {
      return Status::InvalidArgument(
          "gridShape must be empty when there are no arguments");
    }
  } else if (builder.gridShapeTensorIdx < 0 ||
             builder.gridShapeTensorIdx >=
                 static_cast<int32_t>(builder.args.size())) {
    return Status::InvalidArgument("gridShapeTensorIdx is out of bounds");
  }

  int64_t totalKernelArgs = 0;
  for (const ArgDesc &arg : builder.args) {
    totalKernelArgs += getTotalArgs(arg);
  }
  if (totalKernelArgs > std::numeric_limits<int32_t>::max()) {
    return Status::InvalidArgument("total kernel argument count is too large");
  }

  if (!builder.gridShape.empty()) {
    const auto *shapeDesc =
        std::get_if<TensorArgDesc>(&builder.args[builder.gridShapeTensorIdx]);
    for (auto [dimSize, tensorDim] :
         llvm::zip_equal(builder.gridShape, builder.gridShapeDimMapping)) {
      if (dimSize == TensorArgDesc::kDynamic) {
        if (!shapeDesc || tensorDim < 0 || tensorDim >= shapeDesc->rank() ||
            shapeDesc->shape()[tensorDim] != TensorArgDesc::kDynamic) {
          return Status::InvalidArgument(
              "gridShapeDimMapping entry does not refer to a dynamic "
              "dimension of the grid-shape tensor");
        }
      } else if (dimSize < 0 || tensorDim != -1) {
        return Status::InvalidArgument(
            "static gridShape entries must have a -1 dimension mapping");
      }
    }
  }

  return KernelMetadata(std::move(builder),
                        static_cast<int32_t>(totalKernelArgs));
}

KernelMetadataBuilder KernelMetadata::toBuilder() const {
  return KernelMetadataBuilder{
      args,      tileSizes,           numInputs,   gridShapeTensorIdx,
      gridShape, gridShapeDimMapping, persistence, smCount,
      occupancy};
}

int32_t KernelMetadata::outputTensorStartIdx() const { return numInputs; }

int32_t KernelMetadata::getArgOffset(int32_t tensorIdx) const {
  int32_t off = 0;
  for (int32_t i = 0; i < tensorIdx; ++i) {
    off += getTotalArgs(args[i]);
  }
  return off;
}

bool KernelMetadata::operator==(const KernelMetadata &other) const {
  return args == other.args && tileSizes == other.tileSizes &&
         numInputs == other.numInputs &&
         totalKernelArgs == other.totalKernelArgs &&
         gridShapeTensorIdx == other.gridShapeTensorIdx &&
         gridShape == other.gridShape &&
         gridShapeDimMapping == other.gridShapeDimMapping &&
         persistence == other.persistence && smCount == other.smCount &&
         occupancy == other.occupancy;
}

KernelMetadata::KernelMetadata(KernelMetadataBuilder builder,
                               int32_t totalKernelArgs)
    : args(std::move(builder.args)), tileSizes(std::move(builder.tileSizes)),
      numInputs(builder.numInputs), totalKernelArgs(totalKernelArgs),
      gridShapeTensorIdx(builder.gridShapeTensorIdx),
      gridShape(std::move(builder.gridShape)),
      gridShapeDimMapping(std::move(builder.gridShapeDimMapping)),
      persistence(builder.persistence), smCount(builder.smCount),
      occupancy(builder.occupancy) {}

} // namespace tensor_ir::rt
