// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Artifact/TensorIRArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>

using mlir::nv_tensor_ir::SmTarget;
using mlir::nv_tensor_ir::Status;
using mlir::nv_tensor_ir::StatusOr;

namespace tensor_ir::rt {

Status TensorIRArtifact::validateFields(
    llvm::StringRef kernelName, llvm::StringRef funcName, const Binary &binary,
    const KernelArgLayout &argLayout,
    const std::optional<std::array<int32_t, 3>> &staticGrid) {
  bool binaryEmpty =
      std::visit([](const auto &b) { return b.bytes.empty(); }, binary);
  if (binaryEmpty) {
    return Status::InvalidArgument("TensorIRArtifact: binary is empty");
  }
  if (kernelName.empty() || funcName.empty()) {
    return Status::InvalidArgument(
        "TensorIRArtifact: kernelName/funcName must not be empty");
  }

  if (argLayout.numInputs < 0 ||
      argLayout.numInputs >
          static_cast<int32_t>(argLayout.tensorDescs.size())) {
    return Status::InvalidArgument("TensorIRArtifact: numInputs is out of "
                                   "bounds");
  }

  int32_t summedArgs = 0;
  for (const auto &d : argLayout.tensorDescs) {
    if (d.isScalar) {
      // Scalars are packed by memcpy'ing elementInfo.byteWidth() bytes into
      // an int64 slot.
      if (d.elementInfo.byteWidth() <= 0 ||
          static_cast<size_t>(d.elementInfo.byteWidth()) > sizeof(int64_t)) {
        return Status::InvalidArgument(
            "TensorIRArtifact: scalar argument has an invalid byte width");
      }
      if (d.rank != 0 || !d.staticShape.empty() || !d.staticStrides.empty() ||
          d.numDynSizes != 0 || d.numDynStrides != 0 || d.hasExplicitStrides) {
        return Status::InvalidArgument(
            "TensorIRArtifact: scalar argument has non-trivial tensor-only "
            "fields set");
      }
    } else {
      if (d.rank < 0) {
        return Status::InvalidArgument(
            "TensorIRArtifact: tensor argument has a negative rank");
      }
      if (d.staticShape.size() != static_cast<size_t>(d.rank)) {
        return Status::InvalidArgument(
            "TensorIRArtifact: tensor argument shape length does not match "
            "its rank");
      }
      if (d.hasExplicitStrides) {
        if (d.staticStrides.size() != static_cast<size_t>(d.rank)) {
          return Status::InvalidArgument(
              "TensorIRArtifact: tensor argument strides length does not "
              "match its rank");
        }
      } else if (!d.staticStrides.empty()) {
        return Status::InvalidArgument(
            "TensorIRArtifact: tensor argument has strides set without "
            "hasExplicitStrides");
      }
      auto isValidDim = [](int64_t v) {
        return v >= 0 || v == TensorArgDesc::kDynamic;
      };
      if (!llvm::all_of(d.staticShape, isValidDim) ||
          (d.hasExplicitStrides &&
           !llvm::all_of(d.staticStrides, isValidDim))) {
        return Status::InvalidArgument(
            "TensorIRArtifact: tensor argument has a shape or stride entry "
            "that is neither non-negative nor kDynamic");
      }
      int32_t expectedDynSizes =
          argLayout.uniformSignature ? d.rank : countDynamicDims(d.staticShape);
      int32_t expectedDynStrides =
          d.hasExplicitStrides
              ? (argLayout.uniformSignature ? d.rank
                                            : countDynamicDims(d.staticStrides))
              : 0;
      if (d.numDynSizes != expectedDynSizes ||
          d.numDynStrides != expectedDynStrides) {
        return Status::InvalidArgument(
            "TensorIRArtifact: tensor argument numDynSizes/numDynStrides "
            "does not match its shape/strides");
      }
    }
    summedArgs += d.totalArgs();
  }

  if (argLayout.smCount < 0) {
    return Status::InvalidArgument(
        "TensorIRArtifact: smCount must not be negative");
  }
  if (argLayout.occupancy < 1) {
    return Status::InvalidArgument(
        "TensorIRArtifact: occupancy must be at least 1");
  }

  if (staticGrid &&
      llvm::any_of(*staticGrid, [](int32_t dim) { return dim < 1; })) {
    return Status::InvalidArgument(
        "TensorIRArtifact: staticGrid dimensions must be at least 1");
  }

  if (argLayout.gridShape.size() != argLayout.gridShapeDimMapping.size()) {
    return Status::InvalidArgument(
        "TensorIRArtifact: gridShape and gridShapeDimMapping must have the "
        "same length");
  }
  if (!argLayout.gridShape.empty() &&
      argLayout.gridShape.size() != argLayout.tileSizes.size()) {
    return Status::InvalidArgument(
        "TensorIRArtifact: gridShape length must match tileSizes when "
        "gridShape is present");
  }

  if (argLayout.tensorDescs.empty()) {
    if (argLayout.gridShapeTensorIdx != 0) {
      return Status::InvalidArgument(
          "TensorIRArtifact: gridShapeTensorIdx must be 0 when there are no "
          "tensor arguments");
    }
    if (!argLayout.gridShape.empty()) {
      return Status::InvalidArgument(
          "TensorIRArtifact: gridShape must be empty when there are no "
          "tensor arguments");
    }
  } else if (argLayout.gridShapeTensorIdx < 0 ||
             argLayout.gridShapeTensorIdx >=
                 static_cast<int32_t>(argLayout.tensorDescs.size())) {
    return Status::InvalidArgument(
        "TensorIRArtifact: gridShapeTensorIdx is out of bounds");
  }
  if (summedArgs != argLayout.totalKernelArgs) {
    return Status::InvalidArgument(
        "TensorIRArtifact: totalKernelArgs does not match the sum of "
        "per-argument arg counts");
  }

  if (!argLayout.gridShape.empty()) {
    const TensorArgDesc &shapeDesc =
        argLayout.tensorDescs[argLayout.gridShapeTensorIdx];
    for (auto [dimSize, tensorDim] :
         llvm::zip_equal(argLayout.gridShape, argLayout.gridShapeDimMapping)) {
      if (dimSize == TensorArgDesc::kDynamic) {
        if (tensorDim < 0 || tensorDim >= shapeDesc.rank ||
            shapeDesc.staticShape[tensorDim] != TensorArgDesc::kDynamic) {
          return Status::InvalidArgument(
              "TensorIRArtifact: gridShapeDimMapping entry does not refer "
              "to a dynamic dimension of the grid-shape tensor");
        }
      } else if (dimSize < 0 || tensorDim != -1) {
        return Status::InvalidArgument(
            "TensorIRArtifact: static gridShape entries must have a -1 "
            "dimension mapping");
      }
    }
  }

  return Status::Ok();
}

TensorIRArtifact::TensorIRArtifact(
    std::string kernelName, std::string funcName, Binary &&binary,
    SmTarget arch, KernelArgLayout argLayout,
    const std::optional<std::array<int32_t, 3>> &staticGrid)
    : kernelName(std::move(kernelName)), funcName(std::move(funcName)),
      binary(std::move(binary)), arch(arch), argLayout(std::move(argLayout)),
      staticGrid(staticGrid) {}

StatusOr<std::unique_ptr<TensorIRArtifact>> TensorIRArtifact::create(
    std::string kernelName, std::string funcName, Binary &&binary,
    SmTarget arch, KernelArgLayout argLayout,
    const std::optional<std::array<int32_t, 3>> &staticGrid) {
  if (Status validation =
          validateFields(kernelName, funcName, binary, argLayout, staticGrid);
      !validation.ok()) {
    return validation;
  }
  return std::unique_ptr<TensorIRArtifact>(new TensorIRArtifact(
      std::move(kernelName), std::move(funcName), std::move(binary), arch,
      std::move(argLayout), staticGrid));
}

} // namespace tensor_ir::rt
