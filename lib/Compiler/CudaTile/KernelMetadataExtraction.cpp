// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Builds a KernelMetadata from a TensorIR GraphOp and CUDA Tile ABI options.
//===----------------------------------------------------------------------===//

#include "tensor_ir/Compiler/CudaTile/KernelMetadataExtraction.h"

#include "tensor_ir/Analysis/TileAnalyzer.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Options/Options.h"
#include "tensor_ir/Utils/Utils.h"

#include <optional>

namespace mlir::nv_tensor_ir::compiler::cuda_tile {

using ::tensor_ir::rt::ArgDesc;
using ::tensor_ir::rt::ElementType;
using ::tensor_ir::rt::KernelMetadata;
using ::tensor_ir::rt::KernelMetadataBuilder;
using ::tensor_ir::rt::ScalarArgDesc;
using ::tensor_ir::rt::TensorArgDesc;

/// Maps an MLIR type to the runtime ElementType enum, or nullopt if it
/// doesn't match one of the recognized dtypes.
static std::optional<ElementType> toElementType(mlir::Type type) {
  if (type.isSignlessInteger(1)) {
    return ElementType::Bool;
  }
  if (type.isSignedInteger(8)) {
    return ElementType::SI8;
  }
  if (type.isSignedInteger(16)) {
    return ElementType::SI16;
  }
  if (type.isSignedInteger(32)) {
    return ElementType::SI32;
  }
  if (type.isSignedInteger(64)) {
    return ElementType::SI64;
  }
  if (type.isUnsignedInteger(8)) {
    return ElementType::UI8;
  }
  if (type.isUnsignedInteger(16)) {
    return ElementType::UI16;
  }
  if (type.isUnsignedInteger(32)) {
    return ElementType::UI32;
  }
  if (type.isUnsignedInteger(64)) {
    return ElementType::UI64;
  }
  if (type.isF16()) {
    return ElementType::F16;
  }
  if (type.isBF16()) {
    return ElementType::BF16;
  }
  if (type.isF32()) {
    return ElementType::F32;
  }
  if (type.isF64()) {
    return ElementType::F64;
  }
  if (type.isF8E4M3FN()) {
    return ElementType::F8E4M3FN;
  }
  if (type.isF8E5M2()) {
    return ElementType::F8E5M2;
  }
  // Only add a dtype here once it's been verified to compile and run
  // through TensorIR's own tests -- an unrecognized dtype failing to
  // compile is preferable to a dtype mismatch surfacing silently later at
  // runtime, with no breadcrumb pointing back to this point.
  return std::nullopt;
}

/// Resolves an MLIR type to an ElementType, failing with an MLIR
/// diagnostic (anchored at loc) if the type isn't one of the recognized
/// runtime dtypes.
static mlir::FailureOr<ElementType> resolveElementType(mlir::Location loc,
                                                       mlir::Type type) {
  std::optional<ElementType> elemType = toElementType(type);
  if (!elemType) {
    return mlir::emitError(loc)
           << "kernel argument element type " << type
           << " is not a recognized TensorIR runtime dtype";
  }
  return *elemType;
}

mlir::FailureOr<KernelMetadata>
extractKernelMetadata(mlir::nv_tensor_ir::GraphOp graphOp,
                      const mlir::nv_tensor_ir::CodegenOptions &codegen,
                      const mlir::nv_tensor_ir::CudaTileOptions &cudaTile) {
  KernelMetadataBuilder layout;
  // Runtime launch metadata must use the same persistence contract as lowering
  // for every code-generation strategy.
  layout.persistence = cudaTile.persistenceMode;
  layout.smCount = cudaTile.smCount;
  layout.occupancy = cudaTile.occupancy;
  auto funcType = graphOp.getFunctionType();
  layout.numInputs = funcType.getNumInputs();

  // Copy tile sizes from conversion options.
  layout.tileSizes.assign(codegen.tileSize.begin(), codegen.tileSize.end());

  // Retrieve stride information from the tensor descriptors.
  using TensorDescriptor = mlir::nv_tensor_ir::TensorDescriptor;

  auto argInfos = mlir::nv_tensor_ir::getTensorDescriptors(
      graphOp.getArgumentTypes(), graphOp.getAllArgAttrs());
  assert(succeeded(argInfos) && "failed to get tensor info for input tensors");

  auto resInfos = mlir::nv_tensor_ir::getTensorDescriptors(
      graphOp.getResultTypes(), graphOp.getAllResultAttrs());
  assert(succeeded(resInfos) && "failed to get tensor info for output tensors");

  // Resolve compiler types into validated runtime descriptors.
  mlir::Location loc = graphOp.getLoc();
  auto buildDesc =
      [loc](mlir::RankedTensorType tensorTy,
            const TensorDescriptor &info) -> mlir::FailureOr<ArgDesc> {
    mlir::FailureOr<ElementType> elementType =
        resolveElementType(loc, tensorTy.getElementType());
    if (failed(elementType)) {
      return mlir::failure();
    }

    auto shape = tensorTy.getShape();
    llvm::SmallVector<int64_t> staticShape(shape.begin(), shape.end());
    std::optional<llvm::SmallVector<int64_t>> staticStrides;
    if (!info.strides.empty()) {
      staticStrides.emplace();
      for (const auto &stride : info.strides) {
        staticStrides->push_back(stride.staticValue);
      }
    }
    auto desc = TensorArgDesc::create(
        *elementType, staticShape,
        staticStrides ? std::optional<llvm::ArrayRef<int64_t>>(*staticStrides)
                      : std::nullopt);
    if (!desc.ok()) {
      return mlir::emitError(loc) << desc.status().message();
    }
    return ArgDesc(std::move(*desc));
  };

  // Helper: build a scalar operand descriptor from a non-tensor MLIR type.
  // The kernel signature for a scalar graph argument is a single by-value
  // kernel parameter, so we record the element type so the runtime
  // arg-packer can memcpy the value (instead of packing a pointer).
  auto buildScalarDesc = [loc](mlir::Type type) -> mlir::FailureOr<ArgDesc> {
    mlir::FailureOr<ElementType> elementType = resolveElementType(loc, type);
    if (failed(elementType)) {
      return mlir::failure();
    }
    auto desc = ScalarArgDesc::create(*elementType);
    if (!desc.ok()) {
      return mlir::emitError(loc) << desc.status().message();
    }
    return ArgDesc(std::move(*desc));
  };

  // Process input and output operands (same logic, different arrays).
  // Track how many valid input operands we add to args (tensors and
  // scalars both count).
  int32_t numValidInputTensors = 0;
  auto processTensors = [&](mlir::TypeRange types,
                            llvm::ArrayRef<TensorDescriptor> infos,
                            bool isInput) -> mlir::LogicalResult {
    for (auto [idx, type] : llvm::enumerate(types)) {
      auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(type);
      if (!tensorTy) {
        // Scalar (non-tensor) operand. The MLIR signature converter maps
        // it to a single by-value kernel parameter; record its size so
        // the runtime can pack it by value.
        if (!type.isIntOrFloat()) {
          continue; // Unsupported scalar kind, skip.
        }
        mlir::FailureOr<ArgDesc> desc = buildScalarDesc(type);
        if (failed(desc)) {
          return mlir::failure();
        }
        layout.args.push_back(*desc);
        if (isInput) {
          ++numValidInputTensors;
        }
        continue;
      }
      mlir::FailureOr<ArgDesc> desc = buildDesc(tensorTy, infos[idx]);
      if (failed(desc)) {
        return mlir::failure();
      }
      layout.args.push_back(*desc);
      if (isInput) {
        ++numValidInputTensors;
      }
    }
    return mlir::success();
  };

  if (failed(processTensors(funcType.getInputs(), *argInfos,
                            /*isInput=*/true)) ||
      failed(processTensors(funcType.getResults(), *resInfos,
                            /*isInput=*/false))) {
    return mlir::failure();
  }

  // Update numInputs to reflect the actual count of valid input tensors in
  // args.
  layout.numInputs = numValidInputTensors;

  // Determine which tensor's runtime shape drives grid computation.
  //
  // This uses the shared useOutputTensorShapeForGrid() function to ensure
  // alignment with TileAnalyzer::calculateGridSizeForGraph. Both compile-time
  // and runtime must use the same logic to select which tensor's shape drives
  // grid computation.
  auto findFirstValidTensorIdx = [&](mlir::TypeRange types,
                                     bool isOutput) -> int32_t {
    int32_t tensorDescIdx = isOutput ? layout.numInputs : 0;
    for (auto type : types) {
      auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(type);
      if (!tensorTy) {
        // Scalar operands do occupy a slot in args, so advance the
        // running index. Unsupported non-tensor types are not emitted and
        // therefore do not consume a slot.
        if (type.isIntOrFloat()) {
          ++tensorDescIdx;
        }
        continue;
      }
      if (tensorTy.getRank() == 0) {
        // Rank-0 tensors occupy an ABI slot but cannot drive grid shape.
        ++tensorDescIdx;
        continue;
      }
      return tensorDescIdx;
    }
    return isOutput ? layout.numInputs : 0;
  };

  bool useOutputTensor =
      mlir::nv_tensor_ir::useOutputTensorShapeForGrid(graphOp);

  if (useOutputTensor) {
    layout.gridShapeTensorIdx =
        findFirstValidTensorIdx(funcType.getResults(), /*isOutput=*/true);
  } else {
    layout.gridShapeTensorIdx =
        findFirstValidTensorIdx(funcType.getInputs(), /*isOutput=*/false);
  }

  auto validatedLayout = KernelMetadata::create(std::move(layout));
  if (!validatedLayout.ok()) {
    return mlir::emitError(graphOp.getLoc())
           << validatedLayout.status().message();
  }
  return std::move(*validatedLayout);
}

} // namespace mlir::nv_tensor_ir::compiler::cuda_tile
