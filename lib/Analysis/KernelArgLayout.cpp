// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Builds a KernelArgLayout from a TensorIR GraphOp and CUDA Tile ABI options.
//===----------------------------------------------------------------------===//

#include "tensor_ir/Compiler/CudaTile/KernelArgLayout.h"

#include "tensor_ir/Analysis/TileAnalyzer.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Utils/Utils.h"

#include <optional>

namespace tensor_ir {

using rt::countDynamicDims;
using rt::ElementTypeInfo;
using rt::KernelArgLayout;
using rt::TensorArgDesc;

/// Maps an MLIR type to the runtime ElementType enum, or nullopt if it
/// doesn't match one of the recognized dtypes.
static std::optional<rt::ElementType> toElementType(mlir::Type type) {
  if (type.isSignlessInteger(1)) {
    return rt::ElementType::Bool;
  }
  if (type.isSignedInteger(8)) {
    return rt::ElementType::SI8;
  }
  if (type.isSignedInteger(16)) {
    return rt::ElementType::SI16;
  }
  if (type.isSignedInteger(32)) {
    return rt::ElementType::SI32;
  }
  if (type.isSignedInteger(64)) {
    return rt::ElementType::SI64;
  }
  if (type.isUnsignedInteger(8)) {
    return rt::ElementType::UI8;
  }
  if (type.isUnsignedInteger(16)) {
    return rt::ElementType::UI16;
  }
  if (type.isUnsignedInteger(32)) {
    return rt::ElementType::UI32;
  }
  if (type.isUnsignedInteger(64)) {
    return rt::ElementType::UI64;
  }
  if (type.isF16()) {
    return rt::ElementType::F16;
  }
  if (type.isBF16()) {
    return rt::ElementType::BF16;
  }
  if (type.isF32()) {
    return rt::ElementType::F32;
  }
  if (type.isF64()) {
    return rt::ElementType::F64;
  }
  if (type.isF8E4M3FN()) {
    return rt::ElementType::F8E4M3FN;
  }
  if (type.isF8E5M2()) {
    return rt::ElementType::F8E5M2;
  }
  // Only add a dtype here once it's been verified to compile and run
  // through TensorIR's own tests -- an unrecognized dtype failing to
  // compile is preferable to a dtype mismatch surfacing silently later at
  // runtime, with no breadcrumb pointing back to this point.
  return std::nullopt;
}

/// Resolves an MLIR type to an ElementTypeInfo, failing with an MLIR
/// diagnostic (anchored at loc) if the type isn't one of the recognized
/// runtime dtypes.
static mlir::FailureOr<ElementTypeInfo>
resolveElementTypeInfo(mlir::Location loc, mlir::Type type) {
  std::optional<rt::ElementType> elemType = toElementType(type);
  if (!elemType) {
    return mlir::emitError(loc)
           << "kernel argument element type " << type
           << " is not a recognized TensorIR runtime dtype";
  }
  return ElementTypeInfo(*elemType);
}

mlir::FailureOr<KernelArgLayout> extractKernelArgLayout(
    mlir::nv_tensor_ir::GraphOp graphOp,
    const mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions &options) {
  KernelArgLayout layout;
  layout.uniformSignature = options.uniformSignature;
  // Runtime launch metadata must use the same persistence contract as lowering
  // for every code-generation strategy.
  layout.persistence = options.persistence;
  layout.smCount = options.smCount;
  layout.occupancy = options.occupancy;
  auto funcType = graphOp.getFunctionType();
  layout.numInputs = funcType.getNumInputs();

  // Copy tile sizes from conversion options.
  layout.tileSizes.assign(options.tileSize.begin(), options.tileSize.end());

  // Retrieve stride information from the tensor descriptors.
  using TensorDescriptor = mlir::nv_tensor_ir::TensorDescriptor;

  auto argInfos = mlir::nv_tensor_ir::getTensorDescriptors(
      graphOp.getArgumentTypes(), graphOp.getAllArgAttrs());
  assert(succeeded(argInfos) && "failed to get tensor info for input tensors");

  auto resInfos = mlir::nv_tensor_ir::getTensorDescriptors(
      graphOp.getResultTypes(), graphOp.getAllResultAttrs());
  assert(succeeded(resInfos) && "failed to get tensor info for output tensors");

  // Helper: create a TensorArgDesc from a TensorType and its stride info.
  //
  // When uniformSignature is true, all sizes (and all strides when explicit)
  // are emitted as kernel arguments, even for statically-known dimensions.
  // When false, only dynamic dims produce kernel arguments (original behavior).
  bool uniform = layout.uniformSignature;
  mlir::Location loc = graphOp.getLoc();
  auto buildDesc =
      [uniform,
       loc](mlir::nv_tensor_ir::TensorType tensorTy,
            const TensorDescriptor &info) -> mlir::FailureOr<TensorArgDesc> {
    mlir::FailureOr<ElementTypeInfo> elementInfo =
        resolveElementTypeInfo(loc, tensorTy.getElementType());
    if (failed(elementInfo)) {
      return mlir::failure();
    }

    auto shape = tensorTy.getShape();
    int32_t rank = static_cast<int32_t>(shape.size());
    llvm::SmallVector<int64_t> staticShape(shape.begin(), shape.end());
    int32_t numDynSizes = uniform ? rank : countDynamicDims(shape);
    bool hasExplicitStrides = !info.strides.empty();

    llvm::SmallVector<int64_t> staticStrides;
    int32_t numDynStrides = 0;
    if (hasExplicitStrides) {
      for (const auto &stride : info.strides) {
        staticStrides.push_back(stride.staticValue);
      }
      numDynStrides = uniform ? rank : countDynamicDims(staticStrides);
    }

    return TensorArgDesc{rank,
                         std::move(staticShape),
                         std::move(staticStrides),
                         numDynSizes,
                         numDynStrides,
                         hasExplicitStrides,
                         /*isScalar=*/false,
                         *elementInfo};
  };

  // Helper: build a scalar operand descriptor from a non-tensor MLIR type.
  // The kernel signature for a scalar graph argument is a single by-value
  // kernel parameter, so we record the element type so the runtime
  // arg-packer can memcpy the value (instead of packing a pointer).
  auto buildScalarDesc =
      [loc](mlir::Type type) -> mlir::FailureOr<TensorArgDesc> {
    mlir::FailureOr<ElementTypeInfo> elementInfo =
        resolveElementTypeInfo(loc, type);
    if (failed(elementInfo)) {
      return mlir::failure();
    }
    return TensorArgDesc{/*rank=*/0,
                         /*staticShape=*/{},
                         /*staticStrides=*/{},
                         /*numDynSizes=*/0,
                         /*numDynStrides=*/0,
                         /*hasExplicitStrides=*/false,
                         /*isScalar=*/true,
                         *elementInfo};
  };

  // Process input and output operands (same logic, different arrays).
  // Track how many valid input operands we add to tensorDescs (tensors and
  // scalars both count; rank-0 tensors are skipped).
  int32_t numValidInputTensors = 0;
  auto processTensors = [&](mlir::TypeRange types,
                            llvm::ArrayRef<TensorDescriptor> infos,
                            bool isInput) -> mlir::LogicalResult {
    for (auto [idx, type] : llvm::enumerate(types)) {
      auto tensorTy = mlir::dyn_cast<mlir::nv_tensor_ir::TensorType>(type);
      if (!tensorTy) {
        // Scalar (non-tensor) operand. The MLIR signature converter maps
        // it to a single by-value kernel parameter; record its size so
        // the runtime can pack it by value.
        if (!type.isIntOrFloat()) {
          continue; // Unsupported scalar kind, skip.
        }
        mlir::FailureOr<TensorArgDesc> desc = buildScalarDesc(type);
        if (failed(desc)) {
          return mlir::failure();
        }
        layout.tensorDescs.push_back(*desc);
        if (isInput) {
          ++numValidInputTensors;
        }
        continue;
      }
      // Skip rank-0 tensors (scalars) - they don't need dynamic shape/stride
      // info.
      if (tensorTy.getRank() == 0) {
        continue;
      }
      mlir::FailureOr<TensorArgDesc> desc = buildDesc(tensorTy, infos[idx]);
      if (failed(desc)) {
        return mlir::failure();
      }
      layout.tensorDescs.push_back(*desc);
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
  // tensorDescs.
  layout.numInputs = numValidInputTensors;

  // Compute total kernel args.
  layout.totalKernelArgs = 0;
  for (const auto &desc : layout.tensorDescs) {
    layout.totalKernelArgs += desc.totalArgs();
  }

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
      auto tensorTy = mlir::dyn_cast<mlir::nv_tensor_ir::TensorType>(type);
      if (!tensorTy) {
        // Scalar operands do occupy a slot in tensorDescs, so advance the
        // running index. Unsupported non-tensor types are not emitted and
        // therefore do not consume a slot.
        if (type.isIntOrFloat()) {
          ++tensorDescIdx;
        }
        continue;
      }
      if (tensorTy.getRank() == 0) {
        // Rank-0 tensors are skipped entirely and consume no slot.
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

  return layout;
}

} // namespace tensor_ir
