// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Builds a KernelArgLayout from a TensorIR GraphOp and CUDA Tile ABI options.
//===----------------------------------------------------------------------===//

#include "tensor_ir/Compiler/CudaTile/KernelArgLayout.h"

#include "tensor_ir/Analysis/TileAnalyzer.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Utils/Utils.h"

namespace tensor_ir {

using rt::KernelArgLayout;
using rt::TensorArgDesc;

/// Count elements equal to kDynamic in an int64_t array.
static int countDynamic(llvm::ArrayRef<int64_t> vals) {
  return llvm::count_if(vals,
                        [](int64_t v) { return v == TensorArgDesc::kDynamic; });
}

KernelArgLayout extractKernelArgLayout(mlir::nv_tensor_ir::GraphOp graphOp,
                                       llvm::ArrayRef<int32_t> tileSizes,
                                       bool uniformSignature) {
  KernelArgLayout layout;
  layout.uniformSignature = uniformSignature;
  auto funcType = graphOp.getFunctionType();
  layout.numInputs = funcType.getNumInputs();

  // Copy tile sizes from conversion options.
  layout.tileSizes.assign(tileSizes.begin(), tileSizes.end());

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
  auto buildDesc = [uniform](mlir::nv_tensor_ir::TensorType tensorTy,
                             const TensorDescriptor &info) -> TensorArgDesc {
    TensorArgDesc desc;
    auto shape = tensorTy.getShape();
    desc.rank = static_cast<int32_t>(shape.size());
    desc.staticShape.assign(shape.begin(), shape.end());
    desc.numDynSizes = uniform ? desc.rank : countDynamic(shape);
    desc.hasExplicitStrides = !info.strides.empty();
    desc.numDynStrides = 0;

    if (desc.hasExplicitStrides) {
      for (const auto &stride : info.strides) {
        desc.staticStrides.push_back(stride.staticValue);
      }
      desc.numDynStrides =
          uniform ? desc.rank : countDynamic(desc.staticStrides);
    }
    return desc;
  };

  // Helper: build a scalar operand descriptor from a non-tensor MLIR type.
  // The kernel signature for a scalar graph argument is a single by-value
  // kernel parameter, so we record the element size so the runtime
  // arg-packer can memcpy the value (instead of packing a pointer).
  auto buildScalarDesc = [](mlir::Type type) -> TensorArgDesc {
    TensorArgDesc desc;
    desc.isScalar = true;
    desc.rank = 0;
    if (type.isIntOrFloat()) {
      desc.scalarSizeInBytes =
          static_cast<int32_t>((type.getIntOrFloatBitWidth() + 7) / 8);
    }
    return desc;
  };

  // Process input and output operands (same logic, different arrays).
  // Track how many valid input operands we add to tensorDescs (tensors and
  // scalars both count; rank-0 tensors are skipped).
  int32_t numValidInputTensors = 0;
  auto processTensors = [&](mlir::TypeRange types,
                            llvm::ArrayRef<TensorDescriptor> infos,
                            bool isInput) {
    for (auto [idx, type] : llvm::enumerate(types)) {
      auto tensorTy = mlir::dyn_cast<mlir::nv_tensor_ir::TensorType>(type);
      if (!tensorTy) {
        // Scalar (non-tensor) operand. The MLIR signature converter maps
        // it to a single by-value kernel parameter; record its size so
        // the runtime can pack it by value.
        if (!type.isIntOrFloat()) {
          continue; // Unsupported scalar kind, skip.
        }
        layout.tensorDescs.push_back(buildScalarDesc(type));
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
      layout.tensorDescs.push_back(buildDesc(tensorTy, infos[idx]));
      if (isInput) {
        ++numValidInputTensors;
      }
    }
  };

  processTensors(funcType.getInputs(), *argInfos, /*isInput=*/true);
  processTensors(funcType.getResults(), *resInfos, /*isInput=*/false);

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
