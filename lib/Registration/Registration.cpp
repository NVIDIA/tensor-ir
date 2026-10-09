// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Registration/Registration.h"

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/PtrToLLVM/PtrToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/Transforms/IndexedAccessOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/IndexedAccessOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

using namespace mlir;

void nv_tensor_ir::registerDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    arith::ArithDialect,
    bufferization::BufferizationDialect,
    func::FuncDialect,
    gpu::GPUDialect,
    LLVM::LLVMDialect,
    linalg::LinalgDialect,
    math::MathDialect,
    memref::MemRefDialect,
    ptr::PtrDialect,
    scf::SCFDialect,
    tensor::TensorDialect,
    vector::VectorDialect,
    cuda_tile::CudaTileDialect,
    nv_tensor_ir::TensorIRDialect
  >();
  // clang-format on

  // Register inliner extensions for every upstream dialect that provides one.
  func::registerInlinerExtension(registry);
  LLVM::registerInlinerInterface(registry);

  // Register the remaining interfaces that extend the LLVM dialect itself.
  registerLLVMDialectImport(registry);
  registerLLVMDialectTranslation(registry);

  // Register ConvertToLLVM interfaces for every upstream dialect above that
  // provides one.
  arith::registerConvertArithToLLVMInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  gpu::registerConvertGpuToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  ptr::registerConvertPtrToLLVMInterface(registry);
  vector::registerConvertVectorToLLVMInterface(registry);

  // Register external operation interface models for the upstream dialects.
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  gpu::registerIndexedAccessOpInterfaceExternalModels(registry);
  gpu::registerValueBoundsOpInterfaceExternalModels(registry);
  linalg::registerSubsetOpInterfaceExternalModels(registry);
  linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  memref::registerValueBoundsOpInterfaceExternalModels(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  vector::registerIndexedAccessOpInterfaceExternalModels(registry);
  vector::registerSubsetOpInterfaceExternalModels(registry);
  vector::registerValueBoundsOpInterfaceExternalModels(registry);
  builtin::registerCastOpInterfaceExternalModels(registry);
}

void nv_tensor_ir::registerPasses() {
  registerTensorToCudaTileConversionPasses();
  registerNVTensorIRTransformPasses();
  mlir::registerTransformsPasses();
}
