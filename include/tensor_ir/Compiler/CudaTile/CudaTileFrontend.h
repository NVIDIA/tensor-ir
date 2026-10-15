// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_CUDATILE_CUDATILEFRONTEND_H
#define TENSOR_IR_COMPILER_CUDATILE_CUDATILEFRONTEND_H

#include "tensor_ir/Artifact/KernelMetadata.h"
#include "tensor_ir/Options/Options.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include <array>
#include <functional>
#include <string>

namespace mlir::nv_tensor_ir::compiler::cuda_tile {

/// Result of the shared TensorIR-to-CudaTile frontend. Artifact backends
/// consume this and diverge only at serialization/compilation time.
struct CudaTileFrontendResult {
  explicit CudaTileFrontendResult(::tensor_ir::rt::KernelMetadata argLayout)
      : argLayout(std::move(argLayout)) {}

  OwningOpRef<ModuleOp> module;
  ::mlir::cuda_tile::ModuleOp cudaTileModule;
  std::string runtimeKernelName;
  std::string entryFunctionName;
  ::tensor_ir::rt::KernelMetadata argLayout;
  SmallVector<int64_t> resolvedIterationSpaceShape;
  bool useRuntimeGrid = false;
  /// True when the kernel waits on a grid dependency (`gdc_wait_tko`) and can
  /// therefore be launched with programmatic dependent launch.
  bool usesPDL = false;
};

/// Lightweight legality check for the frontend. This intentionally avoids
/// running graph analysis or conversion so public `canCompile` remains cheap.
bool isCudaTileFrontendSupported(ModuleOp module,
                                 const CompilerInvocation &options);

/// Lower TensorIR to CudaTile and return the frontend artifact metadata.
/// `timing` optionally nests frontend pass timing under the supplied scope.
/// `onTensorIRReady` is invoked after analysis and tile selection, before
/// conversion to CudaTile.
FailureOr<CudaTileFrontendResult>
lowerTensorIRToCudaTile(ModuleOp module, const CompilerInvocation &options,
                        TimingScope *timing = nullptr,
                        std::function<void(ModuleOp)> onTensorIRReady = {});

/// Compute a fixed grid for static-shape frontend results.
std::array<int32_t, 3>
computeStaticGridSize(const CudaTileFrontendResult &result);

} // namespace mlir::nv_tensor_ir::compiler::cuda_tile

#endif // TENSOR_IR_COMPILER_CUDATILE_CUDATILEFRONTEND_H
