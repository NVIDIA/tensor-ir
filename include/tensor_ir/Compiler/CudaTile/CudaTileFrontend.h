// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_CUDATILE_CUDATILEFRONTEND_H
#define TENSOR_IR_COMPILER_CUDATILE_CUDATILEFRONTEND_H

#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "tensor_ir/Runtime/CudaTile/KernelArgLayout.h"

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

/// Debug controls owned by the shared TensorIR-to-CudaTile frontend.
struct CudaTileFrontendDebugOptions {
  std::string dumpCudaTileIRPath;
  std::string printIRTreeDir;
  bool printIRAfterAll = false;
  bool printCudaTileIR = false;
  bool enableTiming = false;
};

/// TensorIR-to-CudaTile lowering options.
struct CudaTileFrontendOptions {
  TensorToCudaTilePipelineOptions pipelineOptions;
  CudaTileFrontendDebugOptions debug;

  /// Optional parent scope for frontend phase and pass timing.
  TimingScope *timing = nullptr;

  /// Optional hook invoked on TensorIR after graph analysis and tile selection
  /// but before conversion to CudaTile. The affine-map path invokes the hook
  /// immediately before its combined conversion pipeline.
  std::function<void(ModuleOp)> onTensorIRReady;
};

/// Result of the shared TensorIR-to-CudaTile frontend. Artifact backends
/// consume this and diverge only at serialization/compilation time.
struct CudaTileFrontendResult {
  OwningOpRef<ModuleOp> module;
  ::mlir::cuda_tile::ModuleOp cudaTileModule;
  std::string runtimeKernelName;
  std::string entryFunctionName;
  ::tensor_ir::rt::KernelArgLayout argLayout;
  SmallVector<int64_t> resolvedIterationSpaceShape;
  bool useRuntimeGrid = false;
};

/// Lightweight legality check for the frontend. This intentionally avoids
/// running graph analysis or conversion so public `canCompile` remains cheap.
bool isCudaTileFrontendSupported(ModuleOp module,
                                 const CudaTileFrontendOptions &options);

/// Lower TensorIR to CudaTile and return the frontend artifact metadata.
FailureOr<CudaTileFrontendResult>
lowerTensorIRToCudaTile(ModuleOp module,
                        const CudaTileFrontendOptions &options);

/// Compute a fixed grid for static-shape frontend results.
std::array<int32_t, 3>
computeStaticGridSize(const CudaTileFrontendResult &result);

} // namespace mlir::nv_tensor_ir::compiler::cuda_tile

#endif // TENSOR_IR_COMPILER_CUDATILE_CUDATILEFRONTEND_H
