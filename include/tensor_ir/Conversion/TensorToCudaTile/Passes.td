// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===-- Passes.td - TensorToCudaTile conversion pass definitions ---*- tablegen -*-===//

#ifndef TENSOR_IR_CONVERSION_TENSORTOCUDATILE_PASSES
#define TENSOR_IR_CONVERSION_TENSORTOCUDATILE_PASSES

include "mlir/Pass/PassBase.td"

def TensorToCudaTileConversionPass : Pass<"convert-tensor-to-cuda-tile", "mlir::ModuleOp"> {
  let summary = "Convert Tensor IR to Cuda Tile IR";
  let description = [{
    Lower TensorIR programs to the `cuda_tile` dialect.

    Two codegen strategies (controlled by `codegen-strategy`):
      - `layout_propagation` (default): lowers destination-passing
        `func.func` operations marked `gpu.kernel` inside `gpu.module`
        operations. Run grid formation, strip-mining, and outlining before this
        pass. Retained host wrappers and their staging `gpu.launch_func`
        operations are consumed before kernel conversion; the emptied GPU
        modules and `gpu.container_module` attribute are then removed. Analysis
        attributes and `nv_tensor_ir.graph` are consumed or rejected at this
        boundary.
      - `affine_map`: uses affine maps per op; `tile-size` is interpreted
        directly from the pass option.

    Option groups by strategy:
      - Always read: `num-ctas`, `occupancy`, `num-warps`, and
        `uniform-signature`.
      - `affine_map`-only: `tile-size`, `persistence`, and `sm-count`.
      - Tiling, reduction strip-mining, and layout-path persistence are owned by
        the preceding formation passes.
  }];
  let dependentDialects = [
    "mlir::cuda_tile::CudaTileDialect",
  ];
  let options = [
    Option<"codegen_strategy", "codegen-strategy",
           "mlir::nv_tensor_ir::CudaTileCodegenStrategy",
           /*default=*/"mlir::nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation",
           "Which codegen strategy to use.",
           [{llvm::cl::values(
               clEnumValN(mlir::nv_tensor_ir::CudaTileCodegenStrategy::AffineMap,
                          "affine_map",
                          "Per-op affine-map codegen."),
               clEnumValN(mlir::nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation,
                          "layout_propagation",
                          "Lower an explicitly formed and outlined TensorIR "
                          "tiled program (default).")
           )}]>,
    Option<"num_ctas", "num-ctas", "int32_t", /*default=*/"1",
           "Number of CTAs per cluster (CGA). Used to set num_cta_in_cga in "
           "cuda_tile.entry optimization_hints.">,
    Option<"occupancy", "occupancy", "int32_t", /*default=*/"1",
           "Target occupancy (CTAs per SM). Used to set occupancy in "
           "cuda_tile.entry optimization_hints. For static persistent kernels, "
           "the grid size is calculated as sm_count * occupancy.">,
    Option<"num_warps", "num-warps", "int32_t", /*default=*/"4",
           "Number of worker warps per CTA (must be a power of 2). "
           "Sets the num_worker_warps_per_cta hint on the cuda_tile.entry op. "
           "In warp-specialized kernels this controls the worker agent's warp "
           "count; additional warps may be allocated for MMA, TMA, and "
           "control agents.">,
    Option<"persistence", "persistence",
           "mlir::nv_tensor_ir::PersistenceMode",
           /*default=*/"mlir::nv_tensor_ir::PersistenceMode::None",
           "Kernel persistence mode for affine-map conversion.",
           [{llvm::cl::values(
               clEnumValN(mlir::nv_tensor_ir::PersistenceMode::None,
                          "none",
                          "Normal kernel launch, one CTA per tile, CTAs "
                          "terminate after processing."),
               clEnumValN(mlir::nv_tensor_ir::PersistenceMode::Static,
                          "static",
                          "Static persistent kernel (Ampere+): launches fixed "
                          "grid (sm_count * occupancy CTAs) that loops over "
                          "all tiles; falls back to normal mode when grid "
                          "size >= total tiles.")
           )}]>,
    Option<"sm_count", "sm-count", "int32_t", /*default=*/"0",
           "Runtime SM count for affine-map persistent kernels. Required for static persistence mode.">,
    Option<"uniform_signature", "uniform-signature", "bool",
           /*default=*/"false",
           "When true, the kernel signature always includes arguments for "
           "ALL tensor sizes and strides, even for statically-known "
           "dimensions (uniform layout). When false, only dynamic dims "
           "produce kernel arguments (original behavior).">,
    ListOption<"tile_size", "tile-size", "int32_t",
               "A list of integers for affine-map tile selection.">,
  ];
}

#endif // TENSOR_IR_CONVERSION_TENSORTOCUDATILE_PASSES
