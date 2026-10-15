// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Passes.td - TensorIR transformation passes --------*- tablegen -*-===//

#ifndef TENSOR_IR_TRANSFORM_PASSES
#define TENSOR_IR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def MaterializeDefaultStridesPass : Pass<"materialize-default-strides", "nv_tensor_ir::GraphOp"> {
  let summary = "Attach row-major stride attributes to graph args/results that lack them";
  let description = [{
    Walks the GraphOp's argument and result attribute dictionaries. For each
    tensor-typed argument or result that does not already carry a
    `nv_tensor_ir.stride` attribute, computes a row-major stride from the
    tensor's shape (`tcg::layout(shape, gen_row_major{}).stride()`) and
    attaches it.

    Running this pass at the front of every TensorIR backend pipeline makes
    the row-major default explicit in the IR. After this pass, downstream
    code can read strides from the graph attributes without thinking about
    layout conventions, and the per-call-site row-major defaults scattered
    across the codebase become dead.
  }];
}

def DiscoverIterationSpaceInfoPass : Pass<"discover-iteration-space-info", "nv_tensor_ir::GraphOp"> {
  let summary = "Discover iteration space structure and attach iteration space maps to operations";
  let description = [{
    This pass analyzes Tensor IR operations to discover their iteration space
    structure and attaches iteration space map attributes to operations and
    function arguments.
  }];
}

def RemoveIterationSpaceMapsPass : Pass<"remove-iteration-space-maps", "nv_tensor_ir::GraphOp"> {
  let summary = "Remove iteration space map attributes from operations and function arguments";
  let description = [{
    This pass removes all iteration space map attributes that were previously
    attached by the DiscoverIterationSpaceInfo pass.
  }];
}

def TileAnalyzerPass : Pass<"tile-analyzer", "nv_tensor_ir::GraphOp"> {
  let summary = "Analyze iteration space and generate tile candidates";
  let description = [{
    This pass computes the normalized layout-propagation iteration space,
    extracts its shape and source strides, and generates tile candidates.

    Candidate generation is skipped when a tile shape is already available:
    - "tile_size" attribute already exists on GraphOp (tile from tensor-ir)
    - tile_size pass option is non-empty (tile from compilation options)

    When neither source provides a tile, generates candidates based on
    max_candidates (must be >= 1). Tile selection and verification are handled
    by TileSelectionPass.

    Note: tile_size is duplicated on TileSelectionPass by design — here it
    gates candidate generation (skip when tile is already known), while
    TileSelectionPass uses it as the resolved tile source. Both are populated
    from the same compilation options in Compiler/CudaTile/Pipelines.cpp.
  }];

  let options = [
    ListOption<"tile_size", "tile-size", "int32_t",
               "Skip candidate generation when a tile is already provided "
               "via compilation options (TileSelectionPass uses this value "
               "as the resolved tile).">,
    Option<"max_candidates", "max-candidates", "int32_t", /*default=*/"1",
           "Number of tile candidates to generate (must be >= 1)">,
    Option<"computeCapability", "compute-capability", "int32_t", /*default=*/"100",
           "Target GPU compute capability as an integer (e.g., 90/100/120). Used for architecture-aware tiling heuristics in TileAnalyzerPass.">,
    Option<"sm_count", "sm-count", "int32_t", /*default=*/"0",
           "Runtime SM count. When 0 (default), a conservative per-architecture "
           "fallback is used.">
  ];
}

def TileSelectionPass : Pass<"tile-selection", "nv_tensor_ir::GraphOp"> {
  let summary = "Select and verify final tile size for codegen";
  let description = [{
    Resolves the final tile shape from one of three sources (in priority order):

    1. tile_size pass option (from compilation options — highest priority)
    2. "tile_size" attribute already on GraphOp (set through tensor-ir
       by the graph producer)
    3. "tile_candidates" attribute on GraphOp (from TileAnalyzerPass)
    If none of the above are available, the pass emits an error.

    Verifies the selected tile against the normalized layout-propagation
    iteration space (rank match and validity). On failure, emits an error including the selected tile shape
    and the current iteration space shape.

    Removes "tile_candidates" (if present), and sets "tile_size" as a
    DenseI32ArrayAttr on GraphOp. After this pass, "tile_size" is guaranteed
    to be present — no empty tile_size is allowed in codegen from this point on.
  }];

  let options = [
    ListOption<"tile_size", "tile-size", "int32_t",
               "Tile size from compilation options. Overrides tile_size "
               "already present on the GraphOp.">
  ];
}

def BufferizePass : Pass<"tir-bufferize", "mlir::ModuleOp"> {
  let summary = "Create destination-passing TensorIR buffer boundaries";
  let description = [{
    Replaces each TensorIR graph with a void function whose tensor inputs and
    appended result destinations are strided memrefs carrying
    `nv_tensor_ir.alignment` attributes. Restrict `bufferization.to_tensor` and
    `bufferization.materialize_in_destination` operations bridge these buffers
    to the unchanged tensor computation. The pass emits no memory accesses,
    preserves tile-selection metadata, and performs no tiling or grid
    construction.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::nv_tensor_ir::TensorIRDialect",
    "::mlir::ptr::PtrDialect"
  ];
}

def FormGridPass : Pass<"tir-form-grid", "mlir::func::FuncOp"> {
  let summary = "Tile a bufferized TensorIR function onto a GPU grid";
  let description = [{
    Consumes a function produced by `tir-bufferize` and the selected tile
    shape, and computes the layout propagation and normalized iteration space
    of its tensor computation from the bufferization boundary. The function
    signature remains unchanged. A rank-one `scf.forall` mapped to
    `#gpu.block<x>` delinearizes the block coordinate, loads fixed-shape tiles,
    computes graph results, and stores them. These tiled `nv_tensor_ir.load` and
    `nv_tensor_ir.store` operations replace the bufferization bridges.

    Layout-only reshape, transpose, broadcast, and slice operations are folded
    into the explicit access metadata. Tile-selection attributes do not
    survive formation.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::nv_tensor_ir::TensorIRDialect",
    "::mlir::ptr::PtrDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def TileReductionsPass : Pass<"tir-tile-reductions", "mlir::func::FuncOp"> {
  let summary = "Materialize TensorIR contraction loops";
  let description = [{
    Rewrites coarse tile-shaped reductions and matrix products into explicit
    `scf.for` nests. The loop induction variables become contracting
    coordinates on TensorIR tile loads and tensor `iter_args` carry partial
    accumulators. Contractions whose complete domain fits the selected
    reduction tile remain loop-free.

    Static persistence caps each flattened `scf.forall` grid at
    `sm-count * occupancy` blocks and adds a grid-stride `scf.for` over the
    logical tile IDs. Runtime-shaped grids compute the cap in SSA. Dynamic
    persistence retains the non-persistent fallback.
  }];
  let options = [
    Option<"reduction_tile_size", "reduction-tile-size", "int64_t", "128",
           "Maximum number of contracting elements in one tile">,
    Option<"persistence", "persistence", "std::string", "\"none\"",
           "Persistence mode: none, static, or dynamic">,
    Option<"sm_count", "sm-count", "int64_t", "0",
           "Streaming multiprocessor count for static persistence">,
    Option<"occupancy", "occupancy", "int64_t", "1",
           "Persistent blocks per streaming multiprocessor">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::nv_tensor_ir::TensorIRDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def OutlineKernelPass : Pass<"outline-tensor-ir-kernel", "mlir::ModuleOp"> {
  let summary = "Outline mapped TensorIR forall bodies as GPU device functions";
  let description = [{
    Outlines the body of each rank-one `scf.forall` mapped to
    `#gpu.block<x>` into a destination-passing `func.func` marked `gpu.kernel`
    inside a shared, collision-free `gpu.module`. The original host function is
    retained or collision-safely renamed with a `_dispatch` suffix, and its
    forall is replaced by a synchronous `gpu.launch_func` that forwards the
    function arguments and preserves the grid's x dimension. The other grid
    dimensions and all thread dimensions are unity placeholders; CudaTile
    selects worker configuration later from entry-point optimization hints.

    Pure, regionless shape, index, and TensorIR view setup captured by the
    forall is cloned into the kernel. The forall induction variable and dynamic
    grid bound become `gpu.block_id x` and `gpu.grid_dim x`; all other captures
    must derive from wrapper arguments through pure setup operations.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::nv_tensor_ir::TensorIRDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def HostAbiBridgePass : Pass<"tir-host-abi-bridge", "ModuleOp"> {
  let summary = "Build the native host ABI entry point";
  let description = [{
    Threads the host runtime context and an error status through host code.
    Every defined top-level `func.func` receives a leading `!llvm.ptr` argument
    marked `nv_tensor_ir.runtime_context` and a leading `i32` status result;
    declarations and functions inside `gpu.module` keep their signatures.
    Existing returns report success (0). Each call to a defined host function
    forwards the caller's context; a nonzero callee status branches to a block
    that returns it, with poison for the caller's other results.

    Public functions must return nothing. Calls to defined host functions must
    be directly in the caller's body, so SCF must be lowered to CF before this
    pass.

    The pass then exports a C entry point,
    `func.func @tensor_ir_host_entry_<name>_v1`, for each public function. It
    has the `TensorIRHostEntryV1` signature
    `(runtime, packedArgs, packedArgCount) -> i32`, passes `runtime` as the
    context, and decodes the remaining arguments in order from the `int64_t`
    slots of `packedArgs`: a memref takes its base pointer, sizes, and strides
    and gets a zero offset; a scalar takes one slot holding its value bits in
    the low bytes. The entry point returns the function's status.
  }];
  let dependentDialects = [
    "::mlir::nv_tensor_ir::TensorIRDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::func::FuncDialect",
    "::mlir::ub::UBDialect"
  ];
}


def HostToLLVMPass : Pass<"tir-host-to-llvm", "ModuleOp"> {
  let summary = "Lower host operations to LLVM while preserving GPU modules";
  let description = [{
    Replaces host `gpu.launch_func` operations with indirect calls to the
    `TensorIRHostRuntimeV1` launch callback and converts arith, cf, func, memref,
    ptr, and ub operations to LLVM. Device `gpu.module` bodies remain intact for
    subsequent device compilation. Unlowered host operations or unresolved
    conversion casts cause pass failure.

    Requires SCF to be lowered to CF and the ABI bridge's marked runtime pointer
    argument and leading i32 status result. Launches must be synchronous and
    have no cluster dimensions. Preserves explicit kernel ordinals and assigns
    missing ordinals in host launch order.

    Returns -3 on callback failure. Invalid launch configuration ranges return
    -5; unrepresentable dynamic device sizes or strides return -2. Range checks
    precede narrowing and callback invocation.
  }];
  let dependentDialects = [
    "::mlir::nv_tensor_ir::TensorIRDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::func::FuncDialect",
    "::mlir::ub::UBDialect"
  ];
}

def CudaTileInsertGdcWaitPass : Pass<"tir-cuda-tile-insert-gdc-wait", "::mlir::cuda_tile::EntryOp"> {
  let summary = "Conservatively insert a grid dependency wait into a CUDA Tile entry";
  let description = [{
    Makes a CUDA Tile kernel safe to launch with programmatic dependent launch
    (PDL). A single `gdc_wait_tko` is inserted before the first top-level
    operation of the entry that contains a global-memory access. The wait's
    token is attached to every global-memory operation it dominates, including
    operations nested in `for`/`if` regions: operations without a token
    consume it directly, operations that already consume a token consume a
    `join_tokens` of the wait token and their existing token.

    No `gdc_launch_dependents_tko` is emitted; dependent grids are triggered
    implicitly when the grid exits. Entries that already contain an explicit
    `gdc_wait_tko`, and entries without any global-memory access, are left
    unchanged. An explicit `gdc_launch_dependents_tko` alone does not prevent
    the insertion, since it does not order the kernel's own memory accesses
    after the preceding grid. The pass fails if a global-memory access nested
    under another isolated-from-above operation would need the wait token.
  }];
  let dependentDialects = ["::mlir::cuda_tile::CudaTileDialect"];
}

def HostToObjectPass : Pass<"tir-host-to-object", "ModuleOp"> {
  let summary = "Translate host program to a relocatable object";
  let description = [{
    This pass translates the host program to a native relocatable object
    and embeds the ELF binary as an MLIR ASM Resource. The
    ASM resource is referenced via an attribute,
    'nv_tensor_ir.host_object', on the top-level ModuleOp.

    By default, the pass retains the MLIR LLVMIR functions
    in the ModuleOp. Set 'drop-host-ir=true' to drop the
    functions after the host ELF image is assembled and
    embedded.
  }];
  let options = [
    Option<"hostArch", "host-arch",
              "::mlir::nv_tensor_ir::HostArch",
              [{::mlir::nv_tensor_ir::HostArch::Native}],
              "Host architecture. Default is uses host architecture.",
              [{::llvm::cl::values(
               clEnumValN(::mlir::nv_tensor_ir::HostArch::Native, "native", "Host architecture"),
               clEnumValN(::mlir::nv_tensor_ir::HostArch::X86_64, "x86_64", "X86_64 architecture"),
               clEnumValN(::mlir::nv_tensor_ir::HostArch::AArch64, "aarch64", "AArch64 architecture")
            )}]>,
    Option<"hostOS", "host-os",
              "::mlir::nv_tensor_ir::HostOS",
              [{::mlir::nv_tensor_ir::HostOS::Linux}],
              "Host OS. Ignored if 'host-arch=native'.",
              [{::llvm::cl::values(
               clEnumValN(::mlir::nv_tensor_ir::HostOS::Linux, "linux", "Linux OS"),
               clEnumValN(::mlir::nv_tensor_ir::HostOS::Windows, "windows", "Windows OS")
            )}]>,
    Option<"dropHostIR", "drop-host-ir", "bool", "false",
           "Drop the host IR after translation. Default is `false`">,
  ];
}

#endif // TENSOR_IR_TRANSFORM_PASSES
