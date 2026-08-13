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

def LayoutPropagationAnnotationPass : Pass<"layout-propagation-annotation", "nv_tensor_ir::GraphOp"> {
  let summary = "Add layout source annotations for graph operations";
  let description = [{
    This pass walks graph operations in forward SSA order and adds a `layout`
    attribute that describes the tensor sources and layout transformations
    needed to compute each operation result.

    For operations that implement `LayoutPropInterface`, the pass collects one
    layout source per operand, in operand order, and calls `inferResultLayout`.
    Pointwise operations combine their distinct operand sources. Operand-free
    tensor producers such as constants, splats, and iotas seed propagation with
    an initial layout source.

    Inferred result layouts retain their operand layout provenance. The later
    graph-splitting phase uses that provenance to materialize operand layouts
    while traversing from a requested result layout back toward graph inputs.
    That traversal is backward; the result layout inference performed by this
    pass is forward.
  }];
}

def LayoutPropagationNormalizationPass : Pass<"layout-propagation-normalization", "nv_tensor_ir::GraphOp"> {
  let summary = "Add iteration space annotations for graph operations";
  let description = [{
    This pass adds the "iteration_space" attribute to the graph operations that
    define the iteration space: the graph output and any ops that modify the
    iteration space (have hidden dimensions), i.e. reductions and matmuls.
  }];
}

def TileAnalyzerPass : Pass<"tile-analyzer", "nv_tensor_ir::GraphOp"> {
  let summary = "Analyze iteration space and generate tile candidates";
  let description = [{
    This pass reads normalized layout attributes (TensorSourceAttr/CompositeSourceAttr),
    extracts the common iteration space, and generates tile candidates.

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

    Verifies the selected tile against the iteration space (rank match and
    validity). On failure, emits an error including the selected tile shape
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

def GraphSplittingPass : Pass<"graph-splitting", "nv_tensor_ir::GraphOp"> {
  let summary = "Split reconvergent graphs into layout-specialized trees";
  let description = [{
    This pass rewrites a TensorIR graph into an equivalent tree by cloning
    operations for each required layout. The pass is intended to run after
    layout propagation/annotation so that every tensor-producing operation has
    a "layout" attribute.
  }];
}

#endif // TENSOR_IR_TRANSFORM_PASSES
