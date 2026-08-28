# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUDA Tile backend support for the TensorIR tracing DSL."""

from dataclasses import dataclass, field, replace

from nv_tensor_ir._mlir.dialects import nv_tensor_ir

from .dsl import (
    BackendImpl,
    CodegenStrategy,
    CompileOptions,
    CompiledKernel,
    _register_backend_impl,
)
from .module_builder import TensorInfoOverrides
from .tracing import MovementOp, NodeKind, TensorInfo, TraceGraph


_DYNAMIC_SHAPE_ALLOWED_NODE_KINDS = {
    NodeKind.INPUT,
    NodeKind.OUTPUT_REF,
    NodeKind.CONSTANT,
    NodeKind.SPLAT,
    NodeKind.UNARY,
    NodeKind.BINARY,
    NodeKind.CONVERT,
    NodeKind.CMP,
    NodeKind.MATMUL,
    NodeKind.SELECT,
}

_DYNAMIC_SHAPE_ALLOWED_MOVEMENT_OPS = {
    MovementOp.TRANSPOSE,
}


@dataclass
class CudaTileCompiledKernel(CompiledKernel):
    """TensorIR DSL kernel compiled for the CUDA Tile backend.

    ``graph`` is the trace captured during compilation. It is exposed for
    lightweight inspection and statistics; execution uses ``program``.
    """

    program: nv_tensor_ir.Program
    module: object
    graph: TraceGraph
    input_count: int
    output_count: int
    _context: object | None = field(default=None, repr=False)

    def get_bytecode(self) -> bytes:
        return self.program.get_bytecode()

    def run(self, *inputs: object, output: object | tuple[object, ...]) -> None:
        if len(inputs) != self.input_count:
            raise TypeError(
                f"Expected {self.input_count} input tensors, got {len(inputs)}"
            )
        outputs = output if isinstance(output, tuple) else (output,)
        if len(outputs) != self.output_count:
            raise TypeError(
                f"Expected {self.output_count} output tensors, got {len(outputs)}"
            )
        self.program.launch(*inputs, *outputs)


def prepare_tensor_infos(
    graph: TraceGraph,
    *,
    dynamic_shape: bool = False,
) -> TensorInfoOverrides:
    """Prepare CUDA Tile tensor metadata for the shared module builder."""
    if not dynamic_shape:
        return {}
    _check_dynamic_shape_supported(graph)
    return _dynamic_ir_metadata_overrides(graph)


def _check_dynamic_shape_supported(graph: TraceGraph) -> None:
    for node in graph.nodes:
        if node.kind in _DYNAMIC_SHAPE_ALLOWED_NODE_KINDS:
            continue
        if (
            node.kind == NodeKind.MOVEMENT
            and node.op in _DYNAMIC_SHAPE_ALLOWED_MOVEMENT_OPS
        ):
            continue
        raise TypeError(
            "dynamic_shape=True currently supports pointwise graphs and selected "
            f"movement ops; op '{node.op_name}' is not supported"
        )


def _dynamic_ir_metadata_overrides(graph: TraceGraph) -> dict[int, TensorInfo]:
    """Return CUDA Tile dynamic tensor metadata keyed by trace node ID.

    Inputs and output references seed the dynamic set. Nodes are visited once
    in trace creation order, so each producer is classified before its users;
    SSA and explicit ``like`` dependencies propagate dynamic metadata forward.
    Overrides set the unified ``dynamic_shape`` flag; the module builder turns
    all shape dimensions and non-unit strides into ``?``. The concrete trace
    graph is not modified.
    """
    dynamic_node_ids = set(graph.input_ids) | set(graph.output_ref_ids)
    for node_id, node in enumerate(graph.nodes):
        like_id = node.kwargs.get("like")
        if like_id in dynamic_node_ids or any(
            argument_id in dynamic_node_ids for argument_id in node.args
        ):
            dynamic_node_ids.add(node_id)

    overrides: dict[int, TensorInfo] = {}
    for node_id in dynamic_node_ids:
        info = graph.nodes[node_id].tensor_info
        overrides[node_id] = replace(info, dynamic_shape=True)
    return overrides


def compile_cuda_tile(
    module: object,
    context: object,
    graph: TraceGraph,
    options: CompileOptions | None = None,
    tile_sizes: tuple[int, ...] = (),
    dynamic_shape: bool = False,
) -> CudaTileCompiledKernel:
    """Compile a traced kernel with the CUDA Tile backend."""
    if options is not None and tile_sizes:
        raise TypeError("Pass either options= or tile_sizes, not both")
    if dynamic_shape and any(node.kind == NodeKind.MATMUL for node in graph.nodes):
        codegen_strategy = (
            options.codegen_strategy
            if options is not None
            else CodegenStrategy.LayoutPropagation
        )
        if codegen_strategy != CodegenStrategy.AffineMap:
            raise ValueError(
                "Dynamic matmul requires CodegenStrategy.AffineMap; "
                "layout propagation does not support dynamic matmul"
            )
    if options is not None:
        program = nv_tensor_ir.compile(module, options=options)
    else:
        program = nv_tensor_ir.compile(module, tile_sizes=tile_sizes)
    return CudaTileCompiledKernel(
        program, module, graph, len(graph.input_ids), len(graph.output_ref_ids), context
    )


_register_backend_impl(
    "cuda_tile",
    BackendImpl(
        prepare_tensor_infos=prepare_tensor_infos,
        compile=compile_cuda_tile,
    ),
)
