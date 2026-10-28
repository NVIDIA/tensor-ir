# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUDA Tile backend support for the TensorIR tracing DSL."""

from dataclasses import dataclass, field, replace
from hashlib import sha256
from math import gcd

from nv_tensor_ir._mlir.dialects import nv_tensor_ir

from .dsl import (
    BackendImpl,
    CodegenStrategy,
    CompileOptions,
    CompiledKernel,
    ArtifactCache,
    _register_backend_impl,
)
from .module_builder import TensorInfoOverrides
from .tracing import MovementOp, NodeKind, ScalarInfo, TensorInfo, TraceGraph


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
    lightweight inspection and statistics; execution loads ``artifact`` on first use.
    """

    artifact: nv_tensor_ir.Artifact
    module: object
    graph: TraceGraph
    input_count: int
    output_count: int
    _context: object | None = field(default=None, repr=False)

    _runtime_kernel: nv_tensor_ir.RuntimeKernel | None = field(
        default=None, init=False, repr=False
    )

    def get_bytecode(self) -> bytes:
        return self.artifact.get_bytecode()

    def run(
        self, *inputs: object, output: object | tuple[object, ...], stream: int = 0
    ) -> None:
        """Launch after checking input/output alignment against the module.

        Raises ValueError if a runtime data address violates a compiled
        alignment requirement.
        """
        if len(inputs) != self.input_count:
            raise TypeError(
                f"Expected {self.input_count} input tensors, got {len(inputs)}"
            )
        outputs = output if isinstance(output, tuple) else (output,)
        if len(outputs) != self.output_count:
            raise TypeError(
                f"Expected {self.output_count} output tensors, got {len(outputs)}"
            )
        self._check_alignment(inputs, outputs)
        if self._runtime_kernel is None or self._runtime_kernel.is_destroyed:
            self._runtime_kernel = self.artifact.load()
        self._runtime_kernel.launch(*inputs, *outputs, stream=stream)

    def _check_alignment(
        self, inputs: tuple[object, ...], outputs: tuple[object, ...]
    ) -> None:
        """Validate runtime addresses against the compiled graph's attributes."""
        from nv_tensor_ir._mlir import ir
        from nv_tensor_ir._mlir._mlir_libs import _tensor_ir

        graph_op = next(
            op
            for op in self.module.body.operations
            if op.operation.name == "nv_tensor_ir.graph"
        )
        for attr_name, kind, tensors in (
            ("arg_attrs", "Input", inputs),
            ("res_attrs", "Output", outputs),
        ):
            if attr_name not in graph_op.attributes:
                continue
            attrs = ir.ArrayAttr(graph_op.attributes[attr_name])
            for index, tensor in enumerate(tensors):
                tensor_attrs = ir.DictAttr(attrs[index])
                if "nv_tensor_ir.alignment" not in tensor_attrs:
                    continue
                alignment = ir.IntegerAttr(tensor_attrs["nv_tensor_ir.alignment"]).value
                if alignment <= 1:
                    continue
                _, _, _, pointer = _tensor_ir._tensor_metadata_from_dlpack(tensor)
                if not pointer or pointer % alignment:
                    raise ValueError(
                        f"{kind} tensor {index} requires a non-null data pointer "
                        f"aligned to {alignment} bytes; got {pointer:#x}"
                    )


def prepare_tensor_infos(
    graph: TraceGraph,
    *,
    options: CompileOptions | None = None,
    dynamic_shape: bool = False,
) -> TensorInfoOverrides:
    """Reject unsupported divisibility; prepare alignment for MLIR emission.

    Traced DLPack tensors use observed pointers in static mode and layout rules
    in dynamic mode. AffineMap skips inference; explicit alignment is preserved.
    """
    if any("dynamic_dimensions" in node.kwargs for node in graph.nodes):
        raise TypeError(
            "The CUDA Tile backend does not yet support per-dimension dynamic shapes"
        )
    if any(
        node.value_info.stride_divisibility is not None
        for node in graph.nodes
        if not isinstance(node.value_info, ScalarInfo)
    ):
        raise TypeError("The CUDA Tile backend does not support stride_divisibility")
    overrides = {}
    if dynamic_shape:
        _check_dynamic_shape_supported(graph)
        overrides = _dynamic_ir_metadata_overrides(graph)
    options = options if options is not None else CompileOptions()
    if options.codegen_strategy == CodegenStrategy.AffineMap:
        return overrides
    for node_id in graph.input_ids + graph.output_ref_ids:
        info = overrides.get(node_id, graph.nodes[node_id].value_info)
        if isinstance(info, ScalarInfo):
            continue
        pointer = info._traced_tensor_ptr
        if info.alignment is not None or pointer is None:
            continue
        if dynamic_shape:
            # Infer from the concrete trace, not the dynamic IR overrides.
            alignment = _dynamic_tensor_alignment(graph.nodes[node_id].value_info)
        else:
            # Cap at 32B for 256-bit vectorization without specializing on
            # over-aligned pointers. Null data gives no alignment guarantee.
            alignment = min(pointer & -pointer, 32) if pointer else None
        overrides[node_id] = replace(info, alignment=alignment)
    return overrides


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
    Dynamic shapes become all ``?`` and only unit strides remain static. The
    concrete trace graph is not modified.
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
        info = graph.nodes[node_id].value_info
        if isinstance(info, ScalarInfo):
            continue
        overrides[node_id] = replace(
            info,
            shape=tuple(None for _ in info.shape),
            stride=tuple(1 if stride == 1 else None for stride in info.stride),
        )
    return overrides


def _dynamic_tensor_alignment(info: TensorInfo) -> int | None:
    """Infer byte alignment from the contiguous extent and TMA requirement.

    Prefer the last non-unit extent with stride 1, falling back to the last
    stride-1 dimension. Scalar layouts and layouts without a stride-1 dimension
    have no inferred alignment annotation. Otherwise use the GCD of the
    contiguous extent's bit size and the TMA alignment requirement, with a
    minimum byte alignment of 1 for sub-byte results.
    """
    if all(size == 1 for size in info.shape):
        return None
    leading_dims = [i for i, stride in enumerate(info.stride) if stride == 1]
    if not leading_dims:
        return None
    non_unit_dims = [i for i in leading_dims if info.shape[i] != 1]
    leading_dim = (non_unit_dims or leading_dims)[-1]
    # DataType values use MLIR scalar spellings (f32, bf16, si8, ui8, i1).
    width = int(info.dtype.value.lstrip("bfsui"))
    tma_alignment_bits = 128 if width >= 8 else 256
    return max(1, gcd(info.shape[leading_dim] * width, tma_alignment_bits) // 8)


def _calculate_cache_key(
    module: object,
    options: CompileOptions | None = None,
    *,
    tile_sizes: tuple[int, ...] = (),
) -> str:
    """Return the SHA-256 cache key for a CUDA Tile MLIR module and options."""
    options = _resolve_compile_options(options, tile_sizes)
    module_key = "cuda_tile_backend-" + nv_tensor_ir.calculate_cache_key(
        module, options
    )
    return sha256(module_key.encode(encoding="utf-8")).hexdigest()


def _resolve_compile_options(
    options: CompileOptions | None,
    tile_sizes: tuple[int, ...],
) -> CompileOptions:
    if options is not None and tile_sizes:
        raise TypeError("Pass either options= or tile_sizes, not both")
    if options is None:
        options = CompileOptions()
    if tile_sizes:
        options.tile_sizes = list(tile_sizes)
    return options


def compile_cuda_tile(
    module: object,
    context: object,
    graph: TraceGraph,
    options: CompileOptions | None = None,
    tile_sizes: tuple[int, ...] = (),
    dynamic_shape: bool = False,
    artifact_cache: ArtifactCache | None = None,
) -> CudaTileCompiledKernel:
    """Compile a traced kernel with the CUDA Tile backend."""
    options = _resolve_compile_options(options, tile_sizes)
    if dynamic_shape and any(node.kind == NodeKind.MATMUL for node in graph.nodes):
        codegen_strategy = options.codegen_strategy
        if codegen_strategy != CodegenStrategy.AffineMap:
            raise ValueError(
                "Dynamic matmul requires CodegenStrategy.AffineMap; "
                "layout propagation does not support dynamic matmul"
            )

    if artifact_cache is None:
        program = nv_tensor_ir.compile(module, options=options)
    else:
        cache_key = _calculate_cache_key(module, options=options)
        program = artifact_cache._get(cache_key)
        if program is None:
            program = nv_tensor_ir.compile(module, options=options)
            artifact_cache._put(cache_key, program)

    return CudaTileCompiledKernel(
        program,
        module,
        graph,
        len(graph.input_ids),
        len(graph.output_ref_ids),
        context,
    )


_register_backend_impl(
    "cuda_tile",
    BackendImpl(
        prepare_tensor_infos=prepare_tensor_infos,
        calculate_cache_key=_calculate_cache_key,
        compile=compile_cuda_tile,
    ),
)
