# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .module_builder import build_mlir_module
from .tensor_spec import tensor_spec_from_value
from .tracing import MovementOp, NodeKind, TensorInfo, TraceGraph, _trace_graph_context

from nv_tensor_ir._mlir.dialects import nv_tensor_ir

CompileOptions = nv_tensor_ir.CompileOptions
CodegenStrategy = nv_tensor_ir.CodegenStrategy
BytecodeVersion = nv_tensor_ir.BytecodeVersion
CudaTileArtifactKind = nv_tensor_ir.CudaTileArtifactKind


@dataclass(frozen=True)
class KernelFunction:
    func: Callable[..., object]

    @property
    def __name__(self) -> str:
        return self.func.__name__


@dataclass
class CompiledKernel:
    """Compiled TensorIR DSL kernel.

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


def kernel(
    func: Callable[..., object] | None = None,
) -> KernelFunction | Callable[[Callable[..., object]], KernelFunction]:
    def decorate(raw_func: Callable[..., object]) -> KernelFunction:
        if not callable(raw_func):
            raise TypeError("@tir.kernel expects a callable")
        return KernelFunction(func=raw_func)

    if func is not None:
        return decorate(func)
    return decorate


def _trace(
    kernel_func: KernelFunction,
    *inputs: object,
    output: object | tuple[object, ...],
    dynamic_shape: bool = False,
) -> TraceGraph:
    raw_func = _unwrap(kernel_func.func)
    signature = inspect.signature(raw_func, eval_str=True)
    graph = TraceGraph()
    _register_output_refs(graph, output, dynamic_shape=dynamic_shape)
    traced_args = []
    input_index = 0
    for param in signature.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError("TensorIR DSL kernels do not support *args or **kwargs")
        if param.name == "output":
            raise TypeError(
                "TensorIR DSL kernels should return computed tensors; pass "
                "output= to tir.compile instead of declaring an output parameter"
            )
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(
                "TensorIR DSL kernels do not support keyword-only input parameters"
            )
        if input_index >= len(inputs):
            raise TypeError(f"Missing tensor argument for parameter '{param.name}'")
        traced_args.append(
            graph.add_input(
                _tensor_info_from_value(
                    inputs[input_index],
                    param.name,
                    dynamic_shape=dynamic_shape,
                )
            )
        )
        input_index += 1
    if input_index != len(inputs):
        raise TypeError(f"Expected {input_index} input tensors, got {len(inputs)}")
    with _trace_graph_context(graph):
        result = raw_func(*traced_args)
    results = list(result) if isinstance(result, tuple) else [result]
    graph.set_results(results)
    if dynamic_shape:
        _check_dynamic_shape_supported(graph)
    return graph


def _build_module(
    kernel_func: KernelFunction,
    *inputs: object,
    output: object | tuple[object, ...],
    name: str | None = None,
    dynamic_shape: bool = False,
) -> tuple[object, object, TraceGraph, int, int]:
    raw_func = _unwrap(kernel_func.func)
    graph_name = name or _mangle_name(raw_func.__name__)
    graph = _trace(kernel_func, *inputs, output=output, dynamic_shape=dynamic_shape)
    module, context = build_mlir_module(graph, graph_name)
    return module, context, graph, len(graph.input_ids), len(graph.output_ref_ids)


def _compile_module(
    module: object,
    context: object,
    graph: TraceGraph,
    input_count: int,
    output_count: int,
    options: CompileOptions | None = None,
    tile_sizes: tuple[int, ...] = (),
) -> CompiledKernel:
    if options is not None and tile_sizes:
        raise TypeError("Pass either options= or tile_sizes, not both")
    if any(
        node.kind == NodeKind.MATMUL and node.tensor_info.dynamic_shape
        for node in graph.nodes
    ):
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
    return CompiledKernel(program, module, graph, input_count, output_count, context)


def compile(
    kernel_func: KernelFunction,
    *inputs: object,
    output: object | tuple[object, ...],
    options: CompileOptions | None = None,
    tile_sizes: tuple[int, ...] = (),
    name: str | None = None,
    dynamic_shape: bool = False,
) -> CompiledKernel:
    if not isinstance(kernel_func, KernelFunction):
        raise TypeError("tir.compile expects a function decorated with @tir.kernel")
    module, context, graph, input_count, output_count = _build_module(
        kernel_func,
        *inputs,
        output=output,
        name=name,
        dynamic_shape=dynamic_shape,
    )
    return _compile_module(
        module,
        context,
        graph,
        input_count,
        output_count,
        options=options,
        tile_sizes=tile_sizes,
    )


def _tensor_info_from_value(
    value: object, name: str, *, dynamic_shape: bool = False
) -> TensorInfo:
    spec = tensor_spec_from_value(value)
    return TensorInfo(
        name=name,
        dtype=spec.dtype,
        shape=spec.shape,
        stride=spec.stride,
        dynamic_shape=dynamic_shape,
    )


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


def _register_output_refs(
    graph: TraceGraph,
    output: object | tuple[object, ...],
    *,
    dynamic_shape: bool = False,
) -> None:
    if isinstance(output, tuple):
        if not output:
            raise TypeError("TensorIR DSL compile output tuple must not be empty")
        for i, value in enumerate(output):
            graph.add_output_ref(
                _tensor_info_from_value(
                    value,
                    f"output_{i}",
                    dynamic_shape=dynamic_shape,
                )
            )
        return
    graph.add_output_ref(
        _tensor_info_from_value(output, "output", dynamic_shape=dynamic_shape)
    )


def _unwrap(func):
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def _mangle_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name)
    return f"kernel_{cleaned}"
