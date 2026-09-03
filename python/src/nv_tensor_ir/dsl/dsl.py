# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import copy
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass

from .module_builder import TensorInfoOverrides, build_mlir_module
from .profile import ProfilingConfig, profile_launches
from .tensor_spec import tensor_spec_from_value
from .tracing import TensorInfo, TraceGraph, _trace_graph_context

from nv_tensor_ir._mlir.dialects import nv_tensor_ir

CompileOptions = nv_tensor_ir.CompileOptions
CodegenStrategy = nv_tensor_ir.CodegenStrategy
BytecodeVersion = nv_tensor_ir.BytecodeVersion
CudaTileArtifactKind = nv_tensor_ir.CudaTileArtifactKind


class ProgramCache:
    """Explicit cache for compiled TensorIR programs.

    Pass an instance to :func:`compile` or :func:`compile_traced` to reuse
    programs compiled by the selected backend.
    """

    def __init__(self) -> None:
        self._programs: dict[str, nv_tensor_ir.Program] = {}

    def flush(self) -> None:
        """Remove all cached programs without invalidating returned programs."""
        self._programs.clear()

    def _get(self, key: str) -> nv_tensor_ir.Program | None:
        program = self._programs.get(key)
        return copy.copy(program) if program is not None else None

    def _put(self, key: str, program: nv_tensor_ir.Program) -> None:
        self._programs[key] = copy.copy(program)


class CompiledKernel:
    """Common execution interface returned by TensorIR DSL backends."""

    def run(
        self,
        *inputs: object,
        output: object | tuple[object, ...],
    ) -> None:
        raise NotImplementedError("CompiledKernel must implement run")

    def run_profile(
        self,
        *inputs: object,
        output: object | tuple[object, ...],
        warmup: int = 1,
        iterations: int = 10,
    ) -> None:
        """Run this kernel under CUPTI and print kernel timings."""
        config = ProfilingConfig(warmup=warmup, iterations=iterations)

        def launch() -> None:
            self.run(*inputs, output=output)

        profile_launches(launch, config)


@dataclass(frozen=True)
class KernelFunction:
    func: Callable[..., object]

    @property
    def __name__(self) -> str:
        return self.func.__name__


@dataclass(frozen=True)
class BackendImpl:
    """Backend hooks invoked around the shared MLIR module builder."""

    prepare_tensor_infos: Callable[..., TensorInfoOverrides]
    calculate_cache_key: Callable[..., str]
    compile: Callable[..., CompiledKernel]


_BACKEND_IMPLS: dict[str, BackendImpl] = {}


def _register_backend_impl(backend: str, impl: BackendImpl) -> None:
    """Register tensor metadata preparation and compilation for a backend."""
    if not isinstance(backend, str) or not backend:
        raise TypeError("TensorIR DSL backend names must be non-empty strings")
    if backend in _BACKEND_IMPLS:
        raise ValueError(f"TensorIR DSL backend '{backend}' is already registered")
    _BACKEND_IMPLS[backend] = impl


def _get_backend_impl(backend: str) -> BackendImpl:
    """Return the implementation registered for ``backend``."""
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    try:
        return _BACKEND_IMPLS[backend]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKEND_IMPLS))
        raise ValueError(
            f"Unsupported TensorIR DSL backend '{backend}'; available: {available}"
        ) from exc


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
) -> TraceGraph:
    raw_func = _unwrap(kernel_func.func)
    signature = inspect.signature(raw_func, eval_str=True)
    graph = TraceGraph()
    _register_output_refs(graph, output)
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
    return graph


def trace(
    kernel_func: KernelFunction,
    *inputs: object,
    output: object | tuple[object, ...],
) -> TraceGraph:
    """Trace a decorated TensorIR DSL kernel into a concrete graph."""
    if not isinstance(kernel_func, KernelFunction):
        raise TypeError("tir.trace expects a function decorated with @tir.kernel")
    return _trace(kernel_func, *inputs, output=output)


def calculate_cache_key(
    module: object,
    options: CompileOptions | None = None,
    *,
    tile_sizes: tuple[int, ...] = (),
    backend: str = "cuda_tile",
) -> str:
    """Return the cache key for an MLIR module and selected backend options."""
    return _get_backend_impl(backend).calculate_cache_key(
        module,
        options=options,
        tile_sizes=tile_sizes,
    )


def compile_traced(
    graph: TraceGraph,
    *,
    options: CompileOptions | None = None,
    tile_sizes: tuple[int, ...] = (),
    name: str = "kernel_traced",
    dynamic_shape: bool = False,
    backend: str = "cuda_tile",
    program_cache: ProgramCache | None = None,
) -> CompiledKernel:
    """Compile a concrete TensorIR trace with the selected backend."""
    if not isinstance(graph, TraceGraph):
        raise TypeError("tir.compile_traced expects a graph returned by tir.trace")
    if program_cache is not None and not isinstance(program_cache, ProgramCache):
        raise TypeError("program_cache must be a ProgramCache or None")
    backend_impl = _get_backend_impl(backend)
    tensor_info_overrides = backend_impl.prepare_tensor_infos(
        graph,
        dynamic_shape=dynamic_shape,
    )
    module, context = build_mlir_module(
        graph,
        name,
        tensor_info_overrides=tensor_info_overrides,
    )
    return backend_impl.compile(
        module,
        context,
        graph,
        options=options,
        tile_sizes=tile_sizes,
        dynamic_shape=dynamic_shape,
        program_cache=program_cache,
    )


def compile(
    kernel_func: KernelFunction,
    *inputs: object,
    output: object | tuple[object, ...],
    options: CompileOptions | None = None,
    tile_sizes: tuple[int, ...] = (),
    name: str | None = None,
    dynamic_shape: bool = False,
    backend: str = "cuda_tile",
    program_cache: ProgramCache | None = None,
) -> CompiledKernel:
    """Compile a TensorIR DSL kernel with the selected registered backend."""
    if not isinstance(kernel_func, KernelFunction):
        raise TypeError("tir.compile expects a function decorated with @tir.kernel")
    if program_cache is not None and not isinstance(program_cache, ProgramCache):
        raise TypeError("program_cache must be a ProgramCache or None")
    raw_func = _unwrap(kernel_func.func)
    graph = trace(kernel_func, *inputs, output=output)
    graph_name = name or _mangle_name(raw_func.__name__)
    return compile_traced(
        graph,
        options=options,
        tile_sizes=tile_sizes,
        name=graph_name,
        dynamic_shape=dynamic_shape,
        backend=backend,
        program_cache=program_cache,
    )


def _tensor_info_from_value(value: object, name: str) -> TensorInfo:
    spec = tensor_spec_from_value(value)
    return TensorInfo(
        name=name,
        dtype=spec.dtype,
        shape=spec.shape,
        stride=spec.stride,
    )


def _register_output_refs(
    graph: TraceGraph,
    output: object | tuple[object, ...],
) -> None:
    if isinstance(output, tuple):
        if not output:
            raise TypeError("TensorIR DSL compile output tuple must not be empty")
        for i, value in enumerate(output):
            graph.add_output_ref(
                _tensor_info_from_value(
                    value,
                    f"output_{i}",
                )
            )
        return
    graph.add_output_ref(_tensor_info_from_value(output, "output"))


def _unwrap(func):
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def _mangle_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name)
    return f"kernel_{cleaned}"
