# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os

import pytest
import torch

from nv_tensor_ir import dsl as tir
from dsl_reference import evaluate_trace_reference
from nv_tensor_ir.dsl.dsl import KernelFunction


def _require_runtime_launch() -> None:
    if os.environ.get("TENSOR_IR_SKIP_RUNTIME_LAUNCH") == "1":
        pytest.skip("runtime launch requires a PTX JIT compiler")


def _compile_run_assert(
    kernel_func: KernelFunction,
    *inputs: torch.Tensor,
    output: torch.Tensor,
    name: str,
    tile_sizes: tuple[int, ...] | None = None,
    options: tir.CompileOptions | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    mlir_probes: tuple[str, ...] = (),
):
    if options is not None:
        compile_kwargs = {"options": options}
    elif tile_sizes is not None:
        compile_kwargs = {"tile_sizes": tile_sizes}
    else:
        compile_kwargs = {}
    compiled = tir.compile(
        kernel_func,
        *inputs,
        output=output,
        name=name,
        **compile_kwargs,
    )
    mlir = _module_text(compiled)
    for probe in mlir_probes:
        assert probe in mlir

    _run_assert_trace(compiled, *inputs, output=output, rtol=rtol, atol=atol)
    return compiled


def _run_assert_trace(
    compiled: tir.CompiledKernel,
    *inputs: torch.Tensor,
    output: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    _require_runtime_launch()
    compiled.run(*inputs, output=output)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        evaluate_trace_reference(compiled.graph, *inputs, device=output.device),
        rtol=rtol,
        atol=atol,
    )


def _assert_compile_error(
    error_type: type[Exception],
    match: str,
    kernel_func: KernelFunction,
    *inputs: torch.Tensor,
    output: object,
) -> None:
    with pytest.raises(error_type, match=match):
        tir.compile(kernel_func, *inputs, output=output)


def _module_text(compiled) -> str:
    assert compiled.module is not None
    return str(compiled.module.operation)


def _layout_propagation_options(tile_sizes: tuple[int, ...]) -> tir.CompileOptions:
    options = tir.CompileOptions()
    options.tile_sizes = list(tile_sizes)
    options.codegen_strategy = tir.CodegenStrategy.LayoutPropagation
    return options


def _layout_propagation_auto_options() -> tir.CompileOptions:
    options = tir.CompileOptions()
    options.codegen_strategy = tir.CodegenStrategy.LayoutPropagation
    return options
