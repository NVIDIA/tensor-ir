# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s
# REQUIRES: cuda-sm100

"""Tests expm1 and log1p against FP64 references."""

import pytest
import torch
from nv_tensor_ir import dsl as tir
from nv_tensor_ir import runtime
from nv_tensor_ir.dsl.module_builder import build_mlir_module

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="These CUDA Tile regression configurations target SM100.",
)


def run_kernel(kernel, inputs, expected, tiles):
    outputs = tuple(torch.full_like(value, float("nan")) for value in expected)
    graph = tir.trace(kernel, *inputs, output=outputs)
    module, context = build_mlir_module(graph, "exp_policy_gpu")
    with (
        context,
        runtime.compile(module, compute_capability=100, tile_sizes=tiles) as program,
    ):
        with program.load() as kernel:
            kernel.launch(*inputs, *outputs, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
    return outputs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_log1p_accuracy(dtype):
    @tir.kernel
    def kernel(value):
        return tir.log1p(value)

    value = torch.linspace(-8.0, 8.0, 128, device="cuda", dtype=dtype)
    value[:4] = torch.tensor(
        [-float("inf"), float("inf"), float("nan"), -0.0], device="cuda", dtype=dtype
    )
    expected = torch.log1p(value.double()).to(dtype)
    (actual,) = run_kernel(kernel, (value,), (expected,), [128])
    tolerance = {torch.float16: 0.002, torch.bfloat16: 0.016, torch.float32: 3e-6}[
        dtype
    ]
    torch.testing.assert_close(
        actual, expected, atol=0.0, rtol=tolerance, equal_nan=True
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_expm1_accuracy(dtype):
    @tir.kernel
    def kernel(value):
        return tir.expm1(value)

    value = torch.linspace(-8.0, 8.0, 128, device="cuda", dtype=dtype)
    value[:4] = torch.tensor(
        [-float("inf"), float("inf"), float("nan"), -0.0], device="cuda", dtype=dtype
    )
    expected = torch.expm1(value.double()).to(dtype)
    (actual,) = run_kernel(kernel, (value,), (expected,), [128])
    tolerance = {torch.float16: 0.002, torch.bfloat16: 0.016, torch.float32: 3e-6}[
        dtype
    ]
    torch.testing.assert_close(
        actual, expected, atol=0.0, rtol=tolerance, equal_nan=True
    )
