# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON -m pytest --capture=no -vv %s
# REQUIRES: cuda-sm100

"""Explicit exp policies and a basic-math gated RMSNorm against FP64 references."""

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


@pytest.mark.parametrize("mode", [None, "full", "approx"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_exp_accuracy(mode, dtype):
    @tir.kernel
    def kernel(value):
        return tir.exp(value) if mode is None else tir.exp(value, rounding_mode=mode)

    # Finite range avoids overflow in half; special values check the endpoint
    # contract separately without imposing bit-identical exp implementations.
    value = torch.linspace(-8.0, 8.0, 128, device="cuda", dtype=dtype)
    value[:4] = torch.tensor(
        [-float("inf"), float("inf"), float("nan"), -0.0], device="cuda", dtype=dtype
    )
    expected = torch.exp(value.double()).to(dtype)
    (actual,) = run_kernel(kernel, (value,), (expected,), [128])
    tolerance = {torch.float16: 0.002, torch.bfloat16: 0.016, torch.float32: 3e-6}[
        dtype
    ]
    torch.testing.assert_close(
        actual, expected, atol=0.0, rtol=tolerance, equal_nan=True
    )


@pytest.mark.parametrize("mode", ["full", "approx"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_basic_math_gated_rmsnorm(mode, dtype):
    @tir.kernel
    def kernel(value, gate, weight):
        shape = value.tensor_info.shape
        x = tir.convert(value, dtype=tir.DataType.F32)
        z = tir.convert(gate, dtype=tir.DataType.F32)
        one = tir.constant(1.0, shape=shape, dtype=tir.DataType.F32)
        sigmoid = one / (one + tir.exp(-z, rounding_mode=mode))
        gated = x * (z * sigmoid)
        squared_sum = tir.reduce(gated * gated, dimensions=(1,))
        count = tir.constant(128.0, shape=(4, 1), dtype=tir.DataType.F32)
        epsilon = tir.constant(1e-5, shape=(4, 1), dtype=tir.DataType.F32)
        inverse_rms = tir.rsqrt(squared_sum / count + epsilon)
        normalized = gated * tir.broadcast(inverse_rms, shape=shape)
        scale = tir.broadcast(tir.convert(weight, dtype=tir.DataType.F32), shape=shape)
        return tir.convert(
            normalized * scale, dtype=value.tensor_info.dtype
        ), inverse_rms

    value = torch.linspace(-2.0, 2.0, 512, device="cuda", dtype=dtype).reshape(4, 128)
    gate = torch.linspace(-80.0, 80.0, 128, device="cuda", dtype=dtype).repeat(4, 1)
    weight = torch.linspace(0.5, 1.5, 128, device="cuda", dtype=dtype).reshape(1, 128)
    gated = value.double() * torch.nn.functional.silu(gate.double())
    inverse_rms = torch.rsqrt(gated.square().mean(dim=1, keepdim=True) + 1e-5)
    expected = ((gated * inverse_rms * weight.double()).to(dtype), inverse_rms.float())
    actual = run_kernel(kernel, (value, gate, weight), expected, [1, 128])
    torch.testing.assert_close(
        actual[0],
        expected[0],
        atol=0.02 if dtype == torch.bfloat16 else 2e-5,
        rtol=0.02 if dtype == torch.bfloat16 else 2e-5,
    )
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-6)
