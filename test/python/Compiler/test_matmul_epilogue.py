# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON -m pytest --capture=no -vv %s
# REQUIRES: cuda-gpu

from __future__ import annotations

import pytest
import torch

from nv_tensor_ir._mlir import ir
from nv_tensor_ir._mlir.dialects import nv_tensor_ir
from dsl_test_utils import _require_runtime_launch


M, K, N = 8, 32, 16
A_SHAPE = (M, K)
B_SHAPE = (K, N)
C_SHAPE = (M, N)
A_STRIDES = (K, 1)
B_STRIDES = (N, 1)
C_STRIDES = (N, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "name": "matmul_neg",
                "dtype": torch.float32,
                "mlir_dtype": "f32",
                "epilogue": "%out = neg %m : tensor<8x16xf32>",
                "extra_args": (),
                "make_extra_inputs": lambda: (),
                "reference": lambda m, extras: -m,
                "tolerance": {"rtol": 1e-4, "atol": 1e-4},
            },
            id="neg",
        ),
        pytest.param(
            {
                "name": "matmul_scale_add",
                "dtype": torch.float32,
                "mlir_dtype": "f32",
                "epilogue": """
  %scaled = mul %m, %scale : tensor<8x16xf32>
  %out = add %scaled, %bias : tensor<8x16xf32>
""",
                "extra_args": (("scale", "f32"), ("bias", "f32")),
                "make_extra_inputs": lambda: (
                    torch.randn(C_SHAPE, device="cuda"),
                    torch.randn(C_SHAPE, device="cuda"),
                ),
                "reference": lambda m, extras: m * extras[0] + extras[1],
                "tolerance": {"rtol": 1e-4, "atol": 1e-4},
            },
            id="scale_add",
        ),
        pytest.param(
            {
                "name": "matmul_f16_bias_add",
                "dtype": torch.float16,
                "mlir_dtype": "f16",
                "epilogue": "%out = add %m, %bias : tensor<8x16xf16>",
                "extra_args": (("bias", "f16"),),
                "make_extra_inputs": lambda: (
                    torch.randn(C_SHAPE, device="cuda", dtype=torch.float16),
                ),
                "reference": lambda m, extras: m + extras[0],
                "tolerance": {"rtol": 2e-2, "atol": 1e-2},
            },
            id="f16_bias_add",
        ),
    ],
)
def test_matmul_epilogue(case: dict[str, object]) -> None:
    torch.manual_seed(0)
    dtype = case["dtype"]
    a = torch.randn(A_SHAPE, device="cuda", dtype=dtype)
    b = torch.randn(B_SHAPE, device="cuda", dtype=dtype)
    output = torch.empty(C_SHAPE, device="cuda", dtype=dtype)
    extra_inputs = case["make_extra_inputs"]()

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        nv_tensor_ir.register_dialect(ctx, load=True)
        module = ir.Module.parse(
            _matmul_epilogue_mlir(
                case["name"],
                case["mlir_dtype"],
                case["epilogue"],
                case["extra_args"],
            )
        )
        options = nv_tensor_ir.CompileOptions()
        options.tile_sizes = [2, 2]
        assert nv_tensor_ir.can_compile(module, options=options)
        with nv_tensor_ir.compile(module, options=options) as program:
            _require_runtime_launch()
            program.launch(a, b, *extra_inputs, output)

    torch.cuda.synchronize()
    expected = case["reference"](torch.matmul(a, b), extra_inputs)
    torch.testing.assert_close(output, expected, **case["tolerance"])


def _matmul_epilogue_mlir(
    name: str,
    mlir_dtype: str,
    epilogue: str,
    extra_args: tuple[tuple[str, str], ...],
) -> str:
    extra_arg_text = "".join(
        f",\n    %{arg_name}: {_tensor_type(arg_dtype, C_SHAPE)} "
        f'{{nv_tensor_ir.stride = "{_shape(C_STRIDES)}"}}'
        for arg_name, arg_dtype in extra_args
    )
    return f"""
module {{
  nv_tensor_ir.graph @{name}(
    %a: {_tensor_type(mlir_dtype, A_SHAPE)} {{nv_tensor_ir.stride = "{_shape(A_STRIDES)}"}},
    %b: {_tensor_type(mlir_dtype, B_SHAPE)} {{nv_tensor_ir.stride = "{_shape(B_STRIDES)}"}}{extra_arg_text}
  ) -> ({_tensor_type(mlir_dtype, C_SHAPE)} {{nv_tensor_ir.stride = "{_shape(C_STRIDES)}"}}) {{
  %m = "nv_tensor_ir.matmul"(%a, %b) : ({_tensor_type(mlir_dtype, A_SHAPE)}, {_tensor_type(mlir_dtype, B_SHAPE)}) -> {_tensor_type(mlir_dtype, C_SHAPE)}
{epilogue}
  results %out : {_tensor_type(mlir_dtype, C_SHAPE)}
  }}
}}
"""


def _tensor_type(dtype: str, shape: tuple[int, ...]) -> str:
    dimensions = "x".join(str(value) for value in shape)
    return f"tensor<{dimensions}x{dtype}>"


def _shape(values: tuple[int, ...]) -> str:
    return f"({','.join(str(value) for value in values)})"
