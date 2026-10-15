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


TEST_SIZE = 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
@pytest.mark.parametrize(
    "op_name,num_inputs,reference",
    [
        ("add", 2, lambda a, b: a + b),
        ("relu_fwd", 1, torch.relu),
    ],
)
@pytest.mark.parametrize("analyze", [False, True])
def test_pointwise_op_from_python_bindings(
    op_name: str, num_inputs: int, reference, analyze: bool
) -> None:
    torch.manual_seed(0)
    inputs = [
        torch.randn((TEST_SIZE,), device="cuda", dtype=torch.float32)
        for _ in range(num_inputs)
    ]
    output = torch.empty_like(inputs[0])

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        nv_tensor_ir.register_dialect(ctx, load=True)
        module = ir.Module.parse(_pointwise_mlir(op_name, num_inputs))
        tile_sizes = [TEST_SIZE // 2]
        if analyze:
            candidates = nv_tensor_ir.analyze_module(module, max_tile_candidates=3)
            assert candidates
            tile_sizes = candidates[0]
        assert nv_tensor_ir.can_compile(module, tile_sizes=tile_sizes)
        with nv_tensor_ir.compile(
            module,
            tile_sizes=tile_sizes,
        ) as artifact:
            assert artifact.get_bytecode()
            _require_runtime_launch()
            with artifact.load() as first, artifact.load() as second:
                artifact.destroy()
                first.launch(*inputs, output)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output, reference(*inputs), rtol=1e-3, atol=1e-3
                )
                first.destroy()
                output.zero_()
                second.launch(*inputs, output)

    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference(*inputs), rtol=1e-3, atol=1e-3)


def _pointwise_mlir(op_name: str, num_inputs: int) -> str:
    tensor_type = f"tensor<{TEST_SIZE}xf32>"
    args = ",\n    ".join(
        f'%arg{i}: {tensor_type} {{nv_tensor_ir.stride = "(1)"}}'
        for i in range(num_inputs)
    )
    operands = ", ".join(f"%arg{i}" for i in range(num_inputs))
    return f"""
module {{
  nv_tensor_ir.graph @test_{op_name}(
    {args}
  ) -> ({tensor_type} {{nv_tensor_ir.stride = "(1)"}}) {{
    %out = {op_name} {operands} : {tensor_type}
    results %out : {tensor_type}
  }}
}}
"""
