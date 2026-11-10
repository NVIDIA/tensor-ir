# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s -x | FileCheck %s


import pytest
from nv_tensor_ir import dsl as tir
from nv_tensor_ir.dsl.module_builder import build_mlir_module


@pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div", "mod", "pow"])
def test_tensor_scalar_binary_ops(op_name):
    operation = {
        "add": lambda lhs, rhs: lhs + rhs,
        "sub": lambda lhs, rhs: lhs - rhs,
        "mul": lambda lhs, rhs: lhs * rhs,
        "div": lambda lhs, rhs: lhs / rhs,
        "mod": lambda lhs, rhs: lhs % rhs,
        "pow": lambda lhs, rhs: lhs**rhs,
    }.get(op_name)

    @tir.kernel
    def _tir_binary_ops(lhs, rhs):
        return operation(lhs, rhs)

    tensor = tir.TensorSpec((8, 8), dtype=tir.DataType.F16)
    output = tir.TensorSpec((8, 8), dtype=tir.DataType.F16)

    scalar = 3.14
    graph = tir.trace(_tir_binary_ops, tensor, scalar, output=output)
    module, _ = build_mlir_module(graph, "_tir_binary_ops")
    # CHECK-LABEL: @_tir_binary_ops
    # CHECK-SAME: %[[in0:.*]]: tensor<8x8xf16>
    # CHECK-SAME: %[[in1:.*]]: f32
    # CHECK: %[[splat:.*]] = splat %[[in1]] : tensor<8x8xf32>
    # CHECK: %[[convert:.*]] = convert %[[splat]] : tensor<8x8xf32> -> tensor<8x8xf16>
    # CHECK: %[[in0]], %[[convert]] : tensor<8x8xf16>
    print(module)

    graph = tir.trace(_tir_binary_ops, scalar, tensor, output=output)
    module, _ = build_mlir_module(graph, "_tir_binary_reflected_ops")
    # CHECK-LABEL: @_tir_binary_reflected_ops
    # CHECK-SAME: %[[in0:.*]]: f32
    # CHECK-SAME: %[[in1:.*]]: tensor<8x8xf16>
    # CHECK: %[[splat:.*]] = splat %[[in0]] : tensor<8x8xf32>
    # CHECK: %[[convert:.*]] = convert %[[splat]] : tensor<8x8xf32> -> tensor<8x8xf16>
    # CHECK: %[[convert]], %[[in1]] : tensor<8x8xf16>
    print(module)
