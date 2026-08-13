# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON -m pytest --capture=no -vv %s
# REQUIRES: cuda-gpu

from __future__ import annotations

import pytest
import torch

from nv_tensor_ir import dsl as tir
from dsl_reference import evaluate_trace_reference
from dsl_test_utils import (
    _compile_run_assert,
    _layout_propagation_auto_options,
    _layout_propagation_options,
    _module_text,
    _run_assert_trace,
)
from nv_tensor_ir.dsl.dsl import KernelFunction, _trace
from nv_tensor_ir.dsl.tracing import NodeKind

_FLOAT_CONVERT_TYPES = (
    (tir.DataType.F32, "f32"),
    (tir.DataType.F32, "f32"),
    (tir.DataType.F16, "f16"),
    (tir.DataType.F32, "f32"),
    (tir.DataType.BF16, "bf16"),
    (tir.DataType.F32, "f32"),
    (tir.DataType.F64, "f64"),
    (tir.DataType.F16, "f16"),
    (tir.DataType.F16, "f16"),
    (tir.DataType.BF16, "bf16"),
    (tir.DataType.F16, "f16"),
    (tir.DataType.F64, "f64"),
    (tir.DataType.BF16, "bf16"),
    (tir.DataType.BF16, "bf16"),
    (tir.DataType.F64, "f64"),
    (tir.DataType.F64, "f64"),
    (tir.DataType.F32, "f32"),
)

_INT_FLOAT_CONVERT_TYPES = (
    (tir.DataType.SI32, "si32"),
    (tir.DataType.SI32, "si32"),
    (tir.DataType.F32, "f32"),
    (tir.DataType.SI32, "si32"),
    (tir.DataType.F16, "f16"),
    (tir.DataType.SI32, "si32"),
    (tir.DataType.BF16, "bf16"),
    (tir.DataType.SI32, "si32"),
    (tir.DataType.F64, "f64"),
    (tir.DataType.SI32, "si32"),
)

_SIGNED_INTEGER_TYPES = (
    (torch.int8, "si8"),
    (torch.int16, "si16"),
    (torch.int32, "si32"),
    (torch.int64, "si64"),
)

_UNSIGNED_INTEGER_TYPES = (
    (torch.uint8, "ui8"),
    (torch.uint16, "ui16"),
    (torch.uint32, "ui32"),
    (torch.uint64, "ui64"),
)

_TORCH_DTYPE_TO_TENSOR_IR_DTYPE = {
    torch.int8: tir.DataType.SI8,
    torch.int16: tir.DataType.SI16,
    torch.int32: tir.DataType.SI32,
    torch.int64: tir.DataType.SI64,
    torch.uint8: tir.DataType.UI8,
    torch.uint16: tir.DataType.UI16,
    torch.uint32: tir.DataType.UI32,
    torch.uint64: tir.DataType.UI64,
}


def _convert_chain(value, dtype_chain):
    for dtype, _ in dtype_chain[1:]:
        value = tir.convert(value, dtype=dtype)
    return value


def _tensor_ir_type_pairs(dtype_chain):
    return zip(dtype_chain, dtype_chain[1:])


@tir.kernel
def add_kernel(a, b):
    return a + b


@tir.kernel
def sub_kernel(a, b):
    return a - b


@tir.kernel
def mul_kernel(a, b):
    return a * b


@tir.kernel
def div_kernel(a, b):
    return a / b


@tir.kernel
def rem_kernel(a, b):
    return tir.rem(a, b)


@tir.kernel
def extended_float_ops_kernel(a, b):
    sqrt_value = tir.sqrt(a)
    rsqrt_value = tir.rsqrt(a)
    sigmoid_value = tir.sigmoid(a)
    tanh_value = tir.tanh(a)
    max_value = tir.max(sqrt_value, rsqrt_value)
    min_value = tir.min(sigmoid_value, tanh_value)
    return (
        tir.relu(a)
        + tir.abs(a)
        + tir.neg(a)
        + tir.ceil(a)
        + tir.floor(a)
        + tir.exp(a)
        + tir.log(b)
        + tir.sin(a)
        + tir.cos(a)
        + tir.tan(a)
        + tir.reciprocal(b)
        + sigmoid_value
        + tanh_value
        + tir.gelu_approx_tanh(a)
        + tir.softplus(a, beta=2.0)
        + tir.swish(a, beta=1.5)
        + tir.elu(a, beta=0.25)
        + tir.mod(b, a)
        + tir.pow(b, a)
        + tir.add_square(max_value, min_value)
        + tir.atan2(a, b)
    )


@tir.kernel
def extended_operator_overload_kernel(a, b):
    return (-a) + (b % a) + (b**a)


@tir.kernel
def mixed_dtype_pow_kernel(base, exponent):
    return tir.pow(base, exponent)


@tir.kernel
def mixed_dtype_pow_operator_kernel(base, exponent):
    return base**exponent


@tir.kernel
def convert_kernel(a):
    return tir.convert(a, dtype=tir.DataType.F16)


@tir.kernel
def convert_supported_float_types_kernel(a):
    return _convert_chain(a, _FLOAT_CONVERT_TYPES)


@tir.kernel
def convert_int_float_kernel(a):
    return _convert_chain(a, _INT_FLOAT_CONVERT_TYPES)


@tir.kernel
def int8_add_kernel(a, b):
    return a + b


@tir.kernel
def integer_add_kernel(a, b):
    return a + b


@tir.kernel
def convert_int8_kernel(a):
    i8_to_i32 = tir.convert(a, dtype=tir.DataType.SI32)
    return tir.convert(i8_to_i32, dtype=tir.DataType.SI8)


def _make_convert_to_kernel(dtype: tir.DataType) -> KernelFunction:
    @tir.kernel
    def convert_to_kernel(a):
        return tir.convert(a, dtype=dtype)

    return convert_to_kernel


@tir.kernel
def integer_full_like_kernel(a):
    return tir.full_like(a, value=1)


@tir.kernel
def constant_kernel(a):
    return tir.constant(2.5, shape=(8, 8), dtype=tir.DataType.F32)


@tir.kernel
def splat_kernel(a):
    return tir.splat(2.5, shape=(8, 8), dtype=tir.DataType.F32)


@tir.kernel
def splat_cmp_kernel(a, b):
    bias = tir.full_like(a, value=0.25)
    fallback = tir.full_like(a, value=1.0)
    shifted = a + bias
    mask = tir.cmp(shifted, b, predicate="ole")
    mask = tir.logical_and(mask, a <= b)
    mask = tir.logical_or(mask, b <= fallback)
    return tir.where(mask, shifted, fallback)


@tir.kernel
def logical_not_kernel(a, b):
    return tir.logical_not(a <= b)


@tir.kernel
def int_cmp_kernel(a, b):
    return tir.cmp(a, b, predicate="ge")


@tir.kernel
def int_operator_cmp_kernel(a, b):
    return a >= b


@tir.kernel
def reshape_kernel(a):
    return tir.reshape(a, shape=(2, 8))


@tir.kernel
def transpose_kernel(a):
    return tir.transpose(a, permutation=(1, 0))


@tir.kernel
def broadcast_kernel(a, b):
    return tir.broadcast(a, shape=(4, 8)) + b


@tir.kernel
def slice_kernel(a):
    return tir.slice(a, starts=(2, 1), limits=(6, 5), strides=(1, 1))


@tir.kernel
def concatenate_kernel(a, b):
    return tir.concatenate((a, b), dimension=1)


@tir.kernel
def iota_like_kernel(a):
    return a + tir.iota_like(a, dimension=1)


@tir.kernel
def iota_kernel():
    return tir.iota(shape=(4, 8), dtype=tir.DataType.F32, dimension=1)


def _make_reduce_kernel(dimensions: tuple[int, ...], mode: str) -> KernelFunction:
    @tir.kernel
    def reduce_kernel(a):
        return tir.reduce(a, dimensions=dimensions, mode=mode)

    return reduce_kernel


def test_float64_input_output_type() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float64)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float64)
    output = torch.empty_like(a)

    _compile_run_assert(
        add_kernel,
        a,
        b,
        output=output,
        name="float64_input_output",
        options=_layout_propagation_auto_options(),
        mlir_probes=("tensor<8x8xf64>",),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "kernel", [mixed_dtype_pow_kernel, mixed_dtype_pow_operator_kernel]
)
def test_mixed_dtype_pow(kernel: KernelFunction) -> None:
    base = torch.rand((8, 8), device="cuda", dtype=torch.float32) + 0.5
    exponent = torch.randint(0, 4, (8, 8), device="cuda", dtype=torch.int32)
    output = torch.empty_like(base)

    compiled = tir.compile(
        kernel,
        base,
        exponent,
        output=output,
        name=kernel.__name__,
        options=_layout_propagation_auto_options(),
    )
    mlir = _module_text(compiled)
    assert ": (tensor<8x8xf32>, tensor<8x8xsi32>) -> tensor<8x8xf32>" in mlir
    _run_assert_trace(compiled, base, exponent, output=output)


def test_trace_frontend_op_mapping() -> None:
    @tir.kernel
    def frontend_mapping_kernel(a, b):
        # Operator overloads and explicit binary helpers.
        add_value = a + b
        sub_value = a - b
        mul_value = a * b
        div_value = a / b
        mod_value = a % b
        rem_value = tir.rem(a, b)
        pow_value = a**b
        matmul_value = tir.matmul(a, b)

        # Unary and activation helpers.
        neg_value = -add_value
        relu_value = tir.relu(add_value)
        abs_value = tir.abs(sub_value)
        ceil_value = tir.ceil(add_value)
        cos_value = tir.cos(add_value)
        exp_value = tir.exp(add_value)
        floor_value = tir.floor(add_value)
        log_value = tir.log(add_value)
        convert_value = tir.convert(add_value, dtype=tir.DataType.F16)
        reciprocal_value = tir.reciprocal(add_value)
        sqrt_value = tir.sqrt(abs_value)
        rsqrt_value = tir.rsqrt(abs_value)
        sin_value = tir.sin(add_value)
        tan_value = tir.tan(add_value)
        sigmoid_value = tir.sigmoid(mul_value)
        tanh_value = tir.tanh(div_value)
        gelu_value = tir.gelu_approx_tanh(add_value)

        # Type conversion, constants, movement, iota, and reduction helpers.
        splat_value = tir.full_like(a, value=1.0)
        cmp_value = tir.cmp(add_value, splat_value, predicate="oeq")
        sliced_value = tir.slice(a, starts=(0, 0), limits=(8, 1))
        # Trace-only values force coverage for ops that do not feed the result.
        _broadcast_value = tir.broadcast(sliced_value, shape=(8, 8))
        _reshape_value = tir.reshape(a, shape=(4, 16))
        _transpose_value = tir.transpose(a, permutation=(1, 0))
        _concatenate_value = tir.concatenate((a, b), dimension=1)
        iota_value = tir.iota_like(a, dimension=1)
        reduce_value = tir.reduce(a, dimensions=(1,), mode="add")
        _broadcast_reduce_value = tir.broadcast(reduce_value, shape=(8, 8))

        # Selection and the final expression keep representative values live.
        softplus_value = tir.softplus(add_value, beta=2.0)
        swish_value = tir.swish(add_value, beta=1.5)
        elu_value = tir.elu(add_value, beta=0.25)
        max_value = tir.max(sqrt_value, rsqrt_value)
        min_value = tir.min(sigmoid_value, tanh_value)
        add_square_value = tir.add_square(max_value, min_value)
        atan2_value = tir.atan2(a, b)
        mask = tir.logical_and(a > b, b > a)
        mask = tir.logical_or(mask, a <= b)
        _not_mask = tir.logical_not(mask)
        selected = tir.where(mask, add_square_value, gelu_value)
        selected = tir.where(cmp_value, selected, splat_value)
        selected = (
            selected
            + relu_value
            + softplus_value
            + swish_value
            + elu_value
            + iota_value
            + atan2_value
        )
        return tir.where(
            a <= b,
            selected + mod_value + rem_value + pow_value + neg_value + ceil_value,
            matmul_value
            + cos_value
            + exp_value
            + floor_value
            + log_value
            + tir.convert(convert_value, dtype=tir.DataType.F32)
            + reciprocal_value
            + sin_value
            + tan_value
            + gelu_value,
        )

    a = torch.randn((8, 8), dtype=torch.float32)
    b = torch.randn((8, 8), dtype=torch.float32)
    graph = _trace(frontend_mapping_kernel, a, b, output=torch.empty_like(a))
    observed = [
        (node.kind, node.op_name)
        for node in graph.nodes
        if node.kind not in (NodeKind.INPUT, NodeKind.OUTPUT_REF)
    ]
    assert set(observed) >= {
        (NodeKind.BINARY, "add"),
        (NodeKind.BINARY, "sub"),
        (NodeKind.BINARY, "mul"),
        (NodeKind.BINARY, "div"),
        (NodeKind.BINARY, "mod"),
        (NodeKind.BINARY, "rem"),
        (NodeKind.BINARY, "pow"),
        (NodeKind.MATMUL, "matmul"),
        (NodeKind.CONSTANT, "constant"),
        (NodeKind.SPLAT, "splat"),
        (NodeKind.UNARY, "neg"),
        (NodeKind.UNARY, "relu_fwd"),
        (NodeKind.UNARY, "abs"),
        (NodeKind.UNARY, "ceil"),
        (NodeKind.UNARY, "cos"),
        (NodeKind.UNARY, "exp"),
        (NodeKind.UNARY, "floor"),
        (NodeKind.UNARY, "log"),
        (NodeKind.UNARY, "reciprocal"),
        (NodeKind.UNARY, "sqrt"),
        (NodeKind.UNARY, "rsqrt"),
        (NodeKind.UNARY, "sin"),
        (NodeKind.UNARY, "tan"),
        (NodeKind.UNARY, "sigmoid_fwd"),
        (NodeKind.UNARY, "tanh_fwd"),
        (NodeKind.UNARY, "gelu_approx_tanh_fwd"),
        (NodeKind.UNARY, "softplus_fwd"),
        (NodeKind.UNARY, "swish_fwd"),
        (NodeKind.UNARY, "elu_fwd"),
        (NodeKind.CONVERT, "convert"),
        (NodeKind.MOVEMENT, "slice"),
        (NodeKind.MOVEMENT, "broadcast"),
        (NodeKind.MOVEMENT, "reshape"),
        (NodeKind.MOVEMENT, "transpose"),
        (NodeKind.CONCATENATE, "concatenate"),
        (NodeKind.IOTA, "iota"),
        (NodeKind.REDUCE, "reduce"),
        (NodeKind.BINARY, "max"),
        (NodeKind.BINARY, "min"),
        (NodeKind.BINARY, "add_square"),
        (NodeKind.BINARY, "atan2"),
        (NodeKind.BINARY, "and"),
        (NodeKind.BINARY, "or"),
        (NodeKind.UNARY, "not"),
        (NodeKind.CMP, "cmp"),
        (NodeKind.SELECT, "binary_select"),
    }
    assert [
        node.kwargs["comparator"] for node in graph.nodes if node.kind == NodeKind.CMP
    ] == ["oeq", "ogt", "ogt", "ole", "ole"]


@pytest.mark.parametrize(
    ("name", "kernel"),
    [
        ("sub_op_simple", sub_kernel),
        ("mul_op_simple", mul_kernel),
        ("div_op_simple", div_kernel),
    ],
    ids=["sub", "mul", "div"],
)
def test_binary_operator_overloads(name: str, kernel: KernelFunction) -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32) + 2.0
    output = torch.empty_like(a)
    _compile_run_assert(
        kernel,
        a,
        b,
        output=output,
        name=name,
        options=_layout_propagation_auto_options(),
    )


def test_extended_pointwise_ops() -> None:
    a = torch.rand((8, 8), device="cuda", dtype=torch.float32) + 0.25
    b = torch.rand((8, 8), device="cuda", dtype=torch.float32) + 1.25
    output = torch.empty_like(a)

    _compile_run_assert(
        extended_float_ops_kernel,
        a,
        b,
        output=output,
        name="extended_float_ops",
        options=_layout_propagation_auto_options(),
        mlir_probes=(
            "beta = 2.000000e+00 : f64",
            "beta = 1.500000e+00 : f64",
            "beta = 2.500000e-01 : f64",
        ),
        rtol=1e-4,
        atol=1e-4,
    )


def test_extended_operator_overloads() -> None:
    a = torch.rand((8, 8), device="cuda", dtype=torch.float32) + 0.25
    b = torch.rand((8, 8), device="cuda", dtype=torch.float32) + 1.25
    output = torch.empty_like(a)

    _compile_run_assert(
        extended_operator_overload_kernel,
        a,
        b,
        output=output,
        name="extended_operator_overloads",
        options=_layout_propagation_auto_options(),
        rtol=1e-4,
        atol=1e-4,
    )


def test_atan2_rejects_integer_operands() -> None:
    @tir.kernel
    def integer_atan2_kernel(lhs, rhs):
        return tir.atan2(lhs, rhs)

    lhs = torch.ones((8, 8), dtype=torch.int32)
    rhs = torch.ones((8, 8), dtype=torch.int32)
    output = torch.empty_like(lhs)

    with pytest.raises(TypeError, match="expects floating-point tensor operands"):
        tir.compile(integer_atan2_kernel, lhs, rhs, output=output)


def test_rem_op() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.rand((8, 8), device="cuda", dtype=torch.float32) + 1.0
    output = torch.empty_like(a)

    _compile_run_assert(
        rem_kernel,
        a,
        b,
        output=output,
        name="rem_op",
        options=_layout_propagation_auto_options(),
        mlir_probes=("= rem ",),
        rtol=1e-4,
        atol=1e-4,
    )


def test_convert_op() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty((8, 8), device="cuda", dtype=torch.float16)

    _compile_run_assert(
        convert_kernel,
        a,
        output=output,
        name="convert_f32_to_f16",
        options=_layout_propagation_auto_options(),
        mlir_probes=("= convert ", "-> tensor<8x8xf16>"),
        rtol=1e-3,
        atol=1e-3,
    )


def test_convert_supported_float_types_op() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    compiled = tir.compile(
        convert_supported_float_types_kernel,
        a,
        output=output,
        name="convert_supported_float_types",
        options=_layout_propagation_auto_options(),
    )
    mlir = _module_text(compiled)
    for (_, src), (_, dst) in _tensor_ir_type_pairs(_FLOAT_CONVERT_TYPES):
        assert f": tensor<8x8x{src}> -> tensor<8x8x{dst}>" in mlir

    _run_assert_trace(compiled, a, output=output)


def test_convert_int_float_op() -> None:
    a = torch.randint(-8, 8, (8, 8), device="cuda", dtype=torch.int32)
    output = torch.empty_like(a)

    compiled = tir.compile(
        convert_int_float_kernel,
        a,
        output=output,
        name="convert_int_float",
        options=_layout_propagation_auto_options(),
    )
    mlir = _module_text(compiled)
    for (_, src), (_, dst) in _tensor_ir_type_pairs(_INT_FLOAT_CONVERT_TYPES):
        assert f": tensor<8x8x{src}> -> tensor<8x8x{dst}>" in mlir

    _run_assert_trace(compiled, a, output=output, rtol=0, atol=0)


def test_int8_input_output_type() -> None:
    a = torch.randint(-8, 8, (8, 8), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (8, 8), device="cuda", dtype=torch.int8)
    output = torch.empty_like(a)

    compiled = tir.compile(
        int8_add_kernel,
        a,
        b,
        output=output,
        name="int8_input_output",
        options=_layout_propagation_auto_options(),
    )
    mlir = _module_text(compiled)
    assert "tensor<8x8xsi8>" in mlir

    _run_assert_trace(compiled, a, b, output=output, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("dtype", "tensor_ir_dtype"),
    _SIGNED_INTEGER_TYPES[1:],
    ids=[name for _, name in _SIGNED_INTEGER_TYPES[1:]],
)
def test_signed_integer_input_output_types(
    dtype: torch.dtype, tensor_ir_dtype: str
) -> None:
    a = torch.randint(-8, 8, (8, 8), device="cuda", dtype=dtype)
    b = torch.randint(-8, 8, (8, 8), device="cuda", dtype=dtype)
    output = torch.empty_like(a)

    compiled = tir.compile(
        integer_add_kernel,
        a,
        b,
        output=output,
        name=f"{tensor_ir_dtype}_input_output",
        options=_layout_propagation_auto_options(),
    )
    mlir = _module_text(compiled)
    assert f"tensor<8x8x{tensor_ir_dtype}>" in mlir

    _run_assert_trace(compiled, a, b, output=output, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("dtype", "tensor_ir_dtype"),
    _UNSIGNED_INTEGER_TYPES,
    ids=[name for _, name in _UNSIGNED_INTEGER_TYPES],
)
def test_unsigned_integer_compile_path(
    dtype: torch.dtype, tensor_ir_dtype: str
) -> None:
    a = torch.randint(0, 8, (8, 8), dtype=dtype)
    b = torch.randint(0, 8, (8, 8), dtype=dtype)
    output = torch.empty_like(a)

    compiled = tir.compile(
        integer_add_kernel,
        a,
        b,
        output=output,
        name=f"{tensor_ir_dtype}_compile_path",
        options=_layout_propagation_auto_options(),
    )
    mlir = _module_text(compiled)
    assert f"tensor<8x8x{tensor_ir_dtype}>" in mlir

    target = (a.to(torch.int64) + b.to(torch.int64)).to(dtype)
    reference = evaluate_trace_reference(compiled.graph, a, b)
    assert torch.equal(reference, target)


@pytest.mark.parametrize(
    ("dtype", "tensor_ir_dtype"),
    _UNSIGNED_INTEGER_TYPES,
    ids=[name for _, name in _UNSIGNED_INTEGER_TYPES],
)
def test_unsigned_integer_abs_op(dtype: torch.dtype, tensor_ir_dtype: str) -> None:
    @tir.kernel
    def abs_kernel(a):
        return tir.abs(a)

    a = torch.randint(0, 8, (8, 4), device="cuda").to(dtype)
    output = torch.empty_like(a)

    _compile_run_assert(
        abs_kernel,
        a,
        output=output,
        name=f"unsigned_abs_{tensor_ir_dtype}",
        options=_layout_propagation_auto_options(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("dtype", "tensor_ir_dtype"),
    _UNSIGNED_INTEGER_TYPES,
    ids=[name for _, name in _UNSIGNED_INTEGER_TYPES],
)
def test_unsigned_integer_avg_reduce_op(
    dtype: torch.dtype, tensor_ir_dtype: str
) -> None:
    @tir.kernel
    def avg_reduce_kernel(a):
        return tir.reduce(a, dimensions=(1,), mode="avg")

    a = torch.randint(0, 8, (8, 4), device="cuda").to(dtype)
    output = torch.empty((8, 1), device="cuda", dtype=dtype)

    _compile_run_assert(
        avg_reduce_kernel,
        a,
        output=output,
        name=f"unsigned_avg_reduce_{tensor_ir_dtype}",
        options=_layout_propagation_auto_options(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("dtype", "tensor_ir_dtype"),
    _UNSIGNED_INTEGER_TYPES,
    ids=[name for _, name in _UNSIGNED_INTEGER_TYPES],
)
def test_unsigned_integer_cmp_op(dtype: torch.dtype, tensor_ir_dtype: str) -> None:
    @tir.kernel
    def cmp_kernel(a, b):
        return tir.cmp(a, b, predicate="ge")

    a = torch.randint(0, 8, (8, 4), device="cuda").to(dtype)
    b = torch.randint(0, 8, (8, 4), device="cuda").to(dtype)
    output = torch.empty_like(a, dtype=torch.bool)

    _compile_run_assert(
        cmp_kernel,
        a,
        b,
        output=output,
        name=f"unsigned_cmp_{tensor_ir_dtype}",
        options=_layout_propagation_auto_options(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("dtype", "tensor_ir_dtype"),
    _UNSIGNED_INTEGER_TYPES,
    ids=[name for _, name in _UNSIGNED_INTEGER_TYPES],
)
def test_unsigned_integer_select_op(dtype: torch.dtype, tensor_ir_dtype: str) -> None:
    @tir.kernel
    def select_kernel(mask, a, b):
        return tir.where(mask, a, b)

    a = torch.randint(0, 8, (8, 4), device="cuda").to(dtype)
    b = torch.randint(0, 8, (8, 4), device="cuda").to(dtype)
    mask = a.to(torch.int64) >= b.to(torch.int64)
    output = torch.empty_like(a)

    _compile_run_assert(
        select_kernel,
        mask,
        a,
        b,
        output=output,
        name=f"unsigned_select_{tensor_ir_dtype}",
        options=_layout_propagation_auto_options(),
        rtol=0,
        atol=0,
    )


def test_signed_integer_div_op_rounds_toward_zero() -> None:
    @tir.kernel
    def div_kernel(a, b):
        return a / b

    a = torch.tensor([[-7, -5, 5, 7]], device="cuda", dtype=torch.int32).repeat(8, 1)
    b = torch.tensor([[3, 2, 2, 3]], device="cuda", dtype=torch.int32).repeat(8, 1)
    output = torch.empty_like(a)

    compiled = _compile_run_assert(
        div_kernel,
        a,
        b,
        output=output,
        name="signed_div_round_toward_zero",
        options=_layout_propagation_auto_options(),
        rtol=0,
        atol=0,
    )
    reference = evaluate_trace_reference(compiled.graph, a, b)
    expected = torch.tensor([[-2, -2, 2, 2]], device="cuda", dtype=a.dtype).repeat(8, 1)
    assert torch.equal(reference, expected)


def test_convert_int8_op() -> None:
    a = torch.randint(-8, 8, (8, 8), device="cuda", dtype=torch.int8)
    output = torch.empty_like(a)

    compiled = tir.compile(
        convert_int8_kernel,
        a,
        output=output,
        name="convert_int8",
        options=_layout_propagation_auto_options(),
    )
    mlir = _module_text(compiled)
    assert ": tensor<8x8xsi8> -> tensor<8x8xsi32>" in mlir
    assert ": tensor<8x8xsi32> -> tensor<8x8xsi8>" in mlir

    _run_assert_trace(compiled, a, output=output, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("dtype", "tensor_ir_dtype"),
    _SIGNED_INTEGER_TYPES + _UNSIGNED_INTEGER_TYPES,
    ids=[name for _, name in _SIGNED_INTEGER_TYPES + _UNSIGNED_INTEGER_TYPES],
)
def test_integer_convert_compile_path(dtype: torch.dtype, tensor_ir_dtype: str) -> None:
    a = torch.randint(0, 8, (8, 8), dtype=dtype)
    output = torch.empty_like(a)

    compiled = tir.compile(
        _make_convert_to_kernel(_TORCH_DTYPE_TO_TENSOR_IR_DTYPE[dtype]),
        a,
        output=output,
        name=f"convert_{tensor_ir_dtype}",
        options=_layout_propagation_auto_options(),
    )
    mlir = _module_text(compiled)
    assert f": tensor<8x8x{tensor_ir_dtype}> -> tensor<8x8x{tensor_ir_dtype}>" in mlir
    reference = evaluate_trace_reference(compiled.graph, a)
    assert reference.dtype == dtype
    assert torch.equal(reference, a)


@pytest.mark.parametrize(
    ("dtype", "tensor_ir_dtype"),
    [(torch.int16, "si16"), (torch.uint32, "ui32")],
    ids=["si16", "ui32"],
)
def test_integer_full_like_rejects_unsupported_scalar_dtype(
    dtype: torch.dtype, tensor_ir_dtype: str
) -> None:
    a = torch.randint(0, 8, (8, 8), dtype=dtype)
    output = torch.empty_like(a)

    with pytest.raises(TypeError, match=f"tensor dtype {tensor_ir_dtype}"):
        tir.compile(
            integer_full_like_kernel,
            a,
            output=output,
            name=f"full_like_{tensor_ir_dtype}",
            options=_layout_propagation_auto_options(),
        )


def test_reduce_int8_max_op() -> None:
    a = torch.randint(-8, 8, (8, 4), device="cuda", dtype=torch.int8)
    output = torch.empty((8, 1), device="cuda", dtype=torch.int8)

    compiled = tir.compile(
        _make_reduce_kernel((1,), "max"),
        a,
        output=output,
        name="reduce_int8_max",
        options=_layout_propagation_options((4,)),
    )
    mlir = _module_text(compiled)
    assert "= reduce(" in mlir
    assert ": tensor<8x4xsi8> -> tensor<8x1xsi8>" in mlir
    assert "reduction_mode = <max>" in mlir

    _run_assert_trace(compiled, a, output=output, rtol=0, atol=0)


def test_splat_and_cmp_ops() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    _compile_run_assert(
        splat_cmp_kernel,
        a,
        b,
        output=output,
        name="splat_cmp_ops",
        options=_layout_propagation_auto_options(),
        mlir_probes=("= constant ", "= splat ", "= cmp "),
    )


def test_iota_op_without_graph_input() -> None:
    output = torch.empty((4, 8), device="cuda", dtype=torch.float32)

    compiled = _compile_run_assert(
        iota_kernel,
        output=output,
        name="iota_without_graph_input",
        options=_layout_propagation_options((4, 8)),
        mlir_probes=("graph @iota_without_graph_input()", "= iota ", "dimension = 1"),
        rtol=0,
        atol=0,
    )
    assert not compiled.graph.input_ids


def test_constant_op() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    _compile_run_assert(
        constant_kernel,
        a,
        output=output,
        name="constant_op",
        options=_layout_propagation_auto_options(),
        mlir_probes=("= constant dense<2.500000e+00> : tensor<8x8xf32>",),
        rtol=0,
        atol=0,
    )


def test_constant_rejects_call_outside_kernel() -> None:
    with pytest.raises(RuntimeError, match="must be called from a @tir.kernel"):
        tir.constant(2.5, shape=(8, 8), dtype=tir.DataType.F32)


def test_splat_op() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    _compile_run_assert(
        splat_kernel,
        a,
        output=output,
        name="splat_op",
        options=_layout_propagation_auto_options(),
        mlir_probes=("= constant 2.500000e+00 : f32", "= splat "),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "tile_sizes",
    [(64,), (16,), (8,)],
    ids=["tile_64", "tile_16", "tile_8"],
)
def test_logical_not_op(tile_sizes: tuple[int, ...]) -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty((8, 8), device="cuda", dtype=torch.bool)

    _compile_run_assert(
        logical_not_kernel,
        a,
        b,
        output=output,
        name=f"logical_not_{tile_sizes[0]}",
        options=_layout_propagation_options(tile_sizes),
        mlir_probes=("= not ",),
        rtol=0,
        atol=0,
    )


def test_logical_not_rejects_non_bool_operand() -> None:
    @tir.kernel
    def bad_logical_not_kernel(a):
        return tir.logical_not(a)

    a = torch.randn((8, 8), dtype=torch.float32)
    output = torch.empty((8, 8), dtype=torch.bool)

    with pytest.raises(TypeError, match="expects an i1 tensor operand"):
        tir.compile(
            bad_logical_not_kernel,
            a,
            output=output,
            name="bad_logical_not",
            options=_layout_propagation_options((8, 8)),
        )


@pytest.mark.parametrize(
    ("name", "kernel"),
    [
        ("int_cmp_op", int_cmp_kernel),
        ("int_operator_cmp_op", int_operator_cmp_kernel),
    ],
    ids=["explicit_cmp", "operator_cmp"],
)
def test_int_cmp_op(name: str, kernel: KernelFunction) -> None:
    a = torch.randint(0, 8, (8, 8), device="cuda", dtype=torch.int32)
    b = torch.randint(0, 8, (8, 8), device="cuda", dtype=torch.int32)
    output = torch.empty((8, 8), device="cuda", dtype=torch.bool)

    _compile_run_assert(
        kernel,
        a,
        b,
        output=output,
        name=name,
        options=_layout_propagation_auto_options(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    (
        "name",
        "kernel",
        "input_shapes",
        "output_shape",
        "tile_sizes",
        "auto_options",
        "mlir_probes",
    ),
    [
        (
            "reshape_op",
            reshape_kernel,
            ((4, 4),),
            (2, 8),
            (16,),
            False,
            ("= reshape ",),
        ),
        (
            "transpose_op",
            transpose_kernel,
            ((4, 8),),
            (8, 4),
            (8, 4),
            False,
            ("= transpose ", "permutation = [1, 0]"),
        ),
        (
            "broadcast_op",
            broadcast_kernel,
            ((4, 1), (4, 8)),
            (4, 8),
            (4, 8),
            False,
            ("= broadcast ", "tensor<4x1xf32> -> tensor<4x8xf32>"),
        ),
        (
            "slice_op",
            slice_kernel,
            ((8, 8),),
            (4, 4),
            (4, 4),
            False,
            ("= slice ", "starts = [2, 1]", "limits = [6, 5]", "strides = [1, 1]"),
        ),
        (
            "concatenate_op",
            concatenate_kernel,
            ((4, 4), (4, 4)),
            (4, 8),
            (),
            True,
            ("= concatenate ", "dimension = 1"),
        ),
        (
            "iota_like_op",
            iota_like_kernel,
            ((4, 8),),
            (4, 8),
            (4, 8),
            False,
            ("= iota ", "dimension = 1"),
        ),
    ],
    ids=["reshape", "transpose", "broadcast", "slice", "concatenate", "iota_like"],
)
def test_movement_ops(
    name: str,
    kernel: KernelFunction,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shape: tuple[int, ...],
    tile_sizes: tuple[int, ...],
    auto_options: bool,
    mlir_probes: tuple[str, ...],
) -> None:
    inputs = tuple(
        torch.randn(shape, device="cuda", dtype=torch.float32) for shape in input_shapes
    )
    output = torch.empty(output_shape, device="cuda", dtype=torch.float32)
    options = (
        _layout_propagation_auto_options()
        if auto_options
        else _layout_propagation_options(tile_sizes)
    )
    _compile_run_assert(
        kernel,
        *inputs,
        output=output,
        name=name,
        options=options,
        mlir_probes=mlir_probes,
    )


@pytest.mark.parametrize(
    (
        "name",
        "input_shape",
        "output_shape",
        "tile_sizes",
        "mode",
        "dimensions",
    ),
    [
        ("reduce_add_dim1", (8, 4), (8, 1), (4,), "add", (1,)),
        ("reduce_avg_dim0", (4, 8), (1, 8), (8,), "avg", (0,)),
        (
            "reduce_max_dim2",
            (4, 4, 8),
            (4, 4, 1),
            (4,),
            "max",
            (2,),
        ),
        (
            "reduce_min_dims02",
            (4, 8, 4),
            (1, 8, 1),
            (8,),
            "min",
            (0, 2),
        ),
        ("reduce_mul_dim1", (8, 4), (8, 1), (4,), "mul", (1,)),
        (
            "reduce_mul_no_zeros_dim1",
            (8, 4),
            (8, 1),
            (4,),
            "mul_no_zeros",
            (1,),
        ),
        (
            "reduce_amax_dim0",
            (4, 8),
            (1, 8),
            (8,),
            "amax",
            (0,),
        ),
        (
            "reduce_norm1_dims12",
            (4, 4, 8),
            (4, 1, 1),
            (4,),
            "norm1",
            (1, 2),
        ),
        (
            "reduce_norm2_dim2",
            (4, 4, 8),
            (4, 4, 1),
            (4,),
            "norm2",
            (2,),
        ),
    ],
    ids=[
        "add_dim1",
        "avg_dim0",
        "max_dim2",
        "min_dims02",
        "mul_dim1",
        "mul_no_zeros_dim1",
        "amax_dim0",
        "norm1_dims12",
        "norm2_dim2",
    ],
)
def test_reduce_ops(
    name: str,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    tile_sizes: tuple[int, ...],
    mode: str,
    dimensions: tuple[int, ...],
) -> None:
    if mode in ("mul", "mul_no_zeros"):
        a = torch.rand(input_shape, device="cuda", dtype=torch.float32) + 0.5
        if mode == "mul_no_zeros":
            a[:, ::2] = 0
    else:
        a = torch.randn(input_shape, device="cuda", dtype=torch.float32)
    output = torch.empty(output_shape, device="cuda", dtype=torch.float32)

    compiled = tir.compile(
        _make_reduce_kernel(dimensions, mode),
        a,
        output=output,
        name=name,
        options=_layout_propagation_options(tile_sizes),
    )
    mlir = _module_text(compiled)
    assert "= reduce(" in mlir
    dimension_list = ", ".join(str(dim) for dim in dimensions)
    assert f"dimensions = [{dimension_list}]" in mlir
    assert f"reduction_mode = <{mode}>" in mlir

    _run_assert_trace(compiled, a, output=output)
