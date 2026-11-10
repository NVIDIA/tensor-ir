# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s


from __future__ import annotations

import operator
import shutil

import pytest
import torch

from nv_tensor_ir import dsl as tir
from nv_tensor_ir import runtime
from nv_tensor_ir._mlir import ir
from nv_tensor_ir._mlir.dialects import nv_tensor_ir
from nv_tensor_ir.dsl.module_builder import build_mlir_module
from dsl_test_utils import _require_runtime_launch


TEST_SIZE = 64

_requires_cuda_tile = pytest.mark.skipif(
    not torch.cuda.is_available() or shutil.which("tileiras") is None,
    reason="CUDA GPU and tileiras are required",
)


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


def _run_program(text, inputs, strategy=runtime.CodegenStrategy.LayoutPropagation):
    with ir.Context() as context, ir.Location.unknown():
        nv_tensor_ir.register_dialect(context, load=True)
        module = ir.Module.parse(text)
        major, minor = torch.cuda.get_device_capability()
        with runtime.compile(
            module,
            compute_capability=major * 10 + minor,
            codegen_strategy=strategy,
            tile_sizes=[128],
            bytecode_version=runtime.BytecodeVersion.current(),
        ) as program:
            output = (
                torch.full_like(inputs[0], float("nan"))
                if inputs[0].is_floating_point()
                else torch.empty_like(inputs[0])
            )
            _require_runtime_launch()
            with program.load() as kernel:
                kernel.launch(*inputs, output, stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
    return output.cpu()


def _run_binary(operation, element_type, dtype, lhs, rhs, expected, strategy, attrs=""):
    lhs, rhs, expected = [(values * 128)[:128] for values in (lhs, rhs, expected)]
    tensor_type = f"tensor<128x{element_type}>"
    text = f"""
    nv_tensor_ir.graph @pointwise_gpu(%lhs: {tensor_type}, %rhs: {tensor_type})
        -> {tensor_type} {{
      %result = {operation} %lhs, %rhs {attrs} : {tensor_type}
      results %result : {tensor_type}
    }}
    """
    inputs = [torch.tensor(values, dtype=dtype, device="cuda") for values in (lhs, rhs)]
    actual = _run_program(text, inputs, strategy)
    assert torch.equal(actual, torch.tensor(expected, dtype=dtype))


@_requires_cuda_tile
@pytest.mark.parametrize(
    "strategy",
    [runtime.CodegenStrategy.AffineMap, runtime.CodegenStrategy.LayoutPropagation],
)
@pytest.mark.parametrize(
    "element_type,dtype,width,signed",
    [
        ("i1", torch.bool, 1, False),
        ("si8", torch.int8, 8, True),
        ("si64", torch.int64, 64, True),
        ("ui8", torch.uint8, 8, False),
        ("ui64", torch.uint64, 64, False),
    ],
)
@pytest.mark.parametrize("operation", ["and", "or", "xor"])
def test_native_bitwise_gpu(strategy, element_type, dtype, width, signed, operation):
    if width == 1:
        lhs, rhs = [False, False, True, True], [False, True, False, True]
    elif signed:
        lhs = [0, 1, -1, -(1 << (width - 1)), (1 << (width - 1)) - 1]
        rhs = [-1, 0, 2, -1, 1]
    else:
        mask = (1 << width) - 1
        lhs, rhs = [0, 1, mask, 1 << (width - 1), mask - 1], [mask, 0, 2, mask, 1]
    function = {"and": operator.and_, "or": operator.or_, "xor": operator.xor}[
        operation
    ]
    expected = [function(a, b) for a, b in zip(lhs, rhs)]
    _run_binary(operation, element_type, dtype, lhs, rhs, expected, strategy)


@_requires_cuda_tile
@pytest.mark.parametrize(
    "strategy",
    [runtime.CodegenStrategy.AffineMap, runtime.CodegenStrategy.LayoutPropagation],
)
@pytest.mark.parametrize(
    "overflow", ["none", "no_signed_wrap", "no_unsigned_wrap", "no_wrap"]
)
def test_native_shift_left_overflow_gpu(strategy, overflow):
    _run_binary(
        "shift_left",
        "si32",
        torch.int32,
        [1, 3, 7, 15],
        [0, 1, 2, 3],
        [1, 6, 28, 120],
        strategy,
        attrs=f"overflow <{overflow}>",
    )


@_requires_cuda_tile
@pytest.mark.parametrize(
    "strategy",
    [runtime.CodegenStrategy.AffineMap, runtime.CodegenStrategy.LayoutPropagation],
)
@pytest.mark.parametrize(
    "element_type,dtype,width,signed",
    [
        ("si8", torch.int8, 8, True),
        ("si64", torch.int64, 64, True),
        ("ui8", torch.uint8, 8, False),
        ("ui64", torch.uint64, 64, False),
    ],
)
@pytest.mark.parametrize("operation", ["shift_left", "shift_right"])
def test_native_shift_boundaries_gpu(
    strategy, element_type, dtype, width, signed, operation
):
    mask = (1 << width) - 1
    lhs = [
        -(1 << (width - 1)) if signed else mask,
        1,
        1,
        -1 if signed else mask,
        -1 if signed else mask,
        1,
        1,
        1,
    ]
    rhs = [
        0,
        width - 1,
        width,
        width + 1,
        -1 if signed else mask,
        (1 << 32) if width == 64 else width,
        width,
        1,
    ]
    expected = []
    for value, amount in zip(lhs, rhs):
        amount &= mask
        if operation == "shift_right":
            result = value >> min(amount, width)
        else:
            result = ((value << amount) & mask) if amount < width else 0
            if signed and result >= (1 << (width - 1)):
                result -= 1 << width
        expected.append(result)
    _run_binary(operation, element_type, dtype, lhs, rhs, expected, strategy)


@_requires_cuda_tile
@pytest.mark.parametrize(
    "strategy",
    [runtime.CodegenStrategy.AffineMap, runtime.CodegenStrategy.LayoutPropagation],
)
@pytest.mark.parametrize("operation", ["shift_left", "shift_right"])
def test_native_boolean_shifts_gpu(strategy, operation):
    _run_binary(
        operation,
        "i1",
        torch.bool,
        [False, False, True, True],
        [False, True, False, True],
        [False, False, True, False],
        strategy,
    )


@_requires_cuda_tile
@pytest.mark.parametrize(
    "strategy",
    [runtime.CodegenStrategy.AffineMap, runtime.CodegenStrategy.LayoutPropagation],
)
@pytest.mark.parametrize(
    "element_type,dtype,width",
    [
        ("i1", torch.bool, 1),
        ("ui8", torch.uint8, 8),
        ("ui16", torch.uint16, 16),
        ("ui32", torch.uint32, 32),
        ("ui64", torch.uint64, 64),
    ],
)
def test_native_mul_high_gpu(strategy, element_type, dtype, width):
    if width == 1:
        lhs, rhs = [False, False, True, True], [False, True, False, True]
    else:
        mask = (1 << width) - 1
        lhs, rhs = [0, 1, mask, mask, 1 << (width - 1)], [mask, mask, mask, 2, 2]
    expected = [(a * b) >> width for a, b in zip(lhs, rhs)]
    _run_binary("mul_high", element_type, dtype, lhs, rhs, expected, strategy)


@_requires_cuda_tile
@pytest.mark.parametrize(
    "element_type,dtype,mantissa_bits",
    [
        ("f16", torch.float16, 10),
        ("bf16", torch.bfloat16, 7),
        ("f32", torch.float32, 23),
        ("f64", torch.float64, 52),
    ],
)
def test_native_fma_types_gpu(element_type, dtype, mantissa_bits):
    tensor_type = f"tensor<128x{element_type}>"
    text = f"""
    nv_tensor_ir.graph @fma_types_gpu(%a: {tensor_type}, %b: {tensor_type},
                                     %c: {tensor_type}) -> {tensor_type} {{
      %result = fma %a, %b, %c : {tensor_type}
      results %result : {tensor_type}
    }}
    """
    epsilon = 2.0**-mantissa_bits
    a = torch.full((128,), 1.0 + epsilon, dtype=dtype, device="cuda")
    b = a.clone()
    c = torch.full_like(a, -(1.0 + 2.0 * epsilon))
    actual = _run_program(text, [a, b, c])
    # Separate multiplication rounds away epsilon squared before cancellation.
    expected_fused = torch.full((128,), epsilon * epsilon, dtype=dtype)
    assert torch.equal(actual, expected_fused)


@_requires_cuda_tile
@pytest.mark.parametrize(
    "mode", ["nearest_even", "zero", "negative_inf", "positive_inf"]
)
def test_native_fma_rounding_gpu(mode):
    text = f"""
    nv_tensor_ir.graph @fma_rounding_gpu(%a: tensor<128xf32>, %b: tensor<128xf32>,
                                        %c: tensor<128xf32>) -> tensor<128xf32> {{
      %result = fma %a, %b, %c <rounding_mode = {mode}>
          : tensor<128xf32>
      results %result : tensor<128xf32>
    }}
    """
    a = torch.tensor([1.0, -1.0, 1.0, -1.0] * 32, dtype=torch.float32, device="cuda")
    b = torch.ones_like(a)
    half_ulp = 2.0**-24
    c = torch.tensor(
        [half_ulp, -half_ulp, 1.5 * half_ulp, -1.5 * half_ulp] * 32,
        dtype=torch.float32,
        device="cuda",
    )
    next_value = 1.0 + 2.0**-23
    expected = {
        "nearest_even": [1.0, -1.0, next_value, -next_value],
        "zero": [1.0, -1.0, 1.0, -1.0],
        "negative_inf": [1.0, -next_value, 1.0, -next_value],
        "positive_inf": [next_value, -1.0, next_value, -1.0],
    }[mode]
    actual = _run_program(text, [a, b, c])
    assert torch.equal(actual, torch.tensor(expected * 32, dtype=torch.float32))


@_requires_cuda_tile
def test_native_fma_ftz_gpu():
    text = """
    nv_tensor_ir.graph @fma_ftz_gpu(%a: tensor<128xf32>, %b: tensor<128xf32>,
                                    %c: tensor<128xf32>) -> tensor<128xf32> {
      %fma = fma %a, %b, %c <ftz> : tensor<128xf32>
      results %fma : tensor<128xf32>
    }
    """
    minimum_normal = 2.0**-126
    minimum_subnormal = 2.0**-149
    values = [
        [minimum_subnormal, -minimum_subnormal, minimum_normal, -minimum_normal],
        [1.0, 1.0, 0.5, 0.5],
        [0.0, -0.0, 0.0, -0.0],
    ]
    inputs = [
        torch.tensor(value * 32, dtype=torch.float32, device="cuda") for value in values
    ]
    actual = _run_program(text, inputs)
    expected = torch.tensor([0.0, -0.0, 0.0, -0.0] * 32)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@_requires_cuda_tile
@pytest.mark.parametrize(
    "element_type,dtype,rtol",
    [
        ("f16", torch.float16, 0.002),
        ("bf16", torch.bfloat16, 0.016),
        ("f32", torch.float32, 3e-6),
        ("f64", torch.float64, 2e-13),
    ],
)
@pytest.mark.parametrize("operation", ["exp2", "log2", "sinh", "cosh"])
def test_native_float_unary_types_gpu(operation, element_type, dtype, rtol):
    tensor_type = f"tensor<128x{element_type}>"
    text = f"""
    nv_tensor_ir.graph @float_unary_gpu(%x: {tensor_type}) -> {tensor_type} {{
      %result = {operation} %x : {tensor_type}
      results %result : {tensor_type}
    }}
    """
    values = torch.tensor([0.25, 0.5, 1.0, 2.0] * 32, dtype=dtype, device="cuda")
    actual = _run_program(text, [values])
    reference = getattr(torch, operation)(values.cpu().double()).to(dtype)
    torch.testing.assert_close(actual, reference, atol=0.0, rtol=rtol)


@_requires_cuda_tile
def test_native_exp2_ftz_gpu():
    text = """
    nv_tensor_ir.graph @exp_ftz_gpu(%x: tensor<128xf32>) -> tensor<128xf32> {
      %exp = exp2 %x <ftz> : tensor<128xf32>
      results %exp : tensor<128xf32>
    }
    """
    values = torch.tensor(
        [-127.0, -128.0, -149.0, -126.0] * 32, dtype=torch.float32, device="cuda"
    )
    actual = _run_program(text, [values])
    expected = torch.tensor([0.0, 0.0, 0.0, 2.0**-126] * 32)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@_requires_cuda_tile
@pytest.mark.parametrize(
    "strategy",
    [runtime.CodegenStrategy.AffineMap, runtime.CodegenStrategy.LayoutPropagation],
)
@pytest.mark.parametrize(
    "element_type,dtype,overflow,values,expected",
    [
        ("si8", torch.int8, "none", [-128, -1, 0, 1], [-128, 1, 0, -1]),
        ("si32", torch.int32, "no_signed_wrap", [-127, -1, 0, 1], [127, 1, 0, -1]),
        ("si32", torch.int32, "no_wrap", [0], [0]),
        ("ui64", torch.uint64, "none", [0, 1, 2**64 - 1], [0, 2**64 - 1, 1]),
        ("i1", torch.bool, "none", [False, True], [False, True]),
        ("i1", torch.bool, "no_signed_wrap", [False], [False]),
        ("i1", torch.bool, "no_wrap", [False], [False]),
    ],
)
def test_native_negi_gpu(strategy, element_type, dtype, overflow, values, expected):
    values = (values * 128)[:128]
    expected = (expected * 128)[:128]
    tensor_type = f"tensor<128x{element_type}>"
    text = f"""
    nv_tensor_ir.graph @negi_gpu(%input: {tensor_type}) -> {tensor_type} {{
      %result = negi %input overflow <{overflow}> : {tensor_type}
      results %result : {tensor_type}
    }}
    """
    inputs = torch.tensor(values, dtype=dtype, device="cuda")
    actual = _run_program(text, [inputs], strategy)
    assert torch.equal(actual, torch.tensor(expected, dtype=dtype))


@pytest.mark.parametrize(
    "dtype", [tir.DataType.BOOL, tir.DataType.SI8, tir.DataType.UI64, tir.DataType.F32]
)
@pytest.mark.parametrize("operation", [tir.neg, lambda value: -value])
def test_negation_selects_integer_or_float_op(dtype, operation):
    @tir.kernel
    def kernel(value):
        return operation(value)

    spec = tir.TensorSpec((4,), dtype=dtype)
    module, _ = build_mlir_module(tir.trace(kernel, spec, output=spec), "negation")
    assert module.operation.verify()
    operations = list(module.body.operations[0].regions[0].blocks[0].operations)
    expected = "nv_tensor_ir.neg" if dtype == tir.DataType.F32 else "nv_tensor_ir.negi"
    assert operations[0].operation.name == expected


@pytest.mark.parametrize("overflow", ["none", "no_signed_wrap", "no_wrap"])
@pytest.mark.parametrize("dtype", [tir.DataType.BOOL, tir.DataType.SI32])
def test_negi_overflow_policy(overflow, dtype):
    @tir.kernel
    def kernel(value):
        return tir.negi(value, overflow=overflow)

    spec = tir.TensorSpec((4,), dtype=dtype)
    module, _ = build_mlir_module(tir.trace(kernel, spec, output=spec), "negi")
    assert module.operation.verify()
    op = module.body.operations[0].regions[0].blocks[0].operations[0]
    assert str(op.overflow) == f"#nv_tensor_ir.overflow<{overflow}>"


@pytest.mark.parametrize("dtype", [tir.DataType.F32, tir.DataType.F64])
def test_negi_rejects_non_integer_tensor(dtype):
    @tir.kernel
    def kernel(value):
        return tir.negi(value)

    spec = tir.TensorSpec((4,), dtype=dtype)
    with pytest.raises(TypeError, match="requires an integer"):
        tir.trace(kernel, spec, output=spec)


@pytest.mark.parametrize("overflow", ["no_unsigned_wrap", "invalid"])
def test_negi_rejects_invalid_overflow(overflow):
    @tir.kernel
    def kernel(value):
        return tir.negi(value, overflow=overflow)

    spec = tir.TensorSpec((4,), dtype=tir.DataType.SI32)
    with pytest.raises(ValueError, match="overflow must be"):
        tir.trace(kernel, spec, output=spec)


def test_overflow_attribute_is_first_class():
    with ir.Context(), ir.Location.unknown():
        nv_tensor_ir.register_dialect()
        for policy in nv_tensor_ir.IntegerOverflow:
            attr = ir.Attribute.parse(f"#nv_tensor_ir.overflow<{policy}>")
            assert str(attr) == f"#nv_tensor_ir.overflow<{policy}>"


@pytest.mark.parametrize(
    "overflow", ["none", "no_signed_wrap", "no_unsigned_wrap", "no_wrap"]
)
@pytest.mark.parametrize("dtype", [tir.DataType.BOOL, tir.DataType.SI32])
def test_shift_left_overflow_policy(overflow, dtype):
    @tir.kernel
    def kernel(lhs, rhs):
        return tir.shift_left(lhs, rhs, overflow=overflow)

    spec = tir.TensorSpec((4,), dtype=dtype)
    module, _ = build_mlir_module(tir.trace(kernel, spec, spec, output=spec), "shift")
    assert module.operation.verify()
    op = module.body.operations[0].regions[0].blocks[0].operations[0]
    assert str(op.overflow) == f"#nv_tensor_ir.overflow<{overflow}>"


@pytest.mark.parametrize("operation", [tir.shift_left, operator.lshift])
def test_shift_left_defaults_to_wrapping(operation):
    @tir.kernel
    def kernel(lhs, rhs):
        return operation(lhs, rhs)

    spec = tir.TensorSpec((4,), dtype=tir.DataType.SI32)
    module, _ = build_mlir_module(tir.trace(kernel, spec, spec, output=spec), "shift")
    assert module.operation.verify()
    op = module.body.operations[0].regions[0].blocks[0].operations[0]
    assert str(op.overflow) == "#nv_tensor_ir.overflow<none>"


def test_shift_left_rejects_invalid_overflow():
    @tir.kernel
    def kernel(lhs, rhs):
        return tir.shift_left(lhs, rhs, overflow="invalid")

    spec = tir.TensorSpec((4,), dtype=tir.DataType.SI32)
    with pytest.raises(ValueError, match="shift_left overflow must be"):
        tir.trace(kernel, spec, spec, output=spec)


@pytest.mark.parametrize(
    "operation",
    [
        tir.xor,
        operator.xor,
        tir.shift_left,
        operator.lshift,
        tir.shift_right,
        operator.rshift,
        tir.mul_high,
    ],
)
def test_boolean_integer_ops(operation):
    @tir.kernel
    def kernel(lhs, rhs):
        return operation(lhs, rhs)

    spec = tir.TensorSpec((4,), dtype=tir.DataType.BOOL)
    module, _ = build_mlir_module(tir.trace(kernel, spec, spec, output=spec), "boolean")
    assert module.operation.verify()


@pytest.mark.parametrize("op_name", ["exp2", "log2", "sinh", "cosh", "fma"])
@pytest.mark.parametrize("dtype", [tir.DataType.BOOL, tir.DataType.SI32])
def test_float_ops_reject_integer_types(op_name, dtype):
    @tir.kernel
    def kernel(value):
        operation = getattr(tir, op_name)
        return operation(value, value, value) if op_name == "fma" else operation(value)

    spec = tir.TensorSpec((4,), dtype=dtype)
    with pytest.raises(TypeError, match="floating-point"):
        tir.trace(kernel, spec, output=spec)


@pytest.mark.parametrize("op_name", ["exp2", "log2", "sinh", "cosh", "fma"])
@pytest.mark.parametrize(
    "dtype", [tir.DataType.F16, tir.DataType.BF16, tir.DataType.F32, tir.DataType.F64]
)
def test_float_ops_accept_supported_types(op_name, dtype):
    @tir.kernel
    def kernel(value):
        operation = getattr(tir, op_name)
        return operation(value, value, value) if op_name == "fma" else operation(value)

    spec = tir.TensorSpec((4,), dtype=dtype)
    module, _ = build_mlir_module(tir.trace(kernel, spec, output=spec), "floating")
    assert module.operation.verify()


@pytest.mark.parametrize("operation", [tir.logical_and, tir.logical_or])
@pytest.mark.parametrize(
    "dtype", [tir.DataType.SI32, tir.DataType.UI32, tir.DataType.F32]
)
def test_logical_binary_rejects_non_boolean_operands(operation, dtype):
    @tir.kernel
    def kernel(lhs, rhs):
        return operation(lhs, rhs)

    spec = tir.TensorSpec((4,), dtype=dtype)
    with pytest.raises(TypeError, match="expects boolean tensor operands"):
        tir.trace(kernel, spec, spec, output=spec)


@pytest.mark.parametrize("operation", [tir.logical_and, tir.logical_or])
def test_logical_binary_accepts_boolean_operands(operation):
    @tir.kernel
    def kernel(lhs, rhs):
        return operation(lhs, rhs)

    spec = tir.TensorSpec((4,), dtype=tir.DataType.BOOL)
    module, _ = build_mlir_module(tir.trace(kernel, spec, spec, output=spec), "logical")
    assert module.operation.verify()


@pytest.mark.parametrize(
    "backend",
    [
        "cuda_tile",
    ],
)
@pytest.mark.parametrize("shape", [(8, 8), (1, 8), (8, 1)])
def test_fma_backend_shapes(backend, shape):
    @tir.kernel
    def kernel(a, b, c):
        return tir.fma(a, b, c)

    spec = tir.TensorSpec(shape, dtype=tir.DataType.F32)
    graph = tir.trace(kernel, spec, spec, spec, output=spec)
    if backend == "cuda_tile":
        from nv_tensor_ir.dsl.cuda_tile_backend import prepare_tensor_infos

        overrides = prepare_tensor_infos(graph, dynamic_shape=True)
    module, _ = build_mlir_module(graph, "fma_shapes", overrides)
    assert module.operation.verify()
    operations = module.body.operations[0].regions[0].blocks[0].operations
    fma = next(op for op in operations if op.operation.name == "nv_tensor_ir.fma")
    assert all(operand.type == fma.result.type for operand in fma.operands)
