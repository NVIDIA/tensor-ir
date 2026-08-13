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
    _assert_compile_error,
    _module_text,
    _require_runtime_launch,
    _run_assert_trace,
)
from nv_tensor_ir.dsl.dsl import KernelFunction
from nv_tensor_ir.dsl.tracing import TensorInfo, TraceGraph


@tir.kernel
def add_kernel(a, b):
    return a + b


@tir.kernel
def dynamic_pointwise_kernel(a, b):
    shifted = a + b
    activated = tir.softplus(shifted, beta=1.5)
    zero = tir.full_like(activated, value=0.0)
    upper = tir.full_like(activated, value=10.0)
    positive = tir.cmp(activated, zero, predicate="ogt")
    bounded = tir.cmp(activated, upper, predicate="olt")
    mask = tir.logical_or(tir.logical_and(positive, bounded), positive)
    selected = tir.where(mask, activated, zero)
    return tir.convert(selected, dtype=tir.DataType.F32)


@tir.kernel
def dynamic_transpose_add_kernel(a, b):
    return tir.transpose(a, permutation=(1, 0)) + b


@tir.kernel
def matmul_static_control_flow_kernel(a, b, c):
    result = a @ b
    for i in range(3):
        if i == 0:
            result = result + c
        result = tir.relu(result)
    return result


@tir.kernel
def complex_poc_style_kernel(a, b, c, threshold):
    mm = a @ b
    biased = mm + c
    activated = tir.relu(biased)
    scaled = activated * c
    safe = tir.sqrt(tir.abs(scaled))
    mask = safe > threshold
    return tir.where(mask, tir.sigmoid(safe), safe)


@tir.kernel
def dynamic_shape_unsupported_reshape_kernel(a):
    return tir.reshape(a, shape=(4, 16))


@tir.kernel
def dynamic_matmul_kernel(a, b):
    return a @ b


@tir.kernel
def mixed_output_matmul_kernel(a, b):
    return tir.matmul(a, b, dtype=tir.DataType.F32)


def test_add_sample_module_builder() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    compiled = tir.compile(add_kernel, a, b, output=output, name="add_op_simple")
    assert isinstance(compiled, tir.CompiledKernel)
    assert compiled.graph.input_ids
    assert compiled.graph.result_ids
    mlir = _module_text(compiled)
    assert "nv_tensor_ir.graph @add_op_simple" in mlir
    assert mlir.count("tensor<8x8xf32>") >= 4
    assert 'nv_tensor_ir.stride = "(8,1)"' in mlir
    assert "= add " in mlir
    assert ": tensor<8x8xf32>" in mlir
    assert "results " in mlir
    bytecode = compiled.get_bytecode()
    assert isinstance(bytecode, bytes)
    assert bytecode

    _run_assert_trace(compiled, a, b, output=output)


def test_add_kernel_ground_truth() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    compiled = tir.compile(add_kernel, a, b, output=output, name="add_ground_truth")
    _require_runtime_launch()
    compiled.run(a, b, output=output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, a + b, rtol=1e-3, atol=1e-3)


def test_dynamic_shape_pointwise_graph() -> None:
    sample_a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    sample_b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    sample_output = torch.empty_like(sample_a)

    compiled = tir.compile(
        dynamic_pointwise_kernel,
        sample_a,
        sample_b,
        output=sample_output,
        name="dynamic_pointwise",
        tile_sizes=(4, 4),
        dynamic_shape=True,
    )
    mlir = _module_text(compiled)
    assert "tensor<?x?xf32>" in mlir
    assert 'nv_tensor_ir.stride = "(?,1)"' in mlir
    assert ": tensor<?x?xf32>" in mlir
    assert "= add " in mlir
    assert "= softplus_fwd" in mlir
    assert "= splat " in mlir
    assert "= cmp " in mlir
    assert "= and " in mlir
    assert "= or " in mlir
    assert "= binary_select " in mlir
    assert "= convert " in mlir

    a = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    _require_runtime_launch()
    compiled.run(a, b, output=output)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        evaluate_trace_reference(compiled.graph, a, b),
        rtol=1e-3,
        atol=1e-3,
    )


def test_dynamic_shape_accepts_tensor_specs() -> None:
    spec = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)
    compiled = tir.compile(
        dynamic_pointwise_kernel,
        spec,
        spec,
        output=spec,
        name="dynamic_pointwise_from_specs",
        tile_sizes=(4, 4),
        dynamic_shape=True,
    )

    mlir = _module_text(compiled)
    assert "nv_tensor_ir.graph @dynamic_pointwise_from_specs" in mlir
    assert "tensor<?x?xf32>" in mlir
    assert 'nv_tensor_ir.stride = "(?,1)"' in mlir


def test_dynamic_shape_transpose_graph() -> None:
    sample_a = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    sample_b = torch.randn((8, 4), device="cuda", dtype=torch.float32)
    sample_output = torch.empty_like(sample_b)

    compiled = tir.compile(
        dynamic_transpose_add_kernel,
        sample_a,
        sample_b,
        output=sample_output,
        name="dynamic_transpose_add",
        tile_sizes=(4, 4),
        dynamic_shape=True,
    )

    mlir = _module_text(compiled)
    assert "tensor<?x?xf32>" in mlir
    assert 'nv_tensor_ir.stride = "(?,1)"' in mlir
    assert "transpose" in mlir

    a = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    b = torch.randn((16, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(b)
    _run_assert_trace(compiled, a, b, output=output)


def test_matmul_result_dtype() -> None:
    lhs = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    rhs = torch.randn((16, 32), device="cuda", dtype=torch.float16)
    output = torch.empty((8, 32), device="cuda", dtype=torch.float32)

    compiled = tir.compile(
        mixed_output_matmul_kernel,
        lhs,
        rhs,
        output=output,
        name="mixed_output_matmul",
        tile_sizes=(8, 16),
    )

    mlir = _module_text(compiled)
    assert "(tensor<8x16xf16>, tensor<16x32xf16>) -> tensor<8x32xf32>" in mlir
    _run_assert_trace(compiled, lhs, rhs, output=output, rtol=1e-2, atol=1e-2)


def test_dynamic_shape_matmul_graph() -> None:
    lhs = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    rhs = torch.randn((16, 32), device="cuda", dtype=torch.float16)
    output = torch.empty((8, 32), device="cuda", dtype=torch.float16)
    options = tir.CompileOptions()
    options.codegen_strategy = tir.CodegenStrategy.AffineMap
    options.tile_sizes = [8, 16]

    compiled = tir.compile(
        dynamic_matmul_kernel,
        lhs,
        rhs,
        output=output,
        name="dynamic_matmul",
        options=options,
        dynamic_shape=True,
    )

    mlir = _module_text(compiled)
    assert mlir.count("tensor<?x?xf16>") >= 3
    assert "matmul" in mlir
    _run_assert_trace(compiled, lhs, rhs, output=output, rtol=1e-2, atol=1e-2)


def test_dynamic_shape_matmul_rejects_layout_propagation() -> None:
    lhs = torch.randn((8, 16), dtype=torch.float16)
    rhs = torch.randn((16, 32), dtype=torch.float16)
    output = torch.empty((8, 32), dtype=torch.float16)

    with pytest.raises(ValueError, match="requires CodegenStrategy.AffineMap"):
        tir.compile(
            dynamic_matmul_kernel,
            lhs,
            rhs,
            output=output,
            dynamic_shape=True,
        )


def test_dynamic_shape_rejects_mismatched_trace_shapes() -> None:
    a = torch.randn((8, 8), dtype=torch.float32)
    b = torch.randn((4, 8), dtype=torch.float32)
    output = torch.empty_like(a)

    with pytest.raises(TypeError, match="same dtype and shape"):
        tir.compile(
            add_kernel,
            a,
            b,
            output=output,
            dynamic_shape=True,
        )


@pytest.mark.parametrize(
    "kernel_func,input_shapes,output_shape",
    [
        (
            dynamic_shape_unsupported_reshape_kernel,
            ((8, 8),),
            (4, 16),
        ),
    ],
)
def test_dynamic_shape_rejects_unsupported_ops(
    kernel_func: KernelFunction,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shape: tuple[int, ...],
) -> None:
    args = tuple(
        torch.randn(shape, device="cuda", dtype=torch.float32) for shape in input_shapes
    )
    output = torch.empty(output_shape, device="cuda", dtype=torch.float32)
    with pytest.raises(TypeError, match="dynamic_shape=True"):
        tir.compile(
            kernel_func,
            *args,
            output=output,
            dynamic_shape=True,
        )


def test_compile_rejects_duplicate_compile_options() -> None:
    a = torch.randn((8, 8), dtype=torch.float32)
    b = torch.randn((8, 8), dtype=torch.float32)
    output = torch.empty_like(a)

    with pytest.raises(TypeError):
        tir.compile(
            add_kernel,
            a,
            b,
            output=output,
            options=tir.CompileOptions(),
            tile_sizes=(8, 8),
        )


def test_kernel_decorator_returns_kernel_function() -> None:
    @tir.kernel
    def decorated_kernel(a):
        return a

    assert isinstance(decorated_kernel, KernelFunction)
    assert decorated_kernel.__name__ == "decorated_kernel"


def test_traced_tensor_equality_guards_preserve_hashability() -> None:
    graph = TraceGraph()
    tensor = graph.add_input(TensorInfo("a", tir.DataType.F32, (8, 8), (8, 1)))

    assert {tensor: "value"}[tensor] == "value"
    with pytest.raises(TypeError, match="does not support =="):
        tensor == tensor
    with pytest.raises(TypeError, match="does not support !="):
        tensor != tensor


def test_compile_rejects_undecorated_function() -> None:
    def undecorated_kernel(a):
        return a

    with pytest.raises(
        TypeError,
        match=r"tir\.compile expects a function decorated with @tir\.kernel",
    ):
        tir.compile(undecorated_kernel, output=object())


def test_trace_rejects_invalid_kernel_contracts() -> None:
    @tir.kernel
    def declares_output_param(output):
        return output

    @tir.kernel
    def returns_python_value(a):
        return 1

    @tir.kernel
    def mismatched_add(a, b):
        return a + b

    @tir.kernel
    def mismatched_matmul(a, b):
        return a @ b

    a = torch.randn((8, 8), dtype=torch.float32)
    b = torch.randn((4, 8), dtype=torch.float32)
    out = torch.empty_like(a)

    _assert_compile_error(
        TypeError, "declaring an output parameter", declares_output_param, output=out
    )
    _assert_compile_error(
        TypeError, "traced tensor", returns_python_value, a, output=out
    )
    _assert_compile_error(
        TypeError, "same dtype and shape", mismatched_add, a, b, output=out
    )

    bad_rhs = torch.randn((4, 8), dtype=torch.float32)
    _assert_compile_error(
        TypeError, "K dimension mismatch", mismatched_matmul, a, bad_rhs, output=out
    )
    _assert_compile_error(
        TypeError, "output tuple must not be empty", add_kernel, a, a, output=()
    )


def test_cmp_rejects_invalid_predicates() -> None:
    @tir.kernel
    def bad_float_cmp(a, b):
        return tir.cmp(a, b, predicate="gt")

    @tir.kernel
    def bad_int_cmp(a, b):
        return tir.cmp(a, b, predicate="ogt")

    float_lhs = torch.randn((8, 8), dtype=torch.float32)
    float_rhs = torch.randn((8, 8), dtype=torch.float32)
    float_out = torch.empty((8, 8), dtype=torch.bool)
    _assert_compile_error(
        TypeError,
        "floating-point",
        bad_float_cmp,
        float_lhs,
        float_rhs,
        output=float_out,
    )

    int_lhs = torch.ones((8, 8), dtype=torch.int32)
    int_rhs = torch.zeros((8, 8), dtype=torch.int32)
    int_out = torch.empty((8, 8), dtype=torch.bool)
    _assert_compile_error(
        TypeError, "integer", bad_int_cmp, int_lhs, int_rhs, output=int_out
    )


def test_movement_ops_reject_invalid_metadata() -> None:
    @tir.kernel
    def bad_reshape(a):
        return tir.reshape(a, shape=(3, 7))

    @tir.kernel
    def bad_transpose(a):
        return tir.transpose(a, permutation=(0, 0))

    @tir.kernel
    def bad_broadcast(a):
        return tir.broadcast(a, shape=(4, 4))

    @tir.kernel
    def bad_slice(a):
        return tir.slice(a, starts=(0, 0), limits=(5, 4))

    @tir.kernel
    def bad_concatenate(a, b):
        return tir.concatenate((a, b), dimension=1)

    @tir.kernel
    def bad_iota(a):
        return tir.iota_like(a, dimension=2)

    @tir.kernel
    def bad_reduce(a):
        return tir.reduce(a, dimensions=(0, 0), mode="add")

    a = torch.randn((4, 4), dtype=torch.float32)
    b = torch.randn((8, 4), dtype=torch.float32)
    out = torch.empty_like(a)

    _assert_compile_error(
        TypeError, "preserve the number of elements", bad_reshape, a, output=out
    )
    _assert_compile_error(TypeError, "valid permutation", bad_transpose, a, output=out)
    _assert_compile_error(
        TypeError, "expand at least one dimension", bad_broadcast, a, output=out
    )
    _assert_compile_error(TypeError, "within the input shape", bad_slice, a, output=out)
    _assert_compile_error(
        TypeError,
        "outside the concat dimension",
        bad_concatenate,
        a,
        b,
        output=torch.empty((4, 8)),
    )
    _assert_compile_error(
        TypeError, "dimension must be in range", bad_iota, a, output=out
    )
    _assert_compile_error(
        TypeError,
        "dimensions must be unique",
        bad_reduce,
        a,
        output=torch.empty((1, 4)),
    )


def test_compile_accepts_tensor_specs() -> None:
    a = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)
    b = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)
    output = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)

    compiled = tir.compile(
        add_kernel,
        a,
        b,
        output=output,
        name="add_from_tensor_specs",
    )

    mlir = _module_text(compiled)
    assert "nv_tensor_ir.graph @add_from_tensor_specs" in mlir
    assert "tensor<8x8xf32>" in mlir
    assert 'nv_tensor_ir.stride = "(8,1)"' in mlir
    assert compiled.graph.nodes[compiled.graph.input_ids[0]].tensor_info.dtype is (
        tir.DataType.F32
    )


def test_compile_accepts_numpy_dlpack_metadata() -> None:
    np = pytest.importorskip("numpy")

    a_np = np.random.randn(8, 8).astype(np.float32)
    b_np = np.random.randn(8, 8).astype(np.float32)
    output_np = np.empty_like(a_np)

    compiled = tir.compile(
        add_kernel,
        a_np,
        b_np,
        output=output_np,
        name="add_from_numpy_metadata",
    )

    mlir = _module_text(compiled)
    assert "tensor<8x8xf32>" in mlir
    assert compiled.graph.nodes[compiled.graph.input_ids[0]].tensor_info.dtype is (
        tir.DataType.F32
    )

    a = torch.from_numpy(a_np).cuda()
    b = torch.from_numpy(b_np).cuda()
    output = torch.empty_like(a)
    _require_runtime_launch()
    compiled.run(a, b, output=output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, a + b, rtol=1e-5, atol=1e-5)


def test_tensor_metadata_rejects_object_without_metadata() -> None:
    from nv_tensor_ir.dsl.dsl import _tensor_info_from_value

    with pytest.raises(TypeError, match="DLPack-compatible tensors or TensorSpec"):
        _tensor_info_from_value(object(), "object")


def test_tensor_metadata_rejects_partial_dlpack_protocol() -> None:
    from nv_tensor_ir.dsl.dsl import _tensor_info_from_value

    class MissingDeviceDLPack:
        def __dlpack__(self):
            return None

    class MissingCapsuleDLPack:
        def __dlpack_device__(self):
            return (1, 0)

    with pytest.raises(TypeError, match="DLPack-compatible tensors or TensorSpec"):
        _tensor_info_from_value(MissingDeviceDLPack(), "value")
    with pytest.raises(TypeError, match="DLPack-compatible tensors or TensorSpec"):
        _tensor_info_from_value(MissingCapsuleDLPack(), "value")


def test_tensor_metadata_rejects_invalid_dlpack_capsule() -> None:
    from nv_tensor_ir.dsl.dsl import _tensor_info_from_value

    class InvalidDLPack:
        def __dlpack__(self):
            return object()

        def __dlpack_device__(self):
            return (1, 0)

    with pytest.raises(TypeError, match="did not return a PyCapsule"):
        _tensor_info_from_value(InvalidDLPack(), "value")


def test_tensor_spec_rejects_string_dtype() -> None:
    with pytest.raises(TypeError, match="tir.DataType"):
        tir.TensorSpec((8, 8), dtype="f32")


def test_tensor_spec_rejects_string_shape_or_stride() -> None:
    with pytest.raises(TypeError, match="shape must be an int sequence"):
        tir.TensorSpec("8,8", dtype=tir.DataType.F32)
    with pytest.raises(TypeError, match="stride must be an int sequence"):
        tir.TensorSpec((8, 8), dtype=tir.DataType.F32, stride="8,1")


@pytest.mark.parametrize(
    ("dtype", "tensor_ir_dtype"),
    [
        (torch.int8, tir.DataType.SI8),
        (torch.int16, tir.DataType.SI16),
        (torch.int32, tir.DataType.SI32),
        (torch.int64, tir.DataType.SI64),
        (torch.uint8, tir.DataType.UI8),
        (torch.uint16, tir.DataType.UI16),
        (torch.uint32, tir.DataType.UI32),
        (torch.uint64, tir.DataType.UI64),
    ],
)
def test_tensor_metadata_accepts_integer_dtypes(
    dtype: torch.dtype, tensor_ir_dtype: tir.DataType
) -> None:
    from nv_tensor_ir.dsl.dsl import _tensor_info_from_value

    value = torch.empty((8, 8), dtype=dtype)

    assert _tensor_info_from_value(value, "value").dtype == tensor_ir_dtype


def test_tensor_metadata_rejects_unsupported_torch_dtype() -> None:
    from nv_tensor_ir.dsl.dsl import _tensor_info_from_value

    value = torch.empty((8, 8), dtype=torch.complex64)

    with pytest.raises(TypeError, match="Unsupported TensorIR tensor dtype"):
        _tensor_info_from_value(value, "value")


def test_tensor_metadata_accepts_float64() -> None:
    from nv_tensor_ir.dsl.dsl import _tensor_info_from_value

    value = torch.empty((8, 8), dtype=torch.float64)

    assert _tensor_info_from_value(value, "value").dtype is tir.DataType.F64


def test_convert_accepts_data_type() -> None:
    @tir.kernel
    def convert_kernel(a):
        return tir.convert(a, dtype=tir.DataType.F32)

    graph = tir.compile(
        convert_kernel,
        tir.TensorSpec((8, 8), dtype=tir.DataType.F16),
        output=tir.TensorSpec((8, 8), dtype=tir.DataType.F32),
    ).graph

    result_id = graph.result_ids[0]
    assert graph.nodes[result_id].tensor_info.dtype is tir.DataType.F32


def test_public_api_surface() -> None:
    expected_ops = {
        "abs",
        "add_square",
        "atan2",
        "broadcast",
        "ceil",
        "cmp",
        "concatenate",
        "constant",
        "convert",
        "cos",
        "elu",
        "exp",
        "floor",
        "full_like",
        "gelu_approx_tanh",
        "iota",
        "iota_like",
        "log",
        "logical_and",
        "logical_not",
        "logical_or",
        "matmul",
        "max",
        "min",
        "mod",
        "neg",
        "pow",
        "reduce",
        "rem",
        "relu",
        "reciprocal",
        "reshape",
        "rsqrt",
        "sigmoid",
        "sin",
        "slice",
        "softplus",
        "splat",
        "sqrt",
        "swish",
        "tan",
        "tanh",
        "transpose",
        "where",
    }
    assert (
        set(tir.__all__)
        == {
            "BytecodeVersion",
            "CodegenStrategy",
            "CompileOptions",
            "CompiledKernel",
            "DataType",
            "TensorSpec",
            "CudaTileArtifactKind",
            "compile",
            "kernel",
        }
        | expected_ops
    )
    assert not hasattr(tir, "KernelFunction")
    assert not hasattr(tir, "ops")
    assert callable(tir.logical_not)
    assert callable(tir.relu)
    assert callable(tir.where)

    @tir.kernel
    def public_api_kernel(a):
        return a

    assert isinstance(public_api_kernel, KernelFunction)


def test_multiple_outputs() -> None:
    @tir.kernel
    def add_sub_kernel(a, b):
        return a + b, a - b

    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    add_output = torch.empty_like(a)
    sub_output = torch.empty_like(a)
    options = tir.CompileOptions()
    options.codegen_strategy = tir.CodegenStrategy.AffineMap
    options.tile_sizes = [8, 8]

    compiled = tir.compile(
        add_sub_kernel,
        a,
        b,
        output=(add_output, sub_output),
        options=options,
    )
    with pytest.raises(TypeError, match="Expected 2 input tensors"):
        compiled.run(a, output=(add_output, sub_output))

    with pytest.raises(TypeError, match="Expected 2 input tensors"):
        compiled.run(a, b, a, output=(add_output, sub_output))

    with pytest.raises(TypeError, match="Expected 2 output tensors"):
        compiled.run(a, b, output=add_output)

    _require_runtime_launch()
    compiled.run(a, b, output=(add_output, sub_output))
    torch.cuda.synchronize()

    add_reference, sub_reference = evaluate_trace_reference(compiled.graph, a, b)
    torch.testing.assert_close(add_output, add_reference, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(sub_output, sub_reference, rtol=1e-3, atol=1e-3)


def test_matmul_and_static_control_flow_examples() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    c = torch.randn((8, 8), device="cuda", dtype=torch.float32)

    output = torch.empty_like(a)
    options = tir.CompileOptions()
    options.tile_sizes = [2, 2]
    compiled = tir.compile(
        matmul_static_control_flow_kernel,
        a,
        b,
        c,
        output=output,
        name="matmul_static_control_flow",
        options=options,
    )
    _run_assert_trace(compiled, a, b, c, output=output)


def test_complex_poc_style_kernel() -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    c = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    threshold = torch.rand((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(c)

    options = tir.CompileOptions()
    options.tile_sizes = [2, 2]
    compiled = tir.compile(
        complex_poc_style_kernel,
        a,
        b,
        c,
        threshold,
        output=output,
        name="complex_poc_style",
        options=options,
    )

    _run_assert_trace(compiled, a, b, c, threshold, output=output)
