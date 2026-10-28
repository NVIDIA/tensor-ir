# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON -m pytest --capture=no -vv %s
# REQUIRES: cuda-gpu

from __future__ import annotations

import pytest
import torch

from nv_tensor_ir import dsl as tir
from nv_tensor_ir.runtime import Artifact, ArchPortability
from dsl_reference import evaluate_trace_reference
from dsl_test_utils import (
    _assert_compile_error,
    _module_text,
    _require_runtime_launch,
    _run_assert_trace,
    _run_kernel,
)
from nv_tensor_ir.dsl.dsl import KernelFunction
from nv_tensor_ir.dsl.tracing import TensorInfo, TraceGraph


@tir.kernel
def add_kernel(a, b):
    return a + b


def _compile_cached_add(
    artifact_cache: tir.ArtifactCache,
    spec: tir.TensorSpec,
    *,
    name: str = "cache_test",
    options: tir.CompileOptions | None = None,
    tile_sizes: tuple[int, ...] = (),
    dynamic_shape: bool = False,
) -> tir.CompiledKernel:
    return tir.compile(
        add_kernel,
        spec,
        spec,
        output=spec,
        name=name,
        options=options,
        tile_sizes=tile_sizes,
        dynamic_shape=dynamic_shape,
        artifact_cache=artifact_cache,
    )


def test_compile_kernel_cache_normalizes_graph_name() -> None:
    artifact_cache = tir.ArtifactCache()
    spec = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)

    first = _compile_cached_add(artifact_cache, spec, name="cache_test")
    renamed = _compile_cached_add(
        artifact_cache,
        spec,
        name="cache_test_renamed",
    )
    first_key = tir.calculate_cache_key(first.module)
    renamed_key = tir.calculate_cache_key(renamed.module)

    assert first_key == renamed_key
    assert first.artifact._native is renamed.artifact._native
    assert "nv_tensor_ir.graph @cache_test_renamed" in _module_text(renamed)


def test_compile_kernel_cache_survives_returned_program_destroy() -> None:
    artifact_cache = tir.ArtifactCache()
    spec = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)

    first = _compile_cached_add(artifact_cache, spec)
    native_program = first.artifact._native
    first.artifact.destroy()
    cached_after_destroy = _compile_cached_add(artifact_cache, spec)

    assert not cached_after_destroy.artifact.is_destroyed
    assert cached_after_destroy.artifact._native is native_program


def test_compile_kernel_cache_flush() -> None:
    artifact_cache = tir.ArtifactCache()
    spec = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)

    first = _compile_cached_add(artifact_cache, spec)

    artifact_cache.flush()
    second = _compile_cached_add(artifact_cache, spec)

    assert second.artifact._native is not first.artifact._native


def test_compile_kernel_cache_distinguishes_compile_options() -> None:
    artifact_cache = tir.ArtifactCache()
    spec = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)

    default_options = _compile_cached_add(artifact_cache, spec)

    different_options = tir.CompileOptions()
    different_options.warp_count = 8
    custom_options = _compile_cached_add(
        artifact_cache,
        spec,
        options=different_options,
    )
    default_key = tir.calculate_cache_key(default_options.module)
    custom_key = tir.calculate_cache_key(
        custom_options.module,
        options=different_options,
    )

    assert custom_key != default_key
    assert custom_options.artifact._native is not default_options.artifact._native


def test_compile_kernel_cache_distinguishes_modules() -> None:
    artifact_cache = tir.ArtifactCache()
    spec_8x8 = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)
    spec_16x8 = tir.TensorSpec((16, 8), dtype=tir.DataType.F32)

    kernel_8x8 = _compile_cached_add(artifact_cache, spec_8x8)
    kernel_16x8 = _compile_cached_add(artifact_cache, spec_16x8)
    key_8x8 = tir.calculate_cache_key(kernel_8x8.module)
    key_16x8 = tir.calculate_cache_key(kernel_16x8.module)

    assert key_16x8 != key_8x8
    assert kernel_16x8.artifact._native is not kernel_8x8.artifact._native


def test_compile_kernel_cache_normalizes_dynamic_shapes() -> None:
    artifact_cache = tir.ArtifactCache()
    spec_8x8 = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)
    spec_16x8 = tir.TensorSpec((16, 8), dtype=tir.DataType.F32)

    static_kernel = _compile_cached_add(artifact_cache, spec_8x8)
    dynamic_8x8 = _compile_cached_add(
        artifact_cache,
        spec_8x8,
        tile_sizes=(4, 4),
        dynamic_shape=True,
    )
    dynamic_16x8 = _compile_cached_add(
        artifact_cache,
        spec_16x8,
        tile_sizes=(4, 4),
        dynamic_shape=True,
    )
    static_key = tir.calculate_cache_key(static_kernel.module)
    dynamic_8x8_key = tir.calculate_cache_key(
        dynamic_8x8.module,
        tile_sizes=(4, 4),
    )
    dynamic_16x8_key = tir.calculate_cache_key(
        dynamic_16x8.module,
        tile_sizes=(4, 4),
    )

    assert static_key != dynamic_8x8_key
    assert dynamic_8x8_key == dynamic_16x8_key
    assert dynamic_8x8.artifact._native is dynamic_16x8.artifact._native


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


def test_add_sample_module_builder(profiling: bool) -> None:
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

    _run_assert_trace(compiled, a, b, output=output, profile=profiling)


def test_dynamic_tensors_without_unit_stride(profiling: bool) -> None:
    a = torch.randn(16, device="cuda", dtype=torch.float32)[::2]
    b = torch.randn(16, device="cuda", dtype=torch.float32)[::2]
    output = torch.full((16,), float("nan"), device="cuda")[::2]
    options = tir.CompileOptions()
    options.codegen_strategy = tir.CodegenStrategy.LayoutPropagation
    # Explicit tiles bypass automatic tile analysis, which needs static layouts.
    options.tile_sizes = [8]

    compiled = tir.compile(
        add_kernel, a, b, output=output, options=options, dynamic_shape=True
    )

    for node_id in compiled.graph.input_ids + compiled.graph.output_ref_ids:
        assert compiled.graph.nodes[node_id].value_info.stride == (2,)
    assert compiled.module.operation.verify()
    mlir = _module_text(compiled)
    assert "tensor<?xf32>" in mlir
    assert mlir.count('nv_tensor_ir.stride = "(?)"') == 3
    assert "nv_tensor_ir.alignment" not in mlir
    assert compiled.get_bytecode()

    _require_runtime_launch()
    _run_kernel(compiled, a, b, output=output, profile=profiling)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, a + b)


def test_trace_then_compile(profiling: bool) -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    graph = tir.trace(add_kernel, a, b, output=output)
    compiled = tir.compile_traced(graph, name="traced_add")

    assert compiled.graph is graph
    assert "nv_tensor_ir.graph @traced_add" in _module_text(compiled)
    _run_assert_trace(compiled, a, b, output=output, profile=profiling)


def test_add_kernel_ground_truth(profiling: bool) -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    compiled = tir.compile(add_kernel, a, b, output=output, name="add_ground_truth")
    _require_runtime_launch()
    _run_kernel(compiled, a, b, output=output, profile=profiling)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, a + b, rtol=1e-3, atol=1e-3)


def test_serialize_deserialize_round_trip(tmp_path) -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    compiled = tir.compile(add_kernel, a, b, output=output, name="serialize_smoke")
    _require_runtime_launch()

    # Mirror the intended cross-process usage: the compile-time process
    # persists the serialized program to disk, and a separate runtime
    # process reloads and launches it without recompiling.
    program_path = tmp_path / "serialize_smoke.tirprogram"
    program_path.write_bytes(compiled.artifact.serialize())
    reloaded_program = Artifact.deserialize(program_path.read_bytes())

    with reloaded_program.load() as kernel:
        kernel.launch(a, b, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, a + b, rtol=1e-3, atol=1e-3)


def test_run_profile_executes_kernel() -> None:
    import_or_skip = True
    if import_or_skip:
        pytest.importorskip("cupti")

    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)
    compiled = tir.compile(add_kernel, a, b, output=output, name="profile_add")

    _require_runtime_launch()
    result = compiled.run_profile(a, b, output=output, warmup=1, iterations=3)
    assert len(result.kernels) > 0
    assert result.kernels[0].name
    assert len(result.kernels[0].durations_us) > 0


def test_optional_profile_result(profiling: bool) -> None:
    a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)
    compiled = tir.compile(add_kernel, a, b, output=output, name="optional_profile")

    _require_runtime_launch()
    result = _run_kernel(compiled, a, b, output=output, profile=profiling)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, a + b, rtol=1e-3, atol=1e-3)

    if profiling:
        assert isinstance(result, tir.ProfileResult)
        assert result.kernels
        for kernel_result in result.kernels:
            assert kernel_result.name
            assert len(kernel_result.durations_us) == 10
            assert kernel_result.minimum_us == min(kernel_result.durations_us)
            assert kernel_result.maximum_us == max(kernel_result.durations_us)
            assert (
                kernel_result.minimum_us
                <= kernel_result.median_us
                <= kernel_result.maximum_us
            )
            assert (
                kernel_result.minimum_us
                <= kernel_result.average_us
                <= kernel_result.maximum_us
            )
    else:
        assert result is None


def test_dynamic_shape_pointwise_graph(profiling: bool) -> None:
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
    _run_kernel(compiled, a, b, output=output, profile=profiling)
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


def test_dynamic_shape_transpose_graph(profiling: bool) -> None:
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
    _run_assert_trace(compiled, a, b, output=output, profile=profiling)


def test_matmul_result_dtype(profiling: bool) -> None:
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
    _run_assert_trace(
        compiled,
        lhs,
        rhs,
        output=output,
        rtol=1e-2,
        atol=1e-2,
        profile=profiling,
    )


def test_dynamic_shape_matmul_graph(profiling: bool) -> None:
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
    _run_assert_trace(
        compiled,
        lhs,
        rhs,
        output=output,
        rtol=1e-2,
        atol=1e-2,
        profile=profiling,
    )


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


def test_compile_rejects_invalid_artifact_cache() -> None:
    with pytest.raises(TypeError, match="artifact_cache must be a ArtifactCache"):
        tir.compile(add_kernel, output=object(), artifact_cache={})


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
    assert compiled.graph.nodes[compiled.graph.input_ids[0]].value_info.dtype is (
        tir.DataType.F32
    )


@pytest.mark.parametrize("dynamic_shape", [False, True])
@pytest.mark.parametrize("alignment", [None, 16])
def test_tensor_spec_alignment_in_compiled_mlir(
    alignment: int | None, dynamic_shape: bool
) -> None:
    a = tir.TensorSpec((8, 8), dtype=tir.DataType.F32, alignment=alignment)
    b = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)
    output = tir.TensorSpec((8, 8), dtype=tir.DataType.F32)
    options = tir.CompileOptions()
    if dynamic_shape:
        options.tile_sizes = [8, 8]
        options.codegen_strategy = tir.CodegenStrategy.AffineMap

    compiled = tir.compile(
        add_kernel,
        a,
        b,
        output=output,
        dynamic_shape=dynamic_shape,
        options=options,
        name="add_with_input_alignment",
    )

    mlir = _module_text(compiled)
    if alignment is None:
        assert "nv_tensor_ir.alignment" not in mlir
    else:
        tensor_type = "tensor<?x?xf32>" if dynamic_shape else "tensor<8x8xf32>"
        assert f"{tensor_type} {{nv_tensor_ir.alignment = {alignment} : i64" in mlir
        # Only the annotated input gets alignment, not the other input or output.
        assert mlir.count("nv_tensor_ir.alignment") == 1


def test_compile_accepts_numpy_dlpack_metadata(profiling: bool) -> None:
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
    assert compiled.graph.nodes[compiled.graph.input_ids[0]].value_info.dtype is (
        tir.DataType.F32
    )

    a = torch.from_numpy(a_np).cuda()
    b = torch.from_numpy(b_np).cuda()
    output = torch.empty_like(a)
    _require_runtime_launch()
    _run_kernel(compiled, a, b, output=output, profile=profiling)
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
    assert graph.nodes[result_id].value_info.dtype is tir.DataType.F32


def test_public_api_surface() -> None:
    expected_ops = {
        "abs",
        "add",
        "add_square",
        "atan2",
        "broadcast",
        "ceil",
        "cmp",
        "concatenate",
        "constant",
        "convert",
        "cos",
        "div",
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
        "mul",
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
        "scan_slice",
        "sub",
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
            "KernelProfile",
            "ProfileResult",
            "TensorSpec",
            "CudaTileArtifactKind",
            "ArtifactCache",
            "calculate_cache_key",
            "compile",
            "compile_traced",
            "kernel",
            "trace",
        }
        | expected_ops
    )
    assert not hasattr(tir, "KernelFunction")
    assert not hasattr(tir, "ops")
    assert callable(tir.logical_not)
    assert callable(tir.add)
    assert callable(tir.sub)
    assert callable(tir.mul)
    assert callable(tir.div)
    assert callable(tir.relu)
    assert callable(tir.where)

    @tir.kernel
    def public_api_kernel(a):
        return a

    assert isinstance(public_api_kernel, KernelFunction)


def test_multiple_outputs(profiling: bool) -> None:
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
    _run_kernel(
        compiled,
        a,
        b,
        output=(add_output, sub_output),
        profile=profiling,
    )
    torch.cuda.synchronize()

    add_reference, sub_reference = evaluate_trace_reference(compiled.graph, a, b)
    torch.testing.assert_close(add_output, add_reference, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(sub_output, sub_reference, rtol=1e-3, atol=1e-3)


def test_matmul_and_static_control_flow_examples(profiling: bool) -> None:
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
    _run_assert_trace(compiled, a, b, c, output=output, profile=profiling)


def test_complex_poc_style_kernel(profiling: bool) -> None:
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

    _run_assert_trace(
        compiled,
        a,
        b,
        c,
        threshold,
        output=output,
        profile=profiling,
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("fp32_carry", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "op_name",
    [
        "sub",
        "div",
        "min",
        "max",
        "neg",
        "abs",
        "log",
        "sqrt",
        "rsqrt",
        "reciprocal",
        "sin",
        "cos",
        "tanh",
        "ceil",
        "floor",
    ],
)
def test_scan_slice_float_pointwise(op_name, dtype, reverse, fp32_carry):
    _require_runtime_launch()
    tensor_dtype = {
        torch.float32: tir.DataType.F32,
        torch.float16: tir.DataType.F16,
        torch.bfloat16: tir.DataType.BF16,
    }[dtype]
    carry_dtype = torch.float32 if fp32_carry else dtype
    carry_tensor_dtype = tir.DataType.F32 if fp32_carry else tensor_dtype
    binary = op_name in ("sub", "div", "min", "max")
    tensor_op = getattr(tir, op_name)
    torch_op = {"min": torch.minimum, "max": torch.maximum}.get(
        op_name, getattr(torch, op_name)
    )

    @tir.kernel
    def kernel(values, initial):
        def body(carry, value):
            carry = tir.convert(carry, dtype=tensor_dtype)
            result = tensor_op(carry, value) if binary else carry + tensor_op(value)
            return tir.convert(result, dtype=carry_tensor_dtype), result

        return tir.scan_slice(
            initial,
            (values,),
            body,
            scan_dimensions=(0,),
            output_dimensions=(0,),
            reverse=reverse,
        )

    torch.manual_seed(31)
    values = (torch.rand(5, 3, 8, device="cuda") + 0.5).to(dtype)
    initial = torch.randn(3, 8, device="cuda", dtype=carry_dtype)
    if op_name in ("neg", "abs", "sin", "cos", "tanh", "ceil", "floor"):
        values[:, :, ::2] = -values[:, :, ::2]
    if op_name in ("min", "max"):
        values[1, 0, :4] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0],
            dtype=dtype,
            device="cuda",
        )
    carry = initial
    history = torch.empty_like(values)
    steps = reversed(range(5)) if reverse else range(5)
    for step in steps:
        carry = carry.to(dtype)
        result = (
            torch_op(carry, values[step]) if binary else carry + torch_op(values[step])
        )
        history[step] = result
        carry = result.to(carry_dtype)
    outputs = (torch.empty_like(initial), torch.empty_like(values))
    options = tir.CompileOptions()
    options.tile_sizes = [2, 8]
    compiled = tir.compile(
        kernel,
        values,
        initial,
        output=outputs,
        name=f"scan_slice_{op_name}",
        options=options,
    )
    compiled.run(values, initial, output=outputs)
    tolerance = 2e-5 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps
    for actual, expected in zip(outputs, (carry, history), strict=True):
        torch.testing.assert_close(
            actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
        )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.uint32,
    ],
)
def test_scan_slice_carry_reduction(dtype, reverse):
    _require_runtime_launch()

    @tir.kernel
    def kernel(values, initial):
        def body(carry, value):
            next_carry = carry + value
            total = tir.reduce(next_carry, dimensions=(1,))
            return next_carry, total

        return tir.scan_slice(
            initial,
            (values,),
            body,
            scan_dimensions=(0,),
            output_dimensions=(0,),
            reverse=reverse,
        )

    # Dyadic values keep the reference independent of reduction order while
    # exercising a genuinely low-precision loop carry and reduction.
    values = torch.randint(-8, 9, (5, 2, 8), device="cuda").to(dtype)
    initial = torch.randint(-8, 9, (2, 8), device="cuda").to(dtype)
    if dtype.is_floating_point:
        values = values / 8
        initial = initial / 8
    carry = initial.clone()
    totals = torch.empty(5, 2, 1, device="cuda", dtype=dtype)
    for step in reversed(range(5)) if reverse else range(5):
        if dtype.is_floating_point:
            carry = carry + values[step]
            totals[step] = carry.sum(dim=1, keepdim=True)
        else:
            carry = (carry.to(torch.int64) + values[step].to(torch.int64)).to(dtype)
            totals[step] = carry.to(torch.int64).sum(dim=1, keepdim=True).to(dtype)
    outputs = (torch.empty_like(initial), torch.empty_like(totals))
    compiled = tir.compile(kernel, values, initial, output=outputs, tile_sizes=(1, 8))
    _run_assert_trace(compiled, values, initial, output=outputs, rtol=0, atol=0)
    for actual, expected in zip(outputs, (carry, totals), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ],
)
@pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div", "min", "max", "abs"])
def test_scan_slice_integer_pointwise(dtype, op_name, reverse):
    _require_runtime_launch()
    tensor_op = getattr(tir, op_name)

    @tir.kernel
    def kernel(values, initial):
        def body(carry, value):
            result = tensor_op(carry) if op_name == "abs" else tensor_op(carry, value)
            return result, result

        return tir.scan_slice(
            initial,
            (values,),
            body,
            scan_dimensions=(0,),
            output_dimensions=(0,),
            reverse=reverse,
        )

    # Include the unsigned high bit to distinguish signed/unsigned min, max,
    # division and abs. Keep uint64 within int64 range for the reference.
    unsigned = dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64)
    initial = torch.tensor([[-7, 5, -3, 9], [2, -6, 8, -4]], device="cuda")
    if unsigned:
        initial = initial.abs()
        if dtype != torch.uint64:
            initial[:, ::2] += 1 << (torch.iinfo(dtype).bits - 1)
    initial = initial.to(dtype)
    values = torch.randint(1, 4, (4, 2, 4), device="cuda").to(dtype)
    carry = initial
    history = torch.empty_like(values)
    for step in reversed(range(4)) if reverse else range(4):
        lhs, rhs = carry.to(torch.int64), values[step].to(torch.int64)
        if op_name == "abs":
            result = lhs.abs()
        elif op_name == "div":
            result = torch.div(lhs, rhs, rounding_mode="trunc")
        else:
            operation = {"min": torch.minimum, "max": torch.maximum}.get(
                op_name, getattr(torch, op_name)
            )
            result = operation(lhs, rhs)
        carry = result.to(dtype)
        history[step] = carry
    outputs = (torch.empty_like(initial), torch.empty_like(values))
    compiled = tir.compile(kernel, values, initial, output=outputs, tile_sizes=(1, 4))
    compiled.run(values, initial, output=outputs)
    for actual, expected in zip(outputs, (carry, history), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "input_dtype,carry_dtype,tensor_dtype",
    [
        (torch.int8, torch.int32, tir.DataType.SI32),
        (torch.uint8, torch.int32, tir.DataType.SI32),
        (torch.int64, torch.int32, tir.DataType.SI32),
        (torch.int32, torch.uint32, tir.DataType.UI32),
        (torch.int32, torch.float32, tir.DataType.F32),
        (torch.float32, torch.int32, tir.DataType.SI32),
        (torch.bool, torch.int32, tir.DataType.SI32),
        (torch.int32, torch.bool, tir.DataType.BOOL),
        (torch.float32, torch.bool, tir.DataType.BOOL),
    ],
)
def test_scan_slice_carry_conversion(input_dtype, carry_dtype, tensor_dtype, reverse):
    _require_runtime_launch()

    @tir.kernel
    def kernel(values, initial):
        def body(carry, value):
            converted = tir.convert(value, dtype=tensor_dtype)
            return converted, carry, value

        return tir.scan_slice(
            initial,
            (values,),
            body,
            scan_dimensions=(0,),
            output_dimensions=(0, 0),
            reverse=reverse,
        )

    values = torch.tensor([-3, 0, 2, 7, -8, 1, 0, 4], device="cuda").reshape(4, 1, 2)
    values = values.to(input_dtype)
    if input_dtype == torch.float32:
        values = values + 0.25
    initial = torch.ones(1, 2, device="cuda", dtype=carry_dtype)
    carry = initial
    history = torch.empty(4, 1, 2, device="cuda", dtype=carry_dtype)
    for step in reversed(range(4)) if reverse else range(4):
        history[step] = carry
        carry = values[step].to(carry_dtype)
    outputs = (
        torch.empty_like(initial),
        torch.empty_like(history),
        torch.empty_like(values),
    )
    compiled = tir.compile(kernel, values, initial, output=outputs, tile_sizes=(1, 2))
    compiled.run(values, initial, output=outputs)
    for actual, expected in zip(outputs, (carry, history, values), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
def test_scan_slice_two_inputs_different_axes(reverse, with_history):
    _require_runtime_launch()

    @tir.kernel
    def kernel(values, scales, initial):
        def body(carry, value, scale):
            scale = tir.reshape(scale, shape=(2, 1))
            scale = tir.broadcast(scale, shape=(2, 8))
            next_carry = carry * scale + value
            if with_history:
                total = tir.reduce(next_carry, dimensions=(1,))
                return next_carry, carry, total
            return next_carry

        result = tir.scan_slice(
            initial,
            (values, scales),
            body,
            scan_dimensions=(0, 1),
            output_dimensions=(1, 0) if with_history else (),
            reverse=reverse,
        )
        if with_history:
            final, previous, totals = result
            # Reorder the scan_slice results and consume the final carry downstream.
            return previous, -final, totals
        return -result[0]

    # Different native slice ranks and noncontiguous input strides exercise
    # independent scan axes, step offsets, and body reshape/broadcast handling.
    values = (torch.randint(-8, 9, (5, 2, 16), device="cuda").float() / 8)[:, :, ::2]
    scales = (torch.randint(2, 6, (2, 10), device="cuda").float() / 4)[:, ::2]
    initial = torch.randint(-8, 9, (2, 8), device="cuda").float() / 8
    outputs = (
        (
            initial.new_empty((2, 5, 8)),
            torch.empty_like(initial),
            initial.new_empty((5, 2, 1)),
        )
        if with_history
        else torch.empty_like(initial)
    )
    compiled = tir.compile(
        kernel, values, scales, initial, output=outputs, tile_sizes=(1, 8)
    )
    _run_assert_trace(compiled, values, scales, initial, output=outputs, rtol=0, atol=0)


def test_reduction_reshape_preserves_contracting_axes():
    _require_runtime_launch()

    @tir.kernel
    def kernel(seed, bias):
        reduced = tir.reduce(seed, dimensions=(0, 2))
        result = tir.reshape(reduced, shape=(2, 4))
        return result + tir.transpose(bias, permutation=(1, 0))

    seed = torch.arange(128, device="cuda", dtype=torch.float32).reshape(4, 8, 4)
    bias = torch.arange(8, device="cuda", dtype=torch.float32).reshape(4, 2)
    output = torch.empty((2, 4), device="cuda", dtype=torch.float32)
    compiled = tir.compile(kernel, seed, bias, output=output, tile_sizes=(1, 2))
    _run_assert_trace(compiled, seed, bias, output=output, rtol=0, atol=0)


@pytest.mark.parametrize("cc", [80, 90, 100])
def test_cubin_compilation(cc):
    @tir.kernel
    def _tir_kernel(A, B):
        result = tir.matmul(A, B)
        return result

    options = tir.CompileOptions()
    options.compute_capability = cc
    options.artifact_kind = 1
    if cc >= 90:
        options.arch_portability = ArchPortability.arch_conditional

    a = tir.TensorSpec((8, 8), dtype=tir.DataType.F16)
    b = tir.TensorSpec((8, 8), dtype=tir.DataType.F16)
    c = tir.TensorSpec((8, 8), dtype=tir.DataType.F16)
    tir.compile(_tir_kernel, a, b, output=c, options=options)
