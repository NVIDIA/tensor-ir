# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON -m pytest --capture=no -vv %s

"""Per-exp accuracy policy: tracing, ODS bindings, cache identity and lowering."""

import io

import pytest
from nv_tensor_ir import dsl as tir
from nv_tensor_ir import runtime
from nv_tensor_ir._mlir import ir
from nv_tensor_ir._mlir.dialects import nv_tensor_ir
from nv_tensor_ir.dsl.module_builder import build_mlir_module
from nv_tensor_ir.dsl.tracing import TensorInfo, TraceGraph


def make_graph(mode: str | None, dtype: tir.DataType = tir.DataType.F32) -> TraceGraph:
    graph = TraceGraph()
    value = graph.add_input(TensorInfo("x", dtype, (128,), (1,)))
    result = tir.exp(value) if mode is None else tir.exp(value, rounding_mode=mode)
    graph.add_output_ref(result.tensor_info)
    graph.set_results([result])
    return graph


@pytest.mark.parametrize("mode", [None, "full", "approx"])
@pytest.mark.parametrize(
    "dtype", [tir.DataType.F16, tir.DataType.BF16, tir.DataType.F32]
)
def test_exp_rounding_mode(mode, dtype, tmp_path):
    graph = make_graph(mode, dtype)
    assert graph.nodes[1].kwargs == {"rounding_mode": mode or "full"}
    module, context = build_mlir_module(graph, "exp_policy")
    with context:
        assert module.operation.verify()
        source = str(module)
        assert ("rounding_mode" in source) == (mode == "approx")
        if mode == "approx":
            exp_op = module.body.operations[0].regions[0].blocks[0].operations[0]
            assert (
                str(exp_op.attributes["rounding_mode"])
                == "#nv_tensor_ir.exp_rounding_mode<approx>"
            )
        # Text round-tripping must preserve the semantic attribute.
        assert str(ir.Module.parse(source)) == source
        bytecode = io.BytesIO()
        module.operation.write_bytecode(bytecode)
        assert str(ir.Module.parse(bytecode.getvalue())) == source
        dump = tmp_path / "generated.mlir"
        with runtime.compile(
            module,
            compute_capability=100,
            tile_sizes=[128],
            dump_cuda_tile_ir_path=str(dump),
        ) as program:
            assert program.get_bytecode()
    lowered = dump.read_text()
    exp_lines = [line for line in lowered.splitlines() if " = exp " in line]
    assert len(exp_lines) == 1
    expected_approx = mode == "approx"
    assert ("rounding<approx>" in exp_lines[0]) == expected_approx
    assert "xf32>" in exp_lines[0]


@pytest.mark.parametrize("mode", [None, "full"])
def test_fp64_exp_policy(mode):
    module, context = build_mlir_module(make_graph(mode, tir.DataType.F64), "fp64_exp")
    with context:
        assert module.operation.verify()


@pytest.mark.parametrize("mode", ["fast", "nearest_even", "FULL", ""])
def test_invalid_exp_policy(mode):
    with pytest.raises(ValueError, match="rounding_mode"):
        make_graph(mode)


@pytest.mark.parametrize("mode", [True, 1, 0.0, []])
def test_invalid_exp_policy_type(mode):
    with pytest.raises(TypeError, match="rounding_mode"):
        make_graph(mode)


def test_approx_fp64_rejected():
    with pytest.raises(ValueError, match="FP64"):
        make_graph("approx", tir.DataType.F64)


def test_exp_policy_requires_float():
    with pytest.raises(TypeError, match="floating-point"):
        make_graph("full", tir.DataType.SI32)


def test_exp_policy_changes_cache_identity():
    keys = []
    for mode in (None, "full", "approx"):
        module, context = build_mlir_module(make_graph(mode), "same_name")
        with context:
            keys.append(tir.calculate_cache_key(module))
    assert keys[0] == keys[1]
    assert keys[1] != keys[2]


def test_generated_exp_builder_accepts_mode():
    with ir.Context() as context, ir.Location.unknown():
        nv_tensor_ir.register_dialect(context, load=True)
        value_type = ir.RankedTensorType.get((128,), ir.F32Type.get())
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            source = nv_tensor_ir.ConstantOp(
                ir.DenseElementsAttr.get_splat(value_type, ir.FloatAttr.get_f32(1.0)),
                results=[value_type],
            )
            op = nv_tensor_ir.ExpOp(
                source.result,
                rounding_mode=nv_tensor_ir.ExpRoundingMode.Approx,
                results=[value_type],
            )
            assert op.operation.verify()
            assert (
                str(op.operation.attributes["rounding_mode"])
                == "#nv_tensor_ir.exp_rounding_mode<approx>"
            )


@pytest.mark.parametrize("mode", ["full", "approx"])
def test_pre_13_3_bytecode_policy(mode, capfd):
    module, context = build_mlir_module(make_graph(mode), "old_bytecode")
    options = runtime.CompileOptions.parse("bytecode-version=13.2")
    options.compute_capability = 100
    options.tile_sizes = [128]
    with context:
        if mode == "full":
            with runtime.compile(module, options=options) as program:
                assert program.get_bytecode()
        else:
            with pytest.raises(
                RuntimeError, match="Failed to lower TensorIR to CudaTile"
            ):
                runtime.compile(module, options=options)
            assert "requires CUDA Tile bytecode version 13.3" in capfd.readouterr().err
