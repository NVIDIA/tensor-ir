# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Mapping

from .dtypes import (
    DataType,
    FLOAT_DTYPES,
    SIGNED_INTEGER_DTYPES,
    UNSIGNED_INTEGER_DTYPES,
    normalize_dtype,
)
from .tracing import BinaryOp, MovementOp, NodeKind, TensorInfo, TraceGraph, UnaryOp

TensorInfoOverrides = Mapping[int, TensorInfo]


def build_mlir_module(
    graph: TraceGraph,
    graph_name: str,
    tensor_info_overrides: TensorInfoOverrides | None = None,
):
    """Build MLIR, optionally overriding tensor metadata by trace node ID."""
    from nv_tensor_ir._mlir import ir
    from nv_tensor_ir._mlir.dialects import nv_tensor_ir

    tensor_info_overrides = tensor_info_overrides or {}
    context = ir.Context()
    # CompiledKernel keeps the MLIR context alive with the generated module.
    # Disable per-context worker threads to avoid accumulating thread pools.
    context.enable_multithreading(False)
    nv_tensor_ir.register_dialect(context, load=True)
    with context, ir.Location.unknown(context):
        module = ir.Module.create()
        values: dict[int, object] = {}
        output_infos = _output_infos(graph, tensor_info_overrides)

        input_types = []
        input_attrs = []
        for node_id in graph.input_ids:
            info = _tensor_info(graph, node_id, tensor_info_overrides)
            input_types.append(_tensor_type(info, ir))
            input_attrs.append(_tensor_attr(info, ir))

        result_types = [_tensor_type(info, ir) for info in output_infos]
        result_attrs = [_tensor_attr(info, ir) for info in output_infos]

        graph_type = ir.FunctionType.get(input_types, result_types)
        with ir.InsertionPoint(module.body):
            graph_op = nv_tensor_ir.GraphOp(
                graph_name,
                ir.TypeAttr.get(graph_type),
                arg_attrs=ir.ArrayAttr.get(input_attrs),
                res_attrs=ir.ArrayAttr.get(result_attrs),
            )

        block = graph_op.regions[0].blocks.append(*input_types)
        for node_id, block_arg in zip(graph.input_ids, block.arguments):
            values[node_id] = block_arg

        with ir.InsertionPoint(block):
            for node_id, node in enumerate(graph.nodes):
                if node.kind == NodeKind.INPUT:
                    continue
                if node.kind == NodeKind.OUTPUT_REF:
                    # OUTPUT_REF nodes define graph result metadata only; they do
                    # not produce SSA values in the TensorIR graph body.
                    continue
                if node.kind == NodeKind.RESULT:
                    if node_id not in values:
                        raise RuntimeError("missing multi-result value")
                    continue
                result_info = _tensor_info(graph, node_id, tensor_info_overrides)
                result_type = _node_result_type(node, result_info, ir)
                result = _build_node(
                    node,
                    result_info,
                    result_type,
                    values,
                    ir,
                    nv_tensor_ir,
                    tensor_info_overrides,
                )
                values[node_id] = result

            nv_tensor_ir.ResultsOp(
                [values[result_id] for result_id in graph.result_ids]
            )

        return module, context


def _build_scan_slice(node, values: dict[int, object], ir, nv_tensor_ir, overrides):
    """Build one isolated scan_slice region and bind all of its SSA results."""
    result_infos = [
        overrides.get(node_id, info)
        for node_id, info in zip(
            node.kwargs["result_node_ids"],
            node.kwargs["result_tensor_infos"],
            strict=True,
        )
    ]
    result_types = [_tensor_type(info, ir) for info in result_infos]
    operation = ir.Operation.create(
        "nv_tensor_ir.scan_slice",
        results=result_types,
        operands=[values[arg] for arg in node.args],
        attributes={
            "scan_dimensions": ir.DenseI64ArrayAttr.get(node.kwargs["scan_dimensions"]),
            "output_dimensions": ir.DenseI64ArrayAttr.get(
                node.kwargs["output_dimensions"]
            ),
            "reverse": ir.BoolAttr.get(bool(node.kwargs["reverse"])),
        },
        regions=1,
    )
    region_graph = node.kwargs["region"]
    argument_types = [
        _tensor_type(region_graph.nodes[node_id].tensor_info, ir)
        for node_id in region_graph.input_ids
    ]
    block = operation.regions[0].blocks.append(*argument_types)
    region_values = dict(zip(region_graph.input_ids, block.arguments, strict=True))
    with ir.InsertionPoint(block):
        for region_node_id, region_node in enumerate(region_graph.nodes):
            if region_node.kind == NodeKind.INPUT:
                continue
            if region_node.kind == NodeKind.RESULT:
                raise NotImplementedError(
                    "Multi-result operations (RESULT nodes) inside a scan_slice body "
                    "are not supported"
                )
            region_info = region_node.tensor_info
            region_type = _node_result_type(region_node, region_info, ir)
            region_values[region_node_id] = _build_node(
                region_node,
                region_info,
                region_type,
                region_values,
                ir,
                nv_tensor_ir,
            )
        nv_tensor_ir.YieldOp(
            [region_values[result_id] for result_id in region_graph.result_ids]
        )
    result_node_ids = node.kwargs["result_node_ids"]
    for result_node_id, result in zip(result_node_ids, operation.results, strict=True):
        values[result_node_id] = result
    return operation.results[0]


def _build_node(
    node,
    result_info: TensorInfo,
    result_type,
    values: dict[int, object],
    ir,
    nv_tensor_ir,
    tensor_info_overrides: TensorInfoOverrides | None = None,
):
    if node.kind == NodeKind.CONSTANT:
        value = _scalar_attr(node.kwargs["value"], result_info.dtype, ir)
        if node.kwargs.get("is_tensor", False):
            value = ir.DenseElementsAttr.get_splat(result_type, value)
        op = nv_tensor_ir.ConstantOp(
            value,
            results=[result_type],
        )
        return op.operation.results[0]
    if node.kind == NodeKind.SPLAT:
        (scalar,) = node.args
        op = nv_tensor_ir.SplatOp(result_type, values[scalar], [])
        return op.operation.results[0]
    if node.kind == NodeKind.BINARY:
        lhs, rhs = node.args
        op = _binary_op(node.op, nv_tensor_ir)(
            values[lhs], values[rhs], results=[result_type]
        )
        return op.operation.results[0]
    if node.kind == NodeKind.UNARY:
        (operand,) = node.args
        if node.op in _UNARY_WITH_BETA_OPS:
            beta = ir.FloatAttr.get_f64(float(node.kwargs["beta"]))
            op = _unary_with_beta_op(node.op, nv_tensor_ir)(
                values[operand],
                beta,
                results=[result_type],
            )
        elif node.op == UnaryOp.EXP and "rounding_mode" in node.kwargs:
            mode = {
                "full": nv_tensor_ir.ExpRoundingMode.Full,
                "approx": nv_tensor_ir.ExpRoundingMode.Approx,
            }[node.kwargs["rounding_mode"]]
            op = nv_tensor_ir.ExpOp(
                values[operand], rounding_mode=mode, results=[result_type]
            )
        else:
            op = _unary_op(node.op, nv_tensor_ir)(
                values[operand], results=[result_type]
            )
        return op.operation.results[0]
    if node.kind == NodeKind.CONVERT:
        (operand,) = node.args
        op = nv_tensor_ir.ConvertOp(result_type, values[operand])
        return op.operation.results[0]
    if node.kind == NodeKind.MOVEMENT:
        (operand,) = node.args
        if node.op == MovementOp.RESHAPE:
            op = nv_tensor_ir.ReshapeOp(result_type, values[operand], [])
        elif node.op == MovementOp.TRANSPOSE:
            op = nv_tensor_ir.TransposeOp(
                result_type, values[operand], node.kwargs["permutation"]
            )
        elif node.op == MovementOp.BROADCAST:
            op = nv_tensor_ir.BroadcastOp(result_type, values[operand], [])
        elif node.op == MovementOp.SLICE:
            op = nv_tensor_ir.SliceOp(
                result_type,
                values[operand],
                node.kwargs["starts"],
                node.kwargs["limits"],
                node.kwargs["strides"],
            )
        else:
            raise NotImplementedError(f"Cannot build TensorIR op {node.op_name}")
        return op.operation.results[0]
    if node.kind == NodeKind.CONCATENATE:
        op = nv_tensor_ir.ConcatenateOp(
            result_type,
            [values[arg] for arg in node.args],
            int(node.kwargs["dimension"]),
        )
        return op.operation.results[0]
    if node.kind == NodeKind.SCAN_SLICE:
        return _build_scan_slice(
            node, values, ir, nv_tensor_ir, tensor_info_overrides or {}
        )
    if node.kind == NodeKind.IOTA:
        op = nv_tensor_ir.IotaOp(result_type, int(node.kwargs["dimension"]), [])
        return op.operation.results[0]
    if node.kind == NodeKind.REDUCE:
        (operand,) = node.args
        op = nv_tensor_ir.ReduceOp(
            result_type,
            values[operand],
            list(node.kwargs["dimensions"]),
            node.kwargs["mode"],
        )
        return op.operation.results[0]
    if node.kind == NodeKind.CMP:
        lhs, rhs = node.args
        op = nv_tensor_ir.CmpOp(
            _comparator(node.kwargs["comparator"], nv_tensor_ir),
            values[lhs],
            values[rhs],
            results=[result_type],
        )
        return op.operation.results[0]
    if node.kind == NodeKind.SELECT:
        selector, lhs, rhs = node.args
        op = nv_tensor_ir.BinarySelectOp(
            values[selector], values[lhs], values[rhs], results=[result_type]
        )
        return op.operation.results[0]
    if node.kind == NodeKind.MATMUL:
        lhs, rhs = node.args
        op = nv_tensor_ir.MatmulOp(result_type, values[lhs], values[rhs])
        return op.operation.results[0]
    raise NotImplementedError(f"Cannot build TensorIR op for node kind {node.kind}")


def _node_result_type(node, result_info: TensorInfo, ir):
    if node.kind == NodeKind.CONSTANT and not node.kwargs.get("is_tensor", False):
        return _scalar_type(result_info.dtype, ir)
    return _tensor_type(result_info, ir)


def _tensor_type(info: TensorInfo, ir):
    dynamic_size = ir.ShapedType.get_dynamic_size()
    shape = [
        dynamic_size if dimension is None else dimension
        for dimension in _ir_shape(info)
    ]
    return ir.RankedTensorType.get(
        shape,
        _scalar_type(info.dtype, ir),
    )


def _scalar_type(dtype: object, ir):
    normalized = normalize_dtype(dtype)
    if normalized == DataType.F32:
        return ir.F32Type.get()
    if normalized == DataType.F64:
        return ir.F64Type.get()
    if normalized == DataType.F16:
        return ir.F16Type.get()
    if normalized == DataType.BF16:
        return ir.BF16Type.get()
    if normalized == DataType.BOOL:
        return ir.IntegerType.get_signless(1)
    if normalized in SIGNED_INTEGER_DTYPES:
        return ir.IntegerType.get_signed(int(normalized.value.removeprefix("si")))
    if normalized in UNSIGNED_INTEGER_DTYPES:
        return ir.IntegerType.get_unsigned(int(normalized.value.removeprefix("ui")))
    raise TypeError(f"Unsupported TensorIR scalar dtype: {normalized}")


def _scalar_attr(value: object, dtype: object, ir):
    normalized = normalize_dtype(dtype)
    ty = _scalar_type(normalized, ir)
    if normalized in FLOAT_DTYPES:
        return ir.FloatAttr.get(ty, float(value))
    if normalized == DataType.BOOL:
        return ir.IntegerAttr.get(ty, int(bool(value)))
    if normalized in SIGNED_INTEGER_DTYPES | UNSIGNED_INTEGER_DTYPES:
        integer_value = int(value)
        if normalized == DataType.UI64 and integer_value > (1 << 63) - 1:
            return ir.Attribute.parse(f"{integer_value} : ui64")
        return ir.IntegerAttr.get(ty, integer_value)
    raise TypeError(f"Unsupported TensorIR scalar dtype: {normalized}")


def _comparator(comparator: object, nv_tensor_ir):
    if not isinstance(comparator, str):
        return comparator
    return getattr(nv_tensor_ir.Comparator, comparator)


def _tensor_attr(info: TensorInfo, ir):
    attrs = {
        "nv_tensor_ir.stride": ir.StringAttr.get(
            _stride_string(_ir_stride(info), info.stride_divisibility)
        )
    }
    if info.alignment is not None:
        attrs["nv_tensor_ir.alignment"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), info.alignment
        )
    return ir.DictAttr.get(attrs)


def _tensor_info(
    graph: TraceGraph,
    node_id: int,
    overrides: TensorInfoOverrides,
) -> TensorInfo:
    return overrides.get(node_id, graph.nodes[node_id].tensor_info)


def _output_infos(
    graph: TraceGraph,
    overrides: TensorInfoOverrides,
) -> list[TensorInfo]:
    infos: list[TensorInfo] = []
    for node_id in graph.output_ref_ids:
        infos.append(_tensor_info(graph, node_id, overrides))
    return infos


def _ir_shape(info: TensorInfo) -> tuple[int | None, ...]:
    if not info.dynamic_shape:
        return info.shape
    return tuple(None for _ in info.shape)


def _ir_stride(info: TensorInfo) -> tuple[int | None, ...]:
    if not info.dynamic_shape:
        return info.stride
    return tuple(1 if stride == 1 else None for stride in info.stride)


def _stride_string(
    values: tuple[int | None, ...],
    divisibility: tuple[int | None, ...] | None = None,
) -> str:
    elements = []
    for index, value in enumerate(values):
        if value is not None:
            elements.append(str(value))
            continue
        divisor = None if divisibility is None else divisibility[index]
        elements.append("?" if divisor is None else f"?{{div={divisor}}}")
    return f"({','.join(elements)})"


def _binary_op(op: BinaryOp, nv_tensor_ir):
    return {
        BinaryOp.ADD: nv_tensor_ir.AddOp,
        BinaryOp.ATAN2: nv_tensor_ir.Atan2Op,
        BinaryOp.SUB: nv_tensor_ir.SubOp,
        BinaryOp.MUL: nv_tensor_ir.MulOp,
        BinaryOp.DIV: nv_tensor_ir.DivOp,
        BinaryOp.MOD: nv_tensor_ir.ModOp,
        BinaryOp.REM: nv_tensor_ir.RemOp,
        BinaryOp.POW: nv_tensor_ir.PowOp,
        BinaryOp.MAX: nv_tensor_ir.MaxOp,
        BinaryOp.MIN: nv_tensor_ir.MinOp,
        BinaryOp.ADD_SQUARE: nv_tensor_ir.AddSquareOp,
        BinaryOp.LOGICAL_AND: nv_tensor_ir.LogicalAndOp,
        BinaryOp.LOGICAL_OR: nv_tensor_ir.LogicalOrOp,
    }[op]


def _unary_op(op: UnaryOp, nv_tensor_ir):
    return {
        UnaryOp.ABS: nv_tensor_ir.AbsOp,
        UnaryOp.CEIL: nv_tensor_ir.CeilOp,
        UnaryOp.COS: nv_tensor_ir.CosOp,
        UnaryOp.EXP: nv_tensor_ir.ExpOp,
        UnaryOp.FLOOR: nv_tensor_ir.FloorOp,
        UnaryOp.LOG: nv_tensor_ir.LogOp,
        UnaryOp.LOGICAL_NOT: nv_tensor_ir.LogicalNotOp,
        UnaryOp.NEG: nv_tensor_ir.NegOp,
        UnaryOp.RECIPROCAL: nv_tensor_ir.ReciprocalOp,
        UnaryOp.SQRT: nv_tensor_ir.SqrtOp,
        UnaryOp.RSQRT: nv_tensor_ir.RsqrtOp,
        UnaryOp.SIN: nv_tensor_ir.SinOp,
        UnaryOp.TAN: nv_tensor_ir.TanOp,
        UnaryOp.RELU_FWD: nv_tensor_ir.ReluFwdOp,
        UnaryOp.SIGMOID_FWD: nv_tensor_ir.SigmoidFwdOp,
        UnaryOp.TANH_FWD: nv_tensor_ir.TanhFwdOp,
        UnaryOp.GELU_APPROX_TANH_FWD: nv_tensor_ir.GeluApproxTanhFwdOp,
    }[op]


_UNARY_WITH_BETA_OPS = {
    UnaryOp.ELU_FWD,
    UnaryOp.SOFTPLUS_FWD,
    UnaryOp.SWISH_FWD,
}


def _unary_with_beta_op(op: UnaryOp, nv_tensor_ir):
    return {
        UnaryOp.ELU_FWD: nv_tensor_ir.EluFwdOp,
        UnaryOp.SOFTPLUS_FWD: nv_tensor_ir.SoftplusFwdOp,
        UnaryOp.SWISH_FWD: nv_tensor_ir.SwishFwdOp,
    }[op]
