# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable, Mapping

from nv_tensor_ir._mlir import ir
from nv_tensor_ir._mlir.dialects import nv_tensor_ir

from .dtypes import (
    DataType,
    FLOAT_DTYPES,
    SIGNED_INTEGER_DTYPES,
    UNSIGNED_INTEGER_DTYPES,
    normalize_dtype,
)
from .tracing import (
    BinaryOp,
    MovementOp,
    NodeKind,
    ScalarInfo,
    TensorInfo,
    TraceGraph,
    TraceOpEnum,
    UnaryOp,
)

OperationBuilder = Callable[..., object]
TensorInfoOverrides = Mapping[int, TensorInfo]

def _elementwise_op(op: BinaryOp | UnaryOp):
    # Tracing enum names follow the generated OpView class names.
    return getattr(nv_tensor_ir, op.name.title().replace("_", "") + "Op")


_UNARY_WITH_BETA_OPS: dict[TraceOpEnum, OperationBuilder] = {
    op: _elementwise_op(op)
    for op in (UnaryOp.ELU_FWD, UnaryOp.SOFTPLUS_FWD, UnaryOp.SWISH_FWD)
}
_UNARY_OPS: dict[TraceOpEnum, OperationBuilder] = {
    op: _elementwise_op(op) for op in UnaryOp if op not in _UNARY_WITH_BETA_OPS
}
_BINARY_OPS: dict[TraceOpEnum, OperationBuilder] = {
    op: _elementwise_op(op) for op in BinaryOp
}
_BINARY_WITH_BETA_OPS: dict[TraceOpEnum, OperationBuilder] = {}


def build_mlir_module(
    graph: TraceGraph,
    graph_name: str,
    tensor_info_overrides: TensorInfoOverrides | None = None,
):
    """Build MLIR, optionally overriding tensor metadata by trace node ID."""
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
            info = _value_info(graph, node_id, tensor_info_overrides)
            input_types.append(_arg_type(info))
            input_attrs.append(_arg_attr(info))

        result_types = [_tensor_type(info) for info in output_infos]
        result_attrs = [_tensor_attr(info) for info in output_infos]

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
                result_info = _value_info(graph, node_id, tensor_info_overrides)
                result_type = _node_result_type(node, result_info)
                result = _build_node(
                    node, result_info, result_type, values, tensor_info_overrides
                )
                values[node_id] = result

            nv_tensor_ir.ResultsOp(
                [values[result_id] for result_id in graph.result_ids]
            )

        return module, context


def _build_scan_slice(node, values: dict[int, object], overrides):
    """Build one isolated scan_slice region and bind all of its SSA results."""
    result_infos = [
        overrides.get(node_id, info)
        for node_id, info in zip(
            node.kwargs["result_node_ids"],
            node.kwargs["result_tensor_infos"],
            strict=True,
        )
    ]
    result_types = [_tensor_type(info) for info in result_infos]
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
        _tensor_type(region_graph.nodes[node_id].value_info)
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
            region_info = region_node.value_info
            region_type = _node_result_type(region_node, region_info)
            region_values[region_node_id] = _build_node(
                region_node,
                region_info,
                region_type,
                region_values,
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
    result_info: TensorInfo | ScalarInfo,
    result_type,
    values: dict[int, object],
    tensor_info_overrides: TensorInfoOverrides | None = None,
):
    if node.kind == NodeKind.CONSTANT:
        value = _scalar_attr(node.kwargs["value"], result_info.dtype)
        if node.kwargs.get("is_tensor", False):
            value = ir.DenseElementsAttr.get_splat(result_type, value)
        op = nv_tensor_ir.ConstantOp(
            value,
            results=[result_type],
        )
    elif node.kind == NodeKind.SPLAT:
        (scalar,) = node.args
        op = nv_tensor_ir.SplatOp(result_type, values[scalar], [])
    elif node.kind == NodeKind.BINARY:
        lhs, rhs = node.args
        if node.op in _BINARY_WITH_BETA_OPS:
            beta = ir.FloatAttr.get_f64(float(node.kwargs["beta"]))
            op = _BINARY_WITH_BETA_OPS[node.op](
                values[lhs], values[rhs], beta, results=[result_type]
            )
        else:
            op = _BINARY_OPS[node.op](values[lhs], values[rhs], results=[result_type])
    elif node.kind == NodeKind.UNARY:
        (operand,) = node.args
        if node.op in _UNARY_WITH_BETA_OPS:
            beta = ir.FloatAttr.get_f64(float(node.kwargs["beta"]))
            op = _UNARY_WITH_BETA_OPS[node.op](
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
            op = _UNARY_OPS[node.op](values[operand], results=[result_type])
    elif node.kind == NodeKind.CONVERT:
        (operand,) = node.args
        op = nv_tensor_ir.ConvertOp(result_type, values[operand])
    elif node.kind == NodeKind.MOVEMENT:
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
    elif node.kind == NodeKind.CONCATENATE:
        op = nv_tensor_ir.ConcatenateOp(
            result_type,
            [values[arg] for arg in node.args],
            int(node.kwargs["dimension"]),
        )
    elif node.kind == NodeKind.SCAN_SLICE:
        return _build_scan_slice(node, values, tensor_info_overrides or {})
    elif node.kind == NodeKind.IOTA:
        op = nv_tensor_ir.IotaOp(result_type, int(node.kwargs["dimension"]), [])
    elif node.kind == NodeKind.REDUCE:
        (operand,) = node.args
        op = nv_tensor_ir.ReduceOp(
            result_type,
            values[operand],
            list(node.kwargs["dimensions"]),
            node.kwargs["mode"],
        )
    elif node.kind == NodeKind.CMP:
        lhs, rhs = node.args
        op = nv_tensor_ir.CmpOp(
            _comparator(node.kwargs["comparator"]),
            values[lhs],
            values[rhs],
            results=[result_type],
        )
    elif node.kind == NodeKind.SELECT:
        selector, lhs, rhs = node.args
        op = nv_tensor_ir.BinarySelectOp(
            values[selector], values[lhs], values[rhs], results=[result_type]
        )
    elif node.kind == NodeKind.MATMUL:
        lhs, rhs = node.args
        op = nv_tensor_ir.MatmulOp(result_type, values[lhs], values[rhs])
    else:
        raise NotImplementedError(f"Cannot build TensorIR op for node kind {node.kind}")
    return op.operation.results[0]


def _node_result_type(node, result_info: TensorInfo | ScalarInfo):
    if isinstance(result_info, ScalarInfo) or (
        node.kind == NodeKind.CONSTANT and not node.kwargs.get("is_tensor", False)
    ):
        return _scalar_type(result_info.dtype)
    return _tensor_type(result_info)


def _arg_type(info: TensorInfo | ScalarInfo):
    """MLIR type for a graph argument: a bare scalar type for by-value
    scalars, or a ranked tensor type otherwise."""
    if isinstance(info, ScalarInfo):
        return _scalar_type(info.dtype)
    return _tensor_type(info)


def _arg_attr(info: TensorInfo | ScalarInfo):
    """Argument dict attr: empty for scalars (no stride to encode)."""
    if isinstance(info, ScalarInfo):
        return ir.DictAttr.get({})
    return _tensor_attr(info)


def _tensor_type(info: TensorInfo):
    dynamic_size = ir.ShapedType.get_dynamic_size()
    shape = [
        dynamic_size if dimension is None else dimension for dimension in info.shape
    ]
    return ir.RankedTensorType.get(
        shape,
        _scalar_type(info.dtype),
    )


def _scalar_type(dtype: object):
    return ir.Type.parse(normalize_dtype(dtype).value)


def _scalar_attr(value: object, dtype: object):
    normalized = normalize_dtype(dtype)
    ty = _scalar_type(normalized)
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


def _comparator(comparator: object):
    if not isinstance(comparator, str):
        return comparator
    return getattr(nv_tensor_ir.Comparator, comparator)


def _tensor_attr(info: TensorInfo):
    attrs = {
        "nv_tensor_ir.stride": ir.StringAttr.get(
            _stride_string(info.stride, info.stride_divisibility)
        )
    }
    if info.alignment is not None:
        attrs["nv_tensor_ir.alignment"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), info.alignment
        )
    return ir.DictAttr.get(attrs)


def _value_info(
    graph: TraceGraph,
    node_id: int,
    overrides: TensorInfoOverrides,
) -> TensorInfo | ScalarInfo:
    return overrides.get(node_id, graph.nodes[node_id].value_info)


def _output_infos(
    graph: TraceGraph,
    overrides: TensorInfoOverrides,
) -> list[TensorInfo]:
    infos: list[TensorInfo] = []
    for node_id in graph.output_ref_ids:
        infos.append(_value_info(graph, node_id, overrides))
    return infos


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


def _register_unary_op(op: TraceOpEnum, builder: OperationBuilder) -> None:
    """Register a unary operation builder."""
    _register_op(_UNARY_OPS, op, builder)


def _register_binary_op(op: TraceOpEnum, builder: OperationBuilder) -> None:
    """Register a binary operation builder."""
    _register_op(_BINARY_OPS, op, builder)


def _register_binary_with_beta_op(op: TraceOpEnum, builder: OperationBuilder) -> None:
    """Register a binary operation builder with a beta attribute."""
    _register_op(_BINARY_WITH_BETA_OPS, op, builder)


def _register_op(
    registry: dict[TraceOpEnum, OperationBuilder],
    op: TraceOpEnum,
    builder: OperationBuilder,
) -> None:
    if not isinstance(op, TraceOpEnum):
        raise TypeError("Registered TensorIR operations must be TraceOpEnum values")
    if op in registry:
        raise ValueError(f"TensorIR operation {op} is already registered")
    registry[op] = builder
