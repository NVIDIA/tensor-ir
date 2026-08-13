# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

from .dtypes import DataType, normalize_dtype, supports_scalar_attr
from .tracing import (
    BinaryOp,
    MovementOp,
    NodeKind,
    TensorInfo,
    TracedTensor,
    UnaryOp,
    _check_binary_tensor_types,
    _check_same_graph,
    _check_same_tensor_shape,
    _check_same_tensor_type,
    _cmp as _trace_cmp,
    _contiguous_stride,
    _current_trace_graph,
    _matmul as _trace_matmul,
)

__all__ = [
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
]


def _ensure_traced(value: object, op_name: str) -> TracedTensor:
    if not isinstance(value, TracedTensor):
        raise TypeError(f"tir.{op_name} expects a traced tensor")
    return value


def _scalar_info(dtype: DataType) -> TensorInfo:
    return TensorInfo(
        name="constant",
        dtype=dtype,
        shape=(),
        stride=(),
    )


def _constant(graph, dtype: DataType, value: object) -> TracedTensor:
    return graph.add_node(
        NodeKind.CONSTANT,
        "constant",
        (),
        _scalar_info(dtype),
        kwargs={"value": value},
    )


def _shape_tuple(shape: object, op_name: str) -> tuple[int, ...]:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"tir.{op_name} shape must be a list or tuple")
    parsed = tuple(int(dim) for dim in shape)
    if not parsed or any(dim <= 0 for dim in parsed):
        raise TypeError(f"tir.{op_name} shape dimensions must be positive")
    return parsed


def _num_elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _permutation_tuple(permutation: object, rank: int) -> tuple[int, ...]:
    if not isinstance(permutation, (list, tuple)):
        raise TypeError("tir.transpose permutation must be a list or tuple")
    parsed = tuple(int(dim) for dim in permutation)
    if sorted(parsed) != list(range(rank)):
        raise TypeError("tir.transpose permutation must be a valid permutation")
    return parsed


def _index_tuple(values: object, op_name: str, attr_name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"tir.{op_name} {attr_name} must be a list or tuple")
    return tuple(int(value) for value in values)


def _dimension(value: object, rank: int, op_name: str) -> int:
    dimension = int(value)
    if dimension < 0 or dimension >= rank:
        raise TypeError(f"tir.{op_name} dimension must be in range [0, {rank})")
    return dimension


def _dimensions(values: object, rank: int, op_name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"tir.{op_name} dimensions must be a list or tuple")
    parsed = tuple(_dimension(value, rank, op_name) for value in values)
    if not parsed:
        raise TypeError(f"tir.{op_name} dimensions must not be empty")
    if len(set(parsed)) != len(parsed):
        raise TypeError(f"tir.{op_name} dimensions must be unique")
    return parsed


def _unary(op: UnaryOp, value: object) -> TracedTensor:
    op_name = op.value
    tensor = _ensure_traced(value, op_name)
    return tensor._graph.add_node(NodeKind.UNARY, op, (tensor._id,), tensor.tensor_info)


def _unary_with_beta(op: UnaryOp, value: object, beta: float) -> TracedTensor:
    op_name = op.value
    tensor = _ensure_traced(value, op_name)
    return tensor._graph.add_node(
        NodeKind.UNARY,
        op,
        (tensor._id,),
        tensor.tensor_info,
        kwargs={"beta": float(beta)},
    )


def _binary(op: BinaryOp, lhs: object, rhs: object) -> TracedTensor:
    op_name = op.value
    lhs_tensor = _ensure_traced(lhs, op_name)
    rhs_tensor = _ensure_traced(rhs, op_name)
    _check_same_graph(lhs_tensor, rhs_tensor, op_name)
    _check_binary_tensor_types(lhs_tensor.tensor_info, rhs_tensor.tensor_info, op)
    return lhs_tensor._graph.add_node(
        NodeKind.BINARY,
        op,
        (lhs_tensor._id, rhs_tensor._id),
        lhs_tensor.tensor_info,
    )


def _logical_binary(op: BinaryOp, lhs: object, rhs: object) -> TracedTensor:
    op_name = op.value
    lhs_tensor = _ensure_traced(lhs, op_name)
    rhs_tensor = _ensure_traced(rhs, op_name)
    _check_same_graph(lhs_tensor, rhs_tensor, op_name)
    _check_same_tensor_type(lhs_tensor.tensor_info, rhs_tensor.tensor_info, op_name)
    if lhs_tensor.tensor_info.dtype != DataType.BOOL:
        raise TypeError(f"tir.{op_name} expects i1 tensor operands")
    return lhs_tensor._graph.add_node(
        NodeKind.BINARY,
        op,
        (lhs_tensor._id, rhs_tensor._id),
        lhs_tensor.tensor_info,
    )


def _logical_not(value: object) -> TracedTensor:
    tensor = _ensure_traced(value, "logical_not")
    if tensor.tensor_info.dtype != DataType.BOOL:
        raise TypeError("tir.logical_not expects an i1 tensor operand")
    return tensor._graph.add_node(
        NodeKind.UNARY,
        UnaryOp.LOGICAL_NOT,
        (tensor._id,),
        tensor.tensor_info,
    )


def relu(value: object) -> TracedTensor:
    return _unary(UnaryOp.RELU_FWD, value)


def abs(value: object) -> TracedTensor:
    return _unary(UnaryOp.ABS, value)


def ceil(value: object) -> TracedTensor:
    return _unary(UnaryOp.CEIL, value)


def cos(value: object) -> TracedTensor:
    return _unary(UnaryOp.COS, value)


def exp(value: object) -> TracedTensor:
    return _unary(UnaryOp.EXP, value)


def floor(value: object) -> TracedTensor:
    return _unary(UnaryOp.FLOOR, value)


def log(value: object) -> TracedTensor:
    return _unary(UnaryOp.LOG, value)


def neg(value: object) -> TracedTensor:
    return _unary(UnaryOp.NEG, value)


def sqrt(value: object) -> TracedTensor:
    return _unary(UnaryOp.SQRT, value)


def reciprocal(value: object) -> TracedTensor:
    return _unary(UnaryOp.RECIPROCAL, value)


def rsqrt(value: object) -> TracedTensor:
    return _unary(UnaryOp.RSQRT, value)


def sin(value: object) -> TracedTensor:
    return _unary(UnaryOp.SIN, value)


def tan(value: object) -> TracedTensor:
    return _unary(UnaryOp.TAN, value)


def sigmoid(value: object) -> TracedTensor:
    return _unary(UnaryOp.SIGMOID_FWD, value)


def tanh(value: object) -> TracedTensor:
    return _unary(UnaryOp.TANH_FWD, value)


def gelu_approx_tanh(value: object) -> TracedTensor:
    return _unary(UnaryOp.GELU_APPROX_TANH_FWD, value)


def softplus(value: object, *, beta: float = 1.0) -> TracedTensor:
    """Apply TensorIR softplus with the given beta parameter."""
    return _unary_with_beta(UnaryOp.SOFTPLUS_FWD, value, beta)


def swish(value: object, *, beta: float = 1.0) -> TracedTensor:
    """Apply TensorIR swish: ``x * sigmoid(beta * x)``."""
    return _unary_with_beta(UnaryOp.SWISH_FWD, value, beta)


def elu(value: object, *, beta: float = 1.0) -> TracedTensor:
    """Apply TensorIR ELU; beta corresponds to PyTorch's ``alpha``."""
    return _unary_with_beta(UnaryOp.ELU_FWD, value, beta)


def convert(value: object, *, dtype: object) -> TracedTensor:
    tensor = _ensure_traced(value, "convert")
    result_info = replace(tensor.tensor_info, dtype=normalize_dtype(dtype))
    return tensor._graph.add_node(
        NodeKind.CONVERT, "convert", (tensor._id,), result_info
    )


def reshape(value: object, *, shape: object) -> TracedTensor:
    tensor = _ensure_traced(value, "reshape")
    result_shape = _shape_tuple(shape, "reshape")
    if _num_elements(result_shape) != _num_elements(tensor.tensor_info.shape):
        raise TypeError("reshape must preserve the number of elements")
    result_info = replace(
        tensor.tensor_info,
        shape=result_shape,
        stride=_contiguous_stride(result_shape),
    )
    return tensor._graph.add_node(
        NodeKind.MOVEMENT,
        MovementOp.RESHAPE,
        (tensor._id,),
        result_info,
    )


def transpose(value: object, *, permutation: object) -> TracedTensor:
    tensor = _ensure_traced(value, "transpose")
    permutation_tuple = _permutation_tuple(permutation, len(tensor.tensor_info.shape))
    result_info = replace(
        tensor.tensor_info,
        shape=tuple(tensor.tensor_info.shape[dim] for dim in permutation_tuple),
        stride=tuple(tensor.tensor_info.stride[dim] for dim in permutation_tuple),
    )
    return tensor._graph.add_node(
        NodeKind.MOVEMENT,
        MovementOp.TRANSPOSE,
        (tensor._id,),
        result_info,
        kwargs={"permutation": permutation_tuple},
    )


def broadcast(value: object, *, shape: object) -> TracedTensor:
    tensor = _ensure_traced(value, "broadcast")
    result_shape = _shape_tuple(shape, "broadcast")
    if len(result_shape) != len(tensor.tensor_info.shape):
        raise TypeError("broadcast input and output must have the same rank")

    changed_dim = False
    for input_dim, output_dim in zip(tensor.tensor_info.shape, result_shape):
        if input_dim == output_dim:
            continue
        if input_dim == 1 and output_dim > 1:
            changed_dim = True
            continue
        raise TypeError("broadcast can only expand dimensions whose input size is 1")
    if not changed_dim:
        raise TypeError("broadcast must expand at least one dimension")

    result_info = replace(
        tensor.tensor_info,
        shape=result_shape,
        stride=_contiguous_stride(result_shape),
    )
    return tensor._graph.add_node(
        NodeKind.MOVEMENT,
        MovementOp.BROADCAST,
        (tensor._id,),
        result_info,
    )


def slice(
    value: object,
    *,
    starts: object,
    limits: object,
    strides: object | None = None,
) -> TracedTensor:
    tensor = _ensure_traced(value, "slice")
    rank = len(tensor.tensor_info.shape)
    start_values = _index_tuple(starts, "slice", "starts")
    limit_values = _index_tuple(limits, "slice", "limits")
    stride_values = (
        (1,) * rank if strides is None else _index_tuple(strides, "slice", "strides")
    )
    if len(start_values) != rank or len(limit_values) != rank:
        raise TypeError("slice starts and limits must match input rank")
    if len(stride_values) != rank:
        raise TypeError("slice strides must match input rank")

    result_shape: list[int] = []
    for dim, start, limit, stride in zip(
        tensor.tensor_info.shape, start_values, limit_values, stride_values
    ):
        if start < 0 or limit < 0:
            raise TypeError("slice starts and limits must be non-negative")
        if stride <= 0:
            raise TypeError("slice strides must be positive")
        if limit <= start:
            raise TypeError("slice limits must be greater than starts")
        if limit > dim:
            raise TypeError("slice limits must be within the input shape")
        result_shape.append(1 + (limit - 1 - start) // stride)

    result_info = replace(
        tensor.tensor_info,
        shape=tuple(result_shape),
        stride=_contiguous_stride(tuple(result_shape)),
    )
    return tensor._graph.add_node(
        NodeKind.MOVEMENT,
        MovementOp.SLICE,
        (tensor._id,),
        result_info,
        kwargs={
            "starts": start_values,
            "limits": limit_values,
            "strides": stride_values,
        },
    )


def concatenate(values: object, *, dimension: int) -> TracedTensor:
    if not isinstance(values, (list, tuple)):
        raise TypeError("tir.concatenate values must be a list or tuple")
    if len(values) < 2:
        raise TypeError("tir.concatenate expects at least two tensors")

    tensors = tuple(_ensure_traced(value, "concatenate") for value in values)
    first = tensors[0]
    rank = len(first.tensor_info.shape)
    concat_dim = _dimension(dimension, rank, "concatenate")

    result_shape = list(first.tensor_info.shape)
    for tensor in tensors[1:]:
        _check_same_graph(first, tensor, "concatenate")
        if tensor.tensor_info.dtype != first.tensor_info.dtype:
            raise TypeError("concatenate operands must have the same dtype")
        if len(tensor.tensor_info.shape) != rank:
            raise TypeError("concatenate operands must have the same rank")
        for dim, (expected, actual) in enumerate(
            zip(first.tensor_info.shape, tensor.tensor_info.shape, strict=True)
        ):
            if dim == concat_dim:
                continue
            if expected != actual:
                raise TypeError(
                    "concatenate operands must match outside the concat dimension"
                )
        result_shape[concat_dim] += tensor.tensor_info.shape[concat_dim]

    shape = tuple(result_shape)
    result_info = replace(
        first.tensor_info,
        shape=shape,
        stride=_contiguous_stride(shape),
    )
    return first._graph.add_node(
        NodeKind.CONCATENATE,
        "concatenate",
        tuple(tensor._id for tensor in tensors),
        result_info,
        kwargs={"dimension": concat_dim},
    )


def iota_like(like: object, *, dimension: int) -> TracedTensor:
    like_tensor = _ensure_traced(like, "iota_like")
    if like_tensor.tensor_info.dtype == DataType.BOOL:
        raise TypeError("iota_like does not support bool tensors")
    iota_dim = _dimension(dimension, len(like_tensor.tensor_info.shape), "iota_like")
    return like_tensor._graph.add_node(
        NodeKind.IOTA,
        "iota",
        (like_tensor._id,),
        like_tensor.tensor_info,
        kwargs={"dimension": iota_dim},
    )


def iota(*, shape: object, dtype: DataType, dimension: int) -> TracedTensor:
    """Create an iota tensor without adding a graph input."""
    iota_shape = _shape_tuple(shape, "iota")
    iota_dtype = normalize_dtype(dtype)
    if iota_dtype == DataType.BOOL:
        raise TypeError("iota does not support bool tensors")
    iota_dim = _dimension(dimension, len(iota_shape), "iota")
    return _current_trace_graph("iota").add_node(
        NodeKind.IOTA,
        "iota",
        (),
        TensorInfo(
            name="iota",
            dtype=iota_dtype,
            shape=iota_shape,
            stride=_contiguous_stride(iota_shape),
        ),
        kwargs={"dimension": iota_dim},
    )


_REDUCTION_MODES = {
    "add",
    "amax",
    "avg",
    "max",
    "min",
    "mul",
    "mul_no_zeros",
    "norm1",
    "norm2",
}


def reduce(value: object, *, dimensions: object, mode: str = "add") -> TracedTensor:
    """Reduce a tensor over dimensions while keeping reduced dimensions.

    Supported modes are ``add`` (sum), ``avg`` (mean), ``max``, ``min``,
    ``mul`` (product), ``mul_no_zeros`` (product with zeros treated as one),
    ``amax`` (max of absolute values), ``norm1`` (sum of absolute values), and
    ``norm2`` (sqrt of summed squares).
    """
    tensor = _ensure_traced(value, "reduce")
    if tensor.tensor_info.dtype == DataType.BOOL:
        raise TypeError("reduce does not support bool tensors")
    if mode not in _REDUCTION_MODES:
        raise TypeError(
            f"reduce mode must be one of {sorted(_REDUCTION_MODES)}, got {mode!r}"
        )
    reduce_dims = _dimensions(dimensions, len(tensor.tensor_info.shape), "reduce")
    result_shape = list(tensor.tensor_info.shape)
    for dim in reduce_dims:
        result_shape[dim] = 1
    shape = tuple(result_shape)
    result_info = replace(
        tensor.tensor_info,
        shape=shape,
        stride=_contiguous_stride(shape),
    )
    return tensor._graph.add_node(
        NodeKind.REDUCE,
        "reduce",
        (tensor._id,),
        result_info,
        kwargs={"dimensions": reduce_dims, "mode": mode},
    )


def full_like(like: object, *, value: object) -> TracedTensor:
    """Create a tensor filled with a Python scalar and shaped like ``like``.

    ``value`` must be a Python scalar. TracedTensor scalar operands are not part
    of the public DSL until scalar-consuming operations are added.
    """
    like_tensor = _ensure_traced(like, "full_like")
    if isinstance(value, TracedTensor):
        raise TypeError("full_like value must be a Python scalar")
    if not supports_scalar_attr(like_tensor.tensor_info.dtype):
        raise TypeError(
            "full_like does not support scalar constants for tensor dtype "
            f"{like_tensor.tensor_info.dtype}"
        )
    scalar = _constant(like_tensor._graph, like_tensor.tensor_info.dtype, value)
    return like_tensor._graph.add_node(
        NodeKind.SPLAT,
        "splat",
        (scalar._id,),
        like_tensor.tensor_info,
        kwargs={"like": like_tensor._id},
    )


def constant(value: object, *, shape: object, dtype: DataType) -> TracedTensor:
    """Create a statically shaped dense tensor constant."""
    if isinstance(value, TracedTensor):
        raise TypeError("constant value must be a Python scalar")
    constant_shape = _shape_tuple(shape, "constant")
    constant_dtype = normalize_dtype(dtype)
    return _current_trace_graph("constant").add_node(
        NodeKind.CONSTANT,
        "constant",
        (),
        TensorInfo(
            name="constant",
            dtype=constant_dtype,
            shape=constant_shape,
            stride=_contiguous_stride(constant_shape),
        ),
        kwargs={"value": value, "is_tensor": True},
    )


def splat(value: object, *, shape: object, dtype: DataType) -> TracedTensor:
    """Splat a scalar value into a statically shaped tensor."""
    if isinstance(value, TracedTensor):
        raise TypeError("splat value must be a Python scalar")
    splat_shape = _shape_tuple(shape, "splat")
    splat_dtype = normalize_dtype(dtype)
    graph = _current_trace_graph("splat")
    scalar = _constant(graph, splat_dtype, value)
    return graph.add_node(
        NodeKind.SPLAT,
        "splat",
        (scalar._id,),
        TensorInfo(
            name="splat",
            dtype=splat_dtype,
            shape=splat_shape,
            stride=_contiguous_stride(splat_shape),
        ),
    )


def cmp(lhs: object, rhs: object, *, predicate: str) -> TracedTensor:
    lhs_tensor = _ensure_traced(lhs, "cmp")
    rhs_tensor = _ensure_traced(rhs, "cmp")
    return _trace_cmp(lhs_tensor, rhs_tensor, predicate)


def max(lhs: object, rhs: object) -> TracedTensor:
    return _binary(BinaryOp.MAX, lhs, rhs)


def min(lhs: object, rhs: object) -> TracedTensor:
    return _binary(BinaryOp.MIN, lhs, rhs)


def add_square(lhs: object, rhs: object) -> TracedTensor:
    return _binary(BinaryOp.ADD_SQUARE, lhs, rhs)


def atan2(lhs: object, rhs: object) -> TracedTensor:
    """Compute elementwise ``atan2(lhs, rhs)``."""
    return _binary(BinaryOp.ATAN2, lhs, rhs)


def mod(lhs: object, rhs: object) -> TracedTensor:
    """Remainder with Python ``%`` / ``torch.remainder`` sign semantics."""
    return _binary(BinaryOp.MOD, lhs, rhs)


def rem(lhs: object, rhs: object) -> TracedTensor:
    """Remainder with C ``fmod`` / ``torch.fmod`` sign semantics."""
    return _binary(BinaryOp.REM, lhs, rhs)


def pow(lhs: object, rhs: object) -> TracedTensor:
    return _binary(BinaryOp.POW, lhs, rhs)


def logical_and(lhs: object, rhs: object) -> TracedTensor:
    return _logical_binary(BinaryOp.LOGICAL_AND, lhs, rhs)


def logical_not(value: object) -> TracedTensor:
    return _logical_not(value)


def logical_or(lhs: object, rhs: object) -> TracedTensor:
    return _logical_binary(BinaryOp.LOGICAL_OR, lhs, rhs)


def where(selector: object, lhs: object, rhs: object) -> TracedTensor:
    selector_tensor = _ensure_traced(selector, "where")
    lhs_tensor = _ensure_traced(lhs, "where")
    rhs_tensor = _ensure_traced(rhs, "where")
    _check_same_graph(selector_tensor, lhs_tensor, "where")
    _check_same_graph(lhs_tensor, rhs_tensor, "where")
    if selector_tensor.tensor_info.dtype != DataType.BOOL:
        raise TypeError("where selector must be an i1 tensor")
    _check_same_tensor_shape(
        selector_tensor.tensor_info, lhs_tensor.tensor_info, "where"
    )
    _check_same_tensor_type(lhs_tensor.tensor_info, rhs_tensor.tensor_info, "where")
    return lhs_tensor._graph.add_node(
        NodeKind.SELECT,
        "binary_select",
        (selector_tensor._id, lhs_tensor._id, rhs_tensor._id),
        lhs_tensor.tensor_info,
    )


def matmul(lhs: object, rhs: object, *, dtype: object | None = None) -> TracedTensor:
    """Multiply tensors, optionally selecting the result element type."""
    lhs_tensor = _ensure_traced(lhs, "matmul")
    rhs_tensor = _ensure_traced(rhs, "matmul")
    return _trace_matmul(lhs_tensor, rhs_tensor, dtype)
