# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum, auto

from .dtypes import DataType, is_float_dtype, is_integer_dtype, normalize_dtype


class NodeKind(Enum):
    INPUT = auto()
    OUTPUT_REF = auto()
    CONSTANT = auto()
    SPLAT = auto()
    UNARY = auto()
    BINARY = auto()
    CONVERT = auto()
    CMP = auto()
    MOVEMENT = auto()
    CONCATENATE = auto()
    IOTA = auto()
    REDUCE = auto()
    SELECT = auto()
    MATMUL = auto()


class UnaryOp(Enum):
    ABS = "abs"
    CEIL = "ceil"
    COS = "cos"
    ELU_FWD = "elu_fwd"
    EXP = "exp"
    FLOOR = "floor"
    GELU_APPROX_TANH_FWD = "gelu_approx_tanh_fwd"
    LOG = "log"
    LOGICAL_NOT = "not"
    NEG = "neg"
    RECIPROCAL = "reciprocal"
    RELU_FWD = "relu_fwd"
    RSQRT = "rsqrt"
    SIGMOID_FWD = "sigmoid_fwd"
    SIN = "sin"
    SOFTPLUS_FWD = "softplus_fwd"
    SQRT = "sqrt"
    SWISH_FWD = "swish_fwd"
    TAN = "tan"
    TANH_FWD = "tanh_fwd"


class BinaryOp(Enum):
    ADD = "add"
    ADD_SQUARE = "add_square"
    ATAN2 = "atan2"
    DIV = "div"
    LOGICAL_AND = "and"
    LOGICAL_OR = "or"
    MAX = "max"
    MIN = "min"
    MOD = "mod"
    MUL = "mul"
    POW = "pow"
    REM = "rem"
    SUB = "sub"


class MovementOp(Enum):
    BROADCAST = "broadcast"
    RESHAPE = "reshape"
    SLICE = "slice"
    TRANSPOSE = "transpose"


TraceOp = UnaryOp | BinaryOp | MovementOp | str


def trace_op_name(op: TraceOp) -> str:
    if isinstance(op, (UnaryOp, BinaryOp, MovementOp)):
        return op.value
    return op


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: DataType
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dynamic_shape: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))


@dataclass
class TraceNode:
    kind: NodeKind
    op: TraceOp
    tensor_info: TensorInfo
    args: tuple[int, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)

    @property
    def op_name(self) -> str:
        return trace_op_name(self.op)


class TraceGraph:
    def __init__(self):
        self.nodes: list[TraceNode] = []
        self.input_ids: list[int] = []
        self.output_ref_ids: list[int] = []
        self.result_ids: list[int] = []

    def add_input(self, tensor_info: TensorInfo) -> TracedTensor:
        node_id = len(self.nodes)
        self.nodes.append(TraceNode(NodeKind.INPUT, "input", tensor_info))
        self.input_ids.append(node_id)
        return TracedTensor(self, node_id)

    def add_output_ref(self, tensor_info: TensorInfo) -> TracedTensor:
        node_id = len(self.nodes)
        self.nodes.append(TraceNode(NodeKind.OUTPUT_REF, "output_ref", tensor_info))
        self.output_ref_ids.append(node_id)
        return TracedTensor(self, node_id)

    def add_node(
        self,
        kind: NodeKind,
        op: TraceOp,
        args: tuple[int, ...],
        tensor_info: TensorInfo,
        kwargs: dict[str, object] | None = None,
    ) -> TracedTensor:
        node_id = len(self.nodes)
        op_name = trace_op_name(op)
        tensor_info = replace(tensor_info, name=f"{op_name}_{node_id}")
        self.nodes.append(
            TraceNode(
                kind,
                op,
                tensor_info=tensor_info,
                args=args,
                kwargs=kwargs or {},
            )
        )
        return TracedTensor(self, node_id)

    def set_results(self, results: list[object]) -> None:
        if len(results) != len(self.output_ref_ids):
            raise TypeError(
                "Kernel must return exactly one traced tensor for each output tensor"
            )
        for result, output_ref_id in zip(results, self.output_ref_ids):
            if not isinstance(result, TracedTensor):
                raise TypeError("Kernel must return traced tensor values")
            if result._graph is not self:
                raise TypeError("Kernel returned a traced tensor from another graph")
            if result._id in self.output_ref_ids:
                raise TypeError("Kernel must return computed tensors, not output refs")
            _check_same_tensor_type(
                result.tensor_info,
                self.nodes[output_ref_id].tensor_info,
                "kernel result",
            )
        self.result_ids = [result._id for result in results]

    def __repr__(self) -> str:
        lines = ["TraceGraph:"]
        for node_id, node in enumerate(self.nodes):
            tag = ""
            if node_id in self.input_ids:
                tag = " [INPUT]"
            elif node_id in self.output_ref_ids:
                tag = " [OUTPUT_REF]"
            elif node_id in self.result_ids:
                tag = " [RESULT]"
            lines.append(f"  %{node_id} = {node.op_name}{node.args}{tag}")
        return "\n".join(lines)


_CURRENT_TRACE_GRAPH: ContextVar[TraceGraph | None] = ContextVar(
    "tensor_ir_current_trace_graph", default=None
)


@contextmanager
def _trace_graph_context(graph: TraceGraph) -> Iterator[None]:
    """Expose the active graph to operand-free DSL ops during kernel tracing."""
    token = _CURRENT_TRACE_GRAPH.set(graph)
    try:
        yield
    finally:
        _CURRENT_TRACE_GRAPH.reset(token)


def _current_trace_graph(op_name: str) -> TraceGraph:
    graph = _CURRENT_TRACE_GRAPH.get()
    if graph is None:
        raise RuntimeError(f"tir.{op_name} must be called from a @tir.kernel function")
    return graph


class TracedTensor:
    def __init__(self, graph: TraceGraph, node_id: int):
        self._graph = graph
        self._id = node_id

    @property
    def tensor_info(self) -> TensorInfo:
        return self._graph.nodes[self._id].tensor_info

    def __add__(self, other: TracedTensor) -> TracedTensor:
        return _binary(self, other, BinaryOp.ADD)

    def __sub__(self, other: TracedTensor) -> TracedTensor:
        return _binary(self, other, BinaryOp.SUB)

    def __mul__(self, other: TracedTensor) -> TracedTensor:
        return _binary(self, other, BinaryOp.MUL)

    def __truediv__(self, other: TracedTensor) -> TracedTensor:
        return _binary(self, other, BinaryOp.DIV)

    def __mod__(self, other: TracedTensor) -> TracedTensor:
        return _binary(self, other, BinaryOp.MOD)

    def __pow__(
        self, other: TracedTensor, modulo: object | None = None
    ) -> TracedTensor:
        if modulo is not None:
            raise TypeError("TensorIR DSL pow does not support a modulo argument")
        return _binary(self, other, BinaryOp.POW)

    def __neg__(self) -> TracedTensor:
        return self._graph.add_node(
            NodeKind.UNARY, UnaryOp.NEG, (self._id,), self.tensor_info
        )

    def __gt__(self, other: TracedTensor) -> TracedTensor:
        return _ordered_cmp(self, other, "ogt", "gt")

    def __ge__(self, other: TracedTensor) -> TracedTensor:
        return _ordered_cmp(self, other, "oge", "ge")

    def __lt__(self, other: TracedTensor) -> TracedTensor:
        return _ordered_cmp(self, other, "olt", "lt")

    def __le__(self, other: TracedTensor) -> TracedTensor:
        return _ordered_cmp(self, other, "ole", "le")

    def __eq__(self, other: object) -> bool:
        raise TypeError(
            "TensorIR DSL does not support == on tensors; use tir.cmp(...) "
            "with predicate='oeq' for floating-point tensors or predicate='eq' "
            "for integer tensors"
        )

    def __ne__(self, other: object) -> bool:
        raise TypeError(
            "TensorIR DSL does not support != on tensors; use tir.cmp(...) "
            "with predicate='one' for floating-point tensors or predicate='neq' "
            "for integer tensors"
        )

    __hash__ = object.__hash__

    def __matmul__(self, other: TracedTensor) -> TracedTensor:
        if not isinstance(other, TracedTensor):
            return NotImplemented
        return _matmul(self, other)

    def __bool__(self) -> bool:
        raise TypeError(
            "TensorIR DSL does not support tensor-valued Python control flow; "
            "use tir.where(...) for elementwise selection"
        )

    def __repr__(self) -> str:
        return f"TracedTensor(%{self._id})"


def _binary(lhs: TracedTensor, rhs: TracedTensor, op: BinaryOp) -> TracedTensor:
    if not isinstance(rhs, TracedTensor):
        return NotImplemented
    op_name = op.value
    _check_same_graph(lhs, rhs, op_name)
    _check_binary_tensor_types(lhs.tensor_info, rhs.tensor_info, op)
    return lhs._graph.add_node(NodeKind.BINARY, op, (lhs._id, rhs._id), lhs.tensor_info)


def _cmp(lhs: TracedTensor, rhs: TracedTensor, comparator: str) -> TracedTensor:
    if not isinstance(rhs, TracedTensor):
        return NotImplemented
    _check_same_graph(lhs, rhs, "cmp")
    _check_same_tensor_type(lhs.tensor_info, rhs.tensor_info, "cmp")
    _check_comparator_supported(lhs.tensor_info.dtype, comparator)
    result_info = replace(lhs.tensor_info, dtype=DataType.BOOL)
    return lhs._graph.add_node(
        NodeKind.CMP,
        "cmp",
        (lhs._id, rhs._id),
        result_info,
        kwargs={"comparator": comparator},
    )


def _ordered_cmp(
    lhs: TracedTensor, rhs: TracedTensor, float_predicate: str, int_predicate: str
) -> TracedTensor:
    if not isinstance(rhs, TracedTensor):
        return NotImplemented
    predicate = (
        float_predicate if is_float_dtype(lhs.tensor_info.dtype) else int_predicate
    )
    return _cmp(lhs, rhs, predicate)


def _matmul_result_info(
    lhs: TracedTensor, rhs: TracedTensor, result_dtype: object | None = None
) -> TensorInfo:
    _check_same_graph(lhs, rhs, "matmul")
    lhs_info = lhs.tensor_info
    rhs_info = rhs.tensor_info
    if lhs_info.dtype != rhs_info.dtype:
        raise TypeError("matmul operands must have the same dtype")
    if len(lhs_info.shape) < 2 or len(rhs_info.shape) < 2:
        raise TypeError("matmul operands must be at least rank-2 tensors")
    if lhs_info.shape[:-2] != rhs_info.shape[:-2]:
        raise TypeError("matmul operands must have matching batch dimensions")
    if lhs_info.shape[-1] != rhs_info.shape[-2]:
        raise TypeError("matmul K dimension mismatch")
    shape = (*lhs_info.shape[:-1], rhs_info.shape[-1])
    return TensorInfo(
        name="matmul",
        dtype=(
            lhs_info.dtype if result_dtype is None else normalize_dtype(result_dtype)
        ),
        shape=shape,
        stride=_contiguous_stride(shape),
        dynamic_shape=lhs_info.dynamic_shape,
    )


def _matmul(
    lhs: TracedTensor, rhs: TracedTensor, result_dtype: object | None = None
) -> TracedTensor:
    return lhs._graph.add_node(
        NodeKind.MATMUL,
        "matmul",
        (lhs._id, rhs._id),
        _matmul_result_info(lhs, rhs, result_dtype),
    )


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


_INTEGER_CMP_PREDICATES = {"eq", "neq", "gt", "ge", "lt", "le"}
_FLOAT_CMP_PREDICATES = {
    "ueq",
    "une",
    "ugt",
    "uge",
    "ult",
    "ule",
    "oeq",
    "one",
    "ogt",
    "oge",
    "olt",
    "ole",
}


def _check_comparator_supported(dtype: DataType, comparator: str) -> None:
    if is_float_dtype(dtype):
        if comparator not in _FLOAT_CMP_PREDICATES:
            raise TypeError(
                f"cmp predicate '{comparator}' is not valid for floating-point tensors"
            )
        return
    if is_integer_dtype(dtype):
        if comparator not in _INTEGER_CMP_PREDICATES:
            raise TypeError(
                f"cmp predicate '{comparator}' is not valid for integer tensors"
            )
        return
    raise TypeError(f"cmp does not support tensor dtype {dtype}")


def _check_same_graph(lhs: TracedTensor, rhs: TracedTensor, op_name: str) -> None:
    if lhs._graph is not rhs._graph:
        raise TypeError(f"{op_name} operands must belong to the same trace graph")


def _check_same_tensor_type(lhs: TensorInfo, rhs: TensorInfo, op_name: str) -> None:
    if lhs.dtype != rhs.dtype:
        raise TypeError(
            f"{op_name} operands must have the same dtype: {lhs.dtype} vs {rhs.dtype}"
        )
    _check_same_tensor_shape(lhs, rhs, op_name)


def _check_binary_tensor_types(lhs: TensorInfo, rhs: TensorInfo, op: BinaryOp) -> None:
    if op == BinaryOp.ATAN2 and not is_float_dtype(lhs.dtype):
        raise TypeError("tir.atan2 expects floating-point tensor operands")
    if (
        op == BinaryOp.POW
        and is_float_dtype(lhs.dtype)
        and rhs.dtype in (DataType.SI32, DataType.UI32)
    ):
        _check_same_tensor_shape(lhs, rhs, op.value)
        return
    _check_same_tensor_type(lhs, rhs, op.value)


def _check_same_tensor_shape(lhs: TensorInfo, rhs: TensorInfo, op_name: str) -> None:
    if lhs.shape != rhs.shape:
        raise TypeError(
            f"{op_name} operands must have the same dtype and shape: "
            f"{lhs.dtype}{lhs.shape} vs {rhs.dtype}{rhs.shape}"
        )
