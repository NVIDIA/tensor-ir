# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable

import torch

from nv_tensor_ir import dsl as tir
from nv_tensor_ir.dsl.dsl import KernelFunction, _trace
from nv_tensor_ir.dsl.tracing import NodeKind, TraceGraph

UnaryReference = Callable[[torch.Tensor], torch.Tensor]
BinaryReference = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_DATA_TYPE_TO_TORCH_DTYPE: dict[tir.DataType, torch.dtype] | None = None

_UNARY_REFERENCES: dict[str, UnaryReference] = {
    "relu_fwd": torch.relu,
    "abs": torch.abs,
    "ceil": torch.ceil,
    "cos": torch.cos,
    "exp": torch.exp,
    "floor": torch.floor,
    "log": torch.log,
    "neg": torch.neg,
    "reciprocal": torch.reciprocal,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "sin": torch.sin,
    "tan": torch.tan,
    "sigmoid_fwd": torch.sigmoid,
    "tanh_fwd": torch.tanh,
    "gelu_approx_tanh_fwd": lambda value: torch.nn.functional.gelu(
        value, approximate="tanh"
    ),
    "not": torch.logical_not,
}

_UNARY_WITH_BETA_REFERENCES: dict[
    str, Callable[[torch.Tensor, float], torch.Tensor]
] = {
    "elu_fwd": lambda value, beta: torch.where(
        value > 0, value, beta * (torch.exp(value) - 1)
    ),
    "softplus_fwd": lambda value, beta: torch.nn.functional.softplus(value, beta=beta),
    "swish_fwd": lambda value, beta: value * torch.sigmoid(beta * value),
}

_BINARY_REFERENCES: dict[str, BinaryReference] = {
    "add": lambda lhs, rhs: lhs + rhs,
    "atan2": torch.atan2,
    "sub": lambda lhs, rhs: lhs - rhs,
    "mul": lambda lhs, rhs: lhs * rhs,
    "div": lambda lhs, rhs: lhs / rhs,
    "mod": torch.remainder,
    "rem": torch.fmod,
    "pow": torch.pow,
    "max": torch.maximum,
    "min": torch.minimum,
    "add_square": lambda lhs, rhs: lhs + rhs * rhs,
    "and": torch.logical_and,
    "or": torch.logical_or,
}

_CMP_REFERENCES: dict[str, BinaryReference] = {
    "eq": torch.eq,
    "neq": torch.ne,
    "gt": torch.gt,
    "ge": torch.ge,
    "lt": torch.lt,
    "le": torch.le,
    "ueq": lambda lhs, rhs: torch.isnan(lhs) | torch.isnan(rhs) | torch.eq(lhs, rhs),
    "une": lambda lhs, rhs: torch.isnan(lhs) | torch.isnan(rhs) | torch.ne(lhs, rhs),
    "ugt": lambda lhs, rhs: torch.isnan(lhs) | torch.isnan(rhs) | torch.gt(lhs, rhs),
    "uge": lambda lhs, rhs: torch.isnan(lhs) | torch.isnan(rhs) | torch.ge(lhs, rhs),
    "ult": lambda lhs, rhs: torch.isnan(lhs) | torch.isnan(rhs) | torch.lt(lhs, rhs),
    "ule": lambda lhs, rhs: torch.isnan(lhs) | torch.isnan(rhs) | torch.le(lhs, rhs),
    "oeq": torch.eq,
    "one": lambda lhs, rhs: (
        (~torch.isnan(lhs)) & (~torch.isnan(rhs)) & torch.ne(lhs, rhs)
    ),
    "ogt": torch.gt,
    "oge": torch.ge,
    "olt": torch.lt,
    "ole": torch.le,
}


def convert_unsupported_torch_dtype(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in (torch.uint16, torch.uint32, torch.uint64):
        return x.to(torch.int64)
    return x


def evaluate_kernel_reference(
    kernel_func: KernelFunction,
    *inputs: torch.Tensor,
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Evaluate a DSL kernel by interpreting its trace with PyTorch.

    The output tensor is used only to trace result shape and dtype, matching the
    `tir.compile(..., output=...)` contract.
    """
    graph = _trace(kernel_func, *inputs, output=output)
    outputs = output if isinstance(output, tuple) else (output,)
    return evaluate_trace_reference(graph, *inputs, device=outputs[0].device)


def evaluate_trace_reference(
    graph: TraceGraph,
    *inputs: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if len(inputs) != len(graph.input_ids):
        raise TypeError(f"Expected {len(graph.input_ids)} inputs, got {len(inputs)}")

    values: dict[int, object] = {}
    reference_device = inputs[0].device if inputs else device
    for node_id, value in zip(graph.input_ids, inputs):
        values[node_id] = value

    for node_id, node in enumerate(graph.nodes):
        if node.kind in (NodeKind.INPUT, NodeKind.OUTPUT_REF):
            continue
        args = [values[arg_id] for arg_id in node.args]
        if node.kind == NodeKind.CONSTANT:
            if node.kwargs.get("is_tensor", False):
                values[node_id] = torch.full(
                    node.tensor_info.shape,
                    node.kwargs["value"],
                    dtype=_torch_dtype_from_tensor_ir(node.tensor_info.dtype),
                    device=reference_device,
                )
            else:
                values[node_id] = node.kwargs["value"]
        elif node.kind == NodeKind.SPLAT:
            shape = node.tensor_info.shape
            if node.tensor_info.dynamic_shape:
                shape = tuple(values[int(node.kwargs["like"])].shape)
            values[node_id] = torch.full(
                shape,
                args[0],
                dtype=_torch_dtype_from_tensor_ir(node.tensor_info.dtype),
                device=reference_device,
            )
        elif node.kind == NodeKind.UNARY:
            if node.op_name == "abs" and args[0].dtype in (
                torch.uint16,
                torch.uint32,
                torch.uint64,
            ):
                values[node_id] = args[0]  # no pytorch reference for uint types
            elif node.op_name in _UNARY_WITH_BETA_REFERENCES:
                values[node_id] = _run_torch_reference(
                    node.op_name,
                    _UNARY_WITH_BETA_REFERENCES[node.op_name],
                    args[0],
                    float(node.kwargs["beta"]),
                )
            else:
                values[node_id] = _run_torch_reference(
                    node.op_name, _UNARY_REFERENCES[node.op_name], *args
                )
        elif node.kind == NodeKind.BINARY:
            values[node_id] = _run_torch_reference(
                node.op_name,
                _BINARY_REFERENCES[node.op_name],
                *map(convert_unsupported_torch_dtype, args),
            )
            if node.op_name == "div" and not args[0].dtype.is_floating_point:
                # seems tensor_ir div with int types has rounding mode of round towards zero
                values[node_id] = values[node_id].trunc().to(args[0].dtype)
            else:
                values[node_id] = values[node_id].to(args[0].dtype)
        elif node.kind == NodeKind.CONVERT:
            values[node_id] = args[0].to(
                _torch_dtype_from_tensor_ir(node.tensor_info.dtype)
            )
        elif node.kind == NodeKind.MOVEMENT:
            if node.op_name == "reshape":
                values[node_id] = torch.reshape(args[0], node.tensor_info.shape)
            elif node.op_name == "transpose":
                values[node_id] = torch.permute(
                    args[0], tuple(node.kwargs["permutation"])
                )
            elif node.op_name == "broadcast":
                values[node_id] = torch.broadcast_to(args[0], node.tensor_info.shape)
            elif node.op_name == "slice":
                slices = tuple(
                    slice(start, limit, stride)
                    for start, limit, stride in zip(
                        node.kwargs["starts"],
                        node.kwargs["limits"],
                        node.kwargs["strides"],
                    )
                )
                values[node_id] = args[0][slices]
            else:
                raise NotImplementedError(f"No PyTorch reference for {node.op_name}")
        elif node.kind == NodeKind.CONCATENATE:
            values[node_id] = torch.cat(args, dim=int(node.kwargs["dimension"]))
        elif node.kind == NodeKind.IOTA:
            dtype = _torch_dtype_from_tensor_ir(node.tensor_info.dtype)
            dim = int(node.kwargs["dimension"])
            shape = node.tensor_info.shape
            view_shape = [1] * len(shape)
            view_shape[dim] = shape[dim]
            iota_device = args[0].device if args else reference_device
            values[node_id] = (
                torch.arange(shape[dim], dtype=dtype, device=iota_device)
                .reshape(view_shape)
                .expand(shape)
            )
        elif node.kind == NodeKind.REDUCE:
            reduction = _reduce_reference(
                convert_unsupported_torch_dtype(args[0]),
                tuple(node.kwargs["dimensions"]),
                str(node.kwargs["mode"]),
                args[0].dtype,
            )
            values[node_id] = reduction.to(args[0].dtype)
        elif node.kind == NodeKind.CMP:
            comparator = node.kwargs["comparator"]
            if not isinstance(comparator, str) or comparator not in _CMP_REFERENCES:
                raise NotImplementedError(f"No PyTorch reference for cmp {comparator}")
            values[node_id] = _run_torch_reference(
                f"cmp {comparator}",
                _CMP_REFERENCES[comparator],
                *map(convert_unsupported_torch_dtype, args),
            )
        elif node.kind == NodeKind.SELECT:
            values[node_id] = torch.where(
                *map(convert_unsupported_torch_dtype, args)
            ).to(args[1].dtype)
        elif node.kind == NodeKind.MATMUL:
            result_dtype = _torch_dtype_from_tensor_ir(node.tensor_info.dtype)
            lhs, rhs = args
            if lhs.dtype in (torch.int8, torch.uint8) and result_dtype == torch.int32:
                lhs = lhs.to(torch.int32)
                rhs = rhs.to(torch.int32)
            values[node_id] = (lhs @ rhs).to(result_dtype)
        else:
            raise NotImplementedError(f"No PyTorch reference for {node.kind}")

    results = tuple(values[result_id] for result_id in graph.result_ids)
    return results[0] if len(results) == 1 else results


def _run_torch_reference(op_name: str, fn: Callable, *args: object):
    try:
        return fn(*args)
    except NotImplementedError as exc:
        raise NotImplementedError(
            _unsupported_reference_message(op_name, args)
        ) from exc
    except RuntimeError as exc:
        message = str(exc)
        if "not implemented" in message or "not supported" in message:
            raise NotImplementedError(
                _unsupported_reference_message(op_name, args)
            ) from exc
        raise


def _unsupported_reference_message(op_name: str, args: tuple[object, ...]) -> str:
    dtypes = ", ".join(str(arg.dtype) for arg in args if isinstance(arg, torch.Tensor))
    return f"No PyTorch reference for TensorIR op '{op_name}' with dtypes: {dtypes}"


def _torch_dtype_from_tensor_ir(dtype: object) -> torch.dtype:
    global _DATA_TYPE_TO_TORCH_DTYPE
    if _DATA_TYPE_TO_TORCH_DTYPE is None:
        data_type_to_torch_dtype = {
            tir.DataType.BOOL: torch.bool,
            tir.DataType.F32: torch.float32,
            tir.DataType.F64: torch.float64,
            tir.DataType.F16: torch.float16,
            tir.DataType.BF16: torch.bfloat16,
            tir.DataType.SI8: torch.int8,
            tir.DataType.SI16: torch.int16,
            tir.DataType.SI32: torch.int32,
            tir.DataType.SI64: torch.int64,
        }
        for name, data_type in (
            ("uint8", tir.DataType.UI8),
            ("uint16", tir.DataType.UI16),
            ("uint32", tir.DataType.UI32),
            ("uint64", tir.DataType.UI64),
        ):
            torch_dtype = getattr(torch, name, None)
            if torch_dtype is not None:
                data_type_to_torch_dtype[data_type] = torch_dtype
        _DATA_TYPE_TO_TORCH_DTYPE = data_type_to_torch_dtype
    normalized = tir.DataType(dtype)
    if normalized in _DATA_TYPE_TO_TORCH_DTYPE:
        return _DATA_TYPE_TO_TORCH_DTYPE[normalized]
    raise TypeError(f"TensorIR DSL reference evaluation does not support {normalized}")


def _reduce_reference(
    value: torch.Tensor,
    dimensions: tuple[int, ...],
    mode: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reference implementation of TensorIR reductions with keepdim semantics."""
    if mode == "add":
        return value.sum(dim=dimensions, keepdim=True)
    if mode == "amax":
        return _reduce_extrema(value.abs(), dimensions, torch.max)
    if mode == "avg":
        # torch.mean does not support integer types
        element_count = 1
        for dimension in dimensions:
            element_count *= value.shape[dimension]
        return (
            # emulate the overflow behavior of the original dtype
            value.sum(dim=dimensions, keepdim=True).to(dtype).to(value.dtype)
            / element_count
        )
    if mode == "max":
        return _reduce_extrema(value, dimensions, torch.max)
    if mode == "min":
        return _reduce_extrema(value, dimensions, torch.min)
    if mode == "mul":
        return _reduce_product(value, dimensions)
    if mode == "mul_no_zeros":
        return _reduce_product(value.masked_fill(value == 0, 1), dimensions)
    if mode == "norm1":
        return value.abs().sum(dim=dimensions, keepdim=True)
    if mode == "norm2":
        return torch.sqrt((value * value).sum(dim=dimensions, keepdim=True))
    raise NotImplementedError(f"No PyTorch reference for reduce mode {mode}")


def _reduce_extrema(
    value: torch.Tensor, dimensions: tuple[int, ...], fn
) -> torch.Tensor:
    """Apply a torch extrema reducer over dimensions without dropping rank."""
    result = value
    for dim in sorted(dimensions):
        result = fn(result, dim=dim, keepdim=True).values
    return result


def _reduce_product(value: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    """Apply product reduction over dimensions without dropping rank.

    CUDA ``torch.prod`` is JIT compiled and unavailable on some CI systems, so
    run it on CPU.
    """
    device = value.device
    result = value.cpu()
    for dim in sorted(dimensions):
        result = torch.prod(result, dim=dim, keepdim=True)
    return result.to(device=device)
