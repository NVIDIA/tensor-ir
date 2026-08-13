# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

from .dtypes import DataType, normalize_dtype


@dataclass(frozen=True)
class TensorSpec:
    """Framework-independent tensor metadata for tracing and compilation.

    ``TensorSpec`` lets users compile from metadata without allocating a
    framework tensor. ``shape`` and ``stride`` are element counts, not bytes.
    If ``stride`` is omitted, a contiguous row-major stride is inferred.

    Example:
        ``tir.TensorSpec((8, 8), dtype=tir.DataType.F32)``
    """

    shape: tuple[int, ...]
    _: KW_ONLY
    dtype: DataType
    stride: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        parsed_shape = _shape_tuple(self.shape)
        parsed_stride = (
            _contiguous_stride(parsed_shape)
            if self.stride is None
            else _stride_tuple(self.stride)
        )
        if len(parsed_shape) != len(parsed_stride):
            raise TypeError("TensorSpec shape and stride must have the same rank")
        object.__setattr__(self, "shape", parsed_shape)
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))
        object.__setattr__(self, "stride", parsed_stride)


def tensor_spec_from_value(value: object) -> TensorSpec:
    """Extract tensor metadata from ``TensorSpec`` or a DLPack tensor.

    DLPack inputs must expose both ``__dlpack__`` and ``__dlpack_device__``.
    Only metadata is consumed here; runtime launch still requires a
    CUDA-accessible DLPack tensor.
    """
    if isinstance(value, TensorSpec):
        return value
    if not (hasattr(value, "__dlpack__") and hasattr(value, "__dlpack_device__")):
        raise TypeError(
            "TensorIR DSL tensor arguments must be DLPack-compatible tensors "
            "or TensorSpec"
        )
    # Avoid importing the native extension when the pure-Python DSL is imported.
    from nv_tensor_ir._mlir._mlir_libs import _tensor_ir

    shape, stride, dtype = _tensor_ir._tensor_metadata_from_dlpack(value)
    return TensorSpec(shape, dtype=DataType(dtype), stride=stride)


def _shape_tuple(shape: object) -> tuple[int, ...]:
    if isinstance(shape, str):
        raise TypeError("TensorSpec shape must be an int sequence, not str")
    if not isinstance(shape, (list, tuple)):
        try:
            shape = tuple(shape)
        except TypeError as exc:
            raise TypeError("TensorSpec shape must be iterable") from exc
    parsed = tuple(int(dim) for dim in shape)
    if any(dim <= 0 for dim in parsed):
        raise TypeError("TensorSpec shape dimensions must be positive")
    return parsed


def _stride_tuple(stride: object) -> tuple[int, ...]:
    if isinstance(stride, str):
        raise TypeError("TensorSpec stride must be an int sequence, not str")
    if not isinstance(stride, (list, tuple)):
        try:
            stride = tuple(stride)
        except TypeError as exc:
            raise TypeError("TensorSpec stride must be iterable") from exc
    return tuple(int(dim) for dim in stride)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))
