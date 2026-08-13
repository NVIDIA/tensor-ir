# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from enum import Enum


class DataType(str, Enum):
    """Framework-independent TensorIR element types used by the DSL trace."""

    BOOL = "i1"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"
    SI8 = "si8"
    SI16 = "si16"
    SI32 = "si32"
    SI64 = "si64"
    UI8 = "ui8"
    UI16 = "ui16"
    UI32 = "ui32"
    UI64 = "ui64"

    def __str__(self) -> str:
        return self.value


FLOAT_DTYPES = frozenset({DataType.BF16, DataType.F16, DataType.F32, DataType.F64})
SIGNED_INTEGER_DTYPES = frozenset(
    {DataType.SI8, DataType.SI16, DataType.SI32, DataType.SI64}
)
UNSIGNED_INTEGER_DTYPES = frozenset(
    {DataType.UI8, DataType.UI16, DataType.UI32, DataType.UI64}
)
INTEGER_DTYPES = (
    frozenset({DataType.BOOL}) | SIGNED_INTEGER_DTYPES | UNSIGNED_INTEGER_DTYPES
)
TENSOR_DTYPES = FLOAT_DTYPES | INTEGER_DTYPES
# Dtypes supported by the scalar constant + splat path used by full_like.
# Dense tensor constants support every TensorIR tensor dtype.
SCALAR_DTYPES = FLOAT_DTYPES | frozenset({DataType.BOOL, DataType.SI32})


def normalize_dtype(dtype: object) -> DataType:
    if isinstance(dtype, DataType):
        return dtype
    raise TypeError(
        f"Unsupported TensorIR tensor dtype: {dtype}. "
        "Use tir.DataType.* for explicit dtype metadata."
    )


def is_float_dtype(dtype: object) -> bool:
    return normalize_dtype(dtype) in FLOAT_DTYPES


def is_integer_dtype(dtype: object) -> bool:
    return normalize_dtype(dtype) in INTEGER_DTYPES


def element_type_asm(dtype: object) -> str:
    """Return the MLIR assembly spelling for a TensorIR dtype."""
    return normalize_dtype(dtype).value


def supports_scalar_attr(dtype: object) -> bool:
    return normalize_dtype(dtype) in SCALAR_DTYPES
