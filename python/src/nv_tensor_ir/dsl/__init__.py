# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TensorIR Python DSL.

Use ``from nv_tensor_ir import dsl as tir``. Wildcard imports are discouraged
because TensorIR ops such as ``abs``, ``max``, and ``pow`` shadow Python builtins.
"""

from .dsl import (
    BytecodeVersion,
    CodegenStrategy,
    CompiledKernel,
    CompileOptions,
    CudaTileArtifactKind,
    compile,
    kernel,
)
from .dtypes import DataType
from .tensor_spec import TensorSpec
from . import ops as _ops

for _op_name in _ops.__all__:
    globals()[_op_name] = getattr(_ops, _op_name)

__all__ = [
    "BytecodeVersion",
    "CodegenStrategy",
    "CompileOptions",
    "CompiledKernel",
    "DataType",
    "TensorSpec",
    "CudaTileArtifactKind",
    "compile",
    "kernel",
    *_ops.__all__,
]

# Importing a package submodule automatically binds it as `dsl.ops`; remove that
# implementation detail so the public API is only `dsl.<opname>`.
globals().pop("ops", None)
del _ops
