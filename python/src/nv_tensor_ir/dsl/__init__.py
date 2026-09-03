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
    ProgramCache,
    calculate_cache_key,
    compile,
    compile_traced,
    kernel,
    trace,
)
from . import cuda_tile_backend as _cuda_tile_backend
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
    "ProgramCache",
    "calculate_cache_key",
    "compile",
    "compile_traced",
    "kernel",
    "trace",
    *_ops.__all__,
]

# Importing a package submodule automatically binds it as `dsl.ops`; remove that
# implementation detail so the public API is only `dsl.<opname>`.
globals().pop("ops", None)
globals().pop("cuda_tile_backend", None)
globals().pop("profile", None)
del _cuda_tile_backend, _op_name, _ops
