# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public TensorIR compiler and runtime API shared by all package variants."""

from __future__ import annotations

from nv_tensor_ir._mlir._mlir_libs import _tensor_ir as _tensor_ir_module

CompileOptions = _tensor_ir_module.CompileOptions
ArchPortability = _tensor_ir_module.ArchPortability
CodegenStrategy = _tensor_ir_module.CodegenStrategy
BytecodeVersion = _tensor_ir_module.BytecodeVersion
CudaTileArtifactKind = _tensor_ir_module.CudaTileArtifactKind

__all__ = [
    "ArchPortability",
    "BytecodeVersion",
    "CodegenStrategy",
    "CompileOptions",
    "Program",
    "CudaTileArtifactKind",
    "can_compile",
    "compile",
]


def _detect_compute_capability() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return props.major * 10 + props.minor
    except ImportError:
        pass
    return 100


def _make_workspace(size: int):
    """Allocate default CUDA workspace using PyTorch."""
    if size == 0:
        return None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "TensorIR launch requires workspace= when PyTorch is not available; "
            "the default workspace allocator uses a torch CUDA tensor"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorIR launch requires workspace= when CUDA is not available "
            "through PyTorch; the default workspace allocator uses a torch CUDA "
            "tensor"
        )

    return torch.empty((size,), device="cuda", dtype=torch.uint8)


class Program:
    """Executable TensorIR program returned by :func:`compile`.

    ``Program`` owns native compiler/runtime state. Use it as a context manager
    or call :meth:`destroy` when the compiled program is no longer needed.
    """

    def __init__(self, native_program):
        self._native_program = native_program

    @property
    def is_destroyed(self) -> bool:
        return self._native_program.is_destroyed

    @property
    def is_initialized(self) -> bool:
        return self._native_program.is_initialized

    def __repr__(self) -> str:
        return repr(self._native_program)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.destroy()

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    def destroy(self) -> None:
        """Release the native program state.

        Calling ``destroy`` more than once is allowed. Other methods raise after
        the program has been destroyed.
        """
        self._native_program.destroy()

    def initialize(self) -> None:
        """Initialize host-side runtime state for this program."""
        self._native_program.initialize()

    def check_support(self, *args) -> bool:
        """Return whether this program can run with the provided arguments.

        Tensor arguments must be DLPack-compatible objects that provide
        ``__dlpack__`` and ``__dlpack_device__``.
        """
        return self._native_program.check_support(args)

    def query_workspace_size(self, *args) -> int:
        """Return required workspace bytes for the provided runtime arguments.

        Tensor arguments must be DLPack-compatible objects that provide
        ``__dlpack__`` and ``__dlpack_device__``.
        """
        return self._native_program.query_workspace_size(args)

    def launch(self, *args, workspace=None, stream=None) -> None:
        """Launch the program.

        Args:
            *args: Runtime scalar and tensor arguments, followed by output
                tensors. Tensor arguments must be DLPack-compatible objects.
            workspace: Optional DLPack-compatible workspace buffer. If omitted,
                the runtime queries the required size and allocates a torch CUDA
                tensor.
            stream: Optional CUDA stream object or integer stream pointer.
        """
        workspace_tensor = workspace
        if workspace_tensor is None:
            workspace_tensor = _make_workspace(self.query_workspace_size(*args))
        self._native_program.launch(args, workspace_tensor, stream)

    def get_bytecode(self) -> bytes:
        """Return the selected TileIR bytecode or cuBin artifact as ``bytes``."""
        return self._native_program.get_bytecode()


_COMPILE_OPTION_NAMES = frozenset(
    {
        "compute_capability",
        "arch_portability",
        "cta_count",
        "warp_count",
        "tile_sizes",
        "uniform_signature",
        "codegen_strategy",
        "max_tile_candidates",
        "dump_cuda_tile_ir_path",
        "dump_tileir_bc_path",
        "load_tileir_bc_path",
        "print_ir_tree_dir",
        "print_ir_after_all",
        "enable_timing",
        "bytecode_version",
        "artifact_kind",
    }
)

_ARCH_PORTABILITY_BY_VALUE = {value.value: value for value in ArchPortability}

_CODEGEN_STRATEGY_BY_VALUE = {value.value: value for value in CodegenStrategy}

_CUDA_TILE_ARTIFACT_KIND_BY_VALUE = {
    value.value: value for value in CudaTileArtifactKind
}


def _enum_by_value(name: str, values: dict[int, object], value: int):
    try:
        return values[value]
    except KeyError as exc:
        raise ValueError(f"Unknown TensorIR compile option {name}: {value}") from exc


def _make_compile_options(options=None, **kwargs) -> CompileOptions:
    unknown_options = sorted(set(kwargs) - _COMPILE_OPTION_NAMES)
    if unknown_options:
        raise TypeError(
            "Unknown TensorIR compile option(s): " + ", ".join(unknown_options)
        )
    if options is not None and kwargs:
        raise TypeError("Pass either options= or compile option keyword arguments")

    if options is None:
        options = CompileOptions()
        options.compute_capability = _detect_compute_capability()
    elif not isinstance(options, CompileOptions):
        raise TypeError("options must be a TensorIR CompileOptions object")

    for name, value in kwargs.items():
        if name == "compute_capability":
            value = value or _detect_compute_capability()
        elif name == "arch_portability" and isinstance(value, int):
            value = _enum_by_value(name, _ARCH_PORTABILITY_BY_VALUE, value)
        elif name == "codegen_strategy" and isinstance(value, int):
            value = _enum_by_value(name, _CODEGEN_STRATEGY_BY_VALUE, value)
        elif name == "artifact_kind" and isinstance(value, int):
            value = _enum_by_value(name, _CUDA_TILE_ARTIFACT_KIND_BY_VALUE, value)
        elif name == "tile_sizes":
            value = list(value)
        setattr(options, name, value)
    return options


def can_compile(module, *, options=None, **kwargs) -> bool:
    """Return whether a TensorIR MLIR module can be compiled.

    Args:
        module: MLIR ``ir.Module`` object.
        options: Optional ``CompileOptions`` object. If provided, keyword
            option overrides are rejected.
        **kwargs: Individual ``CompileOptions`` field overrides.
    """
    return _tensor_ir_module.can_compile(
        module, _make_compile_options(options, **kwargs)
    )


def compile(module, *, options=None, **kwargs) -> Program:
    """Compile a TensorIR MLIR module to an executable :class:`Program`.

    Args:
        module: MLIR ``ir.Module`` object.
        options: Optional ``CompileOptions`` object. If provided, keyword
            option overrides are rejected.
        **kwargs: Individual ``CompileOptions`` field overrides.
    """
    native_program = _tensor_ir_module.compile(
        module, _make_compile_options(options, **kwargs)
    )
    return Program(native_program)
