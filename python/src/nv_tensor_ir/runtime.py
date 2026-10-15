# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public TensorIR compiler and runtime API shared by all package variants."""

from __future__ import annotations

from nv_tensor_ir._mlir._mlir_libs import _tensor_ir as _tensor_ir_module

CompileOptions = _tensor_ir_module.CompileOptions
ArchPortability = _tensor_ir_module.ArchPortability
AutoGdcInsertion = _tensor_ir_module.AutoGdcInsertion
CodegenStrategy = _tensor_ir_module.CodegenStrategy
BytecodeVersion = _tensor_ir_module.BytecodeVersion
CudaTileArtifactKind = _tensor_ir_module.CudaTileArtifactKind
HostTarget = _tensor_ir_module.HostTarget
HostArch = _tensor_ir_module.HostArch
HostOS = _tensor_ir_module.HostOS

__all__ = [
    "ArchPortability",
    "AutoGdcInsertion",
    "BytecodeVersion",
    "CodegenStrategy",
    "CompileOptions",
    "Artifact",
    "RuntimeKernel",
    "CudaTileArtifactKind",
    "analyze_module",
    "HostTarget",
    "HostArch",
    "HostOS",
    "can_compile",
    "compile",
    "parse_compile_options",
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


class _OwnedObject:
    def __init__(self, native):
        self._native = native

    @property
    def is_destroyed(self) -> bool:
        return self._native is None

    def _require_native(self):
        if self.is_destroyed:
            raise RuntimeError(f"{type(self).__name__} has been destroyed")
        return self._native

    def __repr__(self) -> str:
        state = "destroyed" if self.is_destroyed else "ready"
        return f"{type(self).__name__}(state='{state}')"

    def __enter__(self):
        self._require_native()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.destroy()

    def destroy(self) -> None:
        """Release this object's ownership. Repeated calls are allowed."""
        self._native = None


class Artifact(_OwnedObject):
    """Immutable compiled code and metadata, with no loaded runtime state.

    Compilation and deserialization do not load host or device code. Use
    :meth:`load` to create an independently owned executable kernel.
    """

    def get_bytecode(self) -> bytes:
        """Return the original TileIR bytecode or cubin device binary."""
        return self._require_native().get_bytecode()

    def serialize(self) -> bytes:
        """Serialize the device binary and launch metadata."""
        return self._require_native().serialize()

    @staticmethod
    def deserialize(data: bytes) -> "Artifact":
        """Restore an artifact without loading host or device code."""
        return Artifact(_tensor_ir_module._Artifact.deserialize(data))

    def load(self) -> "RuntimeKernel":
        """Initialize runtime state and return a ready-to-launch kernel.

        Loading can fail for an incompatible device. Each call creates
        an independent kernel that remains usable after this artifact is
        destroyed. Device bytecode may be compiled during loading.
        """
        return RuntimeKernel(self._require_native().load())


class RuntimeKernel(_OwnedObject):
    """Loaded executable state created by :meth:`Artifact.load`.

    Use as a context manager or call :meth:`destroy` to unload the kernel.
    """

    def check_support(self, *args) -> bool:
        """Check runtime arguments; tensors must support DLPack."""
        return self._require_native().check_support(args)

    def query_workspace_size(self, *args) -> int:
        """Return required workspace bytes for these runtime arguments."""
        return self._require_native().query_workspace_size(args)

    def launch(self, *args, workspace=None, stream=None) -> None:
        """Enqueue the kernel with inputs followed by output tensors.

        Tensors and an optional workspace must support DLPack. If workspace
        is omitted, allocate the required size as a torch CUDA tensor. Stream
        may be a CUDA stream object or an integer stream pointer.
        """
        native = self._require_native()
        if workspace is None:
            workspace = _make_workspace(self.query_workspace_size(*args))
        native.launch(args, workspace, stream)


_COMPILE_OPTION_NAMES = frozenset(
    {
        "compute_capability",
        "arch_portability",
        "cta_count",
        "warp_count",
        "tile_sizes",
        "host_target",
        "host_arch",
        "host_os",
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

_HOST_OPTIONS_BY_NAME = {
    "host_target": {
        "auto": HostTarget.Auto,
        "native": HostTarget.Native,
        "interpreter": HostTarget.Interpreter,
    },
    "host_arch": {
        "native": HostArch.Native,
        "x86_64": HostArch.X86_64,
        "aarch64": HostArch.AArch64,
    },
    "host_os": {"windows": HostOS.Windows, "linux": HostOS.Linux},
}

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
        elif name in _HOST_OPTIONS_BY_NAME and isinstance(value, str):
            try:
                value = _HOST_OPTIONS_BY_NAME[name][value]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown TensorIR compile option {name}: {value}"
                ) from exc
        elif name == "tile_sizes":
            value = list(value)
        setattr(options, name, value)
    return options


def parse_compile_options(text: str) -> CompileOptions:
    """Parse globally keyed compile options from a textual invocation string."""
    return CompileOptions.parse(text)


def analyze_module(module, *, options=None, **kwargs) -> list[list[int]]:
    """Analyze one graph in place and return tile shapes in best-first order.

    Requires a positive ``max_tile_candidates`` or a tile hint. Explicit
    ``tile_sizes`` yield an empty list. The module may change even on failure.
    Pass either a ``CompileOptions`` object or individual option keywords,
    as for :func:`compile`.
    """
    return _tensor_ir_module.analyze_module(
        module, _make_compile_options(options, **kwargs)
    )


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


def compile(module, *, options=None, **kwargs) -> Artifact:
    """Compile a TensorIR MLIR module to an immutable :class:`Artifact`.

    Args:
        module: MLIR ``ir.Module`` object.
        options: Optional ``CompileOptions`` object. If provided, keyword
            option overrides are rejected.
        **kwargs: Individual ``CompileOptions`` field overrides.
    """
    native_artifact = _tensor_ir_module.compile(
        module, _make_compile_options(options, **kwargs)
    )
    return Artifact(native_artifact)
