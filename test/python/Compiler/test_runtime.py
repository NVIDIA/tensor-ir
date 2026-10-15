# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON -m pytest --capture=no -vv %s

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from nv_tensor_ir import runtime as tensor_ir_runtime
from nv_tensor_ir._mlir import ir
from nv_tensor_ir._mlir.dialects import nv_tensor_ir


class _FakeWorkspace:
    def __init__(self, ptr: int, size: int):
        self._ptr = ptr
        self._size = size

    def data_ptr(self) -> int:
        return self._ptr

    def numel(self) -> int:
        return self._size

    def element_size(self) -> int:
        return 1


class _FakeNativeKernel:
    def __init__(self) -> None:
        self.check_support_calls = []
        self.query_workspace_calls = []
        self.launch_calls = []

    def check_support(self, args) -> bool:
        self.check_support_calls.append(args)
        return True

    def query_workspace_size(self, args) -> int:
        self.query_workspace_calls.append(args)
        return 128

    def launch(self, args, workspace=None, stream=None) -> None:
        self.launch_calls.append((args, workspace, stream))


class _FakeNativeArtifact:
    def __init__(self) -> None:
        self.load_calls = 0

    def get_bytecode(self) -> bytes:
        return b"tensor-ir-bytecode"

    def serialize(self) -> bytes:
        return b"serialized-artifact"

    def load(self):
        self.load_calls += 1
        return _FakeNativeKernel()


def test_launch_uses_provided_workspace_without_query() -> None:
    native_program = _FakeNativeKernel()
    program = tensor_ir_runtime.RuntimeKernel(native_program)
    workspace = _FakeWorkspace(ptr=0x1000, size=64)

    program.launch(1, workspace=workspace)

    assert native_program.query_workspace_calls == []
    assert native_program.launch_calls == [((1,), workspace, None)]


def test_launch_queries_workspace_when_workspace_is_not_provided(monkeypatch) -> None:
    native_program = _FakeNativeKernel()
    program = tensor_ir_runtime.RuntimeKernel(native_program)
    allocated_workspaces = []

    def make_workspace(size: int):
        workspace = _FakeWorkspace(ptr=0x2000, size=size)
        allocated_workspaces.append(workspace)
        return workspace

    monkeypatch.setattr(tensor_ir_runtime, "_make_workspace", make_workspace)

    program.launch(1)

    assert native_program.query_workspace_calls == [(1,)]
    assert len(allocated_workspaces) == 1
    assert allocated_workspaces[0]._size == 128
    assert native_program.launch_calls == [((1,), allocated_workspaces[0], None)]


def test_runtime_all_exports_only_public_api() -> None:
    assert tensor_ir_runtime.__name__ == "nv_tensor_ir.runtime"
    assert tensor_ir_runtime.__all__ == [
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


def test_artifact_and_runtime_lifetimes_are_independent() -> None:
    native = _FakeNativeArtifact()
    artifact = tensor_ir_runtime.Artifact(native)
    assert repr(artifact) == "Artifact(state='ready')"
    assert artifact.get_bytecode() == b"tensor-ir-bytecode"
    assert artifact.serialize() == b"serialized-artifact"
    assert native.load_calls == 0
    assert not hasattr(artifact, "launch")
    assert not hasattr(artifact, "initialize")

    first, second = artifact.load(), artifact.load()
    assert native.load_calls == 2
    assert first._native is not second._native
    artifact.destroy()
    artifact.destroy()
    assert repr(artifact) == "Artifact(state='destroyed')"
    with pytest.raises(RuntimeError, match="Artifact has been destroyed"):
        artifact.load()
    with pytest.raises(RuntimeError, match="Artifact has been destroyed"):
        artifact.serialize()
    assert first.check_support(1)
    first.destroy()
    assert second.check_support(1)
    with pytest.raises(RuntimeError, match="RuntimeKernel has been destroyed"):
        first.launch(1)
    with second:
        assert not second.is_destroyed
    assert second.is_destroyed


def test_make_compile_options_from_kwargs(monkeypatch) -> None:
    monkeypatch.setattr(tensor_ir_runtime, "_detect_compute_capability", lambda: 120)

    options = tensor_ir_runtime._make_compile_options(
        compute_capability=None,
        arch_portability=tensor_ir_runtime.ArchPortability.arch_conditional,
        codegen_strategy=tensor_ir_runtime.CodegenStrategy.LayoutPropagation,
        bytecode_version=tensor_ir_runtime.BytecodeVersion.compatibility(),
        tile_sizes=(2, 4, 8),
        print_ir_after_all=True,
    )

    assert options.compute_capability == 120
    assert (
        options.arch_portability == tensor_ir_runtime.ArchPortability.arch_conditional
    )
    assert (
        options.codegen_strategy == tensor_ir_runtime.CodegenStrategy.LayoutPropagation
    )
    assert options.bytecode_version == tensor_ir_runtime.BytecodeVersion.compatibility()
    assert options.tile_sizes == [2, 4, 8]
    assert options.print_ir_after_all is True


def test_make_compile_options_accepts_int_enum_values() -> None:
    options = tensor_ir_runtime._make_compile_options(
        arch_portability=0,
        codegen_strategy=1,
        artifact_kind=1,
    )

    assert options.arch_portability == tensor_ir_runtime.ArchPortability.portable
    assert (
        options.codegen_strategy == tensor_ir_runtime.CodegenStrategy.LayoutPropagation
    )
    assert options.artifact_kind == tensor_ir_runtime.CudaTileArtifactKind.Cubin


def test_make_compile_options_rejects_unknown_int_enum_values() -> None:
    with pytest.raises(ValueError, match="Unknown TensorIR compile option"):
        tensor_ir_runtime._make_compile_options(arch_portability=99)
    with pytest.raises(ValueError, match="Unknown TensorIR compile option"):
        tensor_ir_runtime._make_compile_options(artifact_kind=99)


def test_compile_options_enum_properties() -> None:
    options = tensor_ir_runtime.CompileOptions()
    current_bytecode_version = tensor_ir_runtime.BytecodeVersion.current()
    compatibility_bytecode_version = tensor_ir_runtime.BytecodeVersion.compatibility()
    default_bytecode_version = tensor_ir_runtime.BytecodeVersion.default()

    assert options.arch_portability == tensor_ir_runtime.ArchPortability.family_portable
    assert (
        options.codegen_strategy == tensor_ir_runtime.CodegenStrategy.LayoutPropagation
    )
    assert options.bytecode_version == default_bytecode_version
    assert options.artifact_kind == tensor_ir_runtime.CudaTileArtifactKind.TileIR
    assert "." in str(compatibility_bytecode_version)
    assert compatibility_bytecode_version in {
        tensor_ir_runtime.BytecodeVersion.compatibility()
    }

    options.arch_portability = tensor_ir_runtime.ArchPortability.arch_conditional
    options.codegen_strategy = tensor_ir_runtime.CodegenStrategy.LayoutPropagation
    options.bytecode_version = tensor_ir_runtime.BytecodeVersion.current()
    assert (
        options.arch_portability == tensor_ir_runtime.ArchPortability.arch_conditional
    )
    assert (
        options.codegen_strategy == tensor_ir_runtime.CodegenStrategy.LayoutPropagation
    )
    assert options.bytecode_version == tensor_ir_runtime.BytecodeVersion.current()
    assert current_bytecode_version == tensor_ir_runtime.BytecodeVersion.current()


def test_make_compile_options_rejects_mixed_options_and_kwargs() -> None:
    options = tensor_ir_runtime.CompileOptions()

    with pytest.raises(TypeError, match="Pass either options="):
        tensor_ir_runtime._make_compile_options(options, tile_sizes=[8])


def test_compile_apis_accept_mlir_module_objects(monkeypatch) -> None:
    calls = []
    native_program = _FakeNativeArtifact()

    class FakeModule:
        pass

    def fake_can_compile(module, options):
        calls.append(("can_compile", module, options))
        return True

    def fake_compile(module, options):
        calls.append(("compile", module, options))
        return native_program

    monkeypatch.setattr(
        tensor_ir_runtime._tensor_ir_module, "can_compile", fake_can_compile
    )
    monkeypatch.setattr(tensor_ir_runtime._tensor_ir_module, "compile", fake_compile)

    module = FakeModule()

    assert tensor_ir_runtime.can_compile(module)
    program = tensor_ir_runtime.compile(module)
    assert isinstance(program, tensor_ir_runtime.Artifact)
    assert program._native is native_program

    assert calls[0][0] == "can_compile"
    assert calls[0][1] is module
    assert calls[1][0] == "compile"
    assert calls[1][1] is module


def test_register_dialect_requires_context() -> None:
    """Outside a context there is nothing to default to, so this must diagnose.

    Matching on the message keeps this honest: an opaque argument-conversion
    error from the native binding would also be a `TypeError`.
    """
    with pytest.raises(TypeError, match="needs an MLIR context"):
        nv_tensor_ir.register_dialect()


def test_site_initializer_registers_tensor_ir_dialect() -> None:
    with ir.Context() as context:
        descriptor = context.get_dialect_descriptor("nv_tensor_ir")

    assert descriptor.namespace == "nv_tensor_ir"


_ARTIFACT_TEST_MODULE = """
module {
  nv_tensor_ir.graph @artifact_test(
    %a: tensor<8xf32> {nv_tensor_ir.stride = "(1)"},
    %b: tensor<8xf32> {nv_tensor_ir.stride = "(1)"}
  ) -> (tensor<8xf32> {nv_tensor_ir.stride = "(1)"}) {
    %result = add %a, %b : tensor<8xf32>
    results %result : tensor<8xf32>
  }
}
"""


def _install_fake_tileiras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    versions: str | None = None,
) -> Path:
    """Generate a fake `tileiras` executable and put it first on PATH."""
    argv_log = tmp_path / "tileiras-argv.txt"
    tileiras = tmp_path / "tileiras"
    tileiras.write_text(
        """#!/bin/sh
set -eu
: "${TENSOR_IR_TEST_TILEIRAS_ARGV_LOG:?}"
: "${TENSOR_IR_TEST_TILEIRAS_VERSIONS:?}"
printf '%s\\n' "$@" >> "$TENSOR_IR_TEST_TILEIRAS_ARGV_LOG"
if [ "$1" = "--list-versions" ]; then
  for version in $TENSOR_IR_TEST_TILEIRAS_VERSIONS; do
    printf '%s\\n' "$version"
  done
  exit 0
fi
output=
for arg in "$@"; do
  case "$arg" in
    --output-file=*) output=${arg#--output-file=} ;;
  esac
done
if [ -z "$output" ]; then
  printf 'missing --output-file\\n' >&2
  exit 2
fi
printf 'fake-cubin' > "$output"
"""
    )
    tileiras.chmod(0o755)
    if versions is None:
        versions = str(tensor_ir_runtime.BytecodeVersion.default())
    monkeypatch.setenv("TENSOR_IR_TEST_TILEIRAS_ARGV_LOG", str(argv_log))
    monkeypatch.setenv("TENSOR_IR_TEST_TILEIRAS_VERSIONS", versions)
    monkeypatch.setenv(
        "PATH", f"{tileiras.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    )
    assert shutil.which("tileiras") == str(tileiras)
    return argv_log


def _compile_artifact_test(
    arch_portability: tensor_ir_runtime.ArchPortability,
) -> bytes:
    with ir.Context() as context, ir.Location.unknown(context):
        nv_tensor_ir.register_dialect(context, load=True)
        module = ir.Module.parse(_ARTIFACT_TEST_MODULE)
        with tensor_ir_runtime.compile(
            module,
            tile_sizes=[8],
            compute_capability=100,
            arch_portability=arch_portability,
            artifact_kind=(tensor_ir_runtime.CudaTileArtifactKind.Cubin),
        ) as program:
            return program.get_bytecode()


def test_cubin_artifact_request_assembles_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv_log = _install_fake_tileiras(tmp_path, monkeypatch)
    artifact = _compile_artifact_test(
        tensor_ir_runtime.ArchPortability.arch_conditional
    )

    argv = argv_log.read_text().splitlines()
    assert argv[0] == "--list-versions"
    assert "--gpu-name=sm_100" in argv
    assert artifact == b"fake-cubin"


def test_cubin_artifact_request_rejects_portable_target() -> None:
    with pytest.raises(RuntimeError, match="arch-conditional target"):
        _compile_artifact_test(tensor_ir_runtime.ArchPortability.family_portable)


def test_cubin_artifact_request_without_executable_falls_back_to_tileir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    artifact = _compile_artifact_test(
        tensor_ir_runtime.ArchPortability.arch_conditional
    )
    assert artifact.startswith(b"\x7fTileIR\0")


def test_cubin_artifact_request_handles_incompatible_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv_log = _install_fake_tileiras(tmp_path, monkeypatch, versions="0.0")
    artifact = _compile_artifact_test(
        tensor_ir_runtime.ArchPortability.arch_conditional
    )
    assert argv_log.read_text().splitlines() == ["--list-versions"]
    assert artifact.startswith(b"\x7fTileIR\0")


def test_debug_option_properties_round_trip():
    options = tensor_ir_runtime.CompileOptions()
    options.cuda_toolkit_path = "/requested/toolkit"
    options.print_cuda_tile_ir = True
    parsed = tensor_ir_runtime.CompileOptions.parse(str(options))
    assert parsed.cuda_toolkit_path == "/requested/toolkit"
    assert parsed.print_cuda_tile_ir is True
    parsed.print_cuda_tile_ir = False
    parsed.cuda_toolkit_path = ""
    assert parsed.print_cuda_tile_ir is False
    assert parsed.cuda_toolkit_path == ""


@pytest.mark.parametrize("use_options", [False, True])
def test_analyze_module_generates_candidates(use_options):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(_ARTIFACT_TEST_MODULE)
        kwargs = dict(compute_capability=100, max_tile_candidates=3)
        if use_options:
            kwargs = dict(options=tensor_ir_runtime._make_compile_options(**kwargs))
        candidates = nv_tensor_ir.analyze_module(module, **kwargs)
        assert 0 < len(candidates) <= 3
        graph = next(iter(module.body.operations))
        stored = graph.attributes["tile_candidates"]
        assert candidates == [list(shape) for shape in stored]
        assert all(len(shape) == 1 and shape[0] > 0 for shape in candidates)


@pytest.mark.parametrize(
    "hint,expected",
    [
        ("tile_size = array<i32: 4>", [[4]]),
        ("tile_candidates = [array<i32: 8>, array<i32: 4>]", [[8], [4]]),
    ],
)
def test_analyze_module_preserves_tile_hints(hint, expected):
    with ir.Context(), ir.Location.unknown():
        source = _ARTIFACT_TEST_MODULE.replace(
            ') -> (tensor<8xf32> {nv_tensor_ir.stride = "(1)"}) {',
            ') -> (tensor<8xf32> {nv_tensor_ir.stride = "(1)"}) attributes {'
            + hint
            + "} {",
        )
        module = ir.Module.parse(source)
        assert (
            tensor_ir_runtime.analyze_module(
                module, compute_capability=100, max_tile_candidates=0
            )
            == expected
        )


def test_analyze_module_explicit_tile_sizes():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(_ARTIFACT_TEST_MODULE)
        assert (
            tensor_ir_runtime.analyze_module(
                module, compute_capability=100, tile_sizes=[8]
            )
            == []
        )


def test_analyze_module_errors():
    with ir.Context(), ir.Location.unknown():
        with pytest.raises(RuntimeError, match="expected exactly one graph op"):
            tensor_ir_runtime.analyze_module(ir.Module.parse("module {}"))
        module = ir.Module.parse(_ARTIFACT_TEST_MODULE)
        with pytest.raises(RuntimeError, match="maxTileCandidates must be > 0"):
            tensor_ir_runtime.analyze_module(module, max_tile_candidates=0)
        options = tensor_ir_runtime.CompileOptions()
        options.cta_count = 0
        with pytest.raises(RuntimeError, match="cta-count"):
            tensor_ir_runtime.analyze_module(module, options=options)


def test_artifact_outlives_compilation_context():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(_ARTIFACT_TEST_MODULE)
        artifact = tensor_ir_runtime.compile(
            module, compute_capability=100, tile_sizes=[8]
        )
    encoded = artifact.serialize()
    bytecode = artifact.get_bytecode()
    artifact.destroy()
    with tensor_ir_runtime.Artifact.deserialize(encoded) as restored:
        assert restored.serialize() == encoded
        assert restored.get_bytecode() == bytecode
    with pytest.raises(RuntimeError, match="Artifact has been destroyed"):
        restored.get_bytecode()


def test_host_compile_options():
    assert (
        tensor_ir_runtime.CompileOptions().host_target
        == tensor_ir_runtime.HostTarget.Auto
    )
    assert (
        tensor_ir_runtime._make_compile_options(host_target="auto").host_target
        == tensor_ir_runtime.HostTarget.Auto
    )
    assert (
        tensor_ir_runtime.parse_compile_options("host-target=auto").host_target
        == tensor_ir_runtime.HostTarget.Auto
    )
    assert (
        tensor_ir_runtime._make_compile_options(host_arch="x86_64").host_arch
        == tensor_ir_runtime.HostArch.X86_64
    )
    assert (
        tensor_ir_runtime._make_compile_options(host_arch="native").host_arch
        == tensor_ir_runtime.HostArch.Native
    )
    options = tensor_ir_runtime.parse_compile_options(
        "host-target=interpreter host-arch=aarch64 host-os=windows"
    )
    assert options.host_target == tensor_ir_runtime.HostTarget.Interpreter
    assert options.host_arch == tensor_ir_runtime.HostArch.AArch64
    assert options.host_os == tensor_ir_runtime.HostOS.Windows
    for value in ["invalid", "jit"]:
        with pytest.raises(ValueError, match="host_target"):
            tensor_ir_runtime._make_compile_options(host_target=value)
    combinations = ["host-target=native codegen-strategy=affine-map"]
    with ir.Context() as context:
        module = ir.Module.parse("module {}")
        for combination in combinations:
            options = tensor_ir_runtime.parse_compile_options(combination)
            with pytest.raises(RuntimeError, match="host-target=native requires"):
                tensor_ir_runtime.can_compile(module, options=options)


@pytest.mark.parametrize("host_target", [None, "auto", "native", "interpreter"])
@pytest.mark.parametrize(
    "codegen_strategy",
    [
        tensor_ir_runtime.CodegenStrategy.LayoutPropagation,
        tensor_ir_runtime.CodegenStrategy.AffineMap,
    ],
)
def test_native_host_artifact_round_trip(host_target, codegen_strategy):
    if (
        host_target == "native"
        and codegen_strategy == tensor_ir_runtime.CodegenStrategy.AffineMap
    ):
        pytest.skip("Explicit native requires layout propagation")
    options = {} if host_target is None else {"host_target": host_target}
    native = (
        codegen_strategy == tensor_ir_runtime.CodegenStrategy.LayoutPropagation
        and host_target != "interpreter"
    )
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(_ARTIFACT_TEST_MODULE)
        with tensor_ir_runtime.compile(
            module,
            compute_capability=100,
            tile_sizes=[4],
            codegen_strategy=codegen_strategy,
            **options,
        ) as program:
            encoded = program.serialize()
            symbol = b"tensor_ir_host_entry_artifact_test_dispatch_v1"
            assert (symbol in encoded) == native
            assert (b"\x7fELF" in encoded) == native
            with tensor_ir_runtime.Artifact.deserialize(encoded) as restored:
                assert restored.serialize() == encoded
