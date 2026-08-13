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


class _FakeNativeProgram:
    def __init__(self) -> None:
        self.destroy_calls = 0
        self.initialize_calls = 0
        self.check_support_calls = []
        self.query_workspace_calls = []
        self.launch_calls = []
        self.bytecode = b"tensor-ir-bytecode"
        self.is_destroyed = False
        self.is_initialized = False

    def __repr__(self) -> str:
        if self.is_destroyed:
            return "Program(state='destroyed')"
        if self.is_initialized:
            return "Program(state='initialized')"
        return "Program(state='uninitialized')"

    def destroy(self) -> None:
        self.destroy_calls += 1
        self.is_destroyed = True
        self.is_initialized = False

    def initialize(self) -> None:
        self.initialize_calls += 1
        self.is_initialized = True

    def check_support(self, args) -> bool:
        self.check_support_calls.append(args)
        return True

    def query_workspace_size(self, args) -> int:
        self.query_workspace_calls.append(args)
        return 128

    def launch(self, args, workspace=None, stream=None) -> None:
        self.launch_calls.append((args, workspace, stream))
        self.is_initialized = True

    def get_bytecode(self) -> bytes:
        return self.bytecode


def test_launch_uses_provided_workspace_without_query() -> None:
    native_program = _FakeNativeProgram()
    program = tensor_ir_runtime.Program(native_program)
    workspace = _FakeWorkspace(ptr=0x1000, size=64)

    program.launch(1, workspace=workspace)

    assert native_program.query_workspace_calls == []
    assert native_program.launch_calls == [((1,), workspace, None)]


def test_launch_queries_workspace_when_workspace_is_not_provided(monkeypatch) -> None:
    native_program = _FakeNativeProgram()
    program = tensor_ir_runtime.Program(native_program)
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
        "BytecodeVersion",
        "CodegenStrategy",
        "CompileOptions",
        "Program",
        "CudaTileArtifactKind",
        "can_compile",
        "compile",
    ]


def test_program_delegates_state_and_lifecycle() -> None:
    native_program = _FakeNativeProgram()
    program = tensor_ir_runtime.Program(native_program)

    assert repr(program) == "Program(state='uninitialized')"
    assert not program.is_destroyed
    assert not program.is_initialized

    program.initialize()
    assert repr(program) == "Program(state='initialized')"
    assert program.is_initialized
    assert native_program.initialize_calls == 1

    program.destroy()
    assert repr(program) == "Program(state='destroyed')"
    assert program.is_destroyed
    assert native_program.destroy_calls == 1


def test_program_check_support_and_bytecode_delegate_to_native() -> None:
    native_program = _FakeNativeProgram()
    program = tensor_ir_runtime.Program(native_program)

    assert program.check_support(1)
    assert native_program.check_support_calls == [(1,)]
    assert program.get_bytecode() == b"tensor-ir-bytecode"


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
    native_program = _FakeNativeProgram()

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
    assert isinstance(program, tensor_ir_runtime.Program)
    assert program._native_program is native_program

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
