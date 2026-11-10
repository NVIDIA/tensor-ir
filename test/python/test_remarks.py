# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: tensor-ir-tileiras
# RUN: %pytest %s

import copy
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from nv_tensor_ir import runtime as rt
from nv_tensor_ir._mlir import ir
from nv_tensor_ir._mlir.dialects import nv_tensor_ir
from nv_tensor_ir.dsl import cuda_tile_backend as backend
from nv_tensor_ir.dsl.dsl import ArtifactCache

SOURCE = (
    Path(__file__).parents[1] / "Tools/tensor-ir-compiler-debuginfo.mlir"
).read_text()


@pytest.fixture
def compile_artifact():
    def compile(**kwargs):
        kwargs.setdefault("tileiras_path", shutil.which("tileiras"))
        opts = rt._make_compile_options(
            tile_sizes=[8],
            host_target=rt.HostTarget.Interpreter,
            remarks="all",
            arch_portability=rt.ArchPortability.arch_conditional,
            **kwargs,
        )
        with ir.Context() as context, ir.Location.unknown():
            nv_tensor_ir.register_dialect(context, load=True)
            return rt.compile(ir.Module.parse(SOURCE), options=opts)

    return compile


@pytest.mark.parametrize(
    "kind", [rt.CudaTileArtifactKind.TileIR, rt.CudaTileArtifactKind.Cubin]
)
@pytest.mark.parametrize("destination", ["", "-", "file"])
def test_capture_and_artifact_lifecycle(
    compile_artifact, tmp_path, capfd, kind, destination
):
    output = str(tmp_path / "remarks.yaml") if destination == "file" else destination
    artifact = compile_artifact(
        artifact_kind=kind, remarks_output_file=output, remarks_missed="none"
    )
    yaml = artifact.get_remarks()
    assert "RemarkMemoryLoadInstructionSelected" in yaml
    assert "File: source.py" in yaml.replace("'", "")
    assert "--- !Missed" not in yaml
    out, err = capfd.readouterr()
    if destination == "file":
        assert Path(output).read_text() == yaml
    else:
        selected, other = (out, err) if destination == "-" else (err, out)
        assert yaml in selected and yaml not in other
    assert copy.copy(artifact).get_remarks() == yaml
    decoded = rt.Artifact.deserialize(artifact.serialize())
    assert decoded.get_remarks() == yaml
    assert decoded.get_bytecode() == artifact.get_bytecode()


@pytest.mark.parametrize("selected", ["explicit", "toolkit", "cuda-root", "path"])
def test_tool_lookup_order(compile_artifact, tmp_path, monkeypatch, selected):
    for marker in ("explicit", "toolkit", "cuda-root", "path"):
        path = tmp_path / marker / "bin" / "tileiras"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!/bin/sh\necho 'selected {marker}' >&2\nexit 99\n")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "path" / "bin"))
    monkeypatch.setenv("CUDA_ROOT", str(tmp_path / "cuda-root"))
    toolkit = "" if selected == "cuda-root" else str(tmp_path / "toolkit")
    if selected == "path":
        toolkit = str(tmp_path / "absent-toolkit")
    with pytest.raises(RuntimeError, match=f"selected {selected}"):
        compile_artifact(
            tileiras_path=str(tmp_path / "explicit/bin/tileiras")
            if selected == "explicit"
            else "",
            cuda_toolkit_path=toolkit,
        )


@pytest.mark.parametrize(
    "option,path,match",
    [
        ("remarks_output_file", "missing/output", "cannot open remark output"),
        ("remarks_output_file", "/dev/full", "cannot write remark output"),
        ("tileiras_path", "absent", "configured --tileiras-path is unavailable"),
    ],
)
def test_invalid_paths(compile_artifact, tmp_path, option, path, match):
    if path == "/dev/full" and not Path(path).exists():
        pytest.skip("requires /dev/full")
    with pytest.raises(RuntimeError, match=match):
        compile_artifact(**{option: str(tmp_path / path)})


def test_disabled_remarks_skip_assembly(compile_artifact, tmp_path):
    output = tmp_path / "existing"
    output.write_text("untouched")
    options = dict(
        tileiras_path=str(tmp_path / "absent"),
        remarks_output_file=str(output),
        **dict.fromkeys(
            (f"remarks_{kind}" for kind in ("passed", "missed", "failed", "analysis")),
            "none",
        ),
    )
    assert compile_artifact(**options).get_remarks() == ""
    assert output.read_text() == "untouched"
    with pytest.raises(RuntimeError, match="configured --tileiras-path is unavailable"):
        compile_artifact(artifact_kind=rt.CudaTileArtifactKind.Cubin, **options)


@pytest.mark.parametrize("enabled", [False, True])
def test_cache_bypass(monkeypatch, enabled):
    opts = rt._make_compile_options(
        remarks=rt.RemarkCategory.ALL if enabled else rt.RemarkCategory.NONE
    )
    cache, accesses = ArtifactCache(), []
    cached = SimpleNamespace(get_bytecode=lambda: b"cached")
    compiled = SimpleNamespace(get_bytecode=lambda: b"compiled")
    cache._artifacts["key"] = cached
    monkeypatch.setattr(cache, "_get", lambda key: accesses.append("get") or cached)
    monkeypatch.setattr(cache, "_put", lambda *args: accesses.append("put"))
    monkeypatch.setattr(backend, "_calculate_cache_key", lambda *args, **kw: "key")
    monkeypatch.setattr(backend.nv_tensor_ir, "compile", lambda *args, **kw: compiled)
    graph = SimpleNamespace(nodes=[], input_ids=[], output_ref_ids=[])
    kernel = backend.compile_cuda_tile(
        None, None, graph, options=opts, artifact_cache=cache
    )
    assert kernel.artifact is (compiled if enabled else cached)
    assert accesses == ([] if enabled else ["get"])
    assert cache._artifacts["key"] is cached
