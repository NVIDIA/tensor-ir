# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON -m pytest --capture=no -vv %s

"""CPU-only coverage of DLPack alignment inference and graph attributes."""

import ctypes
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from nv_tensor_ir import dsl as tir
from nv_tensor_ir._mlir import ir
from nv_tensor_ir.dsl.cuda_tile_backend import (
    CudaTileCompiledKernel,
    prepare_tensor_infos,
)
from nv_tensor_ir.dsl.dsl import _tensor_info_from_value
from nv_tensor_ir.dsl.module_builder import build_mlir_module
from nv_tensor_ir.dsl.tracing import TraceGraph
from nv_tensor_ir.dsl.tensor_spec import tensor_spec_from_value


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


class _MetadataTensor:
    """DLPack-only producer with a synthetic address that must not be read."""

    def __init__(
        self, pointer, byte_offset=0, *, shape=(8, 8), stride=None, dtype=(2, 32)
    ):
        self.shape = (ctypes.c_int64 * len(shape))(*shape)
        self.stride = (
            (ctypes.c_int64 * len(stride))(*stride) if stride is not None else None
        )
        self.managed = _DLManagedTensor(
            _DLTensor(
                pointer,
                _DLDevice(1, 0),
                len(shape),
                _DLDataType(*dtype, 1),
                self.shape,
                self.stride,
                byte_offset,
            ),
            None,
            None,
        )

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self):
        make_capsule = ctypes.pythonapi.PyCapsule_New
        make_capsule.restype = ctypes.py_object
        make_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        return make_capsule(ctypes.addressof(self.managed), b"dltensor", None)


@pytest.mark.parametrize(
    "pointer,offset,expected",
    [
        (0x1000, 0, 32),
        (0x1010, 0, 16),
        (0x1008, 0, 8),
        (0x1004, 0, 4),
        (0x1002, 0, 2),
        (0x1001, 0, 1),
        (0x1000, 4, 4),
        (0x1004, 12, 16),
        (0x1000, 32, 32),
        (None, 0, None),
        (None, 16, None),
    ],
)
def test_dlpack_pointer_alignment(pointer, offset, expected):
    info = _tensor_info_from_value(_MetadataTensor(pointer, offset), "tensor")
    assert info._traced_tensor_ptr == (pointer + offset if pointer else 0)
    assert info.alignment is None
    assert info.shape == (8, 8)
    assert info.stride == (8, 1)
    assert info.dtype == tir.DataType.F32
    graph = TraceGraph()
    graph.add_input(info)
    graph.add_output_ref(info)
    overrides = prepare_tensor_infos(graph)
    for node_id in graph.input_ids + graph.output_ref_ids:
        assert overrides[node_id].alignment == expected
        assert graph.nodes[node_id].value_info.alignment is None


@pytest.mark.parametrize(
    "view", ["contiguous", "offset", "transpose", "strided", "scalar"]
)
def test_numpy_alignment(view):
    base = np.empty((16, 16), dtype=np.float32)
    tensor = {
        "contiguous": base,
        "offset": base[:, 1:],
        "transpose": base.T,
        "strided": base[::2, ::3],
        "scalar": base[0, :1].reshape(()),
    }[view]
    pointer = tensor.ctypes.data
    info = _tensor_info_from_value(tensor, "tensor")
    spec = tensor_spec_from_value(tensor)
    assert spec.shape == info.shape
    assert spec.stride == info.stride
    assert spec.dtype == info.dtype
    assert spec.alignment is None
    assert info._traced_tensor_ptr == pointer
    assert info.alignment is None
    assert info.shape == tensor.shape
    assert info.stride == tuple(s // tensor.itemsize for s in tensor.strides)
    graph = TraceGraph()
    graph.add_input(info)
    overrides = prepare_tensor_infos(graph)
    assert overrides[graph.input_ids[0]].alignment == min(pointer & -pointer, 32)


@pytest.mark.parametrize("alignment", [None, 16, 128])
@pytest.mark.parametrize("dynamic_shape", [False, True])
def test_explicit_tensor_spec_is_preserved(alignment, dynamic_shape):
    spec = tir.TensorSpec((8, 8), dtype=tir.DataType.F32, alignment=alignment)
    assert tensor_spec_from_value(spec) is spec
    assert spec.alignment == alignment
    graph = TraceGraph()
    info = _tensor_info_from_value(spec, "tensor")
    assert info._traced_tensor_ptr is None
    graph.add_input(info)
    graph.add_output_ref(info)
    overrides = prepare_tensor_infos(graph, dynamic_shape=dynamic_shape)
    for node_id in graph.input_ids + graph.output_ref_ids:
        assert overrides.get(node_id, info).alignment == alignment


@pytest.mark.parametrize(
    "shape,stride,dtype,expected",
    [
        ((8, 8), (8, 1), (2, 32), 16),
        ((8, 3), (3, 1), (2, 32), 4),
        ((8, 6), (6, 1), (2, 16), 4),
        ((8, 3), (3, 1), (4, 16), 2),  # bf16
        ((8, 3), (3, 1), (2, 64), 8),
        ((8, 3), (3, 1), (0, 8), 1),  # si8
        ((8, 3), (3, 1), (1, 16), 2),  # ui16
        ((6, 8), (1, 6), (2, 32), 8),  # transposed
        ((3, 1), (1, 1), (2, 32), 4),  # prefer non-unit extent
        ((3, 5), (1, 1), (2, 16), 2),  # prefer last stride-1 dimension
        ((3, 1), (2, 1), (2, 32), 4),  # fall back to unit extent
        ((2, 3, 1), (3, 1, 1), (2, 32), 4),
        ((8, 8), (0, 1), (2, 32), 16),  # broadcast
        ((8, 3), (3, 1), (6, 8), 1),  # DLPack bool maps to i1
        ((8, 16), (16, 1), (6, 8), 2),
        ((8, 256), (256, 1), (6, 8), 32),
        ((1, 1), (1, 1), (2, 32), None),
        ((1, 1), (1, 1), (6, 8), None),
        ((), (), (2, 32), None),
    ],
)
def test_dynamic_alignment_follows_layout(shape, stride, dtype, expected):
    tensor = _MetadataTensor(0x1000, shape=shape, stride=stride, dtype=dtype)
    info = _tensor_info_from_value(tensor, "tensor")
    assert info.alignment is None
    assert info._traced_tensor_ptr == 0x1000
    graph = TraceGraph()
    graph.add_input(info)
    graph.add_output_ref(info)
    assert prepare_tensor_infos(graph)[graph.input_ids[0]].alignment == 32
    overrides = prepare_tensor_infos(graph, dynamic_shape=True)
    for node_id in graph.input_ids + graph.output_ref_ids:
        assert overrides[node_id].alignment == expected
        assert overrides[node_id].shape == (None,) * len(shape)
        assert graph.nodes[node_id].value_info.alignment is None


@pytest.mark.parametrize("pointer", [0x1000, 0x1004, None])
def test_dynamic_alignment_is_independent_of_pointer(pointer):
    graph = TraceGraph()
    graph.add_input(_tensor_info_from_value(_MetadataTensor(pointer), "tensor"))
    overrides = prepare_tensor_infos(graph, dynamic_shape=True)
    assert overrides[graph.input_ids[0]].alignment == 16


@pytest.mark.parametrize("stride", [(16, 2), (24, 3)])
@pytest.mark.parametrize("explicit_alignment", [None, 8])
def test_dynamic_alignment_without_unit_stride(stride, explicit_alignment):
    graph = TraceGraph()
    info = _tensor_info_from_value(_MetadataTensor(0x1000, stride=stride), "tensor")
    info = replace(info, alignment=explicit_alignment)
    graph.add_input(info)
    graph.add_output_ref(info)
    static = prepare_tensor_infos(graph)
    dynamic = prepare_tensor_infos(graph, dynamic_shape=True)
    for node_id in graph.input_ids + graph.output_ref_ids:
        assert static.get(node_id, info).alignment == (explicit_alignment or 32)
        assert dynamic[node_id].alignment == explicit_alignment
        assert all(dim is None for dim in dynamic[node_id].shape)
        assert dynamic[node_id].stride == (None,) * len(stride)
        assert graph.nodes[node_id].value_info.stride == stride


@pytest.mark.parametrize("dynamic_shape", [False, True])
def test_explicit_trace_alignment_overrides_inference(dynamic_shape):
    info = _tensor_info_from_value(_MetadataTensor(0x1000), "tensor")
    info = replace(info, alignment=8)
    graph = TraceGraph()
    graph.add_input(info)
    graph.add_output_ref(info)
    overrides = prepare_tensor_infos(graph, dynamic_shape=dynamic_shape)
    for node_id in graph.input_ids + graph.output_ref_ids:
        assert overrides.get(node_id, info).alignment == 8


def _launch_test_kernel(alignments=(32, 16, 16, 32), *, dynamic_shape=False):
    tensor_type = "tensor<?x?xf32>" if dynamic_shape else "tensor<8x8xf32>"
    stride = "(?,1)" if dynamic_shape else "(8,1)"
    attrs = []
    for alignment in alignments:
        attr = f'nv_tensor_ir.stride = "{stride}"'
        if alignment is not None:
            attr += f", nv_tensor_ir.alignment = {alignment} : i64"
        attrs.append("{" + attr + "}")
    context = ir.Context()
    with context:
        module = ir.Module.parse(
            f"""module {{
              nv_tensor_ir.graph @launch_alignment(
                %a: {tensor_type} {attrs[0]}, %b: {tensor_type} {attrs[1]}
              ) -> ({tensor_type} {attrs[2]}, {tensor_type} {attrs[3]}) {{
                results %a, %b : {tensor_type}, {tensor_type}
              }}
            }}"""
        )
    # Only launch is mocked: alignment attributes come from a real MLIR module,
    # and runtime addresses are extracted through the native DLPack binding.
    artifact = Mock()
    artifact.load.return_value.is_destroyed = False
    return CudaTileCompiledKernel(artifact, module, TraceGraph(), 2, 2, context)


@pytest.mark.parametrize("dynamic_shape", [False, True])
@pytest.mark.parametrize("index", range(4))
def test_launch_rejects_misaligned_input_or_output(index, dynamic_shape):
    kernel = _launch_test_kernel(dynamic_shape=dynamic_shape)
    tensors = [_MetadataTensor(0x1000 * (i + 1)) for i in range(4)]
    tensors[index] = _MetadataTensor(0x1000 * (index + 1), byte_offset=4)
    kind = "Input" if index < 2 else "Output"
    alignment = (32, 16, 16, 32)[index]
    with pytest.raises(
        ValueError, match=f"{kind} tensor {index % 2}.*{alignment} bytes"
    ):
        kernel.run(*tensors[:2], output=tuple(tensors[2:]))
    kernel.artifact.load.assert_not_called()
    kernel.artifact.load.return_value.launch.assert_not_called()


@pytest.mark.parametrize("dynamic_shape", [False, True])
def test_launch_accepts_aligned_addresses(dynamic_shape):
    kernel = _launch_test_kernel(dynamic_shape=dynamic_shape)
    tensors = [_MetadataTensor(0x1000 * (i + 1), byte_offset=32) for i in range(4)]
    kernel.run(*tensors[:2], output=tuple(tensors[2:]))
    kernel.artifact.load.return_value.launch.assert_called_once_with(*tensors, stream=0)


@pytest.mark.parametrize("alignment", [None, 1])
def test_launch_without_alignment_requirement(alignment):
    kernel = _launch_test_kernel((alignment,) * 4)
    tensors = [_MetadataTensor(0x1001 + i * 32) for i in range(4)]
    kernel.run(*tensors[:2], output=tuple(tensors[2:]))
    kernel.artifact.load.return_value.launch.assert_called_once_with(*tensors, stream=0)


def test_launch_rechecks_replacement_tensor():
    kernel = _launch_test_kernel()
    tensor = _MetadataTensor(0x1000)
    kernel.run(tensor, tensor, output=(tensor, tensor))
    with pytest.raises(ValueError, match="Input tensor 0.*32 bytes"):
        kernel.run(_MetadataTensor(0x1004), tensor, output=(tensor, tensor))
    kernel.artifact.load.return_value.launch.assert_called_once()


def test_launch_rejects_null_pointer_with_alignment_requirement():
    kernel = _launch_test_kernel()
    tensor = _MetadataTensor(0x1000)
    with pytest.raises(ValueError, match="Input tensor 0.*non-null"):
        kernel.run(_MetadataTensor(None), tensor, output=(tensor, tensor))
    kernel.artifact.load.assert_not_called()
    kernel.artifact.load.return_value.launch.assert_not_called()


@tir.kernel
def _matmul(a, b):
    return a @ b


def _matmul_module(input_pointer, dynamic_shape):
    graph = tir.trace(
        _matmul,
        _MetadataTensor(input_pointer),
        _MetadataTensor(0x2000),
        output=_MetadataTensor(0x3010),
    )
    overrides = prepare_tensor_infos(graph, dynamic_shape=dynamic_shape)
    module, context = build_mlir_module(
        graph, "aligned_matmul", tensor_info_overrides=overrides
    )
    return graph, module, context


@pytest.mark.parametrize("dynamic_shape", [False, True])
def test_matmul_alignment_attributes_and_cache_key(dynamic_shape):
    graph, module, context = _matmul_module(0x1000, dynamic_shape)
    text = str(module)
    assert module.operation.verify()
    assert text.count("nv_tensor_ir.alignment = 32 : i64") == (
        0 if dynamic_shape else 2
    )
    assert text.count("nv_tensor_ir.alignment = 16 : i64") == (
        3 if dynamic_shape else 1
    )
    assert graph.nodes[graph.input_ids[0]].value_info.alignment is None
    assert graph.nodes[graph.input_ids[0]].value_info._traced_tensor_ptr == 0x1000
    assert graph.nodes[graph.output_ref_ids[0]].value_info.alignment is None
    assert graph.nodes[graph.output_ref_ids[0]].value_info._traced_tensor_ptr == 0x3010
    if dynamic_shape:
        assert "tensor<?x?xf32>" in text

    _, offset_module, offset_context = _matmul_module(0x1004, dynamic_shape)
    _, over_aligned_module, over_aligned_context = _matmul_module(0x8000, dynamic_shape)
    key = tir.calculate_cache_key(module)
    if dynamic_shape:
        assert tir.calculate_cache_key(offset_module) == key
    else:
        assert tir.calculate_cache_key(offset_module) != key
    assert tir.calculate_cache_key(over_aligned_module) == key
