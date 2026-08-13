# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections.abc as _collections_abc

# These imports intentionally populate this module's public dialect API. The
# dialect enums come from the generated enum module; do not hand-write copies
# here, they would shadow the generated ones and drift from the .td files.
from ._nv_tensor_ir_ops_gen import *  # noqa: F403
from ._nv_tensor_ir_ops_gen import (
    _Dialect,  # noqa: F401
    broadcast as _generated_broadcast,
    iota as _generated_iota,
    reshape as _generated_reshape,
    splat as _generated_splat,
)
from ._nv_tensor_ir_enum_gen import *  # noqa: F403

from .._mlir_libs import _tensor_ir as _tensor_ir_cext  # noqa: E402

# The two imports below must stay in this order: the wildcard exports the native
# `compile`/`can_compile`, and `nv_tensor_ir.runtime` re-exports the Python
# wrappers that are meant to shadow them. Sorting these imports breaks that.
from .._mlir_libs._tensor_ir import *  # noqa: F403, E402
from nv_tensor_ir.runtime import (  # noqa: F401, E402
    CompileOptions,
    Program,
    CudaTileArtifactKind,
    can_compile,
    compile,
)

try:
    from ._ods_common import _cext as _ods_cext
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

import nv_tensor_ir._mlir.ir as ir  # noqa: E402


def broadcast(
    output: ir.Type,
    input: ir.Value,
    dynamic_sizes: _collections_abc.Sequence[ir.Value] = (),
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """Build a broadcast, defaulting to legacy implicit shape metadata."""
    return _generated_broadcast(output, input, dynamic_sizes, loc=loc, ip=ip)


def iota(
    output: ir.Type,
    dimension: int | ir.IntegerAttr,
    dynamic_sizes: _collections_abc.Sequence[ir.Value] = (),
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """Build an iota, defaulting to legacy implicit shape metadata."""
    return _generated_iota(output, dimension, dynamic_sizes, loc=loc, ip=ip)


def reshape(
    output: ir.Type,
    input: ir.Value,
    dynamic_sizes: _collections_abc.Sequence[ir.Value] = (),
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """Build a reshape, defaulting to legacy implicit shape metadata."""
    return _generated_reshape(output, input, dynamic_sizes, loc=loc, ip=ip)


def splat(
    output: ir.Type,
    input: ir.Value,
    dynamic_sizes: _collections_abc.Sequence[ir.Value] = (),
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """Build a splat, defaulting to legacy implicit shape metadata."""
    return _generated_splat(output, input, dynamic_sizes, loc=loc, ip=ip)


def register_dialect(context: ir.Context | None = None, load: bool = True) -> None:
    """Register the TensorIR dialect with an MLIR context.

    `context` defaults to the innermost active `ir.Context`, i.e. calling this
    inside a `with ir.Context():` block needs no argument. Existing callers
    depend on that default, so it is part of the supported API.
    """
    if context is None:
        context = ir.Context.current
        if context is None:
            raise TypeError(
                "register_dialect() needs an MLIR context: pass one explicitly, "
                "or call it inside a `with ir.Context():` block"
            )
    _tensor_ir_cext.register_dialect(context, load=load)


def get_tensor_datatype(type: ir.Type) -> ir.Type:
    """Return the element type of a TensorIR-compatible tensor type."""
    return ir.ShapedType(type).element_type


def get_value_datatype(type: ir.Value) -> ir.Type:
    """Return the element type of a TensorIR-compatible tensor value."""
    return get_tensor_datatype(type.type)


def get_tensor_type(
    shape: _collections_abc.Sequence[int],
    datatype: ir.Type,
    *,
    context: ir.Context | None = None,
) -> ir.RankedTensorType:
    """Create a TensorIR-compatible ranked tensor type."""
    dynamic_size = ir.ShapedType.get_dynamic_size()
    normalized_shape = [
        dynamic_size if dimension == -1 else dimension for dimension in shape
    ]
    encoding = None
    return ir.RankedTensorType.get(
        normalized_shape,
        datatype,
        encoding=encoding,
        loc=ir.Location.unknown(context),
    )


def get_tensor_type_from_tensor_type(
    type: ir.Type,
    dataType: ir.Type,
    *,
    context: ir.Context | None = None,
) -> ir.RankedTensorType:
    """Clone a ranked tensor type with a different element type."""
    ranked_type = ir.RankedTensorType(type)
    return ir.RankedTensorType.get(
        ranked_type.shape,
        dataType,
        encoding=ranked_type.encoding,
        loc=ir.Location.unknown(context),
    )


def get_tensor_type_from_value_type(
    type: ir.Value,
    dataType: ir.Type,
    *,
    context: ir.Context | None = None,
) -> ir.RankedTensorType:
    """Clone a value's ranked tensor type with a different element type."""
    return get_tensor_type_from_tensor_type(
        type=type.type, dataType=dataType, context=context
    )


class _TensorTypeMeta(type):
    """Preserve runtime type checks for the former custom tensor type."""

    def __instancecheck__(cls, instance: object) -> bool:
        """Return whether an object is an MLIR ranked tensor type."""
        return isinstance(instance, ir.RankedTensorType)


class TensorType(metaclass=_TensorTypeMeta):
    """Compatibility API backed by MLIR's builtin ranked tensor type."""

    get = staticmethod(get_tensor_type)
    get_from_tensor_type = staticmethod(get_tensor_type_from_tensor_type)
    get_from_value_type = staticmethod(get_tensor_type_from_value_type)


def _tensorir_float_attr(x, context=None):
    if context is None:
        context = ir.Context.current
    f32_type = ir.F32Type.get(context=context)
    return ir.FloatAttr.get(f32_type, float(x))


_ods_cext.ir.AttrBuilder.insert("TensorIR_FloatAttr", _tensorir_float_attr)


class TensorIRTensorDescriptor:
    def __init__(self, tensor, overwrite_strides=None, overwrite_shape=None):
        self.tensor = tensor
        if not (hasattr(tensor, "__dlpack__") and hasattr(tensor, "__dlpack_device__")):
            raise ValueError(
                "Tensor must support DLPack protocol (__dlpack__ and __dlpack_device__)"
            )

        if hasattr(tensor, "device") and hasattr(tensor, "numel"):
            if tensor.device.type == "cpu" and tensor.numel() > 1:
                raise ValueError("Multi-element tensors must be on GPU for TensorIR")
            if tensor.device.type not in ["cpu", "cuda"]:
                raise ValueError(
                    f"Unsupported device type: {tensor.device.type}. Only 'cpu' and 'cuda' are supported"
                )
        try:
            self.dlpack_capsule = tensor.__dlpack__()
        except (RuntimeError, BufferError) as e:
            if "float8 types are not supported by dlpack" in str(e):
                # TODO: remove this when PyTorch natively supports FP8 in DLPack
                self.dlpack_capsule = self._create_fp8_dlpack_workaround(tensor)
            else:
                raise RuntimeError(f"Failed to create DLPack capsule: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create DLPack capsule: {e}")
        if overwrite_strides is not None:
            self.overwrite_strides(overwrite_strides)
        if overwrite_shape is not None:
            self.overwrite_shape(overwrite_shape)

    def overwrite_strides(self, overwrite_strides):
        if self.dlpack_capsule is None:
            raise RuntimeError("DLPack capsule is not set")
        managed_tensor = self._get_managed_tensor(self.dlpack_capsule)
        if managed_tensor.dl_tensor.ndim != len(overwrite_strides):
            raise RuntimeError(
                f"Number of dimensions mismatch: {managed_tensor.dl_tensor.ndim} != {len(overwrite_strides)}"
            )
        for dim in range(managed_tensor.dl_tensor.ndim):
            managed_tensor.dl_tensor.strides[dim] = overwrite_strides[dim]

    def overwrite_shape(self, overwrite_shape):
        """Overwrite the shape field in the DLPack tensor descriptor.

        This is needed for packed data types (e.g. FP4, where 2 elements share
        one byte) where the physical allocation size differs from the logical
        element count.  The kernel uses the shape to infer problem dimensions
        (M, N, K), so the logical element counts must be provided here.
        """
        if self.dlpack_capsule is None:
            raise RuntimeError("DLPack capsule is not set")
        managed_tensor = self._get_managed_tensor(self.dlpack_capsule)
        if managed_tensor.dl_tensor.ndim != len(overwrite_shape):
            raise RuntimeError(
                f"Number of dimensions mismatch: {managed_tensor.dl_tensor.ndim} != {len(overwrite_shape)}"
            )
        for dim in range(managed_tensor.dl_tensor.ndim):
            managed_tensor.dl_tensor.shape[dim] = overwrite_shape[dim]

    def _create_fp8_dlpack_workaround(self, tensor):
        """
        Creates a DLPack workaround for FP8 tensors by converting to uint8 and patching metadata.

        This method orchestrates the complete FP8 workaround process:
        1. Creates a uint8 view of the FP8 tensor (preserving raw bytes)
        2. Generates a DLPack capsule from the uint8 tensor
        3. Patches the DLPack dtype metadata to reflect the original FP8 type

        Args:
            tensor: A PyTorch tensor with FP8 dtype (float8_e4m3fn or float8_e5m2)

        Returns:
            DLPack capsule with corrected FP8 dtype metadata

        Raises:
            RuntimeError: If tensor is not FP8 or if DLPack creation fails

        Note:
            TODO: Remove this method when PyTorch natively supports FP8 in DLPack.
            This is a high-level orchestration method that calls _patch_dlpack_dtype_for_fp8
            to perform the actual low-level dtype patching.
        """
        import torch

        if tensor.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            raise RuntimeError(f"Expected FP8 tensor, got {tensor.dtype}")

        uint8_tensor = tensor.view(torch.uint8)
        dlpack_capsule = torch.utils.dlpack.to_dlpack(uint8_tensor)
        return self._patch_dlpack_dtype_for_fp8(dlpack_capsule, tensor.dtype)

    def _patch_dlpack_dtype_for_fp8(self, dlpack_capsule, original_dtype):
        """
        Performs low-level patching of DLPack dtype metadata for FP8 tensors.

        This method directly manipulates the DLPack tensor structure in memory to correct
        the dtype code from uint8 to the appropriate FP8 type. It uses ctypes to:
        1. Extract the DLTensor pointer from the DLPack capsule
        2. Access the low-level DLDataType structure
        3. Modify the dtype code field to match the original FP8 format

        Args:
            dlpack_capsule: DLPack capsule containing a uint8 tensor
            original_dtype: Original PyTorch FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2)

        Returns:
            Modified DLPack capsule with corrected FP8 dtype metadata

        Raises:
            RuntimeError: If capsule access fails or dtype patching fails

        Note:
            TODO: Remove this method when PyTorch natively supports FP8 in DLPack.
            This is a low-level implementation method called by _create_fp8_dlpack_workaround.
            It performs the actual memory manipulation while the caller handles orchestration.
        """

        import torch

        managed_tensor = self._get_managed_tensor(dlpack_capsule)

        # Modify the dtype code
        if original_dtype == torch.float8_e4m3fn:
            new_dtype_code = 10  # kDLFloat8_e4m3fn
        elif original_dtype == torch.float8_e5m2:
            new_dtype_code = 12  # kDLFloat8_e5m2
        else:
            raise RuntimeError(f"Unsupported FP8 dtype: {original_dtype}")
        managed_tensor.dl_tensor.dtype.code = new_dtype_code
        return dlpack_capsule

    def _get_managed_tensor(self, dlpack_capsule):
        import ctypes

        PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
        PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        PyCapsule_GetPointer.restype = ctypes.c_void_p

        managed_tensor_ptr = PyCapsule_GetPointer(dlpack_capsule, b"dltensor")
        if not managed_tensor_ptr:
            raise RuntimeError("Failed to get DLPack tensor pointer from capsule")

        class DLDataType(ctypes.Structure):
            _fields_ = [
                ("code", ctypes.c_uint8),
                ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16),
            ]

        class DLDevice(ctypes.Structure):
            _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]

        class DLTensor(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.c_void_p),
                ("device", DLDevice),
                ("ndim", ctypes.c_int32),
                ("dtype", DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                # We don't need the rest of the fields for our purpose
            ]

        class DLManagedTensor(ctypes.Structure):
            _fields_ = [("dl_tensor", DLTensor)]

        managed_tensor = ctypes.cast(
            managed_tensor_ptr, ctypes.POINTER(DLManagedTensor)
        ).contents
        return managed_tensor

    def __str__(self):
        return f"TensorIRTensorDescriptor({self.tensor.shape} {self.tensor.dtype} on {self.tensor.device})"


# Runtime functions for POC
def from_dlpack(tensor_dlpack, assumed_align=None):
    """Convert DLPack tensor to TensorIR descriptor."""
    return TensorIRTensorDescriptor(tensor_dlpack)


def create_tensor_descriptor(tensor):
    """Create TensorIR descriptor from tensor."""
    if isinstance(tensor, TensorIRTensorDescriptor):
        return tensor
    else:
        return TensorIRTensorDescriptor(tensor)


def is_tensor_like(obj):
    """Check if object supports DLPack protocol."""
    return hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__")
