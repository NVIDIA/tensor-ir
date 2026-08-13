# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# -*- Python -*-

#
# GPU detection for lit.cfg.py. Kept out of lit.cfg.py so that it can be unit
# tested with a fake driver; see test/Config/gpu_detection.py.

import ctypes
from collections.abc import Callable
from typing import Protocol


class CudaDriver(Protocol):
    """CUDA Driver API calls used to inspect the first visible device."""

    def cuInit(self, flags: int) -> int: ...

    def cuDeviceGetCount(self, count: object) -> int: ...

    def cuDeviceGet(self, device: object, ordinal: int) -> int: ...

    def cuDeviceGetAttribute(
        self, value: object, attribute: int, device: int
    ) -> int: ...


CudaDriverLoader = Callable[[str], CudaDriver]

_CUDA_SUCCESS = 0
_COMPUTE_CAPABILITY_MAJOR = 75
_COMPUTE_CAPABILITY_MINOR = 76


def detect_visible_gpu_arch(
    driver_loader: CudaDriverLoader = ctypes.CDLL,
) -> str:
    """Return the compute capability of CUDA's first visible device.

    The CUDA Driver API applies ``CUDA_VISIBLE_DEVICES`` before enumerating
    devices, so ordinal zero is the GPU that a test process will actually use.
    The result is formatted the way the rest of the test suite spells a compute
    capability, i.e. major * 10 + minor ("100" for a 10.0 device), matching
    utils.detect_sm_version() and normalize_cubin_chip(). An empty string
    indicates that no usable CUDA device is visible.

    Note that this initializes the CUDA driver in the lit process itself. That
    is safe because lit workers only ever launch tests as child processes, but
    it is the reason this runs once at configuration time instead of per test.
    """
    try:
        driver = driver_loader("libcuda.so.1")
    except OSError:
        return ""

    if driver.cuInit(0) != _CUDA_SUCCESS:
        return ""

    device_count = ctypes.c_int()
    if (
        driver.cuDeviceGetCount(ctypes.byref(device_count)) != _CUDA_SUCCESS
        or device_count.value == 0
    ):
        return ""

    device = ctypes.c_int()
    if driver.cuDeviceGet(ctypes.byref(device), 0) != _CUDA_SUCCESS:
        return ""

    major = ctypes.c_int()
    minor = ctypes.c_int()
    if (
        driver.cuDeviceGetAttribute(
            ctypes.byref(major), _COMPUTE_CAPABILITY_MAJOR, device.value
        )
        != _CUDA_SUCCESS
        or driver.cuDeviceGetAttribute(
            ctypes.byref(minor), _COMPUTE_CAPABILITY_MINOR, device.value
        )
        != _CUDA_SUCCESS
    ):
        return ""
    return str(major.value * 10 + minor.value)
