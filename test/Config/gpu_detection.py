# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

"""Unit tests for the GPU detection used by lit.cfg.py.

detect_visible_gpu_arch decides which tests are reported as UNSUPPORTED, so its
failure paths must not depend on the machine the tests happen to run on. Every
case below drives it with a fake CUDA driver.
"""

import ctypes
import importlib.util
import unittest
from pathlib import Path

_LIT_GPU_PATH = Path(__file__).parents[1] / "lit_gpu.py"
_spec = importlib.util.spec_from_file_location("lit_gpu", _LIT_GPU_PATH)
lit_gpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lit_gpu)

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_INITIALIZED = 3
COMPUTE_CAPABILITY_MAJOR = 75


def set_int(pointer, value):
    ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = value


class FakeCudaDriver:
    """A libcuda.so.1 stand-in covering the four entry points that are used."""

    def __init__(
        self,
        *,
        init_status=CUDA_SUCCESS,
        count_status=CUDA_SUCCESS,
        attribute_status=CUDA_SUCCESS,
        device_count=1,
        major=10,
        minor=0,
    ):
        self.init_status = init_status
        self.count_status = count_status
        self.attribute_status = attribute_status
        self.device_count = device_count
        self.major = major
        self.minor = minor

    def cuInit(self, flags):
        return self.init_status

    def cuDeviceGetCount(self, count):
        set_int(count, self.device_count)
        return self.count_status

    def cuDeviceGet(self, device, ordinal):
        set_int(device, ordinal)
        return CUDA_SUCCESS

    def cuDeviceGetAttribute(self, value, attribute, device):
        set_int(
            value, self.major if attribute == COMPUTE_CAPABILITY_MAJOR else self.minor
        )
        return self.attribute_status


def detect(driver):
    return lit_gpu.detect_visible_gpu_arch(lambda _: driver)


class DetectVisibleGpuArchTest(unittest.TestCase):
    def test_reports_first_visible_device_arch(self):
        self.assertEqual(detect(FakeCudaDriver(major=10, minor=0)), "100")

    def test_formats_arch_as_major_times_ten_plus_minor(self):
        # Spelled this way rather than by concatenating the two numbers so that
        # it agrees with utils.detect_sm_version() for two-digit minors.
        self.assertEqual(detect(FakeCudaDriver(major=9, minor=0)), "90")
        self.assertEqual(detect(FakeCudaDriver(major=12, minor=1)), "121")

    def test_reports_no_gpu_when_driver_is_unavailable(self):
        def fail_to_load(_):
            raise OSError

        self.assertEqual(lit_gpu.detect_visible_gpu_arch(fail_to_load), "")

    def test_reports_no_gpu_when_driver_fails_to_initialize(self):
        driver = FakeCudaDriver(init_status=CUDA_ERROR_NOT_INITIALIZED)
        self.assertEqual(detect(driver), "")

    def test_reports_no_gpu_when_all_devices_are_hidden(self):
        # This is what CUDA_VISIBLE_DEVICES="" looks like to the driver.
        self.assertEqual(detect(FakeCudaDriver(device_count=0)), "")

    def test_reports_no_gpu_when_device_count_query_fails(self):
        driver = FakeCudaDriver(count_status=CUDA_ERROR_NOT_INITIALIZED)
        self.assertEqual(detect(driver), "")

    def test_reports_no_gpu_when_attribute_query_fails(self):
        driver = FakeCudaDriver(attribute_status=CUDA_ERROR_NOT_INITIALIZED)
        self.assertEqual(detect(driver), "")


if __name__ == "__main__":
    unittest.main()
