# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os

import lit.formats

# name: The name of this test suite.
config.name = "TensorIR-CppUnit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.tensor_ir_obj_root, "unittests/")
config.test_source_root = config.test_exec_root

# Run each GoogleTest binary as a single lit test instead of splitting it into
# shards. Lit picks the shard count from the host's core count, independently of
# -j, and every shard is a fresh process that pays for loading the compiler and
# CUDA libraries before it can run a single case. That startup cost dominates
# this suite: disabling sharding takes `check-tensor-ir` from ~570s to ~180s at
# -j20 on a 224-core host.
#
# This is a property of the suite rather than of one build target, so it lives
# here instead of in the `ARGS` of individual `add_lit_testsuite` calls; that way
# it also applies when a developer runs llvm-lit directly or via one of the
# per-directory targets created by `add_lit_testsuites`. Pass
# `--param=gtest-sharding` to re-enable sharding when measuring whether a
# particular host benefits from it; the `--gtest-sharding` flag cannot serve as
# that override because lit cannot distinguish it from the default.
if "gtest-sharding" not in lit_config.params:
    lit_config.gtest_sharding = False

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")

# Propagate the temp directory. Windows requires this because it uses \Windows\
# if none of these are present.
if "TMP" in os.environ:
    config.environment["TMP"] = os.environ["TMP"]
if "TEMP" in os.environ:
    config.environment["TEMP"] = os.environ["TEMP"]

# Propagate HOME as it can be used to override incorrect homedir in passwd
# that causes the tests to fail.
if "HOME" in os.environ:
    config.environment["HOME"] = os.environ["HOME"]

# Propagate sanitizer options.
for var in [
    "ASAN_SYMBOLIZER_PATH",
    "HWASAN_SYMBOLIZER_PATH",
    "MSAN_SYMBOLIZER_PATH",
    "TSAN_SYMBOLIZER_PATH",
    "UBSAN_SYMBOLIZER_PATH",
    "ASAN_OPTIONS",
    "HWASAN_OPTIONS",
    "MSAN_OPTIONS",
    "TSAN_OPTIONS",
    "UBSAN_OPTIONS",
]:
    if var in os.environ:
        config.environment[var] = os.environ[var]

# so that NVPTXSerializer can find ptxas/libdevice
config.environment['CUDA_ROOT'] = config.cuda_toolkit_path

# Match the CUDA device selection used by the main TensorIR test suite.
# Preserve an empty CUDA_VISIBLE_DEVICES value because it hides every device.
for var in ["CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES"]:
    if var in os.environ:
        config.environment[var] = os.environ[var]
