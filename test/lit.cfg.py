# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# -*- Python -*-

import importlib.util
import os

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst


def _load_sibling_module(name):
    """Import a helper module that sits next to this config file."""
    path = os.path.join(os.path.dirname(__file__), f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


detect_visible_gpu_arch = _load_sibling_module("lit_gpu").detect_visible_gpu_arch

# Configuration file for the 'lit' test runner

# name: The name of this test suite
config.name = "TENSOR_IR"

if config.tensor_ir_oss_layout:
    config.available_features.add("tensor-ir-oss")

# ShTest's first arg is execute_external: on Linux, use_lit_shell is False by
# default, so this historically selected the system /bin/sh. LLVM 23 rejects
# execute_external=True unless force_execute_external=True (removed in LLVM 24);
# catch ValueError so older lit still works while we keep external-shell behavior.
# Prefer migrating to lit.formats.ShTest() (internal shell) when RUN lines allow.
try:
    config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
except ValueError:
    config.test_format = lit.formats.ShTest(
        not llvm_config.use_lit_shell, force_execute_external=True
    )

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.tensor_ir_obj_root, "test")

config.excludes = config.tensor_ir_excludes

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))


# Searches for a runtime library with the given name and returns the found path.
# Correctly handles the platforms shared library directory and naming conventions.
def find_runtime(name):
    check_dirs = [config.llvm_lib_dir, config.tensor_ir_lib_dir]
    nt_check_dirs = [
        config.llvm_tools_dir,
        config.tensor_ir_tools_dir,
    ]
    for prefix in ["", "lib"]:
        for check_dir in check_dirs:
            path = os.path.join(check_dir, f"{prefix}{name}{config.llvm_shlib_ext}")
            if os.path.isfile(path):
                return path
        for check_dir in nt_check_dirs:
            path = os.path.join(check_dir, f"{prefix}{name}{config.llvm_shlib_ext}")
            if os.path.isfile(path):
                return path
    return None


# Searches for a runtime library with the given name and returns a tool
# substitution of the same name and the found path.
def add_runtime(name):
    return ToolSubst(f"%{name}", find_runtime(name))


tool_dirs = [
    config.tensor_ir_tools_dir,
    config.llvm_tools_dir,
]

tools = [
    "tensor_ir-compiler",
    "tensor_ir-opt",
    "FileCheck",
    "not",
    ToolSubst("%PYTHON", config.python_executable, unresolved="ignore"),
]



llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment("PATH", config.tensor_ir_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# so that NVPTXSerializer can find ptxas/libdevice
if config.cuda_toolkit_path:
    llvm_config.with_environment("CUDA_ROOT", config.cuda_toolkit_path)

if config.enable_bindings_python:
    paths = [
        (
            os.path.join(config.tensor_ir_install_root, "python_packages")
            if config.tensor_ir_install_root
            else config.tensor_ir_python_package_prefix
        ),
        os.path.join(config.test_source_root, "python"),
        os.path.join(config.test_source_root, "python", "Compiler"),
    ]
    llvm_config.with_environment("PYTHONPATH", paths, append_path=True)

if config.tensor_ir_include_tests:
    config.available_features.add("enabled")


def normalize_cubin_chip(gpu_arch):
    arch = gpu_arch
    if arch.startswith("sm_"):
        arch = arch[len("sm_") :]

    # CI exports TEST_GPU_ARCH as the device compute capability, e.g.
    # `sm_100`. Lit `%cubin_chip` is a codegen target and Blackwell+ needs the
    # arch-conditional suffix used by the auto-detected path below.
    if arch.isdigit() and int(arch) >= 90:
        arch += "a"
    return f"sm_{arch}"


# There are two independent notions of "GPU architecture" in this suite, and
# they are deliberately not tied together:
#
#   * The `cuda-gpu` / `cuda-sm<cc>` features describe the device that test
#     processes will actually get, honoring CUDA_VISIBLE_DEVICES. Tests that
#     launch kernels or query the device gate on these, so a machine without a
#     GPU reports them UNSUPPORTED instead of failing. TEST_GPU_ARCH does not
#     add these features: claiming a device exists cannot make one appear.
#   * `%cubin_chip` is only a codegen target. TEST_GPU_ARCH overrides it so that
#     compile-only tests can target an architecture other than the local one,
#     which means `%cubin_chip` may legitimately disagree with `cuda-sm<cc>`.
config.gpu_arch = detect_visible_gpu_arch()
if config.gpu_arch:
    config.available_features.add("cuda-gpu")
    config.available_features.add(f"cuda-sm{config.gpu_arch}")

config.environment.pop("TENSOR_IR_SKIP_RUNTIME_LAUNCH", None)


if "TEST_GPU_ARCH" in os.environ:
    config.cubin_chip = normalize_cubin_chip(os.environ["TEST_GPU_ARCH"])
elif config.gpu_arch:
    config.cubin_chip = normalize_cubin_chip(config.gpu_arch)
else:
    print("No CUDA GPU detected; using default codegen arch sm_80")
    config.cubin_chip = "sm_80"

config.substitutions.append(("%cubin_chip", config.cubin_chip))

# CUDA treats an empty CUDA_VISIBLE_DEVICES value as "hide every device", so
# preserve it instead of using the truthiness check for optional variables.
for k in ["CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES"]:
    if k in os.environ:
        llvm_config.with_environment(k, os.environ[k])

# set up optional env vars
for k in [
    # sanitizer
    "ASAN_OPTIONS",
    "CUDA_CACHE_DISABLE",
    "CUDA_DISABLE_PTX_JIT",
    "LD_LIBRARY_PATH",
    "LOAD_EXE_BEFORE_EXECUTION",
]:
    if os.environ.get(k):
        llvm_config.with_environment(k, os.environ.get(k))
