# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tensor-ir-profile",
        action="store_true",
        help="Profile tests that explicitly request the profiling fixture.",
    )
    parser.addoption(
        "--tensor-ir-profile-filter",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help=(
            "Only profile tests whose node ID contains this substring. "
            "May be specified multiple times."
        ),
    )


@pytest.fixture
def profiling(request: pytest.FixtureRequest) -> bool:
    """Return whether profiling was requested for this opted-in test."""
    run_profiling = bool(request.config.getoption("--tensor-ir-profile"))
    profile_filters = request.config.getoption("--tensor-ir-profile-filter")
    return run_profiling and (
        not profile_filters
        or any(
            profile_filter in request.node.nodeid for profile_filter in profile_filters
        )
    )


@pytest.fixture(autouse=True)
def seed_rng(request):
    """Seed torch RNG with a deterministic per-test value.

    The seed is derived from the test's node ID (e.g.
    "TensorToCudaTile/test_pointwise_ops.py::test_add_op[float32]"),
    so each test always gets the same seed regardless of execution order,
    parallelism, or -k filtering.
    """

    seed = int(hashlib.sha256(request.node.nodeid.encode()).hexdigest(), 16) % (2**31)
    torch.manual_seed(seed)
