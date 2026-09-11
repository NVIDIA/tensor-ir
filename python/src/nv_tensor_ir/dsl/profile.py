# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUPTI-backed runtime profiling for TensorIR DSL kernels."""

import queue
from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from statistics import fmean, median, variance
from threading import Lock


_OUTLIER_FILTER_MIN_SAMPLES = 30
_PROFILING_INSTALL_HINT = (
    "TensorIR kernel profiling requires the optional cupti-python and "
    "cuda-python packages. Install them with "
    "`python -m pip install cupti-python cuda-python`."
)
_cupti_initialized = False
_cupti_buffer_queue = queue.Queue()
_profile_lock = Lock()


def init_cupti():
    global _cupti_initialized

    if _cupti_initialized:
        return

    try:
        from cupti import cupti
    except ImportError as exc:
        raise ImportError(_PROFILING_INSTALL_HINT) from exc

    def func_buffer_requested():
        buffer_size = 8 * 1024 * 1024  # 8MB buffer
        max_num_records = 0
        return buffer_size, max_num_records

    def func_buffer_completed(activities: list):
        for activity in activities:
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                _cupti_buffer_queue.put((activity.name, activity.end - activity.start))

    cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)

    _cupti_initialized = True


@dataclass(frozen=True)
class ProfilingConfig:
    """Controls warmup and measured launches for a profiled DSL kernel."""

    warmup: int = 1
    iterations: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int):
            raise TypeError("profile warmup must be an integer")
        if self.warmup < 0:
            raise ValueError("profile warmup must be non-negative")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise TypeError("profile iterations must be an integer")
        if self.iterations <= 0:
            raise ValueError("profile iterations must be positive")

    def __repr__(self) -> str:
        return f"ProfilingConfig(warmup={self.warmup}, iterations={self.iterations})"


@dataclass(frozen=True)
class _TimingStats:
    minimum: float
    average: float
    median: float
    maximum: float
    outliers: int | None


@dataclass(frozen=True)
class KernelProfile:
    """Raw CUPTI durations and summary statistics for one CUDA kernel."""

    name: str
    durations_us: tuple[float, ...]
    minimum_us: float
    average_us: float
    median_us: float
    maximum_us: float
    outliers: int | None

    def __str__(self) -> str:
        message = (
            f"{self.name}: calls={len(self.durations_us)}, "
            f"min={self.minimum_us:.3f} us, "
            f"max={self.maximum_us:.3f} us, "
            f"median={self.median_us:.3f} us, "
            f"avg={self.average_us:.3f} us"
        )
        if self.outliers is not None:
            message += f", outliers={self.outliers}"
        return message


@dataclass(frozen=True)
class ProfileResult:
    """Profiling results grouped by CUPTI kernel name."""

    kernels: tuple[KernelProfile, ...]


def profile_launches(
    launch: Callable[[], None],
    config: ProfilingConfig,
    *,
    print_results: bool = True,
) -> ProfileResult:
    """Profile launches directly with CUPTI and optionally print kernel timings."""

    with _profile_lock:
        init_cupti()
        try:
            from cupti import cupti
            from cuda.bindings.runtime import cudaDeviceSynchronize, cudaError_t
        except ImportError as exc:
            raise ImportError(_PROFILING_INSTALL_HINT) from exc

        def synchronize() -> None:
            (error,) = cudaDeviceSynchronize()
            if error != cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaDeviceSynchronize failed with {error}")

        for _ in range(config.warmup):
            launch()
        synchronize()

        kernel_times: dict[str, list[float]] = {}
        for _ in range(config.iterations):
            assert _cupti_buffer_queue.empty()
            cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
            try:
                launch()
                synchronize()
                cupti.activity_flush_all(1)
            finally:
                cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
            while not _cupti_buffer_queue.empty():
                name, duration = _cupti_buffer_queue.get()
                kernel_times.setdefault(name, []).append(duration / 1e3)  # ns to us

    result = ProfileResult(
        kernels=tuple(
            _kernel_profile(name, durations_us)
            for name, durations_us in kernel_times.items()
        )
    )
    if print_results:
        for kernel in result.kernels:
            print(f"[TENSOR_IR_PROFILE] {kernel}")
    return result


def _kernel_profile(name: str, durations_us: list[float]) -> KernelProfile:
    stats = _timing_stats(durations_us)
    return KernelProfile(
        name=name,
        durations_us=tuple(durations_us),
        minimum_us=stats.minimum,
        average_us=stats.average,
        median_us=stats.median,
        maximum_us=stats.maximum,
        outliers=stats.outliers,
    )


def _timing_stats(times: list[float]) -> _TimingStats:
    """Calculate timing statistics, filtering 3-sigma outliers when stable."""
    if not times:
        raise ValueError("Cannot summarize an empty timing sequence")

    filtered_times = times
    outliers = None
    if len(times) >= _OUTLIER_FILTER_MIN_SAMPLES:
        sample_variance = variance(times)
        standard_deviation = sqrt(sample_variance)
        outliers = 0
        if standard_deviation > 0:
            sample_mean = fmean(times)
            lower_bound = sample_mean - 3 * standard_deviation
            upper_bound = sample_mean + 3 * standard_deviation
            filtered_times = [
                time for time in times if lower_bound <= time <= upper_bound
            ]
            outliers = len(times) - len(filtered_times)

    return _TimingStats(
        minimum=min(filtered_times),
        average=fmean(filtered_times),
        median=median(filtered_times),
        maximum=max(filtered_times),
        outliers=outliers,
    )
