// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_ANALYSIS_TILING_ARCH_H_
#define TENSOR_IR_ANALYSIS_TILING_ARCH_H_

#include "tensor_ir/Dialect/TensorIRAttrs.h"

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {

/// Common parameters for all GPU architectures.
/// Number of threads per warp.
constexpr int kWarpSize = 32;
/// Size of a memory sector in bytes.
constexpr int kMemorySectorSize = 32;
/// Maximum number of registers per thread.
constexpr int kMaxRegistersPerThread = 255;
/// Maximum number of registers per SM.
constexpr int kMaxRegistersPerSM = 65536;
/// Size of a register in bytes.
constexpr int kRegisterSize = 4;

/// Tensor core versions (bit values).
enum TensorCoreVersion {
  kMMAv2 = 1 << 2,
  kMMAv3 = 1 << 3,
  kMMAv5 = 1 << 5,
};

/**
 * @brief GPU architecture information.
 */
struct ArchInfo {
  /// Number of Streaming Multiprocessors (SMs).
  int numSMs = 0;
  /// Maximum number of resident blocks per SM.
  int maxBlocksPerSM;
  /// Maximum number of resident warps per SM.
  int maxWarpsPerSM;
  /// Maximum amount of shared memory per SM (in KB).
  int maxSharedMemoryPerSM;
  /// Supported tensor core versions (bit mask).
  int tensorCoreVersions;
};

/**
 * @brief Get the architecture information for a given compute capability.
 * @param cc Compute capability.
 * @param numSMs Number of Streaming Multiprocessors (SMs).
 */
ArchInfo getArchInfo(ComputeCapability cc, int numSMs);

} // namespace tiling_analysis
} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_ANALYSIS_TILING_ARCH_H_
