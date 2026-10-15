// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/Tiling/Arch.h"

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {
namespace {

static const ArchInfo archInfoSm80 = {
    .maxBlocksPerSM = 32,
    .maxWarpsPerSM = 64,
    .maxSharedMemoryPerSM = 164,
    .tensorCoreVersions = kMMAv2,
};

static const ArchInfo archInfoSm86 = {
    .maxBlocksPerSM = 16,
    .maxWarpsPerSM = 48,
    .maxSharedMemoryPerSM = 100,
    .tensorCoreVersions = kMMAv2,
};

static const ArchInfo archInfoSm87 = {
    .maxBlocksPerSM = 16,
    .maxWarpsPerSM = 48,
    .maxSharedMemoryPerSM = 164,
    .tensorCoreVersions = kMMAv2,
};

static const ArchInfo archInfoSm89 = {
    .maxBlocksPerSM = 24,
    .maxWarpsPerSM = 48,
    .maxSharedMemoryPerSM = 100,
    .tensorCoreVersions = kMMAv2,
};

static const ArchInfo archInfoSm90 = {
    .maxBlocksPerSM = 32,
    .maxWarpsPerSM = 64,
    .maxSharedMemoryPerSM = 228,
    .tensorCoreVersions = kMMAv2 | kMMAv3,
};

static const ArchInfo archInfoSm100 = {
    .maxBlocksPerSM = 32,
    .maxWarpsPerSM = 64,
    .maxSharedMemoryPerSM = 228,
    .tensorCoreVersions = kMMAv2 | kMMAv5,
};

static const ArchInfo archInfoSm103 = {
    .maxBlocksPerSM = 32,
    .maxWarpsPerSM = 64,
    .maxSharedMemoryPerSM = 228,
    .tensorCoreVersions = kMMAv2 | kMMAv5,
};

static const ArchInfo archInfoSm110 = {
    .maxBlocksPerSM = 24,
    .maxWarpsPerSM = 48,
    .maxSharedMemoryPerSM = 228,
    .tensorCoreVersions = kMMAv2 | kMMAv5,
};

static const ArchInfo archInfoSm120 = {
    .maxBlocksPerSM = 24,
    .maxWarpsPerSM = 48,
    .maxSharedMemoryPerSM = 100,
    .tensorCoreVersions = kMMAv2,
};

static const ArchInfo archInfoSm121 = {
    .maxBlocksPerSM = 24,
    .maxWarpsPerSM = 48,
    .maxSharedMemoryPerSM = 100,
    .tensorCoreVersions = kMMAv2,
};

static const ArchInfo archInfoGeneric = {
    .maxBlocksPerSM = 16,
    .maxWarpsPerSM = 48,
    .maxSharedMemoryPerSM = 100,
    .tensorCoreVersions = 0,
};

ArchInfo getArchInfo(ComputeCapability cc) {
  switch (cc) {
  case ComputeCapability::Sm80:
    return archInfoSm80;
  case ComputeCapability::Sm86:
    return archInfoSm86;
  case ComputeCapability::Sm87:
    return archInfoSm87;
  case ComputeCapability::Sm89:
    return archInfoSm89;
  case ComputeCapability::Sm90:
    return archInfoSm90;
  case ComputeCapability::Sm100:
    return archInfoSm100;
  case ComputeCapability::Sm103:
    return archInfoSm103;
  case ComputeCapability::Sm110:
    return archInfoSm110;
  case ComputeCapability::Sm120:
    return archInfoSm120;
  case ComputeCapability::Sm121:
    return archInfoSm121;
  default:
    return archInfoGeneric;
  }
}

} // namespace

/**
 * @brief Get the architecture information for a given compute capability.
 * @param cc Compute capability.
 * @param numSMs Number of Streaming Multiprocessors (SMs).
 */
ArchInfo getArchInfo(ComputeCapability cc, int numSMs) {
  ArchInfo archInfo = getArchInfo(cc);
  archInfo.numSMs = std::max(numSMs, 1);
  return archInfo;
}

} // namespace tiling_analysis
} // namespace mlir::nv_tensor_ir
