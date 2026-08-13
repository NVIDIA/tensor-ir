// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Utils/ComputeCapability.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::nv_tensor_ir {
namespace {

constexpr int32_t kFirstArchConditionalCc = 90;
constexpr int32_t kFirstFamilyPortableCc = 100;

bool isFamilyTargetCompatible(ComputeCapability cc,
                              ComputeCapability targetComputeCapability) {
  // Family-specific targets are only defined from compute capability 10.0.
  const int32_t ccVersion = toCcInt(cc);
  const int32_t targetCc = toCcInt(targetComputeCapability);
  if (ccVersion < 100 || targetCc < 100) {
    return false;
  }

  if (isSm100fFamily(cc)) {
    return isSm100fFamily(targetComputeCapability);
  }
  if (isSm120fFamily(cc)) {
    return isSm120fFamily(targetComputeCapability);
  }
  // Unknown family: require exact match to avoid false positives.
  return targetComputeCapability == cc;
}

bool isPortabilitySupported(ComputeCapability cc, ArchPortability portability) {
  const int32_t ccVersion = toCcInt(cc);
  switch (portability) {
  case ArchPortability::portable:
    return true;
  case ArchPortability::family_portable:
    return ccVersion >= kFirstFamilyPortableCc;
  case ArchPortability::arch_conditional:
    return ccVersion >= kFirstArchConditionalCc;
  }
  llvm_unreachable("unrecognized arch portability");
}

llvm::StringRef toComputeCapabilitySuffix(ArchPortability portability) {
  switch (portability) {
  case ArchPortability::portable:
    return "";
  case ArchPortability::family_portable:
    return "f";
  case ArchPortability::arch_conditional:
    return "a";
  }
  llvm_unreachable("unrecognized arch portability");
}

} // namespace

std::optional<ComputeCapability> symbolizeComputeCapability(int32_t cc) {
  if (cc < 0) {
    return std::nullopt;
  }
  return symbolizeComputeCapability(static_cast<uint32_t>(cc));
}

ArchPortability clampPortability(ComputeCapability cc,
                                 ArchPortability preferred) {
  if (isPortabilitySupported(cc, preferred)) {
    return preferred;
  }

  switch (preferred) {
  case ArchPortability::arch_conditional:
    if (isPortabilitySupported(cc, ArchPortability::family_portable)) {
      return ArchPortability::family_portable;
    }
    return ArchPortability::portable;
  case ArchPortability::family_portable:
    return ArchPortability::portable;
  case ArchPortability::portable:
    llvm_unreachable("portable targets are supported on every architecture");
  }
  llvm_unreachable("unrecognized arch portability");
}

FailureOr<SmTarget> SmTarget::fromCc(int32_t cc, ArchPortability portability) {
  auto computeCapability = symbolizeComputeCapability(cc);
  if (!computeCapability) {
    return failure();
  }
  return SmTarget{*computeCapability,
                  clampPortability(*computeCapability, portability)};
}

int32_t SmTarget::getComputeCapabilityVersion() const {
  return toCcInt(computeCapability);
}

std::string SmTarget::toString() const {
  return stringifyComputeCapability(computeCapability).str() +
         toComputeCapabilitySuffix(portability).str();
}

bool SmTarget::validateCodegenTargetCompatibility(int32_t targetCc) const {
  const int32_t cc = getComputeCapabilityVersion();
  switch (portability) {
  case ArchPortability::portable:
    // Baseline feature-set target can run on same or newer architectures.
    if (targetCc < cc) {
      return false;
    }
    break;

  case ArchPortability::family_portable:
    // Family-specific target:
    //   - is only defined for cc >= 10.0
    //   - inherits baseline forward-compatibility (same or newer)
    //   - must stay within the same family
    if (cc < 100 || targetCc < cc) {
      return false;
    }
    if (auto targetComputeCapability = symbolizeComputeCapability(targetCc);
        !targetComputeCapability ||
        !isFamilyTargetCompatible(computeCapability,
                                  *targetComputeCapability)) {
      return false;
    }
    break;

  case ArchPortability::arch_conditional:
    // Architecture-specific target:
    //   - is only defined for cc >= 9.0
    //   - must match exactly
    if (cc < 90 || targetCc != cc) {
      return false;
    }
    break;
  }
  return true;
}

FailureOr<SmTarget> SmTarget::fromString(llvm::StringRef chip) {
  constexpr llvm::StringLiteral kChipPrefix = "sm_";
  if (!chip.starts_with(kChipPrefix)) {
    return failure();
  }

  llvm::StringRef archStr = chip;
  if (archStr.size() <= kChipPrefix.size()) {
    return failure();
  }

  ArchPortability portability = ArchPortability::portable;
  if (archStr.ends_with("a")) {
    portability = ArchPortability::arch_conditional;
    archStr = archStr.drop_back();
  } else if (archStr.ends_with("f")) {
    portability = ArchPortability::family_portable;
    archStr = archStr.drop_back();
  }

  if (archStr.size() <= kChipPrefix.size()) {
    return failure();
  }

  auto computeCapability = symbolizeComputeCapability(archStr);
  if (!computeCapability) {
    return failure();
  }

  if (!isPortabilitySupported(*computeCapability, portability)) {
    return failure();
  }
  return SmTarget{*computeCapability, portability};
}

} // namespace mlir::nv_tensor_ir
