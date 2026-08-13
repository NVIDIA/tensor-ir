// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_INCLUDE_UTILS_COMPUTECAPABILITY_H_
#define TENSOR_IR_INCLUDE_UTILS_COMPUTECAPABILITY_H_

#include "tensor_ir/Dialect/TensorIRAttrs.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::nv_tensor_ir {

/// Returns the integer SM version represented by \p cc.
[[nodiscard]] inline int32_t toCcInt(ComputeCapability cc) {
  return static_cast<int32_t>(cc);
}

/// Returns the recognized compute capability for \p cc, or std::nullopt.
[[nodiscard]] std::optional<ComputeCapability>
symbolizeComputeCapability(int32_t cc);

/// Returns the most specific portability supported by \p cc that is no more
/// specific than \p preferred.
///
/// Portability specificity increases in the order portable, family portable,
/// and architecture conditional. Lowering the specificity only widens the set
/// of devices on which the generated code can run.
[[nodiscard]] ArchPortability clampPortability(ComputeCapability cc,
                                               ArchPortability preferred);

/// Canonical GPU target combining a compute capability and portability mode.
struct SmTarget {
  /// Returns the architecture-conditional SM100 target.
  [[nodiscard]] static SmTarget sm100a() {
    return SmTarget{ComputeCapability::Sm100,
                    ArchPortability::arch_conditional};
  }

  /// Creates a target for \p cc, lowering \p portability to the most specific
  /// supported portability when necessary.
  [[nodiscard]] static FailureOr<SmTarget>
  fromCc(int32_t cc,
         ArchPortability portability = ArchPortability::arch_conditional);

  [[nodiscard]] ComputeCapability getComputeCapability() const {
    return computeCapability;
  }
  [[nodiscard]] ArchPortability getPortability() const { return portability; }
  [[nodiscard]] int32_t getComputeCapabilityVersion() const;
  [[nodiscard]] std::string toString() const;

  /// Validates whether a concrete codegen target is compatible with this
  /// compute capability + portability pair.
  [[nodiscard]] bool validateCodegenTargetCompatibility(int32_t targetCc) const;

  /// Parses a target string strictly, rejecting unsupported explicit suffixes.
  [[nodiscard]] static FailureOr<SmTarget> fromString(llvm::StringRef chip);

private:
  ComputeCapability computeCapability;
  ArchPortability portability;

  SmTarget(ComputeCapability computeCapability, ArchPortability portability)
      : computeCapability(computeCapability), portability(portability) {}
};

/// Returns true if \p cc belongs to the sm_100f family.
inline bool isSm100fFamily(ComputeCapability cc) {
  bool isFamily =
      cc == ComputeCapability::Sm100 || cc == ComputeCapability::Sm103;
  return isFamily;
}

/// Returns true if \p cc belongs to the sm_120f family.
inline bool isSm120fFamily(ComputeCapability cc) {
  bool isFamily =
      cc == ComputeCapability::Sm120 || cc == ComputeCapability::Sm121;
  return isFamily;
}

/// Returns true if \p cc belongs to the Blackwell family.
inline bool isBlackwellFamily(ComputeCapability cc) {
  bool isFamily =
      cc == ComputeCapability::Sm100 || cc == ComputeCapability::Sm103 ||
      cc == ComputeCapability::Sm110 || cc == ComputeCapability::Sm120 ||
      cc == ComputeCapability::Sm121;
  return isFamily;
}

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_INCLUDE_UTILS_COMPUTECAPABILITY_H_
