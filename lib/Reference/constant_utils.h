// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_LIB_REFERENCE_CONSTANT_UTILS_H
#define TENSOR_IR_LIB_REFERENCE_CONSTANT_UTILS_H

#include "tensor_ir/Support/Status.h"

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

namespace mlir::nv_tensor_ir::reference::detail {

/// Convert a finite, representable constant to an integer by truncating toward
/// zero. Values outside the destination type's range are rejected before the
/// floating-point-to-integer conversion.
template <typename T>
StatusOr<T> convertFiniteConstantToInteger(double value, const char *typeName) {
  static_assert(std::is_integral_v<T>, "destination type must be integral");

  if (!std::isfinite(value)) {
    return Status::InvalidArgument(
        std::string("Constant value must be finite for ") + typeName);
  }

  const double lowest = static_cast<double>(std::numeric_limits<T>::lowest());
  const double highest = static_cast<double>(std::numeric_limits<T>::max());
  if (value < lowest || value > highest) {
    return Status::InvalidArgument(
        std::string("Constant value out of range for ") + typeName);
  }

  return static_cast<T>(std::trunc(value));
}

} // namespace mlir::nv_tensor_ir::reference::detail

#endif // TENSOR_IR_LIB_REFERENCE_CONSTANT_UTILS_H
