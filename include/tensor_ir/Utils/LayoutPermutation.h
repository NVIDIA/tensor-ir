// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_UTILS_LAYOUTPERMUTATION_H
#define TENSOR_IR_UTILS_LAYOUTPERMUTATION_H

#include "tensor_ir/Dialect/TensorIRAttrs.h"

#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::nv_tensor_ir {

/// Find a permutation such that source.transpose(permutation) == target.
/// Result dimension i corresponds to source dimension permutation[i].
///
/// matchDimensions optionally imposes additional equivalence constraints on
/// (source dimension, target dimension), such as equal tile extents and
/// coordinates. It must test equality of per-dimension properties, so candidates
/// with equal properties remain interchangeable during greedy matching.
///
/// Dimensions are matched by their contributions to every source layout,
/// including runtime dimension identities. Equal contributions are
/// interchangeable, so matching takes O(rank^2) comparisons without searching
/// permutations. The complete transposed attribute is checked before returning.
/// Unsupported layout kinds conservatively return failure.
FailureOr<SmallVector<int64_t>> findLayoutPermutation(
    LayoutSourceAttrInterface source, LayoutSourceAttrInterface target,
    llvm::function_ref<bool(unsigned, unsigned)> matchDimensions = nullptr);

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_UTILS_LAYOUTPERMUTATION_H
