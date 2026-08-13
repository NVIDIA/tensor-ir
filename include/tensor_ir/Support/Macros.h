// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file contains preprocessor utilities shared across multiple TensorIR libraries.
// that may be used in exported header files. To lessen the liklihood of name colissions downstream,
// each macro is prefixed with TIR_.

#ifndef TENSOR_IR_SUPPORT_MACROS_H
#define TENSOR_IR_SUPPORT_MACROS_H

/// Pastes `left` and `right` into a single token after both have been macro
/// expanded. The extra indirection through the `_IMPL` form is what forces
/// that expansion, so `TIR_CONCAT(x, __COUNTER__)` yields `x0` rather
/// than `x__COUNTER__`. Used to build unique identifiers in macros that
/// declare a temporary.
#define TIR_CONCAT(left, right) TIR_CONCAT_IMPL(left, right)
#define TIR_CONCAT_IMPL(left, right) left##right

#endif // TENSOR_IR_SUPPORT_MACROS_H
