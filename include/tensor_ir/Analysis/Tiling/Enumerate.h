// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_ANALYSIS_TILING_ENUMERATE_H_
#define TENSOR_IR_ANALYSIS_TILING_ENUMERATE_H_

#include "tensor_ir/Analysis/Tiling/Evaluate.h"

#include "mlir/Support/LLVM.h"

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {

/**
  * @brief Get the tiling configurations in the evaluator scope.
  * @param evaluator The tiling config evaluator.
  * @param maxResults The maximum number of results to return.
  */
SmallVector<TilingConfig> enumerateTilings(TilingEvaluator &evaluator,
                                           size_t maxResults = 1000);

} // namespace tiling_analysis
} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_ANALYSIS_TILING_ENUMERATE_H_
