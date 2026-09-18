// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_CONVERSION_TENSORTOCUDATILE_OPTIONS_H_
#define TENSOR_IR_CONVERSION_TENSORTOCUDATILE_OPTIONS_H_

#include "tensor_ir/Options/OptionsEnums.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::nv_tensor_ir {

inline constexpr int64_t kDefaultReductionTileSize = 128;

/// Knobs consumed by the TensorIR-to-CudaTile analysis and conversion
/// pipeline builders (`buildGraphAnalysisPipeline`,
/// `buildTileSelectionPipeline`, `buildTensorToCudaTileConversion*`) and by
/// the affine-map conversion state factory.
struct TensorToCudaTilePipelineOptions {
  CudaTileCodegenStrategy codegenStrategy =
      CudaTileCodegenStrategy::LayoutPropagation;

  /// Tile size used by layout formation or affine-map conversion.
  llvm::SmallVector<int32_t> tileSize;

  /// Tile size for contracting dimensions during tiled-program formation.
  int64_t reductionTileSize = kDefaultReductionTileSize;

  /// Number of CTAs per kernel.
  int32_t numCTAs = 1;

  /// Occupancy hint (number of CTAs per SM).
  int32_t occupancy = 1;

  /// Number of warps per CTA.
  int32_t numWarps = 4;

  /// Runtime SM count for persistent kernels.
  int32_t smCount = 0;

  /// Numeric compute capability for tile analysis (e.g. 100 for sm_100).
  int32_t computeCapability = 100;

  /// Kernel persistence mode used by formation or affine-map conversion.
  PersistenceMode persistence = PersistenceMode::None;

  /// Use the "uniform signature" ABI for the generated kernel.
  bool uniformSignature = false;

  /// Maximum tile candidates to keep during layout-propagation analysis.
  /// Ignored by the affine-map path. Set to 0 to skip TileAnalyzer when an
  /// explicit `tileSize` is already provided.
  int32_t maxCandidates = 1;
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_CONVERSION_TENSORTOCUDATILE_OPTIONS_H_
