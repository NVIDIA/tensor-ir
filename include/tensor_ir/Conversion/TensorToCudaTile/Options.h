// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_CONVERSION_TENSORTOCUDATILE_OPTIONS_H_
#define TENSOR_IR_CONVERSION_TENSORTOCUDATILE_OPTIONS_H_

#include "llvm/ADT/SmallVector.h"

namespace mlir::nv_tensor_ir {

inline constexpr int64_t kDefaultReductionTileSize = 128;

/// Kernel persistence mode controls the lifetime and work distribution of CTAs.
/// Note: if Static persistence mode is used and grid size >= total tiles, the
/// tile compiler will fall back to None mode as persistence provides no
/// benefit.
enum class PersistenceMode {
  /// Normal kernel launch - one CTA per tile, CTAs terminate after processing.
  /// Grid size equals the number of tiles. Standard approach with highest
  /// compatibility but may have launch overhead for many small tiles.
  None = 0,

  /// Static persistent kernel (Ampere GA100+) - launches a fixed grid of
  /// (smCount * occupancy) CTAs that loop over all tiles. Each CTA processes
  /// multiple tiles in a software-managed loop without terminating.
  /// Benefits: Reduces kernel launch overhead, enables dynamic load balancing,
  /// better for workloads with many tiles or irregular tile sizes.
  /// Grid size = smCount * occupancy (independent of problem size).
  Static = 1,

};

/// Codegen strategy for the TensorIR to CudaTile conversion.
enum class CudaTileCodegenStrategy {
  /// Derive iteration space maps from affine analysis.
  AffineMap = 0,

  /// Derive memory access patterns from layout propagation annotations.
  LayoutPropagation = 1
};

/// Knobs consumed by the TensorIR-to-CudaTile analysis and conversion
/// pipeline builders (`buildGraphAnalysisPipeline`,
/// `buildTileSelectionPipeline`, `buildTensorToCudaTileConversion*`) and by
/// the affine-map conversion state factory.
struct TensorToCudaTilePipelineOptions {
  CudaTileCodegenStrategy codegenStrategy =
      CudaTileCodegenStrategy::LayoutPropagation;

  /// Tile size for the generated kernel.
  llvm::SmallVector<int32_t> tileSize;

  /// Tile size for contracting dimensions in layout-propagation lowering.
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

  /// Kernel persistence mode.
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
