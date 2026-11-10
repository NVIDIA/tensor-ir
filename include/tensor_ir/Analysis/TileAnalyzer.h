// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_ANALYSIS_TILEANALYZER_H
#define TENSOR_IR_ANALYSIS_TILEANALYZER_H

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Utils/ComputeCapability.h"
#include "tensor_ir/Utils/Utils.h"

#include <optional>
#include <string>

namespace mlir {
namespace nv_tensor_ir {

/// Basic GPU parameters needed for tiling decisions
class GpuArchitecture {
public:
  /// Default warp size for NVIDIA GPUs
  static constexpr DimSize kDefaultWarpSize = 32;
  /// Default L1 cache line size in bytes
  static constexpr DimSize kDefaultL1CacheLineSize = 128;
  /// Default compute capability used for architecture-aware heuristics
  static constexpr ComputeCapability kDefaultComputeCapability =
      ComputeCapability::Sm100;

  GpuArchitecture()
      : warpSize(kDefaultWarpSize), l1CacheLineSize(kDefaultL1CacheLineSize),
        computeCapability(kDefaultComputeCapability),
        numSMs(defaultSmsForComputeCapability(kDefaultComputeCapability)) {}

  /// \param compute_capability Compute capability, e.g. Sm90/Sm100/Sm120
  /// \param num_sms Runtime SM count. 0 = use per-architecture fallback.
  GpuArchitecture(
      DimSize warp_size, DimSize cache_line,
      ComputeCapability compute_capability = kDefaultComputeCapability,
      int64_t num_sms = 0)
      : warpSize(warp_size), l1CacheLineSize(cache_line),
        computeCapability(compute_capability),
        numSMs(num_sms > 0
                   ? num_sms
                   : defaultSmsForComputeCapability(compute_capability)) {}

  /// Size of a warp
  DimSize warpSize;
  /// L1 cache line size
  DimSize l1CacheLineSize;
  /// Compute capability.
  ComputeCapability computeCapability;
  /// Number of SMs on the target GPU
  int64_t numSMs;

private:
  /// Conservative SM count fallback when runtime value is unavailable.
  /// SM count varies by SKU within an architecture; these are representative
  /// values for the flagship chip of each generation.
  static int64_t defaultSmsForComputeCapability(ComputeCapability cc) {
    const int ccVersion = toCcInt(cc);
    if (ccVersion >= 120) {
      return 148;
    }
    if (ccVersion >= 100) {
      return 132;
    }
    if (ccVersion >= 90) {
      return 132;
    }
    if (ccVersion >= 80) {
      return 108;
    }
    if (ccVersion >= 75) {
      return 80;
    }
    return 64;
  }
};

/// Determines tile sizes for pointwise operations
///
/// This class implements a straightforward algorithm to determine tile sizes
/// for tensor operations on GPUs, focusing on enabling efficient execution
/// while letting the compiler handle resource allocation.
class TileAnalyzer {
public:
  /// Construct a new TileAnalyzer object
  TileAnalyzer(const GpuArchitecture &gpu);

  /// Analyze and determine optimal tile size
  ///
  /// \returns Tile sizes for each dimension, or failure if the tiling heuristic
  /// could not produce a tile size that is valid for \p tensor_shape.
  FailureOr<llvm::SmallVector<DimSize>>
  analyzeTileSize(llvm::ArrayRef<DimSize> tensor_shape,
                  llvm::ArrayRef<DimSize> tensor_strides,
                  DimSize element_size_bytes,
                  llvm::ArrayRef<size_t> reduction_dims = {});

  /// Get explanation of tile size calculation
  ///
  /// \returns Explanation string, or std::nullopt if tile size analysis has not
  /// succeeded yet.
  std::optional<std::string> getTileExplanation() const;

  /// Check if tile size has been calculated
  bool isTileSizeCalculated() const;

  /// Check if a tile size is valid
  ///
  /// Each dimension of a valid tile size must be a power-of-two >= 1 and <=
  /// PowerOf2Ceil(tensor_dim) where tensor_dim is the corresponding dimension of
  /// the tensor.
  ///
  /// The tile size is also reported invalid (returns false) when \p tensor_shape
  /// is itself degenerate: a rank mismatch with \p tile_size or a non-positive
  /// dimension. The RankedTensorType verifier does not forbid such shapes, so they can
  /// originate from user input; this is a recoverable "invalid" answer rather
  /// than a hard error.
  bool isValidTileSize(llvm::ArrayRef<DimSize> tile_size,
                       llvm::ArrayRef<DimSize> tensor_shape,
                       DimSize element_size_bytes) const;

private:
  GpuArchitecture gpu_;
  std::optional<std::string> tileExplanation_;
  std::optional<llvm::SmallVector<DimSize>> cachedTileSize_;
};

/// Analyzes and determines the optimal tile size for a given graph
///
/// \returns The optimal tile size, or failure if the graph signature cannot be
/// analyzed (non-tensor argument, no argument/result, or a non-static shape) or
/// if no valid tile size could be determined. A diagnostic is emitted on the
/// graph for every failure case.
FailureOr<llvm::SmallVector<DimSize>>
analyzeTileSizeForGraph(TileAnalyzer &tile_analyzer,
                        mlir::nv_tensor_ir::GraphOp graph_op);

/// Determines the needed grid size based on the tile size for a graph
///
/// The grid size defines how many tiles have to be calculated in each
/// dimension.
///
/// \returns The grid size, or failure if the graph signature cannot be analyzed
/// (non-tensor argument, no argument/result, or a non-static shape) or if \p
/// tileSize is invalid for the graph's tensor shape. A diagnostic is emitted on
/// the graph for every failure case.

FailureOr<llvm::SmallVector<DimSize>>
calculateGridSizeForGraph(const TileAnalyzer &tileAnalyzer,
                          llvm::ArrayRef<DimSize> tileSize,
                          mlir::nv_tensor_ir::GraphOp graphOp);

/// Calculates the one-dimensional grid for a persistent kernel
///
/// For static persistent kernels, the grid size is determined by the hardware
/// configuration rather than the problem size. Specifically, we launch `smCount
/// * occupancy` CTAs that persist across the entire computation, with each CTA
/// processing multiple tiles in a loop.
///
/// \param persistentCtaCount Positive maximum number of persistent CTAs to
/// launch
/// \param [out] totalTiles Optional output parameter - returns the total number of
/// tiles to process. For static persistent kernels, the grid size may be smaller
/// than totalTiles (grid = smCount * occupancy, but totalTiles = full problem
/// size / tileSize)
///
/// \returns The persistent grid size, clamped to the number of work tiles, or
/// failure (with a diagnostic emitted on the graph) if the base grid-size
/// computation fails.
FailureOr<llvm::SmallVector<DimSize>> calculatePersistentGridSizeForGraph(
    const TileAnalyzer &tileAnalyzer, llvm::ArrayRef<DimSize> tileSize,
    mlir::nv_tensor_ir::GraphOp graphOp, int64_t persistentCtaCount,
    int64_t *totalTiles = nullptr);

/// Determines whether grid computation uses output tensor shape
/// instead of input tensor shape.
///
/// This function encapsulates the logic shared between compile-time analysis
/// (calculateGridSizeForGraph) and runtime grid computation (KernelMetadata).
/// Both must use the same logic to ensure alignment.
///
/// Returns true if output tensor shape is used (for BroadcastOp or MatmulOp),
/// false otherwise (default to input tensor shape).
bool useOutputTensorShapeForGrid(mlir::nv_tensor_ir::GraphOp graphOp);

} // namespace nv_tensor_ir
} // namespace mlir

#endif // TENSOR_IR_ANALYSIS_TILEANALYZER_H
