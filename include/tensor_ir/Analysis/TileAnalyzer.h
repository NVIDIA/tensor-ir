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

/**
 * @class GpuArchitecture
 * @brief Basic GPU parameters needed for tiling decisions
 */
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

  /// @param warp_size Size of a warp
  /// @param cache_line L1 cache line size
  /// @param compute_capability Compute capability, e.g. Sm90/Sm100/Sm120
  /// @param num_sms Runtime SM count. 0 = use per-architecture fallback.
  GpuArchitecture(
      DimSize warp_size, DimSize cache_line,
      ComputeCapability compute_capability = kDefaultComputeCapability,
      int64_t num_sms = 0)
      : warpSize(warp_size), l1CacheLineSize(cache_line),
        computeCapability(compute_capability),
        numSMs(num_sms > 0
                   ? num_sms
                   : defaultSmsForComputeCapability(compute_capability)) {}

  DimSize warpSize;                    ///< Size of a warp
  DimSize l1CacheLineSize;             ///< L1 cache line size
  ComputeCapability computeCapability; ///< Compute capability.
  int64_t numSMs;                      ///< Number of SMs on the target GPU

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

/**
 * @class TileAnalyzer
 * @brief Determines tile sizes for pointwise operations
 *
 * This class implements a straightforward algorithm to determine tile sizes
 * for tensor operations on GPUs, focusing on enabling efficient execution
 * while letting the compiler handle resource allocation.
 */
class TileAnalyzer {
public:
  /**
   * @brief Construct a new TileAnalyzer object
   * @param gpu GPU architecture parameters
   */
  TileAnalyzer(const GpuArchitecture &gpu);

  /**
   * @brief Analyze and determine optimal tile size
   * @param tensor_shape Shape of the tensor
   * @param tensor_strides Strides of the tensor
   * @param element_size_bytes Size of each element in bytes
   * @param reduction_dims Reduction dimensions
   * @return Tile sizes for each dimension, or failure if the tiling heuristic
   * could not produce a tile size that is valid for @p tensor_shape.
   */
  FailureOr<llvm::SmallVector<DimSize>>
  analyzeTileSize(llvm::ArrayRef<DimSize> tensor_shape,
                  llvm::ArrayRef<DimSize> tensor_strides,
                  DimSize element_size_bytes,
                  llvm::ArrayRef<size_t> reduction_dims = {});

  /**
   * @brief Get explanation of tile size calculation
   * @return Explanation string, or std::nullopt if tile size analysis has not
   * succeeded yet.
   */
  std::optional<std::string> getTileExplanation() const;

  /**
   * @brief Check if tile size has been calculated
   * @return True if calculated
   */
  bool isTileSizeCalculated() const;

  /**
   * @brief Check if a tile size is valid
   *
   * Each dimension of a valid tile size must be a power-of-two >= 1 and <=
   * PowerOf2Ceil(tensor_dim) where tensor_dim is the corresponding dimension of
   * the tensor.
   *
   * The tile size is also reported invalid (returns false) when @p tensor_shape
   * is itself degenerate: a rank mismatch with @p tile_size or a non-positive
   * dimension. The TensorType verifier does not forbid such shapes, so they can
   * originate from user input; this is a recoverable "invalid" answer rather
   * than a hard error.
   *
   * @param tile_size The tile size to check
   * @param tensor_shape The shape of the tensor
   * @param element_size_bytes The size of each element in bytes
   * @return true If the tile size is valid
   * @return false If the tile size is invalid
   */
  bool isValidTileSize(llvm::ArrayRef<DimSize> tile_size,
                       llvm::ArrayRef<DimSize> tensor_shape,
                       DimSize element_size_bytes) const;

private:
  GpuArchitecture gpu_;
  std::optional<std::string> tileExplanation_;
  std::optional<llvm::SmallVector<DimSize>> cachedTileSize_;
};

/**
 * @brief Analyzes and determines the optimal tile size for a given graph
 *
 * @param tile_analyzer The tile analyzer to use
 * @param graph_op The TensorIR GraphOp
 * @return The optimal tile size, or failure if the graph signature cannot be
 * analyzed (non-tensor argument, no argument/result, or a non-static shape) or
 * if no valid tile size could be determined. A diagnostic is emitted on the
 * graph for every failure case.
 */
FailureOr<llvm::SmallVector<DimSize>>
analyzeTileSizeForGraph(TileAnalyzer &tile_analyzer,
                        mlir::nv_tensor_ir::GraphOp graph_op);

/**
 * @brief Determines the needed grid size based on the tile size for a graph
 *
 * The grid size defines how many tiles have to be calculated in each
 * dimension.
 *
 * @param tile_analyzer The tile analyzer to use
 * @param tile_size The tile size
 * @param graph_op The TensorIR GraphOp
 * @return The grid size, or failure if the graph signature cannot be analyzed
 * (non-tensor argument, no argument/result, or a non-static shape) or if @p
 * tile_size is invalid for the graph's tensor shape. A diagnostic is emitted on
 * the graph for every failure case.
 */

FailureOr<llvm::SmallVector<DimSize>>
calculateGridSizeForGraph(const TileAnalyzer &tileAnalyzer,
                          llvm::ArrayRef<DimSize> tileSize,
                          mlir::nv_tensor_ir::GraphOp graphOp);

/**
 * @brief Calculates the one-dimensional grid for a persistent kernel
 *
 * For static persistent kernels, the grid size is determined by the hardware
 * configuration rather than the problem size. Specifically, we launch `smCount
 * * occupancy` CTAs that persist across the entire computation, with each CTA
 * processing multiple tiles in a loop.
 *
 * @param tileAnalyzer The tile analyzer to use
 * @param tileSize The tile size
 * @param graphOp The TensorIR GraphOp
 * @param persistentCtaCount Positive maximum number of persistent CTAs to
 * launch
 * @param totalTiles Optional output parameter - returns the total number of
 * tiles to process. For static persistent kernels, the grid size may be smaller
 * than totalTiles (grid = smCount * occupancy, but totalTiles = full problem
 * size / tileSize)
 * @return The persistent grid size, clamped to the number of work tiles, or
 * failure (with a diagnostic emitted on the graph) if the base grid-size
 * computation fails.
 */
FailureOr<llvm::SmallVector<DimSize>> calculatePersistentGridSizeForGraph(
    const TileAnalyzer &tileAnalyzer, llvm::ArrayRef<DimSize> tileSize,
    mlir::nv_tensor_ir::GraphOp graphOp, int64_t persistentCtaCount,
    int64_t *totalTiles = nullptr);

/**
 * @brief Determines whether grid computation uses output tensor shape
 * instead of input tensor shape.
 *
 * This function encapsulates the logic shared between compile-time analysis
 * (calculateGridSizeForGraph) and runtime grid computation (KernelArgLayout).
 * Both must use the same logic to ensure alignment.
 *
 * Returns true if output tensor shape is used (for BroadcastOp or MatmulOp),
 * false otherwise (default to input tensor shape).
 *
 * @param graphOp The GraphOp to analyze
 * @return true if output tensor shape is used for grid computation
 */
bool useOutputTensorShapeForGrid(mlir::nv_tensor_ir::GraphOp graphOp);

} // namespace nv_tensor_ir
} // namespace mlir

#endif // TENSOR_IR_ANALYSIS_TILEANALYZER_H
