// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TILE_CANDIDATE_GENERATOR_H
#define TILE_CANDIDATE_GENERATOR_H

#include "tensor_ir/Analysis/TileAnalyzer.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace nv_tensor_ir {

/// @brief Generates multiple tile shape candidates using a unified
/// constraint-based algorithm.
///
/// The algorithm:
///   1. Apply constraints (reduction dims → full size, op constraints)
///   2. Identify fastest free dimension (from strides across all operands)
///   3. Small tensor fast path (< 100K elements → single tile)
///   4. Generate coalescing options for fastest dim
///   5. Budget × coalescing cross-product with greedy distribution
///   6. Validate, dedup, score, return top-K
///
/// Scoring evaluates per-operand coalescing when multiple stride arrays are
/// provided (composite iteration space with inputs + output).
class TileCandidateGenerator {
public:
  /// Internal candidate representation with score and rationale.
  struct Candidate {
    llvm::SmallVector<int32_t> tileShape;
    float score = 0.0f;    ///< Higher is better. 0.0 = unscored.
    std::string rationale; ///< Human-readable generation rationale.
  };

  /// @brief Construct a new TileCandidateGenerator
  /// @param gpu GPU architecture parameters (warpSize, l1CacheLineSize, numSMs)
  /// @param maxCandidates Maximum number of candidates to return
  TileCandidateGenerator(const GpuArchitecture &gpu, int maxCandidates);

  /// @brief Generate tile candidates with per-operand stride information.
  ///
  /// @param shape Iteration space shape (shared by all operands)
  /// @param allStrides Per-operand strides (one per input/output tensor)
  /// @param allElementSizeBytes Per-operand element sizes in bytes
  /// @param constraints Op-specific constraints: {dim_index →
  /// required_tile_value}.
  /// @return Candidates sorted by score (highest first = best). At most
  ///         maxCandidates returned.
  llvm::SmallVector<Candidate>
  generateCandidates(llvm::ArrayRef<DimSize> shape,
                     llvm::ArrayRef<llvm::SmallVector<DimSize>> allStrides,
                     llvm::ArrayRef<DimSize> allElementSizeBytes,
                     const llvm::DenseMap<size_t, DimSize> &constraints);

  /// @brief Convenience overload: single stride array (backward compatible).
  llvm::SmallVector<Candidate>
  generateCandidates(llvm::ArrayRef<DimSize> shape,
                     llvm::ArrayRef<DimSize> strides, DimSize elementSizeBytes,
                     const llvm::DenseMap<size_t, DimSize> &constraints);

  /// @brief Overload without constraints (uses empty constraints map).
  llvm::SmallVector<Candidate>
  generateCandidates(llvm::ArrayRef<DimSize> shape,
                     llvm::ArrayRef<DimSize> strides, DimSize elementSizeBytes);

  /// @brief Multi-stride overload without constraints.
  llvm::SmallVector<Candidate>
  generateCandidates(llvm::ArrayRef<DimSize> shape,
                     llvm::ArrayRef<llvm::SmallVector<DimSize>> allStrides,
                     llvm::ArrayRef<DimSize> allElementSizeBytes);

  /// @brief Score a single tile candidate using per-operand coalescing.
  ///
  /// @param tileShape The tile shape to score
  /// @param tensorShape The full tensor shape
  /// @param fastestDim Index of the fastest-varying dimension
  /// @param allStrides Per-operand strides
  /// @param allElementSizeBytes Per-operand element sizes
  /// @return Score (higher = better)
  virtual float scoreTile(llvm::ArrayRef<int32_t> tileShape,
                          llvm::ArrayRef<DimSize> tensorShape,
                          size_t fastestDim,
                          llvm::ArrayRef<llvm::SmallVector<DimSize>> allStrides,
                          llvm::ArrayRef<DimSize> allElementSizeBytes);

  /// @brief Backward-compatible single-operand scoring.
  virtual float scoreTile(llvm::ArrayRef<int32_t> tileShape,
                          llvm::ArrayRef<DimSize> tensorShape,
                          size_t fastestDim, DimSize elementSizeBytes);

  /// Check that each tile dim is a power-of-2, >= 1, and <=
  /// PowerOf2Ceil(shape).
  bool isValidTile(llvm::ArrayRef<int32_t> tile,
                   llvm::ArrayRef<DimSize> shape) const;

  virtual ~TileCandidateGenerator() = default;

private:
  /// Distribute remaining element budget across free dims,
  /// fastest-stride-first.
  void distributeBudget(llvm::SmallVectorImpl<int32_t> &tile, int64_t remaining,
                        llvm::ArrayRef<size_t> freeDims,
                        llvm::ArrayRef<DimSize> shape);

  /// Generate smem-aware tile candidates for fusions with 2+ active dims.
  /// These are balanced tiles sized to fit in shared memory, covering the
  /// case where the compiler will use reg→smem→reg layout conversion.
  void generateSmemCandidates(
      llvm::ArrayRef<size_t> activeDims, llvm::ArrayRef<DimSize> shape,
      llvm::ArrayRef<int64_t> shapeLimit,
      const llvm::DenseMap<size_t, DimSize> &constraints, DimSize maxElemBytes,
      size_t ndim, llvm::SmallVectorImpl<Candidate> &allCandidates);

  /// Compute the per-operand contiguous-span cache-line utilization proxy.
  /// Returns average bandwidth utilization (0..1) across all memory operands.
  static float computeCoalescingEfficiency(
      llvm::ArrayRef<int32_t> tileShape, size_t fastestDim,
      llvm::ArrayRef<llvm::SmallVector<DimSize>> allStrides,
      llvm::ArrayRef<DimSize> allElementSizeBytes, DimSize cacheLineSize);

  GpuArchitecture gpu_;
  int maxCandidates_;
};

} // namespace nv_tensor_ir
} // namespace mlir

#endif // TILE_CANDIDATE_GENERATOR_H
