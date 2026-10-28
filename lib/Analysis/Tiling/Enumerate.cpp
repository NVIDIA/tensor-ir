// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/Tiling/Enumerate.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#define DEBUG_TYPE "tiling-analysis-enumerate"

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {

/// Minimum tile size (1 warp, 1 value per thread).
constexpr int kMinTileSizeBits = 5; // 2^5 = 32
/// Maximum tile size (8 warps, 128 values per thread).
constexpr int kMaxTileSizeBits = 15; // 2^15 = 32768

/// Minimum reduction tile size.
constexpr int kMinReductionTileSizeBits = 5; // 2^5 = 32
/// Maximum tile size for dynamic dimensions.
constexpr int kMaxDynamicTileSizeBits = 6; // 2^6 = 64

/// Tile size limits for matmul dimensions.
constexpr int kMinLhsTileSizeBits = 4;         // 2^4 = 16
constexpr int kMaxLhsTileSizeBits = 7;         // 2^7 = 128
constexpr int kMinRhsTileSizeBits = 3;         // 2^3 = 8
constexpr int kMaxRhsTileSizeBits = 8;         // 2^8 = 256
constexpr int kMinContractionTileSizeBits = 4; // 2^4 = 16
constexpr int kMaxContractionTileSizeBits = 8; // 2^8 = 256

/// ----- Dimension analysis ---------------------------------------------------
///
/// Before generating tile shapes, the enumerator collects the properties of
/// each dimension of the main iteration space, by walking the input and output
/// layout trees attached to the graph:
///
/// - Tensor sources count the tensors where the dimension is contiguous
///   (stride 1, the loads and stores can be vectorized), and where it is
///   broadcast (stride 0, the loaded values are reused across the tile).
/// - Concat sources mark their concatenation dimension as non-tileable, as the
///   evaluator requires a unit tile size for it.
/// - Matmul sources mark the dimensions that map to the M and N dimensions of
///   the matmul result, based on their position in the (batch, M, N, K)
///   iteration space. The matmul generator tiles these groups separately.
///
/// Reductions and matmuls start nested iteration spaces. The nested dimensions
/// are mapped back to the main dimensions the same way as in the evaluator, and
/// the reduced and contracting dimensions are mapped to a temporary value,
/// since their tile size is set by the reduction tile size instead.
///
/// The dimensions are then ordered by importance. The combination generators
/// assign the bits depth-first, starting from the largest values in the first
/// dimensions, so the order decides which dimensions get the largest tiles
/// when the budget is limited. Non-tileable dimensions go last, and the other
/// dimensions are ordered by the number of vectorizable and broadcast uses.

/**
 * @brief Information about a dimension.
 */
struct DimensionInfo {
  /// Dimension number in the main iteration space (-1 for reduction dims).
  int index;

  /// Number of tensors where this dimension can be vectorized (stride=1).
  int vectorizableCount = 0;

  /// Number of tensors where this dimension is broadcasted (stride=0).
  int broadcastedCount = 0;

  /// Marks dimensions that cannot be tiled (e.g. concatenation dims).
  bool nonTileable = false;

  /// Marks dimensions that propagate to a matmul result.
  bool usedInMatmulM = false;
  bool usedInMatmulN = false;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const DimensionInfo &info);
};

class DimensionLayoutIterator : public LayoutIterator {
public:
  explicit DimensionLayoutIterator(size_t rank);

  LogicalResult start(LayoutSourceAttrInterface layout, bool writes) {
    this->writes = writes;
    return dispatch(layout);
  }

  LogicalResult processTensor(TensorSourceAttr tensor) override;
  LogicalResult processConcat(ConcatSourceAttr concat) override;
  LogicalResult processReduction(ReductionSourceAttr reduction) override;
  LogicalResult processMatmul(MatmulSourceAttr matmul) override;

  // Returns the ordered dimensions (important first).
  SmallVector<DimensionInfo> getOrderedDimensions() const;

private:
  SmallVector<DimensionInfo> dimensions;
  SmallVector<DimensionInfo *> pointers;
  bool writes = false;
};

DimensionLayoutIterator::DimensionLayoutIterator(size_t rank) {
  dimensions.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    dimensions.push_back({static_cast<int>(i)});
    pointers.push_back(&dimensions.back());
  }
}

LogicalResult DimensionLayoutIterator::processTensor(TensorSourceAttr tensor) {
  // Mark vectorizable and broadcasted tensor dimensions.
  auto cuteLayout = tensor.getCuteLayout();
  for (auto [idx, stride] : llvm::enumerate(cuteLayout.stride().getValues())) {
    if (stride.isLeaf() && stride.isStatic()) {
      if (stride.as_int() == 1) {
        // Increased weights for write operations (read-modify-write).
        pointers[idx]->vectorizableCount += writes ? 2 : 1;
      } else if (stride.as_int() == 0) {
        pointers[idx]->broadcastedCount++;
      }
    }
  }
  return success();
}

LogicalResult DimensionLayoutIterator::processConcat(ConcatSourceAttr concat) {
  // Mark the concatenation dimension as non-tileable.
  pointers[concat.getDimension()]->nonTileable = true;
  return LayoutIterator::processConcat(concat);
}

LogicalResult
DimensionLayoutIterator::processReduction(ReductionSourceAttr reduction) {
  // Build the new pointers for the inner iteration space.
  SmallVector<DimensionInfo *> newPointers;
  auto cuteLayout = reduction.getCuteLayout();
  auto strideValues = cuteLayout.stride().getValues().drop_back();
  for (auto [idx, stride] : llvm::enumerate(strideValues)) {
    if (stride.as_int() > 0) {
      newPointers.push_back(pointers[idx]);
    }
  }
  DimensionInfo reductionDim{-1};
  newPointers.resize(newPointers.size() + reduction.getReductionShape().size(),
                     &reductionDim);

  // Add reduction dimensions (temporary) to the iteration space.
  std::swap(pointers, newPointers);
  LogicalResult result = LayoutIterator::processReduction(reduction);
  std::swap(pointers, newPointers);
  return result;
}

LogicalResult DimensionLayoutIterator::processMatmul(MatmulSourceAttr matmul) {
  size_t rank = pointers.size();
  DimensionInfo reductionDim{-1};
  SmallVector<DimensionInfo *> iterSpace = pointers;

  // Process LHS input.
  if (auto lhsDimensionMap = matmul.getLhsDimensionMap();
      succeeded(lhsDimensionMap)) {
    pointers.clear();
    for (size_t idx : *lhsDimensionMap) {
      pointers.push_back(idx < rank ? iterSpace[idx] : &reductionDim);
    }
    auto lhsResult = dispatch(matmul.getLhs());
    (void)lhsResult;
  }

  // Process RHS input.
  if (auto rhsDimensionMap = matmul.getRhsDimensionMap();
      succeeded(rhsDimensionMap)) {
    pointers.clear();
    for (size_t idx : *rhsDimensionMap) {
      pointers.push_back(idx < rank ? iterSpace[idx] : &reductionDim);
    }
    auto rhsResult = dispatch(matmul.getRhs());
    (void)rhsResult;
  }

  // Mark the dimensions used in the matmul.
  pointers = std::move(iterSpace);
  auto cuteLayout = matmul.getCuteLayout();
  auto strideValues = cuteLayout.stride().getValues().drop_back();
  for (auto [idx, stride] : llvm::enumerate(strideValues)) {
    if (stride.isStatic()) {
      int64_t pos = stride.as_int();
      if (pos >= matmul.getM() * matmul.getN() * matmul.getK()) {
        continue; // Skip batch dimension.
      }
      if (pos >= matmul.getN() * matmul.getK()) {
        pointers[idx]->usedInMatmulM = true;
      } else if (pos >= matmul.getK()) {
        pointers[idx]->usedInMatmulN = true;
      }
    }
  }
  return success();
}

SmallVector<DimensionInfo>
DimensionLayoutIterator::getOrderedDimensions() const {
  // Sort the dimensions in order of importance:
  // - Non-tileable dimensions go to the end.
  // - Vectorizable dimensions first.
  // - Broadcasted dimensions next.
  // - Otherwise, in the iteration space order.
  SmallVector<DimensionInfo> result = dimensions;
  llvm::sort(result, [](const DimensionInfo &a, const DimensionInfo &b) {
    if (a.nonTileable != b.nonTileable) {
      return b.nonTileable;
    }
    if (a.vectorizableCount != b.vectorizableCount) {
      return a.vectorizableCount > b.vectorizableCount;
    }
    if (a.broadcastedCount != b.broadcastedCount) {
      return a.broadcastedCount > b.broadcastedCount;
    }
    return a.index < b.index;
  });
  return result;
}

#ifndef NDEBUG
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const DimensionInfo &info) {
  os << "[" << info.index;
  if (info.vectorizableCount > 0) {
    os << " v" << info.vectorizableCount;
  }
  if (info.broadcastedCount > 0) {
    os << " b" << info.broadcastedCount;
  }
  if (info.nonTileable) {
    os << " nt";
  }
  os << "]";
  return os;
}
#endif

/// ----- Combination generator ------------------------------------------------
///
/// Tile sizes are powers of two, so a tile shape is represented by the number
/// of bits (the exponent) assigned to each dimension, called a slot here. The
/// sum of the bits is the log2 of the number of elements in the tile.
/// A combination is valid if each slot is within its limit (the dimension size,
/// rounded up to a power of two) and the sum is within the requested range.
///
/// The number of combinations grows quickly with the rank, so the generation is
/// bounded by a result budget:
///
/// - `countCombinations` counts the combinations without generating them, used
///   to check whether the whole search space fits the budget.
/// - `generateCombinationsExactCount` generates the combinations with a given
///   sum depth-first, starting from the largest values in the first slots.
/// - `generateCombinationsWithinRange` splits the budget evenly between the
///   sums in the range, starting from the largest or the smallest tiles.
/// - `generateCombinationsBalanced` first reduces the search space by lowering
///   the limits of the least important slots, when the whole space does not fit
///   the budget, so that the truncation does not favor the first slots only.

/**
 * @brief Counts the number of combinations of bits assigned to slots.
 * @param minBits The minimum number of bits to assign.
 * @param maxBits The maximum number of bits to assign.
 * @param limits The maximum number of bits for each slot.
 */
size_t countCombinations(int minBits, int maxBits, ArrayRef<int> limits) {
  // Use dynamic programming to avoid enumerating the combinations: for each
  // bit total, count the ways to reach it as slots are added one at a time.
  // The result is the sum of the counts for totals in [minBits, maxBits].
  SmallVector<size_t> accum(maxBits + 1, 0);
  accum[0] = 1;
  for (int limit : limits) {
    // accum[s] counts the combinations of the previous slots summing to s.
    // Add the current dimension by shifting each count by every value in
    // [0, limit], dropping sums above maxBits.
    SmallVector<size_t> next = accum;
    for (int add = std::min(maxBits, limit); add >= 1; --add) {
      for (int i = maxBits - add; i >= 0; --i) {
        next[i + add] += accum[i];
      }
    }
    accum = std::move(next);
  }
  return llvm::sum_of(ArrayRef(accum).drop_front(minBits));
}

/**
 * @brief Generates all combinations of exact number of bits assigned to slots.
 * @param bits The number of bits to assign.
 * @param limits The maximum number of bits for each slot.
 * @param maxResults The maximum number of results to generate.
 */
SmallVector<SmallVector<int>>
generateCombinationsExactCount(int bits, ArrayRef<int> limits,
                               size_t maxResults) {
  SmallVector<SmallVector<int>> result;
  SmallVector<int> current(limits.size(), 0);

  // Calculate suffix sum for the limits, needed for early stopping.
  // Example: for input [2,3,4], the suffix sum is [7,4,0].
  SmallVector<int> suffixSum(limits.size(), 0);
  for (int i = static_cast<int>(limits.size()) - 1; i > 0; --i) {
    suffixSum[i - 1] = suffixSum[i] + limits[i];
  }

  // Recursively assign bits to slots.
  std::function<void(size_t, int)> generate = [&](size_t idx, int bits) {
    // Stop when enough results are generated.
    if (result.size() == maxResults) {
      return;
    }
    // All bits are assigned, add the current combination.
    if (bits == 0) {
      result.push_back(current);
      return;
    }
    // Try all possible assignments for the current dimension.
    // Start from the higher value, as the dimensions are sorted.
    for (int i = std::min(bits, limits[idx]);
         i >= std::max(bits - suffixSum[idx], 0); --i) {
      current[idx] = i;
      generate(idx + 1, bits - i);
    }
    current[idx] = 0;
  };

  // Start from the first index.
  assert(!limits.empty());
  generate(0, bits);
  return result;
}

/**
 * @brief Generates all combinations of bits assigned to slots.
 * @param minBits The minimum number of bits to assign (must be >= 0).
 * @param maxBits The maximum number of bits to assign (must be >= minBits).
 * @param limits The maximum number of bits for each slot.
 * @param maxResults The maximum number of results to generate.
 * @param descending Whether to start from the upper bound.
 */
SmallVector<SmallVector<int>>
generateCombinationsWithinRange(int minBits, int maxBits, ArrayRef<int> limits,
                                size_t maxResults, bool descending) {
  SmallVector<SmallVector<int>> result;
  assert(minBits >= 0 && maxBits >= minBits);

  // Generate combinations for each number of bits in the range.
  for (int i = 0, n = maxBits - minBits + 1; i < n; ++i) {
    size_t maxResultsWeighted =
        llvm::divideCeil(maxResults - result.size(), n - i);
    auto combinations = generateCombinationsExactCount(
        descending ? maxBits - i : minBits + i, limits, maxResultsWeighted);
    result.append(combinations);
  }

  // If there are no results, then the limits are too low.
  if (result.empty() && maxResults > 0) {
    result.push_back(SmallVector<int>(limits));
  }
  return result;
}

/// Generates the combinations of bits assigned to slots within the budget. If
/// the search space is too large, it is first reduced by lowering the limits of
/// the least important dimensions, so that the budget covers the rest evenly.
/// The lowering stops once the search space fits the budget, or once the limits
/// sum to at most `maxBits`: lowering them further would make even the largest
/// combination smaller than the upper bound.
///
/// The generator assigns the bits depth-first, starting from the largest values
/// of the first slots. When the search space exceeds the budget, the output is
/// truncated and would only contain the combinations favoring the first slots.
/// In case there are more than two vectorizable dimensions, this would result
/// in missed optimization opportunities.
SmallVector<SmallVector<int>>
generateCombinationsBalanced(int minBits, int maxBits, ArrayRef<int> limits,
                             size_t maxResults, bool descending,
                             ArrayRef<DimensionInfo> dimensions) {
  // If the number of combinations is within the budget, then generate all
  // combinations (fast path).
  auto fitsBudget = [&](ArrayRef<int> limits) {
    return llvm::sum_of(limits) <= maxBits ||
           countCombinations(minBits, maxBits, limits) <= maxResults;
  };
  if (fitsBudget(limits)) {
    return generateCombinationsWithinRange(minBits, maxBits, limits, maxResults,
                                           descending);
  }

  // Conservative upper bound for the tile size of the vectorizable dimensions,
  // assuming small element size (one byte).
  int vectorizableCap = llvm::Log2_64(kMemorySectorSize);

  // Try reducing the search space by reducing the limits starting from the
  // less important dimensions. Reduce the limit on vectorizable dimensions and
  // remove the other dimensions.
  SmallVector<int> newLimits(limits);
  for (int dim = limits.size() - 1; dim >= 0; --dim) {
    int cap = dimensions[dim].vectorizableCount > 0 ? vectorizableCap : 0;
    if (limits[dim] > cap) {
      newLimits[dim] = cap;
      if (fitsBudget(newLimits)) {
        return generateCombinationsWithinRange(minBits, maxBits, newLimits,
                                               maxResults, descending);
      }
    }
  }

  // The search space is still too large, so lower the cap on all slots by up
  // to kReductionExponentFactorMax bits (halving the tile size each step).
  // As above, stop at the first cap that fits the budget or brings the limits
  // within `maxBits`. Otherwise, the lowest cap is used and the output is
  // truncated.
  constexpr int kReductionExponentFactorMax = 2;
  for (int i = 1; i <= kReductionExponentFactorMax; ++i) {
    for (int &limit : newLimits) {
      limit = std::min(limit, vectorizableCap - i);
    }
    if (fitsBudget(newLimits)) {
      break;
    }
  }
  return generateCombinationsWithinRange(minBits, maxBits, newLimits,
                                         maxResults, descending);
}

/// ----- Reduction processing -------------------------------------------------
///
/// The default generator, used for graphs without matmuls, and as a fallback
/// when the matmul dimensions are too small to use tensor cores. A tiling
/// configuration consists of the reduction tile size and the tile shape of the
/// main iteration space, both enumerated here.
///
/// Reduction tile size:
///
/// - The sizes range from 32 elements up to the largest reduction size in the
///   graph (rounded up to a power of two), and are enumerated from the largest,
///   as larger reduction tiles need fewer loop iterations.
/// - Graphs without reductions use a unit reduction tile. On the matmul
///   fallback path, the contraction tile sizes are enumerated instead.
///
/// Main tile shape:
///
/// - The upper bound on the number of tile elements is the number of elements
///   per SM if the grid runs in a single wave. Larger tiles would leave SMs
///   idle, but the bound is kept above the minimum tile size (one warp).
/// - The main tile combined with the reduction tile is limited to 2^15
///   elements, so larger reduction tiles leave room for smaller main tiles.
/// - The evaluator is used to lower the upper bound while the estimated
///   register storage exceeds the register file, as such tiles are likely to
///   spill.
/// - Non-tileable dimensions are forced to a unit tile size. They are still
///   counted in the single wave bound, as they multiply the number of blocks.
///
/// The result budget is split evenly between the reduction tile sizes, and the
/// tile shapes are generated by the balanced combination generator, starting
/// from the largest tiles.

/// Calculates the maximum reduction size in the iteration space.
class ReductionSizeEstimator : public LayoutIterator {
public:
  LogicalResult processReduction(ReductionSourceAttr reduction) override {
    maxReductionSize = std::max(maxReductionSize, reduction.getReductionSize());
    return dispatch(reduction.getSource());
  }
  int64_t maxReductionSize = 1;
};

/// Calculates the maximum number of tile size bits in each dimension.
SmallVector<int> getMaxSizeBits(ArrayRef<DimensionInfo> dimensions,
                                ArrayRef<int64_t> shape,
                                bool markNonTileable = true) {
  SmallVector<int> result;
  for (const auto &dim : dimensions) {
    int64_t size = shape[dim.index];
    result.push_back(size > 0 ? llvm::Log2_64_Ceil(size)
                              : kMaxDynamicTileSizeBits);
    if (markNonTileable && dim.nonTileable) {
      result.back() = 0;
    }
  }
  return result;
}

// Calculates the maximum number of tile size bits in each dimension, taking
// into account the estimated storage size needed for the tile.
SmallVector<int> getMaxSizeBits(ArrayRef<DimensionInfo> dimensions,
                                ArrayRef<int> limits, int reductionTileSizeBits,
                                TilingEvaluator &evaluator) {
  SmallVector<int> result(limits);
  for (auto [idx, dim] : llvm::enumerate(dimensions)) {
    if (limits[idx] > 0) {
      // Run the evaluator to estimate the storage size needed for the maximum
      // tile size with the given reduction tile size.
      SmallVector<int32_t> tileShape(limits.size(), 1);
      tileShape[dim.index] = 1 << kMaxTileSizeBits;
      auto evalResult = evaluator.evaluate(
          *TilingConfig::get(tileShape, 1 << reductionTileSizeBits));
      if (succeeded(evalResult)) {
        // Proportionally lower the limit to avoid spilling.
        int64_t ratio = 1 + evalResult->registerStorageBytes /
                                (kMaxRegistersPerSM * kRegisterSize);
        int newLimit = kMaxTileSizeBits - llvm::Log2_64_Ceil(ratio);
        result[idx] = std::min(limits[idx], std::max(newLimit, 0));
      }
    }
  }
  return result;
}

/**
 * @brief Generates tiling configurations for a graph with reductions.
 * @param evaluator The tiling config evaluator.
 * @param dimensions The ordered dimensions of the main iteration space.
 * @param maxResults The maximum number of results to generate.
 * @param fallbackUpperBound Maximum reduction tile size on fallback path.
 */
SmallVector<TilingConfig>
generateReductionConfigs(TilingEvaluator &evaluator,
                         ArrayRef<DimensionInfo> dimensions, size_t maxResults,
                         int fallbackUpperBound = 0) {
  SmallVector<TilingConfig> result;

  // Calculate the maximum number of tile size bits in each dimension.
  auto shape = evaluator.getInputLayout().getShape();
  SmallVector<int> maxTileSizeBits =
      getMaxSizeBits(dimensions, shape, /*markNonTileable=*/false);

  // Set the lower and upper bounds for the reduction tile size.
  ReductionSizeEstimator iter;
  auto iterResult = iter.dispatch(evaluator.getInputLayout());
  (void)iterResult;

  int minReductionTileSizeBits = kMinReductionTileSizeBits;
  int maxReductionTileSizeBits = std::min<int>(
      llvm::Log2_64_Ceil(iter.maxReductionSize), kMaxTileSizeBits);

  if (maxReductionTileSizeBits > 0) {
    maxReductionTileSizeBits =
        std::max(maxReductionTileSizeBits, minReductionTileSizeBits);
    LLVM_DEBUG(llvm::dbgs() << "Max reduction size: " << iter.maxReductionSize
                            << ", max reduction tile size: "
                            << (1 << maxReductionTileSizeBits) << "\n");
  } else {
    minReductionTileSizeBits =
        std::min(kMinContractionTileSizeBits, fallbackUpperBound);
    maxReductionTileSizeBits = fallbackUpperBound;
  }

  // Estimate number of elements per SM if a single wave is used.
  // This defines the upper bound for the tile size, but do not allow values
  // that are too low (i.e. underutilize threads).
  const ArchInfo &archInfo = evaluator.getArchInfo();
  int bitsPerProcessorSingleWave =
      llvm::sum_of(maxTileSizeBits) - llvm::Log2_64(archInfo.numSMs);
  int globalUpperBound = std::max(bitsPerProcessorSingleWave, kMinTileSizeBits);

  LLVM_DEBUG(llvm::dbgs() << "Tile size upper bound: "
                          << (int64_t{1} << globalUpperBound)
                          << " (using SM count: " << archInfo.numSMs << ")\n");

  // Force the tile size limit for non-tileable dimensions.
  for (auto [idx, dim] : llvm::enumerate(dimensions)) {
    if (dim.nonTileable) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Non-tileable dimension: " << dim.index << "\n");
      maxTileSizeBits[idx] = 0;
    }
  }

  // Iterate over all possible reduction tile sizes.
  for (int reductionTileSizeBits = maxReductionTileSizeBits;
       reductionTileSizeBits >= minReductionTileSizeBits;
       --reductionTileSizeBits) {
    // Split the remaining result budget evenly over the remaining sizes.
    size_t maxResultsWeighted =
        llvm::divideCeil(maxResults - result.size(),
                         reductionTileSizeBits - minReductionTileSizeBits + 1);

    // Set the lower bound to the smallest tile for which the grid still runs
    // in a single wave (reductions only), or the default (32).
    // The register storage of a unit main tile estimates how many blocks can
    // be resident on one SM (capped by the hardware limit). The single wave
    // bound assumes one block per SM; if N blocks fit on each SM, the grid can
    // have N times more blocks, so the tile can be N times smaller and still
    // run in a single wave.
    int lowerBound = kMinTileSizeBits;
    if (reductionTileSizeBits > 0) {
      auto evalResult = evaluator.evaluate(
          *TilingConfig::get(SmallVector<int32_t>(dimensions.size(), 1),
                             1 << reductionTileSizeBits));
      if (succeeded(evalResult)) {
        int occupancy = std::clamp<int>((kMaxRegistersPerSM * kRegisterSize) /
                                            evalResult->registerStorageBytes,
                                        1, archInfo.maxBlocksPerSM);
        int bitsRemaining =
            bitsPerProcessorSingleWave - llvm::Log2_32_Ceil(occupancy);
        lowerBound = std::min(lowerBound, std::max(bitsRemaining, 0));
      }
    }

    // Lower the per-dimension limits so that a tile with this reduction tile
    // size fits into the register file. The tile cannot be larger than all
    // dimensions at their limits, nor than the maximum tile size.
    SmallVector<int> limits = getMaxSizeBits(dimensions, maxTileSizeBits,
                                             reductionTileSizeBits, evaluator);
    int upperBound = std::min(kMaxTileSizeBits, llvm::sum_of(limits));

    // Set the bounds for the tile size.
    upperBound = std::min(upperBound, globalUpperBound);
    lowerBound = std::min(lowerBound, upperBound);

    LLVM_DEBUG({
      llvm::dbgs() << "Reduction tile size: " << (1 << reductionTileSizeBits)
                   << ", tile size range: " << (1 << lowerBound) << ".."
                   << (1 << upperBound) << ", bits: [";
      llvm::interleave(limits, llvm::dbgs(), ",");
      llvm::dbgs() << "], limit: " << maxResultsWeighted << "\n";
    });

    // Build the tiling configuration.
    SmallVector<int32_t> tileShape(dimensions.size(), 1);
    int32_t reductionTileSize = 1 << reductionTileSizeBits;

    auto combinations = generateCombinationsBalanced(
        lowerBound, upperBound, limits, maxResultsWeighted,
        /*descending=*/true, dimensions);

    for (const auto &combination : combinations) {
      for (auto [bits, dim] : llvm::zip_equal(combination, dimensions)) {
        tileShape[dim.index] = 1 << bits;
      }
      result.push_back(*TilingConfig::get(tileShape, reductionTileSize));
    }
  }

  return result;
}

/// ----- Matmul processing ----------------------------------------------------
///
/// The generator for graphs with matmuls. The tile shape is built around the
/// MMA tile, so the dimensions are split into three groups, based on the
/// dimension analysis:
///
/// - LHS dimensions map to the M dimension of a matmul result. Their combined
///   tile size ranges from 16 to 128 elements.
/// - RHS dimensions map to the N dimension of a matmul result. Their combined
///   tile size ranges from 8 to 256 elements.
/// - Other dimensions (batch dimensions, and dimensions not used by matmuls)
///   are tiled with the remaining space.
///
/// If either the LHS or the RHS dimensions cannot reach the minimum size,
/// tensor cores cannot be used, and the reduction generator is used instead.
///
/// The configurations are enumerated in nested loops:
///
/// - The LHS and RHS tile combinations are generated separately, starting from
///   the largest tiles, and each pair of them is visited. The result budget is
///   split as a square root between the two sides, so that neither side limits
///   the other when the budget is small. If one side has fewer combinations,
///   its unused share is given to the other side.
/// - The contraction tile size ranges from 16 elements up to the contraction
///   size (at most 256), so that the LHS and RHS operand tiles (M x K and
///   N x K) do not exceed 2^15 elements. The sizes are enumerated from the
///   smallest, so that the small contraction tiles are always included when
///   the budget is too small to cover the full range.
/// - The other dimensions get the bits left after the output tile (M x N) and
///   the operand tiles, starting from the smallest tiles, as they do not
///   contribute to the MMA efficiency.
///
/// At each level, the remaining budget is split evenly between the remaining
/// iterations. The MMA constraints (e.g. the supported tile sizes per tensor
/// core generation) are not checked here, they are accounted for by the
/// evaluator when ranking the configurations.

/// Calculates the maximum contraction size in the iteration space.
class ContractionSizeEstimator : public LayoutIterator {
public:
  LogicalResult processMatmul(MatmulSourceAttr matmul) override {
    maxContractionSize = std::max(
        maxContractionSize, llvm::product_of(matmul.getContractingShape()));
    bool ok = succeeded(dispatch(matmul.getLhs())) &&
              succeeded(dispatch(matmul.getRhs()));
    return LogicalResult::success(ok);
  }
  int64_t maxContractionSize = 1;
};

/**
 * @brief Generates tiling configurations for a graph with matmuls.
 * @param evaluator The tiling config evaluator.
 * @param dimensions The ordered dimensions of the main iteration space.
 * @param maxResults The maximum number of results to generate.
 */
SmallVector<TilingConfig>
generateMatmulConfigs(TilingEvaluator &evaluator,
                      ArrayRef<DimensionInfo> dimensions, size_t maxResults) {
  SmallVector<TilingConfig> result;

  // Set the upper bound for the contraction tile size.
  ContractionSizeEstimator iter;
  auto iterResult = iter.dispatch(evaluator.getInputLayout());
  (void)iterResult;
  int maxContractionTileSizeBits =
      std::clamp<int>(llvm::Log2_64_Ceil(iter.maxContractionSize),
                      kMinContractionTileSizeBits, kMaxContractionTileSizeBits);

  LLVM_DEBUG(llvm::dbgs() << "Max contraction size: " << iter.maxContractionSize
                          << ", max contraction tile size: "
                          << (1 << maxContractionTileSizeBits) << "\n");

  // Separate the dimensions into LHS, RHS, and other dimensions.
  SmallVector<DimensionInfo> lhsDims, rhsDims, otherDims;
  for (const auto &dim : dimensions) {
    if (dim.usedInMatmulM) {
      lhsDims.push_back(dim);
    } else if (dim.usedInMatmulN) {
      rhsDims.push_back(dim);
    } else {
      otherDims.push_back(dim);
    }
  }

  // Calculate the maximum number of tile size bits in each dimension.
  auto shape = evaluator.getInputLayout().getShape();
  SmallVector<int> lhsLimits = getMaxSizeBits(lhsDims, shape);
  SmallVector<int> rhsLimits = getMaxSizeBits(rhsDims, shape);
  SmallVector<int> otherLimits = getMaxSizeBits(otherDims, shape);

  // Count the number of combinations for LHS and RHS dimensions.
  size_t lhsCount =
      countCombinations(kMinLhsTileSizeBits, kMaxLhsTileSizeBits, lhsLimits);
  size_t rhsCount =
      countCombinations(kMinRhsTileSizeBits, kMaxRhsTileSizeBits, rhsLimits);

  LLVM_DEBUG({
    llvm::dbgs() << "LHS dimensions: " << lhsDims.size() << " (" << lhsCount
                 << " combinations total)\n";
    llvm::dbgs() << "RHS dimensions: " << rhsDims.size() << " (" << rhsCount
                 << " combinations total)\n";
    llvm::dbgs() << "Other dimensions: " << otherDims.size() << "\n";
  });

  // If LHS/RHS dimensions are too small, fall back to default generator.
  if (lhsCount == 0 || rhsCount == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Falling back to reduction generator\n");
    return generateReductionConfigs(evaluator, dimensions, maxResults,
                                    maxContractionTileSizeBits);
  }

  // Split the search space by using `sqrt(maxResults)` for LHS/RHS number of
  // combinations, balance this number if any side is too small.
  size_t maxLhsResults = std::min<size_t>(lhsCount, std::sqrt(maxResults));
  size_t maxRhsResults = std::min(rhsCount, maxResults / maxLhsResults);
  maxLhsResults = std::min(lhsCount, maxResults / maxRhsResults);

  // Generate combinations for LHS and RHS dimensions.
  auto lhsCombinations = generateCombinationsWithinRange(
      kMinLhsTileSizeBits, kMaxLhsTileSizeBits, lhsLimits, maxLhsResults,
      /*descending=*/true);
  auto rhsCombinations = generateCombinationsWithinRange(
      kMinRhsTileSizeBits, kMaxRhsTileSizeBits, rhsLimits, maxRhsResults,
      /*descending=*/true);
  size_t remainingCount = lhsCombinations.size() * rhsCombinations.size();

  LLVM_DEBUG(llvm::dbgs() << "Outer iterations: " << lhsCombinations.size()
                          << " * " << rhsCombinations.size() << " = "
                          << remainingCount << "\n");
#ifndef NDEBUG
  DenseSet<std::pair<int, int>> visited;
#endif

  // Iterate over all generated LHS/RHS tile combinations.
  for (const auto &lhs : lhsCombinations) {
    for (const auto &rhs : rhsCombinations) {
      // Split the remaining result budget evenly over the remaining sizes.
      size_t maxResultsWeighted =
          llvm::divideCeil(maxResults - result.size(), remainingCount--);

      // Build the tiling configuration.
      SmallVector<int32_t> tileShape(dimensions.size(), 1);
      for (auto [bits, dim] : llvm::zip_equal(lhs, lhsDims)) {
        tileShape[dim.index] = 1 << bits;
      }
      for (auto [bits, dim] : llvm::zip_equal(rhs, rhsDims)) {
        tileShape[dim.index] = 1 << bits;
      }

      // Set the upper bound for the contraction tile size.
      int lhsBits = llvm::sum_of(lhs);
      int rhsBits = llvm::sum_of(rhs);
      int upperBound = std::min(kMaxTileSizeBits - std::max(lhsBits, rhsBits),
                                maxContractionTileSizeBits);
      LLVM_DEBUG({
        if (visited.insert({lhsBits, rhsBits}).second) {
          llvm::dbgs() << "LHS tile size: " << (1 << lhsBits)
                       << ", RHS tile size: " << (1 << rhsBits)
                       << ", contraction tile size upper bound: "
                       << (1 << upperBound) << "\n";
        }
      });

      // Iterate over all possible contraction tile sizes.
      // Start from the smaller sizes to handle limited search space (when the
      // budget is too small to cover the full range).
      for (int contractionTileSizeBits = kMinContractionTileSizeBits;
           contractionTileSizeBits <= upperBound; ++contractionTileSizeBits) {
        int32_t reductionTileSize = 1 << contractionTileSizeBits;

        if (!otherDims.empty()) {
          // Split the remaining result budget one last time.
          size_t maxResultsInner = llvm::divideCeil(
              maxResultsWeighted, upperBound - contractionTileSizeBits + 1);

          // Generate combinations for the other dimensions.
          int innerBound =
              kMaxTileSizeBits -
              std::max(lhsBits + rhsBits,
                       contractionTileSizeBits + std::max(lhsBits, rhsBits));
          auto combinations = generateCombinationsBalanced(
              0, innerBound, otherLimits, maxResultsInner, /*descending=*/false,
              otherDims);
          maxResultsWeighted -= combinations.size();

          // Add the generated configurations to the result.
          for (const auto &combination : combinations) {
            for (auto [bits, dim] : llvm::zip_equal(combination, otherDims)) {
              tileShape[dim.index] = 1 << bits;
            }
            result.push_back(*TilingConfig::get(tileShape, reductionTileSize));
          }
        } else if (maxResultsWeighted > 0) {
          // No other dimensions, add the configuration to the result.
          result.push_back(*TilingConfig::get(tileShape, reductionTileSize));
          --maxResultsWeighted;
        }
      }
    }
  }
  return result;
}

/// ----- Enumerator implementation --------------------------------------------
///
/// The entry point analyzes and orders the dimensions of the main iteration
/// space, then dispatches to the matmul generator if the graph has matmuls, or
/// to the reduction generator otherwise. No configurations are returned for
/// rank-0 iteration spaces or a zero budget. If no dimension can be tiled, the
/// only configuration is a unit tile with the largest reduction tile, which
/// the evaluator and the lowering clamp to the actual reduction size.

SmallVector<TilingConfig> enumerateTilings(TilingEvaluator &evaluator,
                                           size_t maxResults) {
  auto shape = evaluator.getInputLayout().getShape();
  if (shape.empty() || maxResults == 0) {
    return {};
  }

  // Collect the dimensions of the main iteration space.
  DimensionLayoutIterator iter(shape.size());
  auto inputResult = iter.start(evaluator.getInputLayout(), /*writes=*/false);
  (void)inputResult;
  auto outputResult = iter.start(evaluator.getOutputLayout(), /*writes=*/true);
  (void)outputResult;

  // Order the dimensions using the importance heuristic.
  auto dimensions = iter.getOrderedDimensions();
  LLVM_DEBUG({
    llvm::dbgs() << "Dimensions:";
    for (const auto &dim : dimensions) {
      llvm::dbgs() << " " << dim;
    }
    llvm::dbgs() << "\n";
  });

  // If the first dimension is non-tileable, return a unit tile shape.
  if (dimensions.front().nonTileable) {
    return {*TilingConfig::get(SmallVector<int32_t>(shape.size(), 1),
                               1 << kMaxTileSizeBits)};
  }

  // Generate tiling configurations.
  if (!evaluator.getGraph().getOps<MatmulOp>().empty()) {
    return generateMatmulConfigs(evaluator, dimensions, maxResults);
  }
  return generateReductionConfigs(evaluator, dimensions, maxResults);
}

} // namespace tiling_analysis
} // namespace mlir::nv_tensor_ir
