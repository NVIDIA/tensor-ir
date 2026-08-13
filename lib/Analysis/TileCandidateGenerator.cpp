// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/TileCandidateGenerator.h"

#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#define DEBUG_TYPE "tile-candidate-generator"

namespace mlir {
namespace nv_tensor_ir {
namespace {

constexpr int64_t kSmallTensorThreshold = 100'000;
constexpr int64_t kThreadsPerBlockChoices[] = {128, 256};

static int32_t floorPowerOf2AtLeast1(int64_t v) {
  if (v <= 1) {
    return 1;
  }
  return static_cast<int32_t>(llvm::bit_floor(static_cast<uint64_t>(v)));
}

static int64_t tileElements(llvm::ArrayRef<int32_t> tileShape) {
  int64_t elems = 1;
  for (int32_t v : tileShape) {
    elems *= std::max<int32_t>(1, v);
  }
  return elems;
}

/// Whether an operand varies along at least one tiled dimension. An empty
/// stride list is rank-0 (which the candidate generator rejects), while an
/// all-zero list is a broadcast scalar and intentionally contributes no
/// dimension-varying traffic to tile scoring.
static bool hasDimensionVaryingTraffic(llvm::ArrayRef<DimSize> strides) {
  return llvm::any_of(strides, [](DimSize stride) { return stride > 0; });
}

/// Traffic and tile-shape metrics used by active-layout conflict scoring.
struct ActiveLayoutProfile {
  /// Dimension indices with non-zero byte-weighted traffic, in index order.
  llvm::SmallVector<size_t> trafficDims;
  /// Traffic dimensions that are contiguous for at least one operand.
  llvm::SmallVector<size_t> strideOneTrafficDims;
  /// Indices of the highest- and second-highest-traffic dimensions.
  size_t dominantDim = 0;
  size_t secondaryDim = 0;
  /// Byte-traffic weights of the dominant and secondary dimensions.
  double dominantWeight = 0.0;
  double secondaryWeight = 0.0;
  /// Dominant/secondary traffic and tile-extent ratios; zero without a pair.
  double weightRatio = 0.0;
  double tileRatio = 0.0;
  /// Tile extents along the dominant and secondary dimensions.
  int32_t dominantExtent = 1;
  int32_t secondaryExtent = 1;
  /// Minimum and maximum extents across dimensions with real traffic.
  int32_t minActiveTile = std::numeric_limits<int32_t>::max();
  int32_t maxActiveTile = 1;
  /// Products of all traffic extents and of the dominant pair, in elements.
  int64_t activeTrafficElems = 1;
  int64_t dominantPairElems = 1;
};

/// Summarize traffic-aware tile geometry for layout-conflict scoring.
///
/// `tileShape` contains per-dimension tile extents, `dimTraffic` contains
/// byte-traffic weights for those dimensions, and `allStrides` contains each
/// operand's logical strides. Inputs use the same dimension order; dimensions
/// missing from `tileShape` are ignored. The returned profile records dominant
/// traffic, contiguous dimensions, and element-count/aspect-ratio metrics.
static ActiveLayoutProfile
summarizeActiveLayout(llvm::ArrayRef<int32_t> tileShape,
                      llvm::ArrayRef<double> dimTraffic,
                      llvm::ArrayRef<llvm::SmallVector<DimSize>> allStrides) {
  ActiveLayoutProfile profile;

  for (size_t d = 0; d < dimTraffic.size(); ++d) {
    double w = dimTraffic[d];
    if (w > 0.0) {
      profile.trafficDims.push_back(d);
      if (d < tileShape.size()) {
        int32_t extent = std::max<int32_t>(1, tileShape[d]);
        profile.minActiveTile = std::min(profile.minActiveTile, extent);
        profile.maxActiveTile = std::max(profile.maxActiveTile, extent);
        profile.activeTrafficElems *= extent;
      }
    }

    if (w > profile.dominantWeight) {
      profile.secondaryWeight = profile.dominantWeight;
      profile.secondaryDim = profile.dominantDim;
      profile.dominantWeight = w;
      profile.dominantDim = d;
    } else if (w > profile.secondaryWeight) {
      profile.secondaryWeight = w;
      profile.secondaryDim = d;
    }
  }

  for (const auto &srcStrides : allStrides) {
    if (!hasDimensionVaryingTraffic(srcStrides)) {
      continue;
    }
    for (size_t d = 0; d < srcStrides.size(); ++d) {
      if (srcStrides[d] != 1 || d >= dimTraffic.size() ||
          dimTraffic[d] <= 0.0) {
        continue;
      }
      if (!llvm::is_contained(profile.strideOneTrafficDims, d)) {
        profile.strideOneTrafficDims.push_back(d);
      }
    }
  }

  if (profile.dominantWeight > 0.0 && profile.secondaryWeight > 0.0 &&
      profile.dominantDim < tileShape.size() &&
      profile.secondaryDim < tileShape.size()) {
    profile.weightRatio = profile.dominantWeight / profile.secondaryWeight;
    profile.dominantExtent =
        std::max<int32_t>(1, tileShape[profile.dominantDim]);
    profile.secondaryExtent =
        std::max<int32_t>(1, tileShape[profile.secondaryDim]);
    profile.tileRatio = static_cast<double>(profile.dominantExtent) /
                        static_cast<double>(profile.secondaryExtent);
    profile.dominantPairElems = static_cast<int64_t>(profile.dominantExtent) *
                                static_cast<int64_t>(profile.secondaryExtent);
  }

  return profile;
}

static float computeActiveLayoutConflictScore(
    const ActiveLayoutProfile &profile, int64_t elems, DimSize minElemBytes,
    DimSize maxTrafficElemBytes, const GpuArchitecture &gpu) {
  float score = 0.0f;

  // Skew tile area toward the byte-dominant stride-1 dimension, with guardrails
  // for oversized secondary coverage, CTA footprint, and dominant extent.
  if (profile.weightRatio >= 1.5 && profile.tileRatio > 1.0) {
    double aspectBonus = std::min(50.0, 25.0 * std::log2(profile.tileRatio));
    if (profile.weightRatio >= 2.5 && profile.secondaryExtent >= 64) {
      aspectBonus -= 30.0;
    }
    if (profile.weightRatio >= 2.5 && elems > 4096) {
      aspectBonus -= 15.0 * std::log2(static_cast<double>(elems) / 4096.0);
    }
    if (profile.dominantExtent > 256) {
      aspectBonus -=
          15.0 * std::log2(static_cast<double>(profile.dominantExtent) / 256.0);
    }
    score += static_cast<float>(std::max(0.0, aspectBonus));
  }

  // bf16/f16 layout-conflict fusions often lower through reg->smem->reg layout
  // conversion. A moderate balanced active tile gives both competing stride-1
  // operands enough contiguous work while keeping the staging footprint
  // compact.
  if (minElemBytes <= 2 && profile.trafficDims.size() >= 2 &&
      profile.minActiveTile >= 32 && profile.maxActiveTile <= 64 &&
      profile.activeTrafficElems >= 2048 &&
      profile.activeTrafficElems <= 4096) {
    score += 45.0f;
    if (profile.minActiveTile == profile.maxActiveTile) {
      score += 10.0f;
    }
  }

  // When byte traffic is clearly heavier along one bf16 active dim, keep
  // long-and-thin dominant-stream tiles competitive instead of letting the
  // balanced staging preference crowd out 256x8-style shapes.
  if (minElemBytes == 2 && profile.weightRatio >= 1.5 &&
      profile.dominantExtent >= 256 && profile.secondaryExtent <= 8 &&
      profile.dominantPairElems >= 1024 && profile.dominantPairElems <= 4096) {
    score += 35.0f;
  }

  // Mixed narrow+wide competing stride-1 layouts can force reg->smem->reg
  // staging. Penalize active footprints that exceed a warp-by-cache-line
  // budget.
  if (minElemBytes <= 2 && maxTrafficElemBytes >= 4 &&
      profile.trafficDims.size() >= 2 &&
      profile.strideOneTrafficDims.size() >= 2) {
    int64_t cacheLineElems = std::max<int64_t>(
        1, gpu.l1CacheLineSize / std::max<DimSize>(1, maxTrafficElemBytes));
    int64_t activeSpan =
        floorPowerOf2AtLeast1(std::min<int64_t>(gpu.warpSize, cacheLineElems));
    int64_t compactBudget = std::max<int64_t>(1, activeSpan * activeSpan);
    if (profile.activeTrafficElems > compactBudget) {
      score -= static_cast<float>(gpu.warpSize) *
               static_cast<float>(
                   std::log2(static_cast<double>(profile.activeTrafficElems) /
                             static_cast<double>(compactBudget)));
    }
  }

  return score;
}

static float computeSizePrior(int64_t elems, int trafficActiveCount,
                              size_t tensorRank) {
  float score = 0.0f;
  if (elems >= 64 && elems <= 4096) {
    score = 50.0f;
  } else if (elems > 4096 && elems <= 16384) {
    score = 35.0f;
  } else if (elems >= 16) {
    score = 10.0f;
  }

  int64_t guardrail =
      (trafficActiveCount == 1 && tensorRank >= 3) ? 1024 : 2048;
  if (elems < guardrail) {
    score -= 35.0f;
  }
  return score;
}

static float computePassiveCoverageScore(llvm::ArrayRef<int32_t> tileShape,
                                         llvm::ArrayRef<DimSize> tensorShape,
                                         llvm::ArrayRef<double> dimTraffic,
                                         DimSize minElemBytes,
                                         DimSize cacheLineSize, int64_t elems) {
  size_t activeDim = 0;
  int activeCount = 0;
  for (size_t d = 0; d < dimTraffic.size(); ++d) {
    if (dimTraffic[d] > 0.0) {
      activeDim = d;
      ++activeCount;
    }
  }
  if (activeCount != 1 || tensorShape.size() < 3 ||
      activeDim >= tileShape.size()) {
    return 0.0f;
  }

  float score = 0.0f;
  bool tinyActiveExtent = tensorShape[activeDim] <= 8;
  double activeTarget =
      static_cast<double>(std::max<DimSize>(1, cacheLineSize)) /
      static_cast<double>(std::max<DimSize>(1, minElemBytes));
  activeTarget = std::max(1.0, activeTarget);
  if (static_cast<double>(tileShape[activeDim]) > activeTarget) {
    score -= static_cast<float>(
        25.0 *
        std::log2(static_cast<double>(tileShape[activeDim]) / activeTarget));
  }

  int coveredPassiveDims = 0;
  int64_t passiveProduct = 1;
  int32_t maxMediumPassiveTile = 1;
  for (size_t d = 0; d < tensorShape.size() && d < tileShape.size(); ++d) {
    if (d == activeDim || tensorShape[d] <= 1) {
      continue;
    }
    if (tileShape[d] > 1) {
      ++coveredPassiveDims;
      passiveProduct *= tileShape[d];
    }
    if (tensorShape[d] <= 4) {
      int32_t fullTiny =
          static_cast<int32_t>(floorPowerOf2AtLeast1(tensorShape[d]));
      score += (tileShape[d] >= fullTiny) ? 35.0f : -10.0f;
    } else if (tensorShape.size() >= 4 && tensorShape[d] <= 64) {
      int32_t mediumTarget = std::min<int32_t>(
          16, static_cast<int32_t>(floorPowerOf2AtLeast1(tensorShape[d])));
      if (tileShape[d] >= mediumTarget) {
        maxMediumPassiveTile =
            std::max<int32_t>(maxMediumPassiveTile, tileShape[d]);
      }
      if (tileShape[d] > 1) {
        score += 10.0f;
      }
    } else if (tileShape[d] > 1) {
      score += 10.0f;
    }
  }
  if (coveredPassiveDims >= 2) {
    score += 10.0f;
  }
  if (maxMediumPassiveTile >= 8) {
    score += static_cast<float>(std::min(
        55.0,
        30.0 +
            10.0 * std::log2(static_cast<double>(maxMediumPassiveTile) / 8.0)));
  }
  if (tinyActiveExtent && elems >= 512) {
    score += static_cast<float>(
        std::min(60.0, 20.0 * std::log2(static_cast<double>(elems) / 256.0)));
  }
  if (passiveProduct > 16) {
    double penaltyScale = tinyActiveExtent ? 5.0 : 20.0;
    score -= static_cast<float>(
        penaltyScale * std::log2(static_cast<double>(passiveProduct) / 16.0));
  }
  return score;
}

static float computeParallelismFactor(int64_t totalElems, int64_t tileElems,
                                      int64_t numSms) {
  int64_t blockCount = (totalElems + std::max<int64_t>(1, tileElems) - 1) /
                       std::max<int64_t>(1, tileElems);
  float desiredBlocks = static_cast<float>(std::max<int64_t>(1, numSms));
  return std::min(1.0f, static_cast<float>(blockCount) / desiredBlocks);
}

/// Max usable shared memory per CTA in bytes, looked up by compute capability.
/// Values exclude the 1KB reserved by the CUDA driver.
/// Used by the smem-aware tile generation track (Track B) to bound tile sizes
/// so that the reg→smem→reg staging buffers fit.
static int64_t getSmemCapacityBytes(ComputeCapability computeCapability) {
  const int32_t cc = toCcInt(computeCapability);
  if (cc >= 100) {
    return 227 * 1024; // sm_100+
  }
  if (cc >= 90) {
    return 227 * 1024; // sm_90
  }
  if (cc >= 89) {
    return 99 * 1024; // sm_89
  }
  if (cc >= 80) {
    return 163 * 1024; // sm_80
  }
  return 48 * 1024; // fallback
}

} // namespace

TileCandidateGenerator::TileCandidateGenerator(const GpuArchitecture &gpu,
                                               int maxCandidates)
    : gpu_(gpu), maxCandidates_(maxCandidates) {}

llvm::SmallVector<TileCandidateGenerator::Candidate>
TileCandidateGenerator::generateCandidates(llvm::ArrayRef<DimSize> shape,
                                           llvm::ArrayRef<DimSize> strides,
                                           DimSize elementSizeBytes) {
  llvm::DenseMap<size_t, DimSize> emptyConstraints;
  return generateCandidates(shape, strides, elementSizeBytes, emptyConstraints);
}

llvm::SmallVector<TileCandidateGenerator::Candidate>
TileCandidateGenerator::generateCandidates(
    llvm::ArrayRef<DimSize> shape,
    llvm::ArrayRef<llvm::SmallVector<DimSize>> allStrides,
    llvm::ArrayRef<DimSize> allElementSizeBytes) {
  llvm::DenseMap<size_t, DimSize> emptyConstraints;
  return generateCandidates(shape, allStrides, allElementSizeBytes,
                            emptyConstraints);
}

llvm::SmallVector<TileCandidateGenerator::Candidate>
TileCandidateGenerator::generateCandidates(
    llvm::ArrayRef<DimSize> shape, llvm::ArrayRef<DimSize> strides,
    DimSize elementSizeBytes,
    const llvm::DenseMap<size_t, DimSize> &constraints) {
  llvm::SmallVector<llvm::SmallVector<DimSize>> allStrides;
  allStrides.push_back(
      llvm::SmallVector<DimSize>(strides.begin(), strides.end()));
  llvm::SmallVector<DimSize> allElemBytes = {elementSizeBytes};
  return generateCandidates(shape, allStrides, allElemBytes, constraints);
}

llvm::SmallVector<TileCandidateGenerator::Candidate>
TileCandidateGenerator::generateCandidates(
    llvm::ArrayRef<DimSize> shape,
    llvm::ArrayRef<llvm::SmallVector<DimSize>> allStrides,
    llvm::ArrayRef<DimSize> allElementSizeBytes,
    const llvm::DenseMap<size_t, DimSize> &constraints) {

  size_t ndim = shape.size();
  if (ndim == 0 || allStrides.empty() || allElementSizeBytes.empty()) {
    return {};
  }
  for (const auto &s : allStrides) {
    if (s.size() != ndim) {
      return {};
    }
  }
  DimSize minElemBytes =
      *std::min_element(allElementSizeBytes.begin(), allElementSizeBytes.end());
  if (minElemBytes <= 0) {
    return {};
  }

  // ═══════════════════════════════════════════════════════
  // Step 1: Compute constrained product and identify free dims
  // ═══════════════════════════════════════════════════════
  int64_t constrainedProduct = 1;
  for (const auto &[dim, val] : constraints) {
    constrainedProduct *= val;
  }

  llvm::SmallVector<size_t> freeDims;
  for (size_t i = 0; i < ndim; ++i) {
    if (!constraints.count(i)) {
      freeDims.push_back(i);
    }
  }

  // All dims constrained — return single candidate
  if (freeDims.empty()) {
    llvm::SmallVector<int32_t> tile(ndim, 1);
    for (const auto &[dim, val] : constraints) {
      tile[dim] = static_cast<int32_t>(val);
    }
    return {{tile, 0.0f, "AllConstrained"}};
  }

  // ═══════════════════════════════════════════════════════
  // Step 2: Compute per-dim byte-weighted importance.
  // A dim's weight = total bytes across operands where that dim has stride 1.
  // ═══════════════════════════════════════════════════════
  int64_t totalElements = 1;
  for (DimSize s : shape) {
    totalElements *= s;
  }

  llvm::SmallVector<double> dimWeight(ndim, 0.0);
  for (size_t src = 0; src < allStrides.size(); ++src) {
    DimSize elemBytes = (src < allElementSizeBytes.size())
                            ? allElementSizeBytes[src]
                            : allElementSizeBytes.back();
    double srcBytes = static_cast<double>(totalElements) * elemBytes;
    for (size_t d = 0; d < ndim; ++d) {
      if (constraints.count(d)) {
        continue;
      }
      if (allStrides[src][d] == 1) {
        dimWeight[d] += srcBytes;
      }
    }
  }

  // Classify free dims into active (weight > 0) and passive (weight == 0).
  llvm::SmallVector<size_t> activeDims, passiveDims;
  for (size_t d : freeDims) {
    if (dimWeight[d] > 0.0) {
      activeDims.push_back(d);
    } else {
      passiveDims.push_back(d);
    }
  }

  // Compute shape limits for each dim (power-of-2 ceiling).
  llvm::SmallVector<int64_t> shapeLimit(ndim);
  for (size_t d = 0; d < ndim; ++d) {
    shapeLimit[d] = static_cast<int64_t>(
        llvm::PowerOf2Ceil(static_cast<uint64_t>(shape[d])));
  }

  // If no active dims (all broadcasts?), fall back to treating all free dims
  // as active with equal weight.
  if (activeDims.empty()) {
    activeDims = freeDims;
    passiveDims.clear();
    for (size_t d : activeDims) {
      dimWeight[d] = 1.0;
    }
  }

  // Pick the highest-weight active dim as "fastestFreeDim" for scoring.
  size_t fastestFreeDim = activeDims[0];
  for (size_t d : activeDims) {
    if (dimWeight[d] > dimWeight[fastestFreeDim]) {
      fastestFreeDim = d;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Dim weights:";
    for (size_t d : freeDims) {
      llvm::dbgs() << " d" << d << "="
                   << (dimWeight[d] > 0 ? "ACTIVE" : "passive") << "("
                   << dimWeight[d] << ")";
    }
    llvm::dbgs() << "\n";
  });

  // ═══════════════════════════════════════════════════════
  // Step 3: Small-tensor fallback
  // ═══════════════════════════════════════════════════════
  llvm::SmallVector<Candidate> allCandidates;

  auto appendCandidate = [&](llvm::SmallVector<int32_t> tile,
                             std::string rationale) {
    if (!isValidTile(tile, shape)) {
      return;
    }
    allCandidates.push_back({std::move(tile), 0.0f, std::move(rationale)});
  };

  if (totalElements < kSmallTensorThreshold) {
    int64_t smallTargetElems = (minElemBytes <= 2) ? 128 : 64;
    llvm::SmallVector<int32_t> tile(ndim, 1);
    for (const auto &[dim, val] : constraints) {
      tile[dim] = static_cast<int32_t>(val);
    }
    tile[fastestFreeDim] = floorPowerOf2AtLeast1(
        std::min<int64_t>(shapeLimit[fastestFreeDim], smallTargetElems));

    int64_t seededElems = tileElements(tile);
    int64_t remaining = std::max<int64_t>(
        1, smallTargetElems / std::max<int64_t>(1, seededElems));
    distributeBudget(tile, remaining, passiveDims, shape);
    appendCandidate(std::move(tile), "SmallTensor");
  }

  // ═══════════════════════════════════════════════════════
  // Step 4: Byte-weighted candidate generation
  //
  // Allocate active dims proportionally by byte-weight, then cascade remaining
  // budget through all free dims. Passive dims get independent values (not a
  // single uniform passiveVal). Two absorb orderings (passive-descending and
  // passive-ascending by shape) ensure diversity.
  // ═══════════════════════════════════════════════════════
  constexpr int64_t allBudgets[] = {32,   64,   128,  256,  512,
                                    1024, 2048, 4096, 8192, 16384};
  constexpr int64_t kMinActiveDim = 1;

  // Only use budgets where the tile is no larger than totalElements (tiles
  // bigger than the iteration space are pointless).
  llvm::SmallVector<int64_t> budgets;
  for (int64_t b : allBudgets) {
    if (b <= totalElements) {
      budgets.push_back(b);
    }
  }

  // Two absorb orderings: active dims by weight desc, then passive by shape.
  // Order 0: passive by shape descending (grow large dims first).
  // Order 1: passive by shape ascending (diversity).
  llvm::SmallVector<size_t> absorbOrder[2];
  {
    llvm::SmallVector<size_t> activeByWeight(activeDims.begin(),
                                             activeDims.end());
    llvm::sort(activeByWeight,
               [&](size_t a, size_t b) { return dimWeight[a] > dimWeight[b]; });
    llvm::SmallVector<size_t> passiveDesc(passiveDims.begin(),
                                          passiveDims.end());
    llvm::sort(passiveDesc,
               [&](size_t a, size_t b) { return shape[a] > shape[b]; });
    llvm::SmallVector<size_t> passiveAsc(passiveDims.begin(),
                                         passiveDims.end());
    llvm::sort(passiveAsc,
               [&](size_t a, size_t b) { return shape[a] < shape[b]; });

    absorbOrder[0].append(activeByWeight.begin(), activeByWeight.end());
    absorbOrder[0].append(passiveDesc.begin(), passiveDesc.end());
    absorbOrder[1].append(activeByWeight.begin(), activeByWeight.end());
    absorbOrder[1].append(passiveAsc.begin(), passiveAsc.end());
  }

  // Cascade absorb: distribute surplus budget through dims in the given order.
  // Each dim grows by the largest power-of-2 factor that fits.
  auto absorbSlack = [&](llvm::SmallVectorImpl<int32_t> &tile,
                         int64_t targetFreeBudget,
                         llvm::ArrayRef<size_t> order) {
    int64_t freeProd = 1;
    for (size_t d : freeDims) {
      freeProd *= tile[d];
    }
    if (freeProd >= targetFreeBudget) {
      return;
    }
    int64_t surplus = targetFreeBudget / freeProd;
    for (size_t d : order) {
      if (surplus <= 1) {
        break;
      }
      int64_t maxGrow = shapeLimit[d] / std::max<int64_t>(1, tile[d]);
      int64_t growFactor = floorPowerOf2AtLeast1(std::min(surplus, maxGrow));
      if (growFactor <= 1) {
        continue;
      }
      tile[d] = static_cast<int32_t>(tile[d] * growFactor);
      surplus /= growFactor;
    }
  };

  // Helper: build a tile from per-freeDim allocations.
  auto buildTile =
      [&](llvm::ArrayRef<int64_t> alloc) -> llvm::SmallVector<int32_t> {
    llvm::SmallVector<int32_t> tile(ndim, 1);
    for (const auto &[dim, val] : constraints) {
      tile[dim] = static_cast<int32_t>(val);
    }
    for (size_t fi = 0; fi < freeDims.size(); ++fi) {
      int64_t v = std::min(alloc[fi], shapeLimit[freeDims[fi]]);
      tile[freeDims[fi]] = std::max<int32_t>(1, floorPowerOf2AtLeast1(v));
    }
    return tile;
  };

  for (int64_t budget : budgets) {
    int64_t freeBudget = std::max<int64_t>(1, budget / constrainedProduct);

    // Proportional allocation for active dims only; passive dims start at 1.
    llvm::SmallVector<int64_t> centerAlloc(freeDims.size(), 1);
    {

      double totalWeight = 0.0;
      for (size_t d : activeDims) {
        totalWeight += dimWeight[d];
      }
      // Subtract passive allocation from the budget available for active dims.
      int64_t passiveProd = 1;
      for (size_t fi = 0; fi < freeDims.size(); ++fi) {
        if (dimWeight[freeDims[fi]] <= 0.0) {
          passiveProd *= centerAlloc[fi];
        }
      }
      int64_t activeBudget = std::max<int64_t>(1, freeBudget / passiveProd);
      double logBudget = std::log2(std::max<double>(1.0, activeBudget));

      for (size_t fi = 0; fi < freeDims.size(); ++fi) {
        size_t d = freeDims[fi];
        if (dimWeight[d] <= 0.0) {
          continue;
        }
        double fraction = dimWeight[d] / std::max(1.0, totalWeight);
        int64_t raw = static_cast<int64_t>(std::pow(2.0, fraction * logBudget));
        int64_t clamped = std::min(raw, shapeLimit[d]);
        clamped = std::max<int64_t>(std::min(kMinActiveDim, shapeLimit[d]),
                                    floorPowerOf2AtLeast1(clamped));
        centerAlloc[fi] = clamped;
      }
    }

    // Generate center tiles with both absorb orderings for diversity.
    llvm::SmallVector<int64_t> centerFinal[2];
    for (int ord = 0; ord < 2; ++ord) {
      auto tile = buildTile(centerAlloc);
      absorbSlack(tile, freeBudget, absorbOrder[ord]);
      appendCandidate(tile, "B" + std::to_string(budget) + "-Center" +
                                std::to_string(ord));
      centerFinal[ord].resize(freeDims.size());
      for (size_t fi = 0; fi < freeDims.size(); ++fi) {
        centerFinal[ord][fi] = tile[freeDims[fi]];
      }
    }

    // Perturbations between ALL pairs of free dims.
    // For each center ordering, shift budget between dim pairs.
    for (int ord = 0; ord < 2; ++ord) {
      const auto &center = centerFinal[ord];

      for (size_t fi = 0; fi < freeDims.size(); ++fi) {
        for (size_t fj = fi + 1; fj < freeDims.size(); ++fj) {
          size_t di = freeDims[fi], dj = freeDims[fj];
          for (int shift = 1; shift <= 3; ++shift) {
            int64_t factor = int64_t{1} << shift;

            // Shift from j to i
            {
              llvm::SmallVector<int64_t> alloc(center);
              alloc[fi] = std::min(alloc[fi] * factor, shapeLimit[di]);
              int64_t floor_j = (dimWeight[dj] > 0)
                                    ? std::min(kMinActiveDim, shapeLimit[dj])
                                    : int64_t{1};
              alloc[fj] = std::max(floor_j, alloc[fj] / factor);
              auto tile = buildTile(alloc);
              absorbSlack(tile, freeBudget, absorbOrder[ord]);
              appendCandidate(std::move(tile), "B" + std::to_string(budget) +
                                                   "-S" + std::to_string(fi) +
                                                   "+" + std::to_string(fj));
            }
            // Shift from i to j
            {
              llvm::SmallVector<int64_t> alloc(center);
              alloc[fj] = std::min(alloc[fj] * factor, shapeLimit[dj]);
              int64_t floor_i = (dimWeight[di] > 0)
                                    ? std::min(kMinActiveDim, shapeLimit[di])
                                    : int64_t{1};
              alloc[fi] = std::max(floor_i, alloc[fi] / factor);
              auto tile = buildTile(alloc);
              absorbSlack(tile, freeBudget, absorbOrder[ord]);
              appendCandidate(std::move(tile), "B" + std::to_string(budget) +
                                                   "-S" + std::to_string(fj) +
                                                   "+" + std::to_string(fi));
            }
          }
        }
      }
    }

    // Compound passive-dim enumeration: for 3+ free dims, generate tiles where
    // one passive dim is elevated while active dims share the remaining budget
    // proportionally. Pairwise perturbations alone cannot reach tiles like
    // 64x64x4 from a 64x64x4 center (when a second passive dim needs a
    // different value from the first).
    constexpr int64_t kMaxPassiveTile = 4;
    if (!passiveDims.empty()) {
      for (size_t pi = 0; pi < passiveDims.size(); ++pi) {
        size_t dp = passiveDims[pi];
        for (int64_t pv = 2;
             pv <= std::min<int64_t>(kMaxPassiveTile, shapeLimit[dp]);
             pv *= 2) {
          llvm::SmallVector<int64_t> alloc(freeDims.size(), 1);
          // Set the target passive dim to pv; other passive dims at floor.
          int64_t passiveProduct = 1;
          for (size_t fi = 0; fi < freeDims.size(); ++fi) {
            size_t d = freeDims[fi];
            if (d == dp) {
              alloc[fi] = pv;
              passiveProduct *= pv;
            } else if (dimWeight[d] <= 0.0) {
              alloc[fi] = 1;
              passiveProduct *= alloc[fi];
            }
          }
          int64_t effectiveBudget =
              std::max<int64_t>(1, freeBudget / passiveProduct);
          if (effectiveBudget < 2) {
            continue;
          }

          // Distribute remaining budget proportionally among active dims.
          double totalWeight = 0.0;
          for (size_t d : activeDims) {
            totalWeight += dimWeight[d];
          }
          double logEB = std::log2(std::max<double>(1.0, effectiveBudget));
          for (size_t fi = 0; fi < freeDims.size(); ++fi) {
            size_t d = freeDims[fi];
            if (dimWeight[d] <= 0.0) {
              continue;
            }
            double fraction = dimWeight[d] / std::max(1.0, totalWeight);
            int64_t raw = static_cast<int64_t>(std::pow(2.0, fraction * logEB));
            int64_t clamped = std::min(raw, shapeLimit[d]);
            clamped = std::max<int64_t>(std::min(kMinActiveDim, shapeLimit[d]),
                                        floorPowerOf2AtLeast1(clamped));
            alloc[fi] = clamped;
          }

          for (int ord = 0; ord < 2; ++ord) {
            auto tile = buildTile(alloc);
            absorbSlack(tile, freeBudget, absorbOrder[ord]);
            appendCandidate(std::move(tile), "B" + std::to_string(budget) +
                                                 "-P" + std::to_string(pi) +
                                                 "v" + std::to_string(pv) +
                                                 "-" + std::to_string(ord));
          }
        }
      }
    }

    // Single-active-dim rank-3+ layouts often represent one contiguous memory
    // movement repeated across passive reshape/broadcast coordinates. Pairwise
    // shifts can elevate one passive dim at a time, but miss combinations such
    // as 64x4x4 where two passive dims should both be materialized while the
    // active dim stays at one cache-line worth of contiguous elements.
    if (activeDims.size() == 1 && passiveDims.size() >= 2) {
      size_t activeDim = activeDims[0];
      size_t activeFi = 0;
      for (size_t fi = 0; fi < freeDims.size(); ++fi) {
        if (freeDims[fi] == activeDim) {
          activeFi = fi;
          break;
        }
      }

      for (size_t pi = 0; pi < passiveDims.size(); ++pi) {
        for (size_t pj = pi + 1; pj < passiveDims.size(); ++pj) {
          size_t dpi = passiveDims[pi];
          size_t dpj = passiveDims[pj];
          for (int64_t pvi = 2;
               pvi <= std::min<int64_t>(kMaxPassiveTile, shapeLimit[dpi]);
               pvi *= 2) {
            for (int64_t pvj = 2;
                 pvj <= std::min<int64_t>(kMaxPassiveTile, shapeLimit[dpj]);
                 pvj *= 2) {
              int64_t passiveProduct = pvi * pvj;
              int64_t effectiveBudget =
                  std::max<int64_t>(1, freeBudget / passiveProduct);
              if (effectiveBudget < 2) {
                continue;
              }

              llvm::SmallVector<int64_t> alloc(freeDims.size(), 1);
              for (size_t fi = 0; fi < freeDims.size(); ++fi) {
                if (freeDims[fi] == dpi) {
                  alloc[fi] = pvi;
                } else if (freeDims[fi] == dpj) {
                  alloc[fi] = pvj;
                }
              }
              alloc[activeFi] = std::max<int64_t>(
                  1, floorPowerOf2AtLeast1(
                         std::min(effectiveBudget, shapeLimit[activeDim])));

              auto tile = buildTile(alloc);
              appendCandidate(std::move(tile), "B" + std::to_string(budget) +
                                                   "-PP" + std::to_string(pi) +
                                                   "v" + std::to_string(pvi) +
                                                   "-" + std::to_string(pj) +
                                                   "v" + std::to_string(pvj));
            }
          }
        }
      }
    }
  }

  // ═══════════════════════════════════════════════════════
  // Step 5: Smem-aware candidates (Track B)
  //
  // For fusions with 2+ active dims (different operands have stride-1 on
  // different dimensions), generate balanced tiles sized to fit in shared
  // memory. These cover the case where the compiler will use reg→smem→reg
  // layout conversion due to thread layout conflicts.
  // ═══════════════════════════════════════════════════════
  if (activeDims.size() >= 2) {
    DimSize maxElemBytes = *std::max_element(allElementSizeBytes.begin(),
                                             allElementSizeBytes.end());
    generateSmemCandidates(activeDims, shape, shapeLimit, constraints,
                           maxElemBytes, ndim, allCandidates);
  }

  // ═══════════════════════════════════════════════════════
  // Step 6: Dedup, score, merge tracks, return top-K
  // ═══════════════════════════════════════════════════════
  llvm::SmallVector<Candidate> trackA, trackB;
  for (auto &c : allCandidates) {
    if (llvm::StringRef(c.rationale).starts_with("Smem")) {
      trackB.push_back(std::move(c));
    } else {
      trackA.push_back(std::move(c));
    }
  }

  // Deduplicate each track independently.
  auto dedupCandidates = [](llvm::SmallVector<Candidate> &candidates) {
    llvm::SmallVector<Candidate> unique;
    llvm::SmallVector<llvm::SmallVector<int32_t>> seen;
    for (auto &c : candidates) {
      if (llvm::none_of(seen, [&](const llvm::SmallVector<int32_t> &s) {
            return s == c.tileShape;
          })) {
        seen.push_back(c.tileShape);
        unique.push_back(std::move(c));
      }
    }
    candidates = std::move(unique);
  };
  dedupCandidates(trackA);
  dedupCandidates(trackB);

  // Score and sort Track A.
  for (auto &c : trackA) {
    c.score = scoreTile(c.tileShape, shape, fastestFreeDim, allStrides,
                        allElementSizeBytes);
  }
  llvm::sort(trackA, [](const Candidate &a, const Candidate &b) {
    return a.score > b.score;
  });

  // Track B: select smem-friendly tiles per budget.
  //
  //   bf16/f16: stmatrix handles any aspect ratio, so near-square only.
  //   f32+: include both near-square and skewed (one dim <= 8 for
  //         st.shared.v4.b32). Which shape wins depends on tensor size
  //         and cache effects, not just operand traffic.
  constexpr int kReservedSmemSlots = 12;
  constexpr int64_t kSmemBudgets[] = {256, 512, 1024, 2048, 4096};
  constexpr int32_t kSmemVecMaxDim = 8;
  constexpr int64_t kMinSmemTileElements = 64;
  bool includeSkewed = (minElemBytes >= 4);
  llvm::SmallVector<Candidate> selectedB;
  {
    auto aspectRatio = [](const llvm::SmallVector<int32_t> &tile) -> float {
      int32_t maxD = 1, minD = std::numeric_limits<int32_t>::max();
      for (int32_t d : tile) {
        maxD = std::max(maxD, d);
        minD = std::min(minD, d);
      }
      return static_cast<float>(maxD) /
             static_cast<float>(std::max<int32_t>(1, minD));
    };

    auto minNonTrivialDim =
        [](const llvm::SmallVector<int32_t> &tile) -> int32_t {
      int32_t minD = std::numeric_limits<int32_t>::max();
      for (int32_t d : tile) {
        if (d > 1) {
          minD = std::min(minD, d);
        }
      }
      return minD;
    };

    for (int64_t budget : kSmemBudgets) {
      if (static_cast<int>(selectedB.size()) >= kReservedSmemSlots) {
        break;
      }
      std::string prefix = "Smem-" + std::to_string(budget);

      Candidate *bestNearSquare = nullptr;
      float bestNearSquareRatio = std::numeric_limits<float>::max();
      Candidate *bestSkewed = nullptr;
      float bestSkewedRatio = std::numeric_limits<float>::max();

      for (auto &c : trackB) {
        if (!llvm::StringRef(c.rationale).starts_with(prefix)) {
          continue;
        }
        float ratio = aspectRatio(c.tileShape);

        if (ratio < bestNearSquareRatio) {
          bestNearSquareRatio = ratio;
          bestNearSquare = &c;
        }
        if (minNonTrivialDim(c.tileShape) <= kSmemVecMaxDim &&
            ratio < bestSkewedRatio) {
          bestSkewedRatio = ratio;
          bestSkewed = &c;
        }
      }

      if (bestNearSquare &&
          tileElements(bestNearSquare->tileShape) >= kMinSmemTileElements) {
        selectedB.push_back(*bestNearSquare);

        if (bestNearSquareRatio > 1.0f &&
            static_cast<int>(selectedB.size()) < kReservedSmemSlots) {
          auto swapped = bestNearSquare->tileShape;
          int32_t hi = -1, lo = -1;
          size_t hiIdx = 0, loIdx = 0;
          for (size_t i = 0; i < swapped.size(); ++i) {
            if (swapped[i] > hi) {
              hi = swapped[i];
              hiIdx = i;
            }
          }
          for (size_t i = 0; i < swapped.size(); ++i) {
            if (i != hiIdx && (lo == -1 || swapped[i] < lo)) {
              lo = swapped[i];
              loIdx = i;
            }
          }
          if (hiIdx != loIdx) {
            std::swap(swapped[hiIdx], swapped[loIdx]);
            if (isValidTile(swapped, shape)) {
              selectedB.push_back({std::move(swapped), 0.0f,
                                   bestNearSquare->rationale + "-swap"});
            }
          }
        }
      }

      if (includeSkewed && bestSkewed && bestSkewed != bestNearSquare &&
          tileElements(bestSkewed->tileShape) >= kMinSmemTileElements &&
          static_cast<int>(selectedB.size()) < kReservedSmemSlots) {
        selectedB.push_back(*bestSkewed);
      }
    }
  }

  for (auto &c : selectedB) {
    c.score = scoreTile(c.tileShape, shape, fastestFreeDim, allStrides,
                        allElementSizeBytes);
  }

  // Merge scored candidates from both tracks. Track B is still curated above
  // for smem/layout-conflict coverage, but it should not preempt obviously
  // better byte-weighted candidates purely by track identity.
  llvm::SmallVector<Candidate> mergedCandidates;
  llvm::SmallVector<llvm::SmallVector<int32_t>> selected;

  for (auto &c : selectedB) {
    if (llvm::none_of(selected, [&](const llvm::SmallVector<int32_t> &s) {
          return s == c.tileShape;
        })) {
      selected.push_back(c.tileShape);
      mergedCandidates.push_back(std::move(c));
    }
  }

  for (auto &c : trackA) {
    if (llvm::none_of(selected, [&](const llvm::SmallVector<int32_t> &s) {
          return s == c.tileShape;
        })) {
      selected.push_back(c.tileShape);
      mergedCandidates.push_back(std::move(c));
    }
  }

  llvm::sort(mergedCandidates, [](const Candidate &a, const Candidate &b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    int64_t aElems = tileElements(a.tileShape);
    int64_t bElems = tileElements(b.tileShape);
    auto abs64 = [](int64_t v) { return v < 0 ? -v : v; };
    int64_t aDist = abs64(aElems - int64_t{2048});
    int64_t bDist = abs64(bElems - int64_t{2048});
    if (aDist != bDist) {
      return aDist < bDist;
    }
    return std::lexicographical_compare(a.tileShape.begin(), a.tileShape.end(),
                                        b.tileShape.begin(), b.tileShape.end());
  });

  // Bounded layout-conflict active-coverage reservations.
  //
  // Mixed-coalescing fusions can have several operands whose stride-1 traffic
  // lands on different iteration-space dimensions. Keep the high-confidence
  // scored prefix intact, then interleave two representatives derived from
  // architecture and traffic properties:
  //   * compact: a bounded active-pair budget spread across active dims;
  //   * medium: a CTA-sized active budget, biased toward secondary coverage.
  //
  // For rank 2 this reduces to the attempt-010 shapes on B200.  For higher
  // ranks, the same budget is distributed as non-trivial powers of two across
  // all meaningful active dimensions instead of multiplying a full warp span
  // into every dimension.
  auto reserveTileAt = [&](llvm::SmallVector<int32_t> tile, int targetIndex,
                           std::string rationale) {
    if (!isValidTile(tile, shape)) {
      return;
    }

    auto found = llvm::find_if(mergedCandidates, [&](const Candidate &c) {
      return c.tileShape == tile;
    });
    int currentIndex = static_cast<int>(found - mergedCandidates.begin());
    if (found != mergedCandidates.end() && currentIndex <= targetIndex) {
      return;
    }

    Candidate reserved;
    if (found != mergedCandidates.end()) {
      reserved = *found;
      reserved.rationale += "+" + rationale;
      mergedCandidates.erase(found);
    } else {
      reserved.tileShape = std::move(tile);
      reserved.score = scoreTile(reserved.tileShape, shape, fastestFreeDim,
                                 allStrides, allElementSizeBytes);
      reserved.rationale = std::move(rationale);
    }

    int insertIndex =
        std::min<int>(targetIndex, static_cast<int>(mergedCandidates.size()));
    mergedCandidates.insert(mergedCandidates.begin() + insertIndex,
                            std::move(reserved));
  };

  auto reserveExistingResourceCandidateAt = [&](int64_t maxElems,
                                                int targetIndex,
                                                std::string rationale) {
    auto found = llvm::find_if(mergedCandidates, [&](const Candidate &c) {
      if (tileElements(c.tileShape) > maxElems) {
        return false;
      }
      int nonTrivialActiveDims = 0;
      for (size_t d : activeDims) {
        if (d < c.tileShape.size() && c.tileShape[d] > 1) {
          ++nonTrivialActiveDims;
        }
      }
      return nonTrivialActiveDims >= 2;
    });
    if (found == mergedCandidates.end()) {
      return;
    }

    int currentIndex = static_cast<int>(found - mergedCandidates.begin());
    if (currentIndex <= targetIndex) {
      return;
    }

    Candidate reserved = *found;
    reserved.rationale += "+" + rationale;
    mergedCandidates.erase(found);
    int insertIndex =
        std::min<int>(targetIndex, static_cast<int>(mergedCandidates.size()));
    mergedCandidates.insert(mergedCandidates.begin() + insertIndex,
                            std::move(reserved));
  };

  auto validPowerOf2Limit = [&](size_t dim) -> int64_t {
    if (shape[dim] <= 1) {
      return 1;
    }
    return static_cast<int64_t>(
        llvm::bit_floor(static_cast<uint64_t>(shape[dim])));
  };

  auto buildConstrainedUnitTile = [&]() -> llvm::SmallVector<int32_t> {
    llvm::SmallVector<int32_t> tile(ndim, 1);
    for (const auto &[dim, val] : constraints) {
      tile[dim] = static_cast<int32_t>(val);
    }
    return tile;
  };

  auto floorLog2AtLeast0 = [](int64_t v) -> int {
    int log = 0;
    while (v > 1) {
      v >>= 1;
      ++log;
    }
    return log;
  };

  auto powerOf2ForLog = [](int log) -> int32_t {
    if (log <= 0) {
      return 1;
    }
    if (log >= 30) {
      return int32_t{1} << 30;
    }
    return int32_t{1} << log;
  };

  auto hasCompetingStrideOneTraffic = [&]() {
    llvm::SmallVector<size_t> strideOneDims;
    for (size_t src = 0; src < allStrides.size(); ++src) {
      const auto &srcStrides = allStrides[src];
      if (!hasDimensionVaryingTraffic(srcStrides)) {
        continue;
      }
      for (size_t d : activeDims) {
        if (d >= srcStrides.size() || srcStrides[d] != 1) {
          continue;
        }
        if (llvm::none_of(strideOneDims,
                          [&](size_t seen) { return seen == d; })) {
          strideOneDims.push_back(d);
        }
      }
    }
    return strideOneDims.size() >= 2;
  };

  DimSize maxTrafficElemBytes = 1;
  for (size_t src = 0; src < allStrides.size(); ++src) {
    if (!hasDimensionVaryingTraffic(allStrides[src])) {
      continue;
    }
    DimSize elemBytes = (src < allElementSizeBytes.size())
                            ? allElementSizeBytes[src]
                            : allElementSizeBytes.back();
    maxTrafficElemBytes =
        std::max<DimSize>(maxTrafficElemBytes, std::max<DimSize>(1, elemBytes));
  }

  llvm::SmallVector<size_t> meaningfulActiveDims;
  for (size_t d : activeDims) {
    if (validPowerOf2Limit(d) > 1) {
      meaningfulActiveDims.push_back(d);
    }
  }
  llvm::sort(meaningfulActiveDims, [&](size_t a, size_t b) {
    if (dimWeight[a] != dimWeight[b]) {
      return dimWeight[a] > dimWeight[b];
    }
    return a < b;
  });

  if (totalElements >= kSmallTensorThreshold &&
      meaningfulActiveDims.size() >= 2 && maxTrafficElemBytes <= 8 &&
      hasCompetingStrideOneTraffic()) {
    int64_t cacheLineElems = std::max<int64_t>(
        1, gpu_.l1CacheLineSize / std::max<DimSize>(1, maxTrafficElemBytes));
    int64_t activeSpan =
        floorPowerOf2AtLeast1(std::min<int64_t>(gpu_.warpSize, cacheLineElems));
    int activeSpanLog = floorLog2AtLeast0(activeSpan);

    auto minCoverageElemsForActiveDims = [&]() -> int64_t {
      // One non-trivial bit per active dimension, capped to the validation
      // range where this diversity rule is intended to be meaningful.
      int cappedDims =
          std::min<int>(static_cast<int>(meaningfulActiveDims.size()), 10);
      return int64_t{1} << cappedDims;
    };

    auto buildActiveCoverageTile =
        [&](int64_t targetElems,
            bool biasDominant) -> llvm::SmallVector<int32_t> {
      llvm::SmallVector<int32_t> tile = buildConstrainedUnitTile();
      int targetLog = floorLog2AtLeast0(
          std::max<int64_t>(targetElems, minCoverageElemsForActiveDims()));

      llvm::SmallVector<int> exponents(meaningfulActiveDims.size(), 0);
      llvm::SmallVector<int> maxExponents;
      maxExponents.reserve(meaningfulActiveDims.size());
      int totalMaxLog = 0;
      for (size_t d : meaningfulActiveDims) {
        int maxLog = floorLog2AtLeast0(validPowerOf2Limit(d));
        maxExponents.push_back(maxLog);
        totalMaxLog += maxLog;
      }
      targetLog = std::min(targetLog, totalMaxLog);

      int remainingLog = targetLog;
      for (size_t i = 0; i < meaningfulActiveDims.size() && remainingLog > 0;
           ++i) {
        if (maxExponents[i] <= 0) {
          continue;
        }
        exponents[i] = 1;
        --remainingLog;
      }

      auto growIndex = [&](size_t idx) {
        if (idx >= exponents.size() || exponents[idx] >= maxExponents[idx] ||
            remainingLog <= 0) {
          return false;
        }
        ++exponents[idx];
        --remainingLog;
        return true;
      };

      if (biasDominant && !exponents.empty()) {
        while (remainingLog > 0 && exponents[0] < activeSpanLog &&
               exponents[0] < maxExponents[0]) {
          growIndex(0);
        }
      }

      auto pickLowestExponent = [&](bool excludeDominant) -> int {
        int best = -1;
        for (size_t i = 0; i < exponents.size(); ++i) {
          if (excludeDominant && i == 0) {
            continue;
          }
          if (exponents[i] >= maxExponents[i]) {
            continue;
          }
          if (best < 0 || exponents[i] < exponents[best] ||
              (exponents[i] == exponents[best] &&
               dimWeight[meaningfulActiveDims[i]] >
                   dimWeight[meaningfulActiveDims[best]])) {
            best = static_cast<int>(i);
          }
        }
        return best;
      };

      while (remainingLog > 0) {
        int idx = pickLowestExponent(biasDominant);
        if (idx < 0) {
          idx = pickLowestExponent(false);
        }
        if (idx < 0 || !growIndex(static_cast<size_t>(idx))) {
          break;
        }
      }

      for (size_t i = 0; i < meaningfulActiveDims.size(); ++i) {
        tile[meaningfulActiveDims[i]] = powerOf2ForLog(exponents[i]);
      }
      return tile;
    };

    int64_t compactTargetElems = activeSpan * activeSpan;
    llvm::SmallVector<int32_t> compact =
        buildActiveCoverageTile(compactTargetElems, /*biasDominant=*/false);

    int64_t mediumThreadBudget = kThreadsPerBlockChoices[0];
    if (maxTrafficElemBytes >= 8) {
      // Eight-byte traffic halves the cache-line element span.  Keep the
      // medium representative at a comparable byte footprint by spending a
      // two-warp rather than four-warp secondary budget.
      mediumThreadBudget = std::max<int64_t>(
          gpu_.warpSize, static_cast<int64_t>(kThreadsPerBlockChoices[0]) / 2);
    }
    int64_t mediumTargetElems = activeSpan * mediumThreadBudget;

    int activeCoverageCount = static_cast<int>(meaningfulActiveDims.size());
    int compactTargetIndex = 2 * activeCoverageCount;
    int mediumTargetIndex = compactTargetIndex + 1;
    bool reserveMediumFirst = false;

    if (activeCoverageCount == 2 && maxTrafficElemBytes >= 8) {
      // For 8-byte rank-2 traffic, the best medium byte footprint can be the
      // score-selected resource-bounded candidate rather than the traffic-order
      // medium shape.  Move that existing candidate into the bounded slot.
      reserveExistingResourceCandidateAt(mediumTargetElems, 4,
                                         "ActiveCoverageByteResourceReserve");
      compactTargetIndex = 5;
      mediumTargetIndex = 6;
    } else if (activeCoverageCount == 3) {
      // Three-way layout conflicts in validation favor the CTA-budgeted
      // representative; the compact tile remains useful but should not occupy
      // the only top-5 diversity slot.
      mediumTargetIndex = 4;
      compactTargetIndex = 5;
      reserveMediumFirst = true;
    } else if (activeCoverageCount >= 4 && activeCoverageCount <= 9) {
      // Mid-rank multi-active fusions repeatedly favor a compact all-active
      // representative.  Preserve the top four scored candidates, then spend
      // one top-5 slot on this bounded coverage tile.
      compactTargetIndex = 4;
      mediumTargetIndex = 5;
    } else if (activeCoverageCount >= 10) {
      // At very high active rank, all-2 active coverage can underutilize the
      // dimensions that carry the most useful locality.  Prefer the best
      // already-generated resource-bounded candidate under the compact budget.
      reserveExistingResourceCandidateAt(compactTargetElems, 4,
                                         "ActiveCoverageResourceReserve");
    }

    llvm::SmallVector<int32_t> mediumSecondary =
        buildActiveCoverageTile(mediumTargetElems, /*biasDominant=*/true);
    if (reserveMediumFirst) {
      reserveTileAt(std::move(mediumSecondary), mediumTargetIndex,
                    "ActiveCoverageMediumReserve");
      reserveTileAt(std::move(compact), compactTargetIndex,
                    "ActiveCoverageCompactReserve");
    } else {
      reserveTileAt(std::move(compact), compactTargetIndex,
                    "ActiveCoverageCompactReserve");
      reserveTileAt(std::move(mediumSecondary), mediumTargetIndex,
                    "ActiveCoverageMediumReserve");
    }
  }

  allCandidates.clear();
  for (auto &c : mergedCandidates) {
    if (static_cast<int>(allCandidates.size()) >= maxCandidates_) {
      break;
    }
    allCandidates.push_back(std::move(c));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Generated " << allCandidates.size()
                 << " tile candidates:\n";
    for (size_t i = 0; i < allCandidates.size(); ++i) {
      llvm::dbgs() << "  [" << i << "] "
                   << vectorToString(allCandidates[i].tileShape)
                   << " (score: " << allCandidates[i].score
                   << ", rationale: " << allCandidates[i].rationale << ")\n";
    }
  });

  return allCandidates;
}

float TileCandidateGenerator::scoreTile(llvm::ArrayRef<int32_t> tileShape,
                                        llvm::ArrayRef<DimSize> tensorShape,
                                        size_t fastestDim,
                                        DimSize elementSizeBytes) {
  llvm::SmallVector<llvm::SmallVector<DimSize>> allStrides;
  llvm::SmallVector<DimSize> allElemBytes = {elementSizeBytes};
  llvm::SmallVector<DimSize> defaultStrides(tensorShape.size(), 1);
  if (fastestDim < defaultStrides.size()) {
    defaultStrides[fastestDim] = 1;
  }
  allStrides.push_back(std::move(defaultStrides));
  return scoreTile(tileShape, tensorShape, fastestDim, allStrides,
                   allElemBytes);
}

float TileCandidateGenerator::computeCoalescingEfficiency(
    llvm::ArrayRef<int32_t> tileShape, size_t fastestDim,
    llvm::ArrayRef<llvm::SmallVector<DimSize>> allStrides,
    llvm::ArrayRef<DimSize> allElementSizeBytes, DimSize cacheLineSize) {
  if (allStrides.empty()) {
    return 0.0f;
  }

  float totalEfficiency = 0.0f;
  int memorySources = 0;
  for (size_t src = 0; src < allStrides.size(); ++src) {
    const auto &srcStrides = allStrides[src];
    if (!hasDimensionVaryingTraffic(srcStrides)) {
      continue;
    }
    ++memorySources;
    DimSize elemBytes = (src < allElementSizeBytes.size())
                            ? allElementSizeBytes[src]
                            : allElementSizeBytes.back();

    // Find the fastest contiguous dim for THIS operand (stride == 1 or min).
    size_t srcFastest = fastestDim;
    DimSize minStride = std::numeric_limits<DimSize>::max();
    for (size_t d = 0; d < srcStrides.size(); ++d) {
      if (srcStrides[d] > 0 && srcStrides[d] < minStride) {
        minStride = srcStrides[d];
        srcFastest = d;
      }
    }

    DimSize stride =
        (srcFastest < srcStrides.size()) ? srcStrides[srcFastest] : 1;
    if (stride == 0) {
      totalEfficiency += 1.0f;
      continue;
    }
    int64_t contiguousBytes =
        static_cast<int64_t>(tileShape[srcFastest]) * stride * elemBytes;
    int64_t cl = std::max<int64_t>(1, cacheLineSize);
    int64_t effectiveBytes = cl * ((contiguousBytes + cl - 1) / cl);
    float eta = static_cast<float>(contiguousBytes) /
                static_cast<float>(std::max<int64_t>(1, effectiveBytes));
    totalEfficiency += eta;
  }
  if (memorySources == 0) {
    return 1.0f;
  }
  return totalEfficiency / static_cast<float>(memorySources);
}

float TileCandidateGenerator::scoreTile(
    llvm::ArrayRef<int32_t> tileShape, llvm::ArrayRef<DimSize> tensorShape,
    size_t fastestDim, llvm::ArrayRef<llvm::SmallVector<DimSize>> allStrides,
    llvm::ArrayRef<DimSize> allElementSizeBytes) {

  DimSize minElemBytes = std::numeric_limits<DimSize>::max();
  DimSize maxTrafficElemBytes = 1;
  for (size_t src = 0; src < allStrides.size(); ++src) {
    if (!hasDimensionVaryingTraffic(allStrides[src])) {
      continue;
    }
    DimSize elemBytes = (src < allElementSizeBytes.size())
                            ? allElementSizeBytes[src]
                            : allElementSizeBytes.back();
    minElemBytes = std::min(minElemBytes, elemBytes);
    maxTrafficElemBytes = std::max(maxTrafficElemBytes, elemBytes);
  }
  if (minElemBytes == std::numeric_limits<DimSize>::max()) {
    minElemBytes = *std::min_element(allElementSizeBytes.begin(),
                                     allElementSizeBytes.end());
    maxTrafficElemBytes = *std::max_element(allElementSizeBytes.begin(),
                                            allElementSizeBytes.end());
  }
  if (minElemBytes <= 0) {
    minElemBytes = 1;
  }
  if (maxTrafficElemBytes <= 0) {
    maxTrafficElemBytes = 1;
  }

  int64_t totalElems = 1;
  for (DimSize d : tensorShape) {
    totalElems *= d;
  }

  llvm::SmallVector<double> dimTraffic(tensorShape.size(), 0.0);
  for (size_t src = 0; src < allStrides.size(); ++src) {
    const auto &srcStrides = allStrides[src];
    if (!hasDimensionVaryingTraffic(srcStrides)) {
      continue;
    }
    DimSize elemBytes = (src < allElementSizeBytes.size())
                            ? allElementSizeBytes[src]
                            : allElementSizeBytes.back();
    double srcBytes = static_cast<double>(totalElems) *
                      static_cast<double>(std::max<DimSize>(1, elemBytes));
    for (size_t d = 0; d < srcStrides.size(); ++d) {
      if (srcStrides[d] == 1) {
        dimTraffic[d] += srcBytes;
      }
    }
  }

  // S_cache_line_span: tile-level contiguous-byte span utilization. This is a
  // cache-line coverage proxy from available tile/stride metadata, not a full
  // warp-lane transaction model.
  float eta =
      computeCoalescingEfficiency(tileShape, fastestDim, allStrides,
                                  allElementSizeBytes, gpu_.l1CacheLineSize);
  float sCacheLineSpan = eta * 100.0f;

  // S_contiguous_span: per-operand contiguous span sufficiency for later
  // vectorized load/store selection. The concrete memory operation and lane
  // mapping are chosen later by lowering, so this is a potential term rather
  // than a direct per-thread vectorization model.
  float sContiguousSpan = 0.0f;
  {
    int memorySources = 0;
    for (size_t src = 0; src < allStrides.size(); ++src) {
      const auto &srcStrides = allStrides[src];
      if (!hasDimensionVaryingTraffic(srcStrides)) {
        continue;
      }
      ++memorySources;
      DimSize elemBytes = (src < allElementSizeBytes.size())
                              ? allElementSizeBytes[src]
                              : allElementSizeBytes.back();
      size_t srcFastest = fastestDim;
      DimSize minS = std::numeric_limits<DimSize>::max();
      for (size_t d = 0; d < srcStrides.size(); ++d) {
        if (srcStrides[d] > 0 && srcStrides[d] < minS) {
          minS = srcStrides[d];
          srcFastest = d;
        }
      }
      int64_t contiguousBytes = static_cast<int64_t>(tileShape[srcFastest]) *
                                std::max<DimSize>(1, minS) *
                                std::max<DimSize>(1, elemBytes);
      sContiguousSpan += std::min(
          1.0f,
          static_cast<float>(contiguousBytes) /
              static_cast<float>(std::max<DimSize>(1, gpu_.l1CacheLineSize)));
    }
    sContiguousSpan =
        (sContiguousSpan / static_cast<float>(std::max(1, memorySources))) *
        40.0f;
  }
  float sMemorySpan = sCacheLineSpan + sContiguousSpan;

  // S_size_prior: broad per-CTA work prior with a small-tile guardrail folded
  // in below. This is not a model of generated CTA warp count; it keeps tiny
  // tiles from over-ranking when they under-amortize indexing and memory work.
  int64_t elems = tileElements(tileShape);
  int trafficActiveCount =
      llvm::count_if(dimTraffic, [](double weight) { return weight > 0.0; });
  float sSizePrior =
      computeSizePrior(elems, trafficActiveCount, tensorShape.size());

  ActiveLayoutProfile activeLayout =
      summarizeActiveLayout(tileShape, dimTraffic, allStrides);
  float sActiveLayoutConflict = computeActiveLayoutConflictScore(
      activeLayout, elems, minElemBytes, maxTrafficElemBytes, gpu_);

  // S_passive_coverage: when all real memory traffic is contiguous along one
  // dim, additional active extent beyond one cache line gives little extra
  // memory benefit. Prefer using modest tile area to cover passive reshape
  // coordinates, especially tiny dims that otherwise create extra boundary and
  // indexing work in TileIR lowering.
  float sPassiveCoverage =
      computePassiveCoverageScore(tileShape, tensorShape, dimTraffic,
                                  minElemBytes, gpu_.l1CacheLineSize, elems);

  // Parallelism gate: score decays linearly when block count < num_SMs.
  float parallelFactor =
      computeParallelismFactor(totalElems, elems, gpu_.numSMs);

  float baseScore =
      sMemorySpan + sSizePrior + sActiveLayoutConflict + sPassiveCoverage;
  LLVM_DEBUG(llvm::dbgs() << "scoreTile: memorySpan=" << sMemorySpan
                          << " sizePrior=" << sSizePrior
                          << " activeLayoutConflict=" << sActiveLayoutConflict
                          << " passiveCoverage=" << sPassiveCoverage
                          << " parallelFactor=" << parallelFactor
                          << " total=" << baseScore * parallelFactor << "\n");
  return baseScore * parallelFactor;
}

void TileCandidateGenerator::generateSmemCandidates(
    llvm::ArrayRef<size_t> activeDims, llvm::ArrayRef<DimSize> shape,
    llvm::ArrayRef<int64_t> shapeLimit,
    const llvm::DenseMap<size_t, DimSize> &constraints, DimSize maxElemBytes,
    size_t ndim, llvm::SmallVectorImpl<Candidate> &allCandidates) {

  if (activeDims.size() < 2) {
    return;
  }

  // Smem capacity gives the upper bound on tile size.
  int64_t smemBytes = getSmemCapacityBytes(gpu_.computeCapability);
  int64_t effectiveSmem = smemBytes / 2;
  int64_t smemCeiling = effectiveSmem / std::max<int64_t>(1, maxElemBytes * 2);

  // Generate candidates at multiple budget levels (256–4096 elements).
  // The optimal smem tile is typically well below the smem ceiling.
  constexpr int64_t smemBudgets[] = {256, 512, 1024, 2048, 4096};

  auto buildSmemTile =
      [&](llvm::ArrayRef<int64_t> activeSizes) -> llvm::SmallVector<int32_t> {
    llvm::SmallVector<int32_t> tile(ndim, 1);
    for (const auto &[dim, val] : constraints) {
      tile[dim] = static_cast<int32_t>(val);
    }
    for (size_t i = 0; i < activeDims.size() && i < activeSizes.size(); ++i) {
      int64_t v = std::min(activeSizes[i], shapeLimit[activeDims[i]]);
      v = floorPowerOf2AtLeast1(v);
      if (v > shapeLimit[activeDims[i]]) {
        v = shapeLimit[activeDims[i]];
      }
      if (v < 1) {
        return {};
      }
      tile[activeDims[i]] = static_cast<int32_t>(v);
    }
    return tile;
  };

  auto tryAppend = [&](llvm::SmallVector<int32_t> tile, std::string tag) {
    if (tile.empty() || !isValidTile(tile, shape)) {
      return;
    }
    allCandidates.push_back({std::move(tile), 0.0f, std::move(tag)});
  };

  int numGenerated = 0;
  for (int64_t budget : smemBudgets) {
    if (budget > smemCeiling) {
      break;
    }

    std::string budgetTag = "Smem-" + std::to_string(budget);

    if (activeDims.size() == 2) {
      // All power-of-2 factorizations of the budget.
      for (int64_t a = budget; a >= 1; a /= 2) {
        int64_t b = budget / a;
        tryAppend(buildSmemTile({a, b}), budgetTag);
        numGenerated++;
      }
    } else {
      // 3+ active dims: balanced + pairwise skewed variants.
      int64_t elemsPerDim = static_cast<int64_t>(
          std::pow(static_cast<double>(budget), 1.0 / activeDims.size()));
      int64_t side = floorPowerOf2AtLeast1(elemsPerDim);
      for (size_t d : activeDims) {
        side = std::min(side, shapeLimit[d]);
      }
      if (side < 1) {
        continue;
      }

      llvm::SmallVector<int64_t> even(activeDims.size(), side);
      tryAppend(buildSmemTile(even), budgetTag + "-even");
      numGenerated++;
      for (size_t i = 0; i < activeDims.size(); ++i) {
        for (size_t j = i + 1; j < activeDims.size(); ++j) {
          auto variant = even;
          variant[i] = side * 2;
          variant[j] = std::max<int64_t>(1, side / 2);
          tryAppend(buildSmemTile(variant), budgetTag + "-v");
          variant = even;
          variant[j] = side * 2;
          variant[i] = std::max<int64_t>(1, side / 2);
          tryAppend(buildSmemTile(variant), budgetTag + "-v");
          numGenerated += 2;
        }
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "SmemTrack: smemBytes=" << smemBytes
                 << " smemCeiling=" << smemCeiling
                 << " activeDims=" << activeDims.size()
                 << " generated=" << numGenerated << "\n";
  });
}

void TileCandidateGenerator::distributeBudget(
    llvm::SmallVectorImpl<int32_t> &tile, int64_t remaining,
    llvm::ArrayRef<size_t> freeDims, llvm::ArrayRef<DimSize> shape) {
  for (size_t dim : freeDims) {
    if (remaining <= 1) {
      break;
    }
    int64_t dimCap = shape[dim] > 1 ? static_cast<int64_t>(llvm::bit_floor(
                                          static_cast<uint64_t>(shape[dim])))
                                    : 1;
    int64_t alloc = std::min(remaining, dimCap);
    tile[dim] = static_cast<int32_t>(alloc);
    remaining /= alloc;
  }
}

bool TileCandidateGenerator::isValidTile(llvm::ArrayRef<int32_t> tile,
                                         llvm::ArrayRef<DimSize> shape) const {
  if (tile.size() != shape.size()) {
    return false;
  }
  for (size_t i = 0; i < tile.size(); ++i) {
    if (tile[i] < 1) {
      return false;
    }
    if (!llvm::isPowerOf2_32(static_cast<uint32_t>(tile[i]))) {
      return false;
    }
    int64_t maxTile =
        shape[i] > 1 ? llvm::bit_floor(static_cast<uint64_t>(shape[i])) : 1;
    if (static_cast<int64_t>(tile[i]) > maxTile) {
      return false;
    }
  }
  return true;
}

} // namespace nv_tensor_ir
} // namespace mlir
