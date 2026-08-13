// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/TileAnalyzer.h"

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <numeric>

#define DEBUG_TYPE "tile-analyzer"
namespace tcg = mlir::nv_tensor_ir::tcutegen;

namespace mlir {
namespace nv_tensor_ir {

namespace {

/**
 * @brief Get dimensions ordered from fastest to slowest based on stride
 * values.
 * @param tensorStrides Strides of the tensor
 * @return Ordered dimension indices
 *
 * A smaller stride indicates faster dimension (elements closer in memory).
 * The returned vector contains dimension indices where:
 * vec[0] is the fastest dimension (smallest stride)
 * vec[vec.size()-1] is the slowest dimension (largest stride)
 */
llvm::SmallVector<size_t>
getDimensionOrder(llvm::ArrayRef<DimSize> tensorStrides) {
  llvm::SmallVector<size_t> order(tensorStrides.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t i, size_t j) {
    return tensorStrides[i] < tensorStrides[j];
  });
  return order;
}

enum TilingStrategy {
  PointwiseBaseline,
  Strategy1,
};

// Use the pointwise baseline strategy until we will have enough benchmarking
// and data to select better strategies.
constexpr TilingStrategy tilingStrategy = PointwiseBaseline;

// A simple strategy, which provides good performance for pointwise kernels.
//
// It makes tiles of size <= 128 ("one element for each CUDA core of an SM"),
// like XLA's MLIR emitter does. This should be a good baseline for more
// advanced strategies. This results in good occupancy: we use as many SMs as
// possible and each core does something. This always returns tile sizes where
// each dimension is a power of 2.
//
// This strategy tries to build a contiguous tile by first growing the
// fastest-varying dimension and then growing the slower dimensions, until we
// reach the desired size.
//
// For example, for a row-major tensor, this could make the following tile
// sizes:
// - tensorShape: [1024, 1024] -> tileSize: [1, 128]
// - tensorShape: [16, 16, 16] -> tileSize: [1, 8, 16]
// - tensorShape: [16, 3, 16] -> tileSize: [2, 4, 16]
llvm::SmallVector<DimSize>
GetTileSize_PointwiseBaseline(llvm::ArrayRef<DimSize> tensorShape,
                              llvm::ArrayRef<size_t> dimOrder,
                              llvm::ArrayRef<size_t> reductionDims = {}) {
  llvm::SmallVector<DimSize> tileSize(tensorShape.size(), 0);

  // On Blackwell and Hopper, there are 128 CUDA cores per SM, and we want to
  // handle 4 elements per thread, so we need 512 elements per tile.
  const int maxTileSizePow2 = 9; // 512=4*128=2^9
  int remainingTileSizePow2 = maxTileSizePow2;

  // Iterating dims in fastest to slowest-varying order.
  for (const size_t dim : dimOrder) {
    // For reduction dimensions, use the input shape as tile size
    if (std::find(reductionDims.begin(), reductionDims.end(), dim) !=
        reductionDims.end()) {
      tileSize[dim] = tensorShape[dim];
      continue;
    }

    // For non-reduction dimensions, calculate tile size as before
    tileSize[dim] = 1;
    while (tileSize[dim] < tensorShape[dim] && remainingTileSizePow2 > 0) {
      tileSize[dim] *= 2;
      remainingTileSizePow2 -= 1;
    }
  }

  return tileSize;
}

namespace strategy1 {

/*
Tile Size Analyzer for GPU Tensor Operations

This algorithm determines tile sizes for tensor operations on GPUs, focusing on
pointwise operations. The process follows these steps:

1. Dimension Ordering:
   - Sort dimensions based on stride values (smallest to largest)
   - Identifies the fastest-changing dimension for memory access optimization

2. Tile Size Selection:
   For fastest-changing dimension:
   - Start with warp size (32)
   - Allow growth up to 128 elements or dimension size
   - Maintain warp alignment for potential memory coalescing

   For other dimensions:
   - Target 64 elements (2 warps) to amortize setup costs
   - Align to warp size only when beneficial
   - Avoid unnecessary padding that could limit parallelism

Key Principles:
- Tile represents a logical unit of computation
- Compiler can map a tile to one or more thread blocks
- Single thread block doesn't process multiple tiles
- Avoid oversized tiles that cause scheduling inefficiency
- Maintain enough elements per tile to amortize setup costs

Design Considerations:
1. Memory Access:
   - Fastest dimension prioritizes memory access efficiency
   - Warp alignment where beneficial for potential coalescing
   - Compiler determines final memory access patterns

2. Scheduling Efficiency:
   - Large tiles can lead to underutilization in last wave
   - Small tiles increase relative overhead costs
   - Moderate sizes enable flexible compiler optimization

3. Parallelism:
   - Compiler decides elements per thread
   - Tile size doesn't directly correspond to thread count
   - Focus on enabling efficient execution patterns

This approach provides simple but effective guidelines for tile size selection
while giving the compiler flexibility in mapping computation to hardware
resources.
*/

/// @brief Maximum tile dimension to avoid oversized scheduling units
constexpr DimSize MAX_TILE_DIM = 128;

/// @brief Minimum tile dimension for setup cost amortization
constexpr DimSize MIN_TILE_DIM = 8;

/// @brief Warp size for alignment considerations
constexpr DimSize WARP_MULTIPLE = 32;

/**
 * @brief Align size to warp multiple if beneficial
 * @param size Size to align
 * @return Aligned size
 */
DimSize alignToWarpSize(DimSize size) {
  if (size > WARP_MULTIPLE / 2) {
    return ((size + WARP_MULTIPLE - 1) / WARP_MULTIPLE) * WARP_MULTIPLE;
  }
  return size;
}

/**
 * @brief Calculate appropriate tile size for a dimension
 * @param dimSize Size of the dimension
 * @param isFastestDim Whether this is the fastest-changing dimension
 * @return Calculated tile size
 */
DimSize calculateTileSize(DimSize dimSize, bool isFastestDim) {
  if (dimSize == 1) {
    return 1;
  }

  // First, ensure we don't exceed dimension size
  DimSize maxSize = std::min(dimSize, MAX_TILE_DIM);

  if (isFastestDim) {
    // Start with dimension size if smaller than MIN_TILE_DIM
    DimSize size = std::min(maxSize, MIN_TILE_DIM);
    while (size * 2 <= maxSize) {
      size *= 2;
    }
    return alignToWarpSize(size);
  }

  // For other dims, target 2 warps worth of elements
  DimSize target = std::min(dimSize, WARP_MULTIPLE * 2);
  return alignToWarpSize(target);
}

llvm::SmallVector<DimSize>
GetTileSize_Strategy1(llvm::ArrayRef<DimSize> tensorShape,
                      llvm::ArrayRef<size_t> dimOrder,
                      llvm::ArrayRef<size_t> reductionDims = {}) {
  llvm::SmallVector<DimSize> tileSize(tensorShape.size());

  for (size_t i = 0; i < tensorShape.size(); ++i) {
    // For reduction dimensions, use the input shape as tile size
    if (llvm::is_contained(reductionDims, i)) {
      tileSize[i] = tensorShape[i];
      continue;
    }

    // For non-reduction dimensions, calculate tile size as before
    tileSize[i] = (tensorShape[i] == 1)
                      ? 1
                      : calculateTileSize(tensorShape[i], i == dimOrder[0]);
  }

  // Ensure that all dimensions of the tile are powers of 2.
  for (DimSize &dimSize : tileSize) {
    dimSize = llvm::PowerOf2Ceil(dimSize);
  }

  return tileSize;
}

} // namespace strategy1

/**
 * @brief Get tile sizes based on dimension characteristics
 * @param tensorShape Shape of the tensor
 * @param dimOrder Dimension order
 * @return Tile sizes
 */
llvm::SmallVector<DimSize>
getTileSize(llvm::ArrayRef<DimSize> tensorShape,
            llvm::ArrayRef<size_t> dimOrder,
            llvm::ArrayRef<size_t> reductionDims = {}) {
  switch (tilingStrategy) {
  case Strategy1:
    return strategy1::GetTileSize_Strategy1(tensorShape, dimOrder,
                                            reductionDims);
  case PointwiseBaseline: // fallthrough
  default:
    return GetTileSize_PointwiseBaseline(tensorShape, dimOrder, reductionDims);
  }
}

} // namespace

TileAnalyzer::TileAnalyzer(const GpuArchitecture &gpu) : gpu_(gpu) {}

FailureOr<llvm::SmallVector<DimSize>> TileAnalyzer::analyzeTileSize(
    llvm::ArrayRef<DimSize> tensorShape, llvm::ArrayRef<DimSize> tensorStrides,
    DimSize elementSizeBytes, llvm::ArrayRef<size_t> reductionDims) {
  if (!cachedTileSize_) {
    llvm::SmallVector<size_t> dimOrder = getDimensionOrder(tensorStrides);
    llvm::SmallVector<DimSize> tileSize =
        getTileSize(tensorShape, dimOrder, reductionDims);

    if (!isValidTileSize(tileSize, tensorShape, elementSizeBytes)) {
      return failure();
    }

    std::string explanation;
    llvm::raw_string_ostream os(explanation);
    os << "Dimension order (fastest to slowest): " << vectorToString(dimOrder)
       << "\n";
    os << "Reduction dimensions: " << vectorToString(reductionDims) << "\n";
    os << "Final tile size: " << vectorToString(tileSize) << "\n";

    cachedTileSize_ = tileSize;
    tileExplanation_ = explanation;
  }
  return *cachedTileSize_;
}

std::optional<std::string> TileAnalyzer::getTileExplanation() const {
  return tileExplanation_;
}

bool TileAnalyzer::isTileSizeCalculated() const {
  return cachedTileSize_.has_value();
}

bool TileAnalyzer::isValidTileSize(llvm::ArrayRef<DimSize> tileSize,
                                   llvm::ArrayRef<DimSize> tensorShape,
                                   DimSize elementSizeBytes) const {
  // Relatively dummy checker.
  if (tileSize.size() != tensorShape.size()) {
    return false;
  }

  for (size_t i = 0; i < tileSize.size(); ++i) {
    if (tensorShape[i] < 1) {
      return false;
    }
    if (tileSize[i] < 1 ||
        tileSize[i] > static_cast<int64_t>(llvm::PowerOf2Ceil(
                          static_cast<uint64_t>(tensorShape[i]))) ||
        !llvm::isPowerOf2_64(static_cast<uint64_t>(tileSize[i]))) {
      return false;
    }
  }

  return true;
}

namespace {

FailureOr<std::pair<mlir::nv_tensor_ir::TensorType, tcg::Stride>>
getTensorTypeAndStrideFromGraph(mlir::nv_tensor_ir::GraphOp graphOp) {
  // Prefer inputs when available
  if (!graphOp.getArgumentTypes().empty()) {
    const mlir::nv_tensor_ir::TensorType tensorType =
        mlir::dyn_cast<mlir::nv_tensor_ir::TensorType>(
            graphOp.getArgumentTypes()[0]);
    if (!tensorType) {
      return graphOp.emitError(
          "the graph's first argument must have a tensor type");
    }
    MLIR_ASSIGN_OR_RETURN(auto stride,
                          getStrideFromGraph(graphOp.getArgument(0)));
    return std::make_pair(tensorType, stride);
  }

  // Zero-input graphs: derive from first result
  if (!graphOp.getResultTypes().empty()) {
    const mlir::nv_tensor_ir::TensorType tensorType =
        mlir::dyn_cast<mlir::nv_tensor_ir::TensorType>(
            graphOp.getResultTypes()[0]);
    if (!tensorType) {
      return graphOp.emitError("the graph's first result must have a tensor "
                               "type");
    }
    MLIR_ASSIGN_OR_RETURN(auto stride,
                          getStrideFromGraph(graphOp.getResults()[0]));
    return std::make_pair(tensorType, stride);
  }

  return graphOp.emitError(
      "the graph must have at least one argument or result");
}

} // namespace

FailureOr<llvm::SmallVector<DimSize>>
analyzeTileSizeForGraph(TileAnalyzer &tileAnalyzer,
                        mlir::nv_tensor_ir::GraphOp graphOp) {
  MLIR_ASSIGN_OR_RETURN(auto typeAndStride,
                        getTensorTypeAndStrideFromGraph(graphOp));
  auto [tensorType, cgStride] = typeAndStride;

  const bool firstAnalysis = !tileAnalyzer.isTileSizeCalculated();
  auto shape = toSmallVector(getShapeRef(tensorType));
  auto stride = toSmallVector(cgStride);
  if (failed(shape) || failed(stride)) {
    return graphOp.emitError("cannot analyze tile size: the graph's tensor "
                             "shape or stride is not statically known");
  }
  int bytesPerElement =
      tensorType.getElementType().getIntOrFloatBitWidth() / CHAR_BIT;

  // Detect reduction dimensions by looking at direct operations in the graph
  // body
  llvm::SmallVector<size_t> reductionDims;
  LLVM_DEBUG(llvm::dbgs() << "Starting to look for reduction operations...\n");

  for (Operation &op : graphOp.getBody()->getOperations()) {
    if (auto reductionOp = mlir::dyn_cast<mlir::nv_tensor_ir::ReduceOp>(&op)) {
      for (int64_t dim : reductionOp.getDimensions()) {
        reductionDims.push_back(dim);
      }
    } else if (auto reductionUDOp =
                   mlir::dyn_cast<mlir::nv_tensor_ir::ReduceUDOp>(&op)) {
      for (int64_t dim : reductionUDOp.getDimensions()) {
        reductionDims.push_back(dim);
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Finished looking for reduction operations. Found "
                          << reductionDims.size()
                          << " reduction dimensions.\n");

  LLVM_DEBUG(llvm::dbgs() << "Analyzing tile size for shape: "
                          << vectorToString(*shape) << " with strides: "
                          << vectorToString(*stride) << "\n");
  FailureOr<llvm::SmallVector<DimSize>> tileSizeOr =
      tileAnalyzer.analyzeTileSize(*shape, *stride, bytesPerElement,
                                   reductionDims);
  if (failed(tileSizeOr)) {
    return graphOp.emitError(
               "could not determine a valid tile size for tensor shape ")
           << vectorToString(*shape);
  }
  const llvm::SmallVector<DimSize> &tileSize = *tileSizeOr;

  if (firstAnalysis) {
    LLVM_DEBUG({
      llvm::dbgs() << "Processing shape: " << vectorToString(*shape)
                   << " with strides: " << vectorToString(*stride) << "\n";
      llvm::dbgs() << "Tensor shape: " << vectorToString(*shape) << "\n";
      llvm::dbgs() << "Tensor strides: " << vectorToString(*stride) << "\n";
      llvm::dbgs() << "Reduction dimensions: " << vectorToString(reductionDims)
                   << "\n";
      if (std::optional<std::string> explanation =
              tileAnalyzer.getTileExplanation()) {
        llvm::dbgs() << "Tile size explanation:\n" << *explanation << "\n";
      }
      llvm::dbgs() << "Calculated tile sizes: " << vectorToString(tileSize)
                   << "\n";
      llvm::dbgs() << "Finished processing shape: " << vectorToString(*shape)
                   << "\n\n";
    });
  }

  return tileSize;
}

FailureOr<llvm::SmallVector<DimSize>>
calculateGridSizeForGraph(const TileAnalyzer &tileAnalyzer,
                          llvm::ArrayRef<DimSize> tileSize,
                          mlir::nv_tensor_ir::GraphOp graphOp) {
  MLIR_ASSIGN_OR_RETURN(auto typeAndStride,
                        getTensorTypeAndStrideFromGraph(graphOp));
  auto [tensorType, _] = typeAndStride;
  auto shape = toSmallVector(getShapeRef(tensorType));
  if (failed(shape)) {
    return graphOp.emitError("cannot compute grid size: the graph's tensor "
                             "shape is not statically known");
  }

  // For broadcast operations, use the output shape for grid size calculation
  llvm::SmallVector<DimSize> gridCalculationShape =
      *shape; // Default to input shape

  if (useOutputTensorShapeForGrid(graphOp)) {
    // Find the first output tensor to use its shape
    for (Type resType : graphOp.getFunctionType().getResults()) {
      if (auto outputType =
              mlir::dyn_cast<mlir::nv_tensor_ir::TensorType>(resType)) {
        auto outputShapeVector = toSmallVector(getShapeRef(outputType));
        if (succeeded(outputShapeVector)) {
          gridCalculationShape = *outputShapeVector;
          break;
        }
      }
    }
  }

  if (!tileAnalyzer.isValidTileSize(
          tileSize, *shape,
          tensorType.getElementType().getIntOrFloatBitWidth() / CHAR_BIT)) {
    return graphOp.emitError(
        "invalid tile size for the given tensor shape and element size");
  }

  llvm::SmallVector<DimSize> gridSize;
  gridSize.reserve(tileSize.size());
  for (size_t i = 0; i < tileSize.size(); ++i) {
    // Use gridCalculationShape (output shape for broadcast ops, input shape
    // otherwise)
    gridSize.push_back((gridCalculationShape[i] + tileSize[i] - 1) /
                       tileSize[i]);
  }
  return gridSize;
}

FailureOr<llvm::SmallVector<DimSize>> calculatePersistentGridSizeForGraph(
    const TileAnalyzer &tileAnalyzer, llvm::ArrayRef<DimSize> tileSize,
    mlir::nv_tensor_ir::GraphOp graphOp, int64_t persistentCtaCount,
    int64_t *totalTiles) {
  if (persistentCtaCount <= 0) {
    graphOp.emitError("persistent CTA count must be positive");
  }

  // Calculate the base grid size (total work to be done).
  MLIR_ASSIGN_OR_RETURN(
      llvm::SmallVector<DimSize> baseGridSize,
      calculateGridSizeForGraph(tileAnalyzer, tileSize, graphOp));
  // Compute total tile count from grid dimensions
  int64_t totalWorkTiles =
      std::accumulate(baseGridSize.begin(), baseGridSize.end(), int64_t{1},
                      std::multiplies<>());

  // Return total tiles via output parameter if requested
  if (totalTiles != nullptr) {
    *totalTiles = totalWorkTiles;
  }

  // Persistent kernels use a one-dimensional grid capped by the available
  // work.
  int64_t persistentGridSize = std::min(persistentCtaCount, totalWorkTiles);
  return llvm::SmallVector<DimSize>{static_cast<DimSize>(persistentGridSize)};
}

bool useOutputTensorShapeForGrid(mlir::nv_tensor_ir::GraphOp graphOp) {
  // Check which operations require using output tensor shape for grid
  // computation. This logic must be kept in sync between compile-time
  // (calculateGridSizeForGraph) and runtime
  // (KernelArgLayout::gridShapeTensorIdx).
  //
  // - BroadcastOp: grid should cover the broadcast output dimensions
  // - ConcatenateOp: grid should cover the full concatenated output dimensions,
  //   not only the first input operand's shape
  // - MatmulOp: grid should cover output dimensions (M, N), not input (M, K)
  //   because the reduction dimension K is handled inside each CTA
  return llvm::any_of(graphOp.getBody()->getOperations(), [](Operation &op) {
    return mlir::isa<mlir::nv_tensor_ir::MatmulOp>(op) ||
           mlir::isa<mlir::nv_tensor_ir::ConcatenateOp>(op) ||
           mlir::isa<mlir::nv_tensor_ir::BroadcastOp>(op);
  });
}

} // namespace nv_tensor_ir
} // namespace mlir
