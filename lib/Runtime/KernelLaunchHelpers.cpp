// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Runtime/KernelLaunchHelpers.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>

using namespace tensor_ir::rt;

bool tensor_ir::rt::detail::packOperandIntoInt64Slot(
    const void *ptr, int64_t &slot, bool scalar, int32_t scalarBytes,
    bool allowUnresolvedPointers) {
  if (!ptr && !allowUnresolvedPointers) {
    return false;
  }
  if (scalar && ptr) {
    if (scalarBytes <= 0 ||
        static_cast<size_t>(scalarBytes) > sizeof(int64_t)) {
      assert(false && "scalar operand size out of range for int64_t arg slot "
                      "(expected 1..8 bytes); see scalarBytes");
      return false;
    }
    slot = 0;
    std::memcpy(&slot, ptr, static_cast<size_t>(scalarBytes));
    return true;
  }
  slot = ptr ? reinterpret_cast<int64_t>(ptr) : 0;
  return true;
}

GridDims tensor_ir::rt::detail::computeTileBasedGrid(
    const KernelMetadata &layout, llvm::function_ref<int64_t(int32_t)> shape) {
  GridDims grid;
  int32_t shapeTensorIdx = layout.gridShapeTensorIdx;
  const auto *shapeDesc =
      shapeTensorIdx >= 0 &&
              shapeTensorIdx < static_cast<int32_t>(layout.args.size())
          ? std::get_if<TensorArgDesc>(&layout.args[shapeTensorIdx])
          : nullptr;
  int32_t tensorRank =
      shapeDesc ? shapeDesc->rank()
      : shapeTensorIdx < static_cast<int32_t>(layout.args.size())
          ? 0
          : static_cast<int32_t>(layout.tileSizes.size());

  const bool hasNormalizedGridShape = !layout.gridShape.empty();
  // Normalized shape/mapping invariants are enforced by immutable layouts.
  if (!hasNormalizedGridShape &&
      (tensorRank < 0 ||
       layout.tileSizes.size() > static_cast<size_t>(tensorRank))) {
    grid.x = 0;
    return grid;
  }

  size_t numDimsToTile = layout.tileSizes.size();

  int64_t totalTiles = 1;
  constexpr int64_t maxI32 = std::numeric_limits<int32_t>::max();
  for (size_t i = 0; i < numDimsToTile; ++i) {
    int64_t dimSize;
    if (hasNormalizedGridShape) {
      dimSize = layout.gridShape[i];
      int32_t tensorDim = layout.gridShapeDimMapping[i];
      if (dimSize == TensorArgDesc::kDynamic) {
        dimSize = shape(tensorDim);
      }
    } else {
      dimSize = shape(static_cast<int32_t>(i));
    }
    int32_t tileSize = layout.tileSizes[i];
    if (tileSize <= 0 || dimSize < 0) {
      grid.x = 0;
      return grid;
    }
    if (dimSize == 0) {
      totalTiles = 0;
      break;
    }
    int64_t tilesInDim =
        dimSize / tileSize + static_cast<int64_t>(dimSize % tileSize != 0);
    if (tilesInDim > maxI32 / totalTiles) {
      totalTiles = maxI32;
      break;
    }
    totalTiles *= tilesInDim;
  }
  if (layout.persistence == mlir::nv_tensor_ir::PersistenceMode::Static &&
      layout.smCount > 0 && layout.occupancy > 0) {
    int64_t persistentGridSize =
        static_cast<int64_t>(layout.smCount) * layout.occupancy;
    totalTiles = std::min(totalTiles, persistentGridSize);
  }
  totalTiles = std::min(totalTiles, maxI32);
  grid.x = static_cast<int32_t>(totalTiles);
  return grid;
}
