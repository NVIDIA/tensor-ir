// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// KernelLaunchHelpers<Accessor> — helpers for kernel argument packing and
// grid size computation at launch time.
//
// Templated on an OperandAccessor so that the same logic works with any
// tensor representation (TensorOperandDescriptor, TensorView, etc.).
//
// OperandAccessor concept:
//   int32_t  size()                          const;
//   void*    pointer(int32_t i)              const;
//   int64_t  shape(int32_t i, int32_t d)     const;
//   int64_t  stride(int32_t i, int32_t d)    const;
//   bool     hasShape(int32_t i)             const;
//   bool     isDense(int32_t i)              const;
//
// Implementations:
//   Argument Packing:
//     PointerOnlyArgPacker — static shapes: packs data pointers only.
//     FlatArgPacker        — dynamic shapes: packs pointers + sizes + strides
//                            based on a KernelArgLayout.
//   Grid Size Computation:
//     StaticGridComputer    — returns fixed grid dims known at compile time.
//     TileBasedGridComputer — computes grid from runtime shapes + tile sizes.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_IR_RUNTIME_CUDATILE_KERNEL_LAUNCH_HELPERS_H_
#define TENSOR_IR_RUNTIME_CUDATILE_KERNEL_LAUNCH_HELPERS_H_

#include "tensor_ir/Runtime/CudaTile/KernelArgLayout.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace tensor_ir::rt {

namespace detail {

/// Copy a by-value scalar kernel argument into a single `int64_t` slot in
/// the flat argument array.
///
/// Validates that the scalar fits into exactly one `int64_t` slot before
/// copying. Returns `true` on success. Returns `false` (without touching
/// `*slot`) when `sizeInBytes` is non-positive or exceeds
/// `sizeof(int64_t)` — for example, 16-byte complex scalars
/// (`ScalarDescriptor::CF64` / `CS64`) which the current packers cannot
/// represent in a single slot and which would otherwise cause
/// `memcpy`-driven buffer overflow / UB. Proper multi-slot packing for
/// such types requires an ABI change (see `TensorArgDesc::totalArgs()`)
/// and is not yet implemented; callers should propagate the failure as a
/// packing error.
///
/// `operandIndex` is accepted for diagnostics only and is surfaced via
/// `assert` in debug builds.
inline bool packScalarIntoInt64Slot(int64_t *slot, const void *src,
                                    int32_t sizeInBytes, int32_t operandIndex) {
  (void)operandIndex;
  if (sizeInBytes <= 0 || static_cast<size_t>(sizeInBytes) > sizeof(int64_t)) {
    assert(false && "scalar operand size out of range for int64_t arg slot "
                    "(expected 1..8 bytes); see operandIndex / sizeInBytes");
    return false;
  }
  *slot = 0;
  std::memcpy(slot, src, static_cast<size_t>(sizeInBytes));
  return true;
}

} // namespace detail

//===----------------------------------------------------------------------===//
// KernelArgPacker — argument packing strategies
//===----------------------------------------------------------------------===//

/// Interface for packing tensor operands into a flat int64_t argument array
/// suitable for nvtileLaunchKernel.
///
/// @tparam Accessor  Satisfies the OperandAccessor concept.
template <typename Accessor>
class KernelArgPacker {
public:
  virtual ~KernelArgPacker() = default;

  /// Pack tensor operands into a flat int64_t argument array.
  ///
  /// @param operands  Runtime tensor operands wrapped in an Accessor.
  /// @param outArgs   Output buffer; must have at least numKernelArgs()
  ///                  entries.
  /// @return number of arguments written, or < 0 on error.
  virtual int32_t packArgs(const Accessor &operands,
                           int64_t *outArgs) const = 0;

  /// Seed the launch argument array during host-reserved initialization.
  ///
  /// The default implementation is strict and delegates to packArgs().
  /// Packers may override this to tolerate unresolved pointer operands that
  /// are expected to be patched in a later lifecycle phase.
  virtual int32_t packArgsAllowPlaceholders(const Accessor &operands,
                                            int64_t *outArgs) const {
    return packArgs(operands, outArgs);
  }

  /// Number of flat kernel arguments that packArgs() will produce.
  virtual int32_t numKernelArgs(int32_t numOperands) const = 0;
};

/// Packs one int64_t per operand.
///
/// For tensor operands this is the data pointer.  For scalar (by-value)
/// operands the scalar's bytes are memcpy'd into the low-order bytes of
/// the int64 slot, so that the CUDA driver reads the actual value when
/// it dereferences `&argValues[i]`.
///
/// Used when all shapes and strides are baked into the kernel at compile time.
template <typename Accessor>
class PointerOnlyArgPacker final : public KernelArgPacker<Accessor> {
public:
  int32_t packArgs(const Accessor &operands, int64_t *outArgs) const override {
    int32_t idx = 0;
    for (int32_t i = 0; i < operands.size(); ++i) {
      void *ptr = operands.pointer(i);
      if (!ptr) {
        return -1;
      }
      if (operands.isScalar(i)) {
        // By-value kernel arg: copy the scalar bytes into the slot so the
        // driver sees the value (not a host pointer) at &outArgs[idx].
        // Reject scalars that do not fit into a single int64_t slot (e.g.
        // 16-byte complex types) instead of overflowing the slot.
        if (!detail::packScalarIntoInt64Slot(
                &outArgs[idx], ptr, operands.scalarSizeInBytes(i), i)) {
          return -1;
        }
        ++idx;
      } else {
        outArgs[idx++] = reinterpret_cast<int64_t>(ptr);
      }
    }
    return idx;
  }

  int32_t packArgsAllowPlaceholders(const Accessor &operands,
                                    int64_t *outArgs) const override {
    int32_t idx = 0;
    for (int32_t i = 0; i < operands.size(); ++i) {
      void *ptr = operands.pointer(i);
      if (operands.isScalar(i)) {
        outArgs[idx] = 0;
        if (ptr) {
          if (!detail::packScalarIntoInt64Slot(
                  &outArgs[idx], ptr, operands.scalarSizeInBytes(i), i)) {
            return -1;
          }
        }
        ++idx;
      } else {
        outArgs[idx++] = ptr ? reinterpret_cast<int64_t>(ptr) : 0;
      }
    }
    return idx;
  }

  int32_t numKernelArgs(int32_t numOperands) const override {
    return numOperands;
  }
};

/// Packs data pointers plus sizes and strides according to a compile-time
/// KernelArgLayout.
///
/// For each tensor the kernel signature is:
///   [ptr] [size_0, size_1, ...] [stride_0, stride_1, ...]
///
/// All sizes (and all strides when present) are always packed, even for
/// statically-known dimensions.  This produces a uniform argument layout
/// for both static and dynamic shapes.
template <typename Accessor>
class FlatArgPacker final : public KernelArgPacker<Accessor> {
public:
  explicit FlatArgPacker(KernelArgLayout layout) : layout_(std::move(layout)) {}

  int32_t packArgs(const Accessor &operands, int64_t *outArgs) const override {
    return packArgsImpl(operands, outArgs, /*allowUnresolvedPointers=*/false);
  }

  int32_t packArgsAllowPlaceholders(const Accessor &operands,
                                    int64_t *outArgs) const override {
    return packArgsImpl(operands, outArgs, /*allowUnresolvedPointers=*/true);
  }

  int32_t numKernelArgs(int32_t /*numOperands*/) const override {
    return layout_.totalKernelArgs;
  }

private:
  int32_t packArgsImpl(const Accessor &operands, int64_t *outArgs,
                       bool allowUnresolvedPointers) const {
    int32_t idx = 0;
    for (size_t tensorIdx = 0; tensorIdx < layout_.tensorDescs.size();
         ++tensorIdx) {
      const auto &desc = layout_.tensorDescs[tensorIdx];
      int32_t i = static_cast<int32_t>(tensorIdx);

      // Scalar operands are a single by-value kernel argument. Copy the
      // scalar's bytes into the low-order bytes of the int64 slot so the
      // CUDA driver reads the value (not a host pointer) at &outArgs[idx].
      if (desc.isScalar) {
        void *ptr = operands.pointer(i);
        if (!ptr) {
          if (!allowUnresolvedPointers) {
            return -1;
          }
          outArgs[idx++] = 0;
          continue;
        }
        // Reject scalars that do not fit into a single int64_t slot (e.g.
        // 16-byte complex types) instead of overflowing the slot.
        if (!detail::packScalarIntoInt64Slot(&outArgs[idx], ptr,
                                             desc.scalarSizeInBytes, i)) {
          return -1;
        }
        ++idx;
        continue;
      }

      // 1. Pointer
      void *ptr = operands.pointer(i);
      if (!ptr) {
        if (!allowUnresolvedPointers) {
          return -1;
        }
        outArgs[idx++] = 0;
      } else {
        outArgs[idx++] = reinterpret_cast<int64_t>(ptr);
      }

      // 2. Sizes
      if (desc.numDynSizes > 0 && operands.hasShape(i)) {
        for (int32_t d = 0; d < desc.rank; ++d) {
          if (layout_.uniformSignature ||
              desc.staticShape[d] == TensorArgDesc::kDynamic) {
            outArgs[idx++] = operands.shape(i, d);
          }
        }
      }

      // 3. Strides (when explicit)
      if (desc.numDynStrides > 0 && operands.isDense(i)) {
        for (int32_t d = 0; d < desc.rank; ++d) {
          if (layout_.uniformSignature ||
              (!desc.staticStrides.empty() &&
               desc.staticStrides[d] == TensorArgDesc::kDynamic)) {
            outArgs[idx++] = operands.stride(i, d);
          }
        }
      }
    }
    return idx;
  }
  KernelArgLayout layout_;
};

//===----------------------------------------------------------------------===//
// GridSizeComputer — grid size computation strategies
//===----------------------------------------------------------------------===//

/// Three-dimensional CUDA grid dimensions.
struct GridDims {
  int32_t x = 1;
  int32_t y = 1;
  int32_t z = 1;
};

/// Interface for computing CUDA grid dimensions at runtime.
///
/// @tparam Accessor  Satisfies the OperandAccessor concept.
template <typename Accessor>
class GridSizeComputer {
public:
  virtual ~GridSizeComputer() = default;

  /// Compute grid dimensions from runtime tensor operands.
  virtual GridDims computeGrid(const Accessor &operands) const = 0;
};

/// Returns fixed grid dimensions that were determined at compile time.
/// For static persistence, the caller (getLayoutPropGridSize) is responsible
/// for computing min(smCount * occupancy, totalTiles).
template <typename Accessor>
class StaticGridComputer final : public GridSizeComputer<Accessor> {
public:
  explicit StaticGridComputer(std::array<int32_t, 3> tiles) : tiles_(tiles) {}

  GridDims computeGrid(const Accessor & /*operands*/) const override {
    return {tiles_[0], tiles_[1], tiles_[2]};
  }

private:
  std::array<int32_t, 3> tiles_;
};

/// Computes grid dimensions at runtime from tensor shapes and tile sizes.
///
/// Mirrors the compile-time logic in TileAnalyzer::calculateGridSizeForGraph:
///   gridSize[i] = ceil(shape[i] / tileSize[i])
/// then flattens all dimensions into gridDimX (matching mapGridSizeTo3D).
///
/// The tensor whose shape drives the grid is selected by
/// KernelArgLayout::gridShapeTensorIdx (defaults to first input; overridden
/// to the output tensor for matmul / broadcast / concat).
template <typename Accessor>
class TileBasedGridComputer final : public GridSizeComputer<Accessor> {
public:
  explicit TileBasedGridComputer(KernelArgLayout layout)
      : layout_(std::move(layout)) {}

  GridDims computeGrid(const Accessor &operands) const override {
    GridDims grid;
    if (layout_.tileSizes.empty()) {
      return grid;
    }

    int32_t shapeTensorIdx = layout_.gridShapeTensorIdx;
    if (shapeTensorIdx >= operands.size() ||
        !operands.hasShape(shapeTensorIdx)) {
      return grid;
    }

    int32_t tensorRank =
        (shapeTensorIdx < static_cast<int32_t>(layout_.tensorDescs.size()))
            ? layout_.tensorDescs[shapeTensorIdx].rank
            : static_cast<int32_t>(layout_.tileSizes.size());

    size_t numDimsToTile =
        std::min(layout_.tileSizes.size(), static_cast<size_t>(tensorRank));

    int64_t totalTiles = 1;
    for (size_t i = 0; i < numDimsToTile; ++i) {
      int64_t dimSize = operands.shape(shapeTensorIdx, static_cast<int32_t>(i));
      int32_t tileSize = layout_.tileSizes[i];
      if (tileSize > 0 && dimSize > 0) {
        totalTiles *= (dimSize + tileSize - 1) / tileSize;
      }
    }
    // TODO: For static persistence, clamp grid to min(smCount * occupancy,
    // totalTiles). Currently always uses totalTiles which launches one CTA
    // per tile (non-persistent). Needs persistence mode and SM count added
    // to KernelArgLayout.
    grid.x = static_cast<int32_t>(totalTiles);
    return grid;
  }

private:
  KernelArgLayout layout_;
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_RUNTIME_CUDATILE_KERNEL_LAUNCH_HELPERS_H_
