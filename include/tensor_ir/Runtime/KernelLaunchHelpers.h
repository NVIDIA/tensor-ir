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
//     ArgumentPacker — packs pointers, scalars, and metadata according to the
//                      dynamic size/stride metadata in KernelMetadata.
//   Grid Size Computation:
//     StaticGridComputer    — returns fixed grid dims known at compile time.
//     TileBasedGridComputer — computes grid from runtime shapes + tile sizes.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_IR_RUNTIME_KERNEL_LAUNCH_HELPERS_H_
#define TENSOR_IR_RUNTIME_KERNEL_LAUNCH_HELPERS_H_

#include "tensor_ir/Artifact/KernelMetadata.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <array>
#include <cstdint>
#include <utility>

namespace tensor_ir::rt {

namespace detail {

/// Pack the common pointer/scalar slot, optionally seeding unresolved operands.
bool packOperandIntoInt64Slot(void *ptr, int64_t &slot, bool scalar,
                              int32_t scalarBytes,
                              bool allowUnresolvedPointers);

} // namespace detail

/// Packs data pointers plus sizes and strides according to a compile-time
/// KernelMetadata.
///
/// For each tensor the kernel signature is:
///   [ptr] [size_0, size_1, ...] [stride_0, stride_1, ...]
///
/// Signatures pack dynamic sizes and explicit dynamic strides. Scalar operands
/// contribute one by-value slot.
///
/// @tparam Accessor Satisfies the OperandAccessor concept.
template <typename Accessor>
class ArgumentPacker final {
public:
  explicit ArgumentPacker(KernelMetadata layout) : layout_(std::move(layout)) {}

  /// Pack arguments strictly; return the slot count or -1 on failure.
  /// Operand count must match the layout, and outArgs must hold numKernelArgs()
  /// slots. Count and capacity errors are rejected before writing.
  int32_t packArgs(const Accessor &operands,
                   llvm::MutableArrayRef<int64_t> outArgs) const {
    return packArgsImpl(operands, outArgs, /*allowUnresolvedPointers=*/false);
  }

  /// Seed unresolved pointers and scalars with zero for later lifecycle updates.
  int32_t
  packArgsAllowPlaceholders(const Accessor &operands,
                            llvm::MutableArrayRef<int64_t> outArgs) const {
    return packArgsImpl(operands, outArgs, /*allowUnresolvedPointers=*/true);
  }

  /// Number of slots in the compiled signature.
  int32_t numKernelArgs() const { return layout_.totalKernelArgs; }

private:
  int32_t packArgsImpl(const Accessor &operands,
                       llvm::MutableArrayRef<int64_t> outArgs,
                       bool allowUnresolvedPointers) const {
    if (static_cast<size_t>(operands.size()) != layout_.args.size() ||
        outArgs.size() < static_cast<size_t>(layout_.totalKernelArgs)) {
      return -1;
    }

    int32_t idx = 0;
    for (size_t tensorIdx = 0; tensorIdx < layout_.args.size(); ++tensorIdx) {
      const auto &arg = layout_.args[tensorIdx];
      int32_t i = static_cast<int32_t>(tensorIdx);

      const auto *scalar = std::get_if<ScalarArgDesc>(&arg);
      if (!detail::packOperandIntoInt64Slot(
              operands.pointer(i), outArgs[idx++], scalar != nullptr,
              scalar ? getElementByteWidth(scalar->elementType) : 0,
              allowUnresolvedPointers)) {
        return -1;
      }
      if (scalar) {
        continue;
      }

      const auto &desc = std::get<TensorArgDesc>(arg);

      // 2. Sizes
      if (operands.hasShape(i)) {
        for (int32_t d = 0; d < desc.rank(); ++d) {
          if (desc.shape()[d] == TensorArgDesc::kDynamic) {
            outArgs[idx++] = operands.shape(i, d);
          }
        }
      }

      // 3. Strides (when explicit)
      if (desc.hasExplicitStrides() && operands.isDense(i)) {
        for (int32_t d = 0; d < desc.rank(); ++d) {
          if (desc.strides()[d] == TensorArgDesc::kDynamic) {
            outArgs[idx++] = operands.stride(i, d);
          }
        }
      }
    }
    return idx;
  }
  KernelMetadata layout_;
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

namespace detail {

/// Compute the tiled grid after the accessor adapter checks shape availability.
GridDims computeTileBasedGrid(const KernelMetadata &layout,
                              llvm::function_ref<int64_t(int32_t)> shape);

} // namespace detail

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
/// For static persistence, the caller (getLayoutPropGridInfo) is responsible
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
/// KernelMetadata::gridShapeTensorIdx (defaults to first input; overridden
/// to the output tensor for matmul / broadcast / concat).
template <typename Accessor>
class TileBasedGridComputer final : public GridSizeComputer<Accessor> {
public:
  explicit TileBasedGridComputer(KernelMetadata layout)
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

    return detail::computeTileBasedGrid(layout_, [&](int32_t dim) {
      return operands.shape(shapeTensorIdx, dim);
    });
  }

private:
  KernelMetadata layout_;
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_RUNTIME_KERNEL_LAUNCH_HELPERS_H_
