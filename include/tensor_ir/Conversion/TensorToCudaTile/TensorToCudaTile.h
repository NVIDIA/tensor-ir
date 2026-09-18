// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_CONVERSION_TENSORTOCUDATILE_TENSORTOCUDATILE_H
#define TENSOR_IR_CONVERSION_TENSORTOCUDATILE_TENSORTOCUDATILE_H

#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/Pass/Pass.h" // IWYU pragma: keep
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "tensor_ir/Conversion/TensorToCudaTile/Passes.h.inc"

/// Temporary module-level attribute used to surface tile sizes resolved by
/// TensorToCudaTileConversionPass after it erases the GraphOp.
inline constexpr const char *kResolvedTileSizeAttrName =
    "tensor_ir.resolved_tile_size";

namespace tensor_to_cuda_tile {

// Default alignment in CUDA is 256B, but we cannot assume that the pointers
// are aligned. The TensorIR must include the "alignment" attribute to enable
// vectorized loads/stores in the CUDA Tile lowering.
constexpr int64_t kDefaultPointerAlignment = 1;

/// Conversion state interface that provides tile values and shapes in the
/// context of the graph and the operation being processed.
class ConversionState {
public:
  virtual ~ConversionState() = default;

  /// Prepare the conversion state for a graph conversion.
  virtual LogicalResult start(GraphOp graphOp) = 0;

  /// Prepare the conversion state for an operation conversion.
  virtual LogicalResult update(ConversionPatternRewriter &rewriter,
                               Operation *op) = 0;

  /// Get the tile for an operand.
  virtual Value getTile(Value operand) = 0;

  /// Get the tile shape for the current iteration space.
  virtual ArrayRef<int64_t> getTileShape() = 0;

  /// Get the resolved tile size to synchronize runtime launch metadata.
  virtual ArrayRef<int64_t> getResolvedTileSize() { return {}; }
};

/// Conversion pattern base class that provides access to the conversion state.
template <typename OpTy>
class ConversionPattern : public OpConversionPattern<OpTy> {
public:
  ConversionPattern(ConversionState &state, const TypeConverter &typeConverter,
                    MLIRContext *ctx, PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, ctx, benefit), state(state) {}

protected:
  ConversionState &state;
};

/// ----- Implemented in `AffineMapImpl.cpp` -----------------------------------

/// State factory for the "affine_map" codegen strategy.
std::unique_ptr<ConversionState>
createAffineMapConversionState(MLIRContext *context,
                               const TypeConverter &typeConverter,
                               const TensorToCudaTilePipelineOptions &options);

} // namespace tensor_to_cuda_tile
} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_CONVERSION_TENSORTOCUDATILE_TENSORTOCUDATILE_H
