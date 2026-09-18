// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_CONVERSION_TENSORTOCUDATILE_TENSORTOCUDATILEINTERNAL_H_
#define TENSOR_IR_CONVERSION_TENSORTOCUDATILE_TENSORTOCUDATILEINTERNAL_H_

// Library-internal plumbing for the TensorIR-to-CudaTile conversion. These
// declarations are shared across the conversion library's translation units
// but are not part of the public interface consumed by backend code.

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

namespace mlir::nv_tensor_ir {
namespace tensor_to_cuda_tile {

/// Convert destination-passing functions marked `gpu.kernel` and nested in
/// `gpu.module` operations from the staged tiled-program contract to CUDA Tile
/// entries. Memref arguments are expanded to pointer/size/stride ABI arguments;
/// retained host wrappers produced by `outline-tensor-ir-kernel` are resolved
/// through `gpu.launch_func` and consumed before kernel conversion. After the
/// GPU modules are consumed, the enclosing module's `gpu.container_module`
/// staging attribute is removed.
LogicalResult convertOutlinedKernels(
    ModuleOp module, cuda_tile::OptimizationHintsAttr optimizationHints,
    bool uniformSignature, bool enableExperimentalCudaTileOps);

/// Populate conversions for tile-shaped TensorIR compute. These patterns use
/// only converted SSA operands and result types; they do not consult layout
/// analysis.
void registerOutlinedComputePatterns(RewritePatternSet &patterns,
                                     const TypeConverter &typeConverter,
                                     bool enableExperimentalCudaTileOps);

/// Verify the state-free type and shape contract of a matmul in an outlined
/// tiled kernel. Emits a diagnostic on `op` when the contract is invalid.
LogicalResult validateOutlinedMatmulLowerable(MatmulOp op);

/// ----- Implemented in `PointwiseOps.cpp` ------------------------------------

/// Shared per-op legality check for pointwise operations. This is a pure
/// predicate over the op's element types / attributes: it does not consult the
/// rewriter or any conversion-driver state. It is invoked both by the pointwise
/// affine-map conversion patterns and the staged outlined-kernel patterns.
///
/// On failure the diagnostic is emitted on `op`, matching the error that the
/// conversion pattern would have produced.
LogicalResult validatePointwiseOpLowerable(Operation *op);

/// Shared pre-rewrite legality check for `ConstantOp`. `ConstantOp` is not a
/// `PointwiseOpInterface` op, so it is dispatched explicitly rather than via
/// `validatePointwiseOpLowerable`. Pure predicate over the constant's result
/// type and value attribute (a tensor-typed constant must be a splat literal);
/// it does not consult the rewriter or the type converter. Shared by both
/// conversion paths and emits the diagnostic on `op`.
LogicalResult validateConstantOpLowerable(ConstantOp op);

/// Emit the CUDA Tile implementation of a validated pointwise operation.
/// `operands` must contain the strategy-adapted values in source operand order.
LogicalResult lowerPointwiseOp(PointwiseOpInterface op,
                               ArrayRef<Value> operands,
                               cuda_tile::TileType resultType,
                               ConversionPatternRewriter &rewriter,
                               bool enableExperimentalCudaTileOps);

/// Shared state-free builders used by both conversion paths.
Value buildExp(OpBuilder &rewriter, Location loc, Value tile,
               Type resultElementType);
Value buildSigmoid(OpBuilder &rewriter, Value tile);
Value buildErfApprox(OpBuilder &rewriter, Location loc, Value tile);

/// Conversion pattern registration.
void registerPointwisePatterns(RewritePatternSet &patterns,
                               ConversionState &state,
                               const TypeConverter &typeConverter,
                               bool enableExperimentalCudaTileOps);

/// ----- Implemented in `EmitHelpers.cpp` -------------------------------------

/// Create load/store optimization hints. Returns null when `allowTma` is true
/// and `latency` is negative, which are the CUDA Tile defaults.
cuda_tile::OptimizationHintsAttr
createLoadStoreOptimizationHints(MLIRContext *ctx, bool allowTma,
                                 int32_t latency);

/// Create entry-point optimization hints with all launch options present.
cuda_tile::OptimizationHintsAttr createEntryOptimizationHints(MLIRContext *ctx,
                                                              int32_t numCTAs,
                                                              int32_t occupancy,
                                                              int32_t numWarps);

/// Create constant tile (floating-point).
/// @param type Shaped type with the original element type.
/// @param value The literal value for the attribute.
Value createConstant(OpBuilder &rewriter, Location loc, ShapedType type,
                     double value);

/// Create constant tile (integer).
/// @param type Shaped type with the converted element type (signless).
/// @param value The literal value for the attribute.
Value createConstant(OpBuilder &rewriter, Location loc, ShapedType type,
                     int64_t value);

/// Get signedness enum value for the given TensorIR integer type.
/// @param type The signed/unsigned integer type.
cuda_tile::Signedness getSignedness(Type type);

/// @brief Reduction emission helper.
/// Currently supports only floating-point types.
struct ReductionEmissionHelper {
  ReductionMode mode;
  Type elementType;

  /// Get the identity value for the reduction operation.
  /// - add, amax, avg, norm1, norm2: 0.0 (additive identity)
  /// - mul, mul_no_zeros: 1.0 (multiplicative identity)
  /// - max: -infinity (all values are greater)
  /// - min: +infinity (all values are smaller)
  Attribute getIdentity(OpBuilder &rewriter);

  /// Build the prologue transformation for a reduction operation.
  /// - amax, norm1: Takes absolute value.
  /// - norm2: Squares the value.
  /// - mul_no_zeros: Replaces zeros with ones.
  /// - other modes: passthrough.
  Value buildPrologue(OpBuilder &rewriter, Value inputTile);

  /// Build the reduction transformation for a reduction operation.
  /// - add, avg, norm1, norm2: Add the value to the accumulator.
  /// - mul, mul_no_zeros: Multiply the value by the accumulator.
  /// - max, amax: Select the maximum value.
  /// - min: Select the minimum value.
  Value buildReduction(OpBuilder &rewriter, Value accumulator, Value tile);

  /// Build the epilogue transformation for a reduction operation.
  /// - norm2: Applies square root (L2 norm).
  /// - avg: Divides by reduction size for average.
  /// - other modes: passthrough.
  Value buildEpilogue(OpBuilder &rewriter, Value outputTile,
                      int64_t reductionSize);
};

} // namespace tensor_to_cuda_tile
} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_CONVERSION_TENSORTOCUDATILE_TENSORTOCUDATILEINTERNAL_H_
