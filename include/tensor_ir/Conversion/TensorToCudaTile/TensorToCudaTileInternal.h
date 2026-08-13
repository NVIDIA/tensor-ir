// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_CONVERSION_TENSORTOCUDATILE_TENSORTOCUDATILEINTERNAL_H_
#define TENSOR_IR_CONVERSION_TENSORTOCUDATILE_TENSORTOCUDATILEINTERNAL_H_

// Library-internal plumbing for the TensorIR-to-CudaTile conversion. These
// declarations are shared across the conversion library's translation units
// (the per-op conversion patterns and the layout-propagation driver) but are
// not part of the public interface consumed by backend code.

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"

namespace mlir::nv_tensor_ir {
namespace tensor_to_cuda_tile {

/// ----- Implemented in `PointwiseOps.cpp` ------------------------------------

/// Shared per-op legality check for pointwise operations. This is a pure
/// predicate over the op's element types / attributes: it does not consult the
/// rewriter or any conversion-driver state. It is invoked both by the pointwise
/// conversion patterns (right after the per-op state update) and by the
/// pre-flight feasibility check, so the two cannot diverge.
///
/// On failure the diagnostic is emitted on `op`, matching the error that the
/// conversion pattern would have produced.
LogicalResult validatePointwiseOpLowerable(Operation *op);

/// Shared pre-rewrite legality check for `ConstantOp`. `ConstantOp` is not a
/// `PointwiseOpInterface` op, so it is dispatched explicitly rather than via
/// `validatePointwiseOpLowerable`. Pure predicate over the constant's result
/// type and value attribute (a tensor-typed constant must be a splat literal);
/// it does not consult the rewriter or the type converter. Shared by the
/// constant conversion pattern and the pre-flight feasibility check, and emits
/// the diagnostic on `op`.
LogicalResult validateConstantOpLowerable(ConstantOp op);

/// Conversion pattern registration.
void registerPointwisePatterns(RewritePatternSet &patterns,
                               ConversionState &state,
                               const TypeConverter &typeConverter,
                               bool enableExperimentalCudaTileOps);

/// ----- Implemented in `ReductionOps.cpp` ------------------------------------

/// Shared per-op legality check for reduction operations. Pure predicate over
/// the reduction's element type and mode; shared by the reduction conversion
/// pattern and the pre-flight feasibility check. Emits the diagnostic on `op`.
LogicalResult validateReductionOpLowerable(ReduceOp op);

/// Shared per-op legality check for `IotaOp`. Pure predicate over the op's
/// layout attribute and the graph tile shape; shared by the iota conversion
/// pattern and the pre-flight feasibility check. Emits the diagnostic on `op`.
LogicalResult validateIotaOpLowerable(IotaOp op, ArrayRef<int64_t> tileShape);

/// Conversion pattern registration.
void registerReductionPatterns(RewritePatternSet &patterns,
                               ConversionState &state,
                               const TypeConverter &typeConverter);

/// ----- Implemented in `MatmulOps.cpp` ------------------------------------

/// Shared per-op legality check for matmul operations. Pure predicate over the
/// matmul's element types and layout attribute; shared by the matmul conversion
/// pattern and the pre-flight feasibility check. Emits the diagnostic on `op`.
LogicalResult validateMatmulOpLowerable(MatmulOp op);

/// Conversion pattern registration.
void registerMatmulPatterns(RewritePatternSet &patterns, ConversionState &state,
                            const TypeConverter &typeConverter);

/// ----- Implemented in `LayoutPropagationImpl.cpp` ---------------------------

/// Run the graph-level legality checks that the layout-propagation
/// TensorIR-to-CudaTile conversion performs before invoking the dialect
/// conversion driver. Verifies that the graph has a single output, an
/// `iteration_space` attribute on the terminator, a valid `tile_size`
/// attribute matching the iteration-space rank, and that the input/output
/// tensor descriptors can be derived from the graph signature.
///
/// Preconditions: Phase 1 (graph analysis) and Phase 2 (tile selection) must
/// have already run on the graph, so the relevant attributes are present.
///
/// Returns `success()` when these graph-level checks pass. Returns `failure()`
/// when the graph has multiple outputs, is missing the `iteration_space` or
/// `tile_size` attribute, or the `tile_size` rank does not match the
/// iteration-space rank; in every failure case a diagnostic is emitted on
/// `graphOp`.
LogicalResult verifyGraphLevelLayoutPropLowerable(GraphOp graphOp);

/// Verify the attribute-level legality that the per-op state update relies on:
/// the presence of the `layout` attribute (and the correct composite/concat/
/// reduction layout kind for the op) and the presence of the iteration-space-id
/// attribute.
///
/// This is the legality half of `ConversionStateImpl::update()`, factored out
/// so the same checks run both during conversion and during the pre-flight
/// feasibility check.
///
/// `iterationSpaces`, when non-null, additionally checks that the op's
/// iteration-space id is a key of the map. The real conversion (`update()`)
/// passes the live map of iteration spaces the driver has discovered. The
/// driver-free pre-flight verifier passes null to skip the membership check,
/// because that map is built recursively during skeleton materialization and
/// cannot be faithfully reconstructed by a static walk (doing so under-collects
/// ids and false-rejects valid multi-iteration-space graphs). Emits the
/// diagnostic on `op`.
LogicalResult
verifyOpStateAttrs(Operation *op,
                   const DenseMap<int, IterationSpace *> *iterationSpaces);

/// ----- Implemented in `StructureBuilder.cpp` --------------------------------

/// Build the block structure and the iteration space data for the graph results
/// and all the operations that modify the iteration space.
///
/// `reductionTileSize` is the tile size for the contracting dimensions (used
/// for reductions and matmuls). The actual shape of the reduction tile is
/// set by a simple heuristic (evenly distribute between the dimensions).
FailureOr<DenseMap<Operation *, BlockStructure>>
buildSkeleton(RewriterBase &rewriter, IterationSpace initial,
              const TypeConverter &typeConverter,
              int64_t reductionTileSize = 128);

/// ----- Implemented in `EmitHelpers.cpp` -------------------------------------

/// Create load/store optimization hints. Returns null when `allowTma` is true
/// and `latency` is negative, which are the CUDA Tile defaults.
cuda_tile::OptimizationHintsAttr
createLoadStoreOptimizationHints(MLIRContext *ctx, bool allowTma,
                                 int32_t latency);

/// Create entry-point optimization hints. Returns null when the launch options
/// have their defaults: one CTA, occupancy one, and four worker warps.
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

/// Modify the tensor descriptor using the layout information.
/// @param desc The tensor descriptor (original pointer and hints).
/// @param tensorSource The layout with the actual sizes and strides.
TensorDescriptor applyLayout(OpBuilder &rewriter, const TensorDescriptor &desc,
                             TensorSourceAttr tensorSource);

/// Emit tile load operation.
/// @param desc The tensor descriptor (pointer, alignment, sizes, strides).
/// @param tileType The shaped type of the tile to load.
/// @param indexValues The index values for the load, same rank as `tileType`.
Value emitLoad(OpBuilder &rewriter, const TensorDescriptor &desc,
               ShapedType tileType, ValueRange indexValues);

/// Emit tile store operation.
/// @param desc The tensor descriptor (pointer, alignment, sizes, strides).
/// @param tile The tile to store.
/// @param indexValues The index values for the store, same rank as `tile` type.
void emitStore(OpBuilder &rewriter, const TensorDescriptor &desc, Value tile,
               ValueRange indexValues);

/// Calculate index values for an iteration space.
/// @param blockId The block ID from `cuda_tile::GetTileBlockIdOp`.
/// @param desc Sizes for the main iteration space.
/// @param tileShape Sizes for the tile shape, same rank as `desc` sizes.
FailureOr<SmallVector<Value>> calculateIndex(OpBuilder &rewriter, Value blockId,
                                             const TensorDescriptor &desc,
                                             ArrayRef<int64_t> tileShape);

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
