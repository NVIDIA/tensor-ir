// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Grid formation (FormGridPass) deliberately tiles only the parallel iteration
// space, leaving reduction and matrix-product contracting dimensions coarse.
// This pass makes those contracting domains explicit and executable on GPU
// hardware.
//
// Key responsibilities:
// 1. Contraction strip-mining:
//    - Evaluates the reduction/contraction shape against `reduction_tile_size`
//      (default 128 elements).
//    - When the contracting domain exceeds the tile size, creates an scf.for
//      nest whose induction variables step through contracting coordinates.
//    - Initializes accumulators using neutral reduction identities and carries
//      partial tile accumulators through scf.for `iter_args`.
//    - Clones upstream producer DAGs (LoadOps, PointwiseOps, IotaOps, Constants)
//      with smaller tile shapes, updating contracting coordinate indices to use
//      the loop induction variables.
//    - Handles out-of-bounds contraction elements using hardware-neutral load
//      padding (Zero, NegInf, PosInf) when safe, or falls back to explicit
//      tile masking (IotaOp + CmpOp + BinarySelectOp).
//    - Applies reduction prologues (abs, square, zero-filtering) and epilogues
//      (sqrt for norm2, division for avg).
//    - Lifts user-defined reduction bodies (arithmetic on scalars) to elementwise
//      operations on tiles.
// 2. Static persistence:
//    - Caps the hardware launch grid at `sm_count * occupancy` blocks.
//    - Serializes remaining logical tiles by wrapping the per-tile computation
//      in a grid-stride scf.for loop inside each persistent block.

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <limits>

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DEF_TILEREDUCTIONSPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace {

/// Partition a multi-dimensional contracting shape into power-of-two tile
/// extents such that their product does not exceed `reductionTileSize`.
///
/// Distributes available tile size bits evenly across contracting dimensions,
/// rounding up so earlier dimensions receive larger tile sizes when bits do
/// not divide evenly.
static SmallVector<int64_t>
calculateReductionTileShape(ArrayRef<int64_t> reductionShape,
                            int64_t reductionTileSize) {
  SmallVector<int64_t> tileShape;
  for (auto [index, size] : llvm::enumerate(reductionShape)) {
    int64_t maximum = int64_t{1} << llvm::Log2_64_Ceil(uint64_t(size));
    size_t remainingDimensions = reductionShape.size() - index;
    int bitsPerDimension = llvm::divideCeil(
        llvm::countr_zero(uint64_t(reductionTileSize)), remainingDimensions);
    int64_t mean = int64_t{1} << bitsPerDimension;
    tileShape.push_back(std::min(mean, maximum));
    reductionTileSize /= tileShape.back();
  }
  return tileShape;
}

/// Return the neutral identity attribute for a given reduction mode and
/// element type.
///
/// Neutral values:
/// - Additive (add, avg, norm1, norm2): 0.0 for float, 0 for integer.
/// - Multiplicative (mul, mul_no_zeros): 1.0 for float, 1 for integer.
/// - Max (max, amax): -infinity for float, signed/unsigned minimum for integer.
/// - Min (min): +infinity for float, signed/unsigned maximum for integer.
static Attribute getIdentity(OpBuilder &builder, ReductionMode mode,
                             Type elementType) {
  auto getInteger = [&](const APInt &value) {
    return builder.getIntegerAttr(elementType, value);
  };
  switch (mode) {
  case ReductionMode::add:
  case ReductionMode::amax:
  case ReductionMode::avg:
  case ReductionMode::norm1:
  case ReductionMode::norm2:
    return elementType.isFloat()
               ? Attribute(builder.getFloatAttr(elementType, 0.0))
               : Attribute(
                     getInteger(APInt(elementType.getIntOrFloatBitWidth(), 0)));
  case ReductionMode::mul:
  case ReductionMode::mul_no_zeros:
    return elementType.isFloat()
               ? Attribute(builder.getFloatAttr(elementType, 1.0))
               : Attribute(
                     getInteger(APInt(elementType.getIntOrFloatBitWidth(), 1)));
  case ReductionMode::max: {
    if (elementType.isFloat()) {
      return builder.getFloatAttr(elementType,
                                  -std::numeric_limits<double>::infinity());
    }
    unsigned width = elementType.getIntOrFloatBitWidth();
    return getInteger(elementType.isSignedInteger()
                          ? APInt::getSignedMinValue(width)
                          : APInt::getMinValue(width));
  }
  case ReductionMode::min: {
    if (elementType.isFloat()) {
      return builder.getFloatAttr(elementType,
                                  std::numeric_limits<double>::infinity());
    }
    unsigned width = elementType.getIntOrFloatBitWidth();
    return getInteger(elementType.isSignedInteger()
                          ? APInt::getSignedMaxValue(width)
                          : APInt::getMaxValue(width));
  }
  default:
    return {};
  }
}

/// Create a constant tensor filled with a splat attribute value.
static Value createSplat(OpBuilder &builder, Location loc,
                         RankedTensorType type, Attribute value) {
  auto dense =
      cast<DenseTypedElementsAttr>(DenseElementsAttr::get(type, value));
  return ConstantOp::create(builder, loc, type, cast<TypedAttr>(dense));
}

/// Return true when padding a source load with a neutral value remains neutral
/// through every operation up to the contraction.
///
/// If true, out-of-bounds elements can be padded directly at the memory
/// boundary via TilePaddingAttr on LoadOp, avoiding expensive explicit
/// masking. Layout operations have already folded into access descriptors,
/// so only floating-point type conversions (ConvertOp) can safely preserve
/// neutrality.
static bool isLoadPaddingNeutral(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (isa<LoadOp>(op)) {
      return true;
    }
    auto convert = dyn_cast<ConvertOp>(op);
    if (!convert ||
        !isa<FloatType>(convert.getInput().getType().getElementType()) ||
        !isa<FloatType>(convert.getOutput().getType().getElementType())) {
      return false;
    }
    value = convert.getInput();
  }
  return false;
}

/// Determine the hardware TilePadding attribute for a built-in reduction.
///
/// Returns TilePadding::Zero for additive modes, NegInf for float max, and
/// PosInf for float min. Returns null if load padding is not neutral.
static TilePaddingAttr getReductionLoadPadding(OpBuilder &builder,
                                               ReduceOp reduce,
                                               Attribute identity) {
  if (!isLoadPaddingNeutral(reduce.getInput())) {
    return {};
  }
  switch (reduce.getReductionMode()) {
  case ReductionMode::add:
  case ReductionMode::amax:
  case ReductionMode::avg:
  case ReductionMode::norm1:
  case ReductionMode::norm2:
    return TilePaddingAttr::get(builder.getContext(), TilePadding::Zero);
  case ReductionMode::max:
    if (isa<FloatAttr>(identity)) {
      return TilePaddingAttr::get(builder.getContext(), TilePadding::NegInf);
    }
    return {};
  case ReductionMode::min:
    if (isa<FloatAttr>(identity)) {
      return TilePaddingAttr::get(builder.getContext(), TilePadding::PosInf);
    }
    return {};
  default:
    return {};
  }
}

/// Inspect a user-defined reduction body to verify that its combiner operation
/// is compatible with the requested hardware load padding.
///
/// Checks that the body combines the accumulator with the input operand
/// using AddFOp/AddIOp for Zero, MaximumFOp for NegInf, or MinimumFOp for
/// PosInf.
static bool reductionBodyPreservesPadding(ReduceUDOp reduce,
                                          TilePadding padding) {
  Block &body = reduce.getBodyRegion().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  size_t inputCount = reduce.getNumOperands();
  if (body.getNumArguments() != inputCount * 2 ||
      yield.getNumOperands() != inputCount) {
    return false;
  }
  for (size_t index = 0; index < inputCount; ++index) {
    Operation *combiner = yield.getOperand(index).getDefiningOp();
    if (!combiner || combiner->getNumOperands() != 2 ||
        combiner->getNumResults() != 1) {
      return false;
    }
    Value accumulator = body.getArgument(index);
    Value input = body.getArgument(inputCount + index);
    if (!((combiner->getOperand(0) == accumulator &&
           combiner->getOperand(1) == input) ||
          (combiner->getOperand(0) == input &&
           combiner->getOperand(1) == accumulator))) {
      return false;
    }
    switch (padding) {
    case TilePadding::Zero:
      if (!isa<arith::AddFOp, arith::AddIOp>(combiner)) {
        return false;
      }
      break;
    case TilePadding::NegInf:
      if (!isa<arith::MaximumFOp>(combiner)) {
        return false;
      }
      break;
    case TilePadding::PosInf:
      if (!isa<arith::MinimumFOp>(combiner)) {
        return false;
      }
      break;
    default:
      return false;
    }
  }
  return true;
}

/// Determine the hardware TilePadding attribute for a user-defined reduction,
/// verifying that all inputs share an identity and that the combiner preserves
/// that identity.
static TilePaddingAttr getReductionLoadPadding(OpBuilder &builder,
                                               ReduceUDOp reduce) {
  if (reduce.getIdentity().empty()) {
    return {};
  }
  Attribute identity = reduce.getIdentity()[0];
  Type elementType = cast<TypedAttr>(identity).getType();
  TilePadding padding;
  if (identity == getIdentity(builder, ReductionMode::add, elementType)) {
    padding = TilePadding::Zero;
  } else if (isa<FloatAttr>(identity) &&
             identity ==
                 getIdentity(builder, ReductionMode::max, elementType)) {
    padding = TilePadding::NegInf;
  } else if (isa<FloatAttr>(identity) &&
             identity ==
                 getIdentity(builder, ReductionMode::min, elementType)) {
    padding = TilePadding::PosInf;
  } else {
    return {};
  }

  for (auto [operand, operandIdentity] :
       llvm::zip_equal(reduce.getInputs(), reduce.getIdentity())) {
    if (operandIdentity != identity || !isLoadPaddingNeutral(operand)) {
      return {};
    }
  }
  if (!reductionBodyPreservesPadding(reduce, padding)) {
    return {};
  }
  return TilePaddingAttr::get(builder.getContext(), padding);
}

/// Build or update a boolean mask tensor indicating valid in-bounds elements
/// along a contracting dimension.
///
/// When contracting extents do not evenly divide the tile size and load padding
/// cannot be used, this creates a 1D coordinate tensor using IotaOp, offsets it
/// by `coordinate * tile`, compares against `extent` with CmpOp(lt), and
/// expands/broadcasts across the contracting tile shape. Combines with previous
/// dimension masks via LogicalAndOp.
static Value updateContractionMask(OpBuilder &builder, Location loc,
                                   Value combined, int64_t extent, int64_t tile,
                                   size_t dimension,
                                   ArrayRef<int64_t> contractionTileShape,
                                   Value coordinate = {}) {
  if (extent % tile == 0) {
    return combined;
  }

  Type indexElementType = IntegerType::get(
      builder.getContext(), 32, IntegerType::SignednessSemantics::Unsigned);
  auto indexType = RankedTensorType::get({tile}, indexElementType);
  Value offsetTile;
  if (coordinate) {
    Value tileSize = arith::ConstantIndexOp::create(builder, loc, tile);
    Value offset = arith::MulIOp::create(builder, loc, coordinate, tileSize);
    Value signless =
        arith::IndexCastOp::create(builder, loc, builder.getI32Type(), offset);
    Value converted = UnrealizedConversionCastOp::create(
                          builder, loc, TypeRange{indexElementType}, signless)
                          .getResult(0);
    offsetTile =
        SplatOp::create(builder, loc, indexType, converted, ValueRange{});
  }
  Value indices = IotaOp::create(builder, loc, indexType, /*dimension=*/0,
                                 /*dynamic_sizes=*/ValueRange{});
  if (offsetTile) {
    indices = AddOp::create(builder, loc, indexType, offsetTile, indices);
  }

  Value bound = createSplat(builder, loc, indexType,
                            builder.getIntegerAttr(indexElementType, extent));
  Value mask =
      CmpOp::create(builder, loc, Comparator::lt, indices, bound).getResult();
  if (contractionTileShape.size() > 1) {
    SmallVector<int64_t> expandedShape(contractionTileShape.size(), 1);
    expandedShape[dimension] = tile;
    mask = ReshapeOp::create(
        builder, loc, RankedTensorType::get(expandedShape, builder.getI1Type()),
        mask);
    mask = BroadcastOp::create(
        builder, loc,
        RankedTensorType::get(contractionTileShape, builder.getI1Type()), mask);
  }
  return combined ? LogicalAndOp::create(builder, loc, combined, mask) : mask;
}

/// Reshape a contraction mask to match the full rank of a source operand by
/// inserting singleton dimensions (extent 1) for non-contracting axes.
static Value reshapeContractionMask(OpBuilder &builder, Location loc,
                                    Value mask, ArrayRef<int64_t> sourceShape,
                                    ArrayRef<int64_t> contractingDimensions,
                                    ArrayRef<int64_t> contractionTileShape) {
  if (!mask) {
    return {};
  }
  SmallVector<int64_t> expandedShape(sourceShape.size(), 1);
  for (auto [dimension, extent] :
       llvm::zip_equal(contractingDimensions, contractionTileShape)) {
    expandedShape[dimension] = extent;
  }
  if (!llvm::equal(cast<RankedTensorType>(mask.getType()).getShape(),
                   expandedShape)) {
    mask = ReshapeOp::create(
        builder, loc, RankedTensorType::get(expandedShape, builder.getI1Type()),
        mask);
  }
  return mask;
}

/// Broadcast singleton non-contracting dimensions of a mask to match the full
/// source operand shape.
static Value broadcastContractionMask(OpBuilder &builder, Location loc,
                                      Value mask,
                                      ArrayRef<int64_t> sourceShape) {
  if (!mask || llvm::equal(cast<RankedTensorType>(mask.getType()).getShape(),
                           sourceShape)) {
    return mask;
  }
  return BroadcastOp::create(
      builder, loc, RankedTensorType::get(sourceShape, builder.getI1Type()),
      mask);
}

/// Replace out-of-bounds tile elements (where mask is false) with the neutral
/// identity value using BinarySelectOp.
static Value applyPaddingMask(OpBuilder &builder, Location loc, Value tile,
                              Value mask, Attribute neutral) {
  if (!mask) {
    return tile;
  }
  auto tileType = cast<RankedTensorType>(tile.getType());
  Value neutralTile = createSplat(builder, loc, tileType, neutral);
  return BinarySelectOp::create(builder, loc, tileType, mask, tile,
                                neutralTile);
}

/// Plan describing how a reduction operation's contracting dimensions will be
/// strip-mined.
struct ReductionTilingPlan {
  /// Indices of the contracting dimensions within the input tensor.
  SmallVector<int64_t> dimensions;
  /// Full original extents of the contracting dimensions.
  SmallVector<int64_t> reductionShape;
  /// Strip-mined tile extents for each contracting dimension.
  SmallVector<int64_t> contractionTileShape;
  /// Operand tile shape with contracting dimensions replaced by their tile sizes.
  SmallVector<int64_t> operandTileShape;
};

/// Validate contracting dimensions and derive the reduction tiling plan.
///
/// Verifies that contracting dimensions are ordered and unique with positive
/// static extents, then calls calculateReductionTileShape to determine the
/// strip-mined contraction tile shape.
static FailureOr<ReductionTilingPlan>
createReductionTilingPlan(Operation *op, RankedTensorType inputType,
                          ArrayRef<int32_t> dimensions,
                          int64_t reductionTileSize) {
  if (dimensions.empty()) {
    return op->emitError("reduction has no contracting dimensions");
  }

  ReductionTilingPlan plan;
  plan.operandTileShape.assign(inputType.getShape().begin(),
                               inputType.getShape().end());
  for (auto [index, dimension] : llvm::enumerate(dimensions)) {
    if (dimension < 0 || dimension >= inputType.getRank() ||
        (index != 0 && dimensions[index - 1] >= dimension)) {
      return op->emitError(
          "strip mining requires ordered unique contracting dimensions");
    }
    int64_t extent = inputType.getDimSize(dimension);
    if (ShapedType::isDynamic(extent) || extent <= 0) {
      return op->emitError(
          "strip mining requires positive static contracting extents");
    }
    plan.dimensions.push_back(dimension);
    plan.reductionShape.push_back(extent);
  }
  plan.contractionTileShape =
      calculateReductionTileShape(plan.reductionShape, reductionTileSize);
  for (auto [dimension, size] :
       llvm::zip_equal(plan.dimensions, plan.contractionTileShape)) {
    plan.operandTileShape[dimension] = size;
  }
  return plan;
}

/// Loop nest and coordinate state generated for strip-mined contractions.
struct ContractionLoopNest {
  /// Generated scf.for loops, from outermost to innermost.
  SmallVector<scf::ForOp> loops;
  /// Loop induction variables corresponding to each contracting dimension
  /// (null for dimensions whose extent fits in a single tile).
  SmallVector<Value> coordinates;
  /// Carried accumulator values inside the innermost loop body.
  SmallVector<Value> carried;
  /// Combined boolean in-bounds mask across all contracting dimensions.
  Value mask;
};

/// Construct an scf.for loop nest over contracting dimensions where extent > tile.
///
/// Loops iterate from 0 to ceil(extent / tile) with step 1. Initial carried
/// values (accumulators) flow through iter_args. Optionally constructs the
/// in-bounds contraction mask inside the loop nest.
static ContractionLoopNest
createContractionLoopNest(OpBuilder &builder, Location loc,
                          ArrayRef<int64_t> reductionShape,
                          ArrayRef<int64_t> contractionTileShape,
                          ValueRange initialValues, bool buildMask) {
  ContractionLoopNest result;
  result.coordinates.resize(reductionShape.size());
  result.carried.assign(initialValues.begin(), initialValues.end());
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);

  for (size_t index = 0; index < reductionShape.size(); ++index) {
    int64_t extent = reductionShape[index];
    int64_t tile = contractionTileShape[index];
    Value coordinate;
    if (extent > tile) {
      Value upper = arith::ConstantIndexOp::create(
          builder, loc, llvm::divideCeil(extent, tile));
      auto loop =
          scf::ForOp::create(builder, loc, zero, upper, one, result.carried);
      result.loops.push_back(loop);
      coordinate = loop.getInductionVar();
      result.coordinates[index] = coordinate;
      result.carried.assign(loop.getRegionIterArgs().begin(),
                            loop.getRegionIterArgs().end());
      builder.setInsertionPointToEnd(loop.getBody());
      auto yield = scf::YieldOp::create(builder, loc, result.carried);
      builder.setInsertionPoint(yield);
    }
    if (buildMask) {
      result.mask =
          updateContractionMask(builder, loc, result.mask, extent, tile, index,
                                contractionTileShape, coordinate);
    }
  }
  return result;
}

/// Build a map from contracting dimension index to its loop induction variable.
static DenseMap<unsigned, Value>
mapContractionCoordinates(ArrayRef<int64_t> dimensions,
                          ArrayRef<Value> coordinates) {
  DenseMap<unsigned, Value> result;
  for (auto [dimension, coordinate] :
       llvm::zip_equal(dimensions, coordinates)) {
    if (coordinate) {
      result[dimension] = coordinate;
    }
  }
  return result;
}

/// Connect accumulator yield values through the nested scf.for hierarchy.
///
/// The innermost loop yields the updated partial accumulator `values`. Each
/// enclosing loop yields the results of its immediately nested child loop.
static void setContractionLoopYields(SmallVectorImpl<scf::ForOp> &loops,
                                     ValueRange values) {
  assert(!loops.empty() && "expected at least one contraction loop");
  cast<scf::YieldOp>(loops.back().getBody()->getTerminator())
      .getResultsMutable()
      .assign(values);
  for (size_t index = loops.size() - 1; index > 0; --index) {
    cast<scf::YieldOp>(loops[index - 1].getBody()->getTerminator())
        .getResultsMutable()
        .assign(loops[index].getResults());
  }
}

/// Emit the reduction combining operation between the carried accumulator
/// and the current tile.
static Value createAccumulatorStep(OpBuilder &builder, Location loc,
                                   ReductionMode mode, Value accumulator,
                                   Value tile) {
  switch (mode) {
  case ReductionMode::add:
  case ReductionMode::avg:
  case ReductionMode::norm1:
  case ReductionMode::norm2:
    return AddOp::create(builder, loc, accumulator, tile);
  case ReductionMode::mul:
  case ReductionMode::mul_no_zeros:
    return MulOp::create(builder, loc, accumulator, tile);
  case ReductionMode::max:
  case ReductionMode::amax:
    return MaxOp::create(builder, loc, accumulator, tile);
  case ReductionMode::min:
    return MinOp::create(builder, loc, accumulator, tile);
  default:
    return {};
  }
}

/// Apply elementwise pre-reduction transformations to a tile before accumulation.
///
/// - amax, norm1: Compute absolute value (AbsOp).
/// - norm2: Square elements (MulOp(x, x)).
/// - mul_no_zeros: Replace zero elements with 1 to avoid zeroing out products.
static Value createReductionPrologue(OpBuilder &builder, Location loc,
                                     ReductionMode mode, Value tile) {
  auto type = cast<RankedTensorType>(tile.getType());
  switch (mode) {
  case ReductionMode::amax:
  case ReductionMode::norm1:
    return AbsOp::create(builder, loc, type, tile);
  case ReductionMode::norm2:
    return MulOp::create(builder, loc, type, tile, tile);
  case ReductionMode::mul_no_zeros: {
    Type elementType = type.getElementType();
    Attribute zero = elementType.isFloat()
                         ? Attribute(builder.getFloatAttr(elementType, 0.0))
                         : Attribute(builder.getIntegerAttr(elementType, 0));
    Attribute one = elementType.isFloat()
                        ? Attribute(builder.getFloatAttr(elementType, 1.0))
                        : Attribute(builder.getIntegerAttr(elementType, 1));
    Value zeroTile = createSplat(builder, loc, type, zero);
    Value oneTile = createSplat(builder, loc, type, one);
    Comparator comparator =
        elementType.isFloat() ? Comparator::oeq : Comparator::eq;
    Value isZero = CmpOp::create(builder, loc, comparator, tile, zeroTile);
    return BinarySelectOp::create(builder, loc, type, isZero, oneTile, tile);
  }
  default:
    return tile;
  }
}

/// Apply post-reduction adjustments to the final accumulated tensor.
///
/// - norm2: SqrtOp to complete the L2 norm.
/// - avg: DivOp by the total reduction extent (handling potential integer overflow).
static Value createDegenerateReductionEpilogue(OpBuilder &builder, Location loc,
                                               ReductionMode mode,
                                               Value accumulated,
                                               int64_t reductionSize) {
  auto type = cast<RankedTensorType>(accumulated.getType());
  Type elementType = type.getElementType();
  if (mode == ReductionMode::norm2) {
    return SqrtOp::create(builder, loc, type, accumulated);
  }
  if (mode != ReductionMode::avg) {
    return accumulated;
  }

  if (auto integer = dyn_cast<IntegerType>(elementType)) {
    APInt maximum = integer.isSigned()
                        ? APInt::getSignedMaxValue(integer.getWidth()) + 1
                        : APInt::getMaxValue(integer.getWidth());
    if (uint64_t(reductionSize) > maximum.getZExtValue()) {
      return createSplat(builder, loc, type,
                         builder.getIntegerAttr(elementType, 0));
    }
  }
  Attribute divisor =
      elementType.isFloat()
          ? Attribute(builder.getFloatAttr(elementType, reductionSize))
          : Attribute(builder.getIntegerAttr(elementType, reductionSize));
  Value divisorTile = createSplat(builder, loc, type, divisor);
  return DivOp::create(builder, loc, type, accumulated, divisorTile);
}

/// Clones the upstream producer DAG feeding a contraction, shrinking tile
/// shapes to the strip-mined size and substituting contracting coordinates
/// with loop induction variables.
///
/// Replaces LoadOps with smaller tiled loads indexed by contracting loop
/// induction variables, re-splats constants, and recursively clones
/// pointwise and iota operations. Cloned values are memoized to preserve
/// reconvergent DAG structure.
class ProducerCloner {
public:
  ProducerCloner(ArrayRef<int64_t> tileShape,
                 const DenseMap<unsigned, Value> &contractingCoordinates,
                 TilePaddingAttr loadPadding, OpBuilder &builder)
      : tileShape(tileShape), contractingCoordinates(contractingCoordinates),
        loadPadding(loadPadding), builder(builder) {}

  FailureOr<Value> clone(Value value) {
    if (!isa<RankedTensorType>(value.getType())) {
      return value;
    }
    if (auto found = cache.find(value); found != cache.end()) {
      return found->second;
    }
    Operation *op = value.getDefiningOp();
    if (!op) {
      return emitError(value.getLoc())
             << "strip-mined producer unexpectedly captures a block argument";
    }

    if (auto load = dyn_cast<LoadOp>(op)) {
      SmallVector<OpFoldResult> indices = load.getMixedIndices();
      for (auto [dimension, coordinate] : contractingCoordinates) {
        if (dimension >= indices.size()) {
          return load.emitError(
              "contracting dimension is outside the tile-access rank");
        }
        indices[dimension] = coordinate;
      }
      Value result = LoadOp::create(
          builder, load.getLoc(),
          RankedTensorType::get(
              tileShape,
              cast<RankedTensorType>(load.getType()).getElementType()),
          load.getBase(), load.getMixedOffset(), load.getMixedSizes(),
          load.getMixedStrides(), indices, load.getAlignmentAttr(),
          loadPadding);
      cache[value] = result;
      return result;
    }

    if (auto constant = dyn_cast<ConstantOp>(op)) {
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!dense || !dense.isSplat()) {
        return constant.emitError(
            "strip mining requires tensor constants to be splats");
      }
      auto oldType = cast<RankedTensorType>(constant.getType());
      auto newType = RankedTensorType::get(tileShape, oldType.getElementType());
      Attribute splat =
          oldType.getElementType().isFloat()
              ? Attribute(builder.getFloatAttr(oldType.getElementType(),
                                               dense.getSplatValue<APFloat>()))
              : Attribute(builder.getIntegerAttr(oldType.getElementType(),
                                                 dense.getSplatValue<APInt>()));
      Value result = createSplat(builder, constant.getLoc(), newType, splat);
      cache[value] = result;
      return result;
    }

    if ((!isa<PointwiseOpInterface>(op) && !isa<IotaOp>(op)) ||
        op->getNumRegions() != 0) {
      return op->emitError(
          "strip mining cannot clone this contraction producer");
    }
    IRMapping mapping;
    for (Value operand : op->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType())) {
        mapping.map(operand, operand);
        continue;
      }
      FailureOr<Value> clonedOperand = clone(operand);
      if (failed(clonedOperand)) {
        return failure();
      }
      mapping.map(operand, *clonedOperand);
    }
    Operation *cloned = builder.clone(*op, mapping);
    for (auto [original, result] :
         llvm::zip(op->getResults(), cloned->getResults())) {
      if (auto type = dyn_cast<RankedTensorType>(original.getType())) {
        result.setType(RankedTensorType::get(tileShape, type.getElementType()));
      }
      cache[original] = result;
    }
    return cache.lookup(value);
  }

private:
  ArrayRef<int64_t> tileShape;
  const DenseMap<unsigned, Value> &contractingCoordinates;
  TilePaddingAttr loadPadding;
  OpBuilder &builder;
  DenseMap<Value, Value> cache;
};

/// Recursively collect all producer operations upstream of a value up to LoadOps.
static void collectProducerOps(Value value, SmallVectorImpl<Operation *> &ops,
                               DenseSet<Operation *> &seen) {
  Operation *op = value.getDefiningOp();
  if (!op || !seen.insert(op).second) {
    return;
  }
  if (isa<LoadOp>(op)) {
    ops.push_back(op);
    return;
  }
  for (Value operand : op->getOperands()) {
    if (isa<RankedTensorType>(operand.getType())) {
      collectProducerOps(operand, ops, seen);
    }
  }
  ops.push_back(op);
}

/// Erase upstream producer operations that have become dead after their
/// strip-mined counterparts were cloned inside contraction loops.
static void eraseDeadProducerOps(ValueRange values) {
  DenseSet<Operation *> seen;
  SmallVector<Operation *> producerOps;
  for (Value value : values) {
    collectProducerOps(value, producerOps, seen);
  }
  for (Operation *op : llvm::reverse(producerOps)) {
    if (op->use_empty()) {
      op->erase();
    }
  }
}

/// Check whether an upstream operand DAG contains another contraction operation
/// (ReduceOp, ReduceUDOp, or MatmulOp).
///
/// When nested contractions exist, the inner contraction must be strip-mined
/// first. Deferring the outer contraction avoids cloning its loop nest across
/// another contracting dimension.
static bool hasNestedContraction(Value value, DenseSet<Operation *> &seen) {
  Operation *op = value.getDefiningOp();
  if (!op || !seen.insert(op).second) {
    return false;
  }
  if (isa<ReduceOp, ReduceUDOp, MatmulOp>(op)) {
    return true;
  }
  return llvm::any_of(op->getOperands(), [&](Value operand) {
    return isa<RankedTensorType>(operand.getType()) &&
           hasNestedContraction(operand, seen);
  });
}

/// Lift a scalar user-defined reduction body (ReduceUDOp) into elementwise
/// tensor operations operating over tile-shaped values.
///
/// Maps scalar block arguments to tensor tiles (inserting UnrealizedConversionCastOp
/// if types differ), clones scalar arith operations with ranked tensor result
/// types, and maps the terminator yields to updated accumulator tiles.
static FailureOr<SmallVector<Value>>
liftReductionBody(ReduceUDOp reduce, ValueRange accumulators, ValueRange tiles,
                  ArrayRef<int64_t> tileShape, OpBuilder &builder) {
  Block &body = reduce.getBodyRegion().front();
  size_t inputCount = reduce.getNumOperands();
  if (body.getNumArguments() != inputCount * 2 ||
      accumulators.size() != inputCount || tiles.size() != inputCount) {
    return reduce.emitError(
        "user-defined reduction body does not match its inputs");
  }

  IRMapping mapping;
  for (size_t index = 0; index < inputCount; ++index) {
    auto mapBodyArgument = [&](BlockArgument argument,
                               Value tensor) -> LogicalResult {
      auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
      if (!tensorType) {
        return reduce.emitError(
            "strip-mined reduction body expects tensor values");
      }
      auto liftedType = RankedTensorType::get(tileShape, argument.getType());
      if (tensorType != liftedType) {
        tensor = UnrealizedConversionCastOp::create(
                     builder, reduce.getLoc(), TypeRange{liftedType}, tensor)
                     .getResult(0);
      }
      mapping.map(argument, tensor);
      return success();
    };
    if (failed(mapBodyArgument(body.getArgument(index), accumulators[index])) ||
        failed(mapBodyArgument(body.getArgument(inputCount + index),
                               tiles[index]))) {
      return failure();
    }
  }

  for (Operation &operation : body.without_terminator()) {
    if (operation.getDialect() !=
            builder.getContext()->getLoadedDialect<arith::ArithDialect>() ||
        operation.getNumRegions() != 0) {
      return operation.emitError(
          "strip mining only supports arithmetic operations in a "
          "user-defined reduction body");
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      if (!constant.getType().isIntOrFloat()) {
        return constant.emitError(
            "strip mining requires an integer or floating-point constant");
      }
      auto type = RankedTensorType::get(tileShape, constant.getType());
      auto value = cast<TypedAttr>(DenseElementsAttr::get(
          type, ArrayRef<Attribute>{constant.getValue()}));
      auto tensorConstant =
          arith::ConstantOp::create(builder, constant.getLoc(), type, value);
      tensorConstant->setAttr("nv_tensor_ir.strip_mined_reduction_body",
                              builder.getUnitAttr());
      mapping.map(constant.getResult(), tensorConstant);
      continue;
    }
    if (operation.getNumResults() != 1 ||
        !operation.getResult(0).getType().isIntOrFloat()) {
      return operation.emitError(
          "strip mining requires single-result scalar arithmetic");
    }
    Operation *clone = builder.clone(operation, mapping);
    clone->setAttr("nv_tensor_ir.strip_mined_reduction_body",
                   builder.getUnitAttr());
    clone->getResult(0).setType(
        RankedTensorType::get(tileShape, operation.getResult(0).getType()));
    mapping.map(operation.getResult(0), clone->getResult(0));
  }

  auto yield = cast<YieldOp>(body.getTerminator());
  SmallVector<Value> results;
  results.reserve(yield.getNumOperands());
  for (auto [index, result] : llvm::enumerate(yield.getOperands())) {
    Value mapped = mapping.lookupOrNull(result);
    if (!mapped) {
      return yield.emitError(
          "user-defined reduction yield is not produced by its body");
    }
    Type accumulatorType = accumulators[index].getType();
    if (mapped.getType() != accumulatorType) {
      mapped = UnrealizedConversionCastOp::create(
                   builder, yield.getLoc(), TypeRange{accumulatorType}, mapped)
                   .getResult(0);
    }
    results.push_back(mapped);
  }
  return results;
}

/// Strip-mine a built-in ReduceOp:
/// 1. Validate contracting dimensions and derive strip-mined tile shape.
/// 2. If the reduction already fits in one tile or has nested contractions, skip.
/// 3. Initialize accumulator tensor with splatted reduction identity.
/// 4. Construct scf.for loop nest over contracting dimensions.
/// 5. Clone producer DAG with shrunken tile shape and loop induction variables.
/// 6. Apply prologue (abs, square, zero-filtering) and accumulation step.
/// 7. Wire yields, clean up dead producers, and apply epilogue (avg, norm2).
static LogicalResult stripMineReduction(ReduceOp reduce,
                                        int64_t reductionTileSize) {
  // 1. Plan strip-mining and check whether contraction fits in a single tile.
  auto inputType = cast<RankedTensorType>(reduce.getInput().getType());
  SmallVector<int32_t> dimensions(reduce.getDimensions().begin(),
                                  reduce.getDimensions().end());
  FailureOr<ReductionTilingPlan> planOr = createReductionTilingPlan(
      reduce, inputType, dimensions, reductionTileSize);
  if (failed(planOr)) {
    return failure();
  }
  ReductionTilingPlan plan = std::move(*planOr);
  ArrayRef<int64_t> reductionShape = plan.reductionShape;
  ArrayRef<int64_t> contractionTileShape = plan.contractionTileShape;
  if (contractionTileShape == reductionShape) {
    return success();
  }
  DenseSet<Operation *> contractionSearch;
  if (hasNestedContraction(reduce.getInput(), contractionSearch)) {
    // The inner contraction is strip-mined independently. Retaining this
    // coarse outer reduction avoids cloning its loop nest across another
    // contracting dimension.
    return success();
  }

  int64_t reductionSize = 1;
  for (int64_t extent : reductionShape) {
    if (llvm::MulOverflow(reductionSize, extent, reductionSize)) {
      return reduce.emitError("reduction extent overflows index range");
    }
  }

  // 2. Initialize the partial accumulator tile with the reduction identity.
  OpBuilder builder(reduce);
  Location loc = reduce.getLoc();
  ArrayRef<int64_t> tileShape = plan.operandTileShape;
  auto tileType = RankedTensorType::get(tileShape, inputType.getElementType());
  Attribute identity = getIdentity(builder, reduce.getReductionMode(),
                                   inputType.getElementType());
  if (!identity) {
    return reduce.emitError("unsupported reduction mode for strip mining");
  }
  TilePaddingAttr loadPadding =
      getReductionLoadPadding(builder, reduce, identity);
  Value initial = createSplat(builder, loc, tileType, identity);

  // 3. Build the scf.for loop nest over contracting dimensions.
  ContractionLoopNest loopNest = createContractionLoopNest(
      builder, loc, reductionShape, contractionTileShape, ValueRange{initial},
      /*buildMask=*/!loadPadding);
  DenseMap<unsigned, Value> contractingCoordinates =
      mapContractionCoordinates(plan.dimensions, loopNest.coordinates);
  Value mask = loopNest.mask;
  if (mask) {
    mask = reshapeContractionMask(builder, loc, mask, tileShape,
                                  plan.dimensions, contractionTileShape);
  }

  // 4. Clone the upstream producer DAG inside the loop with strip-mined shapes.
  ProducerCloner cloner(tileShape, contractingCoordinates, loadPadding,
                        builder);
  FailureOr<Value> tile = cloner.clone(reduce.getInput());
  if (failed(tile)) {
    return failure();
  }
  if (!loadPadding) {
    mask = broadcastContractionMask(builder, loc, mask, tileShape);
    *tile = applyPaddingMask(builder, loc, *tile, mask, identity);
  }

  Value oldInput = reduce.getInput();
  if (loopNest.loops.empty()) {
    reduce.getInputMutable().assign(*tile);
    eraseDeadProducerOps(ValueRange{oldInput});
    return success();
  }

  // 5. Apply prologue, combine with carried accumulator, and yield.
  Value reductionTile =
      createReductionPrologue(builder, loc, reduce.getReductionMode(), *tile);
  if (reductionTile != *tile) {
    reduce->setAttr("nv_tensor_ir.reduction_prologue_applied",
                    builder.getUnitAttr());
  }
  Value combined =
      createAccumulatorStep(builder, loc, reduce.getReductionMode(),
                            loopNest.carried.front(), reductionTile);
  if (!combined) {
    return reduce.emitError("unsupported reduction accumulation mode");
  }
  setContractionLoopYields(loopNest.loops, combined);

  reduce.getInputMutable().assign(loopNest.loops.front().getResult(0));
  eraseDeadProducerOps(ValueRange{oldInput});

  // 6. Handle degenerate local reductions or apply normalization epilogues.
  bool hasDegenerateLocalReduction = llvm::all_of(
      contractionTileShape, [](int64_t extent) { return extent == 1; });
  if (hasDegenerateLocalReduction) {
    // Every local contraction extent is one, so the loop accumulation has
    // already completed the reduction and the residual ReduceOp can be erased.
    builder.setInsertionPoint(reduce);
    Value result = createDegenerateReductionEpilogue(
        builder, loc, reduce.getReductionMode(),
        loopNest.loops.front().getResult(0), reductionSize);
    if (result.getType() != reduce.getType()) {
      result = UnrealizedConversionCastOp::create(
                   builder, loc, TypeRange{reduce.getType()}, result)
                   .getResult(0);
    }
    reduce.replaceAllUsesWith(result);
    reduce.erase();
  } else if (reduce.getReductionMode() == ReductionMode::avg) {
    reduce.setReductionMode(ReductionMode::add);
    builder.setInsertionPointAfter(reduce);
    Value result = createDegenerateReductionEpilogue(
        builder, loc, ReductionMode::avg, reduce.getResult(), reductionSize);
    reduce.getResult().replaceAllUsesExcept(result, result.getDefiningOp());
  }
  return success();
}

/// Strip-mine a user-defined ReduceUDOp:
/// 1. Validate contracting dimensions and derive strip-mined tile shape.
/// 2. Initialize accumulators for each input using its corresponding identity.
/// 3. Construct scf.for loop nest over contracting dimensions.
/// 4. Clone producer DAGs for all inputs with shrunken tile shapes.
/// 5. Lift the user-defined reduction body from scalar arithmetic to tensor tiles.
/// 6. Wire yields, update reduce inputs, and clean up dead producers.
static LogicalResult stripMineReduction(ReduceUDOp reduce,
                                        int64_t reductionTileSize) {
  // 1. Plan strip-mining and check whether contraction fits in a single tile.
  if (reduce.getInputs().empty()) {
    return reduce.emitError("user-defined reduction has no inputs");
  }
  auto inputType = cast<RankedTensorType>(reduce.getInputs().front().getType());
  SmallVector<int32_t> dimensions(reduce.getDimensions().begin(),
                                  reduce.getDimensions().end());
  FailureOr<ReductionTilingPlan> planOr = createReductionTilingPlan(
      reduce, inputType, dimensions, reductionTileSize);
  if (failed(planOr)) {
    return failure();
  }
  ReductionTilingPlan plan = std::move(*planOr);
  ArrayRef<int64_t> reductionShape = plan.reductionShape;
  ArrayRef<int64_t> contractionTileShape = plan.contractionTileShape;
  if (contractionTileShape == reductionShape) {
    return success();
  }
  for (Value input : reduce.getInputs()) {
    DenseSet<Operation *> contractionSearch;
    if (hasNestedContraction(input, contractionSearch)) {
      return success();
    }
  }

  // 2. Initialize partial accumulators for each input using its identity.
  OpBuilder builder(reduce);
  Location loc = reduce.getLoc();
  ArrayRef<int64_t> tileShape = plan.operandTileShape;

  if (reduce.getIdentity().size() != reduce.getNumOperands()) {
    return reduce.emitError(
        "user-defined reduction requires one identity per input");
  }
  TilePaddingAttr loadPadding = getReductionLoadPadding(builder, reduce);
  SmallVector<Value> carried;
  carried.reserve(reduce.getNumOperands());
  for (auto [input, identity] :
       llvm::zip_equal(reduce.getInputs(), reduce.getIdentity())) {
    Type elementType = cast<RankedTensorType>(input.getType()).getElementType();
    carried.push_back(createSplat(
        builder, loc, RankedTensorType::get(tileShape, elementType), identity));
  }

  // 3. Build the scf.for loop nest over contracting dimensions.
  ContractionLoopNest loopNest = createContractionLoopNest(
      builder, loc, reductionShape, contractionTileShape, carried,
      /*buildMask=*/!loadPadding);
  DenseMap<unsigned, Value> contractingCoordinates =
      mapContractionCoordinates(plan.dimensions, loopNest.coordinates);
  Value mask = loopNest.mask;
  if (mask) {
    mask = reshapeContractionMask(builder, loc, mask, tileShape,
                                  plan.dimensions, contractionTileShape);
  }

  // 4. Clone upstream producer DAGs for all inputs with strip-mined shapes.
  ProducerCloner cloner(tileShape, contractingCoordinates, loadPadding,
                        builder);
  SmallVector<Value> tiles;
  tiles.reserve(reduce.getNumOperands());
  for (Value input : reduce.getInputs()) {
    FailureOr<Value> tile = cloner.clone(input);
    if (failed(tile)) {
      return failure();
    }
    tiles.push_back(*tile);
  }
  if (!loadPadding) {
    mask = broadcastContractionMask(builder, loc, mask, tileShape);
    for (auto [index, identity] : llvm::enumerate(reduce.getIdentity())) {
      tiles[index] =
          applyPaddingMask(builder, loc, tiles[index], mask, identity);
    }
  }

  SmallVector<Value> oldInputs(reduce.getInputs());
  if (loopNest.loops.empty()) {
    reduce.getInputsMutable().assign(tiles);
    eraseDeadProducerOps(oldInputs);
    return success();
  }

  // 5. Lift the scalar user-defined reduction body to operate on tensor tiles.
  FailureOr<SmallVector<Value>> combined =
      liftReductionBody(reduce, loopNest.carried, tiles, tileShape, builder);
  if (failed(combined)) {
    return failure();
  }
  setContractionLoopYields(loopNest.loops, *combined);

  // 6. Wire yields through loop nest and clean up dead producers.
  reduce.getInputsMutable().assign(loopNest.loops.front().getResults());
  eraseDeadProducerOps(oldInputs);
  return success();
}

static DenseI64ArrayAttr getMatmulArray(MatmulOp matmul, StringRef name) {
  return matmul->getAttrOfType<DenseI64ArrayAttr>(name);
}

/// Strip temporary formation metadata attributes attached to a MatmulOp and its
/// input reshapes during FormGridPass.
///
/// Once strip-mining has formed explicit contraction loops and shapes, these
/// temporary annotations are dead and must be removed before downstream conversion.
static void removeMatmulFormationAttrs(MatmulOp matmul) {
  matmul->removeAttr("nv_tensor_ir.contraction_shape");
  matmul->removeAttr("nv_tensor_ir.lhs_source_shape");
  matmul->removeAttr("nv_tensor_ir.rhs_source_shape");
  matmul->removeAttr("nv_tensor_ir.lhs_contracting_dimensions");
  matmul->removeAttr("nv_tensor_ir.rhs_contracting_dimensions");
  for (Value operand : {matmul.getA(), matmul.getB()}) {
    if (Operation *producer = operand.getDefiningOp()) {
      producer->removeAttr("nv_tensor_ir.matmul_input_reshape");
    }
  }
}

/// Strip-mine a matrix multiplication (MatmulOp):
/// 1. Read formation attributes attached by FormGridPass (contraction shape,
///    source shapes, contracting dimension indices).
/// 2. Calculate strip-mined contraction tile shape for contracting dimension K.
/// 3. Unwrap formation input reshapes and determine load padding neutrality.
/// 4. Initialize result accumulator tensor with zero.
/// 5. Construct scf.for loop nest over contracting K dimensions.
/// 6. Clone LHS and RHS producer DAGs with shrunken K dimension and loop variables.
/// 7. Apply padding masks if load padding is not neutral.
/// 8. Reshape LHS/RHS to canonical MMA shapes ([..., tileK] and [..., tileK, N]).
/// 9. Emit inner MatmulOp with carried accumulator, wire yields, and replace original.
static LogicalResult stripMineMatmul(MatmulOp matmul,
                                     int64_t contractionTileSize) {
  // 1. Unpack and validate matmul formation metadata.
  DenseI64ArrayAttr contractionShapeAttr =
      getMatmulArray(matmul, "nv_tensor_ir.contraction_shape");
  DenseI64ArrayAttr lhsSourceShapeAttr =
      getMatmulArray(matmul, "nv_tensor_ir.lhs_source_shape");
  DenseI64ArrayAttr rhsSourceShapeAttr =
      getMatmulArray(matmul, "nv_tensor_ir.rhs_source_shape");
  DenseI64ArrayAttr lhsContractingDimsAttr =
      getMatmulArray(matmul, "nv_tensor_ir.lhs_contracting_dimensions");
  DenseI64ArrayAttr rhsContractingDimsAttr =
      getMatmulArray(matmul, "nv_tensor_ir.rhs_contracting_dimensions");
  if (!contractionShapeAttr || !lhsSourceShapeAttr || !rhsSourceShapeAttr ||
      !lhsContractingDimsAttr || !rhsContractingDimsAttr) {
    return matmul.emitError("strip mining requires matmul formation metadata");
  }

  ArrayRef<int64_t> contractionShape = contractionShapeAttr.asArrayRef();
  ArrayRef<int64_t> lhsContractingDims = lhsContractingDimsAttr.asArrayRef();
  ArrayRef<int64_t> rhsContractingDims = rhsContractingDimsAttr.asArrayRef();
  if (contractionShape.empty() ||
      contractionShape.size() != lhsContractingDims.size() ||
      contractionShape.size() != rhsContractingDims.size()) {
    return matmul.emitError("invalid matmul contraction metadata");
  }

  // 2. Calculate contraction tile shape for dimension K.
  SmallVector<int64_t> contractionTileShape =
      calculateReductionTileShape(contractionShape, contractionTileSize);
  if (contractionTileShape == contractionShape) {
    removeMatmulFormationAttrs(matmul);
    return success();
  }

  // 3. Derive LHS and RHS tile shapes with contracting dimensions shrunken.
  SmallVector<int64_t> lhsSourceShape(lhsSourceShapeAttr.asArrayRef());
  SmallVector<int64_t> rhsSourceShape(rhsSourceShapeAttr.asArrayRef());
  for (auto [dimension, tile] :
       llvm::zip_equal(lhsContractingDims, contractionTileShape)) {
    if (dimension < 0 ||
        static_cast<size_t>(dimension) >= lhsSourceShape.size()) {
      return matmul.emitError("invalid LHS contracting dimension");
    }
    lhsSourceShape[dimension] = tile;
  }
  for (auto [dimension, tile] :
       llvm::zip_equal(rhsContractingDims, contractionTileShape)) {
    if (dimension < 0 ||
        static_cast<size_t>(dimension) >= rhsSourceShape.size()) {
      return matmul.emitError("invalid RHS contracting dimension");
    }
    rhsSourceShape[dimension] = tile;
  }

  auto unwrapFormationReshape = [](Value operand) -> Value {
    if (auto reshape = operand.getDefiningOp<ReshapeOp>();
        reshape && reshape->hasAttr("nv_tensor_ir.matmul_input_reshape")) {
      return reshape.getInput();
    }
    return operand;
  };
  Value lhsSource = unwrapFormationReshape(matmul.getA());
  Value rhsSource = unwrapFormationReshape(matmul.getB());

  // 4. Initialize result accumulator tensor with zeros.
  OpBuilder builder(matmul);
  Location loc = matmul.getLoc();
  TilePaddingAttr lhsLoadPadding;
  if (isLoadPaddingNeutral(lhsSource)) {
    lhsLoadPadding =
        TilePaddingAttr::get(builder.getContext(), TilePadding::Zero);
  }
  TilePaddingAttr rhsLoadPadding;
  if (isLoadPaddingNeutral(rhsSource)) {
    rhsLoadPadding =
        TilePaddingAttr::get(builder.getContext(), TilePadding::Zero);
  }
  auto resultType = cast<RankedTensorType>(matmul.getType());
  Type elementType = resultType.getElementType();
  Attribute zero = elementType.isFloat()
                       ? Attribute(builder.getFloatAttr(elementType, 0.0))
                       : Attribute(builder.getIntegerAttr(elementType, 0));
  Value carried = createSplat(builder, loc, resultType, zero);
  bool maskNeeded = !lhsLoadPadding || !rhsLoadPadding;
  for (auto [extent, tile] :
       llvm::zip_equal(contractionShape, contractionTileShape)) {
    if (extent <= 0 || tile <= 0) {
      return matmul.emitError(
          "matmul contracting extents and tiles must be positive");
    }
  }

  // 5. Build scf.for loop nest over contracting K dimensions.
  ContractionLoopNest loopNest = createContractionLoopNest(
      builder, loc, contractionShape, contractionTileShape, ValueRange{carried},
      maskNeeded);
  DenseMap<unsigned, Value> lhsCoordinates =
      mapContractionCoordinates(lhsContractingDims, loopNest.coordinates);
  DenseMap<unsigned, Value> rhsCoordinates =
      mapContractionCoordinates(rhsContractingDims, loopNest.coordinates);
  Value contractionMask = loopNest.mask;
  carried = loopNest.carried.front();

  // 6. Clone LHS and RHS producer DAGs with shrunken K dimension.
  Value lhsMask;
  if (!lhsLoadPadding) {
    lhsMask =
        reshapeContractionMask(builder, loc, contractionMask, lhsSourceShape,
                               lhsContractingDims, contractionTileShape);
  }
  Value rhsMask;
  if (!rhsLoadPadding) {
    rhsMask =
        reshapeContractionMask(builder, loc, contractionMask, rhsSourceShape,
                               rhsContractingDims, contractionTileShape);
  }
  ProducerCloner lhsCloner(lhsSourceShape, lhsCoordinates, lhsLoadPadding,
                           builder);
  ProducerCloner rhsCloner(rhsSourceShape, rhsCoordinates, rhsLoadPadding,
                           builder);
  FailureOr<Value> lhs = lhsCloner.clone(lhsSource);
  FailureOr<Value> rhs = rhsCloner.clone(rhsSource);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }

  // 7. Apply masks if load padding is not neutral.
  if (maskNeeded) {
    if (lhsMask) {
      lhsMask = broadcastContractionMask(builder, loc, lhsMask, lhsSourceShape);
      Type lhsElementType =
          cast<RankedTensorType>((*lhs).getType()).getElementType();
      Attribute zero =
          lhsElementType.isFloat()
              ? Attribute(builder.getFloatAttr(lhsElementType, 0.0))
              : Attribute(builder.getIntegerAttr(lhsElementType, 0));
      *lhs = applyPaddingMask(builder, loc, *lhs, lhsMask, zero);
    }
    if (rhsMask) {
      rhsMask = broadcastContractionMask(builder, loc, rhsMask, rhsSourceShape);
      Type rhsElementType =
          cast<RankedTensorType>((*rhs).getType()).getElementType();
      Attribute zero =
          rhsElementType.isFloat()
              ? Attribute(builder.getFloatAttr(rhsElementType, 0.0))
              : Attribute(builder.getIntegerAttr(rhsElementType, 0));
      *rhs = applyPaddingMask(builder, loc, *rhs, rhsMask, zero);
    }
  }

  // 8. Reshape operands to canonical MMA layout and emit inner MatmulOp.
  int64_t tileK = llvm::product_of(contractionTileShape);
  SmallVector<int64_t> lhsMmaShape(
      cast<RankedTensorType>(matmul.getA().getType()).getShape());
  SmallVector<int64_t> rhsMmaShape(
      cast<RankedTensorType>(matmul.getB().getType()).getShape());
  lhsMmaShape.back() = tileK;
  rhsMmaShape[rhsMmaShape.size() - 2] = tileK;
  auto reshape = [&](Value value, ArrayRef<int64_t> shape) -> Value {
    auto type = cast<RankedTensorType>(value.getType());
    if (type.getShape() == shape) {
      return value;
    }
    return ReshapeOp::create(
        builder, loc, RankedTensorType::get(shape, type.getElementType()),
        value);
  };
  Value localLhs = reshape(*lhs, lhsMmaShape);
  Value localRhs = reshape(*rhs, rhsMmaShape);
  Value oldLhs = matmul.getA();
  Value oldRhs = matmul.getB();
  if (loopNest.loops.empty()) {
    matmul->setOperand(0, localLhs);
    matmul->setOperand(1, localRhs);
    removeMatmulFormationAttrs(matmul);
    eraseDeadProducerOps(ValueRange{oldLhs, oldRhs});
    return success();
  }

  Value localResult =
      MatmulOp::create(builder, loc, resultType, localLhs, localRhs, carried);
  setContractionLoopYields(loopNest.loops, localResult);

  // 9. Wire yields and replace original MatmulOp.
  Value result = loopNest.loops.front().getResult(0);
  matmul.replaceAllUsesWith(result);
  matmul.erase();
  eraseDeadProducerOps(ValueRange{oldLhs, oldRhs});
  return success();
}

/// Limit the hardware grid and let each block visit the remaining flattened
/// tile IDs serially.
///
/// Static persistence caps the launch grid at `sm_count * occupancy` blocks.
/// If the total number of tiles exceeds this limit, the scf.forall upper bound
/// is clamped to the persistent limit, and the entire per-tile computation in the
/// forall body is wrapped in a serial grid-stride loop:
///   scf.for %tileId = %blockId to %totalTiles step %gridSize
/// Uses of the original block ID induction variable inside the body are rewired
/// to the serial loop's induction variable.
///
/// Dynamic persistence intentionally remains on the legacy no-op path; static
/// persistence is useful even when the logical grid extent is only known at
/// runtime.
static LogicalResult formStaticPersistence(func::FuncOp function,
                                           int64_t smCount, int64_t occupancy) {
  // 1. Verify static persistent grid limit does not overflow 32-bit index.
  int64_t persistentGridLimit;
  constexpr int64_t maxI32 = std::numeric_limits<int32_t>::max();
  if (llvm::MulOverflow(smCount, occupancy, persistentGridLimit) ||
      persistentGridLimit > maxI32) {
    return function.emitError(
        "static persistent grid size overflows 32-bit index range");
  }

  // 2. For each rank-one destination-free scf.forall, clamp the grid bound.
  SmallVector<scf::ForallOp> foralls;
  function.walk([&](scf::ForallOp forallOp) { foralls.push_back(forallOp); });
  for (scf::ForallOp forallOp : foralls) {
    if (forallOp.getRank() != 1 || !forallOp.getOutputs().empty()) {
      return forallOp.emitError(
          "static persistence requires a rank-one destination-free forall");
    }

    OpBuilder outerBuilder(forallOp);
    Location loc = forallOp.getLoc();
    OpFoldResult mixedTotalTiles = forallOp.getMixedUpperBound().front();
    Value totalTiles;
    Value gridSize;
    if (auto staticTotal = mixedTotalTiles.dyn_cast<Attribute>()) {
      int64_t total = cast<IntegerAttr>(staticTotal).getInt();
      if (total <= persistentGridLimit) {
        continue;
      }
      int64_t grid = std::max<int64_t>(1, persistentGridLimit);
      totalTiles =
          arith::ConstantIndexOp::create(outerBuilder, loc, total).getResult();
      gridSize =
          arith::ConstantIndexOp::create(outerBuilder, loc, grid).getResult();
      forallOp.setStaticUpperBound({grid});
    } else {
      totalTiles = mixedTotalTiles.dyn_cast<Value>();
      Value limit = arith::ConstantIndexOp::create(outerBuilder, loc,
                                                   persistentGridLimit);
      Value one = arith::ConstantIndexOp::create(outerBuilder, loc, 1);
      Value bounded =
          arith::MinUIOp::create(outerBuilder, loc, totalTiles, limit);
      gridSize = arith::MaxUIOp::create(outerBuilder, loc, bounded, one);
      forallOp.getDynamicUpperBoundMutable().assign(gridSize);
    }

    // 3. Wrap the per-tile computation in a grid-stride scf.for loop.
    Block *body = forallOp.getBody();
    SmallVector<Operation *> bodyOps;
    for (Operation &op : body->without_terminator()) {
      bodyOps.push_back(&op);
    }
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(body);
    auto serialLoop = scf::ForOp::create(
        bodyBuilder, loc, forallOp.getInductionVar(0), totalTiles, gridSize);
    Operation *serialTerminator = serialLoop.getBody()->getTerminator();
    for (Operation *op : bodyOps) {
      op->moveBefore(serialTerminator);
    }
    forallOp.getInductionVar(0).replaceUsesWithIf(
        serialLoop.getInductionVar(), [&](OpOperand &use) {
          return serialLoop->isProperAncestor(use.getOwner());
        });
  }
  return success();
}

struct TileReductionsPass
    : public impl::TileReductionsPassBase<TileReductionsPass> {
  using Base = impl::TileReductionsPassBase<TileReductionsPass>;
  using Base::Base;

  void runOnOperation() override {
    // 1. Validate pass configuration options.
    if (reduction_tile_size < 1 ||
        !llvm::isPowerOf2_64(static_cast<uint64_t>(reduction_tile_size))) {
      getOperation().emitError()
          << "reduction-tile-size must be a positive power of two, got "
          << static_cast<int64_t>(reduction_tile_size);
      signalPassFailure();
      return;
    }
    if (persistence != "none" && persistence != "static" &&
        persistence != "dynamic") {
      getOperation().emitError()
          << "persistence must be none, static, or dynamic, got '"
          << persistence << "'";
      signalPassFailure();
      return;
    }
    if (occupancy < 1 || (persistence == "static" && sm_count < 1)) {
      getOperation().emitError(
          "static persistence requires sm-count >= 1 and occupancy >= 1");
      signalPassFailure();
      return;
    }

    // 2. Collect and strip-mine all contraction operations in the function.
    SmallVector<ReduceOp> reductions;
    SmallVector<ReduceUDOp> userDefinedReductions;
    SmallVector<MatmulOp> matmuls;
    getOperation().walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
    getOperation().walk(
        [&](ReduceUDOp reduce) { userDefinedReductions.push_back(reduce); });
    getOperation().walk([&](MatmulOp matmul) { matmuls.push_back(matmul); });
    for (ReduceOp reduce : reductions) {
      if (failed(stripMineReduction(reduce, reduction_tile_size))) {
        signalPassFailure();
        return;
      }
    }
    for (ReduceUDOp reduce : userDefinedReductions) {
      if (failed(stripMineReduction(reduce, reduction_tile_size))) {
        signalPassFailure();
        return;
      }
    }
    for (MatmulOp matmul : matmuls) {
      if (failed(stripMineMatmul(matmul, reduction_tile_size))) {
        signalPassFailure();
        return;
      }
    }

    // 3. Apply static persistence to limit the launch grid and serialize tiles.
    if (persistence == "static" &&
        failed(formStaticPersistence(getOperation(), sm_count, occupancy))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::nv_tensor_ir
