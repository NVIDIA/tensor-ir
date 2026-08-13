// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"

#include "llvm/ADT/APSInt.h" // IWYU pragma: keep
#include "llvm/ADT/TypeSwitch.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace mlir::nv_tensor_ir {
namespace tensor_to_cuda_tile {
namespace {

namespace tcg = mlir::nv_tensor_ir::tcutegen;

/// Helper for extracting the tensor sources from a layout attribute.
/// Only process the layouts within the iteration space boundary.
void extractTensorSources(LayoutSourceAttrInterface layout,
                          llvm::MapVector<TensorSourceAttr, Value> &result) {
  llvm::TypeSwitch<LayoutSourceAttrInterface>(layout)
      .Case<TensorSourceAttr>([&](auto attr) {
        // Skip constant/iota sources (tensorId < 0).
        if (attr.getTensorId() >= 0) {
          result[attr] = nullptr;
        }
      })
      .Case<CompositeSourceAttr>([&](auto attr) {
        for (auto child : attr.getSources()) {
          extractTensorSources(child, result);
        }
      });
}

llvm::MapVector<TensorSourceAttr, Value>
extractTensorSources(LayoutSourceAttrInterface layout) {
  llvm::MapVector<TensorSourceAttr, Value> result;
  extractTensorSources(layout, result);
  return result;
}

//===----------------------------------------------------------------------===//
// Concatenation operation
//===----------------------------------------------------------------------===//

/// Concatenation operation is implemented as a chain of if/else statements,
/// where each generated block corresponds to an input tile of the concat.
/// The index variable corresponding to the concatenated dimension defines
/// which branch is executed.
///
/// Example: concatenate(A, B, C) dimension=1
/// with shapes: A[4,3], B[4,5], C[4,7]
/// generates pseudo-code:
/// if (%i1 < 3) {
///   [[block for A]], where %i1 is used for indexing loads
/// } else {
///   %i1' = %i1 - 3;
///   if (%i1' < 5) {
///     [[block for B]], where %i1' is used for indexing loads
///   } else {
///     %i1" = %i1' - 5;
///     [[block for C]], where %i1" is used for indexing loads
///   }
/// }
FailureOr<BlockStructure>
buildConcatBlockStructure(RewriterBase &rewriter, ConcatenateOp concatOp,
                          const IterationSpace &iterSpace,
                          const TypeConverter &typeConverter) {
  // Verify that the layout is a concat layout.
  auto concatSource = dyn_cast_if_present<ConcatSourceAttr>(
      concatOp->getAttrOfType<LayoutSourceAttrInterface>(
          TensorIRDialect::getLayoutAttrName()));
  if (!concatSource) {
    return concatOp.emitError("Expected concat layout");
  }

  // The tile shape must have unit size in the concat dimension.
  int64_t concatDim = concatSource.getDimension();
  int64_t concatTileSize = iterSpace.tileShape[concatDim];
  if (concatTileSize != 1) {
    return concatOp.emitError()
           << "Concat dimension " << concatDim
           << " must have tile size 1, but got " << concatTileSize;
  }

  // Infer the resulting tile type.
  Type convertedType =
      typeConverter.convertType(concatOp.getOutput().getType());
  Type resultType = cast<ShapedType>(convertedType).clone(iterSpace.tileShape);

  // Construct the iteration spaces for each operand.
  BlockStructure result;
  SmallVector<Value> index = iterSpace.indexValues;

  auto indexTileType = cast<ShapedType>(index[concatDim].getType());
  Type indexElemType = indexTileType.getElementType();

  // Build if/else branches for all the operands, except the last one.
  Location loc = concatOp.getLoc();
  size_t last = concatSource.size() - 1;
  for (size_t i = 0; i < last; i++) {
    // Create the threshold value for the index.
    int64_t threshold = concatSource.getSource(i).getShape()[concatDim];
    auto denseAttr = cast<DenseTypedElementsAttr>(DenseElementsAttr::get(
        indexTileType, rewriter.getIntegerAttr(indexElemType, threshold)));
    auto thresholdValue =
        cuda_tile::ConstantOp::create(rewriter, loc, indexTileType, denseAttr);

    // Compare the current index value with the threshold value.
    Value predicate = cuda_tile::CmpIOp::create(
        rewriter, loc, cuda_tile::ComparisonPredicate::LESS_THAN,
        index[concatDim], thresholdValue, cuda_tile::Signedness::Unsigned);

    // Create the branch with two blocks.
    auto ifOp = cuda_tile::IfOp::create(rewriter, loc, {resultType}, predicate);
    Block &thenBlock = ifOp.getThenRegion().emplaceBlock();
    Block &elseBlock = ifOp.getElseRegion().emplaceBlock();

    // Create the yield op (inner) or store the result value (outer).
    if (i > 0) {
      cuda_tile::YieldOp::create(rewriter, loc, ifOp.getResults());
    } else {
      result.yieldValues = ifOp.getResults();
    }

    // Add the iteration space for the current operand.
    result.iterationSpaces.push_back(
        {&thenBlock, iterSpace.tileShape, index,
         extractTensorSources(concatSource.getSource(i))});

    // Update the index value for the next operand.
    rewriter.setInsertionPointToStart(&elseBlock);
    index[concatDim] = cuda_tile::SubIOp::create(
        rewriter, loc, index[concatDim], thresholdValue);
  }

  // Add the iteration space for the last operand.
  result.iterationSpaces.push_back(
      {rewriter.getBlock(), iterSpace.tileShape, std::move(index),
       extractTensorSources(concatSource.getSource(last))});
  return result;
}

//===----------------------------------------------------------------------===//
// Reduction operation
//===----------------------------------------------------------------------===//

/// Calculate the reduction tile shape by distributing the reduction tile size
/// evenly between the contracting dimensions.
SmallVector<int64_t>
calculateReductionTileShape(ArrayRef<int64_t> reductionShape,
                            int64_t reductionTileSize) {
  SmallVector<int64_t> tileShape;

  for (auto [idx, size] : llvm::enumerate(reductionShape)) {
    // Calculate the maximum tile size for the dimension.
    // Make sure that the dimension size is divisible by the tile size.
    assert(size > 0 && "Dimension size must be positive");
    int64_t maxTileSize = 1ull << llvm::countr_zero(uint64_t(size));

    // Calculate the average tile size for the remaining dimensions.
    size_t remainingDims = reductionShape.size() - idx;
    int bitsPerDim = llvm::divideCeil(
        llvm::countr_zero(uint64_t(reductionTileSize)), remainingDims);
    int64_t meanTileSize = 1ull << bitsPerDim;

    // Use the minimum of the two values as the dimension tile size.
    tileShape.push_back(std::min(meanTileSize, maxTileSize));
    reductionTileSize /= tileShape.back();
  }

  return tileShape;
}

/// Reduction operation is implemented as nested loops for each reduction
/// dimension. The loop induction variables are used to index the loads in the
/// contracting dimensions.
///
/// Example: reduce(T) <dimensions=[1,2], reduction_mode=<add>>
/// with shape: T[64,256,256] and reduction tile size: 128
/// generates pseudo-code:
/// %acc = constant <f32: 0.0> : tile<Nx16x8xf32>
/// for %i1 in range(0, 16) {
///   for %i2 in range(0, 32) {
///     [[block for T]], where (%i1,%i2) are used for indexing loads
///     yield %acc + tile
///   }
/// }
/// %in = reshape %acc : tile<Nx16x8xf32> -> tile<Nx128xf32>
/// %out = reduce %in dim=1 identities=[0.0 : f32] (%lhs, %rhs) {
///   yield %lhs + %rhs
/// }
FailureOr<BlockStructure> buildCommonReduceBlockStructure(
    RewriterBase &rewriter, Operation *op, const IterationSpace &iterSpace,
    int64_t reductionTileSize, ArrayRef<Attribute> identities) {
  // Verify that the layout is a reduction layout.
  auto reductionSource = dyn_cast_if_present<ReductionSourceAttr>(
      op->getAttrOfType<LayoutSourceAttrInterface>(
          TensorIRDialect::getLayoutAttrName()));
  if (!reductionSource) {
    return op->emitError("Expected reduction layout");
  }

  // Reduction tile size must be a power of two.
  if (!llvm::isPowerOf2_64(reductionTileSize)) {
    return op->emitError()
           << "Reduction tile size must be a power of two, but got "
           << reductionTileSize;
  }

  // Build the shape and the index for the underlying source.
  // Copy values from the iteration space, skip the broadcasted dimensions.
  SmallVector<int64_t> tileShape;
  SmallVector<Value> indexValues;
  auto view = reductionSource.getCuteLayout();
  for (size_t i = 0, n = tcg::rank(view) - 1; i < n; i++) {
    if (tcg::get(view, i).stride().as_int() != 0) {
      tileShape.push_back(iterSpace.tileShape[i]);
      indexValues.push_back(iterSpace.indexValues[i]);
    }
  }

  // Calculate the reduction tile shape.
  SmallVector<int64_t> reductionShape = reductionSource.getReductionShape();
  SmallVector<int64_t> reductionTileShape =
      calculateReductionTileShape(reductionShape, reductionTileSize);
  tileShape.append(reductionTileShape);

  // Use I32 type for the index values.
  ShapedType indexType = cuda_tile::TileType::get(
      /*shape=*/llvm::ArrayRef<int64_t>{}, rewriter.getI32Type());

  Location loc = op->getLoc();
  auto createIndex = [&](int64_t value) {
    return createConstant(rewriter, loc, indexType, value);
  };

  // If the whole tile is reduced, no blocks are needed.
  if (reductionTileShape == reductionShape) {
    indexValues.resize(tileShape.size(), createIndex(0));
    IterationSpace innerIterSpace{
        rewriter.getBlock(), std::move(tileShape), std::move(indexValues),
        extractTensorSources(reductionSource.getSource())};
    return BlockStructure{{std::move(innerIterSpace)},
                          /*yieldValues=*/{},
                          /*iterationSpaceIndexForOperand=*/{}};
  }

  // Generate the accumulator tiles.
  SmallVector<Value> accumulatorTiles;
  for (Attribute identity : identities) {
    // Get the underlying type of the identity literal.
    Type identityType;
    if (auto floatAttr = dyn_cast_if_present<FloatAttr>(identity)) {
      identityType = floatAttr.getType();
    } else if (auto integerAttr = dyn_cast_if_present<IntegerAttr>(identity)) {
      identityType = integerAttr.getType();
    } else {
      return op->emitError() << "Unsupported identity type";
    }

    // Create dense attribute with the tile shape.
    ShapedType tileType = cuda_tile::TileType::get(tileShape, identityType);
    auto denseAttr = cast<DenseTypedElementsAttr>(
        DenseElementsAttr::get(tileType, identity));
    Value accumulator =
        cuda_tile::ConstantOp::create(rewriter, loc, tileType, denseAttr);
    accumulatorTiles.push_back(accumulator);
  }

  // Generate the loop nest.
  BlockStructure result;
  Value startIndex = createIndex(0);
  Value stepValue = createIndex(1);
  ArrayRef<Value> accumulators(accumulatorTiles);

  for (auto [size, tileSize] :
       llvm::zip_equal(reductionShape, reductionTileShape)) {
    // If the tile size covers the whole dimension, skip it.
    if (size == tileSize) {
      indexValues.push_back(startIndex);
      continue;
    }

    // Create the loop operation and store the induction variable.
    Value endIndex = createIndex(llvm::divideCeil(size, tileSize));
    auto forOp = cuda_tile::ForOp::create(rewriter, loc, startIndex, endIndex,
                                          stepValue, accumulators);
    indexValues.push_back(forOp.getInductionVar());

    // Update the accumulators for the next iteration.
    auto blockArgs = forOp.getRegionIterValues();
    accumulators = ArrayRef<Value>(blockArgs.begin(), blockArgs.end());

    // Create the continue op (inner) or store the result value (outer).
    if (!result.yieldValues.empty()) {
      cuda_tile::ContinueOp::create(rewriter, loc, forOp.getResults());
    } else {
      result.yieldValues = forOp.getResults();
    }

    // Update the insertion point.
    rewriter.setInsertionPointToEnd(forOp.getBody());
  }

  // Reduction operation has a single iteration space.
  IterationSpace innerIterSpace{
      rewriter.getBlock(), std::move(tileShape), std::move(indexValues),
      extractTensorSources(reductionSource.getSource())};
  result.iterationSpaces.push_back(std::move(innerIterSpace));

  return result;
}

/// Build the block structure for `ReduceOp`, where the identity is implicitly
/// defined by the reduction mode.
FailureOr<BlockStructure>
buildReduceBlockStructure(RewriterBase &rewriter, ReduceOp reduceOp,
                          const IterationSpace &iterSpace,
                          int64_t reductionTileSize) {
  // Get the reduction identity value.
  Type elementType = reduceOp.getInput().getType().getElementType();
  ReductionEmissionHelper reductionEmission{reduceOp.getReductionMode(),
                                            elementType};
  Attribute identity = reductionEmission.getIdentity(rewriter);
  if (!identity) {
    return reduceOp.emitError() << "Unsupported reduction mode";
  }

  // Build the reduction loop nest.
  return buildCommonReduceBlockStructure(rewriter, reduceOp, iterSpace,
                                         reductionTileSize, {identity});
}

/// Build the block structure for `ReduceUDOp`, where the identities are
/// explicitly set in the IR.
FailureOr<BlockStructure>
buildReduceUDBlockStructure(RewriterBase &rewriter, ReduceUDOp reduceUDOp,
                            const IterationSpace &iterSpace,
                            int64_t reductionTileSize) {
  // Identity attribute must be present for user-defined reduction.
  ArrayAttr identityArrayAttr = reduceUDOp.getIdentityAttr();
  if (!identityArrayAttr) {
    return reduceUDOp.emitError() << "Expected identity attribute";
  }

  // Convert integer types to signless integers.
  SmallVector<Attribute> identities;
  for (Attribute identity : identityArrayAttr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(identity)) {
      Type intType =
          rewriter.getIntegerType(intAttr.getType().getIntOrFloatBitWidth());
      identity = rewriter.getIntegerAttr(intType, intAttr.getAPSInt());
    }
    identities.push_back(identity);
  }
  // Build the reduction loop nest.
  return buildCommonReduceBlockStructure(rewriter, reduceUDOp, iterSpace,
                                         reductionTileSize, identities);
}

//===----------------------------------------------------------------------===//
// Matmul operation
//===----------------------------------------------------------------------===//

/// Matmul operation is implemented as nested loops for each contracting
/// dimension. The loop induction variables are used to index the loads.
///
/// Example: matmul(A, B)
/// with shapes: A[64,1024], B[1024,16] and reduction tile size: 128
/// generates pseudo-code:
/// %acc = constant <f32: 0.0> : tile<64x16xf32>
/// for %idx in range(0, 8) {
///   %tile_A = [[block for A]], where %idx is used for indexing loads
///   %tile_B = [[block for B]], with a different iteration space
///   yield mma(%tile_A, %tile_B, %acc)
/// }
/// If necessary, the result is reshaped, transposed and/or broadcasted.
FailureOr<BlockStructure>
buildMatmulBlockStructure(RewriterBase &rewriter, MatmulOp matmulOp,
                          const IterationSpace &iterSpace,
                          int64_t contractionTileSize) {
  // Verify that the layout is a matmul layout.
  auto matmulSource = dyn_cast_if_present<MatmulSourceAttr>(
      matmulOp->getAttrOfType<LayoutSourceAttrInterface>(
          TensorIRDialect::getLayoutAttrName()));
  if (!matmulSource) {
    return matmulOp->emitError("Expected matmul layout");
  }

  // Contraction tile size must be a power of two.
  if (!llvm::isPowerOf2_64(contractionTileSize)) {
    return matmulOp->emitError()
           << "Contraction tile size must be a power of two, but got "
           << contractionTileSize;
  }

  // Get the dimension maps for the operands.
  auto lhsDimensionMap = matmulSource.getLhsDimensionMap();
  if (llvm::failed(lhsDimensionMap)) {
    return matmulOp->emitError() << "Cannot get LHS dimension map";
  }
  auto rhsDimensionMap = matmulSource.getRhsDimensionMap();
  if (llvm::failed(rhsDimensionMap)) {
    return matmulOp->emitError() << "Cannot get RHS dimension map";
  }

  // Use I32 type for the index values.
  ShapedType indexType = cuda_tile::TileType::get(
      /*shape=*/llvm::ArrayRef<int64_t>{}, rewriter.getI32Type());

  Location loc = matmulOp->getLoc();
  auto createIndex = [&](int64_t value) {
    return createConstant(rewriter, loc, indexType, value);
  };

  // Calculate the contraction tile shape.
  SmallVector<int64_t> contractionShape = matmulSource.getContractingShape();
  SmallVector<int64_t> contractionTileShape =
      calculateReductionTileShape(contractionShape, contractionTileSize);
  size_t iterSpaceRank = iterSpace.tileShape.size();

  // Build the shape and the index for the LHS input.
  SmallVector<int64_t> lhsTileShape;
  SmallVector<Value> lhsIndexValues;
  SmallVector<size_t> lhsContractingDims;

  for (size_t idx : *lhsDimensionMap) {
    if (idx < iterSpaceRank) {
      lhsTileShape.push_back(iterSpace.tileShape[idx]);
      lhsIndexValues.push_back(iterSpace.indexValues[idx]);
    } else {
      lhsTileShape.push_back(contractionTileShape[idx - iterSpaceRank]);
      lhsContractingDims.push_back(lhsIndexValues.size());
      lhsIndexValues.push_back(nullptr);
    }
  }

  // Build the shape and the index for the RHS input.
  SmallVector<int64_t> rhsTileShape;
  SmallVector<Value> rhsIndexValues;
  SmallVector<size_t> rhsContractingDims;

  for (size_t idx : *rhsDimensionMap) {
    if (idx < iterSpaceRank) {
      rhsTileShape.push_back(iterSpace.tileShape[idx]);
      rhsIndexValues.push_back(iterSpace.indexValues[idx]);
    } else {
      rhsTileShape.push_back(contractionTileShape[idx - iterSpaceRank]);
      rhsContractingDims.push_back(rhsIndexValues.size());
      rhsIndexValues.push_back(nullptr);
    }
  }

  // If the whole tile is reduced, no blocks are needed.
  if (contractionTileShape == contractionShape) {
    // Use zero index values for the contracting dimensions.
    Value zero = createIndex(0);
    for (size_t idx : lhsContractingDims) {
      lhsIndexValues[idx] = zero;
    }
    for (size_t idx : rhsContractingDims) {
      rhsIndexValues[idx] = zero;
    }

    // Create the iteration spaces in the same block.
    IterationSpace lhsIterSpace{rewriter.getBlock(), std::move(lhsTileShape),
                                std::move(lhsIndexValues),
                                extractTensorSources(matmulSource.getLhs())};
    IterationSpace rhsIterSpace{rewriter.getBlock(), std::move(rhsTileShape),
                                std::move(rhsIndexValues),
                                extractTensorSources(matmulSource.getRhs())};
    return BlockStructure{{std::move(lhsIterSpace), std::move(rhsIterSpace)},
                          /*yieldValues=*/{},
                          /*iterationSpaceIndexForOperand=*/{}};
  }

  // Calculate the batch tile size from the matmul view layout.
  int64_t tileB = 1;
  tcg::Layout view = matmulSource.getCuteLayout();
  const int64_t batchStride =
      matmulSource.getM() * matmulSource.getN() * matmulSource.getK();
  for (const auto &[idx, stride] : llvm::enumerate(view)) {
    auto part = tcg::coalesce(stride);
    if (tcg::rank(part) != 1) {
      return matmulOp.emitError("matmul view must have a single nested mode");
    }
    if (part.stride().as_int() >= batchStride) {
      tileB *= iterSpace.tileShape[idx];
    }
  }

  // Calculate the tile sizes for each component.
  int64_t tileK = llvm::product_of(contractionTileShape);
  int64_t tileM = llvm::product_of(lhsTileShape) / (tileB * tileK);
  int64_t tileN = llvm::product_of(rhsTileShape) / (tileB * tileK);

  // Create the accumulator tile type.
  SmallVector<int64_t> accumulatorShape = {tileM, tileN};
  if (tileB > 1) {
    accumulatorShape.insert(accumulatorShape.begin(), tileB);
  }

  Type resultType = matmulOp.getResult().getType().getElementType();
  if (resultType.isInteger()) {
    resultType = rewriter.getIntegerType(resultType.getIntOrFloatBitWidth());
  }
  ShapedType accumulatorType =
      cuda_tile::TileType::get(accumulatorShape, resultType);

  // Create the accumulator tile.
  Value accumulator;
  if (resultType.isFloat()) {
    accumulator = createConstant(rewriter, loc, accumulatorType, 0.0);
  } else {
    accumulator = createConstant(rewriter, loc, accumulatorType, int64_t{0});
  }

  // Generate the loop nest.
  BlockStructure result;
  Value startIndex = createIndex(0);
  Value stepValue = createIndex(1);

  for (size_t idx : llvm::seq(contractionShape.size())) {
    // If the tile size covers the whole dimension, skip it.
    int64_t size = contractionShape[idx];
    int64_t tileSize = contractionTileShape[idx];
    if (size == tileSize) {
      continue;
    }

    // Create the loop operation.
    Value endIndex = createIndex(llvm::divideCeil(size, tileSize));
    auto forOp = cuda_tile::ForOp::create(rewriter, loc, startIndex, endIndex,
                                          stepValue, {accumulator});
    accumulator = forOp.getRegionIterValues().front();

    // Update the index values using the dimension maps.
    lhsIndexValues[lhsContractingDims[idx]] = forOp.getInductionVar();
    rhsIndexValues[rhsContractingDims[idx]] = forOp.getInductionVar();

    // Create the continue op (inner) or store the result value (outer).
    if (!result.yieldValues.empty()) {
      cuda_tile::ContinueOp::create(rewriter, loc, forOp.getResults());
    } else {
      result.yieldValues = forOp.getResults();
    }

    // Update the insertion point.
    rewriter.setInsertionPointToEnd(forOp.getBody());
  }

  // Matmul operation has two iteration spaces.
  IterationSpace lhsIterSpace{rewriter.getBlock(), std::move(lhsTileShape),
                              std::move(lhsIndexValues),
                              extractTensorSources(matmulSource.getLhs())};
  result.iterationSpaces.push_back(std::move(lhsIterSpace));

  IterationSpace rhsIterSpace{rewriter.getBlock(), std::move(rhsTileShape),
                              std::move(rhsIndexValues),
                              extractTensorSources(matmulSource.getRhs())};
  result.iterationSpaces.push_back(std::move(rhsIterSpace));

  return result;
}

//===----------------------------------------------------------------------===//
// Block structure builder
//===----------------------------------------------------------------------===//

/// Build the block structure and the iteration space data for operations that
/// modify the iteration space (concatenation, reduction, matmul).
FailureOr<BlockStructure> buildBlockStructure(
    RewriterBase &rewriter, Operation *op, const IterationSpace &iterSpace,
    const TypeConverter &typeConverter, int64_t reductionTileSize) {
  // Update the insertion point.
  rewriter.setInsertionPointToEnd(iterSpace.insertionBlock);

  // Process the concatenation operation.
  if (auto concatOp = dyn_cast<ConcatenateOp>(op)) {
    return buildConcatBlockStructure(rewriter, concatOp, iterSpace,
                                     typeConverter);
  }

  // Process the reduction operations.
  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    return buildReduceBlockStructure(rewriter, reduceOp, iterSpace,
                                     reductionTileSize);
  }
  if (auto reduceUDOp = dyn_cast<ReduceUDOp>(op)) {
    return buildReduceUDBlockStructure(rewriter, reduceUDOp, iterSpace,
                                       reductionTileSize);
  }

  // Process the matmul operation.
  if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
    return buildMatmulBlockStructure(rewriter, matmulOp, iterSpace,
                                     reductionTileSize);
  }

  // No other operations are supported yet.
  return op->emitError() << "Unsupported operation: " << op->getName();
}

/// Derive the iteration-space index for each operand of the owner op `op`:
/// entry `p` of the returned vector is the index into the block structure's
/// iteration spaces that operand `p` feeds, or `kNoIterationSpace` when that
/// operand feeds none (a concatenate operand pruned by a slice). This is the
/// single place the relation is computed; every consumer reads the result.
/// Only owner ops (concatenate, matmul, reduce, reduceUD, results) reach here.
FailureOr<SmallVector<int>> deriveIterationSpaceIndexForOperand(Operation *op) {
  unsigned numOperands = op->getNumOperands();

  // Pruned concatenate -- the only op whose operands don't map 1:1 to spaces:
  // source `i` is fed by operand `argumentIndex[i]`; an operand absent from
  // `argumentIndex` was pruned by a slice and feeds no space. (An empty
  // `argumentIndex` means nothing was pruned -- handled by the 1:1 case below.)
  if (isa<ConcatenateOp>(op)) {
    ArrayRef<int32_t> argumentIndex =
        cast<ConcatSourceAttr>(op->getAttrOfType<LayoutSourceAttrInterface>(
                                   TensorIRDialect::getLayoutAttrName()))
            .getArgumentIndex();
    if (!argumentIndex.empty()) {
      SmallVector<int> spaceIndexForOperand(numOperands,
                                            BlockStructure::kNoIterationSpace);
      for (auto [spaceIndex, operand] : llvm::enumerate(argumentIndex)) {
        // An argument index is an operand position. It is derived from an
        // attribute, so a malformed graph could carry an out-of-range value;
        // reject it with a clear error rather than indexing
        // `spaceIndexForOperand` out of bounds.
        if (static_cast<unsigned>(operand) >= numOperands) {
          return op->emitError()
                 << "concatenate argument index refers to a non-existent "
                    "operand";
        }
        spaceIndexForOperand[operand] = static_cast<int>(spaceIndex);
      }
      return spaceIndexForOperand;
    }
  }

  // Matmul, and an identity (unpruned) concatenate: operand `p` feeds space
  // `p`.
  if (isa<MatmulOp, ConcatenateOp>(op)) {
    SmallVector<int> spaceIndexForOperand(numOperands);
    for (unsigned p = 0; p < numOperands; ++p) {
      spaceIndexForOperand[p] = static_cast<int>(p);
    }
    return spaceIndexForOperand;
  }

  // Reduction (including user-defined, whose operands share a single space) and
  // the results op: every operand feeds the single (first) iteration space.
  if (isa<ReduceOp, ReduceUDOp, ResultsOp>(op)) {
    return SmallVector<int>(numOperands, 0);
  }

  // Not an iteration-space owner. buildSkeleton only calls this for the owner
  // ops above, so this is a guarded error rather than an expected path: a newly
  // added owner op must get its own case here rather than silently fall
  // through.
  return op->emitError()
         << "deriveIterationSpaceIndexForOperand: unsupported iteration-space "
            "owner op";
}

} // namespace

FailureOr<DenseMap<Operation *, BlockStructure>>
buildSkeleton(RewriterBase &rewriter, IterationSpace initial,
              const TypeConverter &typeConverter, int64_t reductionTileSize) {
  DenseMap<Operation *, BlockStructure> result;

  // Get the terminator pointer before modifying the current block.
  auto resultsOp = initial.insertionBlock->getTerminator();

  // Find all the operations that modify the iteration space and group them by
  // their iteration space ID. Only concatenation and reduction ops that carry
  // an `iterSpaceId` attribute are supported for now.
  DenseMap<int, SmallVector<Operation *>> iterSpaceOpMap;
  initial.insertionBlock->walk([&](Operation *op) {
    // Supported operations: concatenation, reduction, matmul.
    if (isa<ConcatenateOp, ReduceOp, ReduceUDOp, MatmulOp>(op)) {
      if (auto iterSpaceId = op->getAttrOfType<IntegerAttr>(
              TensorIRDialect::getIterSpaceIdAttrName())) {
        iterSpaceOpMap[iterSpaceId.getInt()].push_back(op);
      }
    }
  });

  // Recursively process the iteration spaces.
  std::function<LogicalResult(int, const IterationSpace &)>
      processIterationSpace =
          [&](int iterSpaceId,
              const IterationSpace &iterSpace) -> LogicalResult {
    for (Operation *op : iterSpaceOpMap.lookup(iterSpaceId)) {
      // Build the skeleton for the operation.
      MLIR_ASSIGN_OR_RETURN(BlockStructure blockStructure,
                            buildBlockStructure(rewriter, op, iterSpace,
                                                typeConverter,
                                                reductionTileSize));

      // Derive the operand-to-iteration-space relation once, storing it on the
      // block structure so every consumer reads the same relation.
      MLIR_ASSIGN_OR_RETURN(blockStructure.iterationSpaceIndexForOperand,
                            deriveIterationSpaceIndexForOperand(op));

      // For each operand, recursively process its iteration space. An operand
      // with no space (a pruned concatenate operand) is skipped; the mapping
      // already sends every user-defined reduction operand to space 0.
      for (auto [operandIdx, value] : llvm::enumerate(op->getOperands())) {
        int spaceIdx = blockStructure.iterationSpaceIndexForOperand[operandIdx];
        if (spaceIdx == BlockStructure::kNoIterationSpace) {
          continue;
        }
        const IterationSpace &inner = blockStructure.iterationSpaces[spaceIdx];
        if (auto definingOp = value.getDefiningOp()) {
          if (auto iterSpaceId = definingOp->getAttrOfType<IntegerAttr>(
                  TensorIRDialect::getIterSpaceIdAttrName())) {
            MLIR_RETURN_IF_ERROR(
                processIterationSpace(iterSpaceId.getInt(), inner));
          }
        }
      }

      // Update the result.
      result.insert({op, std::move(blockStructure)});
    }
    return success();
  };

  // Start the processing from the root iteration space.
  MLIR_RETURN_IF_ERROR(processIterationSpace(0, initial));

  // Extract the layouts for the default iteration space.
  auto layout = resultsOp->getAttrOfType<LayoutSourceAttrInterface>(
      TensorIRDialect::getIterationSpaceAttrName());
  extractTensorSources(layout, initial.loadedTiles);

  // Add the root iteration space to the result. The results op has a single
  // operand, which feeds the root iteration space (space 0).
  BlockStructure root{{std::move(initial)},
                      /*yieldValues=*/{},
                      /*iterationSpaceIndexForOperand=*/{}};
  MLIR_ASSIGN_OR_RETURN(root.iterationSpaceIndexForOperand,
                        deriveIterationSpaceIndexForOperand(resultsOp));
  result.insert({resultsOp, std::move(root)});

  return result;
}

} // namespace tensor_to_cuda_tile
} // namespace mlir::nv_tensor_ir
