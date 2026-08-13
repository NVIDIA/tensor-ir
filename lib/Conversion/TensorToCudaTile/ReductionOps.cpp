// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// \file ReductionOps.cpp
/// Conversion patterns for reduction TensorIR operations and arithmetic
/// operations in the user-defined reduction.
///
/// This module implements the conversion patterns that lower `ReduceOp`
/// TensorIR operation (has a predefined set of reduction types), `ReduceUDOp`
/// (has user-defined reduction computation) and all supported arithmetic
/// operations (https://mlir.llvm.org/docs/Dialects/ArithOps).
///
/// - multiple reduction dimensions are merged (as CUDA Tile only supports a
///   single contracting dimension), the layout propagation makes sure that
///   such dimensions are adjacent;
/// - if the reduction output tile needs to be broadcasted, a
///   `cuda_tile::BroadcastOp` is emitted;
///
/// If the reduction dimension is small (<=128 by default), a single
/// `cuda_tile::ReduceOp` is emitted, with optional prologue and epilogue.
/// The `yieldValues` array is empty, as there are no blocks generated.
///
/// If the reduction dimension is large, the `buildSkeleton` function creates
/// a loop nest iterating over the tiles in that dimension and accumulates the
/// results, which are then passed to the `cuda_tile::ReduceOp`.
/// The `yieldValues` array is the resulting tile values of the loop nest.
///
/// The patterns are registered via `registerReductionPatterns`.

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"

#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/APSInt.h" // IWYU pragma: keep
#include "llvm/Support/Debug.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#define DEBUG_TYPE "convert-tensor-to-cuda-tile-layout-propagation"

namespace mlir::nv_tensor_ir {
namespace tensor_to_cuda_tile {

namespace tcg = mlir::nv_tensor_ir::tcutegen;

//===----------------------------------------------------------------------===//
// ReduceOp conversion
//===----------------------------------------------------------------------===//

class ReduceOpConversion : public ConversionPattern<ReduceOp> {
public:
  using ConversionPattern<ReduceOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceOp reduceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, reduceOp));
    MLIR_RETURN_IF_ERROR(validateReductionOpLowerable(reduceOp));
    Location loc = reduceOp.getLoc();

    // Retrieve the block structure.
    const BlockStructure *blockStructure = state.getBlockStructure();
    assert(blockStructure && "expected block structure for reduction");

    // Retrieve the reduction source.
    auto reductionSource = reduceOp->getAttrOfType<ReductionSourceAttr>(
        TensorIRDialect::getLayoutAttrName());
    assert(reductionSource && "expected reduction layout");

    // Calculate the shapes for the tiles involved in the reduction.
    // `inputTileShape` is the shape of the underlying source;
    // `resultTileShape` is the shape before the broadcast (if any);
    // `outputTileShape` is the shape in the current iteration space;
    Value inputTile = state.getTile(adaptor.getInput());
    auto inputTileType = cast<ShapedType>(inputTile.getType());

    ArrayRef<int64_t> inputTileShape = inputTileType.getShape();
    ArrayRef<int64_t> outputTileShape = state.getTileShape();
    SmallVector<int64_t> resultTileShape =
        getResultTileShape(reductionSource.getCuteLayout(), inputTileShape);

    LLVM_DEBUG({
      llvm::dbgs() << "Input tile shape: (";
      llvm::interleaveComma(inputTileShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
      llvm::dbgs() << "Result tile shape: (";
      llvm::interleaveComma(resultTileShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
      llvm::dbgs() << "Output tile shape: (";
      llvm::interleaveComma(outputTileShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
    });

    // Create the reduction emission helper.
    ReductionEmissionHelper reductionEmission{
        reduceOp.getReductionMode(),
        reduceOp.getInput().getType().getElementType()};

    // Generate the terminator in the block structure.
    Value blockResult;
    if (!blockStructure->yieldValues.empty()) {
      RewriterBase::InsertionGuard guard(rewriter);

      // Emit prologue and accumulation.
      const IterationSpace &iterationSpace = blockStructure->iterationSpaces[0];
      rewriter.setInsertionPointToEnd(iterationSpace.insertionBlock);
      inputTile = reductionEmission.buildPrologue(rewriter, inputTile);
      Value innerResult = reductionEmission.buildReduction(
          rewriter, iterationSpace.insertionBlock->getArgument(1), inputTile);

      // Emit the loop terminator.
      cuda_tile::ContinueOp::create(rewriter, loc, innerResult);
      blockResult = blockStructure->yieldValues[0];
    } else {
      // Emit only the prologue when no blocks are present.
      blockResult = reductionEmission.buildPrologue(rewriter, inputTile);
    }

    // CUDA Tile only supports reduction in a single dimension.
    // Reshape the reduction input tile, if needed.
    size_t reductionDimCount = reductionSource.getReductionShape().size();
    if (reductionDimCount > 1) {
      auto newTileType = inputTileType.clone(
          getReductionTileShape(reductionDimCount, inputTileShape));
      blockResult =
          cuda_tile::ReshapeOp::create(rewriter, loc, newTileType, blockResult);
    }

    // Emit the tile reduction operation.
    auto identities =
        rewriter.getArrayAttr({reductionEmission.getIdentity(rewriter)});
    auto tileReduceOp = cuda_tile::ReduceOp::create(
        rewriter, loc, blockResult, inputTileShape.size() - reductionDimCount,
        identities);

    // Create the reduction body block.
    Block &body = tileReduceOp.getBodyRegion().emplaceBlock();
    ShapedType argType = inputTileType.clone(ArrayRef<int64_t>{});
    BlockArgument lhs = body.addArgument(argType, loc);
    BlockArgument rhs = body.addArgument(argType, loc);

    // Emit the accumulation operation.
    // Note: in CUDA Tile, the accumulator is the second block argument (RHS).
    rewriter.setInsertionPointToEnd(&body);
    Value innerResult = reductionEmission.buildReduction(rewriter, rhs, lhs);
    cuda_tile::YieldOp::create(rewriter, loc, innerResult);

    // Restore the insertion point and emit epilogue.
    rewriter.setInsertionPointAfter(tileReduceOp);
    Value result =
        reductionEmission.buildEpilogue(rewriter, tileReduceOp.getResult(0),
                                        reductionSource.getReductionSize());

    // Reshape and broadcast the result, if needed.
    if (resultTileShape != outputTileShape) {
      result = cuda_tile::ReshapeOp::create(
          rewriter, loc, inputTileType.clone(resultTileShape), result);
      result = cuda_tile::BroadcastOp::create(
          rewriter, loc, inputTileType.clone(outputTileShape), result);
    }

    // Replace `ReduceOp` with the result of the reduction.
    rewriter.replaceOp(reduceOp, result);
    return success();
  }

private:
  // The below helpers are used by both reduction conversion patterns.
  friend class ReduceUDOpConversion;

  // Calculate the resulting tile shape from the reduction view layout.
  static SmallVector<int64_t>
  getResultTileShape(const tcg::Layout &view,
                     ArrayRef<int64_t> inputTileShape) {
    SmallVector<int64_t> resultTileShape;
    for (size_t i = 0, n = tcg::rank(view) - 1, p = 0; i < n; i++) {
      bool isBroadcast = tcg::get(view, i).stride().as_int() == 0;
      assert((isBroadcast || p < inputTileShape.size()) &&
             "invalid reduction view, shape mismatch");
      resultTileShape.push_back(isBroadcast ? 1 : inputTileShape[p++]);
    }
    return resultTileShape;
  }

  // Calculate the reduction tile shape (merge contracting dimensions).
  static SmallVector<int64_t>
  getReductionTileShape(size_t reductionDimCount,
                        ArrayRef<int64_t> inputTileShape) {
    SmallVector<int64_t> reductionTileShape(
        inputTileShape.drop_back(reductionDimCount));
    int64_t product =
        std::accumulate(inputTileShape.end() - reductionDimCount,
                        inputTileShape.end(), 1, std::multiplies<int64_t>());
    reductionTileShape.push_back(product);
    return reductionTileShape;
  }
};

//===----------------------------------------------------------------------===//
// ReduceUDOp conversion
//===----------------------------------------------------------------------===//

class ReduceUDOpConversion : public ConversionPattern<ReduceUDOp> {
public:
  using ConversionPattern<ReduceUDOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceUDOp reduceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, reduceOp));
    Location loc = reduceOp.getLoc();

    // Retrieve the block structure.
    const BlockStructure *blockStructure = state.getBlockStructure();
    assert(blockStructure && "expected block structure for reduction");

    // Retrieve the reduction source.
    auto reductionSource = reduceOp->getAttrOfType<ReductionSourceAttr>(
        TensorIRDialect::getLayoutAttrName());
    assert(reductionSource && "expected reduction layout");

    // Get the reduction tile operands.
    SmallVector<Value> inputTiles;
    for (auto operand : adaptor.getOperands()) {
      inputTiles.push_back(state.getTile(operand));
    }

    // Copy the reduction operations into the loop body, if present.
    bool hasLoop = !blockStructure->yieldValues.empty();
    if (hasLoop) {
      Block *sourceBlock = &reduceOp.getBodyRegion().front();
      Block *targetBlock = blockStructure->iterationSpaces[0].insertionBlock;

      // Map block arguments to loop body arguments.
      // TensorIR order: acc_1, ..., acc_N, arg_1, ..., arg_N.
      // CUDA Tile order: arg_1, acc_1, ..., arg_N, acc_N.
      IRMapping mapping;
      for (size_t i = 0, n = inputTiles.size(); i < n; i++) {
        mapping.map(sourceBlock->getArgument(n + i), inputTiles[i]);
        mapping.map(sourceBlock->getArgument(i),
                    targetBlock->getArgument(i + 1));
      }

      // Clone the reduction block into the loop body.
      RewriterBase::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(targetBlock);
      for (Operation &op : *sourceBlock) {
        rewriter.clone(op, mapping);
      }
    }

    // Calculate the shapes for the tiles involved in the reduction.
    // `inputTileShape` is the shape of the underlying source;
    // `resultTileShape` is the shape before the broadcast (if any);
    // `outputTileShape` is the shape in the current iteration space;
    ArrayRef<int64_t> inputTileShape =
        cast<ShapedType>(inputTiles[0].getType()).getShape();
    ArrayRef<int64_t> outputTileShape = state.getTileShape();
    SmallVector<int64_t> resultTileShape =
        ReduceOpConversion::getResultTileShape(reductionSource.getCuteLayout(),
                                               inputTileShape);

    LLVM_DEBUG({
      llvm::dbgs() << "Input tile shape: (";
      llvm::interleaveComma(inputTileShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
      llvm::dbgs() << "Result tile shape: (";
      llvm::interleaveComma(resultTileShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
      llvm::dbgs() << "Output tile shape: (";
      llvm::interleaveComma(outputTileShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
    });

    // Convert integer types to signless integers.
    SmallVector<Attribute> identities;
    for (Attribute identity : reduceOp.getIdentityAttr()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(identity)) {
        Type intType =
            rewriter.getIntegerType(intAttr.getType().getIntOrFloatBitWidth());
        identity = rewriter.getIntegerAttr(intType, intAttr.getAPSInt());
      }
      identities.push_back(identity);
    }

    // CUDA Tile only supports reduction in a single dimension.
    // Reshape the reduction input tiles, if needed.
    SmallVector<Value> reductionTiles =
        hasLoop ? blockStructure->yieldValues : inputTiles;
    size_t reductionDimCount = reductionSource.getReductionShape().size();
    if (reductionDimCount > 1) {
      SmallVector<int64_t> reductionTileShape =
          ReduceOpConversion::getReductionTileShape(reductionDimCount,
                                                    inputTileShape);
      for (auto &reductionTile : reductionTiles) {
        auto newTileType =
            cast<ShapedType>(reductionTile.getType()).clone(reductionTileShape);
        reductionTile = cuda_tile::ReshapeOp::create(rewriter, loc, newTileType,
                                                     reductionTile);
      }
    }

    // Emit the tile reduction operation.
    auto tileReduceOp =
        cuda_tile::ReduceOp::create(rewriter, loc, reductionTiles,
                                    inputTileShape.size() - reductionDimCount,
                                    rewriter.getArrayAttr(identities));

    // Create the signature conversion for the body block.
    // TensorIR order: acc_1, ..., acc_N, arg_1, ..., arg_N.
    // CUDA Tile order: arg_1, acc_1, ..., arg_N, acc_N.
    size_t numInputs = inputTiles.size();
    TypeConverter::SignatureConversion signatureConversion(numInputs * 2);
    for (size_t i = 0; i < numInputs; i++) {
      Type inputType =
          getTypeConverter()->convertType(reduceOp.getOperand(i).getType());
      signatureConversion.addInputs(numInputs + i, inputType);
      signatureConversion.addInputs(i, inputType);
    }

    // Move the reduction operations and apply the signature conversion.
    Region *targetRegion = &tileReduceOp.getBodyRegion();
    rewriter.inlineRegionBefore(reduceOp.getBodyRegion(), *targetRegion,
                                targetRegion->end());
    rewriter.applySignatureConversion(&targetRegion->front(),
                                      signatureConversion);

    // Reshape and broadcast the result, if needed.
    SmallVector<Value> results = tileReduceOp.getResults();
    if (resultTileShape != outputTileShape) {
      for (auto &result : results) {
        auto resultType = cast<ShapedType>(result.getType());
        result = cuda_tile::ReshapeOp::create(
            rewriter, loc, resultType.clone(resultTileShape), result);
        result = cuda_tile::BroadcastOp::create(
            rewriter, loc, resultType.clone(outputTileShape), result);
      }
    }

    // Replace `ReduceUDOp` with the result of the reduction.
    rewriter.replaceOp(reduceOp, results);
    return success();
  }
};

/// Yield operation is the terminator of the reduce body block.
class YieldOpConversion : public ConversionPattern<YieldOp> {
public:
  using ConversionPattern<YieldOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp yieldOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands(adaptor.getOperands());

    // Remove unrealized conversion casts.
    for (auto &operand : operands) {
      if (auto unrealizedCast = dyn_cast_if_present<UnrealizedConversionCastOp>(
              operand.getDefiningOp())) {
        operand = unrealizedCast.getOperand(0);
      }
    }

    if (isa<cuda_tile::ForOp>(yieldOp->getParentOp())) {
      // Yield inside a for loop: emit `ContinueOp`.
      rewriter.replaceOpWithNewOp<cuda_tile::ContinueOp>(yieldOp, operands);
    } else {
      // Yield inside a reduce body: emit `YieldOp`.
      rewriter.replaceOpWithNewOp<cuda_tile::YieldOp>(yieldOp, operands);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

template <typename ArithOpTy, typename TileOpTy>
class SimpleArithOpConversion : public OpConversionPattern<ArithOpTy> {
public:
  using Base = OpConversionPattern<ArithOpTy>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOpTy op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TileOpTy>(op, adaptor.getOperands());
    return success();
  }
};

// Signless integer arithmetic operations.
using AddIOpConversion =
    SimpleArithOpConversion<arith::AddIOp, cuda_tile::AddIOp>;
using MulIOpConversion =
    SimpleArithOpConversion<arith::MulIOp, cuda_tile::MulIOp>;
using SubIOpConversion =
    SimpleArithOpConversion<arith::SubIOp, cuda_tile::SubIOp>;

// Signless integer bitwise operations.
using AndIOpConversion =
    SimpleArithOpConversion<arith::AndIOp, cuda_tile::AndIOp>;
using OrIOpConversion = SimpleArithOpConversion<arith::OrIOp, cuda_tile::OrIOp>;
using XOrIOpConversion =
    SimpleArithOpConversion<arith::XOrIOp, cuda_tile::XOrIOp>;

// Floating-point arithmetic operations (no rounding).
using MaximumFOpConversion =
    SimpleArithOpConversion<arith::MaximumFOp, cuda_tile::MaxFOp>;
using MinimumFOpConversion =
    SimpleArithOpConversion<arith::MinimumFOp, cuda_tile::MinFOp>;
using NegFOpConversion =
    SimpleArithOpConversion<arith::NegFOp, cuda_tile::NegFOp>;
using RemFOpConversion =
    SimpleArithOpConversion<arith::RemFOp, cuda_tile::RemFOp>;

// Ternary selection operation.
using SelectOpConversion =
    SimpleArithOpConversion<arith::SelectOp, cuda_tile::SelectOp>;

// Template for floating-point arithmetic operations.
template <typename ArithOpTy, typename TileOpTy>
class FloatArithOpConversion : public OpConversionPattern<ArithOpTy> {
public:
  using Base = OpConversionPattern<ArithOpTy>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOpTy op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TileOpTy>(
        op, adaptor.getLhs(), adaptor.getRhs(),
        cuda_tile::RoundingMode::NEAREST_EVEN);
    return success();
  }
};

// Floating-point arithmetic operations.
using AddFOpConversion =
    FloatArithOpConversion<arith::AddFOp, cuda_tile::AddFOp>;
using DivFOpConversion =
    FloatArithOpConversion<arith::DivFOp, cuda_tile::DivFOp>;
using MulFOpConversion =
    FloatArithOpConversion<arith::MulFOp, cuda_tile::MulFOp>;
using SubFOpConversion =
    FloatArithOpConversion<arith::SubFOp, cuda_tile::SubFOp>;

// Template for integer arithmetic operations.
template <typename ArithOpTy, typename TileOpTy,
          cuda_tile::Signedness Signedness>
class IntegerArithOpConversion : public OpConversionPattern<ArithOpTy> {
public:
  using Base = OpConversionPattern<ArithOpTy>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOpTy op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TileOpTy>(op, adaptor.getLhs(),
                                          adaptor.getRhs(), Signedness);
    return success();
  }
};

template <typename SignedOpTy, typename UnsignedOpTy, typename TileOpTy>
struct IntegerTypedOpConversion {
  using Signed = IntegerArithOpConversion<SignedOpTy, TileOpTy,
                                          cuda_tile::Signedness::Signed>;
  using Unsigned = IntegerArithOpConversion<UnsignedOpTy, TileOpTy,
                                            cuda_tile::Signedness::Unsigned>;
};

// Integer arithmetic operations.
using MaxIOpConversion =
    IntegerTypedOpConversion<arith::MaxSIOp, arith::MaxUIOp, cuda_tile::MaxIOp>;
using MinIOpConversion =
    IntegerTypedOpConversion<arith::MinSIOp, arith::MinUIOp, cuda_tile::MinIOp>;

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

// Floating-point comparison conversion.
class CmpFOpConversion : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    arith::CmpFPredicate comparator = adaptor.getPredicate();
    auto ordering = cuda_tile::ComparisonOrdering::ORDERED;
    if (comparator == arith::CmpFPredicate::UEQ ||
        comparator == arith::CmpFPredicate::UNE ||
        comparator == arith::CmpFPredicate::ULT ||
        comparator == arith::CmpFPredicate::ULE ||
        comparator == arith::CmpFPredicate::UGT ||
        comparator == arith::CmpFPredicate::UGE) {
      ordering = cuda_tile::ComparisonOrdering::UNORDERED;
    }
    rewriter.replaceOpWithNewOp<cuda_tile::CmpFOp>(op, getPredicate(comparator),
                                                   ordering, adaptor.getLhs(),
                                                   adaptor.getRhs());
    return success();
  }

private:
  static cuda_tile::ComparisonPredicate
  getPredicate(arith::CmpFPredicate comparator) {
    switch (comparator) {
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      return cuda_tile::ComparisonPredicate::EQUAL;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      return cuda_tile::ComparisonPredicate::NOT_EQUAL;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      return cuda_tile::ComparisonPredicate::LESS_THAN;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      return cuda_tile::ComparisonPredicate::LESS_THAN_OR_EQUAL;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      return cuda_tile::ComparisonPredicate::GREATER_THAN;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      return cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL;
    default:
      llvm_unreachable("unsupported comparator");
    }
  }
};

// Integer comparison conversion.
class CmpIOpConversion : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    arith::CmpIPredicate comparator = adaptor.getPredicate();
    auto signedness = cuda_tile::Signedness::Signed;
    if (comparator == arith::CmpIPredicate::ult ||
        comparator == arith::CmpIPredicate::ule ||
        comparator == arith::CmpIPredicate::ugt ||
        comparator == arith::CmpIPredicate::uge) {
      signedness = cuda_tile::Signedness::Unsigned;
    }
    rewriter.replaceOpWithNewOp<cuda_tile::CmpIOp>(
        op, getPredicate(comparator), adaptor.getLhs(), adaptor.getRhs(),
        signedness);
    return success();
  }

private:
  static cuda_tile::ComparisonPredicate
  getPredicate(arith::CmpIPredicate comparator) {
    switch (comparator) {
    case arith::CmpIPredicate::eq:
      return cuda_tile::ComparisonPredicate::EQUAL;
    case arith::CmpIPredicate::ne:
      return cuda_tile::ComparisonPredicate::NOT_EQUAL;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return cuda_tile::ComparisonPredicate::LESS_THAN;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return cuda_tile::ComparisonPredicate::LESS_THAN_OR_EQUAL;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return cuda_tile::ComparisonPredicate::GREATER_THAN;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL;
    default:
      llvm_unreachable("unsupported comparator");
    }
  }
};

// Constant literal conversion.
class ScalarConstantOpConversion
    : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If the scalar constant is inside a loop body, get the tile shape from
    // the loop body block arguments.
    ArrayRef<int64_t> tileShape;
    if (auto forOp = dyn_cast<cuda_tile::ForOp>(op->getParentOp())) {
      tileShape = cast<ShapedType>(forOp.getBody()->getArgument(1).getType())
                      .getShape();
    }

    // Get the constant literal for the tile type.
    auto tileType = cuda_tile::TileType::get(tileShape, op.getType());
    auto literal = dyn_cast_if_present<DenseTypedElementsAttr>(
        DenseElementsAttr::get(tileType, op.getValue()));
    if (!literal) {
      return op.emitError("unsupported constant type");
    }

    // Replace the scalar constant with tile constant.
    rewriter.replaceOpWithNewOp<cuda_tile::ConstantOp>(op, tileType, literal);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Per-op legality
//===----------------------------------------------------------------------===//

/// Shared per-op legality check for reduction operations. Single source of
/// truth for the reduction element-type and mode guards: invoked both by the
/// reduction conversion pattern (after the per-op state update) and by the
/// pre-flight feasibility check.
LogicalResult validateReductionOpLowerable(ReduceOp op) {
  // Verify that the reduction type and mode are supported.
  Type outputType = op.getResult().getType().getElementType();
  if (!outputType.isFloat() && !outputType.isInteger()) {
    return op.emitError("Unsupported reduction type");
  }
  if (op.getReductionMode() == ReductionMode::customize ||
      (op.getReductionMode() == ReductionMode::norm2 &&
       outputType.isInteger())) {
    return op.emitError("Unsupported reduction mode");
  }

  auto reductionSource = op->getAttrOfType<ReductionSourceAttr>(
      TensorIRDialect::getLayoutAttrName());
  if (!reductionSource) {
    return op.emitError("expected reduction layout");
  }
  if (!tcg::is_static(reductionSource.getCuteLayout())) {
    return op.emitError(
        "reduction view layout must have static shape and stride");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerReductionPatterns(RewritePatternSet &patterns,
                               ConversionState &state,
                               const TypeConverter &typeConverter) {
  // TensorIR dialect conversion patterns.
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ReduceOpConversion, ReduceUDOpConversion, YieldOpConversion>(
      state, typeConverter, ctx);

  // Arith dialect conversion patterns.
  patterns.add<
      // Floating-point arithmetic operations.
      MaximumFOpConversion, MinimumFOpConversion, NegFOpConversion,
      RemFOpConversion, AddFOpConversion, DivFOpConversion, MulFOpConversion,
      SubFOpConversion,
      // Signless integer arithmetic operations.
      AddIOpConversion, MulIOpConversion, SubIOpConversion, AndIOpConversion,
      OrIOpConversion, XOrIOpConversion,
      // Signed/unsigned integer arithmetic operations.
      MaxIOpConversion::Signed, MaxIOpConversion::Unsigned,
      MinIOpConversion::Signed, MinIOpConversion::Unsigned,
      // Comparison operations.
      CmpFOpConversion, CmpIOpConversion,
      // Miscellaneous operations.
      ScalarConstantOpConversion, SelectOpConversion>(ctx);
}

} // namespace tensor_to_cuda_tile
} // namespace mlir::nv_tensor_ir
