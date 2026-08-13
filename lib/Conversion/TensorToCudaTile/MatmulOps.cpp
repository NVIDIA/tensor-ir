// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"

#include "llvm/Support/Debug.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#define DEBUG_TYPE "convert-tensor-to-cuda-tile-layout-propagation"

namespace mlir::nv_tensor_ir {
namespace tensor_to_cuda_tile {

namespace tcg = mlir::nv_tensor_ir::tcutegen;

//===----------------------------------------------------------------------===//
// MatmulOp conversion
//===----------------------------------------------------------------------===//

static bool isSupportedMatmulTypes(Type lhsType, Type rhsType,
                                   Type outputType) {
  // Floating-point types.
  if (lhsType.isF16() && rhsType == lhsType) {
    return outputType.isF16() || outputType.isF32();
  }
  if ((lhsType.isBF16() || lhsType.isF32()) && rhsType == lhsType) {
    return outputType.isF32();
  }
  if (lhsType.isF64() && rhsType == lhsType) {
    return outputType.isF64();
  }

  // Integer types.
  if (lhsType.isInteger(8) && rhsType.isInteger(8)) {
    return outputType.isSignedInteger(32);
  }
  return false;
}

class MatmulOpConversion : public ConversionPattern<MatmulOp> {
public:
  using ConversionPattern<MatmulOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(MatmulOp matmulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, matmulOp));
    MLIR_RETURN_IF_ERROR(validateMatmulOpLowerable(matmulOp));
    Location loc = matmulOp.getLoc();

    // Retrieve the block structure.
    const BlockStructure *blockStructure = state.getBlockStructure();
    assert(blockStructure && "expected block structure for matmul");

    // Retrieve the matmul source.
    auto matmulSource = matmulOp->getAttrOfType<MatmulSourceAttr>(
        TensorIRDialect::getLayoutAttrName());
    assert(matmulSource && "expected matmul layout");

    // Get the input tiles.
    Value lhsTile = state.getTile(adaptor.getA());
    Value rhsTile = state.getTile(adaptor.getB());

    // Calculate the shapes for the tiles involved in the matmul operation.
    // `lhsTileShape` is the shape of the LHS underlying source;
    // `rhsTileShape` is the shape of the RHS underlying source;
    // `outputTileShape` is the shape in the current iteration space;
    auto lhsTileShape = cast<ShapedType>(lhsTile.getType()).getShape();
    auto rhsTileShape = cast<ShapedType>(rhsTile.getType()).getShape();
    auto outputTileShape = state.getTileShape();

    LLVM_DEBUG({
      llvm::dbgs() << "LHS input tile shape: (";
      llvm::interleaveComma(lhsTileShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
      llvm::dbgs() << "RHS input tile shape: (";
      llvm::interleaveComma(rhsTileShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
      llvm::dbgs() << "Output tile shape: (";
      llvm::interleaveComma(outputTileShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
    });

    // Get dimension information from the matmul layout.
    // Calculate the {B, M, N, K} tile sizes.
    SmallVector<DimensionInfo> dims;
    int64_t tileB = 1, tileM = 1, tileN = 1;

    auto view = matmulSource.getCuteLayout();
    for (size_t i = 0, n = tcg::rank(view) - 1; i < n; i++) {
      int64_t stride = tcg::get(view, i).stride().as_int();

      if (stride == 0) {
        dims.push_back({DimensionType::Broadcast, i, stride});
      } else if (stride >= matmulSource.getM() * matmulSource.getN() *
                               matmulSource.getK()) {
        dims.push_back({DimensionType::Batch, i, stride});
        tileB *= outputTileShape[i];
      } else if (stride >= matmulSource.getN() * matmulSource.getK()) {
        dims.push_back({DimensionType::LeftInput, i, stride});
        tileM *= outputTileShape[i];
      } else {
        dims.push_back({DimensionType::RightInput, i, stride});
        tileN *= outputTileShape[i];
      }
    }

    // Get the contracting tile size from the LHS input.
    size_t contractingDims = matmulSource.getContractingShape().size();
    int64_t tileK = llvm::product_of(lhsTileShape.take_back(contractingDims));

    // Calculate the tile size before the broadcast (if any).
    SmallVector<int64_t> resultTileShape(outputTileShape);
    for (auto [idx, dim] : llvm::enumerate(dims)) {
      if (dim.type == DimensionType::Broadcast) {
        resultTileShape[idx] = 1;
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Result tile shape: (";
      llvm::interleaveComma(resultTileShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
    });

    // MMA output is either 2D [M, N] or 3D [B, M, N].
    // MMA inputs have the same rank as the output: [B?, M, K] and [B?, K, N].
    SmallVector<int64_t> mmaOutputShape{tileM, tileN};
    SmallVector<int64_t> mmaLhsShape{tileM, tileK};
    SmallVector<int64_t> mmaRhsShape{tileK, tileN};

    if (tileB > 1) {
      mmaOutputShape.insert(mmaOutputShape.begin(), tileB);
      mmaLhsShape.insert(mmaLhsShape.begin(), tileB);
      mmaRhsShape.insert(mmaRhsShape.begin(), tileB);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "MMA LHS input tile shape: (";
      llvm::interleaveComma(mmaLhsShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
      llvm::dbgs() << "MMA RHS input tile shape: (";
      llvm::interleaveComma(mmaRhsShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
      llvm::dbgs() << "MMA output tile shape: (";
      llvm::interleaveComma(mmaOutputShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
    });

    // Convert the result type to CUDA Tile type.
    auto resultType = cast<ShapedType>(
        getTypeConverter()->convertType(matmulOp.getResult().getType()));

    Value result;
    {
      // Set the insertion point to the inner iteration space.
      RewriterBase::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(
          blockStructure->iterationSpaces[0].insertionBlock);

      // Reshape LHS input, if needed.
      if (mmaLhsShape != lhsTileShape) {
        auto lhsType = cast<ShapedType>(lhsTile.getType());
        lhsTile = cuda_tile::ReshapeOp::create(
            rewriter, loc, lhsType.clone(mmaLhsShape), lhsTile);
      }

      // Reshape RHS input, if needed.
      if (mmaRhsShape != rhsTileShape) {
        auto rhsType = cast<ShapedType>(rhsTile.getType());
        rhsTile = cuda_tile::ReshapeOp::create(
            rewriter, loc, rhsType.clone(mmaRhsShape), rhsTile);
      }

      // For the inputs, get the original operand types (used for signedness).
      // For the output, get the result tile type.
      SmallVector<ShapedType> operandTypes{matmulOp.getA().getType(),
                                           matmulOp.getB().getType(),
                                           resultType.clone(mmaOutputShape)};

      if (!blockStructure->yieldValues.empty()) {
        // Create MMA operation inside the loop.
        Value accumulator = rewriter.getInsertionBlock()->getArgument(1);
        Value mmaResult = createMmaOp(
            rewriter, loc, {lhsTile, rhsTile, accumulator}, operandTypes);
        cuda_tile::ContinueOp::create(rewriter, loc, mmaResult);
        result = blockStructure->yieldValues[0];
      } else {
        // Create MMA operation without the loop.
        result = createMmaOp(rewriter, loc, {lhsTile, rhsTile}, operandTypes);
      }
    }

    // Normalize the result tile.
    result =
        normalizeOutput(rewriter, result, std::move(dims), resultTileShape);

    // Broadcast the result tile, if needed.
    if (resultTileShape != outputTileShape) {
      result = cuda_tile::BroadcastOp::create(
          rewriter, loc, resultType.clone(outputTileShape), result);
    }

    // Replace `MatmulOp` with the result of the matrix multiplication.
    rewriter.replaceOp(matmulOp, result);
    return success();
  }

private:
  // Dimension types in the matmul layout.
  enum class DimensionType {
    Broadcast = 0,
    Batch = 1,
    LeftInput = 2,
    RightInput = 3,
  };

  // Matmul dimension information.
  struct DimensionInfo {
    DimensionType type;
    size_t index;
    int64_t stride;

    bool operator<(const DimensionInfo &other) const {
      // Order by type [B, M, N, K], then by stride.
      if (type != other.type) {
        return static_cast<int>(type) < static_cast<int>(other.type);
      }
      return stride < other.stride;
    }
  };

  // Apply transpose and/or reshape operations to the output tile, if needed.
  static Value normalizeOutput(RewriterBase &rewriter, Value tile,
                               SmallVector<DimensionInfo> outputDims,
                               ArrayRef<int64_t> resultShape) {
    // Sort the output dimensions by type and stride.
    // This is the actual order after the MMA operation.
    llvm::sort(outputDims);

    LLVM_DEBUG({
      llvm::dbgs() << "Output dimensions index: {";
      for (size_t i = 0, n = outputDims.size(); i < n; i++) {
        llvm::dbgs() << outputDims[i].index << (i < n - 1 ? ", " : "");
      }
      llvm::dbgs() << "}\n";
    });

    // Build the transpose shape and permutation.
    SmallVector<int64_t> targetShape;
    SmallVector<int32_t> permutation(outputDims.size());
    for (auto [idx, dim] : llvm::enumerate(outputDims)) {
      targetShape.push_back(resultShape[dim.index]);
      permutation[dim.index] = idx;
    }

    // If there are fragmented output dimensions, a reshape is needed.
    auto tileType = cast<ShapedType>(tile.getType());
    if (targetShape != tileType.getShape()) {
      LLVM_DEBUG({
        llvm::dbgs() << "Reshape result to: [";
        llvm::interleaveComma(targetShape, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      // Create the reshape operation.
      tile = cuda_tile::ReshapeOp::create(rewriter, tile.getLoc(),
                                          tileType.clone(targetShape), tile);
    }

    // If the dimensions are not ordered, a transpose is needed.
    bool transposeNeeded =
        llvm::any_of(llvm::enumerate(permutation),
                     [](auto it) { return it.value() != (int)it.index(); });
    if (transposeNeeded) {
      LLVM_DEBUG({
        llvm::dbgs() << "Transpose with permutation: {";
        llvm::interleaveComma(permutation, llvm::dbgs());
        llvm::dbgs() << "}\n";
      });

      // Create the transpose operation.
      tile = cuda_tile::PermuteOp::create(rewriter, tile.getLoc(),
                                          tileType.clone(resultShape), tile,
                                          permutation);
    }

    return tile;
  }

  // Create the MMA operation (float or integer).
  // If the accumulator is not provided, create a constant zero tile.
  static Value createMmaOp(RewriterBase &rewriter, Location loc,
                           SmallVector<Value> operands,
                           ArrayRef<ShapedType> types) {
    ShapedType resultType = types.back();
    bool isFloat = resultType.getElementType().isFloat();

    // Create the accumulator, if not provided.
    if (operands.size() == 2) {
      if (isFloat) {
        operands.push_back(createConstant(rewriter, loc, resultType, 0.0));
      } else {
        operands.push_back(
            createConstant(rewriter, loc, resultType, int64_t{0}));
      }
    }

    // Create `cuda_tile::MmaFOp` or `cuda_tile::MmaIOp` based on element type.
    if (isFloat) {
      return cuda_tile::MmaFOp::create(rewriter, loc, {resultType}, operands);
    }
    return cuda_tile::MmaIOp::create(rewriter, loc, resultType, operands[0],
                                     operands[1], operands[2],
                                     getSignedness(types[0].getElementType()),
                                     getSignedness(types[1].getElementType()));
  }
};

//===----------------------------------------------------------------------===//
// Per-op legality
//===----------------------------------------------------------------------===//

LogicalResult validateMatmulOpLowerable(MatmulOp op) {
  Type lhsElementType = op.getA().getType().getElementType();
  Type rhsElementType = op.getB().getType().getElementType();
  Type outElementType = op.getResult().getType().getElementType();
  if (!isSupportedMatmulTypes(lhsElementType, rhsElementType, outElementType)) {
    return op.emitError("unsupported input/output type combination");
  }

  auto matmulSource =
      op->getAttrOfType<MatmulSourceAttr>(TensorIRDialect::getLayoutAttrName());
  if (!matmulSource) {
    return op.emitError("expected matmul layout");
  }
  if (!tcg::is_static(matmulSource.getCuteLayout())) {
    return op.emitError("matmul view layout must have static shape and stride");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerMatmulPatterns(RewritePatternSet &patterns, ConversionState &state,
                            const TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<MatmulOpConversion>(state, typeConverter, ctx);
}

} // namespace tensor_to_cuda_tile
} // namespace mlir::nv_tensor_ir
