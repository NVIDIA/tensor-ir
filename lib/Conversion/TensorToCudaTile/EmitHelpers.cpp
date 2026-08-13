// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"

#include "llvm/ADT/StringRef.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include <type_traits>

namespace tcg = mlir::nv_tensor_ir::tcutegen;

//===----------------------------------------------------------------------===//
// Tensor IR to CUDA Tile conversion emission helpers.
//
// This file contains clearly separated helper functions with no dependencies:
// - `createConstant` is commonly used to build `cuda_tile::ConstantOp`.
// - `applyLayout`, `emitLoad`, `emitStore` are used by `GraphOp` and
//   `ResultsOp` conversion patterns for emitting tile loads and stores.
// - `calculateIndex` is used to generate the index values for the main
//   iteration space.
// - `ReductionEmissionHelper` generates the attributes and operations for
//   `ReduceOp` lowering (initial value, prologue, reduction, epilogue).
//===----------------------------------------------------------------------===//

namespace mlir::nv_tensor_ir {
namespace tensor_to_cuda_tile {
namespace {

/// CUDA Tile optimization-hint names used by TensorIR lowering.
struct OptimizationHintNames {
  StringRef allowTma;
  StringRef latency;
  StringRef numCTAInCGA;
  StringRef occupancy;
  StringRef numWorkerWarpsPerCTA;
};

/// Return optimization-hint names for the configured CUDA Tile.
OptimizationHintNames getOptimizationHintNames() {
  auto resolve = [](auto &&name) -> StringRef {
    if constexpr (std::is_invocable_v<decltype(name)>) {
      return name();
    } else {
      return name;
    }
  };
  return {
      resolve(cuda_tile::OptimizationHintsAttr::kAllowTMA),
      resolve(cuda_tile::OptimizationHintsAttr::kLatency),
      resolve(cuda_tile::OptimizationHintsAttr::kNumCTAInCGA),
      resolve(cuda_tile::OptimizationHintsAttr::kOccupancy),
      resolve(cuda_tile::OptimizationHintsAttr::kNumWorkerWarpsPerCTA),
  };
}

/// Wrap architecture-independent hints in a CUDA Tile attribute.
cuda_tile::OptimizationHintsAttr
createOptimizationHints(MLIRContext *ctx, ArrayRef<NamedAttribute> hintAttrs) {
  if (hintAttrs.empty()) {
    return nullptr;
  }

  NamedAttribute defaultHints("default", DictionaryAttr::get(ctx, hintAttrs));
  return cuda_tile::OptimizationHintsAttr::get(
      ctx, DictionaryAttr::get(ctx, {defaultHints}));
}

} // namespace

/// Create floating-point constant.
Value createConstant(OpBuilder &rewriter, Location loc, ShapedType type,
                     double value) {
  auto scalarAttr = rewriter.getFloatAttr(type.getElementType(), value);
  auto denseAttr =
      cast<DenseTypedElementsAttr>(DenseElementsAttr::get(type, scalarAttr));
  return cuda_tile::ConstantOp::create(rewriter, loc, type, denseAttr);
}

/// Create integer constant.
Value createConstant(OpBuilder &rewriter, Location loc, ShapedType type,
                     int64_t value) {
  auto scalarAttr = rewriter.getIntegerAttr(type.getElementType(), value);
  auto denseAttr =
      cast<DenseTypedElementsAttr>(DenseElementsAttr::get(type, scalarAttr));
  return cuda_tile::ConstantOp::create(rewriter, loc, type, denseAttr);
}

/// Get signedness enum value for the given TensorIR integer type.
/// Treat signless boolean type (i1) as unsigned.
cuda_tile::Signedness getSignedness(Type type) {
  assert(type.isInteger() && "expected integer type");
  return type.isSignedInteger() ? cuda_tile::Signedness::Signed
                                : cuda_tile::Signedness::Unsigned;
}

//===----------------------------------------------------------------------===//
// Load/store emission
//===----------------------------------------------------------------------===//

/// Update the tensor descriptor with the sizes/strides from the layout.
/// If an offset is present, also update the tensor pointer.
/// If an alignment is present, apply `cuda_tile::AssumeOp` to the pointer.
TensorDescriptor applyLayout(OpBuilder &rewriter, const TensorDescriptor &desc,
                             TensorSourceAttr tensorSource) {
  TensorDescriptor result = desc;
  Location loc = desc.pointer.getLoc();

  // Build the array of dynamic values (if present).
  SmallVector<Value> dynamicValues;
  for (auto &size : desc.sizes) {
    if (size.dynamicValue) {
      dynamicValues.push_back(size.dynamicValue);
    }
  }
  for (auto &stride : desc.strides) {
    if (stride.dynamicValue) {
      dynamicValues.push_back(stride.dynamicValue);
    }
  }

  // Apply the dynamic offsets permutation (if present).
  if (!tensorSource.getDynamicValueMapping().empty()) {
    SmallVector<Value> permutedValues;
    for (int idx : tensorSource.getDynamicValueMapping()) {
      permutedValues.push_back(dynamicValues[idx]);
    }
    dynamicValues = std::move(permutedValues);
  }

  // Get the CuTe layout from the tensor source.
  auto layout = tensorSource.getCuteLayout();
  size_t rank = tcg::rank(layout);
  auto dynamicIter = dynamicValues.begin();

  // Update the dimension sizes.
  result.sizes.clear();
  for (size_t i = 0; i < rank; ++i) {
    auto cgSize = tcg::get(layout.shape(), i);
    if (tcg::is_static(cgSize)) {
      result.sizes.push_back({tcg::static_size(cgSize), nullptr});
    } else {
      result.sizes.push_back({ShapedType::kDynamic, *dynamicIter++});
    }
  }

  // Update the dimension strides.
  result.strides.clear();
  for (size_t i = 0; i < rank; ++i) {
    auto cgStride = tcg::get(layout.stride(), i);
    if (tcg::is_static(cgStride)) {
      result.strides.push_back({tcg::static_size(cgStride), nullptr});
    } else {
      result.strides.push_back({ShapedType::kDynamic, *dynamicIter++});
    }
  }

  // Apply the offset, if present.
  if (tensorSource.getOffset() != 0) {
    ShapedType offsetType = cuda_tile::TileType::get(
        /*shape=*/llvm::ArrayRef<int64_t>{}, rewriter.getI64Type());
    Value offsetValue =
        createConstant(rewriter, loc, offsetType, tensorSource.getOffset());
    result.pointer =
        cuda_tile::OffsetOp::create(rewriter, loc, result.pointer, offsetValue);
  }

  // Get or infer the alignment and update the tensor pointer.
  // Applying the alignment to the pointer enables the vectorized load/store
  // operations in the CUDA Tile lowering.

  Type ptrType = cast<ShapedType>(result.pointer.getType()).getElementType();
  Type elemType = cast<cuda_tile::PointerType>(ptrType).getPointeeType();
  int64_t elemSizeInBytes = std::max(1u, elemType.getIntOrFloatBitWidth() / 8);

  // Use per-tensor alignment if explicitly set, otherwise use the default.
  if (result.alignment <= 0) {
    result.alignment = kDefaultPointerAlignment;
  }

  // Adjust the alignment if the offset is not aligned.
  int64_t offsetInBytes = tensorSource.getOffset() * elemSizeInBytes;
  if (offsetInBytes % result.alignment != 0) {
    result.alignment = std::gcd(result.alignment, offsetInBytes);
  }

  // Apply the alignment, if needed.
  if (result.alignment > elemSizeInBytes) {
    auto divByAttr = cuda_tile::DivByAttr::get(
        rewriter.getContext(), result.alignment, /*every=*/std::nullopt,
        /*along=*/std::nullopt);
    result.pointer =
        cuda_tile::AssumeOp::create(rewriter, loc, result.pointer, divByAttr);
  }

  return result;
}

/// Create partition view for a tensor descriptor.
Value createPartitionView(OpBuilder &rewriter, const TensorDescriptor &desc,
                          ArrayRef<int64_t> tileShape) {
  // Get the element type from the descriptor.
  Type ptrType = cast<ShapedType>(desc.pointer.getType()).getElementType();
  Type elemType = cast<cuda_tile::PointerType>(ptrType).getPointeeType();

  // Create static/dynamic vectors from the descriptor.
  SmallVector<int64_t> staticSizes, staticStrides;
  SmallVector<Value> dynamicSizes, dynamicStrides;
  for (auto [size, stride] : llvm::zip_equal(desc.sizes, desc.strides)) {
    staticSizes.push_back(size.staticValue);
    if (size.dynamicValue) {
      dynamicSizes.push_back(size.dynamicValue);
    }
    staticStrides.push_back(stride.staticValue);
    if (stride.dynamicValue) {
      dynamicStrides.push_back(stride.dynamicValue);
    }
    // Fix broadcasted dimensions.
    if (stride.staticValue == 0) {
      assert(!size.dynamicValue && "broadcasted dimension cannot be dynamic");
      staticSizes.back() = 1;
      staticStrides.back() = 1;
    }
  }

  // Create the tensor view.
  auto tensorViewType = cuda_tile::TensorViewType::get(
      rewriter.getContext(), elemType, staticSizes, staticStrides);
  Value tensorView = cuda_tile::MakeTensorViewOp::create(
      rewriter, desc.pointer.getLoc(), tensorViewType, desc.pointer,
      dynamicSizes, dynamicStrides);

  // Create partition view type components.
  SmallVector<int32_t> tileSizes(tileShape.begin(), tileShape.end());
  auto tileSizesAttr = DenseI32ArrayAttr::get(rewriter.getContext(), tileSizes);

  SmallVector<int32_t> dimMap(tileShape.size());
  std::iota(dimMap.begin(), dimMap.end(), 0);

  // Create the partition view.
  auto partitionViewType = cuda_tile::PartitionViewType::get(
      rewriter.getContext(), tileSizesAttr, tensorViewType, dimMap,
      /*padding_value=*/{});
  return cuda_tile::MakePartitionViewOp::create(rewriter, tensorView.getLoc(),
                                                partitionViewType, tensorView);
}

/// Create load/store optimization hints for non-default descriptor options.
cuda_tile::OptimizationHintsAttr
createLoadStoreOptimizationHints(MLIRContext *ctx, bool allowTma,
                                 int32_t latency) {
  SmallVector<NamedAttribute, 2> hintAttrs;
  OptimizationHintNames hintNames = getOptimizationHintNames();

  if (!allowTma) {
    hintAttrs.push_back(
        NamedAttribute(hintNames.allowTma, BoolAttr::get(ctx, allowTma)));
  }
  if (latency >= 0) {
    hintAttrs.push_back(
        NamedAttribute(hintNames.latency,
                       IntegerAttr::get(IntegerType::get(ctx, 32), latency)));
  }

  return createOptimizationHints(ctx, hintAttrs);
}

/// Create entry-point optimization hints for non-default launch options.
cuda_tile::OptimizationHintsAttr
createEntryOptimizationHints(MLIRContext *ctx, int32_t numCTAs,
                             int32_t occupancy, int32_t numWarps) {
  SmallVector<NamedAttribute, 3> hintAttrs;
  Type intTy = IntegerType::get(ctx, 32);
  OptimizationHintNames hintNames = getOptimizationHintNames();

  auto addIfNotDefault = [&](StringRef name, int32_t value,
                             int32_t defaultValue) {
    if (value != defaultValue) {
      hintAttrs.push_back(NamedAttribute(name, IntegerAttr::get(intTy, value)));
    }
  };
  addIfNotDefault(hintNames.numCTAInCGA, numCTAs, 1);
  addIfNotDefault(hintNames.occupancy, occupancy, 1);
  addIfNotDefault(hintNames.numWorkerWarpsPerCTA, numWarps, 4);

  return createOptimizationHints(ctx, hintAttrs);
}

/// Emit load operation (`cuda_tile::LoadViewTkoOp`).
Value emitLoad(OpBuilder &rewriter, const TensorDescriptor &desc,
               ShapedType tileType, ValueRange indexValues) {
  Value partitionView =
      createPartitionView(rewriter, desc, tileType.getShape());
  Type tokenType = cuda_tile::TokenType::get(rewriter.getContext());
  auto optimizationHints = createLoadStoreOptimizationHints(
      rewriter.getContext(), desc.allowTma, desc.cost);

  auto loadOp = cuda_tile::LoadViewTkoOp::create(
      rewriter, partitionView.getLoc(), tileType, tokenType,
      cuda_tile::MemoryOrderingSemantics::WEAK, /*memory_scope=*/nullptr,
      partitionView, indexValues, /*token=*/nullptr, optimizationHints);
  return loadOp.getResult(0);
}

/// Emit store operation (`cuda_tile::StoreViewTkoOp`).
void emitStore(OpBuilder &rewriter, const TensorDescriptor &desc, Value tile,
               ValueRange indexValues) {
  Value partitionView = createPartitionView(
      rewriter, desc, cast<ShapedType>(tile.getType()).getShape());
  Type tokenType = cuda_tile::TokenType::get(rewriter.getContext());
  auto optimizationHints = createLoadStoreOptimizationHints(
      rewriter.getContext(), desc.allowTma, desc.cost);

  cuda_tile::StoreViewTkoOp::create(
      rewriter, partitionView.getLoc(), tokenType,
      cuda_tile::MemoryOrderingSemantics::WEAK, /*memory_scope=*/nullptr, tile,
      partitionView, indexValues, /*token=*/nullptr, optimizationHints);
}

//===----------------------------------------------------------------------===//
// Index calculation
//===----------------------------------------------------------------------===//

/// Calculate index values for an iteration space with static sizes.
/// This should be more efficient than the dynamic version (less div/mod ops).
SmallVector<Value> calculateStaticIndex(OpBuilder &rewriter, Value blockId,
                                        ArrayRef<int64_t> iterationSpaceShape,
                                        ArrayRef<int64_t> tileShape) {
  SmallVector<Value> indexValues;
  Location loc = blockId.getLoc();

  // Use I32 type for load indices.
  ShapedType indexType = cuda_tile::TileType::get(
      /*shape=*/llvm::ArrayRef<int64_t>{}, rewriter.getI32Type());

  // Find the last dimension that is tiled.
  size_t rank = iterationSpaceShape.size();
  size_t lastTiledDim = SIZE_MAX;
  for (size_t i = 0; i < rank; i++) {
    if (iterationSpaceShape[i] > tileShape[i]) {
      lastTiledDim = i;
    }
  }

  // Repeat splitting the block ID into left/right parts.
  // "Left" part is the block ID modulo the number of blocks in the dimension,
  // and the "right" part is the remaining number of blocks.
  for (size_t i = 0; i < rank; i++) {
    int64_t blockCountInDim =
        llvm::divideCeil(iterationSpaceShape[i], tileShape[i]);
    if (blockCountInDim != 1) {
      if (i != lastTiledDim) {
        // Calculate the index value and update the remainder.
        Value cst = createConstant(rewriter, loc, indexType, blockCountInDim);
        indexValues.push_back(cuda_tile::RemIOp::create(
            rewriter, loc, blockId, cst, cuda_tile::Signedness::Unsigned));
        blockId = cuda_tile::DivIOp::create(rewriter, loc, blockId, cst,
                                            cuda_tile::Signedness::Unsigned);
      } else {
        // Last tiled dimension uses the remainder.
        indexValues.push_back(blockId);
      }
    } else {
      // Dimension is not tiled, use zero index.
      indexValues.push_back(
          createConstant(rewriter, loc, indexType, int64_t{0}));
    }
  }

  return indexValues;
}

/// Calculate index values for an iteration space with dynamic sizes.
SmallVector<Value> calculateDynamicIndex(OpBuilder &rewriter, Value blockId,
                                         ValueRange dynamicSizes) {
  SmallVector<Value> indexValues;
  Location loc = blockId.getLoc();

  // Repeat splitting the block ID into left/right parts.
  // "Left" part is the block ID modulo the number of blocks in the dimension,
  // and the "right" part is the remaining number of blocks.
  for (size_t i = 0, lastDim = dynamicSizes.size() - 1; i < lastDim; i++) {
    indexValues.push_back(
        cuda_tile::RemIOp::create(rewriter, loc, blockId, dynamicSizes[i],
                                  cuda_tile::Signedness::Unsigned));
    blockId = cuda_tile::DivIOp::create(rewriter, loc, blockId, dynamicSizes[i],
                                        cuda_tile::Signedness::Unsigned);
  }

  indexValues.push_back(blockId);
  return indexValues;
}

/// Calculate index values for an iteration space.
FailureOr<SmallVector<Value>> calculateIndex(OpBuilder &rewriter, Value blockId,
                                             const TensorDescriptor &desc,
                                             ArrayRef<int64_t> tileShape) {
  // Use static index calculation, if possible.
  bool isStatic =
      llvm::all_of(desc.sizes, [](auto &size) { return !size.dynamicValue; });
  if (isStatic) {
    SmallVector<int64_t> tensorShape;
    for (auto &size : desc.sizes) {
      tensorShape.push_back(size.staticValue);
    }
    return calculateStaticIndex(rewriter, blockId, tensorShape, tileShape);
  }

  // Use `GetIndexSpaceShapeOp` to get the block sizes in the iteration space.
  Value partitionView = createPartitionView(rewriter, desc, tileShape);
  SmallVector<Type> indexTypes(tileShape.size(), blockId.getType());
  auto getIndexSpaceShapeOp = cuda_tile::GetIndexSpaceShapeOp::create(
      rewriter, partitionView.getLoc(), indexTypes, partitionView);
  return calculateDynamicIndex(rewriter, blockId,
                               getIndexSpaceShapeOp.getResults());
}

//===----------------------------------------------------------------------===//
// Reduction emission
//===----------------------------------------------------------------------===//

/// Build the identity value for a reduction operation.
Attribute ReductionEmissionHelper::getIdentity(OpBuilder &rewriter) {
  auto convertIntegerType = [&](Type type) {
    assert(type.isInteger() && "expected integer type");
    return rewriter.getIntegerType(type.getIntOrFloatBitWidth());
  };

  switch (mode) {
  case ReductionMode::add:
  case ReductionMode::amax:
  case ReductionMode::avg:
  case ReductionMode::norm1:
  case ReductionMode::norm2:
    // Use zero identity value.
    if (elementType.isFloat()) {
      return rewriter.getFloatAttr(elementType, 0.0);
    } else {
      return rewriter.getIntegerAttr(convertIntegerType(elementType), 0);
    }

  case ReductionMode::mul:
  case ReductionMode::mul_no_zeros:
    // Use unit identity value for multiplication.
    if (elementType.isFloat()) {
      return rewriter.getFloatAttr(elementType, 1.0);
    } else {
      return rewriter.getIntegerAttr(convertIntegerType(elementType), 1);
    }

  case ReductionMode::max:
    // Use minimum possible value for "max" reduction.
    if (elementType.isFloat()) {
      return rewriter.getFloatAttr(elementType,
                                   -std::numeric_limits<float>::infinity());
    } else {
      unsigned bitWidth = elementType.getIntOrFloatBitWidth();
      return rewriter.getIntegerAttr(convertIntegerType(elementType),
                                     elementType.isSignedInteger()
                                         ? APInt::getSignedMinValue(bitWidth)
                                         : APInt::getMinValue(bitWidth));
    }

  case ReductionMode::min:
    // Use maximum possible value for "min" reduction.
    if (elementType.isFloat()) {
      return rewriter.getFloatAttr(elementType,
                                   std::numeric_limits<float>::infinity());
    } else {
      unsigned bitWidth = elementType.getIntOrFloatBitWidth();
      return rewriter.getIntegerAttr(convertIntegerType(elementType),
                                     elementType.isSignedInteger()
                                         ? APInt::getSignedMaxValue(bitWidth)
                                         : APInt::getMaxValue(bitWidth));
    }

  default:
    return nullptr;
  }
}

// Build the prologue for a reduction operation.
Value ReductionEmissionHelper::buildPrologue(OpBuilder &rewriter,
                                             Value inputTile) {
  Location loc = inputTile.getLoc();

  switch (mode) {
  case ReductionMode::amax:
  case ReductionMode::norm1:
    // Calculate absolute value.
    if (elementType.isFloat()) {
      return cuda_tile::AbsFOp::create(rewriter, loc, inputTile);
    } else if (elementType.isSignedInteger()) {
      return cuda_tile::AbsIOp::create(rewriter, loc, inputTile);
    } else {
      return inputTile;
    }

  case ReductionMode::norm2:
    // Calculate square value for L2 norm.
    if (elementType.isFloat()) {
      return cuda_tile::MulFOp::create(rewriter, loc, inputTile, inputTile,
                                       cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      return cuda_tile::MulIOp::create(rewriter, loc, inputTile, inputTile);
    }

  case ReductionMode::mul_no_zeros:
    // If the value is zero, replace it with one.
    if (elementType.isFloat()) {
      auto inputType = cast<ShapedType>(inputTile.getType());
      Value zero = createConstant(rewriter, loc, inputType, 0.0);
      Value one = createConstant(rewriter, loc, inputType, 1.0);

      auto isZero = cuda_tile::CmpFOp::create(
          rewriter, loc, cuda_tile::ComparisonPredicate::EQUAL,
          cuda_tile::ComparisonOrdering::ORDERED, inputTile, zero);
      return cuda_tile::SelectOp::create(rewriter, loc, isZero, one, inputTile);
    } else {
      auto inputType = cast<ShapedType>(inputTile.getType());
      Value zero = createConstant(rewriter, loc, inputType, int64_t{0});
      Value one = createConstant(rewriter, loc, inputType, int64_t{1});

      auto isZero = cuda_tile::CmpIOp::create(
          rewriter, loc, cuda_tile::ComparisonPredicate::EQUAL, inputTile, zero,
          getSignedness(elementType));
      return cuda_tile::SelectOp::create(rewriter, loc, isZero, one, inputTile);
    }

  default:
    return inputTile;
  }
}

// Accumulate values for a reduction operation.
Value ReductionEmissionHelper::buildReduction(OpBuilder &rewriter,
                                              Value accumulator, Value tile) {
  Location loc = tile.getLoc();

  switch (mode) {
  case ReductionMode::add:
  case ReductionMode::avg:
  case ReductionMode::norm1:
  case ReductionMode::norm2:
    // Apply addition.
    if (elementType.isFloat()) {
      return cuda_tile::AddFOp::create(rewriter, loc, accumulator, tile,
                                       cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      return cuda_tile::AddIOp::create(rewriter, loc, accumulator, tile);
    }

  case ReductionMode::mul:
  case ReductionMode::mul_no_zeros:
    // Apply multiplication.
    if (elementType.isFloat()) {
      return cuda_tile::MulFOp::create(rewriter, loc, accumulator, tile,
                                       cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      return cuda_tile::MulIOp::create(rewriter, loc, accumulator, tile);
    }

  case ReductionMode::max:
  case ReductionMode::amax:
    // Calculate maximum.
    if (elementType.isFloat()) {
      return cuda_tile::MaxFOp::create(rewriter, loc, accumulator, tile,
                                       /*propagate_nan=*/true);
    } else {
      return cuda_tile::MaxIOp::create(rewriter, loc, accumulator, tile,
                                       getSignedness(elementType));
    }

  case ReductionMode::min:
    // Calculate minimum.
    if (elementType.isFloat()) {
      return cuda_tile::MinFOp::create(rewriter, loc, accumulator, tile,
                                       /*propagate_nan=*/true);
    } else {
      return cuda_tile::MinIOp::create(rewriter, loc, accumulator, tile,
                                       getSignedness(elementType));
    }

  default:
    return nullptr;
  }
}

// Build the epilogue for a reduction operation.
Value ReductionEmissionHelper::buildEpilogue(OpBuilder &rewriter,
                                             Value outputTile,
                                             int64_t reductionSize) {
  Location loc = outputTile.getLoc();

  switch (mode) {
  case ReductionMode::norm2:
    // Apply square root for L2 norm.
    if (elementType.isFloat()) {
      return cuda_tile::SqrtOp::create(rewriter, loc, outputTile,
                                       cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      return nullptr;
    }

  case ReductionMode::avg:
    // Divide by reduction size for average.
    if (elementType.isFloat()) {
      auto outputType = cast<ShapedType>(outputTile.getType());
      auto divisor =
          createConstant(rewriter, loc, outputType, double(reductionSize));
      return cuda_tile::DivFOp::create(rewriter, loc, outputTile, divisor,
                                       cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      auto outputType = cast<ShapedType>(outputTile.getType());

      // If the reduction size constant doesn't fit in the element type, the
      // result of the division is always zero (accumulator likely overflows).
      unsigned bitWidth = elementType.getIntOrFloatBitWidth();
      APInt maxIntValue = elementType.isSignedInteger()
                              ? APInt::getSignedMaxValue(bitWidth) + 1
                              : APInt::getMaxValue(bitWidth);
      if (uint64_t(reductionSize) > maxIntValue.getZExtValue()) {
        return createConstant(rewriter, loc, outputType, int64_t{0});
      }

      auto divisor = createConstant(rewriter, loc, outputType, reductionSize);
      return cuda_tile::DivIOp::create(rewriter, loc, outputTile, divisor,
                                       getSignedness(elementType));
    }

  default:
    return outputTile;
  }
}

} // namespace tensor_to_cuda_tile
} // namespace mlir::nv_tensor_ir
