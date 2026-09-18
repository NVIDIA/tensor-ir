// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"
#include "tensor_ir/Dialect/TensorIR.h"

#include "llvm/ADT/StringRef.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include <type_traits>

//===----------------------------------------------------------------------===//
// Tensor IR to CUDA Tile conversion emission helpers.
//
// This file contains clearly separated helper functions with no dependencies:
// - `createConstant` is commonly used to build `cuda_tile::ConstantOp`.
// - optimization-hint helpers construct load/store and entry attributes.
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

/// Create entry-point optimization hints with authoritative launch options.
cuda_tile::OptimizationHintsAttr
createEntryOptimizationHints(MLIRContext *ctx, int32_t numCTAs,
                             int32_t occupancy, int32_t numWarps) {
  SmallVector<NamedAttribute, 3> hintAttrs;
  Type intTy = IntegerType::get(ctx, 32);
  OptimizationHintNames hintNames = getOptimizationHintNames();

  auto addIntegerHint = [&](StringRef name, int32_t value) {
    hintAttrs.push_back(NamedAttribute(name, IntegerAttr::get(intTy, value)));
  };

  addIntegerHint(hintNames.numCTAInCGA, numCTAs);
  addIntegerHint(hintNames.numWorkerWarpsPerCTA, numWarps);
  addIntegerHint(hintNames.occupancy, occupancy);

  return createOptimizationHints(ctx, hintAttrs);
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
