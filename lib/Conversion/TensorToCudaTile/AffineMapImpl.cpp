// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/TileAnalyzer.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Dialect/TensorIRAttrs.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

#define DEBUG_TYPE "convert-tensor-to-cuda-tile-affine-map"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

namespace {

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

static SmallVector<Value>
getFilteredDynamicSizesOrStrides(ArrayRef<int64_t> staticSizes,
                                 ArrayRef<Value> dynamicSizes) {
  SmallVector<Value> filtered;
  size_t dynamicIndex = 0;
  for (int64_t staticSize : staticSizes) {
    if (staticSize == cuda_tile::TensorViewType::kDynamic) {
      assert(dynamicIndex < dynamicSizes.size() && "Not enough dynamic values");
      filtered.push_back(dynamicSizes[dynamicIndex++]);
    }
  }
  return filtered;
}

static SmallVector<int64_t>
getFilteredDynamicDivisibility(ArrayRef<int64_t> staticValues,
                               ArrayRef<int64_t> divisibility) {
  assert(staticValues.size() == divisibility.size() &&
         "Divisibility metadata must match descriptor rank");
  SmallVector<int64_t> filtered;
  for (auto [staticValue, divisor] :
       llvm::zip_equal(staticValues, divisibility)) {
    if (staticValue == cuda_tile::TensorViewType::kDynamic) {
      filtered.push_back(divisor);
    }
  }
  return filtered;
}

static int64_t getIterSpaceDimIdx(AffineExpr expr, int64_t exprIdx) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    return static_cast<int64_t>(dimExpr.getPosition());
  }
  if (isa<AffineConstantExpr>(expr)) {
    return exprIdx;
  }

  llvm_unreachable("Expected affine dim or constant expression");
}

/// Get the iteration space ID for an operation. Returns 0 if not set.
/// TODO: Assert when affine map approach is removed.
static int32_t getIterSpaceIdForOp(Operation *op) {
  auto attr =
      op->getAttrOfType<IntegerAttr>(TensorIRDialect::getIterSpaceIdAttrName());
  if (!attr) {
    return 0;
  }
  return attr.getInt();
}

/// Collect all unique iteration space IDs from a graph operation.
/// Returns a sorted list of unique iteration space IDs found in the graph.
static SmallVector<int32_t> collectIterSpaceIds(nv_tensor_ir::GraphOp graphOp) {
  DenseSet<int32_t> uniqueIds;

  // Collect from all operations in the graph.
  graphOp.walk([&](Operation *op) {
    // Skip the graph and results operations.
    if (isa<nv_tensor_ir::GraphOp>(op) || isa<nv_tensor_ir::ResultsOp>(op)) {
      return;
    }

    auto attr = op->getAttrOfType<IntegerAttr>(
        TensorIRDialect::getIterSpaceIdAttrName());
    if (attr) {
      uniqueIds.insert(attr.getInt());
    }
  });

  // If no iteration space IDs found, default to iteration space 0.
  if (uniqueIds.empty()) {
    uniqueIds.insert(0);
  }

  // Return as sorted vector for deterministic processing.
  SmallVector<int32_t> sortedIds(uniqueIds.begin(), uniqueIds.end());
  llvm::sort(sortedIds);
  return sortedIds;
}

/// Count the number of dynamic dimensions in a tensor shape.
static int countDynamicDimensions(ArrayRef<int64_t> shape) {
  return llvm::count_if(shape, [](int64_t dim) {
    return dim == cuda_tile::TensorViewType::kDynamic;
  });
}

static SmallVector<int64_t> convertInt32ToI64(ArrayRef<int32_t> int32Values) {
  SmallVector<int64_t> i64Values;
  i64Values.reserve(int32Values.size());
  llvm::transform(int32Values, std::back_inserter(i64Values),
                  [](int32_t value) { return static_cast<int64_t>(value); });
  return i64Values;
}

static SmallVector<int32_t> convertInt64ToI32(ArrayRef<int64_t> int64Values) {
  SmallVector<int32_t> i32Values;
  i32Values.reserve(int64Values.size());
  llvm::transform(int64Values, std::back_inserter(i32Values),
                  [](int64_t value) { return static_cast<int32_t>(value); });
  return i32Values;
}

/// Static shape/layout facts decoded from TensorIR tensor descriptors.
///
/// TensorIR no longer encodes dynamic shape divisibility in TensorType. Dynamic
/// stride divisibility still comes from the stride layout string
/// `?{div=N}` and is relayed to make_tensor_view operands.
struct TensorDescriptorStaticMetadata {
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
  SmallVector<int64_t> strideDivisibility;
  bool hasExplicitStrides = false;
};

/// Holds IR references to the different components of a tensor descriptor in
/// CUDA Tile IR. This includes, the pointer, dimension sizes and strides. It's
/// currently used for input and output tensors, hence their specific use of
/// BlockArguments.
class CudaTileTensorDescriptor {
public:
  enum TensorKind { Input, Output };

  CudaTileTensorDescriptor(Value tensorIRTensor, BlockArgument cudaTilePtr,
                           ArrayRef<int64_t> staticSizes,
                           ArrayRef<BlockArgument> dynamicSizes,
                           ArrayRef<int64_t> staticStrides,
                           ArrayRef<BlockArgument> dynamicStrides,
                           ArrayRef<int64_t> strideDivisibility,
                           TensorKind tensorKind, AffineMap iterSpaceMap,
                           ArrayRef<int32_t> iterSpaceIds, bool allowTma = true,
                           int64_t alignment = 0, int32_t cost = -1)
      : tensorIRTensor(tensorIRTensor), cudaTilePtr(cudaTilePtr),
        staticSizes(staticSizes), dynamicSizes(dynamicSizes),
        staticStrides(staticStrides), dynamicStrides(dynamicStrides),
        strideDivisibility(strideDivisibility), tensorKind(tensorKind),
        iterSpaceMap(iterSpaceMap), allowTma(allowTma), alignment(alignment),
        cost(cost) {
    if (!iterSpaceIds.empty()) {
      this->iterSpaceIds.insert(iterSpaceIds.begin(), iterSpaceIds.end());
    } else {
      this->iterSpaceIds.insert(0);
    }
    assert(staticSizes.size() == staticStrides.size() &&
           "Sizes and strides must have the same rank");
    assert(staticStrides.size() == this->strideDivisibility.size() &&
           "Stride divisibility metadata must match rank");
  }

  /// Get the original Tensor IR tensor (before conversion).
  Value getTensorIRTensor() const { return tensorIRTensor; }

  /// Get the CUDA Tile pointer (converted from Tensor IR).
  BlockArgument getCudaTilePtr() const { return cudaTilePtr; }

  ArrayRef<int64_t> getStaticSizes() const { return staticSizes; }

  ArrayRef<Value> getDynamicSizes() const { return dynamicSizes; }

  ArrayRef<int64_t> getStaticStrides() const { return staticStrides; }

  ArrayRef<Value> getDynamicStrides() const { return dynamicStrides; }

  bool isInput() const { return tensorKind == TensorKind::Input; }

  bool isOutput() const { return tensorKind == TensorKind::Output; }

  int64_t getRank() const { return staticSizes.size(); }

  Value getTensorView() const {
    assert(tensorView && "Tensor view not set");
    return tensorView;
  }

  void setTensorView(Value tv) { tensorView = tv; }

  bool hasZeroPaddedPartitionView() const {
    return static_cast<bool>(zeroPaddedPartitionView);
  }

  Value getZeroPaddedPartitionView() const {
    assert(zeroPaddedPartitionView && "Zero-padded partition view not set");
    return zeroPaddedPartitionView;
  }

  void setZeroPaddedPartitionView(Value view) {
    assert(view && "Zero-padded partition view is null");
    zeroPaddedPartitionView = view;
  }

  Value getLoadedTile() const {
    assert(loadedTile && "Loaded tile not set");
    return loadedTile;
  }

  void setLoadedTile(Value tile) { loadedTile = tile; }

  AffineMap getIterSpaceMap() const {
    assert(iterSpaceMap && "Iteration space map hasn't been set");
    return iterSpaceMap;
  }

  int32_t getIterSpaceId() const {
    assert(!iterSpaceIds.empty() && "No iteration space IDs set");
    return *iterSpaceIds.begin();
  }

  /// Returns true if this descriptor belongs to the given iteration space.
  bool belongsToIterSpace(int32_t id) const {
    return iterSpaceIds.contains(id);
  }

  /// Returns true if strides should be computed from sizes (row-major).
  /// This is derived from the state: dynamicStrides is empty but staticStrides
  /// has dynamic components that need to be computed at runtime.
  bool shouldComputeStridesFromSizes() const {
    return dynamicStrides.empty() && countDynamicDimensions(staticStrides) > 0;
  }

  SmallVector<Value> getFilteredDynamicSizes() const {
    return getFilteredDynamicSizesOrStrides(staticSizes, dynamicSizes);
  }

  /// Returns the dynamic strides for dimensions that are not static.
  SmallVector<Value> getFilteredDynamicStrides() const {
    return getFilteredDynamicSizesOrStrides(staticStrides, dynamicStrides);
  }

  SmallVector<int64_t> getFilteredDynamicStrideDivisibility() const {
    return getFilteredDynamicDivisibility(staticStrides, strideDivisibility);
  }

  /// Returns whether TMA is allowed for this tensor.
  bool getAllowTma() const { return allowTma; }

  /// Returns the per-tensor alignment in bytes (0 means use auto-detection).
  int64_t getAlignment() const { return alignment; }

  /// Returns the per-tensor cost/latency hint (-1 means use default).
  int32_t getCost() const { return cost; }

private:
  /// Original Tensor IR tensor (before conversion).
  Value tensorIRTensor;
  /// CUDA Tile pointer (converted from Tensor IR).
  BlockArgument cudaTilePtr;
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  SmallVector<int64_t> strideDivisibility;
  Value tensorView;
  Value zeroPaddedPartitionView;
  Value loadedTile;
  TensorKind tensorKind;
  AffineMap iterSpaceMap;
  /// All iteration spaces this descriptor belongs to.
  llvm::SmallDenseSet<int32_t, 4> iterSpaceIds;
  bool allowTma;
  int64_t alignment;
  int32_t cost;
};

/// Process a single tensor (input or output) and create its descriptor.
/// Returns the number of arguments consumed for this tensor.
/// @param metadata Static descriptor sizes, strides, and divisibility facts.
///                 If metadata has no explicit strides, row-major strides are
///                 computed from sizes and no stride args are expected.
static size_t processTensorDescriptor(
    nv_tensor_ir::TensorType tensorTy, Value tensorIRPtr,
    ArrayRef<BlockArgument> cudaTileArgs, size_t ptrIdx,
    const TensorDescriptorStaticMetadata &metadata,
    CudaTileTensorDescriptor::TensorKind tensorKind, AffineMap iterSpaceMap,
    ArrayRef<int32_t> iterSpaceIds, bool allowTma, int64_t alignment,
    int32_t cost,
    SmallVectorImpl<CudaTileTensorDescriptor> &inOutTensorDescriptors,
    bool uniformSignature = false) {
  assert(llvm::equal(tensorTy.getShape(), metadata.sizes) &&
         "Tensor type shape must match descriptor metadata");

  size_t sizeStartIdx = ptrIdx + 1;

  // When uniformSignature is true, all dims get kernel args.
  // When false, only dynamic dims get kernel args (original behavior).
  int numSizeArgs = uniformSignature ? static_cast<int>(metadata.sizes.size())
                                     : countDynamicDimensions(metadata.sizes);

  int numStrideArgs = 0;
  if (metadata.hasExplicitStrides) {
    numStrideArgs = uniformSignature ? static_cast<int>(metadata.strides.size())
                                     : countDynamicDimensions(metadata.strides);
  }

  size_t strideStartIdx = sizeStartIdx + numSizeArgs;
  size_t numArgsConsumed = 1 + numSizeArgs + numStrideArgs;

  // Extract block args that correspond to dynamic dims.
  // When uniform, all dims have block args but only dynamic ones go into the
  // descriptor (static dim block args become unused dead parameters).
  SmallVector<BlockArgument> dynamicSizes;
  if (uniformSignature) {
    for (int d = 0; d < numSizeArgs; ++d) {
      if (metadata.sizes[d] == cuda_tile::TensorViewType::kDynamic) {
        dynamicSizes.push_back(cudaTileArgs[sizeStartIdx + d]);
      }
    }
  } else {
    dynamicSizes.assign(cudaTileArgs.slice(sizeStartIdx, numSizeArgs).begin(),
                        cudaTileArgs.slice(sizeStartIdx, numSizeArgs).end());
  }

  SmallVector<BlockArgument> dynamicStrides;
  if (metadata.hasExplicitStrides) {
    if (uniformSignature) {
      for (int d = 0; d < numStrideArgs; ++d) {
        if (d < static_cast<int>(metadata.strides.size()) &&
            metadata.strides[d] == cuda_tile::TensorViewType::kDynamic) {
          dynamicStrides.push_back(cudaTileArgs[strideStartIdx + d]);
        }
      }
    } else {
      dynamicStrides.assign(
          cudaTileArgs.slice(strideStartIdx, numStrideArgs).begin(),
          cudaTileArgs.slice(strideStartIdx, numStrideArgs).end());
    }
  }

  CudaTileTensorDescriptor descriptor(
      /*tensorIRTensor=*/tensorIRPtr,
      /*cudaTilePtr=*/cudaTileArgs[ptrIdx],
      /*staticSizes=*/metadata.sizes,
      /*dynamicSizes=*/dynamicSizes,
      /*staticStrides=*/metadata.strides,
      /*dynamicStrides=*/dynamicStrides,
      /*strideDivisibility=*/metadata.strideDivisibility, tensorKind,
      iterSpaceMap, iterSpaceIds, allowTma, alignment, cost);
  inOutTensorDescriptors.push_back(descriptor);

  return numArgsConsumed;
}

/// Strips unrealized conversion casts from a value.
///
/// The conversion infra converts the Tensor IR tensor to a pointer tile type by
/// default. Since the actual reaching converted value is a shaped tile, it will
/// introduce an unrealized conversion op from cuda.tile<XxYxZxelmType> to
/// cuda.tile<ptr<elmType>>. We skip over the unrealized conversion op and use
/// the value directly.
static Value stripUnrealizedConversionCast(Value value) {
  if (auto unrealCast =
          dyn_cast_or_null<UnrealizedConversionCastOp>(value.getDefiningOp())) {
    return unrealCast.getOperand(0);
  }
  return value;
}

/// Get default memory alignment for a given tensor element type.
static int64_t getDefaultAlignment(Type tensorElementType) {
  if (auto intType = dyn_cast<IntegerType>(tensorElementType)) {
    return intType.getWidth() / 8;
  }
  if (auto floatType = dyn_cast<FloatType>(tensorElementType)) {
    return floatType.getWidth() / 8;
  }
  return 1;
}

/// Apply alignment assumption if needed
static Value applyAlignmentIfNeeded(ConversionPatternRewriter &rewriter,
                                    Location loc, Value ptr, Type elementType,
                                    int64_t requestedAlignment) {
  int64_t defaultAlignment = getDefaultAlignment(elementType);
  if (requestedAlignment <= defaultAlignment) {
    return ptr;
  }

  auto alignmentAttr =
      cuda_tile::DivByAttr::get(rewriter.getContext(), requestedAlignment,
                                /*every=*/std::nullopt, /*along=*/std::nullopt);

  return cuda_tile::AssumeOp::create(rewriter, loc, ptr, alignmentAttr);
}

static Value applyDivisibilityIfNeeded(OpBuilder &builder, Location loc,
                                       Value value, int64_t divisibility) {
  if (divisibility <= 1) {
    return value;
  }
  assert(divisibility > 0 && (divisibility & (divisibility - 1)) == 0 &&
         "cuda_tile.div_by requires power-of-two divisibility");

  auto divByAttr = cuda_tile::DivByAttr::get(builder.getContext(), divisibility,
                                             /*every=*/std::nullopt,
                                             /*along=*/std::nullopt);
  return cuda_tile::AssumeOp::create(builder, loc, value, divByAttr);
}

static Value applyNonNegativeBoundIfNeeded(OpBuilder &builder, Location loc,
                                           Value value) {
  auto tileType = dyn_cast<cuda_tile::TileType>(value.getType());
  if (!tileType || !isa<IntegerType>(tileType.getElementType())) {
    return value;
  }

  auto boundedAttr = cuda_tile::BoundedAttr::get(builder.getContext(), /*lb=*/0,
                                                 /*ub=*/std::nullopt);
  return cuda_tile::AssumeOp::create(builder, loc, value, boundedAttr);
}

/// Relay TensorIR stride descriptor facts to one dynamic CUDA Tile descriptor
/// value.
///
/// This is where TensorIR stride divisibility becomes cuda_tile.div_by, and
/// where descriptor values are marked non-negative before make_tensor_view uses
/// them. The second div_by is intentional: assume ops wrap their operand, so
/// the value consumed by make_tensor_view must carry the divisibility fact
/// after the bound assume.
static Value applyDescriptorValueAssumption(OpBuilder &builder, Location loc,
                                            Value value, int64_t divisibility) {
  value = applyDivisibilityIfNeeded(builder, loc, value, divisibility);
  value = applyNonNegativeBoundIfNeeded(builder, loc, value);
  return applyDivisibilityIfNeeded(builder, loc, value, divisibility);
}

static SmallVector<Value>
applyDescriptorBoundAssumptions(OpBuilder &builder, Location loc,
                                ArrayRef<Value> values) {
  SmallVector<Value> assumedValues;
  assumedValues.reserve(values.size());
  for (Value value : values) {
    assumedValues.push_back(applyNonNegativeBoundIfNeeded(builder, loc, value));
  }
  return assumedValues;
}

/// Apply non-negative runtime descriptor bounds and optional stride
/// divisibility facts.
static SmallVector<Value>
applyDescriptorValueAssumptions(OpBuilder &builder, Location loc,
                                ArrayRef<Value> values,
                                ArrayRef<int64_t> divisibilities) {
  assert(values.size() == divisibilities.size() &&
         "Value count must match divisibility metadata");
  SmallVector<Value> assumedValues;
  assumedValues.reserve(values.size());
  for (auto [value, divisibility] : llvm::zip_equal(values, divisibilities)) {
    assumedValues.push_back(
        applyDescriptorValueAssumption(builder, loc, value, divisibility));
  }
  return assumedValues;
}

/// Extract static sizes from a vector of tensor descriptor sizes.
static SmallVector<int64_t>
getStaticSizes(ArrayRef<TensorDescriptor::StaticOrDynamic> sizes) {
  SmallVector<int64_t> staticSizes;
  for (const auto &size : sizes) {
    staticSizes.push_back(size.staticValue);
  }
  return staticSizes;
}

static SmallVector<int64_t> getStoredDynamicDivisibilities(
    ArrayRef<TensorDescriptor::StaticOrDynamic> values) {
  SmallVector<int64_t> divisibilities;
  divisibilities.reserve(values.size());
  for (const auto &value : values) {
    divisibilities.push_back(value.divisibility);
  }
  return divisibilities;
}

static int64_t getStaticDivisibility(int64_t value) {
  if (value <= 0) {
    return 1;
  }
  return int64_t{1} << llvm::countr_zero(static_cast<uint64_t>(value));
}

/// Derive divisibility for implicit row-major strides from static size facts.
///
/// TensorIR dynamic shape divisibility was removed; only static dimensions can
/// prove factors in implicit row-major strides. Explicit strides do not use
/// this path; they relay their own TensorIR stride divisibility metadata from
/// `?{div=N}` in the stride layout string.
static SmallVector<int64_t>
computeRowMajorStrideDivisibility(ArrayRef<int64_t> staticStrides,
                                  ArrayRef<int64_t> staticSizes) {
  SmallVector<int64_t> strideDivisibility(staticStrides.size(), 1);
  int64_t runningDivisibility = 1;
  for (size_t reverseIdx = staticStrides.size(); reverseIdx > 0; --reverseIdx) {
    size_t idx = reverseIdx - 1;
    strideDivisibility[idx] = runningDivisibility;

    int64_t dimDivisibility =
        staticSizes[idx] == cuda_tile::TensorViewType::kDynamic
            ? 1
            : getStaticDivisibility(staticSizes[idx]);
    // Keep the metadata in the range normally useful for CUDA Tile lowering and
    // avoid overflow on high-rank tensor products.
    constexpr int64_t kMaxUsefulDivisibility = 1 << 30;
    if (runningDivisibility <= kMaxUsefulDivisibility / dimDivisibility) {
      runningDivisibility *= dimDivisibility;
    } else {
      runningDivisibility = kMaxUsefulDivisibility;
    }
  }
  return strideDivisibility;
}

/// Compute default row-major strides from shape.
/// TensorIR uses row-major as the default layout.
/// For row-major: stride[last] = 1, stride[i] = stride[i+1] * dim[i+1].
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);

  int64_t runningStride = 1;
  for (size_t reverseIdx = shape.size(); reverseIdx > 0; --reverseIdx) {
    size_t idx = reverseIdx - 1;
    strides[idx] = runningStride;

    // If this dimension is dynamic, all preceding strides depend on it.
    if (shape[idx] < 0) {
      for (size_t j = 0; j < idx; ++j) {
        strides[j] = cuda_tile::TensorViewType::kDynamic;
      }
      break;
    }
    runningStride *= shape[idx];
  }

  return strides;
}

/// Extract TensorIR descriptor metadata consumed by CUDA Tile tensor views.
///
/// This is the front door for relaying TensorIR descriptor facts into CUDA
/// Tile: explicit dynamic strides relay stored stride divisibility, and
/// implicit row-major strides derive any provable divisibility from static
/// later dimensions that form each stride.
static TensorDescriptorStaticMetadata
getTensorDescriptorStaticMetadata(const TensorDescriptor &descriptor) {
  TensorDescriptorStaticMetadata metadata;
  metadata.sizes = getStaticSizes(descriptor.sizes);
  metadata.hasExplicitStrides = !descriptor.strides.empty();
  metadata.strides = metadata.hasExplicitStrides
                         ? getStaticSizes(descriptor.strides)
                         : computeRowMajorStrides(metadata.sizes);
  metadata.strideDivisibility =
      metadata.hasExplicitStrides
          ? getStoredDynamicDivisibilities(descriptor.strides)
          : computeRowMajorStrideDivisibility(metadata.strides, metadata.sizes);
  return metadata;
}

static SmallVector<int64_t>
getTileSizesForIterSpaceMap(AffineMap iterSpaceMap,
                            ArrayRef<int32_t> iterSpaceTileSizes) {
  assert(
      iterSpaceMap.getNumInputs() == iterSpaceTileSizes.size() &&
      "Iteration space map dimensions do not match the number of tile sizes");

  SmallVector<int64_t> i64TileSizes = convertInt32ToI64(iterSpaceTileSizes);
  auto opTileSizes = iterSpaceMap.compose(i64TileSizes);

  LLVM_DEBUG({
    llvm::dbgs() << "Op tiles sizes from iteration space map: \n";
    llvm::dbgs() << llvm::indent(4) << "Map: " << iterSpaceMap << "\n";
    llvm::dbgs() << llvm::indent(4) << "Iteration space tile sizes: (";
    llvm::interleaveComma(i64TileSizes, llvm::dbgs());
    llvm::dbgs() << ")\n";
    llvm::dbgs() << llvm::indent(4) << "Op tile sizes: (";
    llvm::interleaveComma(opTileSizes, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  // Explicitly construct return type to avoid implicit conversion issues
  // between SmallVector<int64_t, 4> (from compose) and SmallVector<int64_t>
  // which fails on GCC 11.2+ due to stricter template type checking.
  return {opTileSizes.begin(), opTileSizes.end()};
}

//===----------------------------------------------------------------------===//
// Tensor To CUDA Tile Conversion State.
//===----------------------------------------------------------------------===//

/// TODO: This is just a sketch of how to map GPU virtual threads to the kernel
/// iteration space and it's subject to change significantly.
///
/// This class models how the kernel iteration space will be mapped and
/// distributed across the virtual threads (grid_dim × block_dim) from the grid
/// configuration.
///
/// For each iteration space dimension, the use one of the following values to
/// indicate that:
///   * kUndefinedVirtualThreadMapping: The mapping/thread distribution has not
///     been defined yet.
///   * kVirtualThreadsMatchIterSpaceDimSize: The iteration space dimension is
///     distributed across threads and the number of virtual threads matches the
///     iteration space dimension size, resulting in a 1:1 thread-to-element
///     mapping for that dimension.
///   * kUnmappedIterSpaceDim: This iteration space dimension is not distributed
///     across virtual threads (e.g., reduction dimension) so there is no direct
///     mapping between virtual threads and the iteration space dimension.
///   * Positive integer: The iteration space dimension is distributed across
///     the provided number of virtual threads from the grid configuration.
///     When the number of virtual threads is less than the iteration space
///     dimension size, each thread must process multiple elements of that
///     dimension inside the kernel.
///
struct ThreadToIterSpaceMapping {

  static constexpr int64_t kUndefinedVirtualThreadMapping = -1;
  static constexpr int64_t kVirtualThreadsMatchIterSpaceDimSize = -2;
  static constexpr int64_t kUnmappedIterSpaceDim = -3;

  ThreadToIterSpaceMapping() = default;
  ThreadToIterSpaceMapping(
      int numIterSpaceDims,
      int64_t defaultNumThreads = kUndefinedVirtualThreadMapping)
      : virtualThreadsToIterSpaceDimMapping(numIterSpaceDims,
                                            defaultNumThreads) {}

  /// Default configuration assumes no tiling is needed inside the kernel.
  static ThreadToIterSpaceMapping getNoTilingConfig(int numDims) {
    return ThreadToIterSpaceMapping(numDims,
                                    kVirtualThreadsMatchIterSpaceDimSize);
  }

  void setDimMapping(int64_t dimIdx, int64_t mapping) {
    virtualThreadsToIterSpaceDimMapping[dimIdx] = mapping;
  }

  bool dimNeedsTiling(int64_t dimIdx) const {
    assert(dimIdx >= 0 &&
           dimIdx < static_cast<int64_t>(
                        virtualThreadsToIterSpaceDimMapping.size()) &&
           "Dimension index is out of bounds");
    assert(virtualThreadsToIterSpaceDimMapping[dimIdx] !=
               kUndefinedVirtualThreadMapping &&
           "Dimension mapping is not initialized");
    return virtualThreadsToIterSpaceDimMapping[dimIdx] !=
           kVirtualThreadsMatchIterSpaceDimSize;
  }

  /// Check if any dimension requires tiling inside the kernel.
  bool anyDimensionNeedsTiling() const {
    return !llvm::all_of(virtualThreadsToIterSpaceDimMapping, [](int64_t size) {
      return size == kVirtualThreadsMatchIterSpaceDimSize;
    });
  }

  void print(llvm::raw_ostream &os) const {
    llvm::interleaveComma(
        virtualThreadsToIterSpaceDimMapping, os, [&](int64_t value) {
          if (value == kUndefinedVirtualThreadMapping) {
            os << "undefined";
          } else if (value == kVirtualThreadsMatchIterSpaceDimSize) {
            os << "match";
          } else if (value == kUnmappedIterSpaceDim) {
            os << "unmapped";
          } else {
            os << value;
          }
        });
  }

private:
  /// Mapping from virtual threads in a grid dimension to an iteration space
  /// dimension.
  SmallVector<int64_t> virtualThreadsToIterSpaceDimMapping;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ThreadToIterSpaceMapping &mapping) {
  mapping.print(os);
  return os;
}

/// The location of an iteration space dimension is defined by the tensor view
/// containing that dimension and an index to the dimension position within that
/// tensor view.
using IterSpaceDimLocation = std::pair<Value, int32_t>;

/// Holds information about a single iteration space, including dimension sizes,
/// tile counts, and tiling loops.
struct IterSpaceDimInfo {
  IterSpaceDimInfo() = default;
  explicit IterSpaceDimInfo(int64_t rank)
      : numTiles(rank, nullptr), tilerLoops(rank, nullptr),
        iterSpaceDimLocations(rank) {}

  /// Number of tiles for each iteration space dimension.
  SmallVector<Value> numTiles;

  /// Holds the loops for dimensions that need tiling. For dimensions that don't
  /// need tiling, the corresponding entry is nullptr operation. Some loops may
  /// be shared across iteration spaces when dimensions have the same size and
  /// are fused.
  SmallVector<cuda_tile::ForOp> tilerLoops;

  /// Iteration space dimension locations - maps each iteration space dimension
  /// to the tensor view and dimension index that defines its size.
  SmallVector<IterSpaceDimLocation> iterSpaceDimLocations;
};

/// Kernel-level information that is common across all iteration spaces.
/// Per-iteration-space info is stored in IterSpaceDimInfo.
struct KernelInfo {
  static constexpr int64_t kUnknownDimSize = -1;

  KernelInfo() = default;
  KernelInfo(int64_t iterSpaceRank)
      : iterSpaceRank(iterSpaceRank),
        tileSizes(iterSpaceRank, kUnknownDimSize) {}

  int64_t getIterationSpaceRank() const {
    assert(iterSpaceRank >= 0 && "Iteration space rank is not initialized");
    return iterSpaceRank;
  }

  /// The rank of the iteration space.
  /// TODO: For now, we assume all iteration spaces share the same rank. Support
  /// per-iteration-space ranks when needed.
  int64_t iterSpaceRank = -1;

  /// Holds whether tiling is available. We may have to bail out on tiling when
  /// there is not enough information in the graph to infer iteration space
  /// information (e.g., empty or passthrough graphs).
  bool isTilingAvailable = true;

  /// Tile sizes for each iteration space dimension.
  /// TODO: For now, we assume tile sizes are the same across all iteration
  /// spaces.
  SmallVector<int32_t> tileSizes;

  /// Use zero padding for input partition views. Matmul reduction tiles need
  /// zero-padding when K is not a multiple of the reduction tile size.
  bool useZeroPaddedInputLoads = false;
};

//===----------------------------------------------------------------------===//
// Persistence Strategy Interface
//===----------------------------------------------------------------------===//

class TensorToCudaTileConversionState;

/// Abstract strategy for persistence mode handling.
/// This pattern minimizes conditional code for different persistence modes.
class PersistenceStrategy {
public:
  virtual ~PersistenceStrategy() = default;

  /// Set up block ID operation and optionally set up persistence loop.
  /// Called during graph conversion before tile skeleton generation.
  /// Always creates the block ID op; for static persistence, also creates
  /// the persistence loop.
  virtual void
  setupBlockIdAndPersistence(OpBuilder &builder, Location loc,
                             TensorToCudaTileConversionState &state) = 0;

  /// Get the effective tile index for coordinate computation.
  virtual Value
  getTileIndex(const TensorToCudaTileConversionState &state) const = 0;

  /// Create the return operation. Called during results finalization.
  /// For persistent kernels, entryOp must be provided to set insertion point
  /// at the end of the entry block. For non-persistent kernels, entryOp can be
  /// null and the return is created at the current insertion point.
  virtual void createReturnOp(OpBuilder &builder, Location loc,
                              cuda_tile::EntryOp entryOp = nullptr) = 0;

  /// Check if persistence loop is active (for conditional logic that can't
  /// be abstracted).
  virtual bool isPersistenceActive() const = 0;
};

/// Default persistence strategy for "none" and "dynamic" persistence modes.
/// Uses block ID as the tile index with no loop wrapping.
class DefaultPersistenceStrategy : public PersistenceStrategy {
public:
  void
  setupBlockIdAndPersistence(OpBuilder &builder, Location loc,
                             TensorToCudaTileConversionState &state) override;

  Value
  getTileIndex(const TensorToCudaTileConversionState &state) const override;

  void createReturnOp(OpBuilder &builder, Location loc,
                      cuda_tile::EntryOp entryOp = nullptr) override;

  bool isPersistenceActive() const override { return false; }
};

/// Static persistence strategy for "static" persistence mode.
/// Creates a for loop over tiles, uses induction variable as tile index.
class StaticPersistenceStrategy : public PersistenceStrategy {
public:
  StaticPersistenceStrategy(int64_t gridSizeX, int64_t totalTiles,
                            int32_t smCount)
      : gridSizeX(gridSizeX), totalTiles(totalTiles), smCount(smCount) {}

  /// Set up block ID and the persistence for loop. Called during graph
  /// conversion.
  void
  setupBlockIdAndPersistence(OpBuilder &builder, Location loc,
                             TensorToCudaTileConversionState &state) override;

  Value
  getTileIndex(const TensorToCudaTileConversionState &state) const override;

  void createReturnOp(OpBuilder &builder, Location loc,
                      cuda_tile::EntryOp entryOp = nullptr) override;

  bool isPersistenceActive() const override { return true; }

private:
  int64_t gridSizeX;
  int64_t totalTiles;
  int32_t smCount;
  Value inductionVar = nullptr;
};

//===----------------------------------------------------------------------===//
// Conversion State
//===----------------------------------------------------------------------===//

/// State that tracks conversion information across the different Tensor To CUDA
/// Tile conversion patterns for a single graph. We should only store
/// information that is generally valid and available across the different
/// conversion patterns. Temporary information or information that is only valid
/// during the conversion of a specific pattern and becomes invalid immediately
/// after that conversion shouldn't be stored here.
class TensorToCudaTileConversionState {
public:
  /// Clear the conversion state. Should be called before converting each new
  /// graph.
  void clear();

  /// Get the input and output tensor descriptors.
  MutableArrayRef<CudaTileTensorDescriptor> getInOutTensorDescriptors();
  MutableArrayRef<CudaTileTensorDescriptor> getInputTensorDescriptors();
  MutableArrayRef<CudaTileTensorDescriptor> getOutputTensorDescriptors();

  /// Compute the tensor descriptors for all input and output tensors in the
  /// CUDA Tile function.
  LogicalResult computeCudaTileInOutTensorDescriptors(
      nv_tensor_ir::GraphOp graphOp, FunctionType origFuncType,
      Block *tensorIREntryBlock, cuda_tile::EntryOp entryOp,
      TypeConverter::SignatureConversion &signatureConversion,
      ArrayAttr tensorIRArgAttrs, ArrayAttr tensorIRResAttrs);

  /// Cache and reuse the GetTileBlockIdOp.
  cuda_tile::GetTileBlockIdOp getGetTileBlockIdOp() const;

  /// Cache GetTileBlockIdOp to avoid creating multiple instances
  void setGetTileBlockIdOp(cuda_tile::GetTileBlockIdOp op);

  /// Map a tensor view to its descriptor.
  void mapTensorViewToDescriptor(Value tensorView,
                                 CudaTileTensorDescriptor *descriptor);

  /// Get the descriptor for a tensor view.
  CudaTileTensorDescriptor &getTensorViewDescriptor(Value tensorView) const;

  /// Find the descriptor for a given input tensor value. The value can be
  /// either a Tensor IR tensor or a CUDA Tile pointer.
  const CudaTileTensorDescriptor *
  findDescriptorForInput(Value inputTensor) const;

  /// Get the tile value for an operand. If the original operand is an input
  /// tensor pointer, return the loaded tile from the descriptor. Otherwise,
  /// return the converted adaptor operand (stripping unrealized conversion
  /// casts if needed).
  Value getOperandTile(Value adaptorOperand) const;

  /// Iteration space utilities.
  AffineMap getIterSpaceMapForValue(Value value) const;
  int32_t getIterSpaceIdForValue(Value value) const;
  AffineMap getOutputIterSpaceMapForOperation(Operation *op) const;
  AffineMap getOpUnionIterSpaceMap(Operation *op) const;
  OpBuilder::InsertPoint getInsertPointForMap(int32_t iterSpaceId,
                                              AffineMap iterSpaceMap) const;
  OpBuilder::InsertPoint getInsertPointForValue(Value dependentValue) const;
  OpBuilder::InsertPoint getInsertPointForOp(Operation *op) const;

  /// Record the insertion point for a specific iteration space dimension.
  /// This is used to track where operations should be inserted based on which
  /// iteration space and dimension they depend on (e.g., inside a tiling loop).
  void setInsertPointForIterSpaceDim(int32_t iterSpaceId,
                                     int64_t iterSpaceDimIdx,
                                     OpBuilder::InsertPoint insertPoint);
  OpBuilder::InsertPoint
  getInsertPointForIterSpaceDim(int32_t iterSpaceId,
                                int64_t iterSpaceDimIdx) const;

  /// Get the list of iteration space IDs in the graph (sorted).
  ArrayRef<int32_t> getIterSpaceIds() const { return iterSpaceIds; }

  /// Set the list of iteration space IDs in the graph.
  void setIterSpaceIds(ArrayRef<int32_t> ids) {
    iterSpaceIds.assign(ids.begin(), ids.end());
  }

  /// Get the per-iteration-space dimension info for a given iteration space ID.
  /// Creates a new entry if it doesn't exist.
  IterSpaceDimInfo &getOrCreateIterSpaceDimInfo(int32_t iterSpaceId) {
    auto it = iterSpaceDimInfos.find(iterSpaceId);
    if (it == iterSpaceDimInfos.end()) {
      int64_t rank = kernelInfo.getIterationSpaceRank();
      it = iterSpaceDimInfos.try_emplace(iterSpaceId, rank).first;
    }
    return it->second;
  }

  /// Get the per-iteration-space dimension info for a given iteration space ID.
  /// Asserts if the entry doesn't exist.
  IterSpaceDimInfo &getIterSpaceDimInfo(int32_t iterSpaceId) {
    auto it = iterSpaceDimInfos.find(iterSpaceId);
    assert(it != iterSpaceDimInfos.end() &&
           "IterSpaceDimInfo not found for iteration space ID");
    return it->second;
  }

  /// Get the effective tile index for coordinate calculation.
  /// For non-persistent kernels, this returns the block ID (program ID).
  /// For persistent kernels, this returns the persistence loop's induction
  /// variable.
  Value getEffectiveTileIndex() const;

  /// Holds the mapping from virtual threads to the elements of the iteration
  /// space.
  ThreadToIterSpaceMapping threadToIterSpaceMapping;

  /// Holds kernel-level information that is common across all iteration spaces.
  KernelInfo kernelInfo;

  /// Per-iteration-space dimension information. Each iteration space may have
  /// different dimension sizes, tile counts, and tiling loops.
  DenseMap<int32_t, IterSpaceDimInfo> iterSpaceDimInfos;

  /// Number of CTAs per cluster (CGA). Default is 1.
  int32_t numCTAs = 1;

  /// Target occupancy (CTAs per SM). Default is 1.
  int32_t occupancy = 1;

  /// Number of warps per CTA. Default is 4.
  int32_t numWarps = 4;

  /// User-specified tile sizes. Empty means use auto-computed tile sizes.
  SmallVector<int32_t> tileSize;

  /// Persistence mode for the kernel.
  PersistenceMode persistence = PersistenceMode::None;

  /// When true, the kernel signature includes args for ALL tensor sizes and
  /// strides.  When false, only dynamic dims produce args (original behavior).
  bool uniformSignature = false;

  /// Grid size X dimension for static persistent kernels.
  int64_t gridSizeX = 0;

  /// Total number of tiles for static persistent kernels.
  int64_t totalTiles = 0;

  /// SM count for static persistent kernels.
  int32_t smCount = 0;

  /// Persistence strategy for handling different persistence modes.
  /// Encapsulates tile index computation and loop setup.
  std::unique_ptr<PersistenceStrategy> persistenceStrategy;

private:
  SmallVector<CudaTileTensorDescriptor, 8> inOutTensorDescriptors;
  size_t numInputDescriptors = 0;

  cuda_tile::GetTileBlockIdOp getTileBlockIdOp;

  // Mapping from tensor view Values to their descriptors.
  DenseMap<Value, CudaTileTensorDescriptor *> tensorViewToDescriptorMap;

  // Mapping from (iter_space_id, dim_idx) to insertion points.
  DenseMap<std::pair<int32_t, int64_t>, OpBuilder::InsertPoint>
      iterSpaceDimInsertPoints;

  // Sorted list of iteration space IDs in the graph.
  SmallVector<int32_t> iterSpaceIds;
};

void TensorToCudaTileConversionState::clear() {
  inOutTensorDescriptors.clear();
  numInputDescriptors = 0;
  getTileBlockIdOp = nullptr;
  tensorViewToDescriptorMap.clear();
  threadToIterSpaceMapping = ThreadToIterSpaceMapping();
  kernelInfo = KernelInfo();
  iterSpaceDimInfos.clear();
  iterSpaceDimInsertPoints.clear();
  persistenceStrategy.reset();
  iterSpaceIds.clear();
}

MutableArrayRef<CudaTileTensorDescriptor>
TensorToCudaTileConversionState::getInOutTensorDescriptors() {
  return inOutTensorDescriptors;
}

MutableArrayRef<CudaTileTensorDescriptor>
TensorToCudaTileConversionState::getInputTensorDescriptors() {
  return MutableArrayRef<CudaTileTensorDescriptor>(inOutTensorDescriptors)
      .take_front(numInputDescriptors);
}

MutableArrayRef<CudaTileTensorDescriptor>
TensorToCudaTileConversionState::getOutputTensorDescriptors() {
  return MutableArrayRef<CudaTileTensorDescriptor>(inOutTensorDescriptors)
      .drop_front(numInputDescriptors);
}

LogicalResult
TensorToCudaTileConversionState::computeCudaTileInOutTensorDescriptors(
    nv_tensor_ir::GraphOp graphOp, FunctionType origFuncType,
    Block *tensorIREntryBlock, cuda_tile::EntryOp entryOp,
    TypeConverter::SignatureConversion &signatureConversion,
    ArrayAttr tensorIRArgAttrs, ArrayAttr tensorIRResAttrs) {

  auto cudaTileArgs = entryOp.getArguments();
  size_t numInputs = origFuncType.getNumInputs();
  size_t numOutputs = origFuncType.getNumResults();

  // This reserve is critical to avoid reallocations during the conversion and
  // invalidating the references to the tensor descriptors.
  inOutTensorDescriptors.reserve(numInputs + numOutputs);

  LLVM_DEBUG(llvm::dbgs() << "Computing CUDA Tile tensor descriptors for "
                          << entryOp.getName() << "\n");

  //===--------------------------------------------------------------------===//
  // Compute input tensor descriptors.
  //===--------------------------------------------------------------------===//

  auto argumentDescriptors =
      getTensorDescriptors(graphOp.getArgumentTypes(), tensorIRArgAttrs);
  if (failed(argumentDescriptors)) {
    return graphOp.emitError(
        "failed to get tensor descriptors for input tensors");
  }

  // Get the original Tensor IR block arguments.
  auto tensorIRArgs = tensorIREntryBlock->getArguments();
  assert(tensorIRArgs.size() == argumentDescriptors->size() &&
         "Number of graph arguments must match number of tensor descriptors");

  size_t lastArgEnd = 0;
  for (auto [tensorIdx, argType] : llvm::enumerate(origFuncType.getInputs())) {
    auto mapping = signatureConversion.getInputMapping(tensorIdx);
    assert(mapping && "No mapping found for tensor IR argument");

    // Non-tensor inputs (e.g. scalar uniform arguments) still consume converted
    // function argument slots even though they do not create tensor
    // descriptors. Keep the output tensor argument cursor after every input.
    size_t mappingEnd = mapping->inputNo + mapping->size;
    if (lastArgEnd < mappingEnd) {
      lastArgEnd = mappingEnd;
    }

    auto tensorTy = dyn_cast<nv_tensor_ir::TensorType>(argType);
    if (!tensorTy) {
      continue;
    }

    // Get iteration space map and iteration space id list from the dictionary
    // attribute set on the tensor IR argument.
    AffineMap iterSpaceMap;
    ArrayRef<int32_t> iterSpaceIds;
    if (tensorIRArgAttrs) {
      if (auto argAttrDict =
              dyn_cast<DictionaryAttr>(tensorIRArgAttrs[tensorIdx])) {
        if (auto iterSpaceMapAttr = argAttrDict.getAs<AffineMapAttr>(
                TensorIRDialect::getIterSpaceMapAttrName())) {
          iterSpaceMap = iterSpaceMapAttr.getAffineMap();
        }
        if (auto idsAttr = argAttrDict.getAs<DenseI32ArrayAttr>(
                TensorIRDialect::getIterSpaceIdsAttrName())) {
          iterSpaceIds = idsAttr.asArrayRef();
        }
      }
    }
    if (!iterSpaceMap) {
      return graphOp.emitError(
          "Iteration space map not found for affine map codegen. Please run "
          "\"discover-iteration-space-info\" pass to discover the iteration "
          "space map.");
    }

    // Get the original Tensor IR argument.
    Value tensorIRArg = tensorIRArgs[tensorIdx];

    const TensorDescriptor &descriptor = (*argumentDescriptors)[tensorIdx];
    TensorDescriptorStaticMetadata metadata =
        getTensorDescriptorStaticMetadata(descriptor);

    size_t ptrIdx = mapping->inputNo;
    size_t numArgsConsumed = processTensorDescriptor(
        tensorTy, tensorIRArg, cudaTileArgs, ptrIdx, metadata,
        CudaTileTensorDescriptor::TensorKind::Input, iterSpaceMap, iterSpaceIds,
        descriptor.allowTma, descriptor.alignment, descriptor.cost,
        inOutTensorDescriptors, uniformSignature);

    assert(mappingEnd == ptrIdx + numArgsConsumed &&
           "Mapping size mismatch: expected tensor to expand to "
           "ptr + dynamic_sizes + dynamic_strides");
  }

  numInputDescriptors = inOutTensorDescriptors.size();

  //===--------------------------------------------------------------------===//
  // Compute output tensor descriptors.
  //===--------------------------------------------------------------------===//

  auto resultDescriptors =
      getTensorDescriptors(graphOp.getResultTypes(), tensorIRResAttrs);
  if (failed(resultDescriptors)) {
    return graphOp.emitError(
        "failed to get tensor descriptors for output tensors");
  }

  // Get the result values from the results operation.
  auto resultsOp =
      cast<nv_tensor_ir::ResultsOp>(entryOp.getBody().back().back());
  ValueRange returnValues = resultsOp.getOperands();
  assert(returnValues.size() == origFuncType.getNumResults() &&
         "Graph return value count mismatch");
  assert(returnValues.size() == resultDescriptors->size() &&
         "Number of graph results must match number of tensor descriptors");

  size_t outputArgIdx = lastArgEnd;
  for (auto [tensorIdx, returnValue] : llvm::enumerate(returnValues)) {
    auto tensorTy = dyn_cast<nv_tensor_ir::TensorType>(returnValue.getType());
    if (!tensorTy) {
      continue;
    }

    // Retrieve the iteration space map from the graph output.
    AffineMap iterSpaceMap = getIterSpaceMapForValue(returnValue);
    int32_t iterSpaceId = getIterSpaceIdForValue(returnValue);
    assert(iterSpaceMap && "Iteration space map not found for input tensor");

    const TensorDescriptor &descriptor = (*resultDescriptors)[tensorIdx];
    TensorDescriptorStaticMetadata metadata =
        getTensorDescriptorStaticMetadata(descriptor);

    size_t ptrIdx = outputArgIdx;
    size_t numArgsConsumed = processTensorDescriptor(
        tensorTy, returnValue, cudaTileArgs, ptrIdx, metadata,
        CudaTileTensorDescriptor::TensorKind::Output, iterSpaceMap,
        {iterSpaceId}, descriptor.allowTma, descriptor.alignment,
        descriptor.cost, inOutTensorDescriptors, uniformSignature);

    outputArgIdx = ptrIdx + numArgsConsumed;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "CUDA Tile input/output tensor descriptors:\n";
    constexpr int indSize = 4;
    for (auto &descriptor : inOutTensorDescriptors) {
      llvm::dbgs() << llvm::indent(indSize)
                   << "Descriptor: " << descriptor.getCudaTilePtr() << "\n";
      llvm::dbgs() << llvm::indent(indSize) << "Tensor kind: "
                   << (descriptor.isInput() ? "Input" : "Output") << "\n";
      llvm::dbgs() << llvm::indent(indSize) << "Static sizes: [";
      llvm::interleaveComma(descriptor.getStaticSizes(), llvm::dbgs());
      llvm::dbgs() << "]\n";
      llvm::dbgs() << llvm::indent(indSize) << "Dynamic sizes: [";
      llvm::interleaveComma(descriptor.getDynamicSizes(), llvm::dbgs());
      llvm::dbgs() << "]\n";
      llvm::dbgs() << llvm::indent(indSize) << "Static strides: [";
      llvm::interleaveComma(descriptor.getStaticStrides(), llvm::dbgs());
      llvm::dbgs() << "]\n";
      if (descriptor.isInput()) {
        llvm::dbgs() << llvm::indent(indSize)
                     << "Iteration space map: " << descriptor.getIterSpaceMap()
                     << "\n";
      }
      llvm::dbgs() << llvm::indent(indSize) << "Dynamic strides: [";
      llvm::interleaveComma(descriptor.getDynamicStrides(), llvm::dbgs());
      llvm::dbgs() << "]\n";
    }
  });

  return success();
}

cuda_tile::GetTileBlockIdOp
TensorToCudaTileConversionState::getGetTileBlockIdOp() const {
  assert(getTileBlockIdOp && "GetTileBlockIdOp has not been created");
  return getTileBlockIdOp;
}

void TensorToCudaTileConversionState::setGetTileBlockIdOp(
    cuda_tile::GetTileBlockIdOp op) {
  assert(op && "GetTileBlockIdOp is null");
  assert(!getTileBlockIdOp && "GetTileBlockIdOp has already been created");
  getTileBlockIdOp = op;
}

void TensorToCudaTileConversionState::mapTensorViewToDescriptor(
    Value tensorView, CudaTileTensorDescriptor *descriptor) {
  assert(isa<cuda_tile::MakeTensorViewOp>(tensorView.getDefiningOp()) &&
         "Tensor view must be a cuda_tile.make_tensor_view op");
  assert(tensorView && "Tensor view must not be null");
  assert(descriptor && "Descriptor must not be null");
  assert(tensorViewToDescriptorMap.count(tensorView) == 0 &&
         "Tensor view already mapped to a descriptor");
  tensorViewToDescriptorMap[tensorView] = descriptor;
}

CudaTileTensorDescriptor &
TensorToCudaTileConversionState::getTensorViewDescriptor(
    Value tensorView) const {
  assert(tensorViewToDescriptorMap.count(tensorView) > 0 &&
         "Tensor view not mapped to a descriptor");
  return *tensorViewToDescriptorMap.lookup(tensorView);
}

const CudaTileTensorDescriptor *
// TODO: Consider using a map for O(1) lookup instead of linear search.
TensorToCudaTileConversionState::findDescriptorForInput(
    Value inputTensor) const {
  // Check if this is a Tensor IR value or a CUDA Tile value based on the type.
  bool isTensorIRValue = isa<RankedTensorType>(inputTensor.getType());

  for (const auto &descriptor : inOutTensorDescriptors) {
    if (isTensorIRValue) {
      if (descriptor.getTensorIRTensor() == inputTensor) {
        return &descriptor;
      }
    } else {
      if (descriptor.getCudaTilePtr() == inputTensor) {
        return &descriptor;
      }
    }
  }
  return nullptr;
}

Value TensorToCudaTileConversionState::getOperandTile(
    Value adaptorOperand) const {
  Value operand = stripUnrealizedConversionCast(adaptorOperand);
  if (isa<BlockArgument>(operand)) {
    if (auto *descriptor = findDescriptorForInput(operand)) {
      return descriptor->getLoadedTile();
    }
  }
  return operand;
}

void TensorToCudaTileConversionState::setInsertPointForIterSpaceDim(
    int32_t iterSpaceId, int64_t iterSpaceDimIdx,
    OpBuilder::InsertPoint insertPoint) {
  auto key = std::make_pair(iterSpaceId, iterSpaceDimIdx);
  assert(iterSpaceDimInsertPoints.count(key) == 0 &&
         "Insertion point already recorded for iteration space and dimension");
  iterSpaceDimInsertPoints[key] = insertPoint;
}

OpBuilder::InsertPoint
TensorToCudaTileConversionState::getInsertPointForIterSpaceDim(
    int32_t iterSpaceId, int64_t iterSpaceDimIdx) const {
  auto key = std::make_pair(iterSpaceId, iterSpaceDimIdx);
  auto it = iterSpaceDimInsertPoints.find(key);
  assert(it != iterSpaceDimInsertPoints.end() &&
         "Insertion point not recorded for iteration space and dimension");
  return it->second;
}

Value TensorToCudaTileConversionState::getEffectiveTileIndex() const {
  assert(persistenceStrategy && "PersistenceStrategy has not been set");
  return persistenceStrategy->getTileIndex(*this);
}

//===----------------------------------------------------------------------===//
// Persistence Strategy Implementations
//===----------------------------------------------------------------------===//

void DefaultPersistenceStrategy::setupBlockIdAndPersistence(
    OpBuilder &builder, Location loc, TensorToCudaTileConversionState &state) {
  // For default persistence, just create the block ID op (no persistence loop)
  auto blockIdOp = cuda_tile::GetTileBlockIdOp::create(builder, loc);
  state.setGetTileBlockIdOp(blockIdOp);
}

Value DefaultPersistenceStrategy::getTileIndex(
    const TensorToCudaTileConversionState &state) const {
  auto blockIdOp = state.getGetTileBlockIdOp();
  assert(blockIdOp && "GetTileBlockIdOp has not been created");
  auto blockIds = blockIdOp->getResults();
  assert(!blockIds.empty() && "No block IDs available");
  return blockIds[0];
}

void DefaultPersistenceStrategy::createReturnOp(OpBuilder &builder,
                                                Location loc,
                                                cuda_tile::EntryOp entryOp) {
  // For non-persistent kernels, create return at current insertion point
  (void)entryOp; // Unused for default persistence
  cuda_tile::ReturnOp::create(builder, loc);
}

void StaticPersistenceStrategy::setupBlockIdAndPersistence(
    OpBuilder &builder, Location loc, TensorToCudaTileConversionState &state) {
  MLIRContext *ctx = builder.getContext();

  // Get block IDs first - needed as the loop's starting value
  auto blockIdOp = cuda_tile::GetTileBlockIdOp::create(builder, loc);
  state.setGetTileBlockIdOp(blockIdOp);
  Value programIdX = blockIdOp->getResults()[0];

  // Determine type based on tile count (use i64 if > INT32_MAX)
  bool needsI64 = (totalTiles > INT32_MAX) || (gridSizeX > INT32_MAX);
  Type indexType =
      needsI64 ? IntegerType::get(ctx, 64) : IntegerType::get(ctx, 32);
  auto tileIndexType =
      cuda_tile::TileType::get(llvm::ArrayRef<int64_t>{}, indexType);

  // Create loop bounds as cuda_tile constants
  // For start value, we need to extend programIdX to i64 if needed
  Value startValue = programIdX;
  if (needsI64) {
    // Extend i32 programId to i64 (unsigned since block IDs are unsigned)
    auto extOp =
        cuda_tile::ExtIOp::create(builder, loc, tileIndexType, programIdX,
                                  cuda_tile::Signedness::Unsigned);
    startValue = extOp.getResult();
  }

  // Create end value (totalTiles) and step value (gridSizeX)
  auto endAttr = cast<DenseTypedElementsAttr>(DenseElementsAttr::get(
      tileIndexType, builder.getIntegerAttr(indexType, totalTiles)));
  auto endValue =
      cuda_tile::ConstantOp::create(builder, loc, tileIndexType, endAttr);

  auto stepAttr = cast<DenseTypedElementsAttr>(DenseElementsAttr::get(
      tileIndexType, builder.getIntegerAttr(indexType, gridSizeX)));
  auto stepValue =
      cuda_tile::ConstantOp::create(builder, loc, tileIndexType, stepAttr);

  // Create the persistence for loop
  auto forOp = cuda_tile::ForOp::create(
      builder, loc, startValue, endValue.getResult(), stepValue.getResult());
  inductionVar = forOp.getInductionVar();

  // Set insertion point inside the loop body
  builder.setInsertionPointToStart(forOp.getBody());

  LLVM_DEBUG({
    llvm::dbgs() << "Generated static persistence loop:\n";
    llvm::dbgs() << "  totalTiles: " << totalTiles << "\n";
    llvm::dbgs() << "  gridSizeX: " << gridSizeX << "\n";
    llvm::dbgs() << "  indexType: " << indexType << "\n";
  });
}

Value StaticPersistenceStrategy::getTileIndex(
    const TensorToCudaTileConversionState &state) const {
  assert(inductionVar && "Persistence loop has not been set up");
  return inductionVar;
}

void StaticPersistenceStrategy::createReturnOp(OpBuilder &builder, Location loc,
                                               cuda_tile::EntryOp entryOp) {
  // For persistent kernels, create return at the end of the entry block
  // (after the persistence loop)
  assert(entryOp && "EntryOp must be provided for persistent kernels");
  builder.setInsertionPointToEnd(&entryOp.getBlocks().front());
  cuda_tile::ReturnOp::create(builder, loc);
}

//===----------------------------------------------------------------------===//
// Tensor to CUDA Tile Type Conversion.
//===----------------------------------------------------------------------===//

/// Convert function argument types. Tensors are expanded into non-aggregated
/// descriptor arguments: pointer, dimension sizes and strides.
/// @param numDynamicStrides Number of dynamic strides from stride attribute.
///                          If nullopt, uses shape dynamicness as fallback.
LogicalResult functionArgumentConverter(Type argType,
                                        const TypeConverter &typeConverter,
                                        SmallVectorImpl<Type> &result,
                                        std::optional<int> numDynamicStrides,
                                        bool uniformSignature) {

  // Convert input type to CUDA Tile type.
  auto converted =
      dyn_cast_or_null<cuda_tile::TileType>(typeConverter.convertType(argType));
  if (!converted) {
    return failure();
  }

  // Handle non-tensor type (e.g. a scalar).
  auto tensorTy = dyn_cast<nv_tensor_ir::TensorType>(argType);
  if (!tensorTy) {
    result.push_back(converted);
    return success();
  }

  // Tensor IR tensor type: expand into descriptor arguments.
  MLIRContext *ctx = tensorTy.getContext();

  // Convert pointer type.
  auto ptrType = cuda_tile::PointerType::get(converted.getElementType());
  auto ptrTileTy =
      cuda_tile::TileType::get(/*shape=*/llvm::ArrayRef<int64_t>{}, ptrType);
  result.push_back(ptrTileTy);

  int numSizeArgs = uniformSignature
                        ? tensorTy.getRank()
                        : countDynamicDimensions(tensorTy.getShape());

  int strideCount =
      numDynamicStrides.has_value() ? *numDynamicStrides : numSizeArgs;

  // Convert dimensions and strides.
  auto sizeAndStrideTy = cuda_tile::TileType::get(
      /*shape=*/llvm::ArrayRef<int64_t>{}, IntegerType::get(ctx, 32));
  result.append(numSizeArgs + strideCount, sizeAndStrideTy);

  LLVM_DEBUG({
    llvm::dbgs() << "DEBUG: functionArgumentConverter for tensor type: "
                 << tensorTy << "\n";
    llvm::dbgs() << "DEBUG: numSizeArgs: " << numSizeArgs << "\n";
    llvm::dbgs() << "DEBUG: numStrideArgs: " << strideCount << "\n";
    llvm::dbgs() << "DEBUG: Total converted arguments: " << result.size()
                 << "\n";
  });
  return success();
}

/// Convert GraphOp function signature with stride awareness.
/// This function uses stride information from the graph to correctly count
/// dynamic strides independently from dynamic shape dimensions.
/// When hasExplicitStride is false, strides are computed from sizes at runtime
/// (row-major default), so no stride args are added to the signature.
Type convertGraphOpSignature(FunctionType funcTy,
                             const TypeConverter &typeConverter,
                             ArrayRef<TensorDescriptor> argDescriptors,
                             ArrayRef<TensorDescriptor> resDescriptors,
                             bool uniformSignature,
                             TypeConverter::SignatureConversion &result) {

  LLVM_DEBUG(llvm::dbgs() << "Converting GraphOp signature with "
                          << funcTy.getNumInputs() << " inputs and "
                          << funcTy.getNumResults() << " results\n");

  // Convert input argument types with stride awareness.
  for (auto [idx, argType] : llvm::enumerate(funcTy.getInputs())) {
    SmallVector<Type, 8> converted;

    int numStrideArgs = 0;
    const TensorDescriptor &descriptor = argDescriptors[idx];
    if (!descriptor.strides.empty()) {
      if (uniformSignature) {
        numStrideArgs = descriptor.strides.size();
      } else {
        numStrideArgs =
            llvm::count_if(descriptor.strides, [](const auto &stride) {
              return stride.staticValue == ShapedType::kDynamic;
            });
      }
    }

    if (failed(functionArgumentConverter(argType, typeConverter, converted,
                                         numStrideArgs, uniformSignature))) {
      return {};
    }

    result.addInputs(idx, converted);
  }

  // Move graph outputs to the end of the argument list with stride awareness.
  for (auto [idx, resType] : llvm::enumerate(funcTy.getResults())) {
    SmallVector<Type, 8> converted;

    int numStrideArgs = 0;
    const TensorDescriptor &descriptor = resDescriptors[idx];
    if (!descriptor.strides.empty()) {
      if (uniformSignature) {
        numStrideArgs = descriptor.strides.size();
      } else {
        numStrideArgs =
            llvm::count_if(descriptor.strides, [](const auto &stride) {
              return stride.staticValue == ShapedType::kDynamic;
            });
      }
    }

    if (failed(functionArgumentConverter(resType, typeConverter, converted,
                                         numStrideArgs, uniformSignature))) {
      return {};
    }

    result.addInputs(converted);
  }

  return FunctionType::get(funcTy.getContext(), result.getConvertedTypes(),
                           /*resultTypes=*/{});
}

//===----------------------------------------------------------------------===//
// Conversion Patterns.
//===----------------------------------------------------------------------===//

/// Convert linear program ID to multi-dimensional block coordinates
///
/// This implements the inverse of linearization that works regardless of memory
/// layout. If linear_id = coord[n-1] + coord[n-2]*blocks[n-1] + ... +
/// coord[0]*blocks[n-1]*...*blocks[1] Then we extract coordinates by iterating
/// in reverse:
/// - coord[n-1] = linear_id % blocks[n-1]
/// - coord[n-2] = (linear_id / blocks[n-1]) % blocks[n-2]
/// - etc.
///
/// For dimensions that have been tiled within the kernel, the tiler loop's
/// induction variable is used accordingly.
static SmallVector<Value> unwrapLinearCoordinates(
    Location loc, ConversionPatternRewriter &builder, Value linearProgramId,
    ArrayRef<Value> blocksPerDimension, ArrayRef<Value> inductionVars) {
  size_t numDimensions = blocksPerDimension.size();
  assert(inductionVars.size() == numDimensions &&
         "Induction vars size must match blocks per dimension size");

  // Guard against empty rank (would cause UB in loop below)
  // Empty result is valid for scalar tensors (rank 0) - no indexing needed
  if (numDimensions == 0) {
    return {};
  }

  SmallVector<Value> blockCoordinates(numDimensions);
  Value remainingId = linearProgramId;

  auto unsignedAttr = cuda_tile::SignednessAttr::get(
      builder.getContext(), cuda_tile::Signedness::Unsigned);

  // Extract coordinates by iterating from last dimension to first
  // This works regardless of memory layout (layout-agnostic algorithm)
  // Use ssize_t to avoid unsigned→signed UB and handle large ranks safely
  for (ssize_t dimensionIndex = static_cast<ssize_t>(numDimensions) - 1;
       dimensionIndex >= 0; --dimensionIndex) {
    Value blocksInThisDimension = blocksPerDimension[dimensionIndex];

    // If this dimension has a loop induction variable, use it directly.
    // The induction variable iterates over the tiles that each block should
    // compute within the kernel.
    if (inductionVars[dimensionIndex]) {
      blockCoordinates[dimensionIndex] = inductionVars[dimensionIndex];
    } else {
      // Extract coordinate for this dimension: remainingId %
      // blocksInThisDimension
      blockCoordinates[dimensionIndex] = cuda_tile::RemIOp::create(
          builder, loc, remainingId, blocksInThisDimension, unsignedAttr);
    }

    // Prepare for next iteration: remainingId = remainingId /
    // blocksInThisDimension
    if (dimensionIndex > 0) {
      remainingId = cuda_tile::DivIOp::create(
          builder, loc, remainingId, blocksInThisDimension, unsignedAttr);
    }
  }

  return blockCoordinates;
}

/// Calculate block indices from program ID and tiler loop's induction variable
/// using layout-agnostic coordinate mapping.
FailureOr<SmallVector<Value>>
calculateBlockIndices(Location loc, ConversionPatternRewriter &builder,
                      Value pid, Value partitionView,
                      const CudaTileTensorDescriptor &tensorDescr,
                      TensorToCudaTileConversionState &conversionState) {
  if (!pid) {
    return emitError(loc) << "Invalid program ID for coordinate calculation";
  }

  AffineMap iterSpaceMap = tensorDescr.getIterSpaceMap();
  const auto &kernelInfo = conversionState.kernelInfo;
  int64_t iterSpaceRank = kernelInfo.getIterationSpaceRank();

  SmallVector<Value> blockCoordinates;

  if (kernelInfo.isTilingAvailable) {
    // Get the per-iteration-space dimension info for this tensor's iteration
    // space.
    int32_t iterSpaceId = tensorDescr.getIterSpaceId();
    IterSpaceDimInfo &dimInfo =
        conversionState.getIterSpaceDimInfo(iterSpaceId);

    // Step 1: Decode grid coordinates at the iteration space level.
    // The grid is defined over iteration space dimensions that do NOT have
    // tiler loops (i.e., grid dimensions). For example, in matmul with
    // iteration space [M, N, K] where K is tiled, the grid is M×N blocks.
    SmallVector<Value> gridDimNumTiles;

    for (int64_t i = 0; i < iterSpaceRank; ++i) {
      // Only include dimensions without tiler loops in the grid
      if (!dimInfo.tilerLoops[i]) {
        gridDimNumTiles.push_back(dimInfo.numTiles[i]);
      }
    }

    SmallVector<Value> gridInductionVars(gridDimNumTiles.size());
    SmallVector<Value> gridCoordinates = unwrapLinearCoordinates(
        loc, builder, pid, gridDimNumTiles, gridInductionVars);

    // Step 2: Construct full iteration space coordinates by combining
    // grid coordinates and tiler loop induction variables.
    SmallVector<Value> iterSpaceCoordinates(iterSpaceRank);
    size_t gridIdx = 0;

    for (int64_t i = 0; i < iterSpaceRank; ++i) {
      if (auto tilerLoop = dimInfo.tilerLoops[i]) {
        // Dimension with tiler loop: use the induction variable
        iterSpaceCoordinates[i] = tilerLoop.getInductionVar();
      } else {
        // Grid dimension: use decoded grid coordinate
        iterSpaceCoordinates[i] = gridCoordinates[gridIdx++];
      }
    }

    // Step 3: Map iteration space coordinates to tensor dimensions
    // using the tensor's iteration space map.
    blockCoordinates =
        applyPermutationMap(iterSpaceMap, ArrayRef(iterSpaceCoordinates));
  } else {
    // Tiling is not available, probably because this is an empty or passthrough
    // graph. Explicitly generate GetIndexSpaceShapeOp for these indices.
    auto partViewType =
        cast<cuda_tile::PartitionViewType>(partitionView.getType());
    auto tileType = cast<cuda_tile::TileType>(pid.getType());
    SmallVector<Type> resultTypes(partViewType.getViewIndexRank(), tileType);
    auto indexSpaceShapeOp = cuda_tile::GetIndexSpaceShapeOp::create(
        builder, loc, resultTypes, partitionView);
    SmallVector<Value> numTilesPerTensorDim(
        indexSpaceShapeOp->getResults().begin(),
        indexSpaceShapeOp->getResults().end());
    SmallVector<Value> emptyInductionVars(numTilesPerTensorDim.size());
    blockCoordinates = unwrapLinearCoordinates(
        loc, builder, pid, numTilesPerTensorDim, emptyInductionVars);
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Block coordinate calculation successful:\n";
    llvm::dbgs() << "  Input program ID: " << pid << "\n";
    llvm::dbgs() << "  Block coordinates: [";
    llvm::interleaveComma(blockCoordinates, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  return blockCoordinates;
}

/// Base class for all conversion patterns that require a conversion state.
template <typename OpTy>
class TensorToCudaTileConversionPattern : public OpConversionPattern<OpTy> {
  using Base = OpConversionPattern<OpTy>;

public:
  TensorToCudaTileConversionPattern(
      TensorToCudaTileConversionState &conversionState,
      const TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<OpTy>(typeConverter, ctx),
        conversionState(conversionState) {}

protected:
  TensorToCudaTileConversionState &conversionState;
};

enum class PartitionViewPadding { None, Zero };

static cuda_tile::MakePartitionViewOp
buildPartitionView(ConversionPatternRewriter &rewriter,
                   const CudaTileTensorDescriptor &tensorDescr,
                   TensorToCudaTileConversionState &conversionState,
                   PartitionViewPadding padding = PartitionViewPadding::None);

/// Compute row-major dynamic strides from sizes.
/// For row-major: stride[last] = 1, stride[i] = product(size[i+1..last]).
/// @param staticStrides Static strides with kDynamic markers
/// @param staticSizes Static sizes with kDynamic markers
/// @param dynamicSizes Dynamic size values in order
/// @return Dynamic stride values for each kDynamic in staticStrides,
///         in left-to-right (increasing index) order.
static SmallVector<Value> computeRowMajorDynamicStrides(
    Location loc, OpBuilder &builder, ArrayRef<int64_t> staticStrides,
    ArrayRef<int64_t> staticSizes, ArrayRef<Value> dynamicSizes) {
  // Early return if no dynamic strides to compute - avoids generating unused
  // ops
  if (countDynamicDimensions(staticStrides) == 0 || staticStrides.empty()) {
    return {};
  }

  // Build a mapping from dimension index to its Value (static or dynamic).
  SmallVector<Value> sizeValues;
  size_t dynamicSizeIdx = 0;
  for (int64_t staticSize : staticSizes) {
    if (staticSize == cuda_tile::TensorViewType::kDynamic) {
      assert(dynamicSizeIdx < dynamicSizes.size());
      sizeValues.push_back(dynamicSizes[dynamicSizeIdx++]);
    } else {
      auto i32TileTy = cuda_tile::TileType::get(llvm::ArrayRef<int64_t>{},
                                                builder.getI32Type());
      auto constAttr =
          llvm::cast<DenseTypedElementsAttr>(DenseElementsAttr::get(
              i32TileTy, ArrayRef<int32_t>(static_cast<int32_t>(staticSize))));
      sizeValues.push_back(
          cuda_tile::ConstantOp::create(builder, loc, i32TileTy, constAttr));
    }
  }

  // Compute row-major strides: stride[i] = product(size[i+1..last]).
  // Walk dims right-to-left, accumulating a running product. stride[last] = 1
  // is always static (handled by staticStrides), so we start with
  // runningStride = size[last] which gives stride[last-1].
  SmallVector<Value> strideValues(staticStrides.size(), nullptr);
  Value runningStride = sizeValues.back();
  for (size_t reverseIdx = staticStrides.size() - 1; reverseIdx > 0;
       --reverseIdx) {
    size_t i = reverseIdx - 1;
    strideValues[i] = runningStride;
    if (i > 0) {
      runningStride =
          cuda_tile::MulIOp::create(builder, loc, runningStride, sizeValues[i]);
    }
  }

  // Emit dynamic strides in left-to-right order matching staticStrides.
  SmallVector<Value> dynamicStrides;
  for (size_t i = 0; i < staticStrides.size(); ++i) {
    if (staticStrides[i] == cuda_tile::TensorViewType::kDynamic) {
      assert(strideValues[i] && "dynamic stride has no computed value");
      dynamicStrides.push_back(strideValues[i]);
    }
  }
  return dynamicStrides;
}

struct DynamicTensorViewOperands {
  SmallVector<Value> sizes;
  SmallVector<Value> strides;
};

/// Build the dynamic operands for make_tensor_view.
///
/// Dynamic sizes get non-negative bounds before they are reused to compute
/// implicit row-major strides. Dynamic strides then get non-negative bounds and
/// any TensorIR stride divisibility facts before make_tensor_view consumes
/// them.
static DynamicTensorViewOperands
buildDynamicTensorViewOperands(OpBuilder &builder, Location loc,
                               const CudaTileTensorDescriptor &descriptor) {
  DynamicTensorViewOperands operands;

  operands.sizes = applyDescriptorBoundAssumptions(
      builder, loc, descriptor.getFilteredDynamicSizes());

  if (descriptor.shouldComputeStridesFromSizes()) {
    operands.strides = computeRowMajorDynamicStrides(
        loc, builder, descriptor.getStaticStrides(),
        descriptor.getStaticSizes(), operands.sizes);
  } else {
    operands.strides = descriptor.getFilteredDynamicStrides();
  }

  operands.strides = applyDescriptorValueAssumptions(
      builder, loc, operands.strides,
      descriptor.getFilteredDynamicStrideDivisibility());
  return operands;
}

/// Generate tensor views for all input and output tensor descriptors with
/// enhanced alignment.
static void
generateTensorViews(ConversionPatternRewriter &rewriter,
                    TensorToCudaTileConversionState &conversionState,
                    cuda_tile::EntryOp entryOp) {
  MLIRContext *ctx = rewriter.getContext();

  // Generate a tensor view for each tensor argument.
  for (auto &descriptor : conversionState.getInOutTensorDescriptors()) {
    BlockArgument tensorPtr = descriptor.getCudaTilePtr();

    auto tileTy = dyn_cast<cuda_tile::TileType>(tensorPtr.getType());
    if (!tileTy || !isa<cuda_tile::PointerType>(tileTy.getElementType())) {
      continue;
    }

    auto elemType =
        cast<cuda_tile::PointerType>(tileTy.getElementType()).getPointeeType();

    // Use per-tensor alignment if explicitly set, otherwise use the default.
    int64_t alignment = descriptor.getAlignment();
    if (alignment <= 0) {
      alignment = tensor_to_cuda_tile::kDefaultPointerAlignment;
    }
    Value alignedPtr = applyAlignmentIfNeeded(rewriter, tensorPtr.getLoc(),
                                              tensorPtr, elemType, alignment);

    DynamicTensorViewOperands dynamicOperands = buildDynamicTensorViewOperands(
        rewriter, tensorPtr.getLoc(), descriptor);

    // Create tensor view with aligned pointer.
    auto viewType = cuda_tile::TensorViewType::get(
        ctx, elemType, descriptor.getStaticSizes(),
        descriptor.getStaticStrides());

    Value tensorView = cuda_tile::MakeTensorViewOp::create(
        rewriter, tensorPtr.getLoc(), viewType, alignedPtr,
        dynamicOperands.sizes, dynamicOperands.strides);
    descriptor.setTensorView(tensorView);

    conversionState.mapTensorViewToDescriptor(tensorView, &descriptor);

    if (descriptor.isInput() &&
        conversionState.kernelInfo.useZeroPaddedInputLoads) {
      auto paddedInputView = buildPartitionView(
          rewriter, descriptor, conversionState, PartitionViewPadding::Zero);
      descriptor.setZeroPaddedPartitionView(paddedInputView.getResult());
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Generating tensor view for: " << tensorPtr << "\n";
      llvm::dbgs() << "  * Alignment: " << alignment << " bytes"
                   << (descriptor.getAlignment() > 0 ? " (from attribute)"
                                                     : " (auto-detected)")
                   << "\n";
      llvm::dbgs() << "  * Compute strides from sizes: "
                   << descriptor.shouldComputeStridesFromSizes() << "\n";
      llvm::dbgs() << "  * TensorViewOp: " << tensorView << "\n";
    });
  }
}

/// Use tensor descriptor information and iteration space affine maps to match
/// each iteration space dimension to a dimension of a tensor view.
static void
locateIterationSpaceDimSizes(TensorToCudaTileConversionState &conversionState) {
  KernelInfo &kernelInfo = conversionState.kernelInfo;
  int64_t iterSpaceRank = kernelInfo.getIterationSpaceRank();

  // Skip empty graph.
  if (iterSpaceRank == 0) {
    return;
  }

  auto allDescriptors = conversionState.getInOutTensorDescriptors();
  assert(!allDescriptors.empty() && "No tensor descriptors found");

  // Process each iteration space separately.
  for (int32_t iterSpaceId : conversionState.getIterSpaceIds()) {
    IterSpaceDimInfo &dimInfo =
        conversionState.getOrCreateIterSpaceDimInfo(iterSpaceId);

    // Initialize dimension locations for this iteration space.
    dimInfo.iterSpaceDimLocations.assign(iterSpaceRank,
                                         std::make_pair(nullptr, -1));

    unsigned numIdentifiedDims = 0;

    // Find tensors belonging to this iteration space.
    for (auto &tensorDescriptor : allDescriptors) {
      if (!tensorDescriptor.belongsToIterSpace(iterSpaceId)) {
        continue;
      }

      auto iterSpaceMap = tensorDescriptor.getIterSpaceMap();
      int numTensorDims = tensorDescriptor.getRank();

      // Traverse all the dimensions this tensor is defined on.
      for (int tensorDimIdx = 0; tensorDimIdx < numTensorDims; ++tensorDimIdx) {
        auto resultExpr = iterSpaceMap.getResults()[tensorDimIdx];

        // Stop search if we have identified all the dimensions.
        if (numIdentifiedDims == static_cast<unsigned>(iterSpaceRank)) {
          break;
        }

        int64_t iterSpaceDimIdx = getIterSpaceDimIdx(resultExpr, tensorDimIdx);
        auto [existingView, existingDimIdx] =
            dimInfo.iterSpaceDimLocations[iterSpaceDimIdx];
        bool hasDimBeenIdentified = (existingView != nullptr);
        int64_t newSize = tensorDescriptor.getStaticSizes()[tensorDimIdx];
        // Static size-1 dims are recorded tentatively without incrementing
        // numIdentifiedDims, so the early break does not fire before a
        // non-broadcast tensor can supply the true extent.
        bool newIsBroadcastPlaceholder = (newSize == 1);
        if (hasDimBeenIdentified) {
          const CudaTileTensorDescriptor &existingDescr =
              conversionState.getTensorViewDescriptor(existingView);
          int64_t existingSize = existingDescr.getStaticSizes()[existingDimIdx];
          bool existingIsBroadcastPlaceholder = (existingSize == 1);
          if (!existingIsBroadcastPlaceholder || newIsBroadcastPlaceholder) {
            continue; // Existing confirmed, or both are broadcast; keep
                      // existing.
          }
          // Override: promote tentative broadcast placeholder to confirmed.
          ++numIdentifiedDims;
        } else if (newIsBroadcastPlaceholder) {
          // Tentative: location set but not yet confirmed.
        } else {
          // Found a new dimension that hasn't been identified yet.
          ++numIdentifiedDims;
        }

        // Map the tensor view and dimension index to the iteration space
        // dimension; may overwrite a tentative broadcast placeholder.
        dimInfo.iterSpaceDimLocations[iterSpaceDimIdx] =
            std::make_pair(tensorDescriptor.getTensorView(), tensorDimIdx);

        LLVM_DEBUG({
          IterSpaceDimLocation &location =
              dimInfo.iterSpaceDimLocations[iterSpaceDimIdx];
          llvm::dbgs() << "Iter space " << iterSpaceId
                       << ": Location found for dimension " << iterSpaceDimIdx
                       << ":\n"
                       << llvm::indent(4) << location.first << ", "
                       << llvm::indent(4) << location.second << "\n";
        });
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Iteration space " << iterSpaceId
                   << " dimension locations identified: ";
      llvm::interleaveComma(dimInfo.iterSpaceDimLocations, llvm::dbgs(),
                            [](const IterSpaceDimLocation &loc) {
                              llvm::dbgs() << "(";
                              if (loc.first) {
                                loc.first.printAsOperand(llvm::dbgs(),
                                                         OpPrintingFlags());
                              } else {
                                llvm::dbgs() << "null";
                              }
                              llvm::dbgs() << ", " << loc.second << ")";
                            });
      llvm::dbgs() << "\n";
    });

    assert(llvm::all_of(dimInfo.iterSpaceDimLocations,
                        [](const IterSpaceDimLocation &location) {
                          return location.first != nullptr &&
                                 location.second != -1;
                        }) &&
           "Not all iteration space dimension locations have been identified");
  }
}

AffineMap
TensorToCudaTileConversionState::getIterSpaceMapForValue(Value value) const {
  assert(value && "Value is null");

  // CUDA Tile tensor view case.
  if (isa<cuda_tile::TensorViewType>(value.getType())) {
    CudaTileTensorDescriptor &descriptor = getTensorViewDescriptor(value);
    return descriptor.getIterSpaceMap();
  }

  // Tensor IR operation case.
  if (Operation *defOp = value.getDefiningOp()) {
    return getOutputIterSpaceMapForOperation(defOp);
  }

  // Block argument case. Retrieve the descriptor based on the input value.
  assert(isa<BlockArgument>(value) && "Value is not a block argument");
  auto *descriptor = findDescriptorForInput(value);
  assert(descriptor && "No descriptor found for block argument");
  return descriptor->getIterSpaceMap();
}

int32_t
TensorToCudaTileConversionState::getIterSpaceIdForValue(Value value) const {
  assert(value && "Value is null");

  // CUDA Tile tensor view case.
  if (isa<cuda_tile::TensorViewType>(value.getType())) {
    CudaTileTensorDescriptor &descriptor = getTensorViewDescriptor(value);
    return descriptor.getIterSpaceId();
  }

  // Tensor IR operation case.
  if (Operation *defOp = value.getDefiningOp()) {
    return getIterSpaceIdForOp(defOp);
  }

  // Block argument case. Retrieve the descriptor based on the input value.
  assert(isa<BlockArgument>(value) && "Value is not a block argument");
  auto *descriptor = findDescriptorForInput(value);
  assert(descriptor && "No descriptor found for block argument");
  return descriptor->getIterSpaceId();
}

AffineMap TensorToCudaTileConversionState::getOutputIterSpaceMapForOperation(
    Operation *op) const {
  auto attr = op->getAttrOfType<AffineMapAttr>(
      TensorIRDialect::getIterSpaceMapAttrName());
  assert(attr && "Iteration space map attribute must exist");
  return attr.getAffineMap();
}

/// Compute the union iteration space map for an operation. This is the
/// union of all iteration space dimensions used by the operation's inputs
/// and output.
AffineMap
TensorToCudaTileConversionState::getOpUnionIterSpaceMap(Operation *op) const {
  auto iterSpaceOp = cast<IterationSpaceInfoInterface>(op);

  AffineMap outputMap = iterSpaceOp.getOutputIterSpaceMap();
  unsigned numIterSpaceDims = outputMap.getNumInputs();

  // Set to true the dimensions used by an affine map.
  SmallVector<bool> usedDims(numIterSpaceDims, false);
  auto setUsedDims = [&](AffineMap map) {
    if (!map) {
      return;
    }
    for (auto expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        usedDims[dimExpr.getPosition()] = true;
      }
    }
  };

  // Set to true the dimensions used by the inputs. We need to skip unrealized
  // conversion casts and retrieve the iteration space map from the original
  // operand.
  for (Value operand : op->getOperands()) {
    Value origOperand = stripUnrealizedConversionCast(operand);
    auto iterSpaceMap = getIterSpaceMapForValue(origOperand);
    setUsedDims(iterSpaceMap);
  }

  // Set to true the dimensions used by the output.
  setUsedDims(outputMap);

  return AffineMap::getFilteredIdentityMap(
      op->getContext(), numIterSpaceDims,
      [&](AffineDimExpr dim) { return usedDims[dim.getPosition()]; });
}

OpBuilder::InsertPoint TensorToCudaTileConversionState::getInsertPointForMap(
    int32_t iterSpaceId, AffineMap iterSpaceMap) const {
  // Find the "deepest" iteration space dimension that this value is defined on.
  int64_t deepestDimIdx = -1;
  for (auto [exprIdx, resExpr] : llvm::enumerate(iterSpaceMap.getResults())) {
    int64_t inputDimIdx = getIterSpaceDimIdx(resExpr, exprIdx);
    deepestDimIdx = std::max(deepestDimIdx, inputDimIdx);
  }

  assert(deepestDimIdx >= 0 && "Deepest dimension couldn't be computed");
  return getInsertPointForIterSpaceDim(iterSpaceId, deepestDimIdx);
}

OpBuilder::InsertPoint
TensorToCudaTileConversionState::getInsertPointForValue(Value value) const {
  auto iterSpaceMap = getIterSpaceMapForValue(value);
  int32_t iterSpaceId = getIterSpaceIdForValue(value);
  return getInsertPointForMap(iterSpaceId, iterSpaceMap);
}

OpBuilder::InsertPoint
TensorToCudaTileConversionState::getInsertPointForOp(Operation *op) const {
  int32_t iterSpaceId = getIterSpaceIdForOp(op);
  auto iterSpaceMap = getOpUnionIterSpaceMap(op);
  return getInsertPointForMap(iterSpaceId, iterSpaceMap);
}

static cuda_tile::MakePartitionViewOp
buildPartitionView(ConversionPatternRewriter &rewriter,
                   const CudaTileTensorDescriptor &tensorDescr,
                   TensorToCudaTileConversionState &conversionState,
                   PartitionViewPadding padding) {
  bool useZeroPadding = padding == PartitionViewPadding::Zero;
  if (useZeroPadding && tensorDescr.hasZeroPaddedPartitionView()) {
    return cast<cuda_tile::MakePartitionViewOp>(
        tensorDescr.getZeroPaddedPartitionView().getDefiningOp());
  }

  Value tensorView = tensorDescr.getTensorView();
  auto tensorViewTy = cast<cuda_tile::TensorViewType>(tensorView.getType());
  MLIRContext *ctx = tensorView.getContext();
  Location loc = tensorView.getLoc();

  const auto &iterSpaceTileSizes = conversionState.kernelInfo.tileSizes;
  assert(!iterSpaceTileSizes.empty() &&
         "Tile sizes should always be computed by computeOptimalTileSizes");

  AffineMap iterSpaceMap = tensorDescr.getIterSpaceMap();
  SmallVector<int64_t> tensorTileSizes =
      getTileSizesForIterSpaceMap(iterSpaceMap, iterSpaceTileSizes);

  // Cap tile sizes to actual tensor dimensions
  auto tensorSizes = tensorDescr.getStaticSizes();
  size_t tensorRank = tensorDescr.getRank();
  for (size_t i = 0; i < tensorRank; ++i) {
    // Only cap if dimension is static (not dynamic)
    if (tensorSizes[i] != cuda_tile::TensorViewType::kDynamic) {
      tensorTileSizes[i] =
          std::min(tensorTileSizes[i], static_cast<int64_t>(tensorSizes[i]));
    }
  }

  auto tileSizesAttr =
      DenseI32ArrayAttr::get(ctx, convertInt64ToI32(tensorTileSizes));
  SmallVector<int32_t> dimMap(tensorRank);
  for (size_t i = 0; i < tensorRank; ++i) {
    dimMap[i] = static_cast<int32_t>(i);
  }

  cuda_tile::PaddingValueAttr paddingValue;
  if (useZeroPadding) {
    paddingValue =
        cuda_tile::PaddingValueAttr::get(ctx, cuda_tile::PaddingValue::zero);
  }
  auto partitionTy = cuda_tile::PartitionViewType::get(
      ctx, tileSizesAttr, tensorViewTy, dimMap, paddingValue);

  return cuda_tile::MakePartitionViewOp::create(rewriter, loc, partitionTy,
                                                tensorDescr.getTensorView());
}

static LogicalResult computeAddressAndTileSizesForLoadOrStore(
    ConversionPatternRewriter &rewriter,
    const CudaTileTensorDescriptor &tensorDescr,
    TensorToCudaTileConversionState &conversionState,
    cuda_tile::MakePartitionViewOp &partViewOp, SmallVectorImpl<Value> &coords,
    cuda_tile::TileType &tileType,
    PartitionViewPadding padding = PartitionViewPadding::None) {

  LLVM_DEBUG({
    llvm::dbgs() << "Computing address and tile sizes for loads/store:\n";
  });

  auto tensorView = tensorDescr.getTensorView();
  auto tensorViewTy = cast<cuda_tile::TensorViewType>(tensorView.getType());
  size_t tensorRank = tensorDescr.getRank();

  MLIRContext *ctx = tensorView.getContext();
  Location loc = tensorView.getLoc();

  partViewOp =
      buildPartitionView(rewriter, tensorDescr, conversionState, padding);

  // Get the effective tile index - either block ID (normal) or persistence loop
  // induction variable (persistent kernels)
  Value tileIndex = conversionState.getEffectiveTileIndex();
  if (!tileIndex) {
    LLVM_DEBUG(llvm::dbgs()
               << "ERROR: Invalid tile index for coordinate calculation\n");
    return failure();
  }

  auto coordsResult = calculateBlockIndices(
      loc, rewriter, tileIndex, partViewOp, tensorDescr, conversionState);
  if (failed(coordsResult)) {
    // Fallback to direct block indexing - get block IDs from conversionState
    auto blockIds = conversionState.getGetTileBlockIdOp()->getResults();
    LLVM_DEBUG({
      llvm::dbgs() << "WARNING: Block coordinate calculation failed, "
                      "falling back to direct block indexing\n";
      llvm::dbgs() << "  Tensor rank: " << tensorRank << "\n";
      llvm::dbgs() << "  Available block IDs: " << blockIds.size() << "\n";
    });

    if (blockIds.size() < tensorRank) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ERROR: Insufficient block IDs for naive fallback\n");
      return failure();
    }

    auto indexing = blockIds.take_front(tensorRank);
    coords = SmallVector<Value>(indexing.begin(), indexing.end());
  } else {
    coords = std::move(*coordsResult);
  }

  // Clamp coordinates to the tensor's actual index space bounds.
  // This is critical for broadcast/smaller tensors where the iteration space
  // is larger than the tensor's own dimensions (e.g., scale tensor 1x1024
  // with main tensor 32000x1024). Without clamping, we'd access out-of-bounds
  // memory for TMA operations.
  //
  // Optimization: Skip clamping when both this tensor and the iteration space
  // have the same number of tiles in a dimension. This happens when:
  // 1. This tensor's size >= defining tensor's size (same numTiles), OR
  // 2. This tensor's size < defining tensor's size, but the iteration space
  //    has only 1 tile (definingSize <= iterSpaceTileSize), so both tensors
  //    have 1 tile in that dimension.
  {
    const auto &kernelInfo = conversionState.kernelInfo;
    AffineMap iterSpaceMap = tensorDescr.getIterSpaceMap();
    ArrayRef<int64_t> staticSizes = tensorDescr.getStaticSizes();
    // Use per-iteration-space dimension locations for this tensor's iteration
    // space.
    int32_t iterSpaceId = tensorDescr.getIterSpaceId();
    IterSpaceDimInfo &dimInfo =
        conversionState.getIterSpaceDimInfo(iterSpaceId);
    const auto &iterSpaceDimLocations = dimInfo.iterSpaceDimLocations;
    const auto &iterSpaceTileSizes = kernelInfo.tileSizes;

    // Determine which dimensions need clamping
    SmallVector<bool> needsClamping(tensorRank, false);
    bool anyNeedsClamping = false;

    for (size_t tensorDimIdx = 0; tensorDimIdx < tensorRank; ++tensorDimIdx) {
      // Map tensor dimension to iteration space dimension
      auto iterSpaceDimExpr = iterSpaceMap.getResult(tensorDimIdx);
      int64_t iterSpaceDimIdx = -1;
      if (auto dimExpr = dyn_cast<AffineDimExpr>(iterSpaceDimExpr)) {
        iterSpaceDimIdx = dimExpr.getPosition();
      }

      int64_t dimSize = staticSizes[tensorDimIdx];

      // Check if clamping is needed
      bool needsClampingForDim = false;
      if (iterSpaceDimIdx >= 0 &&
          iterSpaceDimIdx <
              static_cast<int64_t>(iterSpaceDimLocations.size())) {
        auto [definingTensorView, definingDimIdx] =
            iterSpaceDimLocations[iterSpaceDimIdx];

        // Get the defining tensor's size in this dimension
        const auto &definingDescr =
            conversionState.getTensorViewDescriptor(definingTensorView);
        int64_t definingDimSize =
            definingDescr.getStaticSizes()[definingDimIdx];

        // Get the iteration space tile size for this dimension
        int64_t iterSpaceTileSize = iterSpaceTileSizes[iterSpaceDimIdx];

        // Clamping is needed only if:
        // 1. This tensor's size < defining tensor's size (fewer tiles), AND
        // 2. The iteration space has more than 1 tile (definingSize > tileSize)
        //
        // Why condition 2? If definingSize <= tileSize, then:
        //   numTiles = ceil(definingSize / tileSize) = 1
        // The iteration space has only 1 tile, so coordinates are always 0.
        // Since this tensor also has at most 1 tile (its size <= definingSize
        // <= tileSize), both tensors have 1 tile and no clamping is needed.
        //
        // Example (reduction output, no clamping needed):
        //   Input: (16, 32, 64), Output: (16, 1, 64), TileSize: (8, 32, 64)
        //   Dim 1: definingSize=32, tileSize=32 → 32 <= 32 → 1 tile
        //   Output dim 1 size=1 → also 1 tile → no clamping
        //
        // Example (broadcast scale, clamping needed):
        //   Input: (32000, 1024), Scale: (1, 1024), TileSize: (8, 1024)
        //   Dim 0: definingSize=32000, tileSize=8 → 32000 > 8 → 4000 tiles
        //   Scale dim 0 size=1 → 1 tile → MISMATCH → needs clamping
        //
        // Special case: dimSize==1 and definingDimSize is dynamic. The else-if
        // below requires both sizes to be static; when definingDimSize is
        // unknown at compile time, conservatively clamp. Reduction outputs
        // (definingDimSize static) fall through to the else-if correctly.
        const int64_t kDynamic = cuda_tile::TensorViewType::kDynamic;
        bool isStaticOneDim = (dimSize != kDynamic) && (dimSize == 1);
        if (isStaticOneDim && definingDimSize == kDynamic) {
          needsClampingForDim = true;
        } else if (dimSize != kDynamic && definingDimSize != kDynamic) {
          needsClampingForDim = (dimSize < definingDimSize) &&
                                (definingDimSize > iterSpaceTileSize);
        }
        // If either size is dynamic, clamping cannot be determined at compile
        // time; leave as false.
      }

      needsClamping[tensorDimIdx] = needsClampingForDim;
      if (needsClampingForDim) {
        anyNeedsClamping = true;
      }
    }

    // Only emit clamping operations if at least one dimension needs it
    if (anyNeedsClamping) {
      auto partViewType = partViewOp.getType();
      auto tileType = cast<cuda_tile::TileType>(coords[0].getType());
      SmallVector<Type> resultTypes(partViewType.getViewIndexRank(), tileType);
      auto indexSpaceShapeOp = cuda_tile::GetIndexSpaceShapeOp::create(
          rewriter, loc, resultTypes, partViewOp);
      auto indexSpaceResults = indexSpaceShapeOp->getResults();

      auto unsignedAttr =
          cuda_tile::SignednessAttr::get(ctx, cuda_tile::Signedness::Unsigned);

      for (size_t i = 0; i < tensorRank; ++i) {
        if (needsClamping[i]) {
          // coord[i] = coord[i] % numTilesInDim[i]
          // This clamps the coordinate to the valid range for this tensor
          coords[i] = cuda_tile::RemIOp::create(
              rewriter, loc, coords[i], indexSpaceResults[i], unsignedAttr);
        }
      }
    }
  }

  // Validate coordinates
  if (coords.size() != tensorRank) {
    LLVM_DEBUG(llvm::dbgs() << "ERROR: Coordinate count mismatch: expected "
                            << tensorRank << ", got " << coords.size() << "\n");
    return failure();
  }

  auto tileSizes = convertInt32ToI64(partViewOp.getType().getTileShape());
  tileType = cuda_tile::TileType::get(tileSizes, tensorViewTy.getElementType());

  LLVM_DEBUG({
    llvm::dbgs() << llvm::indent(4) << "Partition view: " << partViewOp << "\n";
    llvm::dbgs() << llvm::indent(4) << "Coordinates: (";
    llvm::interleaveComma(coords, llvm::dbgs());
    llvm::dbgs() << ")\n";
    llvm::dbgs() << llvm::indent(4) << "Tile type: " << tileType << "\n";
  });

  return success();
}

/// Create a load operation for the given tensor descriptor. The load is
/// inserted at the appropriate insertion point based on the tensor's iteration
/// space map. Returns the loaded tile values on success.
static FailureOr<SmallVector<Value>>
createLoad(ConversionPatternRewriter &rewriter,
           TensorToCudaTileConversionState &conversionState,
           CudaTileTensorDescriptor &tensorDescr) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto insertPoint =
      conversionState.getInsertPointForValue(tensorDescr.getTensorView());
  rewriter.restoreInsertionPoint(insertPoint);

  cuda_tile::MakePartitionViewOp partView;
  SmallVector<Value> coords;
  cuda_tile::TileType tileType;
  PartitionViewPadding padding =
      conversionState.kernelInfo.useZeroPaddedInputLoads
          ? PartitionViewPadding::Zero
          : PartitionViewPadding::None;
  if (failed(computeAddressAndTileSizesForLoadOrStore(
          rewriter, tensorDescr, conversionState, partView, coords, tileType,
          padding))) {
    return failure();
  }

  Location loc = partView.getLoc();
  MLIRContext *ctx = partView.getContext();

  // Create optimization hints if any non-default values are set.
  cuda_tile::OptimizationHintsAttr optHints =
      tensor_to_cuda_tile::createLoadStoreOptimizationHints(
          ctx, tensorDescr.getAllowTma(), tensorDescr.getCost());

  auto loadOp = cuda_tile::LoadViewTkoOp::create(
      rewriter, loc, tileType, cuda_tile::TokenType::get(ctx),
      cuda_tile::MemoryOrderingSemantics::WEAK, /*scope=*/nullptr, partView,
      coords, /*token=*/nullptr, optHints);

  SmallVector<Value> results(loadOp->getResults().begin(),
                             loadOp->getResults().end());

  return results;
}

static FailureOr<SmallVector<Value>>
createStore(ConversionPatternRewriter &rewriter,
            TensorToCudaTileConversionState &conversionState,
            CudaTileTensorDescriptor &tensorDescr, Value tileToStore) {
  assert(tileToStore && "No tile to store for output tensor descriptor");

  OpBuilder::InsertionGuard guard(rewriter);
  auto insertPoint =
      conversionState.getInsertPointForValue(tensorDescr.getTensorView());
  rewriter.restoreInsertionPoint(insertPoint);

  cuda_tile::MakePartitionViewOp partView;
  SmallVector<Value> coords;
  cuda_tile::TileType tileType;
  if (failed(computeAddressAndTileSizesForLoadOrStore(rewriter, tensorDescr,
                                                      conversionState, partView,
                                                      coords, tileType))) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Tile to store: " << tileToStore << "\n";
    llvm::dbgs() << "Partition view: " << partView << "\n";
  });

  assert(tileType == tileToStore.getType() &&
         "Tile type should match the tile type of the value to store");

  Location loc = partView.getLoc();
  MLIRContext *ctx = partView.getContext();

  // Create optimization hints if any non-default values are set.
  cuda_tile::OptimizationHintsAttr optHints =
      tensor_to_cuda_tile::createLoadStoreOptimizationHints(
          ctx, tensorDescr.getAllowTma(), tensorDescr.getCost());

  auto storeOp = cuda_tile::StoreViewTkoOp::create(
      rewriter, loc, cuda_tile::TokenType::get(ctx),
      cuda_tile::MemoryOrderingSemantics::WEAK,
      /*scope=*/nullptr,
      /*tile=*/tileToStore, partView, coords,
      /*token=*/nullptr, optHints);

  SmallVector<Value> results;
  results.append(storeOp->getResults().begin(), storeOp->getResults().end());
  return results;
}

/// Returns true if all the tensors in the graph have the same rank as the
/// provided uniform rank. This utility can be used to determine if the rank of
/// all the tensors in the graph matches the rank of the iteration space.
static bool hasUniformTensors(nv_tensor_ir::GraphOp graphOp,
                              int64_t uniformRank) {
  auto walkResult = graphOp.walk([&](Operation *op) -> WalkResult {
    auto uniformTensorRank = [&](Value value) -> bool {
      auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(value.getType());
      return !tensorType || tensorType.getRank() == uniformRank;
    };

    // Check operands.
    for (Value operand : op->getOperands()) {
      if (!uniformTensorRank(operand)) {
        return WalkResult::interrupt();
      }
    }

    // Check results.
    for (Value result : op->getResults()) {
      if (!uniformTensorRank(result)) {
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  return !walkResult.wasInterrupted();
}

/// Check if the graph has tensors with different ranks.
///
/// TileAnalyzer assumes all tensors have the same rank, so we need to detect
/// mixed ranks and use conservative tile sizes in that case.
static bool hasMixedRanks(nv_tensor_ir::GraphOp graphOp) {
  if (graphOp.getArgumentTypes().empty()) {
    return false; // No tensors to compare
  }

  size_t expectedRank = 0;
  bool firstTensor = true;

  // Check input arguments
  for (Type argType : graphOp.getArgumentTypes()) {
    if (auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(argType)) {
      size_t currentRank = tensorType.getRank();
      if (firstTensor) {
        expectedRank = currentRank;
        firstTensor = false;
      } else if (currentRank != expectedRank) {
        return true; // Found different rank
      }
    }
  }

  // Check result types
  for (Type resType : graphOp.getFunctionType().getResults()) {
    if (auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(resType)) {
      size_t currentRank = tensorType.getRank();
      if (currentRank != expectedRank) {
        return true; // Found different rank
      }
    }
  }

  return false; // All tensors have the same rank
}

/// Get the maximum rank among all tensors in the graph.
static size_t getMaxRank(nv_tensor_ir::GraphOp graphOp) {
  size_t maxRank = 2; // Conservative fallback

  // Check input arguments
  for (Type argType : graphOp.getArgumentTypes()) {
    if (auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(argType)) {
      maxRank = std::max(maxRank, static_cast<size_t>(tensorType.getRank()));
    }
  }

  // Check result types
  for (Type resType : graphOp.getFunctionType().getResults()) {
    if (auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(resType)) {
      maxRank = std::max(maxRank, static_cast<size_t>(tensorType.getRank()));
    }
  }

  return maxRank;
}

static bool hasAnyDynamicInputs(nv_tensor_ir::GraphOp graphOp) {
  // Check input tensors only - outputs are derived from inputs
  for (Type type : graphOp.getArgumentTypes()) {
    if (auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(type)) {
      if (!tensorType.hasStaticShape()) {
        return true; // Found a dynamic input tensor
      }
    }
  }
  return false; // All input tensors are static
}

/// Tries to retrieve the iteration space rank from the graph. Returns 0 if no
/// iteration space was found (e.g., empty graph).
static int64_t retrieveIterationSpaceRank(nv_tensor_ir::GraphOp graphOp) {
  int64_t iterSpaceRank = 0;
  // First, try to retrieve an iteration space map from the arguments.
  for (unsigned i = 0, numArgs = graphOp.getNumArguments(); i < numArgs; ++i) {
    auto iterSpaceAttr = dyn_cast_or_null<AffineMapAttr>(
        graphOp.getArgAttr(i, TensorIRDialect::getIterSpaceMapAttrName()));
    if (iterSpaceAttr) {
      iterSpaceRank = iterSpaceAttr.getAffineMap().getNumInputs();
      break;
    }
  }

  // Otherwise, try to retrieve it from an operation.
  if (iterSpaceRank == 0) {
    graphOp.walk([&](Operation *op) -> WalkResult {
      auto iterSpaceMapAttr = op->getAttrOfType<AffineMapAttr>(
          TensorIRDialect::getIterSpaceMapAttrName());
      if (iterSpaceMapAttr) {
        iterSpaceRank = iterSpaceMapAttr.getAffineMap().getNumInputs();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Iteration space rank found in graph: " << iterSpaceRank
                 << "\n";
  });

  return iterSpaceRank;
}

static bool hasMatmulOp(nv_tensor_ir::GraphOp graphOp) {
  bool found = false;
  graphOp.walk([&](nv_tensor_ir::MatmulOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

// TODO: We may need to propagate thread mapping information as part of the
// iteration space information analysis. This methos is currently a placeholder
// for a better implementation based on operation properties.
static ThreadToIterSpaceMapping computeThreadToIterSpaceMapping(
    nv_tensor_ir::GraphOp graphOp,
    TensorToCudaTileConversionState &conversionState, int64_t iterSpaceRank) {
  // Assume no tiling is needed inside the kernel by default.
  auto threadToIterSpaceMapping =
      ThreadToIterSpaceMapping::getNoTilingConfig(iterSpaceRank);

  // TODO: This is just a special-case for matmul operations that sets the
  // reduction dimension to be unmapped. This should be reimplemented based on
  // operation properties.
  graphOp.walk([&](Operation *op) -> WalkResult {
    if (isa<nv_tensor_ir::MatmulOp>(op)) {
      threadToIterSpaceMapping.setDimMapping(
          iterSpaceRank - 1, ThreadToIterSpaceMapping::kUnmappedIterSpaceDim);
    }

    return WalkResult::advance();
  });

  return threadToIterSpaceMapping;
}

/// Compute kernel/grid information and tile sizes for the AffineMap path.
/// This function handles tile heuristics (analyzeTileSizeForGraph) and
/// persistence grid computation from tensor shapes.
static LogicalResult computeKernelInfoAndTileSizes(
    nv_tensor_ir::GraphOp graphOp,
    TensorToCudaTileConversionState &conversionState) {
  constexpr int32_t kDefaultTileSize = 2;

  int64_t iterSpaceRank = retrieveIterationSpaceRank(graphOp);

  conversionState.setIterSpaceIds(collectIterSpaceIds(graphOp));
  conversionState.threadToIterSpaceMapping =
      computeThreadToIterSpaceMapping(graphOp, conversionState, iterSpaceRank);
  conversionState.kernelInfo = KernelInfo(iterSpaceRank);
  KernelInfo &kernelInfo = conversionState.kernelInfo;
  bool containsMatmul = hasMatmulOp(graphOp);
  kernelInfo.useZeroPaddedInputLoads = containsMatmul;

  if (iterSpaceRank == 0) {
    return success();
  }

  // Resolve tile sizes from available sources.
  if (!conversionState.tileSize.empty()) {
    if (static_cast<int64_t>(conversionState.tileSize.size()) !=
        iterSpaceRank) {
      kernelInfo.tileSizes = SmallVector<int32_t>(iterSpaceRank);
      for (int64_t i = 0; i < iterSpaceRank; ++i) {
        if (i < static_cast<int64_t>(conversionState.tileSize.size())) {
          kernelInfo.tileSizes[i] = conversionState.tileSize[i];
        } else {
          kernelInfo.tileSizes[i] = kDefaultTileSize;
        }
      }
    } else {
      kernelInfo.tileSizes = conversionState.tileSize;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "Using user-specified tile sizes: [";
      llvm::interleaveComma(kernelInfo.tileSizes, llvm::dbgs());
      llvm::dbgs() << "] for graph " << graphOp.getSymName() << "\n";
    });
  } else if (!hasUniformTensors(graphOp, iterSpaceRank)) {
    kernelInfo.tileSizes =
        SmallVector<int32_t>(iterSpaceRank, kDefaultTileSize);
  } else if (hasMixedRanks(graphOp)) {
    size_t maxRank = getMaxRank(graphOp);
    kernelInfo.tileSizes = SmallVector<int32_t>(maxRank, kDefaultTileSize);
  } else {
    bool hasInputTensors = !graphOp.getArgumentTypes().empty();
    bool hasDynamicInputs = hasAnyDynamicInputs(graphOp);

    if (hasDynamicInputs) {
      auto tensorType =
          dyn_cast<nv_tensor_ir::TensorType>(graphOp.getArgumentTypes()[0]);
      if (tensorType) {
        size_t rank = tensorType.getRank();
        kernelInfo.tileSizes = SmallVector<int32_t>(rank, kDefaultTileSize);
      } else {
        kernelInfo.tileSizes =
            SmallVector<int32_t>(iterSpaceRank, kDefaultTileSize);
      }
    } else if (hasInputTensors) {
      TileAnalyzer tileAnalyzer{GpuArchitecture{}};
      MLIR_ASSIGN_OR_RETURN(auto analyzedTileSizes,
                            analyzeTileSizeForGraph(tileAnalyzer, graphOp));
      // DimSize is int64_t but tileSizes is int32_t; convert element-wise.
      kernelInfo.tileSizes = SmallVector<int32_t>(analyzedTileSizes.begin(),
                                                  analyzedTileSizes.end());
    } else {
      auto results = graphOp.getFunctionType().getResults();
      auto it = llvm::find_if(results, [](Type t) { return isTensorType(t); });
      if (it == results.end()) {
        kernelInfo.tileSizes =
            SmallVector<int32_t>(iterSpaceRank, kDefaultTileSize);
      } else {
        auto firstOutputTensor = dyn_cast<nv_tensor_ir::TensorType>(*it);
        if (firstOutputTensor) {
          size_t rank = firstOutputTensor.getRank();
          kernelInfo.tileSizes = SmallVector<int32_t>(rank, kDefaultTileSize);
        } else {
          kernelInfo.tileSizes =
              SmallVector<int32_t>(iterSpaceRank, kDefaultTileSize);
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Computed optimal tile sizes: [";
      llvm::interleaveComma(kernelInfo.tileSizes, llvm::dbgs());
      llvm::dbgs() << "] for graph " << graphOp.getSymName() << "\n";
    });
  }

  // Persistence grid: runs for ALL tile-source branches (fixes the bug where
  // user-provided tiles skipped this via early return).
  if (conversionState.persistence == PersistenceMode::Static) {
    TileAnalyzer tileAnalyzer{GpuArchitecture{}};
    SmallVector<int64_t> tileSizeI64(kernelInfo.tileSizes.begin(),
                                     kernelInfo.tileSizes.end());

    int64_t persistentCtaCount =
        static_cast<int64_t>(conversionState.smCount) *
        static_cast<int64_t>(conversionState.occupancy);

    int64_t totalTiles = 0;
    MLIR_ASSIGN_OR_RETURN(auto gridSize, calculatePersistentGridSizeForGraph(
                                             tileAnalyzer, tileSizeI64, graphOp,
                                             persistentCtaCount, &totalTiles));

    conversionState.gridSizeX = gridSize.empty() ? 0 : gridSize[0];
    conversionState.totalTiles = totalTiles;

    LLVM_DEBUG({
      llvm::dbgs() << "Static persistence enabled:\n";
      llvm::dbgs() << "  gridSizeX: " << conversionState.gridSizeX << "\n";
      llvm::dbgs() << "  totalTiles: " << conversionState.totalTiles << "\n";
    });
  }
  return success();
}

/// Generates the operations to compute the number of tiles required to
/// iterate over for each dimension of the iteration space. This computation is
/// based on iteration space dimensions, the tile sizes and the expected virtual
/// threads of the grid configuration.
static LogicalResult generateTileCountComputation(
    ConversionPatternRewriter &rewriter,
    TensorToCudaTileConversionState &conversionState,
    TypeConverter::SignatureConversion &signatureConversion) {
  Location loc = conversionState.getGetTileBlockIdOp().getLoc();

  int64_t iterSpaceRank = conversionState.kernelInfo.getIterationSpaceRank();

  LLVM_DEBUG({
    llvm::dbgs() << "Input kernel dimension info:\n";
    llvm::dbgs() << llvm::indent(4) << "Tile sizes: ";
    llvm::interleaveComma(conversionState.kernelInfo.tileSizes, llvm::dbgs());
    llvm::dbgs() << "\n";
    llvm::dbgs() << llvm::indent(4) << "Grid virtual threads: "
                 << conversionState.threadToIterSpaceMapping << "\n";
  });

  auto i32TileTy = cuda_tile::TileType::get(/*shape=*/llvm::ArrayRef<int64_t>{},
                                            rewriter.getI32Type());

  // Compute tile counts for each iteration space.
  for (int32_t iterSpaceId : conversionState.getIterSpaceIds()) {
    IterSpaceDimInfo &dimInfo =
        conversionState.getIterSpaceDimInfo(iterSpaceId);
    const auto &iterSpaceDimLocations = dimInfo.iterSpaceDimLocations;

    // Initialize numTiles and tilerLoops if not already done.
    if (dimInfo.numTiles.empty()) {
      dimInfo.numTiles.assign(iterSpaceRank, nullptr);
      dimInfo.tilerLoops.assign(iterSpaceRank, nullptr);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Computing tile counts for iteration space "
                   << iterSpaceId << ":\n";
      llvm::dbgs() << llvm::indent(4) << "Dimension locations: ";
      llvm::interleaveComma(iterSpaceDimLocations, llvm::dbgs(),
                            [](const IterSpaceDimLocation &loc) {
                              llvm::dbgs() << "(" << loc.first << ", "
                                           << loc.second << ")";
                            });
      llvm::dbgs() << "\n";
    });

    // Iterate through all dimensions and compute the number of tiles for each.
    for (int dimIdx = 0; dimIdx < iterSpaceRank; ++dimIdx) {
      // Skip if numTiles for this dimension was already computed.
      if (dimInfo.numTiles[dimIdx]) {
        continue;
      }

      // Retrieve the tensor view and dimension index from the pre-computed
      // iteration space dimension locations for this iteration space.
      auto [tensorViewForDim, tensorDimIdx] = iterSpaceDimLocations[dimIdx];
      assert(tensorViewForDim && "Tensor view should have been identified");
      assert(tensorDimIdx >= 0 && "Tensor dimension index should be valid");
      (void)tensorDimIdx;

      // Create a partition view with the tile sizes for the tensor view
      // dimensions.
      CudaTileTensorDescriptor &tensorDescriptor =
          conversionState.getTensorViewDescriptor(tensorViewForDim);
      AffineMap iterSpaceMap = tensorDescriptor.getIterSpaceMap();
      auto partViewOp =
          buildPartitionView(rewriter, tensorDescriptor, conversionState);

      // Generate a get index space shape op that returns the number of tiles
      // for all the tensor view dimensions.
      size_t tensorRank = tensorDescriptor.getRank();
      SmallVector<Type> idxSpaceResTypes(tensorRank, i32TileTy);
      auto getIdxShapeOp = cuda_tile::GetIndexSpaceShapeOp::create(
          rewriter, loc, idxSpaceResTypes, partViewOp);

      LLVM_DEBUG({
        llvm::dbgs() << "Generated GetIndexSpaceShapeOp for iter space "
                     << iterSpaceId << ": " << getIdxShapeOp << "\n";
      });

      // Record the number of tiles for each tensor view dimension.
      for (int resIdx = 0, numResults = iterSpaceMap.getNumResults();
           resIdx < numResults; ++resIdx) {
        auto resultExpr = iterSpaceMap.getResult(resIdx);
        auto dimNumTiles = getIdxShapeOp.getResults()[resIdx];
        int64_t dimIdxFromMap = getIterSpaceDimIdx(resultExpr, resIdx);
        dimInfo.numTiles[dimIdxFromMap] = dimNumTiles;

        LLVM_DEBUG({
          llvm::dbgs() << llvm::indent(4) << "Iter space " << iterSpaceId
                       << ", dimension " << dimIdxFromMap << " has "
                       << dimNumTiles << " tiles\n";
        });
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Iter space " << iterSpaceId << " tile counts: ";
      llvm::interleaveComma(dimInfo.numTiles, llvm::dbgs());
      llvm::dbgs() << "\n";
    });
  }

  return success();
}

static LogicalResult
createLoadsForInputTensors(ConversionPatternRewriter &rewriter,
                           TensorToCudaTileConversionState &conversionState,
                           cuda_tile::EntryOp entryOp) {
  for (auto &tensorDescr : conversionState.getInputTensorDescriptors()) {
    auto loadRes = createLoad(rewriter, conversionState, tensorDescr);
    if (failed(loadRes)) {
      return entryOp.emitError(
          "Failed to create load for input tensor descriptor");
    }

    tensorDescr.setLoadedTile((*loadRes)[0]);
  }

  return success();
}

static LogicalResult
generateTilerLoopsForIterSpace(ConversionPatternRewriter &rewriter,
                               TensorToCudaTileConversionState &conversionState,
                               int32_t iterSpaceId, Location loc) {
  const auto &kernelInfo = conversionState.kernelInfo;
  const auto &threadMapping = conversionState.threadToIterSpaceMapping;

  IterSpaceDimInfo &dimInfo = conversionState.getIterSpaceDimInfo(iterSpaceId);

  OpBuilder::InsertPoint currentInsertPoint = rewriter.saveInsertionPoint();
  int64_t iterSpaceRank = kernelInfo.getIterationSpaceRank();

  LLVM_DEBUG({ llvm::dbgs() << "Generating tiler loop nest:\n"; });

  for (int64_t dimIdx = 0; dimIdx < iterSpaceRank; ++dimIdx) {
    // No tiling needed. Reusing existing block for insertion.
    if (!threadMapping.dimNeedsTiling(dimIdx)) {
      LLVM_DEBUG({
        llvm::dbgs() << llvm::indent(4) << "Dimension " << dimIdx
                     << ": no tiling needed\n"
                     << llvm::indent(8)
                     << "Insertion block: " << currentInsertPoint.getBlock()
                     << "\n";
      });

      conversionState.setInsertPointForIterSpaceDim(iterSpaceId, dimIdx,
                                                    currentInsertPoint);
      continue;
    }

    // Tiling needed. Generate the loop for this dimension.
    Value lb = cuda_tile::ConstantOp::create(
        rewriter, loc,
        cuda_tile::TileType::get(llvm::ArrayRef<int64_t>{},
                                 rewriter.getI32Type()),
        DenseIntElementsAttr::get(
            cuda_tile::TileType::get(llvm::ArrayRef<int64_t>{},
                                     rewriter.getI32Type()),
            0));
    // Use per-iteration-space numTiles for the upper bound.
    Value ub = dimInfo.numTiles[dimIdx];
    Value step = cuda_tile::ConstantOp::create(
        rewriter, loc,
        cuda_tile::TileType::get(llvm::ArrayRef<int64_t>{},
                                 rewriter.getI32Type()),
        DenseIntElementsAttr::get(
            cuda_tile::TileType::get(llvm::ArrayRef<int64_t>{},
                                     rewriter.getI32Type()),
            1));

    // Once a loop is created, it is not possible to change its number of
    // results/iter args. We assume that a tiler loop will always return a value
    // outside of the loop (reduction result) and generate a placeholder for it.
    // This avoids having to replace the loop to add new results/iter args. If
    // this assumption doesn't hold in the future, we should analize the graph
    // to pre-compute the number of required results at or before this point.
    auto placeholderResTy = cuda_tile::TileType::get(llvm::ArrayRef<int64_t>{},
                                                     rewriter.getI32Type());
    auto zeroAttr = DenseIntElementsAttr::get(
        cuda_tile::TileType::get(llvm::ArrayRef<int64_t>{},
                                 rewriter.getI32Type()),
        0);
    Value placeholderInitValue = cuda_tile::ConstantOp::create(
        rewriter, loc, placeholderResTy, zeroAttr);
    auto forOp = cuda_tile::ForOp::create(rewriter, loc, lb, ub, step,
                                          ValueRange(placeholderInitValue));

    // Create the loop terminator with the placeholder result value.
    Block *loopBody = forOp.getBody();
    rewriter.setInsertionPointToEnd(loopBody);
    auto iterArgs = loopBody->getArguments().drop_front();
    auto continueOp = cuda_tile::ContinueOp::create(rewriter, loc, iterArgs);

    // Store the tiler loop for this dimension in the per-iteration-space info.
    dimInfo.tilerLoops[dimIdx] = forOp;

    // Set the insertion point before the loop terminator.
    rewriter.setInsertionPoint(continueOp);
    currentInsertPoint = rewriter.saveInsertionPoint();

    LLVM_DEBUG({
      llvm::dbgs() << llvm::indent(4) << "Tiler loop for iter space "
                   << iterSpaceId << ", dimension " << dimIdx << " generated:\n"
                   << llvm::indent(8) << forOp << "\n"
                   << llvm::indent(8)
                   << "Induction variable: " << forOp.getInductionVar() << "\n"
                   << llvm::indent(8)
                   << "Insertion block: " << currentInsertPoint.getBlock()
                   << "\n";
    });

    conversionState.setInsertPointForIterSpaceDim(iterSpaceId, dimIdx,
                                                  currentInsertPoint);
  }

  return success();
}

/// Tile and fuse operations within the CudaTile FuncOp.
static LogicalResult generateTileAndFuseSkeleton(
    ConversionPatternRewriter &rewriter,
    TensorToCudaTileConversionState &conversionState,
    TypeConverter::SignatureConversion &signatureConversion,
    cuda_tile::EntryOp entryOp) {

  int64_t iterSpaceRank = conversionState.kernelInfo.getIterationSpaceRank();
  // Skip empty graph and signal that tiling information won't be available.
  if (iterSpaceRank == 0) {
    conversionState.kernelInfo.isTilingAvailable = false;
    return success();
  }

  Location loc = entryOp.getLoc();

  // Set up block ID and optionally set up persistence loop for all strategies
  assert(conversionState.persistenceStrategy &&
         "PersistenceStrategy must be set");
  conversionState.persistenceStrategy->setupBlockIdAndPersistence(
      rewriter, loc, conversionState);

  // Generate tile count computation to determine the number of tiles needed to
  // compute for each dimension of the iteration space.
  if (failed(generateTileCountComputation(rewriter, conversionState,
                                          signatureConversion))) {
    return entryOp.emitError("Failed to generate tile count computation");
  }

  // TODOs:
  // TODO: Generate tokens.

  // Generate tiler loop nests for each iteration space.
  for (int32_t iterSpaceId : conversionState.getIterSpaceIds()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Generating tiler loop nest for iteration space "
                   << iterSpaceId << "\n";
    });

    if (failed(generateTilerLoopsForIterSpace(rewriter, conversionState,
                                              iterSpaceId, loc))) {
      return entryOp.emitError("Failed to generate tiler loop nest for "
                               "iteration space ")
             << iterSpaceId;
    }
  }

  if (failed(createLoadsForInputTensors(rewriter, conversionState, entryOp))) {
    return entryOp.emitError("Failed to create loads for input tensors");
  }

  return success();
}

/// Convert TensorIR GraphOp to CudaTile FuncOp with tiled memory operations.
class GraphOpConversion
    : public TensorToCudaTileConversionPattern<nv_tensor_ir::GraphOp> {
public:
  using Base = TensorToCudaTileConversionPattern<nv_tensor_ir::GraphOp>;
  using Base::TensorToCudaTileConversionPattern;

  LogicalResult
  matchAndRewrite(nv_tensor_ir::GraphOp graphOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently only support single-block graphs.
    if (graphOp.getGraphBody().getBlocks().size() != 1) {
      return graphOp.emitError("Only GraphOp with a single block is supported. "
                               "Found ")
             << graphOp.getGraphBody().getBlocks().size() << " blocks";
    }

    // Get stride info from graph attributes for stride-aware signature
    // conversion. Also check which args have explicit stride attributes.
    auto argDescriptors = getTensorDescriptors(graphOp.getArgumentTypes(),
                                               graphOp.getAllArgAttrs());
    if (failed(argDescriptors)) {
      return graphOp.emitError(
          "Failed to get tensor descriptors for input tensors");
    }

    auto resDescriptors = getTensorDescriptors(graphOp.getResultTypes(),
                                               graphOp.getAllResultAttrs());
    if (failed(resDescriptors)) {
      return graphOp.emitError(
          "Failed to get tensor descriptors for output tensors");
    }

    // Convert the function signature using custom converter that expands tensor
    // arguments into non-aggregate descriptor arguments and moves outputs to
    // arguments. Use stride-aware overload for correct dynamic stride handling.
    // If stride is not explicit, strides are computed from sizes (no stride
    // args).
    auto origFuncType = graphOp.getFunctionType();
    TypeConverter::SignatureConversion signatureConversion(
        origFuncType.getNumInputs());
    auto convFuncType = convertGraphOpSignature(
        origFuncType, *getTypeConverter(), *argDescriptors, *resDescriptors,
        conversionState.uniformSignature, signatureConversion);

    if (!convFuncType) {
      return graphOp.emitError("Failed to convert graph signature");
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  Original function type: " << origFuncType << "\n";
      llvm::dbgs() << "  Converted function type: " << convFuncType << "\n";
    });

    // Initialize iteration-space info and tile sizes for this graph BEFORE
    // moving the region.
    MLIR_RETURN_IF_ERROR(
        computeKernelInfoAndTileSizes(graphOp, conversionState));

    LLVM_DEBUG({
      llvm::dbgs() << "Iteration space IDs in graph: [";
      llvm::interleaveComma(conversionState.getIterSpaceIds(), llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    // Extract attribute information from the graph BEFORE moving the region.
    ArrayAttr tensorIRArgAttrs = graphOp.getAllArgAttrs();
    ArrayAttr tensorIRResAttrs = graphOp.getAllResultAttrs();

    MLIRContext *ctx = rewriter.getContext();
    auto entryOpHint = tensor_to_cuda_tile::createEntryOptimizationHints(
        ctx, conversionState.numCTAs, conversionState.occupancy,
        conversionState.numWarps);

    auto entryOp = cuda_tile::EntryOp::create(
        rewriter, graphOp.getLoc(), graphOp.getName(),
        cast<FunctionType>(convFuncType),
        /*argAttrs=*/ArrayAttr{}, /*resAttrs=*/ArrayAttr{}, entryOpHint);

    // Move the graph region into the entry function.
    rewriter.inlineRegionBefore(graphOp.getRegion(), entryOp.getBody(),
                                entryOp.getBody().end());

    // Apply the signature conversion to update block arguments.
    Block *tensorIREntryBlock = &entryOp.getRegion().front();
    rewriter.applySignatureConversion(tensorIREntryBlock, signatureConversion,
                                      typeConverter);

    // Compute and store tensor descriptors for all function arguments.
    // This information will be used by other conversion patterns.
    auto descriptorInitStatus =
        conversionState.computeCudaTileInOutTensorDescriptors(
            graphOp, origFuncType, tensorIREntryBlock, entryOp,
            signatureConversion, tensorIRArgAttrs, tensorIRResAttrs);
    if (failed(descriptorInitStatus)) {
      return failure();
    }

    // Set insertion point to the beginning of the function body.
    OpBuilder::InsertionGuard guard(rewriter);
    Block *entryBlock = &entryOp.getBlocks().front();
    rewriter.setInsertionPointToStart(entryBlock);

    // For every tensor argument, generate a CUDA Tile tensor view and replace
    // all uses of the original tensor argument with the new tensor view.
    generateTensorViews(rewriter, conversionState, entryOp);

    // Map the iteration space dimensions to the corresponding dimension sizes
    // of the CUDA Tile tensor views.
    locateIterationSpaceDimSizes(conversionState);

    // Create persistence strategy based on persistence mode.
    // Guard: Only consider persistence when iteration space and dimensions are
    // valid. This prevents generating a persistence loop with stale or invalid
    // dimensions (e.g., when the graph has no iteration space or dimensions are
    // uninitialized).
    int64_t iterSpaceRank = conversionState.kernelInfo.tileSizes.size();
    bool hasValidIterationSpace = (iterSpaceRank > 0) &&
                                  (conversionState.totalTiles > 0) &&
                                  (conversionState.gridSizeX > 0);

    bool useStaticPersistence =
        hasValidIterationSpace &&
        (conversionState.persistence == PersistenceMode::Static) &&
        (conversionState.gridSizeX < conversionState.totalTiles);

    if (useStaticPersistence) {
      auto strategy = std::make_unique<StaticPersistenceStrategy>(
          conversionState.gridSizeX, conversionState.totalTiles,
          conversionState.smCount);
      conversionState.persistenceStrategy = std::move(strategy);
    } else {
      conversionState.persistenceStrategy =
          std::make_unique<DefaultPersistenceStrategy>();
    }

    if (failed(generateTileAndFuseSkeleton(rewriter, conversionState,
                                           signatureConversion, entryOp))) {
      return failure();
    }

    // Create return op using the persistence strategy
    // For persistent kernels, this creates it at the end of the entry block
    // For non-persistent kernels, it's created in ResultsOpConversion
    if (conversionState.persistenceStrategy->isPersistenceActive()) {
      conversionState.persistenceStrategy->createReturnOp(
          rewriter, entryOp.getLoc(), entryOp);
    }

    rewriter.replaceOp(graphOp, entryOp);
    return success();
  }
};

/// Convert TensorIR ResultsOp to CudaTile store operations.
class ResultsOpConversion
    : public TensorToCudaTileConversionPattern<nv_tensor_ir::ResultsOp> {
public:
  using Base = TensorToCudaTileConversionPattern<nv_tensor_ir::ResultsOp>;
  using Base::TensorToCudaTileConversionPattern;

  LogicalResult
  matchAndRewrite(nv_tensor_ir::ResultsOp resultsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Generate stores for all the output tensors.
    for (auto [tensorDescr, origOperand, adaptorOperand] :
         llvm::zip_equal(conversionState.getOutputTensorDescriptors(),
                         resultsOp.getOperands(), adaptor.getOperands())) {
      Value tileToStore = conversionState.getOperandTile(adaptorOperand);

      LLVM_DEBUG(
          { llvm::dbgs() << "Storing tensor: " << tileToStore << "\n"; });

      if (failed(createStore(rewriter, conversionState, tensorDescr,
                             tileToStore))) {
        return failure();
      }
    }

    // Use the persistence strategy to create the return op.
    // For persistent kernels, the return is already created in
    // GraphOpConversion. For non-persistent kernels, create it here at the
    // current insertion point.
    assert(conversionState.persistenceStrategy &&
           "PersistenceStrategy must be set");
    if (!conversionState.persistenceStrategy->isPersistenceActive()) {
      conversionState.persistenceStrategy->createReturnOp(
          rewriter, resultsOp.getLoc(), nullptr);
    }

    rewriter.eraseOp(resultsOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Broadcast Operation Conversion.
//===----------------------------------------------------------------------===//

/// Convert TensorIR BroadcastOp to CudaTile BroadcastOp.
class BroadcastOpConversion
    : public TensorToCudaTileConversionPattern<nv_tensor_ir::BroadcastOp> {
public:
  using Base = TensorToCudaTileConversionPattern<nv_tensor_ir::BroadcastOp>;
  using Base::TensorToCudaTileConversionPattern;

  LogicalResult
  matchAndRewrite(nv_tensor_ir::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    auto insertPoint = conversionState.getInsertPointForOp(op);
    rewriter.restoreInsertionPoint(insertPoint);

    // Get input tile.
    Value inputTile = conversionState.getOperandTile(adaptor.getInput());

    // Get the output tensor type and compute tile sizes from iteration space.
    auto resultType = cast<nv_tensor_ir::TensorType>(op.getType());

    auto &iterSpaceTileSizes = conversionState.kernelInfo.tileSizes;
    AffineMap outputIterSpaceMap =
        conversionState.getOutputIterSpaceMapForOperation(op);
    auto outputTileSizes =
        getTileSizesForIterSpaceMap(outputIterSpaceMap, iterSpaceTileSizes);

    // Convert TensorIR tensor type to CudaTile tile type with shape.
    Type outputTileType =
        cast<ShapedType>(getTypeConverter()->convertType(resultType))
            .clone(outputTileSizes);

    // Create the broadcast operation.
    rewriter.replaceOpWithNewOp<cuda_tile::BroadcastOp>(op, outputTileType,
                                                        inputTile);

    return success();
  }
};

static Value buildIdentityValue(ConversionPatternRewriter &rewriter,
                                Location loc,
                                cuda_tile::TileType accumulatorType) {
  Type elementType = accumulatorType.getElementType();
  Attribute scalarAttr;

  if (isa<FloatType>(elementType)) {
    scalarAttr =
        rewriter.getFloatAttr(cast<FloatType>(elementType), /*value=*/0.0);
  } else if (isa<IntegerType>(elementType)) {
    scalarAttr = rewriter.getIntegerAttr(elementType, /*value=*/0);
  } else {
    llvm_unreachable("Unsupported accumulator element type");
  }

  auto denseAttr = cast<DenseTypedElementsAttr>(
      DenseElementsAttr::get(accumulatorType, scalarAttr));
  return cuda_tile::ConstantOp::create(rewriter, loc, accumulatorType,
                                       denseAttr);
}

static Value initializeAccTileForReduction(ConversionPatternRewriter &rewriter,
                                           Location loc,
                                           cuda_tile::ForOp reductionForOp,
                                           cuda_tile::TileType accType) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(reductionForOp);
  Value newIdentityValue = buildIdentityValue(rewriter, loc, accType);
  assert(reductionForOp.getInitValues().size() == 1 &&
         "Expected exactly one initial value");
  reductionForOp.setOperand(3, newIdentityValue);

  Block *loopBody = reductionForOp.getBody();
  auto loopBodyArgs = loopBody->getArguments();
  assert(loopBodyArgs.size() == 2 &&
         "Expected induction variable and accumulator arguments");
  loopBodyArgs[1].setType(newIdentityValue.getType());
  reductionForOp.getResults()[0].setType(newIdentityValue.getType());

  Value accTile = loopBodyArgs[1];

  assert(accTile.getNumUses() == 1 &&
         "Expected only one use (continue op) for the accumulator tile");
  return accTile;
}

/// TODO: Codegen is WIP. Matmul is currently replaced with a constant op.
class MatmulOpConversion
    : public TensorToCudaTileConversionPattern<nv_tensor_ir::MatmulOp> {
public:
  using Base = TensorToCudaTileConversionPattern<nv_tensor_ir::MatmulOp>;
  using Base::TensorToCudaTileConversionPattern;

  LogicalResult
  matchAndRewrite(nv_tensor_ir::MatmulOp matmulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    auto insertPoint = conversionState.getInsertPointForOp(matmulOp);
    rewriter.restoreInsertionPoint(insertPoint);

    auto resultType = matmulOp.getResult().getType();
    bool isFloat = resultType.getElementType().isFloat();

    auto &iterSpaceTileSizes = conversionState.kernelInfo.tileSizes;
    AffineMap outputIterSpaceMap =
        conversionState.getOutputIterSpaceMapForOperation(matmulOp);
    auto outputTileSizes =
        getTileSizesForIterSpaceMap(outputIterSpaceMap, iterSpaceTileSizes);
    Type convResType =
        cast<ShapedType>(getTypeConverter()->convertType(resultType))
            .clone(outputTileSizes);

    Value aTile = conversionState.getOperandTile(adaptor.getA());
    Value bTile = conversionState.getOperandTile(adaptor.getB());

    // Update reduction loop.
    auto reductionForOp =
        cast<cuda_tile::ForOp>(insertPoint.getBlock()->getParentOp());
    Value accTile = initializeAccTileForReduction(
        rewriter, matmulOp.getLoc(), reductionForOp,
        cast<cuda_tile::TileType>(convResType));

    Value mmaResTile;
    if (isFloat) {
      mmaResTile = cuda_tile::MmaFOp::create(rewriter, matmulOp.getLoc(), aTile,
                                             bTile, accTile);
    } else {
      auto getSignedness = [](Type type) -> cuda_tile::Signedness {
        return type.isSignedInteger() ? cuda_tile::Signedness::Signed
                                      : cuda_tile::Signedness::Unsigned;
      };
      auto lhsSignedness =
          getSignedness(matmulOp.getA().getType().getElementType());
      auto rhsSignedness =
          getSignedness(matmulOp.getB().getType().getElementType());
      mmaResTile =
          cuda_tile::MmaIOp::create(rewriter, matmulOp.getLoc(), aTile, bTile,
                                    accTile, lhsSignedness, rhsSignedness);
    }

    // Manually replace uses to avoid conditional replacement in rollback mode
    // when using replaceAllUsesExcept.
    for (OpOperand &use : llvm::make_early_inc_range(accTile.getUses())) {
      if (use.getOwner() != mmaResTile.getDefiningOp()) {
        use.set(mmaResTile);
      }
    }

    // Propagate final mma result to the consumer outside the loop.
    rewriter.replaceOp(matmulOp, reductionForOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reduction Operations
//===----------------------------------------------------------------------===//

class ReduceOpConversion
    : public TensorToCudaTileConversionPattern<nv_tensor_ir::ReduceOp> {
public:
  using Base = TensorToCudaTileConversionPattern<nv_tensor_ir::ReduceOp>;
  using Base::TensorToCudaTileConversionPattern;

  LogicalResult
  matchAndRewrite(nv_tensor_ir::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get insertion point for the operation
    OpBuilder::InsertionGuard guard(rewriter);
    auto insertPoint = conversionState.getInsertPointForOp(op);
    rewriter.restoreInsertionPoint(insertPoint);

    // Get input and validate
    Value loadedInput = conversionState.getOperandTile(adaptor.getInput());
    auto inputTileType = dyn_cast<cuda_tile::TileType>(loadedInput.getType());
    if (!inputTileType) {
      return op.emitError("Expected cuda_tile::TileType for reduction input");
    }

    auto elementType = inputTileType.getElementType();
    tensor_to_cuda_tile::ReductionEmissionHelper reductionEmission{
        op.getReductionMode(), elementType};

    // Validate element type (currently only support floating point)
    if (!elementType.isFloat()) {
      return op.emitError(
          "Unsupported element type for reduction operation. Only floating "
          "point types are currently supported.");
    }

    // Only support single dimension reduction for now
    if (adaptor.getDimensions().size() != 1) {
      return op.emitError(
          "Multiple dimensions are not supported for reduction operation.");
    }

    // "customize" reduction mode is not supported.
    if (op.getReductionMode() == nv_tensor_ir::ReductionMode::customize) {
      return op.emitError("Customize reduction mode is not supported.");
    }

    int32_t reductionDim = adaptor.getDimensions()[0];

    auto identityAttr = reductionEmission.getIdentity(rewriter);
    auto identitiesArray = rewriter.getArrayAttr({identityAttr});

    // Pre-process input based on reduction mode
    Value reductionInput =
        reductionEmission.buildPrologue(rewriter, loadedInput);

    // Get input shape to compute output shape
    auto inputShape = inputTileType.getShape();
    SmallVector<int64_t> outputShape;
    outputShape.reserve(inputShape.size() - 1);
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (i != static_cast<size_t>(reductionDim)) {
        outputShape.push_back(inputShape[i]);
      }
    }

    auto outputTileType = cuda_tile::TileType::get(outputShape, elementType);

    // Create the reduce operation
    auto reduceOp = cuda_tile::ReduceOp::create(
        rewriter, op.getLoc(), outputTileType, reductionInput,
        rewriter.getI32IntegerAttr(reductionDim), identitiesArray);

    // Create the combine operation block
    SmallVector<Location> locs(2, op.getLoc());
    Block *combineBlock = rewriter.createBlock(
        &reduceOp.getBody(), /*insertPt=*/{},
        /*argTypes=*/
        TypeRange{
            cuda_tile::TileType::get(llvm::ArrayRef<int64_t>{}, elementType),
            cuda_tile::TileType::get(llvm::ArrayRef<int64_t>{}, elementType)},
        locs);

    // Add the combine operation in the block
    rewriter.setInsertionPointToEnd(combineBlock);
    Value combineResult = reductionEmission.buildReduction(
        rewriter, combineBlock->getArgument(0), combineBlock->getArgument(1));

    // Create the yield operation
    cuda_tile::YieldOp::create(rewriter, op.getLoc(), combineResult);

    // Post-process the result
    rewriter.setInsertionPointAfter(reduceOp);
    Value result = reductionEmission.buildEpilogue(
        rewriter, reduceOp.getResult(0), inputShape[reductionDim]);

    // Reshape the result to match expected output shape (add dimension of size
    // 1)
    SmallVector<int64_t> reshapeShape(outputShape.begin(), outputShape.end());
    reshapeShape.insert(reshapeShape.begin() + reductionDim, 1);
    auto reshapeType = cuda_tile::TileType::get(reshapeShape, elementType);

    rewriter.replaceOpWithNewOp<cuda_tile::ReshapeOp>(op, reshapeType, result);

    return success();
  }
};

class ConversionStateImpl : public tensor_to_cuda_tile::ConversionState {
public:
  ConversionStateImpl(MLIRContext *context, const TypeConverter &typeConverter,
                      TensorToCudaTileConversionState conversionState,
                      bool enableExperimentalCudaTileOps)
      : context(context), typeConverter(typeConverter),
        conversionState(std::move(conversionState)),
        enableExperimentalCudaTileOps(enableExperimentalCudaTileOps) {
    patterns = registerPatterns();
  }

  LogicalResult start(GraphOp graphOp) override {
    conversionState.clear();

    ConversionTarget target(*context);
    target.addLegalDialect<cuda_tile::CudaTileDialect>();
    LogicalResult result = applyFullConversion(graphOp, target, patterns);
    if (failed(result)) {
      return result;
    }

    resolvedTileSize.clear();
    resolvedTileSize.reserve(conversionState.kernelInfo.tileSizes.size());
    for (int32_t tile : conversionState.kernelInfo.tileSizes) {
      resolvedTileSize.push_back(tile);
    }
    return success();
  }

  LogicalResult update(ConversionPatternRewriter &rewriter,
                       Operation *op) override {
    // Set the insertion point for the operation.
    auto insertPoint = isa<ConstantOp, SplatOp>(op)
                           ? conversionState.getInsertPointForIterSpaceDim(
                                 /*iterSpaceId=*/0, /*iterSpaceDimIdx=*/0)
                           : conversionState.getInsertPointForOp(op);
    rewriter.restoreInsertionPoint(insertPoint);

    // Compute the tile shape for ops without tensor inputs.
    if (isa<ConstantOp, SplatOp>(op)) {
      // Get the tile shape from the iteration space map.
      tileShape = getTileSizesForIterSpaceMap(
          conversionState.getOutputIterSpaceMapForOperation(op),
          conversionState.kernelInfo.tileSizes);

      if (auto tensorType = dyn_cast<ShapedType>(op->getResult(0).getType())) {
        // Adjust tile sizes based on tensor shape - if a dimension is 1 in
        // the tensor shape, use tile size 1. This handles reduced dimensions
        // correctly.
        auto tensorShape = tensorType.getShape();
        for (size_t i = 0; i < tileShape.size() && i < tensorShape.size();
             ++i) {
          if (tensorShape[i] == 1) {
            tileShape[i] = 1;
          }
        }
      }
    } else {
      tileShape.clear();
    }
    return success();
  }

  Value getTile(Value operand) override {
    return conversionState.getOperandTile(operand);
  }

  ArrayRef<int64_t> getTileShape() override { return tileShape; }

  ArrayRef<int64_t> getResolvedTileSize() override { return resolvedTileSize; }

private:
  MLIRContext *context;
  const TypeConverter &typeConverter;
  TensorToCudaTileConversionState conversionState;
  bool enableExperimentalCudaTileOps;
  FrozenRewritePatternSet patterns;

  SmallVector<int64_t> tileShape;
  SmallVector<int64_t> resolvedTileSize;

  RewritePatternSet registerPatterns() {
    RewritePatternSet patterns(context);

    // Pointwise operations.
    registerPointwisePatterns(patterns, *this, typeConverter,
                              enableExperimentalCudaTileOps);

    // Function/kernel-level operations.
    patterns.add<GraphOpConversion>(conversionState, typeConverter, context);
    patterns.add<ResultsOpConversion>(conversionState, typeConverter, context);

    // Contraction operations.
    patterns.add<MatmulOpConversion>(conversionState, typeConverter, context);

    // Reduction operations
    patterns.add<ReduceOpConversion>(conversionState, typeConverter, context);

    // Constant and broadcast operations
    patterns.add<BroadcastOpConversion>(conversionState, typeConverter,
                                        context);

    return patterns;
  }
};

} // namespace

namespace mlir::nv_tensor_ir {
namespace tensor_to_cuda_tile {

std::unique_ptr<ConversionState>
createAffineMapConversionState(MLIRContext *context,
                               const TypeConverter &typeConverter,
                               const TensorToCudaTilePipelineOptions &options) {

  TensorToCudaTileConversionState state;
  state.tileSize.assign(options.tileSize.begin(), options.tileSize.end());
  state.numCTAs = options.numCTAs;
  state.occupancy = options.occupancy;
  state.numWarps = options.numWarps;
  state.smCount = options.smCount;
  state.uniformSignature = options.uniformSignature;
  state.persistence = options.persistence;

  bool enableExperimentalCudaTileOps = false;
  return std::make_unique<ConversionStateImpl>(
      context, typeConverter, std::move(state), enableExperimentalCudaTileOps);
}

} // namespace tensor_to_cuda_tile
} // namespace mlir::nv_tensor_ir
