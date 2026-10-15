// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/Tiling/Evaluate.h"

#include "tensor_ir/Analysis/LayoutPropagation.h"
#include "tensor_ir/Support/TCutegen.h"
#include "tensor_ir/Utils/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <utility>

#define DEBUG_TYPE "tiling-analysis-evaluate"

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {

/// Size value used for dynamic dimensions.
constexpr int64_t kDynamicSizeValue = 128;

/// ----- Layout analysis ------------------------------------------------------
///
/// The evaluator computes layout propagation and normalization on demand:
///
/// - `iterationSpace` is the input layout. It is a tree of layout sources
///   describing which tensor elements are loaded for each point of the main
///   iteration space. Tensor sources are the leaves (loads from graph inputs,
///   or constants that need no memory access). Composite sources group
///   independent loads, and concat, reduction and matmul sources start nested
///   iteration spaces over their operands.
/// - `resultViews` describe the stores to the graph outputs, one tensor source
///   per result, expressed in the main iteration space. They are only present
///   for multi-output graphs. For a single result, the output layout is built
///   from the result type and the output stride.
/// - `resultLayouts` are the per-result input layouts of multi-output graphs.
///   `iterationSpace` is their deduplicated union. For a single result, the
///   input layout is used instead.
///
/// The layouts only describe memory accesses, so the evaluator also needs the
/// operations behind the nested iteration spaces: the storage of a concat,
/// reduction or matmul tile depends on the values that are live while its
/// operands are computed. `findPivots` recovers this mapping by walking the
/// graph backwards from the results, carrying the layout of each value:
///
/// - Reshape, transpose, broadcast and slice are already folded into the
///   layouts, so the layout passes through to the operand unchanged.
/// - Concat, reduction and matmul operations are pivots. Their layout is
///   mapped to the operation, and its sources are split between the operands.
///   Operands discarded by a slice after a concat have no source and are
///   skipped.
/// - Pointwise operations with multiple operands have a forward-propagated
///   layout in the analysis, a composite with one source per distinct operand
///   layout. The normalized layout of the result has the same structure, so
///   each operand's forward sources are located in the forward composite and
///   replaced by the normalized sources at the same positions.
/// - Block arguments must have flat tensor sources, and constants must have
///   tensor sources.
///
/// Pivots are keyed by the (uniqued) layout attribute, so the evaluation
/// iterator can find the operation for every nested iteration space it visits.
/// A value can be reached with multiple layouts (e.g. directly and through a
/// transpose), and each value and layout pair is processed once. Any layout
/// that does not match the graph structure is reported as an error instead of
/// being evaluated.

/// Get the normalized result views or the physical layout of a single output.
LayoutSourceAttrInterface
getOutputLayoutInternal(const LayoutBoundary &boundary,
                        const LayoutNormalization &normalization) {
  ArrayRef<TensorSourceAttr> resultViews = normalization.resultViews;
  if (resultViews.empty()) {
    return boundary.outputs.front();
  }

  // Build a composite source for multiple results.
  if (resultViews.size() > 1) {
    SmallVector<LayoutSourceAttrInterface> sources(resultViews.begin(),
                                                   resultViews.end());
    return CompositeSourceAttr::get(boundary.op->getContext(), sources);
  }
  return resultViews.front();
}

/// Map concat layouts to the operands.
FailureOr<SmallVector<LayoutSourceAttrInterface>>
getConcatLayouts(ConcatSourceAttr concat, size_t numOperands) {
  SmallVector<LayoutSourceAttrInterface> layouts(concat.getSources());
  if (!concat.getArgumentIndex().empty()) {
    layouts = SmallVector<LayoutSourceAttrInterface>(numOperands, nullptr);
    for (auto [sourceIdx, operandIdx] :
         llvm::enumerate(concat.getArgumentIndex())) {
      layouts[operandIdx] = concat.getSource(sourceIdx);
    }
  } else if (layouts.size() != numOperands) {
    return failure();
  }
  return layouts;
}

/// Find the operations corresponding to the pivot layouts.
FailureOr<DenseMap<LayoutSourceAttrInterface, Operation *>>
findPivots(GraphOp graph, const LayoutPropagationAnalysis &propagation,
           const LayoutNormalization &normalization) {
  DenseMap<LayoutSourceAttrInterface, Operation *> pivots;
  DenseSet<std::pair<Value, LayoutSourceAttrInterface>> visited;

  // Recursively process all operations in the graph.
  std::function<LogicalResult(Value, LayoutSourceAttrInterface)> process =
      [&](Value value, LayoutSourceAttrInterface layout) -> LogicalResult {
    // Check if the value has already been processed.
    if (!visited.insert(std::make_pair(value, layout)).second) {
      return success();
    }

    // If the value is a block argument, the layout must be a tensor source.
    if (isa<BlockArgument>(value)) {
      if (auto tensorSource = dyn_cast<TensorSourceAttr>(layout);
          !tensorSource || tcutegen::depth(tensorSource.getCuteLayout()) > 1) {
        return graph.emitError() << "expected normalized layout";
      }
      return success();
    }

    // Skip the layout modification operations (already applied).
    Operation *op = value.getDefiningOp();
    if (isa<ReshapeOp, TransposeOp, BroadcastOp, SliceOp>(op)) {
      return process(op->getOperand(0), layout);
    }

    // Concatenation operations define new iteration spaces.
    if (isa<ConcatenateOp>(op)) {
      auto concatLayout = dyn_cast<ConcatSourceAttr>(layout);
      if (!concatLayout) {
        return op->emitError() << "expected concatenation layout";
      }
      pivots[layout] = op;
      auto operandLayouts =
          getConcatLayouts(concatLayout, op->getNumOperands());
      if (failed(operandLayouts)) {
        return op->emitError() << "invalid concatenation layout";
      }
      for (auto [opValue, opLayout] :
           llvm::zip_equal(op->getOperands(), *operandLayouts)) {
        if (opLayout && failed(process(opValue, opLayout))) {
          return failure();
        }
      }
      return success();
    }

    // Reduction operations define new iteration spaces.
    if (isa<ReduceOp, ReduceUDOp>(op)) {
      auto reduceLayout = dyn_cast<ReductionSourceAttr>(layout);
      if (!reduceLayout) {
        return op->emitError() << "expected reduction layout";
      }
      pivots[layout] = op;
      if (op->getNumOperands() == 1) {
        return process(op->getOperand(0), reduceLayout.getSource());
      }
      auto composite = dyn_cast<CompositeSourceAttr>(reduceLayout.getSource());
      if (!composite || composite.getSources().size() != op->getNumOperands()) {
        return op->emitError() << "expected composite reduction layout";
      }
      for (auto [opValue, opLayout] :
           llvm::zip_equal(op->getOperands(), composite.getSources())) {
        if (failed(process(opValue, opLayout))) {
          return failure();
        }
      }
      return success();
    }

    // Matmul operations define new iteration spaces.
    if (isa<MatmulOp>(op)) {
      auto matmulLayout = dyn_cast<MatmulSourceAttr>(layout);
      if (!matmulLayout) {
        return op->emitError() << "expected matmul layout";
      }
      pivots[layout] = op;
      if (failed(process(op->getOperand(0), matmulLayout.getLhs())) ||
          failed(process(op->getOperand(1), matmulLayout.getRhs()))) {
        return failure();
      }
      return success();
    }

    // Pointwise operations split the layout.
    if (isa<PointwiseOpInterface>(op)) {
      if (op->getNumOperands() == 1) {
        return process(op->getOperand(0), layout);
      }
      auto getSources = [](LayoutSourceAttrInterface src) {
        if (auto composite = dyn_cast_if_present<CompositeSourceAttr>(src)) {
          return SmallVector<LayoutSourceAttrInterface>(composite.getSources());
        }
        return SmallVector<LayoutSourceAttrInterface>(1, src);
      };
      auto forward = getSources(propagation.getLayout(value));
      auto backward = getSources(layout);
      if (forward.size() != backward.size()) {
        return op->emitError() << "composite layout size mismatch";
      }

      for (auto operand : op->getOperands()) {
        auto unmapped = getSources(propagation.getLayout(operand));
        SmallVector<LayoutSourceAttrInterface> mapped = llvm::to_vector(
            llvm::map_range(unmapped, [&](LayoutSourceAttrInterface src) {
              auto it = llvm::find(forward, src);
              return it == forward.end()
                         ? nullptr
                         : backward[std::distance(forward.begin(), it)];
            }));
        if (llvm::is_contained(mapped, nullptr)) {
          return op->emitError() << "composite layout mapping error";
        }
        auto next = mapped.size() == 1
                        ? mapped.front()
                        : CompositeSourceAttr::get(op->getContext(), mapped);
        if (failed(process(operand, next))) {
          return failure();
        }
      }
      return success();
    }

    // Constants do not have tensor operands.
    if (isa<ConstantOp, SplatOp, IotaOp>(op)) {
      if (!isa<TensorSourceAttr>(layout)) {
        return op->emitError() << "expected constant layout";
      }
      return success();
    }
    return op->emitError() << "unsupported operation";
  };

  // Use the per-result normalized layouts, or the single-result root.
  Operation *resultsOp = graph.getBody()->getTerminator();
  SmallVector<LayoutSourceAttrInterface> resultLayouts =
      normalization.resultLayouts;
  if (resultLayouts.empty()) {
    resultLayouts.push_back(normalization.iterationSpace);
  }

  // Process all output values.
  for (auto [result, layout] :
       llvm::zip_equal(resultsOp->getOperands(), resultLayouts)) {
    if (failed(process(result, layout))) {
      return failure();
    }
  }
  return pivots;
}

/// ----- Liveness analysis ----------------------------------------------------
///
/// The register storage of a tile is estimated as the number of tile elements
/// times the peak number of bytes that are live per tile element. Each tile
/// element is computed independently, so it is enough to track the element
/// types of the live values, not their tensor shapes.
///
/// The analysis works on the block order of the graph operations, which is
/// used as the schedule; operations are never reordered to reduce the register
/// pressure. A value is live from its definition to its last use, and block
/// arguments are live from the start of the block. Operations are assumed to
/// update their operands in place: operands at their last use are released
/// before the results are allocated, so a chain of unary operations only needs
/// the larger of its input and output sizes. An operand used multiple times by
/// the same operation is released once.
///
/// The live values are restricted to a single iteration space. Starting from
/// the root values of the iteration space, the values are collected backwards
/// through their defining operations, stopping at concat, reduction and matmul
/// operations. Their results belong to the outer iteration space, while their
/// operands are computed in nested iteration spaces with separate storage. The
/// evaluator calculates the live sizes of:
///
/// - the main iteration space, from the graph results;
/// - each concat and matmul operand, as the operands are tiled independently;
/// - all operands of a reduction together (ReduceUD has one iteration space).

/// Liveness range for a value: [start, end)
using LivenessRange = std::pair<int, int>;
using LivenessRangeMap = DenseMap<Value, LivenessRange>;
using IterationSpaceValues = DenseSet<Value>;

/// Calculate the liveness ranges for the given block.
LivenessRangeMap getLivenessRangeMap(Block *block) {
  LivenessRangeMap livenessRanges;
  for (auto arg : block->getArguments()) {
    livenessRanges[arg] = {-1, -1};
  }
  for (auto [idx, op] : llvm::enumerate(block->getOperations())) {
    for (auto operand : op.getOperands()) {
      livenessRanges[operand].second = idx;
    }
    for (auto result : op.getResults()) {
      livenessRanges[result] = {idx, idx};
    }
  }
  return livenessRanges;
}

/// Collect the values in the iteration space of the given inputs.
IterationSpaceValues getIterationSpaceValues(ValueRange initial) {
  IterationSpaceValues result;
  SmallVector<Value> worklist(initial.begin(), initial.end());
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (result.insert(current).second) {
      if (Operation *op = current.getDefiningOp()) {
        // Stop at iteration space boundary.
        if (isa<ConcatenateOp, ReduceOp, ReduceUDOp, MatmulOp>(op)) {
          continue;
        }
        for (auto operand : op->getOperands()) {
          worklist.push_back(operand);
        }
      }
    }
  }
  return result;
}

/// Calculate the storage needed for the element type.
int getSizeBytes(Type type) {
  return llvm::divideCeil(
      mlir::getElementTypeOrSelf(type).getIntOrFloatBitWidth(), 8);
}

/// Calculate the storage needed for one tile element in the iteration space.
int calculateMaxLiveSize(Block *block, const LivenessRangeMap &livenessRanges,
                         const IterationSpaceValues &activeValues) {
  // Mark active block arguments as live.
  int liveSize = 0;
  for (auto arg : block->getArguments()) {
    if (activeValues.contains(arg)) {
      liveSize += getSizeBytes(arg.getType());
    }
  }
  int maxLiveSize = liveSize;

  // Helper for getting the index of a value in a range.
  auto getIndex = [](ValueRange c, Value v) -> unsigned {
    return llvm::find(c, v) - c.begin();
  };

  // Process all operations in the block, while any values are live.
  // Assume the operations can update the values in-place.
  for (auto [idx, op] : llvm::enumerate(block->getOperations())) {
    // Remove values that are no longer live.
    // Make sure to process each value only once.
    for (auto [pos, operand] : llvm::enumerate(op.getOperands())) {
      if (auto it = livenessRanges.find(operand);
          it->second.second == static_cast<int>(idx) &&
          activeValues.contains(operand) &&
          getIndex(op.getOperands(), operand) == pos) {
        liveSize -= getSizeBytes(operand.getType());
      }
    }
    // Add values of the operation results.
    for (auto result : op.getResults()) {
      if (activeValues.contains(result)) {
        liveSize += getSizeBytes(result.getType());
      }
    }
    // Update the maximum live size.
    maxLiveSize = std::max(maxLiveSize, liveSize);
  }
  return maxLiveSize;
}

/// ----- Tiling iterator -------------------------------------------------------
///
/// The tile shape of the configuration is defined in the main iteration space.
/// This iterator keeps the tile shape of the layout being visited in sync while
/// descending into nested iteration spaces:
///
/// - A reduction tile keeps the tile sizes of the dimensions that the reduction
///   operand depends on (dimensions broadcast by the reduction view are
///   dropped), followed by the reduction tile over the reduced dimensions.
/// - A matmul tile is built separately for the LHS and RHS operands, using the
///   dimension maps to select either the output tile sizes or the reduction
///   tile sizes of the contracting dimensions.
///
/// The reduction tile size is split between the reduced (or contracting)
/// dimensions as evenly as possible in powers of two, and each dimension is
/// capped at its size rounded up to a power of two. The tile shape is swapped
/// in for the nested layouts and restored afterwards, so the iterator can be
/// reused for sibling layouts.

class TilingLayoutIterator : public LayoutIterator {
public:
  TilingLayoutIterator(TilingConfig config) : config(std::move(config)) {}

  LogicalResult processReduction(ReductionSourceAttr reduction) override;
  LogicalResult processMatmul(MatmulSourceAttr matmul) override;

protected:
  TilingConfig config;
};

/// Calculate the reduction tile shape by distributing the reduction tile size
/// evenly between the contracting dimensions.
SmallVector<int32_t>
calculateReductionTileShape(ArrayRef<int64_t> reductionShape,
                            int32_t reductionTileSize) {
  SmallVector<int32_t> tileShape;
  int rest = llvm::Log2_64(reductionTileSize);
  int dist = reductionShape.size();

  for (int64_t size : reductionShape) {
    int bits = std::min(llvm::Log2_64_Ceil(uint64_t(size)),
                        llvm::divideCeil(rest, dist--));
    rest -= bits;
    tileShape.push_back(1ull << bits);
  }
  return tileShape;
}

LogicalResult
TilingLayoutIterator::processReduction(ReductionSourceAttr reduction) {
  // Calculate the new tile shape.
  SmallVector<int32_t> newTileShape;
  auto cuteLayout = reduction.getCuteLayout();
  auto strideValues = cuteLayout.stride().getValues().drop_back();
  for (auto [idx, stride] : llvm::enumerate(strideValues)) {
    if (stride.as_int() > 0) {
      newTileShape.push_back(config.tileShape[idx]);
    }
  }
  SmallVector<int32_t> reductionTileShape = calculateReductionTileShape(
      reduction.getReductionShape(), config.reductionTileSize);
  newTileShape.append(reductionTileShape);

  // Swap and process the underlying layout.
  std::swap(newTileShape, config.tileShape);
  if (failed(dispatch(reduction.getSource()))) {
    return failure();
  }

  // Restore the tile shape.
  std::swap(newTileShape, config.tileShape);
  return success();
}

LogicalResult TilingLayoutIterator::processMatmul(MatmulSourceAttr matmul) {
  size_t rank = config.tileShape.size();
  SmallVector<int32_t> reductionTileShape = calculateReductionTileShape(
      matmul.getContractingShape(), config.reductionTileSize);

  // Get the dimension maps.
  auto lhsDimensionMap = matmul.getLhsDimensionMap();
  auto rhsDimensionMap = matmul.getRhsDimensionMap();
  if (failed(lhsDimensionMap) || failed(rhsDimensionMap)) {
    return failure();
  }

  // Calculate the LHS tile shape.
  SmallVector<int32_t> lhsTileShape;
  for (size_t idx : *lhsDimensionMap) {
    lhsTileShape.push_back(idx < rank ? config.tileShape[idx]
                                      : reductionTileShape[idx - rank]);
  }

  // Calculate the RHS tile shape.
  SmallVector<int32_t> rhsTileShape;
  for (size_t idx : *rhsDimensionMap) {
    rhsTileShape.push_back(idx < rank ? config.tileShape[idx]
                                      : reductionTileShape[idx - rank]);
  }

  // Process LHS/RHS underlying layouts.
  std::swap(lhsTileShape, config.tileShape);
  if (failed(dispatch(matmul.getLhs()))) {
    return failure();
  }
  std::swap(rhsTileShape, config.tileShape);
  if (failed(dispatch(matmul.getRhs()))) {
    return failure();
  }

  // Restore the tile shape.
  std::swap(lhsTileShape, config.tileShape);
  return success();
}

/// ----- Matmul evaluation ----------------------------------------------------
///
/// A matmul tile is described by its batch, M, N and K sizes, where K is the
/// reduction tile over the contracting dimensions. The evaluation iterator
/// tries the tensor core generations from the newest to the oldest, and uses
/// the first one that supports the element types, the tile size and the
/// architecture. If none applies, only the register storage of a plain
/// register implementation is reported.
///
/// All generations require equal floating-point operand types. One MMA
/// instruction consumes 32 bytes of K per row, so K must be a multiple of
/// `32 / elementSize` elements.
///
/// - MMAv5 requires M of 64 or 128 and N of at most 256 (multiple of 8). The
///   RHS tile is stored in shared memory, and the accumulator lives in tensor
///   memory, so it does not use registers. One instruction is counted per
///   K step and batch.
/// - MMAv3 requires M of 64 (one warpgroup) with the same N limits. The RHS
///   tile is stored in shared memory, and the accumulator is stored in
///   registers. One instruction is counted per K step and batch.
/// - MMAv2 requires M and N to be multiples of 16 and 8. The LHS, RHS and
///   accumulator tiles are stored in registers, and one instruction is counted
///   per 16x8 output fragment, K step and batch.
///
/// The shared memory tiles must fit into the shared memory of one SM. For MMAv5
/// and MMAv3, the RHS only needs registers if it is computed by the graph: an
/// RHS loaded directly from a graph input (through view operations only) is
/// copied to shared memory without passing through registers. The LHS storage
/// uses the live size of the LHS operand iteration space, since its operand
/// values must pass through registers.

struct MmaTileSize {
  int64_t b, m, n, k;
};

bool checkMmaTypes(MatmulOp matmul) {
  Type aType = matmul.getA().getType().getElementType();
  Type bType = matmul.getB().getType().getElementType();
  return aType == bType && aType.isFloat();
}

bool isGraphInputValue(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<ReshapeOp, TransposeOp, BroadcastOp, SliceOp>(op)) {
      return false;
    }
    value = op->getOperand(0);
  }
  return true;
}

std::optional<TilingEvaluation> evaluateV5(MatmulOp matmul,
                                           const ArchInfo &archInfo,
                                           const MmaTileSize &tile,
                                           ArrayRef<int> liveSizes) {
  // Verify the types and the tile size.
  int kSize = 32 / getSizeBytes(matmul.getA().getType());
  bool isValid = checkMmaTypes(matmul) &&
                 (archInfo.tensorCoreVersions & kMMAv5) != 0 &&
                 (tile.m == 64 || tile.m == 128) &&
                 (tile.n % 8 == 0 && tile.n <= 256) && (tile.k % kSize == 0);

  // RHS must be stored in shared memory.
  int64_t sharedMemorySize =
      tile.b * tile.n * tile.k * getSizeBytes(matmul.getB().getType());
  if (!isValid || sharedMemorySize > archInfo.maxSharedMemoryPerSM * 1024) {
    return std::nullopt;
  }

  // LHS uses registers, RHS uses registers if any operations are applied.
  int64_t storageSize = tile.b * tile.m * tile.k * liveSizes[0];
  if (!isGraphInputValue(matmul.getB())) {
    int64_t rhsSize = tile.b * tile.n * tile.k * liveSizes[1];
    storageSize = std::max(storageSize, rhsSize);
  }

  int64_t mmaCount = tile.b * tile.k / kSize;
  return TilingEvaluation{
      .registerStorageBytes = storageSize,
      .sharedStorageBytes = sharedMemorySize,
      .mmaOperationCount = {0, 0, mmaCount},
  };
}

std::optional<TilingEvaluation> evaluateV3(MatmulOp matmul,
                                           const ArchInfo &archInfo,
                                           const MmaTileSize &tile,
                                           ArrayRef<int> liveSizes) {
  // Verify the types and the tile size.
  int kSize = 32 / getSizeBytes(matmul.getA().getType());
  bool isValid = checkMmaTypes(matmul) &&
                 (archInfo.tensorCoreVersions & kMMAv3) != 0 &&
                 (tile.m == 64) && (tile.n % 8 == 0 && tile.n <= 256) &&
                 (tile.k % kSize == 0);

  // RHS should be stored in shared memory.
  int64_t sharedMemorySize =
      tile.b * tile.n * tile.k * getSizeBytes(matmul.getB().getType());
  if (!isValid || sharedMemorySize > archInfo.maxSharedMemoryPerSM * 1024) {
    return std::nullopt;
  }

  // LHS uses registers, RHS uses registers if any operations are applied.
  int64_t storageSize = tile.b * tile.m * tile.k * liveSizes[0];
  if (!isGraphInputValue(matmul.getB())) {
    int64_t rhsSize = tile.b * tile.n * tile.k * liveSizes[1];
    storageSize = std::max(storageSize, rhsSize);
  }
  // Accumulator is stored in registers, must be live.
  storageSize +=
      tile.b * tile.m * tile.n * getSizeBytes(matmul.getResult().getType());

  int64_t mmaCount = tile.b * tile.k / kSize;
  return TilingEvaluation{
      .registerStorageBytes = storageSize,
      .sharedStorageBytes = sharedMemorySize,
      .mmaOperationCount = {0, mmaCount, 0},
  };
}

int64_t getMatmulRegisterStorage(MatmulOp matmul, const MmaTileSize &tile) {
  // Total storage needed by the matmul operation using registers.
  int64_t size = tile.m * tile.k * getSizeBytes(matmul.getA().getType()) +
                 tile.n * tile.k * getSizeBytes(matmul.getB().getType()) +
                 tile.m * tile.n * getSizeBytes(matmul.getResult().getType());
  return tile.b * size;
}

std::optional<TilingEvaluation> evaluateV2(MatmulOp matmul,
                                           const ArchInfo &archInfo,
                                           const MmaTileSize &tile,
                                           ArrayRef<int> liveSizes) {
  // Verify the types and the tile size.
  int kSize = 32 / getSizeBytes(matmul.getA().getType());
  bool isValid =
      checkMmaTypes(matmul) && (archInfo.tensorCoreVersions & kMMAv2) != 0 &&
      (tile.m % 16 == 0) && (tile.n % 8 == 0) && (tile.k % kSize == 0);
  if (!isValid) {
    return std::nullopt;
  }

  // LHS, RHS and accumulator must be stored in registers.
  int64_t lhsStorageSize = tile.b * tile.m * tile.k * liveSizes[0];
  int64_t rhsStorageSize = tile.b * tile.n * tile.k * liveSizes[1];
  int64_t registerStorage = getMatmulRegisterStorage(matmul, tile);

  int64_t mmaCount = tile.b * (tile.m / 16) * (tile.n / 8) * (tile.k / kSize);
  return TilingEvaluation{
      .registerStorageBytes =
          std::max(std::max(lhsStorageSize, rhsStorageSize), registerStorage),
      .mmaOperationCount = {mmaCount, 0, 0},
  };
}

/// ----- Evaluation iterator --------------------------------------------------
///
/// The evaluation iterator estimates the cost of processing one tile (one
/// block) of the main iteration space. It visits the layout tree depth-first
/// and keeps a stack of partial evaluations: every visited layout pushes one
/// evaluation, and a parent layout replaces the evaluations of its children
/// with their combination.
///
/// Tensor sources are the memory accesses. Each one counts as one access, and
/// the number of cache lines is estimated from the CuTe layout clamped to the
/// tile shape:
///
/// - Broadcast dimensions (stride 0) are skipped.
/// - Dimensions with a stride below the cache line size are densely packed.
///   Together they form one contiguous run per combination of the other
///   dimensions, which may straddle one extra cache line if it is unaligned.
/// - Every step along a larger (or dynamic) stride starts a new run.
///
/// Tensor sources without a graph input or output (e.g. constants) cost
/// nothing. The other layouts combine the evaluations of their children:
///
/// - Composite: the children are accessed together, so their memory accesses,
///   loops and MMA operations are added.
/// - Concat: every block of the concat dimension is covered by exactly one
///   operand, so the costs are averaged over the operands, weighted by their
///   size along the concat dimension and rounded up.
/// - Reduction: the costs of the reduction operand are repeated for every step
///   of the reduction tile over the reduced dimensions, and the steps are
///   counted as loop iterations.
/// - Matmul: the costs of the LHS and RHS operands are repeated for every K
///   step, and the steps are counted as loop iterations. The tensor core
///   evaluation adds the MMA operations and the MMA storage.
///
/// Register and shared storage are peaks rather than sums, so the maximum over
/// the children is used. Concat, reduction and matmul layouts also account for
/// the storage of their operand iteration spaces: the tile size times the live
/// size per tile element, plus a result-typed accumulator when necessary.
///
/// The evaluator then adds the output stores and the storage of the main
/// iteration space. All counts except `blockCount` are per block.

class EvaluationLayoutIterator : public TilingLayoutIterator {
public:
  EvaluationLayoutIterator(TilingEvaluator &parent, TilingConfig config)
      : TilingLayoutIterator(std::move(config)), parent(parent) {}

  LogicalResult processTensor(TensorSourceAttr tensor) override;
  LogicalResult processComposite(CompositeSourceAttr composite) override;
  LogicalResult processConcat(ConcatSourceAttr concat) override;
  LogicalResult processReduction(ReductionSourceAttr reduction) override;
  LogicalResult processMatmul(MatmulSourceAttr matmul) override;

  // Must be called only after processing a layout.
  TilingEvaluation getResult() const { return evals.back(); }

private:
  TilingEvaluator &parent;
  SmallVector<TilingEvaluation> evals;
};

LogicalResult EvaluationLayoutIterator::processTensor(TensorSourceAttr tensor) {
  // Get the element type from the input or the output value.
  // Ignore missing tensor IDs (e.g. constant tensors).
  Type elementType;
  auto argTypes = parent.graph.getArgumentTypes();
  auto resTypes = parent.graph.getResultTypes();
  size_t id = tensor.getTensorId();
  if (id < argTypes.size()) {
    elementType = argTypes[id];
  } else if (id < argTypes.size() + resTypes.size()) {
    elementType = resTypes[id - argTypes.size()];
  } else {
    evals.push_back(TilingEvaluation{});
    return success();
  }

  // Verify that the CuTe layout can be evaluated as flat dimensions.
  auto cuteLayout = tensor.getCuteLayout();
  if (tcutegen::depth(cuteLayout) > 1) {
    return parent.graph.emitError() << "expected flat layout";
  }

  // Get the element type size (bytes).
  int elementSizeBytes = getSizeBytes(elementType);
  int elementsPerCacheLine = kCacheLineSize / elementSizeBytes;

  // If the offset is not aligned to the cache line boundary, an additional
  // cache line access may be needed.
  bool isUnalignedAccess =
      (tensor.getOffset() * elementSizeBytes) % kCacheLineSize != 0;

  // Calculate the number of cache line accessed for the current tile.
  int64_t cacheLineAccesses = 1;
  int64_t innerSpan = 0;
  for (auto [tileSize, dimLayout] :
       llvm::zip_equal(config.tileShape, cuteLayout)) {
    auto cgShape = dimLayout.shape();
    auto cgStride = dimLayout.stride();

    // Clamp the tile size to the dimension shape.
    int64_t maxSize =
        tcutegen::is_static(cgShape) ? cgShape.as_int() : INT64_MAX;
    int64_t actualSize = std::min<int64_t>(tileSize, maxSize);

    // Assume large stride value for dynamic strides.
    int64_t cacheLinesInDim = actualSize;
    if (tcutegen::is_static(cgStride)) {
      // Skip broadcasted dimensions.
      int64_t stride = cgStride.as_int();
      if (stride == 0) {
        continue;
      }
      if (stride < elementsPerCacheLine) {
        // If the accesses are not aligned to the cache line boundary, an
        // additional cache line access may be needed.
        int64_t accessSize = actualSize * stride;
        isUnalignedAccess |= accessSize % elementsPerCacheLine != 0 &&
                             elementsPerCacheLine % accessSize != 0;
        // Extend the contiguous span by this dimension's extent.
        innerSpan += accessSize - stride;
        cacheLinesInDim = 1;
      } else {
        // Each step along a large-stride dimension starts a new contiguous run.
        // Unless the stride is a multiple of the cache line size, these runs
        // start mid-line and may straddle an extra cache line.
        isUnalignedAccess |= stride % elementsPerCacheLine != 0;
      }
    }
    cacheLineAccesses *= cacheLinesInDim;
  }

  // Densely packed dimensions (stride below a cache line size) form a single
  // contiguous run per combination of the remaining dimensions. The run covers
  // `innerSpan` elements and may straddle one extra cache line when the access
  // is unaligned.
  if (innerSpan > 0) {
    cacheLineAccesses *= llvm::divideCeil(innerSpan + 1, elementsPerCacheLine) +
                         isUnalignedAccess;
  }

  // Add the evaluation result.
  evals.push_back(TilingEvaluation{
      .memoryAccessCount = 1,
      .memoryAccessCost = cacheLineAccesses,
  });
  return success();
}

LogicalResult
EvaluationLayoutIterator::processComposite(CompositeSourceAttr composite) {
  // Process the child layouts.
  size_t size = evals.size();
  if (failed(TilingLayoutIterator::processComposite(composite))) {
    return failure();
  }

  // Accumulate the results from the child evals.
  TilingEvaluation result;
  for (const auto &eval : llvm::make_range(evals.begin() + size, evals.end())) {
    result.registerStorageBytes =
        std::max(result.registerStorageBytes, eval.registerStorageBytes);
    result.sharedStorageBytes =
        std::max(result.sharedStorageBytes, eval.sharedStorageBytes);
    result.memoryAccessCount += eval.memoryAccessCount;
    result.memoryAccessCost += eval.memoryAccessCost;
    result.loopIterationCount += eval.loopIterationCount;
    for (int i = 0; i < 3; ++i) {
      result.mmaOperationCount[i] += eval.mmaOperationCount[i];
    }
  }

  // Replace the child evals with the accumulated result.
  evals.resize(size);
  evals.push_back(result);
  return success();
}

LogicalResult EvaluationLayoutIterator::processConcat(ConcatSourceAttr concat) {
  // Get the operation and the live sizes.
  Operation *concatOp = parent.pivots[concat];
  ArrayRef<int> operandLiveSizes = parent.liveSizes[concatOp];
  if (!concatOp) {
    return parent.graph.emitError() << "missing pivot: " << concat;
  }

  // Verify the tile shape (must have unit size for concat dimension).
  if (config.tileShape[concat.getDimension()] != 1) {
    return concatOp->emitError()
           << "expected unit tile size for concat dimension";
  }

  // Process the child layouts.
  size_t size = evals.size();
  if (failed(TilingLayoutIterator::processConcat(concat))) {
    return failure();
  }

  // Accumulate the results from the child evals.
  TilingEvaluation result;
  auto operandLayouts = getConcatLayouts(concat, concatOp->getNumOperands());
  size_t ptr = size;
  for (auto [idx, src] : llvm::enumerate(*operandLayouts)) {
    if (src == nullptr) {
      continue; // The operand is not used.
    }
    int64_t weight = src.getShape()[concat.getDimension()];
    const auto &eval = evals[ptr++];

    result.registerStorageBytes =
        std::max(result.registerStorageBytes, eval.registerStorageBytes);
    result.sharedStorageBytes =
        std::max(result.sharedStorageBytes, eval.sharedStorageBytes);
    result.memoryAccessCount += weight * eval.memoryAccessCount;
    result.memoryAccessCost += weight * eval.memoryAccessCost;
    result.loopIterationCount += weight * eval.loopIterationCount;
    for (int i = 0; i < 3; ++i) {
      result.mmaOperationCount[i] += weight * eval.mmaOperationCount[i];
    }

    // Calculate the total storage needed by the operand iteration space.
    int64_t storageBytes =
        llvm::product_of(config.tileShape, int64_t{1}) * operandLiveSizes[idx];
    result.registerStorageBytes =
        std::max(result.registerStorageBytes, storageBytes);
  }

  // Adjust the weighted results.
  int64_t totalWeight = concat.getShape()[concat.getDimension()];
  result.memoryAccessCount =
      llvm::divideCeil(result.memoryAccessCount, totalWeight);
  result.memoryAccessCost =
      llvm::divideCeil(result.memoryAccessCost, totalWeight);
  result.loopIterationCount =
      llvm::divideCeil(result.loopIterationCount, totalWeight);
  for (int i = 0; i < 3; ++i) {
    result.mmaOperationCount[i] =
        llvm::divideCeil(result.mmaOperationCount[i], totalWeight);
  }

  // Replace the child evals with the accumulated result.
  evals.resize(size);
  evals.push_back(result);
  return success();
}

LogicalResult
EvaluationLayoutIterator::processReduction(ReductionSourceAttr reduction) {
  // Get the operation and the live sizes.
  Operation *reduceOp = parent.pivots[reduction];
  ArrayRef<int> operandLiveSizes = parent.liveSizes[reduceOp];
  if (!reduceOp) {
    return parent.graph.emitError() << "missing pivot: " << reduction;
  }

  // Process the child layout.
  if (failed(TilingLayoutIterator::processReduction(reduction))) {
    return failure();
  }

  // Calculate the number of loop iterations.
  int64_t repeats = 1;
  SmallVector<int32_t> reductionTileShape = calculateReductionTileShape(
      reduction.getReductionShape(), config.reductionTileSize);
  for (auto [size, tileSize] :
       llvm::zip_equal(reduction.getReductionShape(), reductionTileShape)) {
    repeats *= llvm::divideCeil(size, tileSize);
  }

  // Update the evaluation result.
  TilingEvaluation &result = evals.back();
  result.memoryAccessCount *= repeats;
  result.memoryAccessCost *= repeats;
  if (repeats > 1) {
    result.loopIterationCount =
        repeats * std::max(result.loopIterationCount, 1l);
  }
  for (int i = 0; i < 3; ++i) {
    result.mmaOperationCount[i] *= repeats;
  }

  // Calculate the underlying tile size.
  int64_t tileSize = llvm::product_of(reductionTileShape);
  auto cuteLayout = reduction.getCuteLayout();
  auto strideValues = cuteLayout.stride().getValues().drop_back();
  for (auto [idx, stride] : llvm::enumerate(strideValues)) {
    tileSize *= stride.as_int() > 0 ? config.tileShape[idx] : 1;
  }

  // Calculate the total storage needed by the operand iteration space.
  int64_t storageBytes = tileSize * operandLiveSizes[0];
  if (repeats > 1) {
    int64_t accumulatorSize =
        llvm::sum_of(llvm::map_range(reduceOp->getResultTypes(), getSizeBytes));
    storageBytes += tileSize * accumulatorSize;
  }
  result.registerStorageBytes =
      std::max(result.registerStorageBytes, storageBytes);

  return success();
}

LogicalResult EvaluationLayoutIterator::processMatmul(MatmulSourceAttr matmul) {
  // Get the operation and the live sizes.
  Operation *matmulOp = parent.pivots[matmul];
  ArrayRef<int> operandLiveSizes = parent.liveSizes[matmulOp];
  if (!matmulOp) {
    return parent.graph.emitError() << "missing pivot: " << matmul;
  }

  // Process the child layout.
  if (failed(TilingLayoutIterator::processMatmul(matmul))) {
    return failure();
  }

  // Calculate the number of loop iterations.
  int64_t repeats = 1;
  SmallVector<int32_t> reductionTileShape = calculateReductionTileShape(
      matmul.getContractingShape(), config.reductionTileSize);
  for (auto [size, tileSize] :
       llvm::zip_equal(matmul.getContractingShape(), reductionTileShape)) {
    repeats *= llvm::divideCeil(size, tileSize);
  }

  // Calculate tile B/M/N/K sizes.
  MmaTileSize tile = {1, 1, 1, llvm::product_of(reductionTileShape)};
  auto cuteLayout = matmul.getCuteLayout();
  auto strideValues = cuteLayout.stride().getValues().drop_back();
  for (auto [idx, stride] : llvm::enumerate(strideValues)) {
    int64_t pos = stride.as_int();
    if (pos >= matmul.getM() * matmul.getN() * matmul.getK()) {
      tile.b *= config.tileShape[idx];
    } else if (pos >= matmul.getN() * matmul.getK()) {
      tile.m *= config.tileShape[idx];
    } else if (pos >= matmul.getK()) {
      tile.n *= config.tileShape[idx];
    }
  }

  // Evaluate the matmul operation.
  TilingEvaluation result;
  auto matmulCast = cast<MatmulOp>(matmulOp);
  if (auto eval =
          evaluateV5(matmulCast, parent.archInfo, tile, operandLiveSizes);
      eval.has_value()) {
    result = *eval;
  } else if (auto eval = evaluateV3(matmulCast, parent.archInfo, tile,
                                    operandLiveSizes);
             eval.has_value()) {
    result = *eval;
  } else if (auto eval = evaluateV2(matmulCast, parent.archInfo, tile,
                                    operandLiveSizes);
             eval.has_value()) {
    result = *eval;
  } else {
    int64_t lhsStorageSize = tile.b * tile.m * tile.k * operandLiveSizes[0];
    int64_t rhsStorageSize = tile.b * tile.n * tile.k * operandLiveSizes[1];
    int64_t registerStorage = getMatmulRegisterStorage(matmulCast, tile);
    result = TilingEvaluation{
        .registerStorageBytes =
            std::max(std::max(lhsStorageSize, rhsStorageSize), registerStorage),
    };
  }

  // Accumulate the results from the child evals.
  for (const auto &eval : llvm::make_range(evals.end() - 2, evals.end())) {
    result.registerStorageBytes =
        std::max(result.registerStorageBytes, eval.registerStorageBytes);
    result.sharedStorageBytes =
        std::max(result.sharedStorageBytes, eval.sharedStorageBytes);
    result.memoryAccessCount += repeats * eval.memoryAccessCount;
    result.memoryAccessCost += repeats * eval.memoryAccessCost;
    result.loopIterationCount += eval.loopIterationCount;
    for (int i = 0; i < 3; ++i) {
      result.mmaOperationCount[i] += repeats * eval.mmaOperationCount[i];
    }
  }
  if (repeats > 1) {
    result.loopIterationCount =
        repeats * std::max(result.loopIterationCount, 1l);
  }

  // Replace the child evals with the accumulated result.
  evals.resize(evals.size() - 2);
  evals.push_back(result);
  return success();
}

/// ----- Tiling configuration -------------------------------------------------
///
/// All tile sizes must be positive powers of two. The tile shape must not be
/// empty, and both the tile size (product of the tile shape) and the reduction
/// tile size are limited to 2^20 elements, which also keeps the storage
/// calculations from overflowing. The tile rank is checked against the
/// iteration space by the evaluator.

FailureOr<TilingConfig> TilingConfig::get(SmallVector<int32_t> tileShape,
                                          int32_t reductionTileSize) {
  constexpr int64_t maxTileSize = 1 << 20;
  int64_t acc = 1;
  auto isValid = [&](int64_t size) {
    return size > 0 && llvm::isPowerOf2_64(size) &&
           !llvm::MulOverflow(acc, size, acc) && acc <= maxTileSize;
  };
  if (tileShape.empty() || !llvm::all_of(tileShape, isValid) ||
      reductionTileSize <= 0 || !llvm::isPowerOf2_64(reductionTileSize) ||
      reductionTileSize > maxTileSize) {
    return failure();
  }
  return TilingConfig(std::move(tileShape), reductionTileSize);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TilingConfig &config) {
  os << "[";
  llvm::interleave(config.tileShape, os, ",");
  os << "]";
  if (config.reductionTileSize != 1) {
    os << " r" << config.reductionTileSize;
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TilingEvaluation &eval) {
  os << "blocks:" << eval.blockCount;
  if (eval.registerStorageBytes > 0) {
    os << " storage:" << eval.registerStorageBytes;
  }
  if (eval.sharedStorageBytes > 0) {
    os << " shared:" << eval.sharedStorageBytes;
  }
  if (eval.memoryAccessCount > 0) {
    os << " memAccess:" << eval.memoryAccessCount;
  }
  if (eval.memoryAccessCost > 0) {
    os << " memCost:" << eval.memoryAccessCost;
  }
  if (eval.loopIterationCount > 0) {
    os << " loops:" << eval.loopIterationCount;
  }
  if (eval.mmaOperationCount[0] > 0) {
    os << " mmaV2:" << eval.mmaOperationCount[0];
  }
  if (eval.mmaOperationCount[1] > 0) {
    os << " mmaV3:" << eval.mmaOperationCount[1];
  }
  if (eval.mmaOperationCount[2] > 0) {
    os << " mmaV5:" << eval.mmaOperationCount[2];
  }
  return os;
}

bool TilingEvaluation::operator<(const TilingEvaluation &other) const {
  // Total number of memory cache lines accessed is the main discriminator.
  int64_t score = blockCount * memoryAccessCost;
  int64_t otherScore = other.blockCount * other.memoryAccessCost;
  if (score != otherScore) {
    return score < otherScore;
  }
  // Prefer kernels with less loop iterations.
  if (loopIterationCount != other.loopIterationCount) {
    return loopIterationCount < other.loopIterationCount;
  }
  // Prefer kernels with fewer blocks.
  if (blockCount != other.blockCount) {
    return blockCount < other.blockCount;
  }
  // If MMA operations are present, prefer the higher MMA version.
  for (int i = 2; i >= 0; --i) {
    bool hasMma = mmaOperationCount[i] > 0;
    bool otherHasMma = other.mmaOperationCount[i] > 0;
    if (hasMma != otherHasMma) {
      return hasMma;
    }
  }
  // Storage estimation is inaccurate, use it as a tie-breaker only.
  if (sharedStorageBytes != other.sharedStorageBytes) {
    return sharedStorageBytes < other.sharedStorageBytes;
  }
  if (registerStorageBytes != other.registerStorageBytes) {
    return registerStorageBytes < other.registerStorageBytes;
  }
  // Memory access count is dependent on the loop iterations count.
  return memoryAccessCount < other.memoryAccessCount;
}

/// ----- Tiling evaluator -----------------------------------------------------
///
/// The evaluator is split into a per-graph setup and a per-configuration
/// evaluation, so that many tiling configurations of the same graph can be
/// compared cheaply.
///
/// `create` collects everything that does not depend on the configuration:
///
/// - The input layout (`iterationSpace`) and the output layout (the result
///   views, or a tensor source for a single result). The output layout is
///   reshaped to the shape of the input layout, so that one tile shape applies
///   to both. The reshape fails for incompatible shapes, and a hierarchical
///   reshaped layout is rejected during the evaluation.
/// - The pivots, which map the concat, reduction and matmul layouts to their
///   operations.
/// - The live sizes per tile element: one for the main iteration space, keyed
///   by the graph terminator, and one per operand iteration space of every
///   pivot operation (per operand for concats and matmuls, one for all
///   operands of a reduction). An operation reached through multiple layouts
///   is analyzed once.
///
/// `evaluate` rejects invalid configurations and tile shapes whose rank does
/// not match the iteration space without emitting a diagnostic, so callers can
/// probe configurations freely. Layout errors found during the evaluation are
/// reported as diagnostics. A valid configuration is evaluated in four steps:
///
/// 1. The input layout gives the loads, the loop iterations, the MMA
///    operations and the storage of the nested iteration spaces.
/// 2. The output layout adds the stores. Only its memory costs are used.
/// 3. The storage of the main iteration space is the tile size times its live
///    size, and the register storage is the maximum over all iteration spaces.
/// 4. The block count is the number of tiles covering the iteration space,
///    assuming `kDynamicSizeValue` elements for dynamic dimensions.

FailureOr<TilingEvaluator> TilingEvaluator::create(GraphOp graph,
                                                   const ArchInfo &archInfo) {
  MLIR_ASSIGN_OR_RETURN(auto boundary, LayoutBoundary::get(graph));
  MLIR_ASSIGN_OR_RETURN(auto propagation,
                        LayoutPropagationAnalysis::compute(boundary));
  MLIR_ASSIGN_OR_RETURN(auto normalization,
                        LayoutNormalization::compute(propagation));

  // Get the input layout.
  auto inputLayout = normalization.iterationSpace;
  LLVM_DEBUG(llvm::dbgs() << "Input layout: " << inputLayout << "\n");
  if (!inputLayout) {
    return graph.emitError("Failed to get the input layout");
  }

  // Get the output layout.
  auto outputLayout = getOutputLayoutInternal(boundary, normalization);
  LLVM_DEBUG(llvm::dbgs() << "Output layout: " << outputLayout << "\n");
  if (!outputLayout) {
    return graph.emitError("Failed to get the output layout");
  }

  // Align the output layout to the input layout.
  if (outputLayout.getShape() != inputLayout.getShape()) {
    outputLayout = outputLayout.reshape(inputLayout.getShape());
    LLVM_DEBUG(llvm::dbgs() << "Reshaped output: " << outputLayout << "\n");
    if (!outputLayout) {
      return graph.emitError("Failed to reshape the output layout");
    }
  }

  // Find pivots (connect layouts to operations).
  auto pivots = findPivots(graph, propagation, normalization);
  if (failed(pivots)) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Iteration space pivots: " << pivots->size()
                          << "\n");

  // Calculate the live sizes for the main iteration space.
  DenseMap<Operation *, SmallVector<int>> liveSizes;
  LivenessRangeMap livenessRanges = getLivenessRangeMap(graph.getBody());
  Operation *resultsOp = graph.getBody()->getTerminator();
  int mainLiveSize =
      calculateMaxLiveSize(graph.getBody(), livenessRanges,
                           getIterationSpaceValues(resultsOp->getOperands()));
  liveSizes.insert({resultsOp, {mainLiveSize}});
  LLVM_DEBUG(
      llvm::dbgs() << "Each tile element in the main iteration space needs "
                   << mainLiveSize << " bytes of storage\n");

  // For operations that define new iteration spaces, calculate the live sizes.
  for (auto [_, op] : *pivots) {
    if (liveSizes.contains(op)) {
      continue;
    }
    SmallVector<int> operandLiveSizes;
    if (isa<ConcatenateOp, MatmulOp>(op)) {
      for (auto operand : op->getOperands()) {
        operandLiveSizes.push_back(calculateMaxLiveSize(
            graph.getBody(), livenessRanges, getIterationSpaceValues(operand)));
      }
    } else {
      operandLiveSizes.push_back(
          calculateMaxLiveSize(graph.getBody(), livenessRanges,
                               getIterationSpaceValues(op->getOperands())));
    }
    LLVM_DEBUG({
      llvm::dbgs() << op->getName() << " storage: ";
      llvm::interleaveComma(operandLiveSizes, llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    liveSizes.insert({op, std::move(operandLiveSizes)});
  }

  // Create the tiling evaluator.
  return TilingEvaluator(graph, archInfo, inputLayout, outputLayout,
                         std::move(*pivots), std::move(liveSizes));
}

FailureOr<TilingEvaluation>
TilingEvaluator::evaluate(const TilingConfig &config) {
  // Verify the tiling configuration.
  if (config.getTileShape().size() != inputLayout.getShape().size()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Invalid tiling configuration: " << config << "\n");
    return failure();
  }

  // Get the evaluation result for the input layout.
  EvaluationLayoutIterator iterator(*this, config);
  if (failed(iterator.dispatch(inputLayout))) {
    return failure();
  }
  TilingEvaluation result = iterator.getResult();

  // Get the evaluation result for the output layout.
  if (failed(iterator.dispatch(outputLayout))) {
    return failure();
  }
  TilingEvaluation output = iterator.getResult();
  result.memoryAccessCount += output.memoryAccessCount;
  result.memoryAccessCost += output.memoryAccessCost;

  // Calculate the total storage needed by the main iteration space.
  int64_t mainLiveSize = liveSizes[graph.getBody()->getTerminator()][0];
  int64_t storageBytes =
      llvm::product_of(config.getTileShape(), int64_t{1}) * mainLiveSize;
  result.registerStorageBytes =
      std::max(result.registerStorageBytes, storageBytes);

  // Calculate the number of execution blocks.
  for (auto [size, tileSize] :
       llvm::zip_equal(inputLayout.getShape(), config.getTileShape())) {
    if (size == ShapedType::kDynamic) {
      size = kDynamicSizeValue;
    }
    result.blockCount *= llvm::divideCeil(size, tileSize);
  }

  return result;
}

/// ----- Layout iterator ------------------------------------------------------
///
/// Base visitor over the layout source tree. `dispatch` calls the handler for
/// the layout kind, and the default handlers visit the child layouts in order,
/// stopping at the first failure. Subclasses override the handlers to track
/// the tile shape or to accumulate the evaluation results.

LogicalResult LayoutIterator::dispatch(LayoutSourceAttrInterface layout) {
  return llvm::TypeSwitch<LayoutSourceAttrInterface, LogicalResult>(layout)
      .Case<TensorSourceAttr>(
          [&](auto tensor) { return processTensor(tensor); })
      .Case<CompositeSourceAttr>(
          [&](auto composite) { return processComposite(composite); })
      .Case<ConcatSourceAttr>(
          [&](auto concat) { return processConcat(concat); })
      .Case<ReductionSourceAttr>(
          [&](auto reduction) { return processReduction(reduction); })
      .Case<MatmulSourceAttr>(
          [&](auto matmul) { return processMatmul(matmul); })
      .Default(failure());
}

LogicalResult LayoutIterator::processTensor(TensorSourceAttr tensor) {
  return success();
}

LogicalResult LayoutIterator::processComposite(CompositeSourceAttr composite) {
  if (llvm::any_of(composite.getSources(),
                   [&](auto source) { return failed(dispatch(source)); })) {
    return failure();
  }
  return success();
}

LogicalResult LayoutIterator::processConcat(ConcatSourceAttr concat) {
  if (llvm::any_of(concat.getSources(),
                   [&](auto source) { return failed(dispatch(source)); })) {
    return failure();
  }
  return success();
}

LogicalResult LayoutIterator::processReduction(ReductionSourceAttr reduction) {
  return dispatch(reduction.getSource());
}

LogicalResult LayoutIterator::processMatmul(MatmulSourceAttr matmul) {
  if (failed(dispatch(matmul.getLhs())) || failed(dispatch(matmul.getRhs()))) {
    return failure();
  }
  return success();
}

} // namespace tiling_analysis
} // namespace mlir::nv_tensor_ir
