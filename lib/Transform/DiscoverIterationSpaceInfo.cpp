// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- DiscoverIterationSpaceInfo.cpp - Discover Iteration Space ---------===//
//
// Implementation of the transformation pass that discovers iteration space
// structure for TensorIR operations and attaches iteration space attributes.
//
//===----------------------------------------------------------------------===//

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/Passes.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#include <mlir/IR/Iterators.h>

#define DEBUG_TYPE "discover-iteration-space-transform"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

namespace mlir::nv_tensor_ir {

// Include the generated pass definitions
#define GEN_PASS_DEF_DISCOVERITERATIONSPACEINFOPASS
#define GEN_PASS_DEF_REMOVEITERATIONSPACEMAPSPASS
#include "tensor_ir/Transform/Passes.h.inc"

} // namespace mlir::nv_tensor_ir

namespace {

//===----------------------------------------------------------------------===//
// IterationSpaceDiscoverer.
//===----------------------------------------------------------------------===//

/// Helper class that discovers iteration spaces and computes iteration space
/// attributes for TensorIR operations.
class IterationSpaceDiscoverer {
public:
  explicit IterationSpaceDiscoverer(GraphOp op) : graphOp(op) {}

  /// Discover the dimensions of the iteration space in the graph, including
  /// rank and dimension order.
  void discoverIterationSpaceDimensions();

  /// Discover iteration spaces and transitions in the graph.
  void discoverIterSpacesAndTransitions();

  /// Compute and propagate iteration space maps to operations.
  /// Maps are computed within the context of each iteration space.
  void computeAndPropagateIterationSpaceMaps();

  /// Refine dimension domains using iteration space maps.
  void refineDimensionDomains();

private:
  /// The graph being analyzed.
  GraphOp graphOp;

  /// Rank of the iteration space discovered.
  unsigned iterSpaceRank = 0;

  /// Current iteration space ID, incremented at each transition.
  unsigned currentIterSpaceId = 0;

  /// Propagate iteration space IDs and dimension domains. Domain states at this
  /// point are canonical (based on operation output rank, not iteration space
  /// rank).
  void propagateIterSpaceIdsAndCanonicalDimDomains();

  /// Backward and forward propagation phases of the data-flow algorithm.
  bool propagateBackwards();
  bool propagateForward();
};

void IterationSpaceDiscoverer::discoverIterationSpaceDimensions() {
  LLVM_DEBUG(llvm::dbgs() << "Discovering iteration space dimensions for "
                          << graphOp->getName() << "\n");
  assert(iterSpaceRank == 0 && "Iteration space dimensions already discovered");

  graphOp->walk([&](Operation *op) {
    auto iterSpaceOp = dyn_cast<IterationSpaceInfoInterface>(op);
    if (!iterSpaceOp) {
      return;
    }

    unsigned dims = iterSpaceOp.getNumIterSpaceDims();
    iterSpaceRank = std::max(iterSpaceRank, dims);

    LLVM_DEBUG(llvm::dbgs()
               << llvm::indent(2) << "Found operation " << *op << " with "
               << dims << " iteration space dimensions\n");
  });

  LLVM_DEBUG(llvm::dbgs() << "Rank of iteration space discovered: "
                          << iterSpaceRank << "\n");
}

/// Expand canonical dimension domain states (based on operation output rank) to
/// iteration space rank using an affine map.
static SmallVector<DimState>
expandCanonicalDimStatesToIterSpace(ArrayRef<DimState> canonicalStates,
                                    AffineMap iterSpaceMap,
                                    unsigned iterSpaceRank) {
  assert(iterSpaceMap && "expected valid iteration space map");

  SmallVector<DimState> expanded(iterSpaceRank, DimState::Undef);
  for (unsigned tensorDim = 0; tensorDim < iterSpaceMap.getNumResults();
       ++tensorDim) {
    unsigned iterDim = iterSpaceMap.getDimPosition(tensorDim);
    if (iterDim < iterSpaceRank && tensorDim < canonicalStates.size()) {
      expanded[iterDim] = canonicalStates[tensorDim];
    }
  }

  return expanded;
}

void IterationSpaceDiscoverer::propagateIterSpaceIdsAndCanonicalDimDomains() {
  // Map values to iteration space IDs and canonical (based on operation's
  // output rank, not iteration space rank) dimension domain states.
  DenseMap<Value, unsigned> valueToIterSpaceIds;
  DenseMap<Value, SmallVector<DimState>> valueToDimStates;

  // Initialize graph input dimension domains to Undef as arguments don't define
  // iteration space dimensions.
  for (auto arg : graphOp.getArguments()) {
    valueToIterSpaceIds[arg] = 0;
    auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(arg.getType());
    // Scalar graph arguments participate as values but have no tensor
    // dimensions to track.
    valueToDimStates[arg] = SmallVector<DimState>(
        tensorType ? tensorType.getRank() : 0, DimState::Undef);
  }

  // Walk operations to propagate iteration space IDs and domain states.
  graphOp->walk([&](Operation *op) {
    auto iterSpaceOp = dyn_cast<IterationSpaceInfoInterface>(op);
    if (!iterSpaceOp) {
      return;
    }

    // 1. Gather input iteration space ID and domain states from operands.
    // Use canonical input maps to expand operand states to iteration space
    // rank, then join (Def U Undef -> Def).
    unsigned inputIterSpaceId = 0;
    SmallVector<DimState> inputDimStates(iterSpaceRank, DimState::Undef);

    for (auto [operandIdx, operand] : llvm::enumerate(op->getOperands())) {
      auto idIt = valueToIterSpaceIds.find(operand);
      assert(idIt != valueToIterSpaceIds.end() && "operand not tracked");

      // Graph inputs have temporary id=0 that gets reassigned later based on
      // first user. Propagate the iteration space ID from operations only.
      if (!isa<BlockArgument>(operand)) {
        inputIterSpaceId = idIt->second;
      }

      // Get operand's canonical states.
      auto stateIt = valueToDimStates.find(operand);
      assert(stateIt != valueToDimStates.end() && "operand states not tracked");
      ArrayRef<DimState> operandCanonicalStates = stateIt->second;

      // Expand operand's canonical states to iteration space rank.
      AffineMap canonicalMap = iterSpaceOp.computeCanonicalInputIterSpaceMap(
          operandIdx, std::nullopt);
      SmallVector<DimState> expandedStates =
          expandCanonicalDimStatesToIterSpace(operandCanonicalStates,
                                              canonicalMap, iterSpaceRank);

      // Join dimension domain states: Def U Undef -> Def.
      for (unsigned i = 0; i < iterSpaceRank; ++i) {
        if (expandedStates[i] == DimState::Def) {
          inputDimStates[i] = DimState::Def;
        }
      }
    }

    // 2. Mark dimensions based on this operation's input shapes.
    for (auto [operandIdx, operand] : llvm::enumerate(op->getOperands())) {
      auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(operand.getType());
      if (!tensorType) {
        continue;
      }

      // For each tensor dim with non-unit size, mark the corresponding iter
      // space dim as Def.
      ArrayRef<int64_t> shape = tensorType.getShape();
      AffineMap inputMap = iterSpaceOp.computeCanonicalInputIterSpaceMap(
          operandIdx, std::nullopt);

      for (unsigned tensorDim = 0, numTensorDims = inputMap.getNumResults();
           tensorDim < numTensorDims; ++tensorDim) {
        if (tensorDim < shape.size() && shape[tensorDim] > 1) {
          unsigned iterDim = inputMap.getDimPosition(tensorDim);
          if (iterDim < iterSpaceRank) {
            inputDimStates[iterDim] = DimState::Def;
          }
        }
      }
    }

    // 3. Compute canonical output dimension domain states projecting input
    // states through the canonical output map to preserve previous dimension
    // domain definitions.
    SmallVector<DimState> canonicalDimStates =
        iterSpaceOp.getCanonicalOutputDimDomains();

    AffineMap outputMap =
        iterSpaceOp.computeCanonicalOutputIterSpaceMap(std::nullopt);
    assert(outputMap && "expected valid projected permutation map");
    unsigned outputRank = canonicalDimStates.size();
    for (unsigned tensorDim = 0; tensorDim < outputRank; ++tensorDim) {
      unsigned iterDim = outputMap.getDimPosition(tensorDim);
      if (iterDim < iterSpaceRank && inputDimStates[iterDim] == DimState::Def) {
        canonicalDimStates[tensorDim] = DimState::Def;
      }
    }

    // 4. Set the output iteration space ID.
    unsigned outputIterSpaceId = inputIterSpaceId;
    bool isTransition = iterSpaceOp.isIterationSpaceTransition(inputDimStates);

    if (isTransition) {
      // This operation creates a new iteration space.
      outputIterSpaceId = ++currentIterSpaceId;

      LLVM_DEBUG({
        llvm::dbgs() << "  Iteration space transition at " << *op << "\n";
        llvm::dbgs() << "    New iter space ID: " << outputIterSpaceId << "\n";
      });
    }

    // 5. Annotate the operation with the corresponding attributes and record
    // computed values for consumer ops.
    op->setAttr(
        TensorIRDialect::getIterSpaceDimDomainsAttrName(),
        IterSpaceDimDomainsAttr::get(op->getContext(), canonicalDimStates));
    op->setAttr(TensorIRDialect::getIterSpaceIdAttrName(),
                IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                 outputIterSpaceId));

    for (Value result : op->getResults()) {
      valueToIterSpaceIds[result] = outputIterSpaceId;
      valueToDimStates[result] = canonicalDimStates;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  " << op->getName()
                   << " iter_space_id: " << outputIterSpaceId << "\n";
    });
  });
}

void IterationSpaceDiscoverer::discoverIterSpacesAndTransitions() {
  if (iterSpaceRank == 0) {
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Discovering iteration space transitions...\n");

  // Propagate iteration space IDs and canonical domain states through
  // operations.
  propagateIterSpaceIdsAndCanonicalDimDomains();

  // Annotate input arguments with the iteration space IDs.
  // This tells codegen which tile sizes to use when loading each input.
  DenseMap<unsigned, llvm::SmallDenseSet<int32_t>> inputIterSpaceIds;

  graphOp->walk([&](IterationSpaceInfoInterface iterSpaceOp) {
    int32_t opIterSpaceId = iterSpaceOp.getIterSpaceId();
    for (Value operand : iterSpaceOp->getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        inputIterSpaceIds[blockArg.getArgNumber()].insert(opIterSpaceId);
      }
    }
  });

  // Annotate all the input arguments. Those not used by operations get
  // assigned to the first iteration space since they're only used in results,
  // at best.
  for (auto [idx, arg] : llvm::enumerate(graphOp.getArguments())) {
    auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(arg.getType());
    if (!tensorType) {
      continue;
    }

    SmallVector<int32_t> sortedIds;
    auto it = inputIterSpaceIds.find(idx);
    if (it != inputIterSpaceIds.end()) {
      sortedIds.assign(it->second.begin(), it->second.end());
      llvm::sort(sortedIds);
    } else {
      sortedIds.push_back(0);
    }

    SmallVector<DimState> canonicalDomains(tensorType.getRank(),
                                           DimState::Undef);

    graphOp.setArgAttr(idx, TensorIRDialect::getIterSpaceIdsAttrName(),
                       DenseI32ArrayAttr::get(graphOp.getContext(), sortedIds));
    graphOp.setArgAttr(
        idx, TensorIRDialect::getIterSpaceDimDomainsAttrName(),
        IterSpaceDimDomainsAttr::get(graphOp.getContext(), canonicalDomains));
    LLVM_DEBUG({
      llvm::dbgs() << "  Input " << idx << " iter_space_ids: [";
      llvm::interleaveComma(sortedIds, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Total iteration spaces discovered: "
                 << (currentIterSpaceId + 1) << "\n";
  });
}

void IterationSpaceDiscoverer::refineDimensionDomains() {
  if (iterSpaceRank == 0) {
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Refining dimension domains using iteration space "
                             "maps...\n");

  // Map values to their expanded iteration space states.
  DenseMap<Value, SmallVector<DimState>> valueToExpandedStates;

  // Initialize expanded states for graph arguments using their iteration space
  // maps.
  for (auto [idx, arg] : llvm::enumerate(graphOp.getArguments())) {
    // Only tensor arguments carry iteration-space attributes.
    if (!isa<nv_tensor_ir::TensorType>(arg.getType())) {
      continue;
    }

    auto mapAttr = graphOp.getArgAttrOfType<AffineMapAttr>(
        idx, TensorIRDialect::getIterSpaceMapAttrName());
    assert(mapAttr && "expected iter_space_map on argument");
    AffineMap argMap = mapAttr.getValue();

    // Get canonical states from annotation.
    auto domainsAttr = graphOp.getArgAttrOfType<IterSpaceDimDomainsAttr>(
        idx, TensorIRDialect::getIterSpaceDimDomainsAttrName());
    assert(domainsAttr && "expected iter_space_dim_domains on argument");
    SmallVector<DimState> canonicalStates(domainsAttr.getValue());

    // Expand to iteration space rank.
    SmallVector<DimState> expandedStates = expandCanonicalDimStatesToIterSpace(
        canonicalStates, argMap, iterSpaceRank);
    valueToExpandedStates[arg] = expandedStates;

    // Update argument annotation with expanded states.
    graphOp.setArgAttr(
        idx, TensorIRDialect::getIterSpaceDimDomainsAttrName(),
        IterSpaceDimDomainsAttr::get(graphOp.getContext(), expandedStates));
  }

  // Walk operations and expand canonical dim domains to iteration space rank.
  graphOp->walk([&](Operation *op) {
    auto iterSpaceOp = dyn_cast<IterationSpaceInfoInterface>(op);
    if (!iterSpaceOp) {
      return;
    }

    // Get current canonical dim domains from the operation.
    SmallVector<DimState> canonicalDimStates =
        iterSpaceOp.getIterSpaceDimDomains();

    // Get the computed output iteration space map.
    AffineMap outputMap = iterSpaceOp.getOutputIterSpaceMap();
    assert(outputMap && "expected output_iter_space_map");

    // Gather input expanded states by joining all operand states.
    SmallVector<DimState> inputExpandedStates(iterSpaceRank, DimState::Undef);
    for (Value operand : op->getOperands()) {
      auto it = valueToExpandedStates.find(operand);
      if (it != valueToExpandedStates.end()) {
        for (unsigned i = 0; i < iterSpaceRank; ++i) {
          if (it->second[i] == DimState::Def) {
            inputExpandedStates[i] = DimState::Def;
          }
        }
      }
    }

    // Expand canonical dim states to iteration space rank using the output map.
    SmallVector<DimState> expandedDimStates(iterSpaceRank, DimState::Undef);

    // First, propagate states from predecessors for all dimensions to make sure
    // we retrieve any lost dimensions (e.g., K in matmul or reshapes)
    for (unsigned i = 0; i < iterSpaceRank; ++i) {
      expandedDimStates[i] = inputExpandedStates[i];
    }

    // Then, check input shapes for dimensions this operation uses.
    for (auto [operandIdx, operand] : llvm::enumerate(op->getOperands())) {
      auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(operand.getType());
      if (!tensorType) {
        continue;
      }
      ArrayRef<int64_t> shape = tensorType.getShape();
      AffineMap inputMap = iterSpaceOp.computeCanonicalInputIterSpaceMap(
          operandIdx, iterSpaceRank);
      if (!inputMap) {
        continue;
      }

      // For each tensor dim, find which iter space dim it maps to.
      assert(inputMap.isProjectedPermutation() &&
             "expected projected permutation map");
      for (unsigned tensorDim = 0; tensorDim < inputMap.getNumResults();
           ++tensorDim) {
        if (tensorDim < shape.size() && shape[tensorDim] > 1) {
          unsigned iterDim = inputMap.getDimPosition(tensorDim);
          if (iterDim < iterSpaceRank) {
            expandedDimStates[iterDim] = DimState::Def;
          }
        }
      }
    }

    // Apply output canonical states using the output map and propagate lost
    // Def states in previous steps.
    assert(outputMap.isProjectedPermutation() &&
           "expected projected permutation map");
    for (unsigned tensorDim = 0; tensorDim < outputMap.getNumResults();
         ++tensorDim) {
      if (tensorDim < canonicalDimStates.size() &&
          canonicalDimStates[tensorDim] == DimState::Def) {
        unsigned iterDim = outputMap.getDimPosition(tensorDim);
        if (iterDim < iterSpaceRank) {
          expandedDimStates[iterDim] = DimState::Def;
        }
      }
    }

    // Update maps and attributes.
    for (Value result : op->getResults()) {
      valueToExpandedStates[result] = expandedDimStates;
    }

    op->setAttr(
        TensorIRDialect::getIterSpaceDimDomainsAttrName(),
        IterSpaceDimDomainsAttr::get(op->getContext(), expandedDimStates));

    LLVM_DEBUG({
      llvm::dbgs() << "  Expanded dim domains for " << op->getName() << "\n";
    });
  });
}

/// Given a GraphOp with no operations or enough information to compute
/// iteration space information, assign default identity maps, iteration
/// space IDs and dimension domains to any existing input arguments.
static void computeDefaultIterSpaceAttrsForEmptyGraphOp(GraphOp graphOp) {
  MLIRContext *ctx = graphOp.getContext();

  // Collect tensor arguments and find maximum rank in a single pass.
  SmallVector<std::pair<unsigned, int64_t>> tensorArgs;
  int64_t maxRank = 0;
  for (auto [idx, arg] : llvm::enumerate(graphOp.getArguments())) {
    if (auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(arg.getType())) {
      tensorArgs.emplace_back(idx, tensorType.getRank());
      maxRank = std::max(maxRank, tensorType.getRank());
    }
  }

  // Assign the default iteration space attributes to each tensor argument.
  SmallVector<DimState> undefDomains(maxRank, DimState::Undef);
  for (auto [argIdx, rank] : tensorArgs) {
    AffineMap map = AffineMap::getMinorIdentityMap(maxRank, rank, ctx);
    graphOp.setArgAttr(argIdx, TensorIRDialect::getIterSpaceMapAttrName(),
                       AffineMapAttr::get(map));
    graphOp.setArgAttr(argIdx, TensorIRDialect::getIterSpaceIdsAttrName(),
                       DenseI32ArrayAttr::get(ctx, /*iterSpaceIds=*/{0}));
    graphOp.setArgAttr(argIdx,
                       TensorIRDialect::getIterSpaceDimDomainsAttrName(),
                       IterSpaceDimDomainsAttr::get(ctx, undefDomains));
  }
}

void IterationSpaceDiscoverer::computeAndPropagateIterationSpaceMaps() {
  LLVM_DEBUG(llvm::dbgs() << "Starting iteration space map propagation\n");

  if (isEmptyRegionGraphOp(graphOp)) {
    computeDefaultIterSpaceAttrsForEmptyGraphOp(graphOp);
    return;
  }

  // TODO: Consider using the data flow analysis framework if control flow or
  // more sophisticated propagation is needed.

  bool changed;
  [[maybe_unused]] unsigned numIterations = 1;
  do {
    LLVM_DEBUG(llvm::dbgs() << "Iteration #" << numIterations << "\n");
    LLVM_DEBUG(llvm::dbgs() << "- Backward propagation -\n");
    changed = propagateBackwards();

    LLVM_DEBUG(llvm::dbgs() << "- Forward propagation -\n");
    changed |= propagateForward();
    ++numIterations;
  } while (changed);

  LLVM_DEBUG(llvm::dbgs() << "Iteration space map propagation converged in "
                          << numIterations << " iterations\n");
}

/// Propagate an affine map to a block argument or operation result value.
static void propagateMapToValue(AffineMap map, Value value) {
  if (auto defOp = value.getDefiningOp()) {
    if (auto iterSpaceOp = dyn_cast<IterationSpaceInfoInterface>(defOp)) {
      iterSpaceOp->setAttr(TensorIRDialect::getIterSpaceMapAttrName(),
                           AffineMapAttr::get(map));
    }
    return;
  }

  // Set attribute on block arguments.
  auto blockArg = cast<BlockArgument>(value);
  // Do not attach tensor iteration-space maps to scalar block arguments.
  if (!isa<nv_tensor_ir::TensorType>(blockArg.getType())) {
    return;
  }

  cast<GraphOp>(blockArg.getOwner()->getParentOp())
      .setArgAttr(blockArg.getArgNumber(),
                  TensorIRDialect::getIterSpaceMapAttrName(),
                  AffineMapAttr::get(map));
}

bool IterationSpaceDiscoverer::propagateBackwards() {
  bool changed = false;
  graphOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    auto iterSpaceOp = dyn_cast<IterationSpaceInfoInterface>(op);
    if (!iterSpaceOp) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << llvm::indent(2) << "Visiting operation '" << *op << "'\n");

    // Try to compute output iteration space map, if missing.
    auto outputMap = iterSpaceOp.getOutputIterSpaceMap();
    if (outputMap) {
      LLVM_DEBUG(llvm::dbgs() << llvm::indent(4)
                              << "Found map for output: " << outputMap << "\n");
    } else {
      outputMap = iterSpaceOp.computeCanonicalOutputIterSpaceMap(iterSpaceRank);
      if (!outputMap) {
        LLVM_DEBUG(
            llvm::dbgs()
            << llvm::indent(4)
            << "Couldn't compute canonical output iteration space map\n");
        return;
      }

      LLVM_DEBUG(llvm::dbgs() << llvm::indent(4)
                              << "[C] Canonical output iteration space map: "
                              << outputMap << "\n");

      op->setAttr(TensorIRDialect::getIterSpaceMapAttrName(),
                  AffineMapAttr::get(outputMap));
      changed = true;
    }

    // Propagate map to inputs.
    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      // Skip if input map was already computed.
      if (iterSpaceOp.getInputIterSpaceMap(i)) {
        continue;
      }

      auto inputMap = iterSpaceOp.inferInputFromOutputIterSpaceMap(i);
      propagateMapToValue(inputMap, operand);
      changed = true;

      LLVM_DEBUG(llvm::dbgs()
                 << llvm::indent(4)
                 << "[C] Inferred input iteration space map from output: "
                 << inputMap << "\n");
    }
  });

  return changed;
}

bool IterationSpaceDiscoverer::propagateForward() {
  bool changed = false;
  graphOp->walk<WalkOrder::PostOrder, ForwardIterator>([&](Operation *op) {
    auto iterSpaceOp = dyn_cast<IterationSpaceInfoInterface>(op);
    if (!iterSpaceOp) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << llvm::indent(2) << "Visiting operation '" << *op << "'\n");

    // Try to compute input iteration space maps, if missing.
    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      // Skip if input map was already computed.
      auto inputMap = iterSpaceOp.getInputIterSpaceMap(i);
      if (inputMap) {
        LLVM_DEBUG(llvm::dbgs() << llvm::indent(4) << "Found map for input "
                                << i << ": " << inputMap << "\n");
        continue;
      }

      inputMap =
          iterSpaceOp.computeCanonicalInputIterSpaceMap(i, iterSpaceRank);
      if (!inputMap) {
        LLVM_DEBUG(
            llvm::dbgs()
            << llvm::indent(4)
            << "Couldn't compute canonical input iteration space map for input "
            << i << "\n");
        continue;
      }

      propagateMapToValue(inputMap, operand);
      changed = true;
      LLVM_DEBUG(llvm::dbgs() << llvm::indent(4) << "[C] Canonical input " << i
                              << " iteration space map: "
                              << ": " << inputMap << "\n");
    }

    // Propagate map to outputs. Skip if output map was already computed.
    auto outputMap = iterSpaceOp.getOutputIterSpaceMap();
    if (outputMap) {
      LLVM_DEBUG(llvm::dbgs() << llvm::indent(4)
                              << "Found map for output: " << outputMap << "\n");
      return;
    }

    outputMap = iterSpaceOp.inferOutputFromInputIterSpaceMaps();
    if (!outputMap) {
      LLVM_DEBUG(llvm::dbgs()
                 << llvm::indent(4)
                 << "Couldn't infer output iteration space map from inputs\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << llvm::indent(4)
               << "[C] Inferred output iteration space map from inputs: "
               << outputMap << "\n");

    op->setAttr(TensorIRDialect::getIterSpaceMapAttrName(),
                AffineMapAttr::get(outputMap));
    changed = true;
  });

  return changed;
}

/// Verify that all the block arguments have an iteration space map assigned.
[[maybe_unused]] static LogicalResult verifyBlockArguments(GraphOp graphOp) {
  for (auto [i, arg] : llvm::enumerate(graphOp.getArguments())) {
    if (!isa<nv_tensor_ir::TensorType>(arg.getType())) {
      continue;
    }

    auto mapAttr =
        graphOp.getArgAttr(i, TensorIRDialect::getIterSpaceMapAttrName());
    if (!mapAttr) {
      return failure();
    }
  }
  return success();
}

/// Verify that all the IterationSpaceInfoInterface operations have an
/// iteration space map assigned.
[[maybe_unused]] static LogicalResult
verifyIterationSpaceMaps(GraphOp graphOp) {
  WalkResult walkResult = graphOp->walk([&](Operation *op) -> WalkResult {
    if (auto graphOp = dyn_cast<GraphOp>(op)) {
      if (failed(verifyBlockArguments(graphOp))) {
        graphOp->emitError()
            << "GraphOp has block arguments with iteration space maps\n";
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    }

    auto iterSpaceOp = dyn_cast<IterationSpaceInfoInterface>(op);
    if (!iterSpaceOp) {
      return WalkResult::advance();
    }

    auto outputMap = iterSpaceOp.getOutputIterSpaceMap();
    if (!outputMap) {
      op->emitError() << "Operation has no output iteration space map\n";
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return failure();
  }
  return success();
}

static LogicalResult rejectUnsupportedOps(GraphOp graphOp) {
  WalkResult walkResult = graphOp->walk([&](Operation *op) -> WalkResult {
    if (!isa<ConcatenateOp>(op)) {
      return WalkResult::advance();
    }
    op->emitError()
        << "concatenate is not supported by DiscoverIterationSpaceInfoPass";
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted()) {
    LLVM_DEBUG(llvm::dbgs() << "rejectUnsupportedOps: concatenate is not "
                               "supported by DiscoverIterationSpaceInfoPass\n");
    return failure();
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

namespace mlir::nv_tensor_ir {

/// Pass that discovers iteration space structure and attaches iteration space
/// maps to operations.
struct DiscoverIterationSpaceInfoPass
    : public impl::DiscoverIterationSpaceInfoPassBase<
          DiscoverIterationSpaceInfoPass> {

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    if (failed(rejectUnsupportedOps(graphOp))) {
      signalPassFailure();
      return;
    }
    IterationSpaceDiscoverer discoverer(graphOp);

    // 1. Discover the number of dimensions in each iteration space in the
    // graph.
    // TODO: Currently all the iteration spaces in supported graphs have
    // the same number of dimensions as broadcast and reductions are
    // rank-preserving operations in Tensor IR.
    discoverer.discoverIterationSpaceDimensions();

    // 2. Discover iteration spaces and transitions. We compute canonical
    // dimension domains based on operation output shapes. They will be refined
    // later once iteration space maps are computed.
    discoverer.discoverIterSpacesAndTransitions();

    // 3. Compute iteration space maps and propagate them forward and backwards.
    discoverer.computeAndPropagateIterationSpaceMaps();

    // 4. Refine dimension domains using iteration space maps.
    // TODO: We may need to also propagate Undef/Def for cases where
    // dimensions are dropped and created again. For example:
    //   * Matmul (drop K) + ... + reshape adding dim back + ...
    //   * More complex reshape cases.
    discoverer.refineDimensionDomains();

    // 5. Verify that all the ops have an iteration space attributes.
#ifndef NDEBUG
    if (failed(verifyIterationSpaceMaps(graphOp))) {
      graphOp->emitError()
          << "DiscoveryIterationSpaceInfoPass: verification failed";
      signalPassFailure();
    }
#endif
  }
};

/// Pass that removes iteration space map attributes from operations.
struct RemoveIterationSpaceMapsPass
    : public impl::RemoveIterationSpaceMapsPassBase<
          RemoveIterationSpaceMapsPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();

    funcOp->walk([](Operation *op) {
      op->removeAttr(TensorIRDialect::getIterSpaceMapAttrName());
    });

    for (unsigned i = 0, numArgs = funcOp.getNumArguments(); i < numArgs; ++i) {
      funcOp.removeArgAttr(i, TensorIRDialect::getIterSpaceMapAttrName());
    }
  }
};

} // namespace mlir::nv_tensor_ir
