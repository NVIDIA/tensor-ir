// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_ANALYSIS_TILING_EVALUATE_H_
#define TENSOR_IR_ANALYSIS_TILING_EVALUATE_H_

#include "tensor_ir/Analysis/Tiling/Arch.h"
#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/Support/LLVM.h"

#include <array>
#include <cstdint>
#include <utility>

/// Tiling evaluator: an analytical cost model for tiling a TensorIR graph.
///
/// A graph is executed as a grid of blocks, where each block computes one tile
/// of the main iteration space. The evaluator estimates the per-block costs of
/// a tiling configuration, so that configurations can be ranked without
/// generating code. It does not produce timings; the results are only
/// meaningful relative to other configurations.
///
/// The evaluator computes layout propagation and normalization from the graph
/// when it is created. The resulting layouts describe memory accesses as a tree
/// of tensor, composite, concat, reduction and matmul sources.
///
/// A tiling configuration consists of:
///
/// - The tile shape, with one power-of-two size per dimension of the main
///   iteration space. Reduced and contracting dimensions are not part of the
///   iteration space. Concat dimensions must have unit tile size.
/// - The reduction tile size, the number of elements of the reduced (or
///   contracting) dimensions processed per loop iteration. It is distributed
///   between these dimensions.
///
/// The evaluation estimates:
///
/// - The number of blocks covering the iteration space.
/// - The register and shared memory storage, as the peak over the iteration
///   spaces. The register storage assumes that every live value of a tile is
///   kept in registers.
/// - The number of memory accesses and accessed sectors (loads and stores),
///   the number of loop iterations over the reduction tiles, and the number of
///   tensor core operations per MMA generation.
///
/// Typical use:
///
///   auto evaluator = TilingEvaluator::create(graph, archInfo);
///   if (failed(evaluator)) return failure();
///   for (const TilingConfig &config : candidates) {
///     if (auto eval = evaluator->evaluate(config); succeeded(eval)) { ... }
///   }
///
/// `create` fails with a diagnostic when layouts cannot be derived or do not
/// match the graph. `evaluate` fails silently for invalid configurations, and
/// with a diagnostic for layouts that cannot be evaluated with the given tile
/// shape (e.g. non-unit concat dimensions).

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {

/**
 * @brief Tiling configuration.
 */
class TilingConfig {
public:
  /// Returns the tiling config, if the tile sizes are valid.
  static FailureOr<TilingConfig> get(SmallVector<int32_t> tileShape,
                                     int32_t reductionTileSize);

  ArrayRef<int32_t> getTileShape() const { return tileShape; }
  int32_t getReductionTileSize() const { return reductionTileSize; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const TilingConfig &config);

private:
  TilingConfig(SmallVector<int32_t> tileShape, int32_t reductionTileSize)
      : tileShape(std::move(tileShape)), reductionTileSize(reductionTileSize) {}

  /// Tile shape for the main iteration space, for example [16,32,64]
  SmallVector<int32_t> tileShape;
  /// Reduction tile size for reduction/matmul operations.
  int32_t reductionTileSize = 1;

  friend class TilingLayoutIterator;
  friend class EvaluationLayoutIterator;
};

/**
 * @brief Tiling evaluation result.
 */
struct TilingEvaluation {
  /// Number of execution blocks.
  int64_t blockCount = 1;

  /// Estimate register storage size for the tiles.
  int64_t registerStorageBytes = 0;

  /// Estimate shared memory storage size for the tiles.
  int64_t sharedStorageBytes = 0;

  /// Number of memory accesses (loads/stores).
  int64_t memoryAccessCount = 0;

  /// Estimate cost of memory accesses (~number of sectors).
  int64_t memoryAccessCost = 0;

  /// Number of loop iterations.
  int64_t loopIterationCount = 0;

  /// Number of TensorCore MMA operations (v2/v3/v5).
  std::array<int64_t, 3> mmaOperationCount = {0, 0, 0};

  /// Returns true if this tiling is expected to run faster than `other`.
  /// The total memory traffic is the main criterion, followed by the loop
  /// iterations, the number of blocks and the tensor core generation. Only
  /// evaluations of the same graph and architecture are comparable.
  bool operator<(const TilingEvaluation &other) const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const TilingEvaluation &eval);
};

/**
 * @brief Tiling evaluator implementation.
 */
class TilingEvaluator {
public:
  /**
   * @brief Construct a new TilingEvaluator object.
   * @param graph The graph to evaluate.
   * @param archInfo The architecture information.
   */
  static FailureOr<TilingEvaluator> create(GraphOp graph,
                                           const ArchInfo &archInfo);

  /**
   * @brief Evaluate the tiling configuration for the current graph.
   * @param config The tiling configuration.
   */
  FailureOr<TilingEvaluation> evaluate(const TilingConfig &config);

  /// Basic getters.
  GraphOp getGraph() const { return graph; }
  const ArchInfo &getArchInfo() const { return archInfo; }
  LayoutSourceAttrInterface getInputLayout() const { return inputLayout; }
  LayoutSourceAttrInterface getOutputLayout() const { return outputLayout; }

private:
  TilingEvaluator(GraphOp graph, ArchInfo archInfo,
                  LayoutSourceAttrInterface inputLayout,
                  LayoutSourceAttrInterface outputLayout,
                  DenseMap<LayoutSourceAttrInterface, Operation *> pivots,
                  DenseMap<Operation *, SmallVector<int>> liveSizes)
      : graph(graph), archInfo(archInfo), inputLayout(inputLayout),
        outputLayout(outputLayout), pivots(std::move(pivots)),
        liveSizes(std::move(liveSizes)) {}

  GraphOp graph;
  ArchInfo archInfo;
  LayoutSourceAttrInterface inputLayout;
  LayoutSourceAttrInterface outputLayout;
  DenseMap<LayoutSourceAttrInterface, Operation *> pivots;
  DenseMap<Operation *, SmallVector<int>> liveSizes;

  friend class EvaluationLayoutIterator;
};

/**
 * @brief Layout tree iterator (base class).
 */
class LayoutIterator {
public:
  LogicalResult dispatch(LayoutSourceAttrInterface layout);

  virtual LogicalResult processTensor(TensorSourceAttr tensor);
  virtual LogicalResult processComposite(CompositeSourceAttr composite);
  virtual LogicalResult processConcat(ConcatSourceAttr concat);
  virtual LogicalResult processReduction(ReductionSourceAttr reduction);
  virtual LogicalResult processMatmul(MatmulSourceAttr matmul);
};

} // namespace tiling_analysis
} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_ANALYSIS_TILING_EVALUATE_H_
