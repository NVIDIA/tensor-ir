// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_REFERENCE_REFERENCEGRAPH_H
#define TENSOR_IR_REFERENCE_REFERENCEGRAPH_H

#include "tensor_ir/Reference/ReferenceNode.h"
#include "tensor_ir/Reference/SimplifiedTensor.h"
#include "tensor_ir/Support/Status.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlir::nv_tensor_ir::reference {

// ============================================================================
// ReferenceGraph - Computational Graph for Reference Implementation
// ============================================================================

/**
 * @brief ReferenceGraph parses MLIR module and constructs a reference
 * computation graph
 *
 * Responsibilities:
 * 1. Parse MLIR operations and create corresponding Node instances
 * 2. Create and manage SimplifiedTensor instances for all tensors
 * 3. Maintain tensor name -> SimplifiedTensor mapping
 * 4. Support graph execution for verification
 *
 * Usage:
 *   ReferenceGraph graph;
 *   Status status = graph.buildFromMLIR(module);
 *   if (status.ok()) {
 *     status = graph.execute();
 *   }
 */
class ReferenceGraph {
public:
  ReferenceGraph();
  ~ReferenceGraph() = default;

  // ============================================================================
  // Graph Construction
  // ============================================================================

  /// Set runtime values for dynamic '?' dimensions before calling
  /// buildFromMLIR. Values are indexed by dimension position: dynamicDims[d]
  /// resolves the '?' at tensor dimension d.  If a dynamic dim's position
  /// exceeds the list, the last value is reused.
  void setDynamicDims(llvm::ArrayRef<int64_t> dims) {
    dynamicDims.assign(dims.begin(), dims.end());
  }

  /// Set runtime values for dynamic '?' strides before calling buildFromMLIR.
  /// Values are indexed by dimension position: dynamicStrides[d] resolves the
  /// '?' stride at position d. A non-empty list must provide a value for every
  /// dynamic stride position. If not set, dynamic strides are deduced assuming
  /// contiguous (packed) layout.
  void setDynamicStrides(llvm::ArrayRef<int64_t> strides) {
    dynamicStrides.assign(strides.begin(), strides.end());
  }

  /**
   * @brief Build computation graph from MLIR module
   *
   * This method:
   * 1. Walks through all operations in the MLIR module
   * 2. Creates SimplifiedTensor for each tensor (SSA value)
   * 3. Creates Node instances for each operation
   * 4. Extracts tensor shapes directly from MLIR type information
   *
   * If dynamic dimensions are present and setDynamicDims() was called,
   * the '?' dims are resolved to the provided runtime values.
   *
   * @param module MLIR module containing tensor_ir operations
   * @return Status indicating success or failure
   */
  Status buildFromMLIR(mlir::ModuleOp module);

  /// Build only graph input and output tensors, without CPU reference nodes.
  Status buildIOTensorsFromMLIR(mlir::ModuleOp module);

  // ============================================================================
  // Graph Execution
  // ============================================================================

  /**
   * @brief Execute all nodes in the graph in topological order
   * @return Status indicating success or failure
   */
  Status execute();

  // ============================================================================
  // Tensor Access
  // ============================================================================

  llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> getInputTensors() const {
    return inputTensors;
  }
  llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> getOutputTensors() const {
    return outputTensors;
  }

  /// Fill all input tensors with deterministic random data from \p seed.
  ///
  /// @return Status::Ok() if every input tensor is initialized. Returns an
  ///   invalid-argument status if an input has no host storage, does not
  ///   support random filling, or its integer type cannot represent any value
  ///   in [\p minVal, \p maxVal].
  Status fillInputTensorsRandom(float minVal, float maxVal, unsigned seed);

  // ============================================================================
  // Debugging and Visualization
  // ============================================================================

  /**
   * @brief Print graph structure for debugging
   */
  void print() const;

private:
  /// Validate all nodes in the graph.
  Status validate() const;

  /// Remove all nodes and tensors and reset graph construction state.
  void clear();

  // ============================================================================
  // Internal Helper Methods
  // ============================================================================

  /**
   * @brief Create SimplifiedTensor based on MLIR type
   *
   * Extracts shape and data type directly from the MLIR type.
   *
   * @param name Tensor name
   * @param mlirType MLIR tensor type
   * @return Created SimplifiedTensor or an error describing why the MLIR type
   *   cannot be represented by the reference implementation
   */
  /// @param stridePattern  If non-empty, stride pattern from the MLIR attribute
  ///   (with kDynamic for '?'). Dynamic entries are resolved assuming
  ///   contiguous layout. If empty, row-major strides are computed from
  ///   dims.
  StatusOr<std::shared_ptr<SimplifiedTensor>>
  createTensorFromMLIRType(const std::string &name, mlir::Type mlirType,
                           llvm::ArrayRef<int64_t> stridePattern = {});

  /// Create input/output tensors using the graph signature's stride metadata.
  Status createIOTensor(mlir::ModuleOp module);

  /// Collect tensors, construct the operation's node, and validate it.
  Status createNode(mlir::Operation *op);

  /**
   * @brief Get or create tensor for an MLIR SSA value
   *
   * Extracts tensor information directly from the value's type.
   *
   * @param value MLIR SSA value
   * @return Shared pointer to SimplifiedTensor, or a tensor creation error
   */
  StatusOr<std::shared_ptr<SimplifiedTensor>>
  getOrCreateTensor(mlir::Value value, llvm::ArrayRef<int64_t> strides = {});

  /**
   * @brief Generate a unique tensor name for an MLIR SSA value
   *
   * Creates a unique name based on the value type (block argument or operation
   * result). Ensures no naming conflicts by checking existing tensors and
   * adding suffixes if needed.
   *
   * @param value MLIR SSA value to generate name for
   * @return Unique tensor name (e.g., "arg0", "matmulResult0",
   * "addResult01")
   */
  std::string generateUniqueTensorName(mlir::Value value);

  // ============================================================================
  // Graph Data Structures
  // ============================================================================

  // Map from tensor name to SimplifiedTensor
  std::map<std::string, std::shared_ptr<SimplifiedTensor>> tensors;

  std::vector<std::shared_ptr<SimplifiedTensor>> inputTensors;
  std::vector<std::shared_ptr<SimplifiedTensor>> outputTensors;

  // Vector of computation nodes in execution order
  std::vector<std::unique_ptr<Node>> nodes;

  // Map from MLIR Value to tensor name (for SSA value tracking)
  std::map<void *, std::string> valueToName;

  // Counter for generating unique tensor names
  int tensorNameCounter = 0;

  // Runtime values for dynamic dimensions (set via setDynamicDims).
  llvm::SmallVector<int64_t> dynamicDims;

  // Runtime values for dynamic strides (set via setDynamicStrides).
  llvm::SmallVector<int64_t> dynamicStrides;
};

} // namespace mlir::nv_tensor_ir::reference

#endif // TENSOR_IR_REFERENCE_REFERENCEGRAPH_H
