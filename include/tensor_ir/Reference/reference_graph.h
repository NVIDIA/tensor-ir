// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "tensor_ir/Reference/reference_node.h"
#include "tensor_ir/Reference/simplified_tensor.h"
#include "tensor_ir/Support/Status.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
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
  /// '?' stride at position d.  If not set, dynamic strides are deduced
  /// assuming contiguous (packed) layout.
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

  /**
   * @brief Parse tensor shape from MLIR type
   *
   * Extracts shape directly from ranked tensor types.
   * Dynamic dimensions are preserved as ShapedType::kDynamic.
   *
   * @param mlirType MLIR tensor type
   * @return Vector of dimension sizes
   */
  std::vector<int64_t> parseTensorShape(mlir::Type mlirType);

  /**
   * @brief Parse data type from MLIR type
   *
   * @param mlirType MLIR element type
   * @return DataType enum, or an error for an unsupported element type
   */
  StatusOr<DataType> parseDataType(mlir::Type mlirType);

  /**
   * @brief Create input/output tensors from MLIR module signature
   *
   * Analyzes the graph operation to extract input and output tensor
   * information, then creates SimplifiedTensor instances for all inputs and
   * outputs.
   *
   * @param module MLIR module containing the graph
   * @return Status indicating success or failure
   */
  Status createIOTensor(mlir::ModuleOp module);

  /**
   * @brief Create Node for a tensor_ir.matmul operation
   *
   * Extracts tensor shapes directly from the operation's type information.
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createMatmulNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.constant operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createConstantNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.splat operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createSplatNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.iota operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createIotaNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.reduce operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createReduceNode(mlir::Operation *op);

  /**
   * @brief Create Node for binary pointwise operations
   *
   * @param op MLIR operation
   * @param mode BinaryPointwiseMode indicating the operation type
   * @return Status indicating success or failure
   */
  Status createBinaryPointwiseNode(mlir::Operation *op,
                                   BinaryPointwiseMode mode);

  /**
   * @brief Create Node for beta-parameterized binary pointwise operations
   *
   * @param op MLIR operation
   * @param mode ParametricBinaryPointwiseMode indicating the operation type
   * @param beta Floating-point activation parameter
   * @return Status indicating success or failure
   */
  Status createParametricBinaryPointwiseNode(mlir::Operation *op,
                                             ParametricBinaryPointwiseMode mode,
                                             double beta);

  /**
   * @brief Create a one-result node with shared tensor plumbing
   *
   * @param op MLIR operation
   * @param expectedOperands Number of input operands expected by the operation
   * @param nodeFactory Factory that builds the concrete node
   * @return Status indicating success or failure
   */
  Status
  createSingleResultNode(mlir::Operation *op, unsigned expectedOperands,
                         llvm::function_ref<std::unique_ptr<Node>(
                             llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>>,
                             const std::string &)>
                             nodeFactory);

  /**
   * @brief Create Node for unary pointwise operations
   *
   * @param op MLIR operation
   * @param mode UnaryPointwiseMode indicating the operation type
   * @return Status indicating success or failure
   */
  Status createUnaryPointwiseNode(mlir::Operation *op, UnaryPointwiseMode mode);

  /**
   * @brief Create Node for beta-parameterized unary pointwise operations
   *
   * @param op MLIR operation
   * @param mode ParametricUnaryPointwiseMode indicating the operation type
   * @param beta Floating-point activation parameter
   * @return Status indicating success or failure
   */
  Status createParametricUnaryPointwiseNode(mlir::Operation *op,
                                            ParametricUnaryPointwiseMode mode,
                                            double beta);

  /**
   * @brief Create Node for a tensor_ir.cmp operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createCompareNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.binary_select operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createBinarySelectNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.concatenate operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createConcatenateNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.slice operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createSliceNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.reshape operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createReshapeNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.broadcast operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createBroadcastNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.transpose operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createTransposeNode(mlir::Operation *op);

  /**
   * @brief Create Node for a tensor_ir.convert operation
   *
   * @param op MLIR operation
   * @return Status indicating success or failure
   */
  Status createConvertNode(mlir::Operation *op);

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

  /**
   * @brief Helper function to print tensor shape
   *
   * @param dims Dimension sizes
   */
  void printTensorShape(llvm::ArrayRef<int64_t> dims) const;

  /// Given concrete dims and a stride pattern (with kDynamic for '?'),
  /// resolve dynamic stride entries.
  /// @param userStrides  If non-empty, use userStrides[d] for '?' at pos d.
  ///                     If empty, deduce contiguous (packed) strides from
  ///                     dims.
  static std::vector<int64_t>
  resolveStrides(llvm::ArrayRef<int64_t> dims,
                 llvm::ArrayRef<int64_t> stridePattern,
                 llvm::ArrayRef<int64_t> userStrides = {});

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
