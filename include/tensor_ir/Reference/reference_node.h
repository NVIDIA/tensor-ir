// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Reference/simplified_tensor.h"
#include "tensor_ir/Support/Status.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::nv_tensor_ir::reference {

// ============================================================================
// Node Base Class for Graph Construction
// ============================================================================

/**
 * @brief Base class for all operation nodes in the reference graph
 *
 * Each Node represents a computation operation that:
 * 1. Validates input tensors
 * 2. Computes output tensors based on inputs
 * 3. Records the original MLIR operation string
 * 4. Can be connected to form a computation graph
 */
class Node {
public:
  /**
   * @brief Construct a node with optional MLIR operation string
   * @param opName Name/type of the operation
   * @param mlirOp Original MLIR operation string (optional)
   */
  explicit Node(const std::string &opName, const std::string &mlirOp = "")
      : opName(opName), originalOperation(mlirOp) {}

  virtual ~Node() = default;

  /**
   * @brief Validate input tensors for this operation
   * @return Status indicating if inputs are valid
   */
  virtual Status validate() const = 0;

  /**
   * @brief Validate the node and execute its computation
   * @return Status indicating if validation and computation succeeded
   *
   * This method writes results to output tensor's hostPtr()
   */
  Status execute() {
    TIR_RETURN_IF_ERROR(validate());
    return computeValidated();
  }

  // Getters
  const std::string &getOpName() const { return opName; }
  const std::string &getOriginalOperation() const { return originalOperation; }

private:
  friend class ReferenceGraph;

  /// Execute the computation after the caller has validated this node.
  virtual Status computeValidated() = 0;

protected:
  std::string opName;            // Operation name/type
  std::string originalOperation; // Original MLIR operation string
};

// ============================================================================
// Constant Node
// ============================================================================

/**
 * @brief Node for TensorIR constants.
 *
 * Constants are graph-internal producers. The reference implementation stores
 * the scalar literal and fills the output tensor's host buffer.
 */
class ConstantNode : public Node {
public:
  /**
   * @brief Construct a ConstantNode
   * @param output Output tensor
   * @param literalValue Scalar value to fill the tensor with
   * @param mlirOp Original MLIR operation string (optional)
   */
  ConstantNode(std::shared_ptr<SimplifiedTensor> output, double literalValue,
               const std::string &mlirOp = "")
      : Node("constant", mlirOp), outputTensor(std::move(output)),
        scalarValue(literalValue) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> outputTensor;
  double scalarValue;
};

// ============================================================================
// Splat Node
// ============================================================================

/**
 * @brief Node for creating a tensor filled from one scalar element.
 */
class SplatNode : public Node {
public:
  /**
   * @brief Construct a SplatNode
   * @param input Scalar input tensor with exactly one element
   * @param output Output tensor filled with the scalar input
   * @param mlirOp Original MLIR operation string (optional)
   */
  SplatNode(std::shared_ptr<SimplifiedTensor> input,
            std::shared_ptr<SimplifiedTensor> output,
            const std::string &mlirOp = "")
      : Node("splat", mlirOp), inputTensor(std::move(input)),
        outputTensor(std::move(output)) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> inputTensor;
  std::shared_ptr<SimplifiedTensor> outputTensor;
};

// ============================================================================
// Iota Node
// ============================================================================

/**
 * @brief Node for creating an iota tensor along one dimension.
 *
 * Each output element is the logical coordinate of that element in
 * `iotaDimension`, converted to the output tensor element type.
 */
class IotaNode : public Node {
public:
  /**
   * @brief Construct an IotaNode
   * @param output Output tensor filled with per-dimension indices
   * @param iotaDimension Dimension whose coordinate is materialized
   * @param mlirOp Original MLIR operation string (optional)
   */
  IotaNode(std::shared_ptr<SimplifiedTensor> output, int64_t iotaDimension,
           const std::string &mlirOp = "")
      : Node("iota", mlirOp), outputTensor(std::move(output)),
        iotaDimension(iotaDimension) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> outputTensor;
  int64_t iotaDimension;
};

// ============================================================================
// Matrix Multiplication Node
// ============================================================================

/**
 * @brief Node for matrix multiplication: C = A @ B
 *
 * Supports:
 * - 2D tensors: (M, K) @ (K, N) -> (M, N)
 * - 3D tensors (batched): (B, M, K) @ (B, K, N) -> (B, M, N)
 *
 * Supported Data Types (currently requires A, B, C to have same type):
 * - FLOAT32: Input=F32, Compute=F32, Output=F32
 * - FLOAT16: Input=F16, Compute=F16, Output=F16
 * - DOUBLE:  Input=F64, Compute=F64, Output=F64
 *
 * Implementation Notes:
 * - Uses NumericConverter for flexible type conversion
 * - Kernel supports independent Input/Compute/Output types
 * - Currently: Compute type = Output type (can be extended in future)
 * - Uses hostPtr() for all tensors to compute reference results
 * - Alpha and beta parameters: currently hardcoded to 1.0 and 0.0
 *
 * Future Extensions:
 * - Mixed precision: F16 input -> F32 compute -> F16 output
 * - High precision accumulation: F16 input -> F16 compute -> F32 output
 * - Different types for A, B, C
 */
class MatmulNode : public Node {
public:
  /**
   * @brief Construct a MatmulNode
   * @param A Input tensor A (left operand)
   * @param B Input tensor B (right operand)
   * @param C Output tensor C (result)
   * @param mlirOp Original MLIR operation string (optional)
   */
  MatmulNode(std::shared_ptr<SimplifiedTensor> A,
             std::shared_ptr<SimplifiedTensor> B,
             std::shared_ptr<SimplifiedTensor> C,
             const std::string &mlirOp = "")
      : Node("matmul", mlirOp), a(std::move(A)), b(std::move(B)),
        c(std::move(C)) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  // Check if a (input_type, output_type) combination is supported for matmul
  // Returns true if this type combination has a kernel implementation
  static bool isSupportedTypeCombination(DataType input_type,
                                         DataType output_type);

  std::shared_ptr<SimplifiedTensor> a;
  std::shared_ptr<SimplifiedTensor> b;
  std::shared_ptr<SimplifiedTensor> c;
};

// ============================================================================
// Concatenate Node
// ============================================================================

/**
 * @brief Node for tensor concatenation along one dimension.
 *
 * The reference implementation copies raw elements from input hostPtr()
 * buffers to the output hostPtr() buffer using each tensor's explicit
 * strides, so it is independent of element type and memory layout.
 */
class ConcatenateNode : public Node {
public:
  /**
   * @brief Construct a ConcatenateNode
   * @param inputs Input tensors in concatenation order
   * @param output Output tensor
   * @param concatDim Dimension along which inputs are concatenated
   * @param mlirOp Original MLIR operation string (optional)
   */
  ConcatenateNode(std::vector<std::shared_ptr<SimplifiedTensor>> inputs,
                  std::shared_ptr<SimplifiedTensor> output, int64_t concatDim,
                  const std::string &mlirOp = "")
      : Node("concatenate", mlirOp), inputs(std::move(inputs)),
        output(std::move(output)), concatDim(concatDim) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::vector<std::shared_ptr<SimplifiedTensor>> inputs;
  std::shared_ptr<SimplifiedTensor> output;
  int64_t concatDim;
};

// ============================================================================
// Slice Node
// ============================================================================

/**
 * @brief Node for tensor slice extraction.
 *
 * The reference implementation maps each output coordinate to
 * `starts + output_coord * strides` in the input tensor and copies raw
 * elements, so it works for all element types with explicit tensor strides.
 */
class SliceNode : public Node {
public:
  /**
   * @brief Construct a SliceNode
   * @param input Input tensor
   * @param output Sliced output tensor
   * @param starts Per-dimension inclusive start indices
   * @param limits Per-dimension exclusive limit indices
   * @param sliceStrides Per-dimension slice strides
   * @param mlirOp Original MLIR operation string (optional)
   */
  SliceNode(std::shared_ptr<SimplifiedTensor> input,
            std::shared_ptr<SimplifiedTensor> output,
            std::vector<int64_t> starts, std::vector<int64_t> limits,
            std::vector<int64_t> sliceStrides, const std::string &mlirOp = "")
      : Node("slice", mlirOp), input(std::move(input)),
        output(std::move(output)), starts(std::move(starts)),
        limits(std::move(limits)), sliceStrides(std::move(sliceStrides)) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> input;
  std::shared_ptr<SimplifiedTensor> output;
  std::vector<int64_t> starts;
  std::vector<int64_t> limits;
  std::vector<int64_t> sliceStrides;
};

// ============================================================================
// Reshape Node
// ============================================================================

/**
 * @brief Node for tensor reshape.
 *
 * The reference implementation preserves the canonical flattened sequence,
 * in which the last tensor dimension varies fastest.
 */
class ReshapeNode : public Node {
public:
  /**
   * @brief Construct a ReshapeNode
   * @param input Input tensor
   * @param output Reshaped output tensor
   * @param mlirOp Original MLIR operation string (optional)
   */
  ReshapeNode(std::shared_ptr<SimplifiedTensor> input,
              std::shared_ptr<SimplifiedTensor> output,
              const std::string &mlirOp = "")
      : Node("reshape", mlirOp), input(std::move(input)),
        output(std::move(output)) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> input;
  std::shared_ptr<SimplifiedTensor> output;
};

// ============================================================================
// Broadcast Node
// ============================================================================

/**
 * @brief Node for rank-preserving tensor broadcast.
 *
 * The reference implementation maps each expanded output dimension back to
 * coordinate zero in the input tensor and copies raw elements, so it works for
 * all element types with explicit tensor strides.
 */
class BroadcastNode : public Node {
public:
  /**
   * @brief Construct a BroadcastNode
   * @param input Input tensor with one or more dimensions of size 1
   * @param output Broadcasted output tensor
   * @param mlirOp Original MLIR operation string (optional)
   */
  BroadcastNode(std::shared_ptr<SimplifiedTensor> input,
                std::shared_ptr<SimplifiedTensor> output,
                const std::string &mlirOp = "")
      : Node("broadcast", mlirOp), input(std::move(input)),
        output(std::move(output)) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> input;
  std::shared_ptr<SimplifiedTensor> output;
};

// ============================================================================
// Transpose Node
// ============================================================================

/**
 * @brief Node for tensor transpose.
 *
 * The reference implementation maps each output coordinate back to an input
 * coordinate according to the TensorIR transpose permutation and copies raw
 * elements, so it works for all element types with explicit tensor strides.
 */
class TransposeNode : public Node {
public:
  /**
   * @brief Construct a TransposeNode
   * @param input Input tensor
   * @param output Transposed output tensor
   * @param permutation Output-dimension to input-dimension permutation
   * @param mlirOp Original MLIR operation string (optional)
   */
  TransposeNode(std::shared_ptr<SimplifiedTensor> input,
                std::shared_ptr<SimplifiedTensor> output,
                std::vector<int64_t> permutation,
                const std::string &mlirOp = "")
      : Node("transpose", mlirOp), input(std::move(input)),
        output(std::move(output)), permutation(std::move(permutation)) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> input;
  std::shared_ptr<SimplifiedTensor> output;
  std::vector<int64_t> permutation;
};

// ============================================================================
// Reduce Node
// ============================================================================

/**
 * @brief Node for built-in TensorIR reductions.
 *
 * Reduction dimensions remain present in the output with extent one. The
 * reference implementation traverses logical coordinates, so both input and
 * output tensors may use explicit strides.
 */
class ReduceNode : public Node {
public:
  /**
   * @brief Construct a ReduceNode
   * @param input Input tensor
   * @param output Keep-dimension reduction result
   * @param dimensions Input dimensions to reduce
   * @param mode Built-in TensorIR reduction mode
   * @param mlirOp Original MLIR operation string (optional)
   */
  ReduceNode(std::shared_ptr<SimplifiedTensor> input,
             std::shared_ptr<SimplifiedTensor> output,
             std::vector<int64_t> dimensions, ReductionMode mode,
             const std::string &mlirOp = "")
      : Node("reduce", mlirOp), input(std::move(input)),
        output(std::move(output)), dimensions(std::move(dimensions)),
        mode(mode) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> input;
  std::shared_ptr<SimplifiedTensor> output;
  std::vector<int64_t> dimensions;
  ReductionMode mode;
};

// ============================================================================
// Convert Node
// ============================================================================

/**
 * @brief Node for element-wise tensor element-type conversion.
 */
class ConvertNode : public Node {
public:
  /**
   * @brief Construct a ConvertNode
   * @param input Input tensor
   * @param output Output tensor
   * @param mlirOp Original MLIR operation string (optional)
   */
  ConvertNode(std::shared_ptr<SimplifiedTensor> input,
              std::shared_ptr<SimplifiedTensor> output,
              const std::string &mlirOp = "")
      : Node("convert", mlirOp), input(std::move(input)),
        output(std::move(output)) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  static bool isSupportedTypeCombination(DataType inputType,
                                         DataType outputType);

  std::shared_ptr<SimplifiedTensor> input;
  std::shared_ptr<SimplifiedTensor> output;
};

// ============================================================================
// Unary Pointwise Operation Node
// ============================================================================

enum class UnaryPointwiseMode {
  ABS,
  CEIL,
  COS,
  ERF,
  EXP,
  FLOOR,
  GELU_FWD,
  LOG,
  NEG,
  RELU_FWD,
  SIN,
  SQRT,
  RSQRT,
  TAN,
  TANH_FWD,
  RECIPROCAL,
  SIGMOID,
  GELU_APPROX_TANH,
  LOGICAL_NOT,
};

/**
 * @brief Get string representation of UnaryPointwiseMode.
 */
const char *getEnumName(UnaryPointwiseMode mode);

/**
 * @brief Node for unary pointwise operation: B = func(A).
 *
 * Supported Data Types (currently requires A and B to have same type):
 * - FLOAT32, FLOAT16, BFLOAT16, DOUBLE for supported unary modes
 * - INT32, INT8 for ABS
 * - BOOL for LOGICAL_NOT
 */
class UnaryPointwiseNode : public Node {
public:
  /**
   * @brief Construct a UnaryPointwiseNode.
   * @param input Input tensor
   * @param output Output tensor
   * @param mode Unary operation mode
   * @param mlirOp Original MLIR operation string (optional)
   */
  UnaryPointwiseNode(std::shared_ptr<SimplifiedTensor> input,
                     std::shared_ptr<SimplifiedTensor> output,
                     UnaryPointwiseMode mode, const std::string &mlirOp = "")
      : Node("unaryPointwise", mlirOp), input(std::move(input)),
        output(std::move(output)), mode(mode) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> input;
  std::shared_ptr<SimplifiedTensor> output;
  UnaryPointwiseMode mode;
};

// ============================================================================
// Parametric Unary Pointwise Operation Node
// ============================================================================

enum class ParametricUnaryPointwiseMode {
  SOFTPLUS_FWD,
  SWISH_FWD,
  ELU_FWD,
};

/**
 * @brief Get string representation of ParametricUnaryPointwiseMode.
 */
const char *getEnumName(ParametricUnaryPointwiseMode mode);

/**
 * @brief Node for unary pointwise operations with one floating-point parameter.
 *
 * Supported operations correspond to TensorIR forward activations with a
 * `beta` attribute: softplus_fwd, swish_fwd, and elu_fwd.
 */
class ParametricUnaryPointwiseNode : public Node {
public:
  /**
   * @brief Construct a ParametricUnaryPointwiseNode.
   * @param input Input tensor
   * @param output Output tensor
   * @param mode Unary operation mode
   * @param beta Floating-point activation parameter
   * @param mlirOp Original MLIR operation string (optional)
   */
  ParametricUnaryPointwiseNode(std::shared_ptr<SimplifiedTensor> input,
                               std::shared_ptr<SimplifiedTensor> output,
                               ParametricUnaryPointwiseMode mode, double beta,
                               const std::string &mlirOp = "")
      : Node("parametricUnaryPointwise", mlirOp), input(std::move(input)),
        output(std::move(output)), mode(mode), beta(beta) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> input;
  std::shared_ptr<SimplifiedTensor> output;
  ParametricUnaryPointwiseMode mode;
  double beta;
};

// ============================================================================
// Compare Operation Node
// ============================================================================

/**
 * @brief Node for tensor compare: C = compare(A, B).
 *
 * The output tensor must be BOOL and have the same shape as both inputs.
 */
class CompareNode : public Node {
public:
  /**
   * @brief Construct a CompareNode
   * @param lhs Left operand tensor
   * @param rhs Right operand tensor
   * @param output Boolean output tensor
   * @param comparator TensorIR comparison predicate
   * @param mlirOp Original MLIR operation string (optional)
   */
  CompareNode(std::shared_ptr<SimplifiedTensor> lhs,
              std::shared_ptr<SimplifiedTensor> rhs,
              std::shared_ptr<SimplifiedTensor> output,
              mlir::nv_tensor_ir::Comparator comparator,
              const std::string &mlirOp = "")
      : Node("compare", mlirOp), lhsTensor(std::move(lhs)),
        rhsTensor(std::move(rhs)), outputTensor(std::move(output)),
        comparator(comparator) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> lhsTensor;
  std::shared_ptr<SimplifiedTensor> rhsTensor;
  std::shared_ptr<SimplifiedTensor> outputTensor;
  mlir::nv_tensor_ir::Comparator comparator;
};

// ============================================================================
// Binary Pointwise Operation Node
// ============================================================================

enum class BinaryPointwiseMode {
  ADD,
  SUB,
  MUL,
  DIV,
  MAX,
  MIN,
  POW,
  ATAN2,
  MOD,
  REM,
  ADD_SQUARE,
  RELU_BWD,
  GELU_BWD,
  SIGMOID_BWD,
  TANH_BWD,
  GELU_APPROX_TANH_BWD,
  LOGICAL_AND,
  LOGICAL_OR,
};

/**
 * @brief Get string representation of BinaryPointwiseMode
 */
const char *getEnumName(BinaryPointwiseMode mode);

/**
 * @brief Node for binary pointwise operation: C = func(A, B)
 *
 * Supports binary pointwise operations
 *
 * Supported Data Types (currently requires A, B, C to have same type):
 * - FLOAT32, FLOAT16, BFLOAT16, DOUBLE for numeric operations
 * - INT32 for basic arithmetic, min/max, and remainder operations
 * - BOOL for logical operations
 *
 */
class BinaryPointwiseNode : public Node {
public:
  /**
   * @brief Construct a BinaryPointwiseNode
   * @param A Input tensor A (left operand)
   * @param B Input tensor B (right operand)
   * @param C Output tensor C (result)
   * @param mode Binary operation mode
   * @param mlirOp Original MLIR operation string (optional)
   */
  BinaryPointwiseNode(std::shared_ptr<SimplifiedTensor> A,
                      std::shared_ptr<SimplifiedTensor> B,
                      std::shared_ptr<SimplifiedTensor> C,
                      BinaryPointwiseMode mode, const std::string &mlirOp = "")
      : Node("binaryPointwise", mlirOp), a(std::move(A)), b(std::move(B)),
        c(std::move(C)), mode(mode) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> a;
  std::shared_ptr<SimplifiedTensor> b;
  std::shared_ptr<SimplifiedTensor> c;
  BinaryPointwiseMode mode;
};

// ============================================================================
// Parametric Binary Pointwise Operation Node
// ============================================================================

enum class ParametricBinaryPointwiseMode {
  SOFTPLUS_BWD,
  SWISH_BWD,
  ELU_BWD,
};

/**
 * @brief Get string representation of ParametricBinaryPointwiseMode.
 */
const char *getEnumName(ParametricBinaryPointwiseMode mode);

/**
 * @brief Node for binary pointwise operations with one floating-point
 * parameter.
 *
 * The first input is the activation input, the second is the upstream
 * gradient, and `beta` is the corresponding forward activation parameter.
 */
class ParametricBinaryPointwiseNode : public Node {
public:
  /**
   * @brief Construct a ParametricBinaryPointwiseNode.
   * @param input Activation input tensor
   * @param gradient Upstream gradient tensor
   * @param output Output gradient tensor
   * @param mode Binary operation mode
   * @param beta Floating-point activation parameter
   * @param mlirOp Original MLIR operation string (optional)
   */
  ParametricBinaryPointwiseNode(std::shared_ptr<SimplifiedTensor> input,
                                std::shared_ptr<SimplifiedTensor> gradient,
                                std::shared_ptr<SimplifiedTensor> output,
                                ParametricBinaryPointwiseMode mode, double beta,
                                const std::string &mlirOp = "")
      : Node("parametricBinaryPointwise", mlirOp), input(std::move(input)),
        gradient(std::move(gradient)), output(std::move(output)), mode(mode),
        beta(beta) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> input;
  std::shared_ptr<SimplifiedTensor> gradient;
  std::shared_ptr<SimplifiedTensor> output;
  ParametricBinaryPointwiseMode mode;
  double beta;
};

// ============================================================================
// Binary Select Operation Node
// ============================================================================

/**
 * @brief Node for binary select: C = selector ? A : B.
 */
class BinarySelectNode : public Node {
public:
  /**
   * @brief Construct a BinarySelectNode.
   * @param selector Boolean selector tensor
   * @param A Input tensor selected when selector is true
   * @param B Input tensor selected when selector is false
   * @param C Output tensor
   * @param mlir_op Original MLIR operation string (optional)
   */
  BinarySelectNode(std::shared_ptr<SimplifiedTensor> selector,
                   std::shared_ptr<SimplifiedTensor> A,
                   std::shared_ptr<SimplifiedTensor> B,
                   std::shared_ptr<SimplifiedTensor> C,
                   const std::string &mlir_op = "")
      : Node("binary_select", mlir_op), selector_(std::move(selector)),
        A_(std::move(A)), B_(std::move(B)), C_(std::move(C)) {}

  Status validate() const override;

private:
  Status computeValidated() override;

  std::shared_ptr<SimplifiedTensor> selector_;
  std::shared_ptr<SimplifiedTensor> A_;
  std::shared_ptr<SimplifiedTensor> B_;
  std::shared_ptr<SimplifiedTensor> C_;
};
} // namespace mlir::nv_tensor_ir::reference
