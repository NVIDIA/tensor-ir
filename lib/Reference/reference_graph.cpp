// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Reference/reference_graph.h"

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Reference/simplified_tensor.h"
#include "tensor_ir/Support/Status.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <random>
#include <sstream>

namespace mlir::nv_tensor_ir::reference {

// ============================================================================
// ReferenceGraph Implementation
// ============================================================================

ReferenceGraph::ReferenceGraph() : tensorNameCounter(0) {}

static StatusOr<double> getScalarConstantValue(mlir::Attribute attr,
                                               DataType dtype) {
  if (auto denseAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!denseAttr.isSplat()) {
      return Status::InvalidArgument(
          "Only scalar or splat dense constants are supported");
    }
    if (denseAttr.getElementType().isInteger()) {
      const llvm::APInt &intValue = denseAttr.getSplatValue<llvm::APInt>();
      return isUnsignedIntegerDataType(dtype)
                 ? static_cast<double>(intValue.getZExtValue())
                 : static_cast<double>(intValue.getSExtValue());
    }
    if (denseAttr.getElementType().isFloat()) {
      return denseAttr.getSplatValue<llvm::APFloat>().convertToDouble();
    }
    return Status::InvalidArgument("Unsupported dense constant element type");
  }

  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    return floatAttr.getValueAsDouble();
  }
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    const llvm::APInt &intValue = intAttr.getValue();
    return isUnsignedIntegerDataType(dtype)
               ? static_cast<double>(intValue.getZExtValue())
               : static_cast<double>(intValue.getSExtValue());
  }
  return Status::InvalidArgument("Unsupported constant attribute");
}

void ReferenceGraph::clear() {
  nodes.clear();
  tensors.clear();
  valueToName.clear();
  inputTensors.clear();
  outputTensors.clear();
  tensorNameCounter = 0;
}

// ============================================================================
// Graph Construction
// ============================================================================

Status ReferenceGraph::buildFromMLIR(mlir::ModuleOp module) {
  clear();

  // First pass: Create input/output tensors from graph signature
  Status status = createIOTensor(module);
  if (!status.ok()) {
    return status;
  }

  // Second pass: Walk through all operations to build the computational graph
  Status buildStatus = Status::Ok();
  auto walkResult = module.walk([&](mlir::Operation *op) {
    // Skip module and function operations
    if (llvm::isa<mlir::ModuleOp>(op)) {
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.graph (function-like operation)
    if (llvm::isa<nv_tensor_ir::GraphOp>(op)) {
      // This is the graph definition, skip it but process its body
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.constant
    if (llvm::isa<nv_tensor_ir::ConstantOp>(op)) {
      Status status = createConstantNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create constant node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.splat
    if (llvm::isa<nv_tensor_ir::SplatOp>(op)) {
      Status status = createSplatNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create splat node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.iota
    if (llvm::isa<nv_tensor_ir::IotaOp>(op)) {
      Status status = createIotaNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create iota node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.reduce
    if (llvm::isa<nv_tensor_ir::ReduceOp>(op)) {
      Status status = createReduceNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create reduce node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.matmul
    if (llvm::isa<nv_tensor_ir::MatmulOp>(op)) {
      Status status = createMatmulNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create matmul node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    auto handleBinaryPointwise = [&](BinaryPointwiseMode mode) {
      Status status = createBinaryPointwiseNode(op, mode);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create "
                     << op->getName().getStringRef().str()
                     << " node: " << status.message() << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    };

#define HANDLE_BINARY_PW(OpType, mode)                                         \
  if (llvm::isa<nv_tensor_ir::OpType>(op))                                     \
    return handleBinaryPointwise(BinaryPointwiseMode::mode);

    HANDLE_BINARY_PW(AddOp, ADD)
    HANDLE_BINARY_PW(SubOp, SUB)
    HANDLE_BINARY_PW(MulOp, MUL)
    HANDLE_BINARY_PW(DivOp, DIV)
    HANDLE_BINARY_PW(MaxOp, MAX)
    HANDLE_BINARY_PW(MinOp, MIN)
    HANDLE_BINARY_PW(PowOp, POW)
    HANDLE_BINARY_PW(Atan2Op, ATAN2)
    HANDLE_BINARY_PW(ModOp, MOD)
    HANDLE_BINARY_PW(RemOp, REM)
    HANDLE_BINARY_PW(AddSquareOp, ADD_SQUARE)
    HANDLE_BINARY_PW(ReluBwdOp, RELU_BWD)
    HANDLE_BINARY_PW(GeluBwdOp, GELU_BWD)
    HANDLE_BINARY_PW(SigmoidBwdOp, SIGMOID_BWD)
    HANDLE_BINARY_PW(TanhBwdOp, TANH_BWD)
    HANDLE_BINARY_PW(GeluApproxTanhBwdOp, GELU_APPROX_TANH_BWD)
    HANDLE_BINARY_PW(LogicalAndOp, LOGICAL_AND)
    HANDLE_BINARY_PW(LogicalOrOp, LOGICAL_OR)

#undef HANDLE_BINARY_PW

    auto handleUnaryPointwise = [&](UnaryPointwiseMode mode) {
      Status status = createUnaryPointwiseNode(op, mode);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create "
                     << op->getName().getStringRef().str()
                     << " node: " << status.message() << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    };

#define HANDLE_UNARY_PW(OpType, mode)                                          \
  if (llvm::isa<nv_tensor_ir::OpType>(op))                                     \
    return handleUnaryPointwise(UnaryPointwiseMode::mode);

    HANDLE_UNARY_PW(AbsOp, ABS)
    HANDLE_UNARY_PW(CeilOp, CEIL)
    HANDLE_UNARY_PW(CosOp, COS)
    HANDLE_UNARY_PW(ErfOp, ERF)
    HANDLE_UNARY_PW(ExpOp, EXP)
    HANDLE_UNARY_PW(FloorOp, FLOOR)
    HANDLE_UNARY_PW(GeluFwdOp, GELU_FWD)
    HANDLE_UNARY_PW(LogOp, LOG)
    HANDLE_UNARY_PW(NegOp, NEG)
    HANDLE_UNARY_PW(ReluFwdOp, RELU_FWD)
    HANDLE_UNARY_PW(SinOp, SIN)
    HANDLE_UNARY_PW(SqrtOp, SQRT)
    HANDLE_UNARY_PW(RsqrtOp, RSQRT)
    HANDLE_UNARY_PW(TanOp, TAN)
    HANDLE_UNARY_PW(TanhFwdOp, TANH_FWD)
    HANDLE_UNARY_PW(ReciprocalOp, RECIPROCAL)
    HANDLE_UNARY_PW(SigmoidFwdOp, SIGMOID)
    HANDLE_UNARY_PW(GeluApproxTanhFwdOp, GELU_APPROX_TANH)
    HANDLE_UNARY_PW(LogicalNotOp, LOGICAL_NOT)

#undef HANDLE_UNARY_PW

    auto handleParametricUnaryPointwise = [&](ParametricUnaryPointwiseMode mode,
                                              double beta) {
      Status status = createParametricUnaryPointwiseNode(op, mode, beta);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create "
                     << op->getName().getStringRef().str()
                     << " node: " << status.message() << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    };

#define HANDLE_PARAMETRIC_UNARY_PW(OpType, mode)                               \
  if (auto typedOp = llvm::dyn_cast<nv_tensor_ir::OpType>(op))                 \
    return handleParametricUnaryPointwise(                                     \
        ParametricUnaryPointwiseMode::mode,                                    \
        typedOp.getBeta().convertToDouble());

    HANDLE_PARAMETRIC_UNARY_PW(SoftplusFwdOp, SOFTPLUS_FWD)
    HANDLE_PARAMETRIC_UNARY_PW(SwishFwdOp, SWISH_FWD)
    HANDLE_PARAMETRIC_UNARY_PW(EluFwdOp, ELU_FWD)

#undef HANDLE_PARAMETRIC_UNARY_PW

    auto handleParametricBinaryPointwise =
        [&](ParametricBinaryPointwiseMode mode, double beta) {
          Status status = createParametricBinaryPointwiseNode(op, mode, beta);
          if (!status.ok()) {
            buildStatus = status;
            llvm::errs() << "Failed to create "
                         << op->getName().getStringRef().str()
                         << " node: " << status.message() << "\n";
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        };

#define HANDLE_PARAMETRIC_BINARY_PW(OpType, mode)                              \
  if (auto typedOp = llvm::dyn_cast<nv_tensor_ir::OpType>(op))                 \
    return handleParametricBinaryPointwise(                                    \
        ParametricBinaryPointwiseMode::mode,                                   \
        typedOp.getBeta().convertToDouble());

    HANDLE_PARAMETRIC_BINARY_PW(SoftplusBwdOp, SOFTPLUS_BWD)
    HANDLE_PARAMETRIC_BINARY_PW(SwishBwdOp, SWISH_BWD)
    HANDLE_PARAMETRIC_BINARY_PW(EluBwdOp, ELU_BWD)

#undef HANDLE_PARAMETRIC_BINARY_PW

    // Handle nv_tensor_ir.cmp
    if (llvm::isa<nv_tensor_ir::CmpOp>(op)) {
      Status status = createCompareNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create cmp node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.binary_select
    if (llvm::isa<nv_tensor_ir::BinarySelectOp>(op)) {
      Status status = createBinarySelectNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create binary_select node: "
                     << status.message() << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.convert
    if (llvm::isa<nv_tensor_ir::ConvertOp>(op)) {
      Status status = createConvertNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create convert node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.concatenate
    if (llvm::isa<nv_tensor_ir::ConcatenateOp>(op)) {
      Status status = createConcatenateNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create concatenate node: "
                     << status.message() << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.slice
    if (llvm::isa<nv_tensor_ir::SliceOp>(op)) {
      Status status = createSliceNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create slice node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.reshape
    if (llvm::isa<nv_tensor_ir::ReshapeOp>(op)) {
      Status status = createReshapeNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create reshape node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.broadcast
    if (llvm::isa<nv_tensor_ir::BroadcastOp>(op)) {
      Status status = createBroadcastNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create broadcast node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.transpose
    if (llvm::isa<nv_tensor_ir::TransposeOp>(op)) {
      Status status = createTransposeNode(op);
      if (!status.ok()) {
        buildStatus = status;
        llvm::errs() << "Failed to create transpose node: " << status.message()
                     << "\n";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }

    // Handle nv_tensor_ir.results (return operation)
    if (llvm::isa<nv_tensor_ir::ResultsOp>(op)) {
      // No node needed, just marks outputs
      return mlir::WalkResult::advance();
    }

    buildStatus =
        Status::NotSupported("Unsupported operation in reference graph: " +
                             op->getName().getStringRef().str());
    return mlir::WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted()) {
    return buildStatus;
  }

  return Status::Ok();
}

// ============================================================================
// Stride Helpers
// ============================================================================

std::vector<int64_t>
ReferenceGraph::resolveStrides(llvm::ArrayRef<int64_t> dims,
                               llvm::ArrayRef<int64_t> stridePattern,
                               llvm::ArrayRef<int64_t> userStrides) {
  if (stridePattern.empty() || dims.size() != stridePattern.size()) {
    return {};
  }

  std::vector<int64_t> resolved(stridePattern.begin(), stridePattern.end());

  if (!userStrides.empty()) {
    // Use explicit user-provided values for dynamic stride entries.
    for (size_t d = 0, e = resolved.size(); d != e; ++d) {
      if (resolved[d] == mlir::ShapedType::kDynamic) {
        size_t idx = std::min(d, userStrides.size() - 1);
        resolved[d] = userStrides[idx];
      }
    }
  } else {
    // Fallback: deduce contiguous (packed) strides from dims.
    // Work from the last dimension inward (row-major).
    int64_t runningStride = 1;
    for (size_t reverseIdx = resolved.size(); reverseIdx > 0; --reverseIdx) {
      size_t d = reverseIdx - 1;
      if (resolved[d] == mlir::ShapedType::kDynamic) {
        resolved[d] = runningStride;
      }
      runningStride = resolved[d] * dims[d];
    }
  }
  return resolved;
}

// ============================================================================
// I/O Tensor Creation
// ============================================================================

Status ReferenceGraph::createIOTensor(mlir::ModuleOp module) {
  llvm::outs() << "\n=== Graph Signature Analysis ==="
               << "\n";

  // Find graph and results operations in one pass
  mlir::Operation *graphOp = nullptr;
  mlir::Operation *resultsOp = nullptr;

  module.walk([&](mlir::Operation *op) {
    if (llvm::isa<nv_tensor_ir::GraphOp>(op)) {
      graphOp = op;
    } else if (llvm::isa<nv_tensor_ir::ResultsOp>(op)) {
      resultsOp = op;
    }
    return mlir::WalkResult::advance();
  });

  if (!graphOp) {
    return Status::InvalidArgument("No nv_tensor_ir.graph operation found");
  }

  llvm::outs() << "Graph name: " << graphOp->getName().getStringRef().str()
               << "\n";

  auto graph = llvm::cast<nv_tensor_ir::GraphOp>(graphOp);

  // Process input tensors
  if (graphOp->getNumRegions() > 0 && !graphOp->getRegion(0).empty()) {
    mlir::Block &block = graphOp->getRegion(0).front();
    llvm::outs() << "\nInput Tensors (" << block.getNumArguments() << "):"
                 << "\n";

    auto argDescriptors =
        getTensorDescriptors(graph.getArgumentTypes(), graph.getAllArgAttrs());

    for (unsigned i = 0; i < block.getNumArguments(); ++i) {
      const TensorDescriptor &descriptor = (*argDescriptors)[i];
      std::vector<int64_t> strides;
      for (const auto &stride : descriptor.strides) {
        strides.push_back(stride.staticValue);
      }

      // SimplifiedTensor owns the host memory used by the interpreter.
      TIR_ASSIGN_OR_RETURN(auto tensor,
                           getOrCreateTensor(block.getArgument(i), strides));
      inputTensors.push_back(tensor);
      llvm::outs() << "Get input tensor: "
                   << "\n";
      tensor->print();
    }
  }

  // Process output tensors
  llvm::outs() << "\nOutput Tensors:"
               << "\n";
  if (resultsOp) {
    llvm::outs() << "  Found " << resultsOp->getNumOperands() << " output(s):"
                 << "\n";

    auto resDescriptors =
        getTensorDescriptors(graph.getResultTypes(), graph.getAllResultAttrs());

    for (unsigned i = 0; i < resultsOp->getNumOperands(); ++i) {
      const TensorDescriptor &descriptor = (*resDescriptors)[i];
      std::vector<int64_t> strides;
      for (const auto &stride : descriptor.strides) {
        strides.push_back(stride.staticValue);
      }

      // SimplifiedTensor owns the host memory used by the interpreter.
      TIR_ASSIGN_OR_RETURN(
          auto tensor, getOrCreateTensor(resultsOp->getOperand(i), strides));
      outputTensors.push_back(tensor);
      llvm::outs() << "Get output tensor: "
                   << "\n";
      tensor->print();
    }
  } else {
    llvm::outs() << "  Warning: No nv_tensor_ir.results operation found"
                 << "\n";
  }

  llvm::outs() << "=== End Graph Signature ==="
               << "\n"
               << "\n";
  return Status::Ok();
}

// ============================================================================
// Node Creation
// ============================================================================

Status ReferenceGraph::createMatmulNode(mlir::Operation *op) {
  // Get operands (inputs)
  if (op->getNumOperands() != 2) {
    return Status::InvalidArgument("Matmul operation expects 2 operands");
  }

  mlir::Value lhs = op->getOperand(0);
  mlir::Value rhs = op->getOperand(1);

  // Get result (output)
  if (op->getNumResults() != 1) {
    return Status::InvalidArgument("Matmul operation expects 1 result");
  }

  mlir::Value result = op->getResult(0);

  // Extract and print tensor shapes
  llvm::outs() << "\n=== Creating Matmul Node ==="
               << "\n";

  std::vector<int64_t> shapeA = parseTensorShape(lhs.getType());
  std::vector<int64_t> shapeB = parseTensorShape(rhs.getType());
  std::vector<int64_t> shapeC = parseTensorShape(result.getType());

  llvm::outs() << "Tensor A shape: ";
  printTensorShape(shapeA);
  llvm::outs() << "\n";

  llvm::outs() << "Tensor B shape: ";
  printTensorShape(shapeB);
  llvm::outs() << "\n";

  llvm::outs() << "Tensor C shape: ";
  printTensorShape(shapeC);
  llvm::outs() << "\n";

  // Create or get tensors (tensor shapes come directly from types)
  TIR_ASSIGN_OR_RETURN(auto tensorA, getOrCreateTensor(lhs));
  TIR_ASSIGN_OR_RETURN(auto tensorB, getOrCreateTensor(rhs));
  TIR_ASSIGN_OR_RETURN(auto tensorC, getOrCreateTensor(result));

  // Get operation location as string (for debugging)
  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  // Create MatmulNode
  auto node = std::make_unique<MatmulNode>(tensorA, tensorB, tensorC, opStr);

  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for matmul node: " +
                                   status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createConstantNode(mlir::Operation *op) {
  auto constantOp = llvm::cast<nv_tensor_ir::ConstantOp>(op);
  std::string opName = op->getName().getStringRef().str();

  if (op->getNumOperands() != 0) {
    return Status::InvalidArgument(opName + " operation expects 0 operands");
  }
  if (op->getNumResults() != 1) {
    return Status::InvalidArgument(opName + " operation expects 1 result");
  }

  mlir::Value output = op->getResult(0);

  llvm::outs() << "\n=== Creating " << opName << " Node ===\n";
  std::vector<int64_t> outputShape = parseTensorShape(output.getType());
  llvm::outs() << "Output shape: ";
  printTensorShape(outputShape);
  llvm::outs() << "\n";

  TIR_ASSIGN_OR_RETURN(auto outputTensor, getOrCreateTensor(output));

  TIR_ASSIGN_OR_RETURN(double literalValue,
                       getScalarConstantValue(constantOp.getValue(),
                                              outputTensor->getDataType()));

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);

  auto node = std::make_unique<ConstantNode>(outputTensor, literalValue, opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createSplatNode(mlir::Operation *op) {
  auto splatOp = llvm::cast<nv_tensor_ir::SplatOp>(op);
  std::string opName = op->getName().getStringRef().str();

  if (op->getNumOperands() != 1) {
    return Status::InvalidArgument(opName + " operation expects 1 operand");
  }
  if (op->getNumResults() != 1) {
    return Status::InvalidArgument(opName + " operation expects 1 result");
  }

  mlir::Value input = splatOp.getInput();
  mlir::Value output = splatOp.getOutput();

  llvm::outs() << "\n=== Creating " << opName << " Node ===\n";
  std::vector<int64_t> inputShape = parseTensorShape(input.getType());
  std::vector<int64_t> outputShape = parseTensorShape(output.getType());

  llvm::outs() << "Input shape: ";
  printTensorShape(inputShape);
  llvm::outs() << "\n";

  llvm::outs() << "Output shape: ";
  printTensorShape(outputShape);
  llvm::outs() << "\n";

  TIR_ASSIGN_OR_RETURN(auto inputTensor, getOrCreateTensor(input));
  TIR_ASSIGN_OR_RETURN(auto outputTensor, getOrCreateTensor(output));

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);

  auto node = std::make_unique<SplatNode>(inputTensor, outputTensor, opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createIotaNode(mlir::Operation *op) {
  auto iotaOp = llvm::cast<nv_tensor_ir::IotaOp>(op);
  std::string opName = op->getName().getStringRef().str();

  if (op->getNumOperands() != 0) {
    return Status::InvalidArgument(opName + " operation expects 0 operands");
  }
  if (op->getNumResults() != 1) {
    return Status::InvalidArgument(opName + " operation expects 1 result");
  }

  mlir::Value output = iotaOp.getOutput();

  llvm::outs() << "\n=== Creating " << opName << " Node ===\n";
  std::vector<int64_t> outputShape = parseTensorShape(output.getType());
  llvm::outs() << "Output shape: ";
  printTensorShape(outputShape);
  llvm::outs() << "\n";
  llvm::outs() << "Iota dimension: " << iotaOp.getDimension() << "\n";

  TIR_ASSIGN_OR_RETURN(auto outputTensor, getOrCreateTensor(output));

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);

  auto node =
      std::make_unique<IotaNode>(outputTensor, iotaOp.getDimension(), opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createReduceNode(mlir::Operation *op) {
  auto reduceOp = llvm::cast<nv_tensor_ir::ReduceOp>(op);
  std::vector<int64_t> dimensions(reduceOp.getDimensions().begin(),
                                  reduceOp.getDimensions().end());
  return createSingleResultNode(
      op, /*expectedOperands=*/1,
      [dimensions = std::move(dimensions), mode = reduceOp.getReductionMode()](
          llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> tensors,
          const std::string &opStr) -> std::unique_ptr<Node> {
        return std::make_unique<ReduceNode>(tensors[0], tensors[1], dimensions,
                                            mode, opStr);
      });
}

Status ReferenceGraph::createSingleResultNode(
    mlir::Operation *op, unsigned expectedOperands,
    llvm::function_ref<std::unique_ptr<Node>(
        llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>>, const std::string &)>
        nodeFactory) {
  std::string opName = op->getName().getStringRef().str();

  if (op->getNumOperands() != expectedOperands) {
    return Status::InvalidArgument(opName + " operation expects " +
                                   std::to_string(expectedOperands) +
                                   " operands");
  }
  if (op->getNumResults() != 1) {
    return Status::InvalidArgument(opName + " operation expects 1 result");
  }

  llvm::outs() << "\n=== Creating " << opName << " Node ==="
               << "\n";

  std::vector<mlir::Value> values;
  values.reserve(expectedOperands + 1);
  for (mlir::Value operand : op->getOperands()) {
    values.push_back(operand);
  }
  values.push_back(op->getResult(0));

  for (auto [idx, value] : llvm::enumerate(values)) {
    std::string label;
    if (expectedOperands == 1) {
      label = idx == 0 ? "Input" : "Output";
    } else {
      label = "Tensor ";
      label.push_back(static_cast<char>('A' + idx));
    }

    llvm::outs() << label << " shape: ";
    printTensorShape(parseTensorShape(value.getType()));
    llvm::outs() << "\n";
  }

  std::vector<std::shared_ptr<SimplifiedTensor>> tensors;
  tensors.reserve(values.size());
  for (mlir::Value value : values) {
    TIR_ASSIGN_OR_RETURN(auto tensor, getOrCreateTensor(value));
    tensors.push_back(std::move(tensor));
  }

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);
  os.flush();

  auto node = nodeFactory(tensors, opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createUnaryPointwiseNode(mlir::Operation *op,
                                                UnaryPointwiseMode mode) {
  return createSingleResultNode(
      op, /*expectedOperands=*/1,
      [mode](llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> tensors,
             const std::string &opStr) -> std::unique_ptr<Node> {
        return std::make_unique<UnaryPointwiseNode>(tensors[0], tensors[1],
                                                    mode, opStr);
      });
}

Status ReferenceGraph::createParametricUnaryPointwiseNode(
    mlir::Operation *op, ParametricUnaryPointwiseMode mode, double beta) {
  return createSingleResultNode(
      op, /*expectedOperands=*/1,
      [mode, beta](llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> tensors,
                   const std::string &opStr) -> std::unique_ptr<Node> {
        return std::make_unique<ParametricUnaryPointwiseNode>(
            tensors[0], tensors[1], mode, beta, opStr);
      });
}

Status ReferenceGraph::createParametricBinaryPointwiseNode(
    mlir::Operation *op, ParametricBinaryPointwiseMode mode, double beta) {
  return createSingleResultNode(
      op, /*expectedOperands=*/2,
      [mode, beta](llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> tensors,
                   const std::string &opStr) -> std::unique_ptr<Node> {
        return std::make_unique<ParametricBinaryPointwiseNode>(
            tensors[0], tensors[1], tensors[2], mode, beta, opStr);
      });
}

Status ReferenceGraph::createBinaryPointwiseNode(mlir::Operation *op,
                                                 BinaryPointwiseMode mode) {
  return createSingleResultNode(
      op, /*expectedOperands=*/2,
      [mode](llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> tensors,
             const std::string &opStr) -> std::unique_ptr<Node> {
        return std::make_unique<BinaryPointwiseNode>(tensors[0], tensors[1],
                                                     tensors[2], mode, opStr);
      });
}

Status ReferenceGraph::createCompareNode(mlir::Operation *op) {
  auto cmpOp = llvm::cast<nv_tensor_ir::CmpOp>(op);
  return createSingleResultNode(
      op, /*expectedOperands=*/2,
      [comparator = cmpOp.getComparator()](
          llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> tensors,
          const std::string &opStr) -> std::unique_ptr<Node> {
        return std::make_unique<CompareNode>(tensors[0], tensors[1], tensors[2],
                                             comparator, opStr);
      });
}

Status ReferenceGraph::createBinarySelectNode(mlir::Operation *op) {
  auto selectOp = llvm::cast<nv_tensor_ir::BinarySelectOp>(op);
  std::string opName = op->getName().getStringRef().str();

  llvm::outs() << "\n=== Creating " << opName << " Node ===\n";

  mlir::Value selector = selectOp.getSelector();
  mlir::Value lhs = selectOp.getLhs();
  mlir::Value rhs = selectOp.getRhs();
  mlir::Value output = selectOp.getOutput();

  std::vector<int64_t> selectorShape = parseTensorShape(selector.getType());
  std::vector<int64_t> lhsShape = parseTensorShape(lhs.getType());
  std::vector<int64_t> rhsShape = parseTensorShape(rhs.getType());
  std::vector<int64_t> outputShape = parseTensorShape(output.getType());

  llvm::outs() << "Selector shape: ";
  printTensorShape(selectorShape);
  llvm::outs() << "\n";

  llvm::outs() << "Tensor A shape: ";
  printTensorShape(lhsShape);
  llvm::outs() << "\n";

  llvm::outs() << "Tensor B shape: ";
  printTensorShape(rhsShape);
  llvm::outs() << "\n";

  llvm::outs() << "Output shape: ";
  printTensorShape(outputShape);
  llvm::outs() << "\n";

  TIR_ASSIGN_OR_RETURN(auto selectorTensor, getOrCreateTensor(selector));
  TIR_ASSIGN_OR_RETURN(auto tensorA, getOrCreateTensor(lhs));
  TIR_ASSIGN_OR_RETURN(auto tensorB, getOrCreateTensor(rhs));
  TIR_ASSIGN_OR_RETURN(auto outputTensor, getOrCreateTensor(output));

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);

  auto node = std::make_unique<BinarySelectNode>(selectorTensor, tensorA,
                                                 tensorB, outputTensor, opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createConcatenateNode(mlir::Operation *op) {
  auto concatOp = llvm::cast<nv_tensor_ir::ConcatenateOp>(op);

  std::string opName = op->getName().getStringRef().str();
  llvm::outs() << "\n=== Creating " << opName << " Node ==="
               << "\n";

  std::vector<std::shared_ptr<SimplifiedTensor>> inputs;
  inputs.reserve(concatOp.getInputs().size());
  for (auto [idx, input] : llvm::enumerate(concatOp.getInputs())) {
    std::vector<int64_t> shape = parseTensorShape(input.getType());
    llvm::outs() << "Input " << idx << " shape: ";
    printTensorShape(shape);
    llvm::outs() << "\n";

    TIR_ASSIGN_OR_RETURN(auto tensor, getOrCreateTensor(input));
    inputs.push_back(tensor);
  }

  mlir::Value output = concatOp.getOutput();
  std::vector<int64_t> outputShape = parseTensorShape(output.getType());
  llvm::outs() << "Output shape: ";
  printTensorShape(outputShape);
  llvm::outs() << "\n";

  TIR_ASSIGN_OR_RETURN(auto outputTensor, getOrCreateTensor(output));

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);

  auto node = std::make_unique<ConcatenateNode>(std::move(inputs), outputTensor,
                                                concatOp.getDimension(), opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createSliceNode(mlir::Operation *op) {
  auto sliceOp = llvm::cast<nv_tensor_ir::SliceOp>(op);
  std::string opName = op->getName().getStringRef().str();

  llvm::outs() << "\n=== Creating " << opName << " Node ==="
               << "\n";

  mlir::Value input = sliceOp.getInput();
  mlir::Value output = sliceOp.getOutput();

  std::vector<int64_t> inputShape = parseTensorShape(input.getType());
  std::vector<int64_t> outputShape = parseTensorShape(output.getType());

  llvm::outs() << "Input shape: ";
  printTensorShape(inputShape);
  llvm::outs() << "\n";

  llvm::outs() << "Output shape: ";
  printTensorShape(outputShape);
  llvm::outs() << "\n";

  TIR_ASSIGN_OR_RETURN(auto inputTensor, getOrCreateTensor(input));
  TIR_ASSIGN_OR_RETURN(auto outputTensor, getOrCreateTensor(output));

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);

  auto starts = sliceOp.getStarts();
  auto limits = sliceOp.getLimits();
  auto strides = sliceOp.getStrides();
  auto node = std::make_unique<SliceNode>(
      inputTensor, outputTensor,
      std::vector<int64_t>(starts.begin(), starts.end()),
      std::vector<int64_t>(limits.begin(), limits.end()),
      std::vector<int64_t>(strides.begin(), strides.end()), opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createReshapeNode(mlir::Operation *op) {
  auto reshapeOp = llvm::cast<nv_tensor_ir::ReshapeOp>(op);
  std::string opName = op->getName().getStringRef().str();

  llvm::outs() << "\n=== Creating " << opName << " Node ==="
               << "\n";

  mlir::Value input = reshapeOp.getInput();
  mlir::Value output = reshapeOp.getOutput();

  std::vector<int64_t> inputShape = parseTensorShape(input.getType());
  std::vector<int64_t> outputShape = parseTensorShape(output.getType());

  llvm::outs() << "Input shape: ";
  printTensorShape(inputShape);
  llvm::outs() << "\n";

  llvm::outs() << "Output shape: ";
  printTensorShape(outputShape);
  llvm::outs() << "\n";

  TIR_ASSIGN_OR_RETURN(auto inputTensor, getOrCreateTensor(input));
  TIR_ASSIGN_OR_RETURN(auto outputTensor, getOrCreateTensor(output));

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);

  auto node = std::make_unique<ReshapeNode>(inputTensor, outputTensor, opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createBroadcastNode(mlir::Operation *op) {
  auto broadcastOp = llvm::cast<nv_tensor_ir::BroadcastOp>(op);
  std::string opName = op->getName().getStringRef().str();

  llvm::outs() << "\n=== Creating " << opName << " Node ==="
               << "\n";

  mlir::Value input = broadcastOp.getInput();
  mlir::Value output = broadcastOp.getOutput();

  std::vector<int64_t> inputShape = parseTensorShape(input.getType());
  std::vector<int64_t> outputShape = parseTensorShape(output.getType());

  llvm::outs() << "Input shape: ";
  printTensorShape(inputShape);
  llvm::outs() << "\n";

  llvm::outs() << "Output shape: ";
  printTensorShape(outputShape);
  llvm::outs() << "\n";

  TIR_ASSIGN_OR_RETURN(auto inputTensor, getOrCreateTensor(input));
  TIR_ASSIGN_OR_RETURN(auto outputTensor, getOrCreateTensor(output));

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);

  auto node = std::make_unique<BroadcastNode>(inputTensor, outputTensor, opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createTransposeNode(mlir::Operation *op) {
  auto transposeOp = llvm::cast<nv_tensor_ir::TransposeOp>(op);
  std::string opName = op->getName().getStringRef().str();

  llvm::outs() << "\n=== Creating " << opName << " Node ==="
               << "\n";

  mlir::Value input = transposeOp.getInput();
  mlir::Value output = transposeOp.getOutput();

  std::vector<int64_t> inputShape = parseTensorShape(input.getType());
  std::vector<int64_t> outputShape = parseTensorShape(output.getType());

  llvm::outs() << "Input shape: ";
  printTensorShape(inputShape);
  llvm::outs() << "\n";

  llvm::outs() << "Output shape: ";
  printTensorShape(outputShape);
  llvm::outs() << "\n";

  TIR_ASSIGN_OR_RETURN(auto inputTensor, getOrCreateTensor(input));
  TIR_ASSIGN_OR_RETURN(auto outputTensor, getOrCreateTensor(output));

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);

  auto permutation = transposeOp.getPermutation();
  auto node = std::make_unique<TransposeNode>(
      inputTensor, outputTensor,
      std::vector<int64_t>(permutation.begin(), permutation.end()), opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

Status ReferenceGraph::createConvertNode(mlir::Operation *op) {
  std::string opName = op->getName().getStringRef().str();

  if (op->getNumOperands() != 1) {
    return Status::InvalidArgument(opName + " operation expects 1 operand");
  }
  if (op->getNumResults() != 1) {
    return Status::InvalidArgument(opName + " operation expects 1 result");
  }

  mlir::Value input = op->getOperand(0);
  mlir::Value output = op->getResult(0);

  llvm::outs() << "\n=== Creating " << opName << " Node ==="
               << "\n";

  std::vector<int64_t> inputShape = parseTensorShape(input.getType());
  std::vector<int64_t> outputShape = parseTensorShape(output.getType());

  llvm::outs() << "Input shape: ";
  printTensorShape(inputShape);
  llvm::outs() << "\n";

  llvm::outs() << "Output shape: ";
  printTensorShape(outputShape);
  llvm::outs() << "\n";

  TIR_ASSIGN_OR_RETURN(auto inputTensor, getOrCreateTensor(input));
  TIR_ASSIGN_OR_RETURN(auto outputTensor, getOrCreateTensor(output));

  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);

  auto node = std::make_unique<ConvertNode>(inputTensor, outputTensor, opStr);
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " + opName +
                                   " node: " + status.message());
  }
  nodes.push_back(std::move(node));

  return Status::Ok();
}

// ============================================================================
// Tensor Management
// ============================================================================

StatusOr<std::shared_ptr<SimplifiedTensor>>
ReferenceGraph::getOrCreateTensor(mlir::Value value,
                                  llvm::ArrayRef<int64_t> strides) {
  // Check if tensor already exists
  void *valuePtr = value.getAsOpaquePointer();
  auto it = valueToName.find(valuePtr);

  if (it != valueToName.end()) {
    // Tensor already exists
    return tensors[it->second];
  }

  // Create new tensor with unique name
  std::string name = generateUniqueTensorName(value);
  mlir::Type type = value.getType();

  TIR_ASSIGN_OR_RETURN(auto tensor,
                       createTensorFromMLIRType(name, type, strides));

  // Register tensor
  tensors[name] = tensor;
  valueToName[valuePtr] = name;

  return tensor;
}

std::string ReferenceGraph::generateUniqueTensorName(mlir::Value value) {
  // Try to get the name from block argument
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    std::ostringstream oss;
    oss << "arg" << blockArg.getArgNumber();
    return oss.str();
  }

  // For operation results, use result number
  if (auto opResult = llvm::dyn_cast<mlir::OpResult>(value)) {
    mlir::Operation *defOp = opResult.getOwner();

    // Try to get a meaningful name from the operation
    std::string opName = defOp->getName().getStringRef().str();

    // Remove "nv_tensor_ir." prefix if present
    if (opName.find("nv_tensor_ir.") == 0) {
      opName = opName.substr(13); // Length of "nv_tensor_ir."
    }

    // Generate base name with result number
    std::ostringstream oss;
    oss << opName << "Result" << opResult.getResultNumber();
    std::string baseName = oss.str();

    // Check if name already exists, if so add a unique suffix
    std::string uniqueName = baseName;
    int suffix = 0;
    for (const auto tensorsEnd = tensors.end();
         tensors.find(uniqueName) != tensorsEnd; ++suffix) {
      std::ostringstream uniqueOss;
      uniqueOss << baseName << suffix;
      uniqueName = uniqueOss.str();
    }

    return uniqueName;
  }

  // Fallback: generate unique name
  std::ostringstream oss;
  oss << "tensor" << tensorNameCounter++;
  return oss.str();
}

void ReferenceGraph::printTensorShape(llvm::ArrayRef<int64_t> dims) const {
  llvm::outs() << "[";
  for (size_t i = 0, e = dims.size(); i != e; ++i) {
    if (i > 0) {
      llvm::outs() << ", ";
    }
    if (dims[i] == mlir::ShapedType::kDynamic) {
      llvm::outs() << "dynamic";
    } else {
      llvm::outs() << dims[i];
    }
  }
  llvm::outs() << "]";
}

StatusOr<std::shared_ptr<SimplifiedTensor>>
ReferenceGraph::createTensorFromMLIRType(
    const std::string &name, mlir::Type mlirType,
    llvm::ArrayRef<int64_t> stridePattern) {
  // Parse shape and data type directly from MLIR type
  std::vector<int64_t> dims = parseTensorShape(mlirType);

  if (dims.empty()) {
    return Status::InvalidArgument(
        "Unsupported MLIR type for reference tensor " + name);
  }

  TIR_ASSIGN_OR_RETURN(DataType dtype, parseDataType(mlirType));

  // Resolve dynamic dimensions from the user-provided runtime values.
  for (size_t i = 0, e = dims.size(); i != e; ++i) {
    if (dims[i] == mlir::ShapedType::kDynamic) {
      if (dynamicDims.empty()) {
        return Status::InvalidArgument(
            "Reference tensor " + name + " has dynamic dimension at position " +
            std::to_string(i) + " but no --dynamic-dims were provided");
      }
      // Map dim position -> runtime value; reuse last value if out of range.
      size_t idx = std::min(i, dynamicDims.size() - 1);
      dims[i] = dynamicDims[idx];
      llvm::outs() << "  Resolved dynamic dim " << i << " -> " << dims[i]
                   << " for tensor " << name << "\n";
    }
  }

  // Resolve strides from the MLIR stride attribute (if provided).
  // Dynamic stride entries are computed assuming contiguous memory layout.
  std::vector<int64_t> resolvedStrides;
  if (!stridePattern.empty()) {
    resolvedStrides = resolveStrides(
        dims, {stridePattern.begin(), stridePattern.end()}, dynamicStrides);
    if (!resolvedStrides.empty()) {
      llvm::outs() << "  Resolved strides for " << name << ": [";
      for (size_t i = 0, e = resolvedStrides.size(); i != e; ++i) {
        if (i > 0) {
          llvm::outs() << ", ";
        }
        llvm::outs() << resolvedStrides[i];
      }
      llvm::outs() << "]"
                   << "\n";
    }
  }

  std::shared_ptr<SimplifiedTensor> tensor;
  if (!resolvedStrides.empty()) {
    tensor = createTensor(name, dims, resolvedStrides, dtype);
  } else {
    tensor = createTensor(name, dims, dtype);
  }

  if (!tensor) {
    return Status::InvalidArgument("Failed to create reference tensor " + name +
                                   " with data type " + getEnumName(dtype));
  }

  return tensor;
}

std::vector<int64_t> ReferenceGraph::parseTensorShape(mlir::Type mlirType) {
  std::vector<int64_t> dims;

  if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(mlirType)) {
    llvm::ArrayRef<int64_t> shape = tensorType.getShape();
    dims.assign(shape.begin(), shape.end());
  }
  // Scalar constants are represented as one-element internal tensors in the
  // reference graph because SimplifiedTensor does not model rank-0 buffers.
  else if (mlirType.isIntOrFloat()) {
    dims.push_back(1);
  }
  // Unhandled type
  else {
    llvm::errs() << "Warning: Unhandled tensor type in parseTensorShape"
                 << "\n";
    return {};
  }

  // Check for dynamic dimensions (common for both types)
  for (size_t i = 0, e = dims.size(); i != e; ++i) {
    if (dims[i] == mlir::ShapedType::kDynamic) {
      llvm::errs() << "Warning: Dynamic dimension found at position " << i
                   << ", keeping as dynamic"
                   << "\n";
    }
  }

  return dims;
}

StatusOr<DataType> ReferenceGraph::parseDataType(mlir::Type mlirType) {
  // Get element type
  mlir::Type elementType;

  if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(mlirType)) {
    elementType = tensorType.getElementType();
  } else {
    elementType = mlirType;
  }

  // Map MLIR types to DataType enum.
  if (elementType.isF32()) {
    return DataType::FLOAT32;
  }
  if (elementType.isF16()) {
    return DataType::FLOAT16;
  }
  if (elementType.isBF16()) {
    return DataType::BFLOAT16;
  }
  if (llvm::isa<mlir::Float8E4M3FNType>(elementType)) {
    return DataType::F8E4M3FN;
  }
  if (llvm::isa<mlir::Float8E5M2Type>(elementType)) {
    return DataType::F8E5M2;
  }
  if (elementType.isF64()) {
    return DataType::DOUBLE;
  }
  if (auto integerType = llvm::dyn_cast<mlir::IntegerType>(elementType)) {
    if (integerType.getWidth() == 1) {
      return DataType::BOOL;
    }
    if (integerType.getWidth() == 8) {
      return integerType.isUnsigned() ? DataType::UINT8 : DataType::INT8;
    }
    if (integerType.getWidth() == 32) {
      return integerType.isUnsigned() ? DataType::UINT32 : DataType::INT32;
    }
  }

  std::string elementTypeString;
  llvm::raw_string_ostream os(elementTypeString);
  os << elementType;
  os.flush();
  return Status::NotSupported("Unsupported MLIR element type " +
                              elementTypeString);
}

// ============================================================================
// Graph Execution
// ============================================================================

Status ReferenceGraph::validate() const {
  for (const auto &node : nodes) {
    Status status = node->validate();
    if (!status.ok()) {
      llvm::errs() << "Node validation failed: " << node->getOpName() << " - "
                   << status.message() << "\n";
      return status;
    }
  }
  return Status::Ok();
}

Status ReferenceGraph::execute() {
  // Validate first
  Status status = validate();
  if (!status.ok()) {
    return status;
  }

  // Execute each node in order
  for (auto &node : nodes) {
    status = node->computeValidated();
    if (!status.ok()) {
      llvm::errs() << "Node execution failed: " << node->getOpName() << " - "
                   << status.message() << "\n";
      return status;
    }
  }

  return Status::Ok();
}

// ============================================================================
// Debugging
// ============================================================================

void ReferenceGraph::print() const {
  llvm::outs() << "=================================================="
               << "\n";
  llvm::outs() << "ReferenceGraph Structure\n";
  llvm::outs() << "=================================================="
               << "\n";
  llvm::outs() << "\n";

  llvm::outs() << "Tensors (" << tensors.size() << "):"
               << "\n";
  llvm::outs() << "--------------------------------------------------"
               << "\n";
  for (const auto &kv : tensors) {
    llvm::outs() << "  [" << kv.first << "] ";
    kv.second->print();
  }
  llvm::outs() << "\n";

  llvm::outs() << "Nodes (" << nodes.size() << "):"
               << "\n";
  llvm::outs() << "--------------------------------------------------"
               << "\n";
  for (size_t i = 0, e = nodes.size(); i != e; ++i) {
    const auto &node = nodes[i];
    llvm::outs() << "  Node " << i << ": " << node->getOpName() << "\n";
    if (!node->getOriginalOperation().empty()) {
      llvm::outs() << "    MLIR: " << node->getOriginalOperation() << "\n";
    }
  }

  llvm::outs() << "=================================================="
               << "\n";
}

Status ReferenceGraph::fillInputTensorsRandom(float minVal, float maxVal,
                                              unsigned seed) {
  std::mt19937 gen(seed);
  for (size_t i = 0, e = inputTensors.size(); i != e; ++i) {
    if (!inputTensors[i]->fillRandom(minVal, maxVal, gen)) {
      return Status::InvalidArgument(
          "Failed to fill input tensor '" + inputTensors[i]->getName() +
          "' with random data of type " +
          std::string(getEnumName(inputTensors[i]->getDataType())));
    }
  }
  return Status::Ok();
}

} // namespace mlir::nv_tensor_ir::reference
