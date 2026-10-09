// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Reference/ReferenceGraph.h"

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Reference/SimplifiedTensor.h"
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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <random>
#include <utility>

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
    attr = denseAttr.getSplatValue<Attribute>();
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

namespace {

StatusOr<std::vector<int64_t>>
resolveStrides(llvm::ArrayRef<int64_t> dims,
               llvm::ArrayRef<int64_t> stridePattern,
               llvm::ArrayRef<int64_t> userStrides) {
  if (stridePattern.empty() || dims.size() != stridePattern.size()) {
    return std::vector<int64_t>{};
  }

  std::vector<int64_t> resolved(stridePattern.begin(), stridePattern.end());

  // Work backwards so inferred strides use the already resolved inner mode.
  int64_t runningStride = 1;
  for (size_t d = resolved.size(); d-- > 0;) {
    if (resolved[d] == ShapedType::kDynamic) {
      if (!userStrides.empty() && d >= userStrides.size()) {
        return Status::InvalidArgument(
            "Dynamic stride at position " + std::to_string(d) +
            " has no corresponding --dynamic-strides value");
      }
      resolved[d] = userStrides.empty() ? runningStride : userStrides[d];
    }
    if (userStrides.empty()) {
      runningStride = resolved[d] * dims[d];
    }
  }
  return resolved;
}

void printTensorShape(llvm::ArrayRef<int64_t> dims) {
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

std::vector<int64_t> parseTensorShape(mlir::Type mlirType) {
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

StatusOr<DataType> parseDataType(mlir::Type mlirType) {
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
  if (llvm::isa<mlir::Float4E2M1FNType>(elementType)) {
    return DataType::F4E2M1FN;
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

} // namespace

Status ReferenceGraph::buildFromMLIR(mlir::ModuleOp module) {
  TIR_RETURN_IF_ERROR(buildIOTensorsFromMLIR(module));
  Status status;
  module.walk([&](Operation *op) {
    if (isa<ModuleOp, GraphOp, ResultsOp>(op)) {
      return WalkResult::advance();
    }
    status = createNode(op);
    if (!status.ok()) {
      llvm::errs() << "Failed to create " << op->getName()
                   << " node: " << status.message() << "\n";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return status;
}

Status ReferenceGraph::buildIOTensorsFromMLIR(mlir::ModuleOp module) {
  clear();
  return createIOTensor(module);
}

Status ReferenceGraph::createIOTensor(mlir::ModuleOp module) {
  GraphOp graph;
  ResultsOp results;
  module.walk([&](Operation *op) {
    if (auto typed = dyn_cast<GraphOp>(op)) {
      graph = typed;
    } else if (auto typed = dyn_cast<ResultsOp>(op)) {
      results = typed;
    }
  });
  if (!graph) {
    return Status::InvalidArgument("No nv_tensor_ir.graph operation found");
  }
  llvm::outs() << "\n=== Graph Signature Analysis ===\nGraph name: "
               << graph->getName() << "\n";

  auto collect =
      [&](ValueRange values, ArrayRef<Type> types, ArrayAttr attrs,
          std::vector<std::shared_ptr<SimplifiedTensor>> &destination,
          StringRef role) -> Status {
    auto descriptors = getTensorDescriptors(types, attrs);
    if (failed(descriptors)) {
      return Status::InvalidArgument(role.str() +
                                     " tensor descriptor parsing failed");
    }
    if (descriptors->size() != values.size()) {
      return Status::InvalidArgument(
          role.str() + " tensor descriptor count mismatch: expected " +
          std::to_string(values.size()) + ", got " +
          std::to_string(descriptors->size()));
    }
    llvm::outs() << role << " Tensors (" << values.size() << "):\n";
    for (auto [value, descriptor] : llvm::zip_equal(values, *descriptors)) {
      std::vector<int64_t> strides;
      for (const auto &stride : descriptor.strides) {
        strides.push_back(stride.staticValue);
      }
      TIR_ASSIGN_OR_RETURN(auto tensor, getOrCreateTensor(value, strides));
      destination.push_back(tensor);
      tensor->print();
    }
    return Status::Ok();
  };
  if (!graph->getRegion(0).empty()) {
    TIR_RETURN_IF_ERROR(collect(graph->getRegion(0).front().getArguments(),
                                graph.getArgumentTypes(),
                                graph.getAllArgAttrs(), inputTensors, "Input"));
  }
  if (results) {
    TIR_RETURN_IF_ERROR(collect(results.getOperands(), graph.getResultTypes(),
                                graph.getAllResultAttrs(), outputTensors,
                                "Output"));
  } else {
    llvm::outs() << "  Warning: No nv_tensor_ir.results operation found\n";
  }
  llvm::outs() << "=== End Graph Signature ===\n\n";
  return Status::Ok();
}

// ============================================================================
// Node Creation
// ============================================================================

namespace {

using TensorPtr = std::shared_ptr<SimplifiedTensor>;
using NodeResult = StatusOr<std::unique_ptr<Node>>;

/// Adapts an operation's inputs and single output to reference constructors.
struct NodeFactory {
  Operation *op;
  ArrayRef<TensorPtr> tensors;
  const std::string &operation;

  template <typename NodeT, size_t Inputs, typename... Args>
  NodeResult make(Args &&...args) const {
    if (tensors.size() != Inputs + 1) {
      return Status::InvalidArgument(op->getName().getStringRef().str() +
                                     " operation expects " +
                                     std::to_string(Inputs) + " operands");
    }
    return makeImpl<NodeT>(std::make_index_sequence<Inputs + 1>{},
                           std::forward<Args>(args)...);
  }

private:
  template <typename NodeT, size_t... Indices, typename... Args>
  NodeResult makeImpl(std::index_sequence<Indices...>, Args &&...args) const {
    return std::unique_ptr<Node>(std::make_unique<NodeT>(
        tensors[Indices]..., std::forward<Args>(args)..., operation));
  }
};

template <typename Range>
std::vector<int64_t> copyIndices(Range values) {
  return {values.begin(), values.end()};
}

NodeResult makeNode(Operation *op, const NodeFactory &factory) {
  llvm::TypeSwitch<Operation *, NodeResult> result(op);
#define BINARY_NODE(OpT, Mode)                                                 \
  result.Case<OpT>([&](auto) {                                                 \
    return factory.make<BinaryPointwiseNode, 2>(BinaryPointwiseMode::Mode);    \
  })
  BINARY_NODE(AddOp, ADD);
  BINARY_NODE(SubOp, SUB);
  BINARY_NODE(MulOp, MUL);
  BINARY_NODE(DivOp, DIV);
  BINARY_NODE(MaxOp, MAX);
  BINARY_NODE(MinOp, MIN);
  BINARY_NODE(PowOp, POW);
  BINARY_NODE(Atan2Op, ATAN2);
  BINARY_NODE(ModOp, MOD);
  BINARY_NODE(RemOp, REM);
  BINARY_NODE(AddSquareOp, ADD_SQUARE);
  BINARY_NODE(ReluBwdOp, RELU_BWD);
  BINARY_NODE(GeluBwdOp, GELU_BWD);
  BINARY_NODE(SigmoidBwdOp, SIGMOID_BWD);
  BINARY_NODE(TanhBwdOp, TANH_BWD);
  BINARY_NODE(GeluApproxTanhBwdOp, GELU_APPROX_TANH_BWD);
  BINARY_NODE(LogicalAndOp, LOGICAL_AND);
  BINARY_NODE(LogicalOrOp, LOGICAL_OR);
#undef BINARY_NODE
#define UNARY_NODE(OpT, Mode)                                                  \
  result.Case<OpT>([&](auto) {                                                 \
    return factory.make<UnaryPointwiseNode, 1>(UnaryPointwiseMode::Mode);      \
  })
  UNARY_NODE(AbsOp, ABS);
  UNARY_NODE(CeilOp, CEIL);
  UNARY_NODE(CosOp, COS);
  UNARY_NODE(ErfOp, ERF);
  UNARY_NODE(ExpOp, EXP);
  UNARY_NODE(FloorOp, FLOOR);
  UNARY_NODE(GeluFwdOp, GELU_FWD);
  UNARY_NODE(LogOp, LOG);
  UNARY_NODE(NegOp, NEG);
  UNARY_NODE(ReluFwdOp, RELU_FWD);
  UNARY_NODE(SinOp, SIN);
  UNARY_NODE(SqrtOp, SQRT);
  UNARY_NODE(RsqrtOp, RSQRT);
  UNARY_NODE(TanOp, TAN);
  UNARY_NODE(TanhFwdOp, TANH_FWD);
  UNARY_NODE(ReciprocalOp, RECIPROCAL);
  UNARY_NODE(SigmoidFwdOp, SIGMOID);
  UNARY_NODE(GeluApproxTanhFwdOp, GELU_APPROX_TANH);
  UNARY_NODE(LogicalNotOp, LOGICAL_NOT);
#undef UNARY_NODE
#define PARAMETRIC_UNARY_NODE(OpT, Mode)                                       \
  result.Case<OpT>([&](auto typedOp) {                                         \
    return factory.make<ParametricUnaryPointwiseNode, 1>(                      \
        ParametricUnaryPointwiseMode::Mode,                                    \
        typedOp.getBeta().convertToDouble());                                  \
  })
  PARAMETRIC_UNARY_NODE(SoftplusFwdOp, SOFTPLUS_FWD);
  PARAMETRIC_UNARY_NODE(SwishFwdOp, SWISH_FWD);
  PARAMETRIC_UNARY_NODE(EluFwdOp, ELU_FWD);
#undef PARAMETRIC_UNARY_NODE
#define PARAMETRIC_BINARY_NODE(OpT, Mode)                                      \
  result.Case<OpT>([&](auto typedOp) {                                         \
    return factory.make<ParametricBinaryPointwiseNode, 2>(                     \
        ParametricBinaryPointwiseMode::Mode,                                   \
        typedOp.getBeta().convertToDouble());                                  \
  })
  PARAMETRIC_BINARY_NODE(SoftplusBwdOp, SOFTPLUS_BWD);
  PARAMETRIC_BINARY_NODE(SwishBwdOp, SWISH_BWD);
  PARAMETRIC_BINARY_NODE(EluBwdOp, ELU_BWD);
#undef PARAMETRIC_BINARY_NODE

  return result
      .Case<MatmulOp>([&](auto) { return factory.make<MatmulNode, 2>(); })
      .Case<SplatOp>([&](auto) { return factory.make<SplatNode, 1>(); })
      .Case<ReshapeOp>([&](auto) { return factory.make<ReshapeNode, 1>(); })
      .Case<BroadcastOp>([&](auto) { return factory.make<BroadcastNode, 1>(); })
      .Case<ConvertOp>([&](auto) { return factory.make<ConvertNode, 1>(); })
      .Case<BinarySelectOp>(
          [&](auto) { return factory.make<BinarySelectNode, 3>(); })
      .Case<ConstantOp>([&](auto constant) -> NodeResult {
        TIR_ASSIGN_OR_RETURN(
            double value,
            getScalarConstantValue(constant.getValue(),
                                   factory.tensors.back()->getDataType()));
        return factory.make<ConstantNode, 0>(value);
      })
      .Case<IotaOp>([&](auto iota) {
        return factory.make<IotaNode, 0>(iota.getDimension());
      })
      .Case<ReduceOp>([&](auto reduce) {
        return factory.make<ReduceNode, 1>(copyIndices(reduce.getDimensions()),
                                           reduce.getReductionMode());
      })
      .Case<CmpOp>([&](auto cmp) {
        return factory.make<CompareNode, 2>(cmp.getComparator());
      })
      .Case<SliceOp>([&](auto slice) {
        return factory.make<SliceNode, 1>(copyIndices(slice.getStarts()),
                                          copyIndices(slice.getLimits()),
                                          copyIndices(slice.getStrides()));
      })
      .Case<TransposeOp>([&](auto transpose) {
        return factory.make<TransposeNode, 1>(
            copyIndices(transpose.getPermutation()));
      })
      .Case<ConcatenateOp>([&](auto concat) -> NodeResult {
        auto inputs = factory.tensors.drop_back();
        return std::unique_ptr<Node>(std::make_unique<ConcatenateNode>(
            std::vector<TensorPtr>(inputs.begin(), inputs.end()),
            factory.tensors.back(), concat.getDimension(), factory.operation));
      })
      .Default([&](Operation *) -> NodeResult {
        return Status::NotSupported(
            "Unsupported operation in reference graph: " +
            op->getName().getStringRef().str());
      });
}

} // namespace

Status ReferenceGraph::createNode(Operation *op) {
  if (op->getNumResults() != 1) {
    return Status::NotSupported("Unsupported operation in reference graph: " +
                                op->getName().getStringRef().str());
  }
  llvm::outs() << "\n=== Creating " << op->getName() << " Node ===\n";
  SmallVector<TensorPtr> nodeTensors;
  auto collect = [&](ValueRange values, StringRef role) -> Status {
    for (auto [index, value] : llvm::enumerate(values)) {
      llvm::outs() << role << " " << index << " shape: ";
      printTensorShape(parseTensorShape(value.getType()));
      llvm::outs() << "\n";
      TIR_ASSIGN_OR_RETURN(auto tensor, getOrCreateTensor(value));
      nodeTensors.push_back(std::move(tensor));
    }
    return Status::Ok();
  };
  TIR_RETURN_IF_ERROR(collect(op->getOperands(), "Input"));
  TIR_RETURN_IF_ERROR(collect(op->getResults(), "Output"));

  std::string operation;
  llvm::raw_string_ostream os(operation);
  op->print(os);
  TIR_ASSIGN_OR_RETURN(auto node, makeNode(op, {op, nodeTensors, operation}));
  Status status = node->validate();
  if (!status.ok()) {
    return Status::InvalidArgument("Invalid arguments for " +
                                   op->getName().getStringRef().str() +
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
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    return "arg" + std::to_string(arg.getArgNumber());
  }
  if (auto result = dyn_cast<OpResult>(value)) {
    StringRef opName = result.getOwner()->getName().getStringRef();
    opName.consume_front("nv_tensor_ir.");
    std::string base =
        opName.str() + "Result" + std::to_string(result.getResultNumber());
    std::string name = base;
    for (unsigned suffix = 0; tensors.count(name); ++suffix) {
      name = base + std::to_string(suffix);
    }
    return name;
  }
  return "tensor" + std::to_string(tensorNameCounter++);
}

StatusOr<std::shared_ptr<SimplifiedTensor>>
ReferenceGraph::createTensorFromMLIRType(
    const std::string &name, mlir::Type mlirType,
    llvm::ArrayRef<int64_t> stridePattern) {
  // Parse shape and data type directly from MLIR type
  std::vector<int64_t> dims = parseTensorShape(mlirType);

  // An empty shape is valid for a ranked rank-zero tensor. For other types,
  // parseTensorShape returning empty still means the type is unsupported.
  if (dims.empty() && !llvm::isa<mlir::RankedTensorType>(mlirType)) {
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
    TIR_ASSIGN_OR_RETURN(
        resolvedStrides,
        resolveStrides(dims, {stridePattern.begin(), stridePattern.end()},
                       dynamicStrides));
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
