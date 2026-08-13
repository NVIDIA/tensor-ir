// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <limits>
#include <utility>

#define DEBUG_TYPE "tensor-ops"

using namespace mlir;
using namespace mlir::nv_tensor_ir;
namespace tcg = mlir::nv_tensor_ir::tcutegen;

bool mlir::nv_tensor_ir::isGraphInput(Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg) {
    return false;
  }
  return isa<GraphOp>(blockArg.getOwner()->getParentOp());
}

bool mlir::nv_tensor_ir::isGraphOutput(Value value) {
  return llvm::any_of(value.getUsers(),
                      [](Operation *user) { return isa<ResultsOp>(user); });
}

FailureOr<int64_t> mlir::nv_tensor_ir::mergeBroadcastDimensions(int64_t lhs,
                                                                int64_t rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  if (lhs == 1) {
    return rhs;
  }
  if (rhs == 1) {
    return lhs;
  }
  if (ShapedType::isDynamic(lhs)) {
    return rhs > 1 ? rhs : ShapedType::kDynamic;
  }
  if (ShapedType::isDynamic(rhs)) {
    return lhs > 1 ? lhs : ShapedType::kDynamic;
  }
  return failure();
}

FailureOr<int64_t> mlir::nv_tensor_ir::mergeEqualDimensions(int64_t lhs,
                                                            int64_t rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  if (ShapedType::isDynamic(lhs)) {
    return rhs;
  }
  if (ShapedType::isDynamic(rhs)) {
    return lhs;
  }
  return failure();
}

LogicalResult mlir::nv_tensor_ir::verifyMatmulShapes(Operation *op,
                                                     TensorType aType,
                                                     TensorType bType,
                                                     TensorType cType) {
  if (aType.getRank() < 2 || bType.getRank() != aType.getRank() ||
      cType.getRank() != aType.getRank()) {
    if (aType.getRank() < 2) {
      return op->emitOpError()
             << "expects rank of at least 2, but got " << aType.getRank();
    }
    return op->emitOpError("expects A, B, and C to have the same rank");
  }

  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  ArrayRef<int64_t> cShape = cType.getShape();
  unsigned rank = aType.getRank();
  ArrayRef<int64_t> aBatchShape = aShape.drop_back(2);
  ArrayRef<int64_t> bBatchShape = bShape.drop_back(2);
  ArrayRef<int64_t> cBatchShape = cShape.drop_back(2);
  for (auto [aDim, bDim, cDim] :
       llvm::zip_equal(aBatchShape, bBatchShape, cBatchShape)) {
    FailureOr<int64_t> expected = mergeBroadcastDimensions(aDim, bDim);
    if (failed(expected) || cDim != *expected) {
      return op->emitOpError()
             << "expects C batch dimensions to be the broadcasted shape of A "
                "and B, but got "
             << aBatchShape << ", " << bBatchShape << ", and " << cBatchShape;
    }
  }

  if (failed(mergeEqualDimensions(aShape[rank - 1], bShape[rank - 2]))) {
    return op->emitOpError()
           << "expects the contracting dimensions of A and B to match, but got "
           << aShape[rank - 1] << " and " << bShape[rank - 2];
  }

  if (aShape[rank - 2] != cShape[rank - 2] ||
      bShape[rank - 1] != cShape[rank - 1]) {
    return op->emitOpError()
           << "expects C's matrix dimensions to be inherited exactly from "
              "A's rows and B's columns, but got A, B, and C shapes "
           << aType << ", " << bType << ", and " << cType;
  }

  return success();
}

LogicalResult
mlir::nv_tensor_ir::detail::verifyDynamicDimsOpInterface(Operation *op) {
  auto dynamicDimsOp = cast<DynamicDimsOpInterface>(op);
  Operation::operand_range dynamicSizes = dynamicDimsOp.getDynamicDims();
  if (dynamicSizes.empty()) {
    return success();
  }
  size_t expectedCount = 0;
  for (Value result : op->getResults()) {
    expectedCount += cast<ShapedType>(result.getType()).getNumDynamicDims();
  }
  if (dynamicSizes.size() != expectedCount) {
    return op->emitOpError()
           << "requires either no dynamic_dims operands or one for every "
              "dynamic result dimension, but got "
           << dynamicSizes.size() << " operand(s) for " << expectedCount
           << " dynamic dimension(s)";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Layout transformation helpers
//===----------------------------------------------------------------------===//

static ParseResult parseReductionBody(OpAsmParser &parser, Region &body);
static void printReductionBody(OpAsmPrinter &printer, Operation *,
                               Region &body);

#define GET_OP_CLASSES
#include "tensor_ir/Dialect/TensorOps.cpp.inc"

// enum attribute definitions
#include "tensor_ir/Dialect/TensorEnums.cpp.inc"

void GraphOp::getAsmBlockArgumentNames(Region &region,
                                       OpAsmSetValueNameFn setNameFn) {
  // TODO: use an attribute on the graph to name the inputs.
  for (Value v : region.front().getArguments()) {
    setNameFn(v, "in");
  }
}

LogicalResult GraphOp::verify() {
  auto verifyGraphType = [&](Type type, StringRef kind,
                             unsigned idx) -> LogicalResult {
    if (isa<RankedTensorType>(type) && !isTensorType(type)) {
      return emitOpError() << "expects " << kind << " #" << idx
                           << " to be a TensorIR ranked tensor, but got "
                           << type;
    }
    return success();
  };

  for (auto [idx, type] : llvm::enumerate(getArgumentTypes())) {
    if (failed(verifyGraphType(type, "argument", idx))) {
      return failure();
    }
  }
  for (auto [idx, type] : llvm::enumerate(getResultTypes())) {
    if (failed(verifyGraphType(type, "result", idx))) {
      return failure();
    }
  }

  return success();
}

LogicalResult GraphOp::verifyRegions() {
  // The result types should match the return value types.
  ArrayRef<Type> resTys = getResultTypes();
  auto resVals = getResults();
  if (resTys.size() != resVals.size()) {
    return emitOpError() << "expects the # of ResultTypes should match the # "
                            "of return value type, but got "
                         << resTys.size() << " and " << resVals.size();
  }

  for (auto [idx, val] : llvm::enumerate(resVals)) {
    if (val.getType() != resTys[idx]) {
      return emitOpError() << "expects ResultTypes[" << idx
                           << "] should match return value type, but got "
                           << resTys[idx] << " and " << val.getType();
    }
  }
  return success();
}

void GraphOp::print(OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

ParseResult GraphOp::parse(OpAsmParser &parser, OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

static ParseResult parseReductionBody(OpAsmParser &parser, Region &body) {
  SmallVector<OpAsmParser::Argument> arguments;
  if (failed(parser.parseArgumentList(arguments, OpAsmParser::Delimiter::Paren,
                                      /*allowType=*/true))) {
    return failure();
  }
  return parser.parseRegion(body, arguments);
}

static void printReductionBody(OpAsmPrinter &printer, Operation *,
                               Region &body) {
  printer << "(";
  llvm::interleaveComma(body.getArguments(), printer, [&](BlockArgument arg) {
    printer.printRegionArgument(arg);
  });
  printer << ") ";
  printer.printRegion(body, /*printEntryBlockArgs=*/false);
}

namespace {

LogicalResult verifyReductionShape(Operation *op, ArrayRef<int64_t> inputShape,
                                   nv_tensor_ir::TensorType outputType,
                                   ArrayRef<int32_t> dimensions,
                                   bool verifyRank) {
  if (verifyRank &&
      static_cast<int64_t>(inputShape.size()) != outputType.getRank()) {
    return op->emitOpError()
           << "expects input and output to have the same rank, but got "
           << inputShape.size() << " and " << outputType.getRank();
  }
  SmallVector<bool> isReductionDimension(inputShape.size(), false);
  for (int32_t dimension : dimensions) {
    if (dimension < 0 || static_cast<size_t>(dimension) >= inputShape.size()) {
      return op->emitOpError()
             << "reduction dimension " << dimension
             << " is out of bounds for tensor rank " << inputShape.size();
    }
    if (isReductionDimension[dimension]) {
      return op->emitOpError()
             << "reduction dimension " << dimension << " is duplicated";
    }
    isReductionDimension[dimension] = true;
  }

  for (int64_t dimension = 0;
       static_cast<size_t>(dimension) < inputShape.size(); ++dimension) {
    int64_t expected =
        isReductionDimension[dimension] ? 1 : inputShape[dimension];
    int64_t actual = outputType.getDimSize(dimension);
    // A dynamic result may be a less-refined declaration at a graph
    // signature. Otherwise preserve dynamic/static information exactly.
    if (!ShapedType::isDynamic(actual) && actual != expected) {
      if (ShapedType::isDynamic(expected)) {
        return op->emitOpError()
               << "expects output dimension " << dimension
               << " to remain dynamic because the corresponding input "
                  "dimension is dynamic, but got "
               << outputType;
      }
      return op->emitOpError()
             << "expects output dimension " << dimension << " to be "
             << expected << ", but got " << actual;
    }
  }
  return success();
}

} // namespace

LogicalResult ReduceOp::verify() {
  return verifyReductionShape(getOperation(), getInput().getType().getShape(),
                              getOutput().getType(), getDimensions(),
                              /*verifyRank=*/false);
}

LogicalResult ReduceUDOp::verify() {
  // AtLeastNOperands cannot be combined with the VariadicOperands trait
  // generated by ODS, because both provide the same multi-operand API.
  if (getNumOperands() == 0) {
    return emitOpError("requires at least one input");
  }

  // For ReduceUDOp, we expect the same number of inputs and outputs
  if (getNumOperands() != getNumResults()) {
    return emitOpError()
           << "expects number of operands to match number of results, but got "
           << getNumOperands() << " operands and " << getNumResults()
           << " results";
  }

  // Check that identity size matches the number of results
  auto identityAttr = getIdentityAttr();
  if (identityAttr.size() != getNumResults()) {
    return emitOpError() << "expects number of identities to match number of "
                            "results, but got "
                         << identityAttr.size() << " identities and "
                         << getNumResults() << " results";
  }

  auto firstInputType = cast<nv_tensor_ir::TensorType>(getOperand(0).getType());
  SmallVector<int64_t> mergedInputShape(firstInputType.getShape());

  // Equal-merge all operand dimensions before checking results, so a static
  // operand can refine a corresponding dynamic operand dimension.
  for (size_t i = 0; i < getNumOperands(); ++i) {
    auto inputType = cast<nv_tensor_ir::TensorType>(getOperand(i).getType());
    if (inputType.getRank() != firstInputType.getRank()) {
      return emitOpError() << "expects all inputs to have the same rank, but "
                           << "input 0 has type " << firstInputType
                           << " and input " << i << " has type " << inputType;
    }
    for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
      FailureOr<int64_t> merged = mergeEqualDimensions(
          mergedInputShape[dim], inputType.getDimSize(dim));
      if (failed(merged)) {
        return emitOpError()
               << "expects all input dimensions to merge, but input 0 has "
                  "type "
               << firstInputType << " and input " << i << " has type "
               << inputType;
      }
      mergedInputShape[dim] = *merged;
    }
  }

  for (size_t i = 0; i < getNumResults(); ++i) {
    auto outputType = cast<nv_tensor_ir::TensorType>(getResult(i).getType());
    if (failed(verifyReductionShape(getOperation(), mergedInputShape,
                                    outputType, getDimensions(),
                                    /*verifyRank=*/true))) {
      return failure();
    }

    // Check that identity type matches the result element type
    auto identityType = cast<TypedAttr>(identityAttr.getValue()[i]).getType();
    auto resultElementType = outputType.getElementType();

    if (identityType != resultElementType) {
      return emitOpError()
             << "expects identity type to match result element type for index "
             << i << ", but got identity type: " << identityType
             << " and result element type: " << resultElementType;
    }
  }

  return success();
}

LogicalResult ReduceUDOp::verifyRegions() {
  Block &bodyBlock = getBody().front();
  auto bodyBlockArgs = bodyBlock.getArguments();
  auto bodyBlockYields = bodyBlock.getTerminator()->getOperands();

  // The body has N accumulator arguments followed by N current-value
  // arguments and yields N updated accumulator values.
  size_t expectedNumArgs = 2 * getNumResults();
  if (bodyBlockArgs.size() != expectedNumArgs) {
    return emitOpError()
           << "expects body region to have exactly " << expectedNumArgs
           << " arguments (prev_result_i, curr_operand_i), but got "
           << bodyBlockArgs.size() << " arguments";
  }

  if (bodyBlockYields.size() != getNumResults()) {
    return emitOpError() << "expects body region to yield exactly "
                         << getNumResults() << " values, but got "
                         << bodyBlockYields.size() << " values";
  }

  auto identityAttr = getIdentityAttr();
  for (size_t i = 0; i < getNumResults(); ++i) {
    Type identityType = cast<TypedAttr>(identityAttr.getValue()[i]).getType();

    // Check that the block argument types in pair (prev_result_i,
    // curr_operand_i) matches the identity type.
    // Convert signed/unsigned integer types to signless integer types.
    auto argType_prev = bodyBlockArgs[i].getType();
    auto argType_curr = bodyBlockArgs[i + getNumOperands()].getType();
    auto yieldType = bodyBlockYields[i].getType();

    Type convertedIdentityType = identityType;
    if (identityType.isSignedInteger() || identityType.isUnsignedInteger()) {
      convertedIdentityType =
          IntegerType::get(getContext(), identityType.getIntOrFloatBitWidth());
    }

    if (argType_prev != convertedIdentityType) {
      return emitOpError() << "expects block argument " << i
                           << " type to match identity type, but got argument "
                              "type: "
                           << argType_prev
                           << " and identity type: " << convertedIdentityType;
    }
    if (argType_curr != convertedIdentityType) {
      return emitOpError()
             << "expects block argument " << i + getNumOperands()
             << " type to match identity type, but got argument type: "
             << argType_curr << " and identity type: " << convertedIdentityType;
    }

    if (yieldType != convertedIdentityType) {
      return emitOpError()
             << "expects yield operand " << i
             << " type to match identity type, but got yield type: "
             << yieldType << " and identity type: " << convertedIdentityType;
    }
  }

  return success();
}

namespace {

tcg::Layout createReductionView(ArrayRef<int32_t> dimensions,
                                ArrayRef<int64_t> shape) {
  // Build CuTe layout view of the underlying source.
  tcg::Shape cgShape(shape);
  tcg::Layout flat(cgShape);

  // Replace contracting dimensions with unit dimensions.
  std::vector<tcg::Layout> parts;
  for (size_t i = 0, n = tcg::rank(flat); i < n; ++i) {
    if (llvm::find(dimensions, i) == dimensions.end()) {
      parts.push_back(tcg::get(flat, i));
    } else {
      parts.emplace_back(1, 0);
    }
  }

  // Add contracting dimension as the last part.
  std::vector<tcg::Layout> last;
  for (size_t i : dimensions) {
    last.push_back(tcg::get(flat, i));
  }
  parts.push_back(tcg::make_layout(last));

  return tcg::make_layout(parts);
}

} // namespace

LayoutSourceAttrInterface
ReduceOp::inferResultLayout(ArrayRef<LayoutSourceAttrInterface> inputLayouts) {
  assert(inputLayouts.size() == 1 && "expects one input layout");
  SmallVector<int64_t> shape = inputLayouts[0].getShape();
  auto view = createReductionView(getDimensions(), shape);
  return ReductionSourceAttr::get(getContext(), view.toString(),
                                  inputLayouts[0]);
}

LayoutSourceAttrInterface ReduceUDOp::inferResultLayout(
    ArrayRef<LayoutSourceAttrInterface> inputLayouts) {
  assert(inputLayouts.size() == getNumOperands() &&
         "expects one input layout for each operand");
  LayoutSourceAttrInterface underlying = inputLayouts[0];
  if (inputLayouts.size() > 1) {
    underlying = CompositeSourceAttr::get(getContext(), inputLayouts);
  }
  SmallVector<int64_t> shape = underlying.getShape();
  auto view = createReductionView(getDimensions(), shape);
  return ReductionSourceAttr::get(getContext(), view.toString(), underlying);
}

void ConvertOp::build(OpBuilder &builder, OperationState &result, Value operand,
                      Type resultElementTy) {
  auto rankedTy = cast<RankedTensorType>(operand.getType());
  auto resultTy = RankedTensorType::get(rankedTy.getShape(), resultElementTy,
                                        rankedTy.getEncoding());
  build(builder, result, resultTy, operand);
}

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext *context, ::std::optional<Location> location,
    ConstantOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  // InferTypeOpAdaptor generates the long-form interface wrapper.
  auto attrType = dyn_cast<TypedAttr>(adaptor.getValueAttr()).getType();

  inferredReturnTypes.push_back(attrType);

  return success();
}

LogicalResult DimOp::verify() {
  int64_t dimension = getDimension();
  if (dimension < 0 || dimension >= getInput().getType().getRank()) {
    return emitOpError() << "dimension " << dimension
                         << " is out of bounds for tensor rank "
                         << getInput().getType().getRank();
  }
  return success();
}

LogicalResult ReshapeOp::verify() {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();

  // Verify shapes have the same number of elements (if both are static)
  if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
    int64_t inputElements = inputType.getNumElements();
    int64_t outputElements = outputType.getNumElements();

    if (inputElements != outputElements) {
      return emitOpError() << "requires input and output to have the same "
                              "number of elements, "
                           << "but got " << inputElements
                           << " elements in input and " << outputElements
                           << " elements in output";
    }
  }

  return success();
}

LayoutSourceAttrInterface
ReshapeOp::inferResultLayout(ArrayRef<LayoutSourceAttrInterface> inputLayouts) {
  assert(inputLayouts.size() == 1 && "expects one input layout");
  auto outputTy = getOutput().getType();
  return inputLayouts[0].reshape(outputTy.getShape());
}

LogicalResult BroadcastOp::verify() {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();

  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError()
           << "requires input and output element types to match, but got "
           << inputType.getElementType() << " and "
           << outputType.getElementType();
  }
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "requires input and output ranks to match, but got "
                         << inputType.getRank() << " and "
                         << outputType.getRank();
  }
  const auto inputShape = getShapeRef(inputType);
  const auto outputShape = getShapeRef(outputType);

  // Enforce shape=1 + dynamic-stride convention: every differing dim must
  // go from literal 1 to ?/>1, and at least one dim must change.
  bool sawBroadcastDim = false;
  for (size_t i = 0; i < tcg::rank(inputShape); ++i) {
    const auto &inDim = inputShape[i];
    const auto &outDim = outputShape[i];

    if ((!tcg::is_static(inDim) && !tcg::is_static(outDim)) ||
        inDim == outDim) {
      continue;
    }

    if (!tcg::is_static(inDim) || inDim != 1) {
      return emitOpError()
             << "broadcast dim " << i
             << " must change from a literal 1 in the input to ?/>1 in the "
                "output (shape=1 + dynamic-stride convention); the legacy "
                "shape=? + stride=0 convention is no longer accepted";
    }
    if (tcg::is_static(outDim) && outDim == 1) {
      return emitOpError() << "broadcast dim " << i
                           << " has output dim 1; expected ? or >1";
    }
    sawBroadcastDim = true;
  }

  if (!sawBroadcastDim) {
    return emitOpError()
           << "no-op broadcast is not allowed; every broadcast op must change "
              "at least one dim from 1 to ?/>1";
  }

  return success();
}

LogicalResult TransposeOp::verify() {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();

  // Get permutation array
  auto permutationArray = getPermutation();

  // Get input and output shapes
  auto inputShape = getShapeRef(inputType);
  auto outputShape = getShapeRef(outputType);

  int64_t inputRank = tcg::rank(inputShape);

  // Verify permutation array size matches tensor rank
  if (static_cast<int64_t>(permutationArray.size()) != inputRank) {
    return emitOpError()
           << "requires permutation array size to match tensor rank, "
           << "but got permutation size " << permutationArray.size()
           << " and tensor rank " << inputRank;
  }

  // Verify permutation is a valid permutation of [0, 1, ..., rank-1]
  llvm::SmallVector<bool> used(inputRank, false);
  for (int64_t i = 0; i < static_cast<int64_t>(permutationArray.size()); ++i) {
    int64_t dim = permutationArray[i];
    if (dim < 0 || dim >= inputRank) {
      return emitOpError() << "permutation contains invalid dimension index "
                           << dim << ", expected value in range [0, "
                           << (inputRank - 1) << "]";
    }
    if (used[dim]) {
      return emitOpError() << "permutation contains duplicate dimension index "
                           << dim;
    }
    used[dim] = true;
  }

  // Every result extent is inherited from exactly one input extent. Requiring
  // identical static/dynamic information keeps the runtime relation explicit.
  for (int64_t i = 0; i < inputRank; ++i) {
    int64_t inputDim = permutationArray[i];
    int64_t inputSize = inputType.getShape()[inputDim];
    int64_t outputSize = outputType.getShape()[i];

    if (inputSize != outputSize) {
      return emitOpError() << "output dimension " << i << " (size "
                           << outputSize << ") "
                           << "does not match permuted input dimension "
                           << inputDim << " (size " << inputSize << ")";
    }
  }

  return success();
}

LayoutSourceAttrInterface TransposeOp::inferResultLayout(
    ArrayRef<LayoutSourceAttrInterface> inputLayouts) {
  assert(inputLayouts.size() == 1 && "expects one input layout");
  return inputLayouts[0].transpose(getPermutation());
}

LogicalResult SliceOp::verify() {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();

  // Rank equality and element type equality are enforced by ODS traits.
  int64_t inputRank = inputType.getRank();

  // Extract attributes
  auto starts = getStarts();
  auto limits = getLimits();
  auto strides = getStrides();

  // Check attribute vector sizes
  if ((int64_t)starts.size() != inputRank ||
      (int64_t)limits.size() != inputRank ||
      (int64_t)strides.size() != inputRank) {
    return emitOpError() << "expects starts, limits, and strides to have "
                            "length equal to tensor rank ("
                         << inputRank << ")";
  }

  // Validate non-negativity/positivity
  if (llvm::any_of(starts, [](int64_t start) { return start < 0; })) {
    return emitOpError() << "starts must be non-negative";
  }
  if (llvm::any_of(limits, [](int64_t limit) { return limit < 0; })) {
    return emitOpError() << "limits must be non-negative";
  }
  if (llvm::any_of(strides, [](int64_t stride) { return stride <= 0; })) {
    return emitOpError() << "strides must be positive";
  }

  auto inShape = inputType.getShape();
  auto outShape = outputType.getShape();
  for (int64_t i = 0; i < inputRank; ++i) {
    // Statically check bounds whenever the input extent is known. Bounds on a
    // dynamic input extent are a runtime precondition.
    if (!ShapedType::isDynamic(inShape[i])) {
      if (starts[i] > inShape[i] || limits[i] > inShape[i]) {
        return emitOpError()
               << "slice out of bounds at dimension " << i
               << ": start=" << starts[i] << ", limit=" << limits[i]
               << ", stride=" << strides[i] << ", input_dim=" << inShape[i];
      }
    }

    int64_t expected = 0;
    if (limits[i] > starts[i]) {
      expected = 1 + (limits[i] - 1 - starts[i]) / strides[i];
    }

    if (outShape[i] != expected) {
      return emitOpError() << "output dimension " << i << " (size "
                           << outShape[i]
                           << ") does not match computed slice size "
                           << expected;
    }
  }

  return success();
}

LayoutSourceAttrInterface
SliceOp::inferResultLayout(ArrayRef<LayoutSourceAttrInterface> inputLayouts) {
  assert(inputLayouts.size() == 1 && "expects one input layout");
  return inputLayouts[0].slice(getStarts(), getLimits(), getStrides());
}

LogicalResult PowOp::verify() {
  auto baseTensor = getLhs().getType();
  auto exponentTensor = getRhs().getType();

  // Get element types
  Type baseElementType = baseTensor.getElementType();
  Type exponentElementType = exponentTensor.getElementType();

  // Case 1: Both operands have the same element type (both float or both
  // integer)
  if (baseElementType == exponentElementType) {
    if (!isa<FloatType>(baseElementType) &&
        !isa<IntegerType>(baseElementType)) {
      return emitOpError()
             << "requires both operands to have the same element type "
             << "which must be either floating-point or integer, but got "
             << baseElementType;
    }
  }
  // Case 2: Base is float and exponent is integer
  else if (!isa<FloatType>(baseElementType) ||
           !isa<IntegerType>(exponentElementType)) {
    return emitOpError()
           << "requires either both operands to have the same element type "
           << "or base operand to have floating-point element type and "
           << "exponent operand to have integer element type, but got "
           << "base: " << baseElementType
           << ", exponent: " << exponentElementType;
  }

  return success();
}

// Verifier for CmpOp operation
LogicalResult CmpOp::verify() {
  auto lhsType = getLhs().getType();

  // Check that the operands have the same element type
  Type lhsElementType = lhsType.getElementType();

  // Determine if we're dealing with integer or floating point tensors
  bool isIntegerType = mlir::isa<mlir::IntegerType>(lhsElementType);
  bool isFloatType = mlir::isa<mlir::FloatType>(lhsElementType);

  // Get the comparator attribute
  int32_t enumValue = static_cast<int32_t>(getComparator());

  // Check if comparator is valid for the element type
  if (isIntegerType && enumValue > 5) {
    // Integer comparators are 0-5
    return emitOpError() << "integer tensor operands require an integer "
                            "comparator (eq, neq, gt, ge, lt, le)";
  } else if (isFloatType && enumValue < 6) {
    // Float comparators are 6-17
    return emitOpError() << "float tensor operands require a float "
                            "comparator (oeq, one, ogt, etc.)";
  } else if (!isIntegerType && !isFloatType) {
    return emitOpError()
           << "comparison only supports integer or float element types";
  }

  return success();
}

LogicalResult ConcatenateOp::verify() {
  // AtLeastNOperands cannot be combined with the VariadicOperands trait
  // generated by ODS, because both provide the same multi-operand API.
  if (getInputs().empty()) {
    return emitOpError("requires at least one input");
  }

  auto firstInputType =
      cast<nv_tensor_ir::TensorType>(getInputs()[0].getType());
  auto outputType = getOutput().getType();

  // Get dimension info
  int64_t concatDim = getDimension();
  int64_t rank = firstInputType.getRank();

  if (outputType.getRank() != rank) {
    return emitOpError() << "output has rank " << outputType.getRank()
                         << " but expected " << rank;
  }

  // Verify concatenation dimension is valid
  if (concatDim < 0 || concatDim >= rank) {
    return emitOpError() << "concatenation dimension " << concatDim
                         << " is out of bounds for tensor rank " << rank;
  }

  // Verify all inputs have compatible shapes and accumulate a static
  // concatenation extent when possible.
  int64_t totalConcatSize = 0;
  bool hasDynamicConcatSize = false;
  SmallVector<int64_t> mergedShape(firstInputType.getShape());

  for (auto [idx, input] : llvm::enumerate(getInputs())) {
    auto inputType = cast<nv_tensor_ir::TensorType>(input.getType());
    if (inputType.getRank() != rank) {
      return emitOpError() << "input " << idx << " has rank "
                           << inputType.getRank() << " but expected " << rank;
    }

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t inputSize = inputType.getDimSize(dim);
      if (dim == concatDim) {
        if (ShapedType::isDynamic(inputSize)) {
          hasDynamicConcatSize = true;
        } else {
          if (inputSize >
              std::numeric_limits<int64_t>::max() - totalConcatSize) {
            return emitOpError("concatenation dimension size overflows i64");
          }
          totalConcatSize += inputSize;
        }
      } else {
        FailureOr<int64_t> merged =
            mergeEqualDimensions(mergedShape[dim], inputSize);
        if (failed(merged)) {
          return emitOpError()
                 << "expects non-concatenation dimensions of all inputs to "
                    "match, but input 0 and input "
                 << idx << " have types " << firstInputType << " and "
                 << inputType;
        }
        mergedShape[dim] = *merged;
      }
    }
  }

  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t outputSize = outputType.getDimSize(dim);
    if (dim == concatDim) {
      int64_t expected =
          hasDynamicConcatSize ? ShapedType::kDynamic : totalConcatSize;
      if (outputSize != expected) {
        return emitOpError() << "expects output concatenation dimension to "
                                "equal the sum of input extents, but got "
                             << outputType;
      }
    } else if (outputSize != mergedShape[dim]) {
      return emitOpError()
             << "expects output non-concatenation dimensions to equal the "
                "merged input shape, but got "
             << outputType;
    }
  }

  return success();
}

LayoutSourceAttrInterface ConcatenateOp::inferResultLayout(
    ArrayRef<LayoutSourceAttrInterface> inputLayouts) {
  assert(inputLayouts.size() == getInputs().size() &&
         "expects one input layout for each input");
  return ConcatSourceAttr::get(getContext(), getDimension(), inputLayouts,
                               /*argumentIndex=*/{});
}

LogicalResult MatmulOp::verify() {
  if (failed(verifyMatmulShapes(getOperation(), getA().getType(),
                                getB().getType(), getC().getType()))) {
    return failure();
  }
  if (getAcc() && getAcc().getType() != getC().getType()) {
    return emitOpError()
           << "expects accumulator type to match result type, but "
              "got "
           << getAcc().getType() << " and " << getC().getType();
  }
  return success();
}

void MatmulOp::build(OpBuilder &builder, OperationState &result, Type c,
                     Value a, Value b) {
  build(builder, result, c, a, b, Value{});
}

LayoutSourceAttrInterface
MatmulOp::inferResultLayout(ArrayRef<LayoutSourceAttrInterface> inputLayouts) {
  // Layout propagation does not support custom matmul accumulators.
  if (getAcc()) {
    return nullptr;
  }
  assert(inputLayouts.size() == 2 && "MatmulOp must have two inputs");

  // Build a CuTe view layout for the inferred matmul result.
  // The view has rank+1 dimensions: batch dims (...) + M + N + K (reduction).
  // For a result shape (..., M, N) and operands (..., M, K) × (..., K, N),
  // we construct a flattened layout where K is the innermost reduction
  // dimension.

  // Calculate B/M/N/K from the shapes.
  auto shape = getResult().getType().getShape();
  size_t rank = shape.size();
  int64_t sizeB = std::accumulate(shape.begin(), shape.end() - 2, 1,
                                  std::multiplies<int64_t>());
  // Matmul requires rank >= 2 for at least (M, N) dimensions.
  int64_t sizeM = shape[rank - 2];
  int64_t sizeN = shape[rank - 1];
  // Get K from first operand's last dimension: (..., M, K).
  int64_t sizeK = getA().getType().getShape().back();

  // Build the logical view shape, with K as the final reduction dimension.
  tcg::Shape viewShape;
  for (size_t i = 0; i < rank - 2; ++i) {
    viewShape.append(shape[i]);
  }
  viewShape.append(sizeM);
  viewShape.append(sizeN);
  viewShape.append(sizeK);
  tcg::Layout view(viewShape);

  // Build matmul source attribute.
  return MatmulSourceAttr::get(getContext(), view.toString(), sizeB, sizeM,
                               sizeN, sizeK, inputLayouts[0], inputLayouts[1]);
}

//===----------------------------------------------------------------------===//
// IterationSpaceInfoInterface implementations for MatmulOp.
//===----------------------------------------------------------------------===//

unsigned MatmulOp::getNumIterSpaceDims() {
  // Number of dimensions in a matmul is result rank (..., m, n) + 1 (k).
  return getResult().getType().getRank() + 1;
}

AffineMap MatmulOp::computeCanonicalInputIterSpaceMap(
    unsigned inputIndex, std::optional<unsigned> iterSpaceRank) {
  assert(inputIndex < getNumOperands() && "invalid MatmulOp input index");

  // Use provided rank or fall back to operation's iteration space rank.
  unsigned numIterSpaceDims = iterSpaceRank.value_or(getNumIterSpaceDims());

  MLIRContext *ctx = getContext();
  if (inputIndex == 2) {
    return computeCanonicalOutputIterSpaceMap(numIterSpaceDims);
  }
  bool isA = inputIndex == 0;

  // A input - canonical input map result is (..., m, k). Return identity map
  // without the n dimension.
  if (isA) {
    unsigned nDimPos = numIterSpaceDims - 2;
    return AffineMap::getFilteredIdentityMap(
        getContext(), numIterSpaceDims,
        [&](AffineDimExpr dim) { return dim.getPosition() != nDimPos; });
  }

  // B input - canonical input map result is (..., k, n). Drop the m dimension
  // from identity map and swap the n and k dimensions.
  unsigned mDimPos = numIterSpaceDims - 3;
  auto bFilteredIdentityMap = AffineMap::getFilteredIdentityMap(
      getContext(), numIterSpaceDims,
      [&](AffineDimExpr dim) { return dim.getPosition() != mDimPos; });

  auto bResExprs = llvm::to_vector(bFilteredIdentityMap.getResults());
  int numResults = bResExprs.size();
  int nDimPos = numResults - 2;
  int kDimPos = numResults - 1;
  std::swap(bResExprs[nDimPos], bResExprs[kDimPos]);
  return AffineMap::get(numIterSpaceDims, 0, bResExprs, ctx);
}

AffineMap MatmulOp::computeCanonicalOutputIterSpaceMap(
    std::optional<unsigned> iterSpaceRank) {
  // Use provided rank or fall back to operation's iteration space rank.
  unsigned numIterSpaceDims = iterSpaceRank.value_or(getNumIterSpaceDims());

  // Canonical output map result is always (..., m, n), regardless of the output
  // rank. It's an identity map except that we drop the reduction dimension.
  return AffineMap::getFilteredIdentityMap(
      getContext(), numIterSpaceDims, [&](AffineDimExpr dim) {
        return dim.getPosition() != numIterSpaceDims - 1;
      });
}

AffineMap MatmulOp::inferInputFromOutputIterSpaceMap(unsigned inputIndex) {
  assert(inputIndex < getNumOperands() && "invalid MatmulOp input index");

  MLIRContext *ctx = getContext();
  bool isA = inputIndex == 0;

  // Retrieve the output iteration space map.
  auto iterSpaceOp = cast<IterationSpaceInfoInterface>(this->getOperation());
  auto outputMap = iterSpaceOp.getOutputIterSpaceMap();
  if (!outputMap) {
    return AffineMap();
  }
  if (inputIndex == 2) {
    return outputMap;
  }

  auto outputResExprs = outputMap.getResults();
  int numResults = outputResExprs.size();
  auto nDimExpr = outputResExprs[numResults - 1];
  auto kDimExpr = getAffineDimExpr(outputMap.getNumDims() - 1, ctx);

  // Initialize the input results with the output results and the replace the
  // corresponding dimensions based on the input index.
  auto inputExprs = llvm::to_vector(outputResExprs);

  if (isA) {
    // Map result for A is (..., m, k).
    inputExprs[numResults - 1] = kDimExpr;
  } else {
    // Map result for B is (..., k, n).
    inputExprs[numResults - 2] = kDimExpr;
    inputExprs[numResults - 1] = nDimExpr;
  }

  return AffineMap::get(outputMap.getNumDims(), 0, inputExprs, ctx);
}

AffineMap MatmulOp::inferOutputFromInputIterSpaceMaps() {
  AffineMap aMap = getInputIterSpaceMap(0);
  AffineMap bMap = getInputIterSpaceMap(1);
  if (!aMap || !bMap) {
    return AffineMap();
  }

  unsigned iterSpaceRank = aMap.getNumInputs();
  assert(iterSpaceRank == bMap.getNumInputs() &&
         "Iteration space rank doesn't match");
  MLIRContext *ctx = getContext();

  // Map result for output is (..., m, n). Initialize the output results with
  // input A results and then replace the k dimension with the n dimension from
  // input B.
  SmallVector<AffineExpr> outputExprs = llvm::to_vector(aMap.getResults());
  outputExprs.back() = bMap.getResults().back();
  return AffineMap::get(iterSpaceRank, 0, outputExprs, ctx);
}

// ===----------------------------------------------------------------------===//
// BroadcastOp
// ===----------------------------------------------------------------------===//

LayoutSourceAttrInterface BroadcastOp::inferResultLayout(
    ArrayRef<LayoutSourceAttrInterface> inputLayouts) {
  assert(inputLayouts.size() == 1 && "expects one input layout");
  auto outputTy = getOutput().getType();
  return inputLayouts[0].broadcast(outputTy.getShape());
}

// ===----------------------------------------------------------------------===//
// BroadcastOp - IterationSpaceInfoInterface implementation.
// ===----------------------------------------------------------------------===//

/// A broadcast creates an iteration space transition when it expands a unit
/// dimension whose domain was previously defined.
bool BroadcastOp::isIterationSpaceTransition(
    llvm::ArrayRef<DimState> inputDimStates) {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  // Rank-preserving broadcast so input states and output shapes must match.
  // For matmul, the input dim states size is the result rank + 1.
  assert(inputDimStates.size() >= outputShape.size() &&
         "input dim states size must match output shape size");

  for (size_t i = 0; i < inputShape.size(); ++i) {
    // Check if this dimension is being expanded from 1 to N.
    if (inputShape[i] == 1 && outputShape[i] > 1) {
      // Transition occurs if the dimension was previously defined (Def).
      if (inputDimStates[i] == DimState::Def) {
        return true;
      }
    }
  }

  return false;
}

// ===----------------------------------------------------------------------===//
// ConstantOp - IterationSpaceInfoInterface implementation
// ===----------------------------------------------------------------------===//

unsigned ConstantOp::getNumIterSpaceDims() {
  // ConstantOp has no operands, so we get the rank from the result type.
  // Scalar constants (e.g., f32) have 0 iteration space dimensions.
  if (auto resultType = dyn_cast<TensorType>(getResult().getType())) {
    return resultType.getRank();
  }
  return 0;
}

AffineMap ConstantOp::computeCanonicalOutputIterSpaceMap(
    std::optional<unsigned> iterSpaceRank) {
  unsigned numIterSpaceDims = iterSpaceRank.value_or(getNumIterSpaceDims());
  // For scalar constants, return an empty 0-result map since they have no
  // iteration space dimensions.
  if (!isa<TensorType>(getResult().getType())) {
    return AffineMap::get(numIterSpaceDims, /*symbolCount=*/0, /*results=*/{},
                          getContext());
  }

  // For tensor constants, use identity map if ranks match.
  auto resultType = cast<TensorType>(getResult().getType());
  int resultRank = resultType.getRank();
  if (numIterSpaceDims != static_cast<unsigned>(resultRank)) {
    return AffineMap();
  }

  return AffineMap::getMultiDimIdentityMap(numIterSpaceDims, getContext());
}

// ===----------------------------------------------------------------------===//
// SplatOp - IterationSpaceInfoInterface implementation
// ===----------------------------------------------------------------------===//

unsigned SplatOp::getNumIterSpaceDims() {
  // SplatOp takes a scalar input (not a tensor), so we get the rank from the
  // output type.
  auto outputType = getOutput().getType();
  return outputType.getRank();
}

AffineMap SplatOp::computeCanonicalInputIterSpaceMap(
    unsigned inputIdx, std::optional<unsigned> iterSpaceRank) {
  unsigned rank = iterSpaceRank.value_or(getNumIterSpaceDims());
  // SplatOp has a scalar input, not a tensor. Return zero-result map
  // (consistent with getInputIterSpaceMap). A zero-result map (d0, d1) -> ()
  // means "defined but produces nothing", distinct from empty map () -> ()
  // which means "undefined/unknown".
  return AffineMap::get(rank, /*symbolCount=*/0, /*results=*/{}, getContext());
}

AffineMap SplatOp::inferInputFromOutputIterSpaceMap(unsigned inputIndex) {
  // SplatOp has a scalar input, not a tensor. Return zero-result map
  // (consistent with getInputIterSpaceMap and
  // computeCanonicalInputIterSpaceMap).
  unsigned iterSpaceRank = getNumIterSpaceDims();
  return AffineMap::get(iterSpaceRank, /*symbolCount=*/0, /*results=*/{},
                        getContext());
}

AffineMap SplatOp::inferOutputFromInputIterSpaceMaps() {
  // SplatOp's output is independent of scalar input.
  // Return identity map for output rank (same as canonical).
  unsigned rank = getNumIterSpaceDims();
  return AffineMap::getMultiDimIdentityMap(rank, getContext());
}

AffineMap SplatOp::getInputIterSpaceMap(unsigned inputIdx) {
  // SplatOp's input is a scalar, not a tensor. Return a zero-result AffineMap
  // to signal "this input doesn't participate in iteration space propagation"
  // and prevent infinite loops in discover-iteration-space-info.
  unsigned iterSpaceRank = getNumIterSpaceDims();
  return AffineMap::get(iterSpaceRank, /*symbolCount=*/0, /*results=*/{},
                        getContext());
}

// ===----------------------------------------------------------------------===//
// IotaOp
// ===----------------------------------------------------------------------===//

LogicalResult IotaOp::verify() {
  auto outputType = getResult().getType();
  auto dimension = static_cast<int64_t>(getDimension());
  if (dimension < 0 || dimension >= outputType.getRank()) {
    return emitOpError() << "dimension " << dimension
                         << " is out of bounds for tensor rank "
                         << outputType.getRank();
  }

  // Lowering emits cuda_tile.iota in an integer type (i32 for float results).
  // Static extent along the iota dimension must fit in that type.
  Type elemType = outputType.getElementType();
  unsigned iotaBits =
      elemType.isInteger() ? elemType.getIntOrFloatBitWidth() : 32U;
  uint64_t maxIotaValue;
  if (iotaBits >= 64) {
    maxIotaValue = std::numeric_limits<uint64_t>::max();
  } else if (elemType.isSignedInteger()) {
    maxIotaValue = (uint64_t{1} << (iotaBits - 1)) - 1;
  } else {
    maxIotaValue = (uint64_t{1} << iotaBits) - 1;
  }

  int64_t extent = outputType.getShape()[dimension];
  if (!ShapedType::isDynamic(extent) && extent > 0 &&
      static_cast<uint64_t>(extent - 1) > maxIotaValue) {
    return emitOpError() << "extent " << extent << " along iota dimension "
                         << dimension
                         << " exceeds the maximum value representable in the "
                         << "iota element type";
  }
  return success();
}
