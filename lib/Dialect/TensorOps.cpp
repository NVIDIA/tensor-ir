// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#define DEBUG_TYPE "tensor-ops"

using namespace mlir;
using namespace mlir::nv_tensor_ir;
namespace tcg = mlir::nv_tensor_ir::tcutegen;

FailureOr<SmallVector<int64_t>>
mlir::nv_tensor_ir::getScanSliceInputAxes(ArrayRef<int64_t> carryShape,
                                          ArrayRef<int64_t> sliceShape) {
  SmallVector<int64_t> axes;
  size_t carryDim = 0;
  for (int64_t extent : sliceShape) {
    if (extent == 1) {
      axes.push_back(-1);
      continue;
    }
    while (carryDim < carryShape.size() && carryShape[carryDim] == 1) {
      ++carryDim;
    }
    if (carryDim == carryShape.size() || carryShape[carryDim] != extent) {
      return failure();
    }
    axes.push_back(carryDim++);
  }
  return axes;
}

bool mlir::nv_tensor_ir::isGraphInput(Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg) {
    return false;
  }
  Operation *parent = blockArg.getOwner()->getParentOp();
  bool isGraph = isa<GraphOp>(parent);
  return isGraph;
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
                                                     RankedTensorType aType,
                                                     RankedTensorType bType,
                                                     RankedTensorType cType) {
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
    FailureOr<int64_t> expected = mergeEqualDimensions(aDim, bDim);
    if (failed(expected) || cDim != *expected) {
      return op->emitOpError()
             << "expects C batch dimensions to equal the merged shape of A "
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

static ParseResult parseContractionIndexingMaps(OpAsmParser &parser,
                                                ArrayAttr &indexingMaps) {
  SmallVector<std::pair<StringRef, AffineExpr>> dimensions;
  auto parseDimension = [&]() -> ParseResult {
    auto loc = parser.getCurrentLocation();
    StringRef name;
    if (parser.parseKeyword(&name)) {
      return failure();
    }
    if (llvm::any_of(dimensions,
                     [&](const auto &dim) { return dim.first == name; })) {
      return parser.emitError(loc, "duplicate indexing-map dimension ") << name;
    }
    dimensions.emplace_back(
        name, getAffineDimExpr(dimensions.size(), parser.getContext()));
    return success();
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseDimension) ||
      parser.parseArrow()) {
    return failure();
  }

  SmallVector<Attribute> maps;
  for (unsigned i = 0; i < 3; ++i) {
    if (i && parser.parseComma()) {
      return failure();
    }
    SmallVector<AffineExpr> results;
    auto parseResult = [&]() -> ParseResult {
      AffineExpr expr;
      if (parser.parseAffineExpr(dimensions, expr)) {
        return failure();
      }
      results.push_back(expr);
      return success();
    };
    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                       parseResult)) {
      return failure();
    }
    auto map =
        AffineMap::get(dimensions.size(), 0, results, parser.getContext());
    maps.push_back(AffineMapAttr::get(map));
  }
  indexingMaps = parser.getBuilder().getArrayAttr(maps);
  return success();
}

static void printContractionIndexingMaps(OpAsmPrinter &printer, Operation *,
                                         ArrayAttr indexingMaps) {
  auto firstMap = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  printer << '(';
  llvm::interleaveComma(llvm::seq<unsigned>(0, firstMap.getNumDims()), printer,
                        [&](unsigned i) { printer << 'd' << i; });
  printer << ") -> ";
  llvm::interleaveComma(indexingMaps, printer, [&](Attribute attr) {
    printer << '(';
    llvm::interleaveComma(cast<AffineMapAttr>(attr).getValue().getResults(),
                          printer);
    printer << ')';
  });
}

static ParseResult parseReductionBody(OpAsmParser &parser, Region &body);
static void printReductionBody(OpAsmPrinter &printer, Operation *,
                               Region &body);

#define GET_OP_CLASSES
#include "tensor_ir/Dialect/TensorOps.cpp.inc"

// enum attribute definitions
#include "tensor_ir/Dialect/TensorEnums.cpp.inc"

namespace {

template <typename OpTy>
LogicalResult verifyViewOperands(OpTy op) {
  if (llvm::any_of(op.getStaticStrides(), [](int64_t stride) {
        return !ShapedType::isDynamic(stride) && stride < 0;
      })) {
    return op->emitOpError() << "expected nonnegative or dynamic strides";
  }
  size_t rank = op.getStaticSizes().size();
  if (op.getStaticIndices().size() != rank) {
    return op->emitOpError()
           << "expected " << rank << " tile coordinate(s), but got "
           << op.getStaticIndices().size();
  }
  if (failed(verifyListOfOperandsOrIntegers(
          op, "indices", rank, op.getStaticIndices(), op.getIndices()))) {
    return failure();
  }
  auto verifyDynamicValues = [&](ValueRange values,
                                 StringRef field) -> LogicalResult {
    for (auto [position, value] : llvm::enumerate(values)) {
      std::optional<int64_t> constant = getConstantIntValue(value);
      if (constant && *constant < 0) {
        return op->emitOpError()
               << "expected dynamic " << field << " #" << position << " to be "
               << "nonnegative, but got " << *constant;
      }
    }
    return success();
  };
  if (failed(verifyDynamicValues(op.getSizes(), "size")) ||
      failed(verifyDynamicValues(op.getStrides(), "stride")) ||
      failed(verifyDynamicValues(op.getOffsets(), "offset"))) {
    return failure();
  }
  for (auto [position, index] : llvm::enumerate(op.getMixedIndices())) {
    std::optional<int64_t> constant = getConstantIntValue(index);
    if (constant && *constant < 0) {
      return op->emitOpError() << "expected nonnegative tile coordinate #"
                               << position << ", but got " << *constant;
    }
  }
  return success();
}

} // namespace

SmallVector<OpFoldResult> LoadOp::getMixedIndices() {
  return getMixedValues(getStaticIndices(), getIndices(), getContext());
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Type tile,
                   Value base, OpFoldResult offset,
                   ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
                   ArrayRef<OpFoldResult> indices, IntegerAttr alignment,
                   TilePaddingAttr padding, bool allowTma) {
  auto [staticOffsets, dynamicOffsets] = decomposeMixedValues({offset});
  auto [staticSizes, dynamicSizes] = decomposeMixedValues(sizes);
  auto [staticStrides, dynamicStrides] = decomposeMixedValues(strides);
  auto [staticIndices, dynamicIndices] = decomposeMixedValues(indices);
  build(builder, state, tile, base, dynamicOffsets, dynamicSizes,
        dynamicStrides, dynamicIndices,
        builder.getDenseI64ArrayAttr(staticOffsets),
        builder.getDenseI64ArrayAttr(staticSizes),
        builder.getDenseI64ArrayAttr(staticStrides),
        builder.getDenseI64ArrayAttr(staticIndices), alignment, padding,
        builder.getBoolAttr(allowTma));
}

SmallVector<OpFoldResult> StoreOp::getMixedIndices() {
  return getMixedValues(getStaticIndices(), getIndices(), getContext());
}

void StoreOp::build(OpBuilder &builder, OperationState &state, Value tile,
                    Value base, OpFoldResult offset,
                    ArrayRef<OpFoldResult> sizes,
                    ArrayRef<OpFoldResult> strides,
                    ArrayRef<OpFoldResult> indices, IntegerAttr alignment,
                    bool allowTma) {
  auto [staticOffsets, dynamicOffsets] = decomposeMixedValues({offset});
  auto [staticSizes, dynamicSizes] = decomposeMixedValues(sizes);
  auto [staticStrides, dynamicStrides] = decomposeMixedValues(strides);
  auto [staticIndices, dynamicIndices] = decomposeMixedValues(indices);
  build(builder, state, tile, base, dynamicOffsets, dynamicSizes,
        dynamicStrides, dynamicIndices,
        builder.getDenseI64ArrayAttr(staticOffsets),
        builder.getDenseI64ArrayAttr(staticSizes),
        builder.getDenseI64ArrayAttr(staticStrides),
        builder.getDenseI64ArrayAttr(staticIndices), alignment,
        builder.getBoolAttr(allowTma));
}

LogicalResult LoadOp::verify() {
  auto tileType = cast<RankedTensorType>(getTile().getType());
  std::optional<TilePadding> padding = getPadding();
  if (padding && *padding != TilePadding::Zero &&
      !isa<FloatType>(tileType.getElementType())) {
    return emitOpError() << "padding " << stringifyTilePadding(*padding)
                         << " requires a floating-point element type, got "
                         << tileType.getElementType();
  }
  return verifyViewOperands(*this);
}

LogicalResult StoreOp::verify() { return verifyViewOperands(*this); }

void GraphOp::getAsmBlockArgumentNames(Region &region,
                                       OpAsmSetValueNameFn setNameFn) {
  // TODO: use an attribute on the graph to name the inputs.
  for (Value v : region.front().getArguments()) {
    setNameFn(v, "in");
  }
}

static LogicalResult verifyDenseGraphType(Operation *op, Type type,
                                          StringRef kind, unsigned index) {
  if (isa<RankedTensorType>(type) && !isTensorType(type)) {
    return op->emitOpError()
           << "expects " << kind << " #" << index
           << " to be a TensorIR ranked tensor, but got " << type;
  }
  return success();
}

LogicalResult GraphOp::verify() {
  for (auto [idx, type] : llvm::enumerate(getArgumentTypes())) {
    if (failed(verifyDenseGraphType(*this, type, "argument", idx))) {
      return failure();
    }
  }
  for (auto [idx, type] : llvm::enumerate(getResultTypes())) {
    if (failed(verifyDenseGraphType(*this, type, "result", idx))) {
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

LogicalResult
ScanSliceOp::inferReturnTypes(MLIRContext *context,
                              std::optional<Location> location, Adaptor adaptor,
                              SmallVectorImpl<Type> &inferredReturnTypes) {
  auto fail = [&](const Twine &message) {
    return emitOptionalError(location, message);
  };
  auto inputs = adaptor.getInputs();
  auto dimensions = adaptor.getScanDimensions();
  auto outputDimensions = adaptor.getOutputDimensions();
  if (inputs.size() < 2 || dimensions.size() + 1 != inputs.size()) {
    return fail("expects one carry and one scan dimension per scanned input");
  }
  auto carryType = dyn_cast<RankedTensorType>(inputs.front().getType());
  if (!carryType || !carryType.hasStaticShape()) {
    return fail("requires a static ranked carry tensor");
  }
  Region &region = adaptor.getBody();
  if (!region.hasOneBlock() || region.front().empty()) {
    return fail("expects a single body block with a yield terminator");
  }
  Block &body = region.front();
  if (body.getNumArguments() != inputs.size() ||
      body.getArgument(0).getType() != carryType) {
    return fail("body arguments must match the carry and sliced inputs");
  }
  int64_t extent = -1;
  for (auto [index, input] : llvm::enumerate(inputs.drop_front())) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    int64_t dimension = dimensions[index];
    if (!type || !type.hasStaticShape() || dimension < 0 ||
        dimension >= type.getRank()) {
      return fail(
          "scan dimension is out of range or input shape is not static");
    }
    int64_t inputExtent = type.getDimSize(dimension);
    if (inputExtent <= 0 || (extent != -1 && inputExtent != extent)) {
      return fail("requires equal positive scan extents for all inputs");
    }
    extent = inputExtent;
    SmallVector<int64_t> shape(type.getShape());
    shape.erase(shape.begin() + dimension);
    auto sliceType = RankedTensorType::get(shape, type.getElementType());
    if (failed(getScanSliceInputAxes(carryType.getShape(), shape))) {
      return fail(
          "scan_slice input non-unit dimensions must match a prefix of the "
          "non-unit carry dimensions");
    }
    if (body.getArgument(index + 1).getType() != sliceType) {
      return fail("body scanned argument must have the sliced input type");
    }
  }
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != outputDimensions.size() + 1) {
    return fail(
        "body must yield the next carry and one value per scan_slice output");
  }
  if (yield.getOperand(0).getType() != carryType) {
    return fail("body must yield a carry matching the input carry");
  }
  inferredReturnTypes.push_back(carryType);
  for (auto [value, dimension] :
       llvm::zip_equal(yield.getOperands().drop_front(), outputDimensions)) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getRank() != carryType.getRank()) {
      return fail("scan_slice output values must have the carry rank");
    }
    if (dimension < 0 || dimension > type.getRank()) {
      return fail("scan_slice output dimension is out of range");
    }
    for (auto [size, carrySize] :
         llvm::zip_equal(type.getShape(), carryType.getShape())) {
      if (size != 1 && size != carrySize) {
        return fail("scan_slice output dimensions must match carry dimensions "
                    "or be one");
      }
    }
    SmallVector<int64_t> shape(type.getShape());
    shape.insert(shape.begin() + dimension, extent);
    inferredReturnTypes.push_back(
        RankedTensorType::get(shape, type.getElementType()));
  }
  return success();
}

LogicalResult ScanSliceOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  auto graph = dyn_cast<GraphOp>(parent);
  // Bufferization moves scan_slice into a marked function before FormGrid lowers it.
  bool bufferized =
      parent->hasAttr(TensorIRDialect::getBufferizedProgramAttrName());
  if (!graph && !bufferized) {
    return emitOpError("must be directly inside a graph or a bufferized "
                       "program; nested scans are not supported");
  }
  for (Value input : getInputs().drop_front()) {
    auto argument = dyn_cast<BlockArgument>(input);
    auto load = input.getDefiningOp<bufferization::ToTensorOp>();
    bool boundaryLoad =
        bufferized && load && load->getBlock() == getOperation()->getBlock();
    if ((!argument || !graph || argument.getOwner() != graph.getBody()) &&
        !boundaryLoad) {
      return emitOpError("scanned inputs must be graph arguments or bufferized "
                         "boundary loads");
    }
  }
  for (Value result : getResults().drop_front()) {
    if (!result.hasOneUse() ||
        !(isa<ResultsOp>(*result.user_begin()) ||
          (bufferized &&
           isa<bufferization::MaterializeInDestinationOp>(
               *result.user_begin()) &&
           cast<bufferization::MaterializeInDestinationOp>(*result.user_begin())
                   .getSource() == result))) {
      return emitOpError("requires every side result to feed graph results or "
                         "a bufferized boundary store directly");
    }
  }
  return success();
}

BlockArgument ScanSliceOp::getBlockArgumentForOperand(OpOperand &operand) {
  assert(operand.getOwner() == getOperation() &&
         "expected a scan_slice operand");
  return getBody().front().getArgument(operand.getOperandNumber());
}

BlockArgument ScanSliceOp::getBlockArgumentForResult(OpResult result) {
  assert(result.getOwner() == getOperation() && "expected a scan_slice result");
  return result.getResultNumber() == 0 ? getBody().front().getArgument(0)
                                       : BlockArgument{};
}

OpResult ScanSliceOp::getResultForOperand(OpOperand &operand) {
  assert(operand.getOwner() == getOperation() &&
         "expected a scan_slice operand");
  return operand.getOperandNumber() == 0 ? cast<OpResult>(getResult(0))
                                         : OpResult{};
}

YieldOp ScanSliceOp::getBodyTerminator() {
  return cast<YieldOp>(getBody().front().getTerminator());
}

int64_t ScanSliceOp::getScanExtent() {
  if (getInputs().size() < 2 || getScanDimensions().empty()) {
    return ShapedType::kDynamic;
  }
  auto type = dyn_cast<RankedTensorType>(getInputs()[1].getType());
  int64_t axis = getScanDimensions()[0];
  return type && axis >= 0 && axis < type.getRank() ? type.getDimSize(axis)
                                                    : ShapedType::kDynamic;
}

OperandRange ScanSliceOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  return getInputs().take_front(1);
}

ValueRange ScanSliceOp::getSuccessorInputs(RegionSuccessor successor) {
  if (successor.getSuccessor() == nullptr) {
    return getResults().take_front(1);
  }
  // The remaining arguments are produced slices, not forwarded input tensors.
  return getBody().front().getArguments().take_front(
      std::min<unsigned>(1, getBody().front().getNumArguments()));
}

// LLVM versions differ in how a successor returning to the parent is created.
// Keep both expressions dependent so only the supported API is instantiated.
template <typename Successor = RegionSuccessor>
static Successor getParentSuccessor(Operation *parent) {
  if constexpr (std::is_constructible_v<Successor, Operation *>) {
    return Successor(parent);
  } else {
    return Successor::parent();
  }
}

void ScanSliceOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody());
    return;
  }
  // A singleton scan_slice cannot take the backedge. Larger scans may either repeat
  // or exit depending on the implicit iteration index.
  if (getScanExtent() > 1) {
    regions.emplace_back(&getBody());
  }
  regions.emplace_back(getParentSuccessor(getOperation()));
}

void ScanSliceOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  int64_t extent = getScanExtent();
  if (extent < 0 ||
      static_cast<uint64_t>(extent) > std::numeric_limits<unsigned>::max()) {
    bounds.push_back(InvocationBounds::getUnknown());
    return;
  }
  bounds.emplace_back(extent, extent);
}

MutableOperandRange
YieldOp::getMutableSuccessorOperands(RegionSuccessor successor) {
  if (isa<ScanSliceOp>((*this)->getParentOp())) {
    return MutableOperandRange(getOperation(), 0,
                               std::min<unsigned>(1, getNumOperands()));
  }
  return MutableOperandRange(getOperation());
}

void YieldOp::getSuccessorRegions(ArrayRef<Attribute> operands,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  if (auto parent = dyn_cast<RegionBranchOpInterface>((*this)->getParentOp())) {
    parent.getSuccessorRegions(RegionBranchPoint(*this), regions);
    return;
  }
  regions.emplace_back(getParentSuccessor((*this)->getParentOp()));
}

void ScanSliceOp::build(
    OpBuilder &builder, OperationState &state, ValueRange inputs,
    ArrayRef<int64_t> scanDimensions, ArrayRef<int64_t> outputDimensions,
    function_ref<void(OpBuilder &, Location, Block::BlockArgListType)>
        bodyBuilder,
    bool reverse) {
  assert(inputs.size() >= 2 && scanDimensions.size() + 1 == inputs.size() &&
         "expected carry and one dimension per scanned input");
  state.addOperands(inputs);
  auto &properties = state.getOrAddProperties<Properties>();
  properties.scan_dimensions = builder.getDenseI64ArrayAttr(scanDimensions);
  properties.output_dimensions = builder.getDenseI64ArrayAttr(outputDimensions);
  properties.reverse = builder.getBoolAttr(reverse);
  Region *region = state.addRegion();
  auto *body = new Block;
  region->push_back(body);
  body->addArgument(inputs.front().getType(), state.location);
  for (auto [input, dimension] :
       llvm::zip_equal(inputs.drop_front(), scanDimensions)) {
    auto type = cast<RankedTensorType>(input.getType());
    assert(dimension >= 0 && dimension < type.getRank() && "invalid scan axis");
    SmallVector<int64_t> shape(type.getShape());
    shape.erase(shape.begin() + dimension);
    body->addArgument(RankedTensorType::get(shape, type.getElementType()),
                      state.location);
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(body);
  bodyBuilder(builder, state.location, body->getArguments());
  SmallVector<Type> types;
  if (failed(
          inferReturnTypes(builder.getContext(), state.location, inputs,
                           state.attributes.getDictionary(builder.getContext()),
                           state.getRawProperties(), state.regions, types))) {
    llvm::report_fatal_error("invalid scan_slice body constructed by callback");
  }
  state.addTypes(types);
}

LayoutSourceAttrInterface ScanSliceOp::inferResultLayout(
    ArrayRef<LayoutSourceAttrInterface> inputLayouts) {
  if (inputLayouts.size() != getInputs().size()) {
    return nullptr;
  }

  LayoutSourceAttrInterface carry = inputLayouts.front();
  if (!carry) {
    return nullptr;
  }
  auto carryType = cast<TensorType>(getInputs().front().getType());
  ArrayRef<int64_t> carryShape = carryType.getShape();
  if (carry.getShape() != carryShape) {
    carry = carry.reshape(carryShape);
    if (!carry) {
      return nullptr;
    }
  }

  SmallVector<Attribute> scanSliceInputs;
  scanSliceInputs.reserve(getInputs().size() - 1);
  for (auto [input, layout, scanDimension] :
       llvm::zip_equal(getInputs().drop_front(), inputLayouts.drop_front(),
                       getScanDimensions())) {
    auto tensorSource = dyn_cast<TensorSourceAttr>(layout);
    if (!tensorSource) {
      return nullptr;
    }
    auto inputType = cast<TensorType>(input.getType());
    tcg::Layout inputLayout = tensorSource.getCuteLayout();
    if (tcg::rank(inputLayout) != static_cast<size_t>(inputType.getRank()) ||
        !tcg::is_static(inputLayout)) {
      return nullptr;
    }

    SmallVector<int64_t> sliceShape(inputType.getShape());
    sliceShape.erase(sliceShape.begin() + scanDimension);
    SmallVector<int64_t> sliceStrides;
    sliceStrides.reserve(inputType.getRank() - 1);
    for (size_t dimension = 0; dimension < tcg::rank(inputLayout);
         ++dimension) {
      auto stride = tcg::get(inputLayout.stride(), dimension);
      if (static_cast<int64_t>(dimension) != scanDimension) {
        sliceStrides.push_back(stride.as_int());
      }
    }
    int64_t scanStride = tcg::get(inputLayout.stride(), scanDimension).as_int();

    tcg::Stride sliceStride;
    for (int64_t stride : sliceStrides) {
      sliceStride.append(stride);
    }
    tcg::Layout sliceLayout(tcg::Shape(sliceShape), sliceStride);
    auto sliceSource = TensorSourceAttr::get(
        getContext(), tensorSource.getTensorId(), tensorSource.getOffset(),
        sliceLayout.toString(), /*dynamicValueMapping=*/{});

    scanSliceInputs.push_back(ScanSliceInputAttr::get(
        getContext(), sliceSource, scanDimension,
        inputType.getDimSize(scanDimension), scanStride));
  }

  auto outputDimensions =
      DenseI64ArrayAttr::get(getContext(), getOutputDimensions());
  return ScanSliceSourceAttr::get(
      getContext(), carry, outputDimensions,
      ArrayAttr::get(getContext(), scanSliceInputs));
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

/// Checks that `dimensions` are unique and in range for a tensor of `rank`.
/// The returned bitvector is indexed by input dimension.
static FailureOr<llvm::SmallBitVector>
verifyReductionDimensions(Operation *op, int64_t rank,
                          ArrayRef<int32_t> dimensions) {
  llvm::SmallBitVector reducedDimensions(static_cast<unsigned>(rank));
  for (int32_t dimension : dimensions) {
    if (dimension < 0 || static_cast<int64_t>(dimension) >= rank) {
      return op->emitOpError() << "reduction dimension " << dimension
                               << " is out of bounds for tensor rank " << rank;
    }
    if (reducedDimensions.test(dimension)) {
      return op->emitOpError()
             << "reduction dimension " << dimension << " is duplicated";
    }
    reducedDimensions.set(dimension);
  }
  return reducedDimensions;
}

/// Verifies that `outputType` keeps every reduced dimension at static extent 1
/// or drops every reduced dimension. Non-reduced extents come from
/// `inputShape`.
static LogicalResult
verifyReductionResultShape(Operation *op, ArrayRef<int64_t> inputShape,
                           RankedTensorType outputType,
                           const llvm::SmallBitVector &reducedDimensions) {
  int64_t inputRank = inputShape.size();
  int64_t outputRank = outputType.getRank();
  int64_t droppedRank =
      inputRank - static_cast<int64_t>(reducedDimensions.count());
  bool dropsReducedDimensions = outputRank == droppedRank;
  if (outputRank != inputRank && !dropsReducedDimensions) {
    return op->emitOpError()
           << "expects output rank to be " << inputRank
           << " when retaining reduced dimensions or " << droppedRank
           << " when dropping them, but got " << outputRank;
  }

  int64_t outputDimension = 0;
  for (int64_t inputDimension = 0; inputDimension < inputRank;
       ++inputDimension) {
    bool isReduced =
        reducedDimensions.test(static_cast<unsigned>(inputDimension));
    if (dropsReducedDimensions && isReduced) {
      continue;
    }
    int64_t expected = isReduced ? 1 : inputShape[inputDimension];
    int64_t actual = outputType.getDimSize(outputDimension);
    if (actual != expected) {
      if (ShapedType::isDynamic(expected)) {
        return op->emitOpError()
               << "expects output dimension " << outputDimension
               << " to remain dynamic because the corresponding input "
                  "dimension is dynamic, but got "
               << outputType;
      }
      return op->emitOpError()
             << "expects output dimension " << outputDimension << " to be "
             << expected << ", but got " << actual;
    }
    ++outputDimension;
  }
  return success();
}

static bool reductionDropsDimensions(int64_t inputRank, int64_t outputRank,
                                     ArrayRef<int32_t> dimensions) {
  return outputRank == inputRank - static_cast<int64_t>(dimensions.size());
}

static llvm::SmallBitVector reductionDimMask(int64_t inputRank,
                                             ArrayRef<int32_t> dimensions) {
  llvm::SmallBitVector mask(static_cast<unsigned>(inputRank));
  for (int32_t dimension : dimensions) {
    assert(dimension >= 0 && dimension < inputRank &&
           "dimension is out of bounds");
    mask.set(static_cast<unsigned>(dimension));
  }
  return mask;
}

bool ReduceOp::dropsReducedDimensions() {
  return reductionDropsDimensions(getInput().getType().getRank(),
                                  getOutput().getType().getRank(),
                                  getDimensions());
}

llvm::SmallBitVector ReduceOp::getReductionDimMask() {
  return reductionDimMask(getInput().getType().getRank(), getDimensions());
}

AffineMap ReduceOp::computeCanonicalOutputIterSpaceMap(
    std::optional<unsigned> iterSpaceRank) {
  unsigned inputRank = getInput().getType().getRank();
  unsigned numIterSpaceDims = iterSpaceRank.value_or(inputRank);
  if (numIterSpaceDims != inputRank) {
    return {};
  }

  llvm::SmallBitVector reducedDimensions = getReductionDimMask();
  bool dropsDimensions = dropsReducedDimensions();
  return AffineMap::getFilteredIdentityMap(
      getContext(), numIterSpaceDims, [&](AffineDimExpr dimension) {
        return !dropsDimensions ||
               !reducedDimensions.test(dimension.getPosition());
      });
}

AffineMap ReduceOp::inferInputFromOutputIterSpaceMap(unsigned inputIndex) {
  assert(inputIndex == 0 && "invalid ReduceOp input index");
  AffineMap outputMap = getOutputIterSpaceMap();
  if (!outputMap || !dropsReducedDimensions()) {
    return outputMap;
  }

  unsigned inputRank = getInput().getType().getRank();
  llvm::SmallBitVector reducedDimensions = getReductionDimMask();
  if (outputMap.getNumDims() != inputRank ||
      outputMap.getNumResults() != inputRank - reducedDimensions.count()) {
    return {};
  }

  SmallVector<AffineExpr> inputResults;
  unsigned outputIndex = 0;
  for (unsigned inputDim = 0; inputDim < inputRank; ++inputDim) {
    if (reducedDimensions.test(inputDim)) {
      inputResults.push_back(getAffineDimExpr(inputDim, getContext()));
      continue;
    }
    inputResults.push_back(outputMap.getResult(outputIndex++));
  }
  return AffineMap::get(outputMap.getNumDims(), outputMap.getNumSymbols(),
                        inputResults, getContext());
}

AffineMap ReduceOp::inferOutputFromInputIterSpaceMaps() {
  AffineMap inputMap = getInputIterSpaceMap(0);
  if (!inputMap || !dropsReducedDimensions()) {
    return inputMap;
  }

  llvm::SmallBitVector reducedDimensions = getReductionDimMask();
  if (inputMap.getNumResults() != reducedDimensions.size()) {
    return {};
  }

  SmallVector<AffineExpr> outputResults;
  for (auto [index, result] : llvm::enumerate(inputMap.getResults())) {
    if (!reducedDimensions.test(index)) {
      outputResults.push_back(result);
    }
  }
  return AffineMap::get(inputMap.getNumDims(), inputMap.getNumSymbols(),
                        outputResults, getContext());
}

bool ReduceUDOp::dropsReducedDimensions() {
  auto inputType = cast<RankedTensorType>(getOperand(0).getType());
  auto outputType = cast<RankedTensorType>(getResult(0).getType());
  return reductionDropsDimensions(inputType.getRank(), outputType.getRank(),
                                  getDimensions());
}

llvm::SmallBitVector ReduceUDOp::getReductionDimMask() {
  auto inputType = cast<RankedTensorType>(getOperand(0).getType());
  return reductionDimMask(inputType.getRank(), getDimensions());
}

LogicalResult ReduceOp::verify() {
  auto inputType = getInput().getType();
  FailureOr<llvm::SmallBitVector> reducedDimensions = verifyReductionDimensions(
      getOperation(), inputType.getRank(), getDimensions());
  if (failed(reducedDimensions)) {
    return failure();
  }
  return verifyReductionResultShape(getOperation(), inputType.getShape(),
                                    getOutput().getType(), *reducedDimensions);
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

  auto firstInputType = cast<RankedTensorType>(getInputs().front().getType());
  SmallVector<int64_t> mergedInputShape(firstInputType.getShape());

  // Equal-merge all operand dimensions before checking results, so a static
  // operand can refine a corresponding dynamic operand dimension.
  for (auto [i, input] : llvm::enumerate(getInputs().drop_front())) {
    auto inputType = cast<RankedTensorType>(input.getType());
    if (inputType.getRank() != firstInputType.getRank()) {
      return emitOpError() << "expects all inputs to have the same rank, but "
                           << "input 0 has type " << firstInputType
                           << " and input " << i + 1 << " has type "
                           << inputType;
    }
    for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
      FailureOr<int64_t> merged = mergeEqualDimensions(
          mergedInputShape[dim], inputType.getDimSize(dim));
      if (failed(merged)) {
        return emitOpError()
               << "expects all input dimensions to merge, but input 0 has "
                  "type "
               << firstInputType << " and input " << i + 1 << " has type "
               << inputType;
      }
      mergedInputShape[dim] = *merged;
    }
  }

  FailureOr<llvm::SmallBitVector> reducedDimensions = verifyReductionDimensions(
      getOperation(), firstInputType.getRank(), getDimensions());
  if (failed(reducedDimensions)) {
    return failure();
  }

  // Shape verification admits only the fully retained rank or the fully
  // dropped rank, so equal result ranks are the same convention.
  int64_t canonicalRank =
      cast<RankedTensorType>(getResult(0).getType()).getRank();
  for (auto [i, result] : llvm::enumerate(getResults())) {
    auto outputType = cast<RankedTensorType>(result.getType());
    if (failed(verifyReductionResultShape(getOperation(), mergedInputShape,
                                          outputType, *reducedDimensions))) {
      return failure();
    }
    if (outputType.getRank() != canonicalRank) {
      return emitOpError(
          "expects all results to use the same reduced-dimension convention");
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

  // The body has N left-hand arguments followed by N right-hand arguments and
  // yields N combined values.
  size_t expectedNumArgs = 2 * getNumResults();
  if (bodyBlockArgs.size() != expectedNumArgs) {
    return emitOpError() << "expects body region to have exactly "
                         << expectedNumArgs
                         << " arguments (lhs_i, rhs_i), but got "
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

    // Check that the block argument types in pair (lhs_i, rhs_i) match the
    // identity type.
    // Convert signed/unsigned integer types to signless integer types.
    auto argTypeLhs = bodyBlockArgs[i].getType();
    auto argTypeRhs = bodyBlockArgs[i + getNumOperands()].getType();
    auto yieldType = bodyBlockYields[i].getType();

    Type convertedIdentityType = identityType;
    if (identityType.isSignedInteger() || identityType.isUnsignedInteger()) {
      convertedIdentityType =
          IntegerType::get(getContext(), identityType.getIntOrFloatBitWidth());
    }

    if (argTypeLhs != convertedIdentityType) {
      return emitOpError() << "expects block argument " << i
                           << " type to match identity type, but got argument "
                              "type: "
                           << argTypeLhs
                           << " and identity type: " << convertedIdentityType;
    }
    if (argTypeRhs != convertedIdentityType) {
      return emitOpError()
             << "expects block argument " << i + getNumOperands()
             << " type to match identity type, but got argument type: "
             << argTypeRhs << " and identity type: " << convertedIdentityType;
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
                                ArrayRef<int64_t> shape,
                                bool dropsReducedDimensions) {
  // Build CuTe layout view of the underlying source.
  tcg::Shape cgShape(shape);
  tcg::Layout flat(cgShape);

  // Replace contracting dimensions with unit dimensions.
  std::vector<tcg::Layout> parts;
  for (size_t i = 0, n = tcg::rank(flat); i < n; ++i) {
    if (llvm::find(dimensions, i) == dimensions.end()) {
      parts.push_back(tcg::get(flat, i));
    } else if (!dropsReducedDimensions) {
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
  auto view =
      createReductionView(getDimensions(), shape, dropsReducedDimensions());
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
  auto view =
      createReductionView(getDimensions(), shape, dropsReducedDimensions());
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

LogicalResult ExpOp::verify() {
  Type elementType = getInput().getType().getElementType();
  if (!elementType.isF16() && !elementType.isBF16() && !elementType.isF32() &&
      !elementType.isF64()) {
    return emitOpError("rounding_mode requires f16, bf16, f32, or f64");
  }
  if (getRoundingMode() == ExpRoundingMode::Approx && elementType.isF64()) {
    return emitOpError("approx rounding_mode requires f16, bf16, or f32");
  }
  return success();
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

  auto firstInputType = cast<RankedTensorType>(getInputs()[0].getType());
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
    auto inputType = cast<RankedTensorType>(input.getType());
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

LogicalResult ContractionOp::verify() {
  SmallVector<AffineMap> maps;
  for (Attribute attr : getIndexingMaps()) {
    maps.push_back(cast<AffineMapAttr>(attr).getValue());
  }
  RankedTensorType types[] = {getA().getType(), getB().getType(),
                              getC().getType()};
  unsigned numDims = maps.front().getNumDims();
  SmallVector<SmallVector<int64_t>> positions(
      3, SmallVector<int64_t>(numDims, -1));
  for (auto [i, map] : llvm::enumerate(maps)) {
    if (map.getNumSymbols() || !map.isProjectedPermutation()) {
      return emitOpError() << "indexing map " << i
                           << " must be a symbol-free projected permutation";
    }
    if (map.getNumDims() != numDims) {
      return emitOpError("expects indexing maps to have the same domain rank");
    }
    if (map.getNumResults() != types[i].getRank()) {
      return emitOpError() << "indexing map " << i
                           << " result count must match its tensor rank";
    }
    for (auto [j, expr] : llvm::enumerate(map.getResults())) {
      positions[i][cast<AffineDimExpr>(expr).getPosition()] = j;
    }
  }

  // Infer dimension roles from membership in A, B, and C. No iterator
  // attributes or canonical ordering of the domain dimensions are needed.
  SmallVector<unsigned> roles;
  for (unsigned dim = 0; dim < numDims; ++dim) {
    int64_t aDim = positions[0][dim];
    int64_t bDim = positions[1][dim];
    int64_t cDim = positions[2][dim];
    unsigned role =
        (aDim >= 0 ? 1 : 0) | (bDim >= 0 ? 2 : 0) | (cDim >= 0 ? 4 : 0);
    roles.push_back(role);
    switch (role) {
    case 7: { // Batch.
      auto extent = mergeBroadcastDimensions(types[0].getDimSize(aDim),
                                             types[1].getDimSize(bDim));
      if (failed(extent) || types[2].getDimSize(cDim) != *extent) {
        return emitOpError() << "expects C batch dimension " << cDim
                             << " to be the broadcasted extent of A and B";
      }
      break;
    }
    case 5:   // Lhs product.
    case 6: { // Rhs product.
      unsigned input = role == 5 ? 0 : 1;
      int64_t inputDim = role == 5 ? aDim : bDim;
      if (types[2].getDimSize(cDim) != types[input].getDimSize(inputDim)) {
        return emitOpError()
               << "expects C product dimension " << cDim
               << " to be inherited exactly from " << (input == 0 ? "A" : "B");
      }
      break;
    }
    case 3: // Contraction.
      if (failed(mergeEqualDimensions(types[0].getDimSize(aDim),
                                      types[1].getDimSize(bDim)))) {
        return emitOpError() << "expects contraction domain dimension " << dim
                             << " of A and B to match";
      }
      break;
    default:
      return emitOpError()
             << "domain dimension " << dim
             << " must be a batch, product, or contraction dimension";
    }
  }

  unsigned previousGroup = 0;
  for (AffineExpr expr : maps[2].getResults()) {
    unsigned role = roles[cast<AffineDimExpr>(expr).getPosition()];
    unsigned group = role == 7 ? 0 : (role == 5 ? 1 : 2);
    if (group < previousGroup) {
      return emitOpError(
          "expects C dimensions in batch, lhs product, rhs product order");
    }
    previousGroup = group;
  }
  if (getAcc() && getAcc().getType() != getC().getType()) {
    return emitOpError("expects accumulator type to match result type");
  }
  return success();
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
    if (inputShape[i] == 1 &&
        (outputShape[i] > 1 || ShapedType::isDynamic(outputShape[i]))) {
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
  if (auto resultType = dyn_cast<RankedTensorType>(getResult().getType())) {
    return resultType.getRank();
  }
  return 0;
}

AffineMap ConstantOp::computeCanonicalOutputIterSpaceMap(
    std::optional<unsigned> iterSpaceRank) {
  unsigned numIterSpaceDims = iterSpaceRank.value_or(getNumIterSpaceDims());
  // For scalar constants, return an empty 0-result map since they have no
  // iteration space dimensions.
  if (!isa<RankedTensorType>(getResult().getType())) {
    return AffineMap::get(numIterSpaceDims, /*symbolCount=*/0, /*results=*/{},
                          getContext());
  }

  // For tensor constants, use identity map if ranks match.
  auto resultType = cast<RankedTensorType>(getResult().getType());
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
