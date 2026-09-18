// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Bufferize.cpp - Form TensorIR buffer boundaries --------------------===//
//
// This pass changes only the program boundary. Tensor-valued graph arguments
// become strided memref arguments, results become destination memrefs, and
// direct TensorIR loads/stores bridge those buffers to the original tensor
// computation. It deliberately performs no tiling or grid construction.
//
// The buffer binding below is one-to-one today. Keeping the pointer and view
// metadata independent in the resulting IR allows a future buffer-assignment
// pass to share a pointer between logical tensors without changing the memory
// operations. This pass does not infer or encode aliasing.
//
//===----------------------------------------------------------------------===//

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/Passes.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DEF_BUFFERIZEPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kDispatchSuffix = "_dispatch";

/// The physical buffer and the SSA metadata needed by direct memory accesses.
struct BufferBinding {
  Value pointer;
  SmallVector<Value> sizes;
  SmallVector<Value> strides;
  DenseI64ArrayAttr staticOffsets;
  DenseI64ArrayAttr staticSizes;
  DenseI64ArrayAttr staticStrides;
  IntegerAttr alignment;
};

/// Materialize row-major strides when the frontend descriptor omitted them.
static void materializeDefaultStrides(TensorDescriptor &descriptor) {
  if (!descriptor.strides.empty() || descriptor.sizes.empty()) {
    return;
  }
  descriptor.strides.resize(descriptor.sizes.size());
  int64_t running = 1;
  for (size_t position = descriptor.sizes.size(); position > 0; --position) {
    size_t dimension = position - 1;
    descriptor.strides[dimension].staticValue = running;
    if (ShapedType::isDynamic(running) ||
        ShapedType::isDynamic(descriptor.sizes[dimension].staticValue) ||
        llvm::MulOverflow(running, descriptor.sizes[dimension].staticValue,
                          running)) {
      running = ShapedType::kDynamic;
    }
  }
}

static FailureOr<MemRefType>
getMemRefType(Type type, TensorDescriptor &descriptor, Location loc) {
  auto tensorType = dyn_cast<TensorType>(type);
  if (!tensorType) {
    return emitError(loc) << "expected a TensorIR tensor type, got " << type;
  }
  bool hasExplicitStrides = !descriptor.strides.empty();
  // Normalize the descriptor even when the memref can keep its implicit
  // identity layout. The normalized strides are reused by the TensorIR
  // boundary load or store created from this descriptor.
  materializeDefaultStrides(descriptor);
  if (descriptor.strides.size() != static_cast<size_t>(tensorType.getRank())) {
    return emitError(loc) << "expected " << tensorType.getRank()
                          << " descriptor strides, got "
                          << descriptor.strides.size();
  }

  MemRefLayoutAttrInterface layout;
  if (hasExplicitStrides) {
    SmallVector<int64_t> strides;
    llvm::transform(descriptor.strides, std::back_inserter(strides),
                    [](const TensorDescriptor::StaticOrDynamic &stride) {
                      return stride.staticValue;
                    });
    layout = StridedLayoutAttr::get(type.getContext(), /*offset=*/0, strides);
  }
  auto memorySpace = ptr::GenericSpaceAttr::get(type.getContext());
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         layout, memorySpace);
}

static FailureOr<BufferBinding>
createBufferBinding(OpBuilder &builder, Location loc, Value buffer,
                    const TensorDescriptor &descriptor) {
  auto memrefType = dyn_cast<MemRefType>(buffer.getType());
  if (!memrefType) {
    return emitError(loc) << "expected a memref buffer, got "
                          << buffer.getType();
  }
  auto metadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, buffer);
  auto memorySpace =
      cast<ptr::MemorySpaceAttrInterface>(memrefType.getMemorySpace());
  auto pointerType = ptr::PtrType::get(builder.getContext(), memorySpace);
  Value pointer =
      ptr::ToPtrOp::create(builder, loc, pointerType, buffer).getResult();

  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> dynamicStrides;
  shape.reserve(descriptor.sizes.size());
  strides.reserve(descriptor.strides.size());
  for (auto [dimension, size] : llvm::enumerate(descriptor.sizes)) {
    shape.push_back(size.staticValue);
    if (ShapedType::isDynamic(size.staticValue)) {
      dynamicSizes.push_back(metadata.getSizes()[dimension]);
    }
  }
  for (auto [dimension, stride] : llvm::enumerate(descriptor.strides)) {
    strides.push_back(stride.staticValue);
    if (ShapedType::isDynamic(stride.staticValue)) {
      dynamicStrides.push_back(metadata.getStrides()[dimension]);
    }
  }
  int64_t alignment = std::max<int64_t>(1, descriptor.alignment);
  return BufferBinding{pointer,
                       std::move(dynamicSizes),
                       std::move(dynamicStrides),
                       builder.getDenseI64ArrayAttr({0}),
                       builder.getDenseI64ArrayAttr(shape),
                       builder.getDenseI64ArrayAttr(strides),
                       builder.getI64IntegerAttr(alignment)};
}

static SmallVector<Value> createZeroCoordinates(OpBuilder &builder,
                                                Location loc, int64_t rank) {
  if (rank == 0) {
    return {};
  }
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  return SmallVector<Value>(rank, zero);
}

static void copySchedulingMetadata(GraphOp graph, ResultsOp results,
                                   func::FuncOp function) {
  function->setAttr(TensorIRDialect::getBufferizedProgramAttrName(),
                    UnitAttr::get(graph.getContext()));
  for (StringRef name : {TensorIRDialect::getTileSizeAttrName()}) {
    if (Attribute attr = graph->getAttr(name)) {
      function->setAttr(name, attr);
    }
  }
  for (StringRef name : {TensorIRDialect::getIterationSpaceAttrName(),
                         TensorIRDialect::getResultLayoutsAttrName(),
                         TensorIRDialect::getResultViewsAttrName()}) {
    if (Attribute attr = results->getAttr(name)) {
      function->setAttr(name, attr);
    }
  }
}

static LogicalResult bufferizeGraph(GraphOp graph) {
  auto results = dyn_cast<ResultsOp>(graph.getBody()->getTerminator());
  if (!results) {
    return graph.emitError("expected TensorIR graph results terminator");
  }
  MLIR_ASSIGN_OR_RETURN(
      auto inputDescriptors,
      getTensorDescriptors(graph.getArgumentTypes(), graph.getAllArgAttrs()));
  MLIR_ASSIGN_OR_RETURN(
      auto outputDescriptors,
      getTensorDescriptors(graph.getResultTypes(), graph.getAllResultAttrs()));

  SmallVector<Type> functionInputs;
  for (auto [type, descriptor] :
       llvm::zip_equal(graph.getArgumentTypes(), inputDescriptors)) {
    if (!isa<TensorType>(type)) {
      functionInputs.push_back(type);
      continue;
    }
    MLIR_ASSIGN_OR_RETURN(MemRefType memref,
                          getMemRefType(type, descriptor, graph.getLoc()));
    functionInputs.push_back(memref);
  }
  for (auto [type, descriptor] :
       llvm::zip_equal(graph.getResultTypes(), outputDescriptors)) {
    MLIR_ASSIGN_OR_RETURN(MemRefType memref,
                          getMemRefType(type, descriptor, graph.getLoc()));
    functionInputs.push_back(memref);
  }

  std::string functionName = (graph.getName() + kDispatchSuffix).str();
  if (SymbolTable::lookupSymbolIn(graph->getParentOp(), functionName)) {
    return graph.emitError() << "cannot create bufferized function @"
                             << functionName << "; symbol already exists";
  }
  auto functionType =
      FunctionType::get(graph.getContext(), functionInputs, TypeRange{});
  func::FuncOp function =
      func::FuncOp::create(graph.getLoc(), functionName, functionType);
  graph->getBlock()->getOperations().insert(graph->getIterator(), function);
  copySchedulingMetadata(graph, results, function);
  Block *entry = function.addEntryBlock();
  OpBuilder builder = OpBuilder::atBlockBegin(entry);

  IRMapping mapping;
  for (auto [index, pair] : llvm::enumerate(llvm::zip_equal(
           graph.getArguments(),
           function.getArguments().take_front(graph.getNumArguments())))) {
    auto [oldArgument, newArgument] = pair;
    if (!isa<TensorType>(oldArgument.getType())) {
      mapping.map(oldArgument, newArgument);
      continue;
    }
    MLIR_ASSIGN_OR_RETURN(BufferBinding binding,
                          createBufferBinding(builder, graph.getLoc(),
                                              newArgument,
                                              inputDescriptors[index]));
    auto tensorType = cast<TensorType>(oldArgument.getType());
    SmallVector<Value> coordinates =
        createZeroCoordinates(builder, graph.getLoc(), tensorType.getRank());
    Value loaded = LoadOp::create(
        builder, graph.getLoc(), tensorType, binding.pointer,
        /*offsets=*/ValueRange{}, binding.sizes, binding.strides, coordinates,
        binding.staticOffsets, binding.staticSizes, binding.staticStrides,
        builder.getDenseI64ArrayAttr(
            SmallVector<int64_t>(coordinates.size(), ShapedType::kDynamic)),
        binding.alignment,
        /*padding=*/TilePaddingAttr{});
    mapping.map(oldArgument, loaded);
  }

  for (Operation &op : results->getBlock()->without_terminator()) {
    builder.clone(op, mapping);
  }

  size_t outputBase = graph.getNumArguments();
  for (auto [index, result] : llvm::enumerate(results.getOperands())) {
    Value mappedResult = mapping.lookupOrNull(result);
    if (!mappedResult) {
      function.erase();
      return results.emitError() << "failed to map graph result #" << index;
    }
    Value output = function.getArgument(outputBase + index);
    MLIR_ASSIGN_OR_RETURN(BufferBinding binding,
                          createBufferBinding(builder, graph.getLoc(), output,
                                              outputDescriptors[index]));
    auto tensorType = cast<TensorType>(mappedResult.getType());
    SmallVector<Value> coordinates =
        createZeroCoordinates(builder, graph.getLoc(), tensorType.getRank());
    StoreOp::create(builder, graph.getLoc(), mappedResult, binding.pointer,
                    /*offsets=*/ValueRange{}, binding.sizes, binding.strides,
                    coordinates, binding.staticOffsets, binding.staticSizes,
                    binding.staticStrides,
                    builder.getDenseI64ArrayAttr(SmallVector<int64_t>(
                        coordinates.size(), ShapedType::kDynamic)),
                    binding.alignment);
  }
  func::ReturnOp::create(builder, graph.getLoc());
  graph.erase();
  return success();
}

struct BufferizePass : public impl::BufferizePassBase<BufferizePass> {
  using BufferizePassBase::BufferizePassBase;

  void runOnOperation() override {
    SmallVector<GraphOp> graphs;
    getOperation().walk([&](GraphOp graph) { graphs.push_back(graph); });
    for (GraphOp graph : graphs) {
      if (failed(bufferizeGraph(graph))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::nv_tensor_ir
