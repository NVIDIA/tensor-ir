// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// This pass changes only the program boundary. Tensor arguments become
/// strided memrefs, and results become appended destination memrefs.
/// `bufferization.to_tensor` and `bufferization.materialize_in_destination`
/// bridge those buffers to the original tensor computation. The pass creates
/// no memory accesses and performs no tiling or grid construction; grid
/// formation replaces the bridges with tiled TensorIR loads and stores.
///
/// Each buffer binds to exactly one tensor, recorded by `restrict` on both
/// bridge operations.

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/Passes.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DEF_BUFFERIZEPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kDispatchSuffix = "_dispatch";

/// Returns the boundary memref type for a tensor. Descriptors without explicit
/// strides use the identity (row-major) layout.
static FailureOr<MemRefType>
getMemRefType(Type type, const TensorDescriptor &descriptor, Location loc) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType) {
    return emitError(loc) << "expected a TensorIR tensor type, got " << type;
  }

  MemRefLayoutAttrInterface layout;
  if (!descriptor.strides.empty()) {
    if (descriptor.strides.size() !=
        static_cast<size_t>(tensorType.getRank())) {
      return emitError(loc)
             << "expected " << tensorType.getRank()
             << " descriptor strides, got " << descriptor.strides.size();
    }
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
    if (!isa<RankedTensorType>(type)) {
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

  size_t outputBase = graph.getNumArguments();
  auto setAlignment = [&](unsigned index, const TensorDescriptor &descriptor) {
    int64_t alignment = std::max<int64_t>(1, descriptor.alignment);
    function.setArgAttr(index, TensorIRDialect::getAlignmentAttrName(),
                        builder.getI64IntegerAttr(alignment));
  };
  for (auto [index, descriptor] : llvm::enumerate(outputDescriptors)) {
    setAlignment(outputBase + index, descriptor);
  }

  IRMapping mapping;
  for (auto [index, pair] : llvm::enumerate(llvm::zip_equal(
           graph.getArguments(),
           function.getArguments().take_front(graph.getNumArguments())))) {
    auto [oldArgument, newArgument] = pair;
    if (!isa<RankedTensorType>(oldArgument.getType())) {
      mapping.map(oldArgument, newArgument);
      continue;
    }
    setAlignment(index, inputDescriptors[index]);
    Value tensor = bufferization::ToTensorOp::create(
        builder, graph.getLoc(), oldArgument.getType(), newArgument,
        /*restrict=*/true, /*writable=*/false);
    mapping.map(oldArgument, tensor);
  }

  for (Operation &op : results->getBlock()->without_terminator()) {
    builder.clone(op, mapping);
  }

  for (auto [index, result] : llvm::enumerate(results.getOperands())) {
    Value mappedResult = mapping.lookupOrNull(result);
    if (!mappedResult) {
      function.erase();
      return results.emitError() << "failed to map graph result #" << index;
    }
    bufferization::MaterializeInDestinationOp::create(
        builder, graph.getLoc(), /*result=*/Type(), mappedResult,
        function.getArgument(outputBase + index), /*restrict=*/true,
        /*writable=*/true);
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
