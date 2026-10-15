// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/HostLoweringUtils.h"
#include "tensor_ir/Transform/Passes.h" // IWYU pragma: keep

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::nv_tensor_ir {
#define GEN_PASS_DEF_HOSTABIBRIDGEPASS
#include "tensor_ir/Transform/Passes.h.inc"
} // namespace mlir::nv_tensor_ir

using namespace mlir;
using namespace mlir::nv_tensor_ir;

/// Insert an unrealized conversion cast to `type` if types differ.
static Value createUnrealizedCast(OpBuilder &b, Location loc, Type type,
                                  Value value) {
  if (type == value.getType()) {
    return value;
  }
  return UnrealizedConversionCastOp::create(b, loc, TypeRange{type},
                                            ValueRange{value})
      .getResult(0);
}

template <typename Successor = RegionSuccessor>
static Successor getParentSuccessor(Operation *parent) {
  if constexpr (std::is_constructible_v<Successor, Operation *>) {
    return Successor(parent);
  } else {
    return Successor::parent();
  }
}

namespace {
/// Decodes the host packedArgs buffer into the SSA arguments of the host
/// dispatch function. Device kernel parameters are built per launch by
/// `HostToLLVM` from each launch's own operands.
///
/// Memory layout of caller-provided packedArgs (contiguous int64_t slots):
/// +------------------------------------------------------------------------+
/// | Tensor Argument (1 + 2 * rank slots):                                  |
/// |   slot 0:        Base data pointer bits (int64_t)                      |
/// |   slot 1..R:     Sizes for dim 0 .. dim R-1 (int64_t each)             |
/// |   slot R+1..2R:  Strides for dim 0 .. dim R-1 (int64_t each)           |
/// +------------------------------------------------------------------------+
/// | Scalar Argument (1 slot):                                              |
/// |   slot 0:        Scalar value bits right-justified in low bytes        |
/// +------------------------------------------------------------------------+
class ArgumentUnpacker {
public:
  ArgumentUnpacker(const HostLLVMTypeConverter &converter, OpBuilder &b,
                   Location loc, Value packed)
      : b(b), loc(loc), packed(packed), converter(converter) {}

  /// Unpack the next argument of `type` from packedArgs into an SSA value for
  /// the host dispatch function.
  Value unpack(Type type) {
    if (auto memref = dyn_cast<MemRefType>(type)) {
      return unpackMemRef(memref);
    }
    return unpackScalar(type);
  }

private:
  /// Load the int64_t value from the next packedArgs slot and advance.
  Value readHostValue() {
    Value slot = LLVM::GEPOp::create(
        b, loc, packed.getType(), b.getI64Type(), packed,
        ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(nextHostSlot++)});
    return LLVM::LoadOp::create(b, loc, b.getI64Type(), slot);
  }

  /// Unpack a MemRefDescriptor with a zero offset from packedArgs.
  Value unpackMemRef(MemRefType type) {
    auto descriptor =
        MemRefDescriptor::poison(b, loc, converter.convertType(type));
    Value raw = readHostValue();
    auto ptr = LLVM::LLVMPointerType::get(b.getContext());
    Value base = LLVM::IntToPtrOp::create(b, loc, ptr, raw);
    descriptor.setAllocatedPtr(b, loc, base);
    descriptor.setAlignedPtr(b, loc, base);
    descriptor.setConstantOffset(b, loc, 0);

    SmallVector<Value> sizes(type.getRank()), strides(type.getRank());
    for (Value &size : sizes) {
      size = readHostValue();
    }
    for (Value &stride : strides) {
      stride = readHostValue();
    }
    for (int dim = 0; dim < type.getRank(); ++dim) {
      descriptor.setSize(b, loc, dim, sizes[dim]);
      descriptor.setStride(b, loc, dim, strides[dim]);
    }
    return createUnrealizedCast(b, loc, type, descriptor);
  }

  /// Unpack a scalar value from the low bits of the next packedArgs slot.
  Value unpackScalar(Type type) {
    Value raw = readHostValue();
    Type loweredType = converter.convertType(type);
    unsigned width = loweredType.getIntOrFloatBitWidth();
    if (width < 64) {
      raw = LLVM::TruncOp::create(b, loc, b.getIntegerType(width), raw);
    }
    if (isa<FloatType>(loweredType)) {
      raw = LLVM::BitcastOp::create(b, loc, loweredType, raw);
    }
    return createUnrealizedCast(b, loc, type, raw);
  }

  OpBuilder &b;
  Location loc;
  Value packed;
  const HostLLVMTypeConverter &converter;
  unsigned nextHostSlot = 0;
};
} // namespace

/// Build the public C entry wrapper `tensor_ir_host_entry_<func>_v1` for each
/// public host entrypoint in the module.
static LogicalResult
emitTensorIRDispatchCIfaceWrappers(RewriterBase &b, ModuleOp module,
                                   const HostLLVMTypeConverter &converter) {

  // Collect entry points first: the wrappers are public functions too.
  SmallVector<func::FuncOp> entrypoints;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isPublic() && !function.isDeclaration()) {
      entrypoints.push_back(function);
    }
  }

  SmallVector<Attribute> entrySymbols;
  for (func::FuncOp entrypoint : entrypoints) {
    assert(entrypoint.getNumArguments() > 0 &&
           entrypoint.getArgAttrOfType<UnitAttr>(
               0, TensorIRDialect::getRuntimeContextAttrName()) &&
           entrypoint.getNumResults() == 1 &&
           "public host functions take the runtime context and return a "
           "status");

    // Create the dispatch wrapper entrypoint.
    b.setInsertionPointToEnd(module.getBody());
    Location loc = entrypoint->getLoc();
    auto ptr = LLVM::LLVMPointerType::get(b.getContext());
    auto i32 = b.getI32Type();
    std::string name =
        "tensor_ir_host_entry_" + entrypoint.getName().str() + "_v1";
    entrySymbols.push_back(b.getStringAttr(name));
    auto entry = func::FuncOp::create(
        b, loc, name, b.getFunctionType({ptr, ptr, i32}, {i32}));
    Block *start = entry.addEntryBlock();
    b.setInsertionPointToStart(start);

    ArgumentUnpacker unpacker(converter, b, loc, start->getArgument(1));

    SmallVector<Value> arguments{start->getArgument(0)};
    for (Type type : entrypoint.getArgumentTypes().drop_front()) {
      Value argument = unpacker.unpack(type);
      arguments.push_back(argument);
    }
    auto call = func::CallOp::create(b, loc, entrypoint, arguments);
    func::ReturnOp::create(b, loc, call.getResult(0));

    entrypoint.setPrivate();
  }
  module->setAttr(TensorIRDialect::getNativeEntrypointsAttrName(),
                  b.getArrayAttr(entrySymbols));

  return success();
}

/// Every defined host function takes the runtime context and returns a status.
/// Declarations and functions inside `gpu.module` keep their signatures.
static bool takesRuntimeContext(func::FuncOp function, ModuleOp module) {
  return function->getParentOp() == module && !function.isDeclaration();
}

/// Rebuild `op` with an additional leading result of `type`, like
/// `appendOpResults` in upstream OwnershipBasedBufferDeallocation.
static Operation *prependOpResult(RewriterBase &b, Operation *op, Type type) {
  SmallVector<Type> resultTypes{type};
  llvm::append_range(resultTypes, op->getResultTypes());
  Operation *newOp = Operation::create(
      op->getLoc(), op->getName(), resultTypes, op->getOperands(),
      op->getAttrDictionary(), op->getPropertiesStorage(), op->getSuccessors(),
      op->getNumRegions());
  for (auto [oldRegion, newRegion] :
       llvm::zip(op->getRegions(), newOp->getRegions())) {
    newRegion.takeBody(oldRegion);
  }
  b.setInsertionPoint(op);
  b.insert(newOp);
  b.replaceOp(op, newOp->getResults().drop_front());
  return newOp;
}

/// Prepend an empty dictionary to an optional per-argument or per-result
/// attribute array.
static ArrayAttr prependEmptyDictionary(Builder &b, ArrayAttr attrs) {
  if (!attrs) {
    return attrs;
  }
  SmallVector<Attribute> newAttrs{b.getDictionaryAttr({})};
  llvm::append_range(newAttrs, attrs);
  return b.getArrayAttr(newAttrs);
}

namespace {
/// Threads the runtime context and status through one host function, in the
/// style of upstream OwnershipBasedBufferDeallocation: a walk over the blocks
/// of the function applies the handler of every interface an op implements,
/// and the signature is updated afterwards. Callers and callees agree through
/// `takesRuntimeContext`, so functions can be processed in any order.
class RuntimeContextThreading {
public:
  RuntimeContextThreading(RewriterBase &b, SymbolTableCollection &symbolTables,
                          ModuleOp module, func::FuncOp function)
      : b(b), symbolTables(symbolTables), module(module), function(function) {}

  LogicalResult process();

private:
  /// The base case for the recursive template below.
  template <typename... T>
  std::enable_if_t<sizeof...(T) == 0, FailureOr<Operation *>>
  handleOp(Operation *op) {
    return op;
  }

  /// Apply the handlers of all interfaces in the template list that `op`
  /// implements, in list order, to `op` or to its replacement.
  template <typename InterfaceT, typename... InterfacesU>
  FailureOr<Operation *> handleOp(Operation *op) {
    Operation *next = op;
    if (auto concreteOp = dyn_cast<InterfaceT>(op)) {
      FailureOr<Operation *> result = handleInterface(concreteOp);
      if (failed(result)) {
        return failure();
      }
      next = *result;
    }
    return handleOp<InterfacesU...>(next);
  }

  LogicalResult verifyFunctionPreconditions();
  LogicalResult insertContextArgument();
  LogicalResult process(Block *block);

  /// Forward the context to a callee that takes it and rebuild the call with a
  /// leading status result.
  FailureOr<Operation *> handleInterface(CallOpInterface call);

  /// Report success (0) from the function's returns.
  FailureOr<Operation *>
  handleInterface(RegionBranchTerminatorOpInterface terminator);

  /// Branch to the failure block when a call returns a nonzero status. Runs
  /// after the walk because it splits blocks.
  void insertStatusChecks();
  Block *createFailureBlock();
  LogicalResult updateFunctionSignature();

  RewriterBase &b;
  SymbolTableCollection &symbolTables;
  ModuleOp module;
  func::FuncOp function;
  SmallVector<Value> statuses;
};
} // namespace

LogicalResult RuntimeContextThreading::process() {
  if (failed(verifyFunctionPreconditions()) ||
      failed(insertContextArgument())) {
    return failure();
  }
  WalkResult result =
      function->walk<WalkOrder::PostOrder, ForwardDominanceIterator<>>(
          [&](Block *block) {
            if (failed(process(block))) {
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
  if (result.wasInterrupted()) {
    return failure();
  }
  insertStatusChecks();
  return updateFunctionSignature();
}

LogicalResult RuntimeContextThreading::verifyFunctionPreconditions() {
  if (function.isPublic() && function.getNumResults() != 0) {
    return function.emitOpError(
        "expects a public host function to return nothing");
  }
  return success();
}

LogicalResult RuntimeContextThreading::insertContextArgument() {
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  DictionaryAttr argAttrs = b.getDictionaryAttr(b.getNamedAttr(
      TensorIRDialect::getRuntimeContextAttrName(), b.getUnitAttr()));
  LogicalResult inserted = failure();
  b.modifyOpInPlace(function, [&] {
    inserted = function.insertArgument(0, ptr, argAttrs, function.getLoc());
  });
  if (failed(inserted)) {
    return function.emitOpError(
        "failed to insert the runtime context argument");
  }
  return success();
}

LogicalResult RuntimeContextThreading::process(Block *block) {
  for (Operation &op : llvm::make_early_inc_range(*block)) {
    FailureOr<Operation *> result =
        handleOp<CallOpInterface, RegionBranchTerminatorOpInterface>(&op);
    if (failed(result)) {
      return failure();
    }
  }
  return success();
}

FailureOr<Operation *>
RuntimeContextThreading::handleInterface(CallOpInterface call) {
  auto callee = dyn_cast_or_null<func::FuncOp>(
      call.resolveCallableInTable(&symbolTables));
  if (!callee || !takesRuntimeContext(callee, module)) {
    return call.getOperation();
  }
  // The status check splits the block containing the call.
  if (call->getParentOp() != function) {
    return call->emitOpError(
        "calls a function that takes the runtime context, so it must be "
        "directly in a func.func body; lower SCF to CF first");
  }

  SmallVector<Value> operands{function.getArgument(0)};
  llvm::append_range(operands, call.getArgOperands());
  b.modifyOpInPlace(call,
                    [&] { call.getArgOperandsMutable().assign(operands); });
  Operation *statusCall = prependOpResult(b, call, b.getI32Type());
  if (auto attrs = dyn_cast<ArgAndResultAttrsOpInterface>(statusCall)) {
    ArrayAttr argAttrs = prependEmptyDictionary(b, attrs.getArgAttrsAttr());
    ArrayAttr resAttrs = prependEmptyDictionary(b, attrs.getResAttrsAttr());
    b.modifyOpInPlace(statusCall, [&] {
      if (argAttrs) {
        attrs.setArgAttrsAttr(argAttrs);
      }
      if (resAttrs) {
        attrs.setResAttrsAttr(resAttrs);
      }
    });
  }
  statuses.push_back(statusCall->getResult(0));
  return statusCall;
}

FailureOr<Operation *> RuntimeContextThreading::handleInterface(
    RegionBranchTerminatorOpInterface terminator) {
  if (terminator->getParentOp() != function) {
    return terminator.getOperation();
  }
  b.setInsertionPoint(terminator);
  Value zero =
      arith::ConstantOp::create(b, terminator.getLoc(), b.getI32IntegerAttr(0));
  MutableOperandRange operands = terminator.getMutableSuccessorOperands(
      getParentSuccessor(function.getOperation()));
  SmallVector<Value> newOperands{zero};
  llvm::append_range(newOperands, operands.getAsOperandRange());
  b.modifyOpInPlace(terminator, [&] { operands.assign(newOperands); });
  return terminator.getOperation();
}

void RuntimeContextThreading::insertStatusChecks() {
  Block *failureBlock = nullptr;
  for (Value status : statuses) {
    if (!failureBlock) {
      failureBlock = createFailureBlock();
    }
    Operation *call = status.getDefiningOp();
    Location loc = call->getLoc();
    Block *before = call->getBlock();
    Block *continuation = b.splitBlock(before, std::next(call->getIterator()));
    b.setInsertionPointToEnd(before);
    Value zero = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(0));
    Value isError =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, status, zero);
    cf::CondBranchOp::create(b, loc, isError, failureBlock, ValueRange{status},
                             continuation, ValueRange{});
  }
}

/// Create a block at the end of the function that returns its `i32` status
/// argument, with poison for the function's existing results. The status
/// result is not part of the signature yet.
Block *RuntimeContextThreading::createFailureBlock() {
  OpBuilder::InsertionGuard guard(b);
  Location loc = function.getLoc();
  Region &body = function.getBody();
  Type i32 = b.getI32Type();
  Block *block = b.createBlock(&body, body.end(), TypeRange{i32}, {loc});
  SmallVector<Value> results{block->getArgument(0)};
  for (Type type : function.getResultTypes()) {
    Value poison = ub::PoisonOp::create(b, loc, type);
    results.push_back(poison);
  }
  func::ReturnOp::create(b, loc, results);
  return block;
}

LogicalResult RuntimeContextThreading::updateFunctionSignature() {
  LogicalResult inserted = failure();
  b.modifyOpInPlace(function, [&] {
    inserted = function.insertResult(0, b.getI32Type(), DictionaryAttr());
  });
  if (failed(inserted)) {
    return function.emitOpError("failed to insert the status result");
  }
  return success();
}

/// Add a leading runtime context argument (`!llvm.ptr` with the
/// `nv_tensor_ir.runtime_context` UnitAttr) and a leading `i32` status result
/// to every defined host function. Returns report success (0). Each call to a
/// defined host function forwards the caller's context and returns a nonzero
/// callee status from the caller, with poison for the caller's other results.
///
/// Public functions must return nothing. Calls to defined host functions must
/// be directly in the caller's body: structured control flow must already be
/// lowered to CF.
static LogicalResult addRuntimeContextArguments(RewriterBase &b,
                                                ModuleOp module) {
  SymbolTableCollection symbolTables;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (!takesRuntimeContext(function, module)) {
      continue;
    }
    RuntimeContextThreading threading(b, symbolTables, module, function);
    if (failed(threading.process())) {
      return failure();
    }
  }
  return success();
}

namespace {
struct HostAbiBridgePass
    : nv_tensor_ir::impl::HostAbiBridgePassBase<HostAbiBridgePass> {
  using HostAbiBridgePassBase::HostAbiBridgePassBase;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter b(module);

    if (failed(addRuntimeContextArguments(b, module))) {
      return signalPassFailure();
    }

    auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    markAnalysesPreserved<DataLayoutAnalysis>();

    HostLLVMTypeConverter converter(b.getContext(), dataLayoutAnalysis);

    if (failed(emitTensorIRDispatchCIfaceWrappers(b, module, converter))) {
      return signalPassFailure();
    }
  }
};
} // namespace
