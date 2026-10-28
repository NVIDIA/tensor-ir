// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/HostLoweringUtils.h"
#include "tensor_ir/Transform/Passes.h" // IWYU pragma: keep

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/PtrToLLVM/PtrToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h" // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include <cstdint>

namespace mlir::nv_tensor_ir {
#define GEN_PASS_DEF_HOSTTOLLVMPASS
#include "tensor_ir/Transform/Passes.h.inc"
} // namespace mlir::nv_tensor_ir

using namespace mlir;
using namespace mlir::nv_tensor_ir;

/// Count the device kernel parameters of `launch`: the buffer pointer plus
/// dynamic sizes and explicit strides of each memref, and one per scalar.
static FailureOr<unsigned> countDeviceParameters(gpu::LaunchFuncOp launch) {
  unsigned count = 0;
  for (Type type : launch.getKernelOperands().getTypes()) {
    if (auto memref = dyn_cast<MemRefType>(type)) {
      auto [strides, offset] = memref.getStridesAndOffset();
      count += 1 + memref.getNumDynamicDims();
      if (!memref.getLayout().isIdentity()) {
        count += llvm::count_if(strides, ShapedType::isDynamic);
      }
    } else if (type.isIndex() ||
               (type.isIntOrFloat() && type.getIntOrFloatBitWidth() <= 64)) {
      ++count;
    } else {
      return launch.emitError("unsupported device kernel operand type ")
             << type;
    }
  }
  return count;
}

/// Return an error status with poison for the host helper's other results.
static void returnError(OpBuilder &b, Location loc, int32_t code) {
  auto function = b.getInsertionBlock()->getParentOp();
  auto func = cast<func::FuncOp>(function);
  Value status = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(code));
  SmallVector<Value> results{status};
  for (Type type : func.getResultTypes().drop_front()) {
    Value poison = ub::PoisonOp::create(b, loc, type);
    results.push_back(poison);
  }
  func::ReturnOp::create(b, loc, results);
}

/// Continue at the insertion point on success, otherwise return an error.
static void require(OpBuilder &b, Location loc, Value condition,
                    int32_t errorCode) {
  Block *before = b.getInsertionBlock();
  Block *after = before->splitBlock(b.getInsertionPoint());
  Block *error = b.createBlock(before->getParent());
  returnError(b, loc, errorCode);
  b.setInsertionPointToEnd(before);
  cf::CondBranchOp::create(b, loc, condition, after, ValueRange{}, error,
                           ValueRange{});
  b.setInsertionPointToStart(after);
}

/// Guard a signed host value before narrowing it to an ABI field. Continue at
/// the original insertion point only when the value is representable.
static void checkRange(OpBuilder &b, Location loc, Value value, int64_t minimum,
                       int64_t maximum, int32_t errorCode) {
  if (value.getType().isIndex()) {
    value = arith::IndexCastOp::create(b, loc, b.getI64Type(), value);
  } else if (value.getType().getIntOrFloatBitWidth() < 64) {
    value = LLVM::SExtOp::create(b, loc, b.getI64Type(), value);
  }
  Value low = LLVM::ConstantOp::create(b, loc, b.getI64IntegerAttr(minimum));
  Value high = LLVM::ConstantOp::create(b, loc, b.getI64IntegerAttr(maximum));
  Value above =
      LLVM::ICmpOp::create(b, loc, LLVM::ICmpPredicate::sge, value, low);
  Value below =
      LLVM::ICmpOp::create(b, loc, LLVM::ICmpPredicate::sle, value, high);
  Value valid = LLVM::AndOp::create(b, loc, above, below);
  require(b, loc, valid, errorCode);
}

namespace {
/// Populates the `void **kernelParams` table of one launch. The storage holds
/// `capacity` parameter pointers followed by `capacity` i64 value slots:
///
///   struct { void *params[capacity]; int64_t values[capacity]; }
///
/// LLVM computes field offsets and padding from this storage type.
///
/// Memref operands contribute their buffer pointer followed by their dynamic
/// sizes and strides narrowed to i32; scalar operands contribute their value.
class KernelParamsBuilder {
public:
  static LLVM::LLVMStructType getStorageType(OpBuilder &b, unsigned capacity) {
    auto ptr = LLVM::LLVMPointerType::get(b.getContext());
    auto params = LLVM::LLVMArrayType::get(ptr, capacity);
    auto values = LLVM::LLVMArrayType::get(b.getI64Type(), capacity);
    return LLVM::LLVMStructType::getLiteral(b.getContext(), {params, values});
  }

  KernelParamsBuilder(OpBuilder &b, Location loc,
                      const LLVMTypeConverter &converter, Value storage,
                      unsigned capacity)
      : b(b), converter(converter), storage(storage), capacity(capacity) {}

  void append(Value operand) {
    if (auto memref = dyn_cast<MemRefType>(operand.getType())) {
      appendMemRef(operand, memref);
    } else {
      appendScalar(operand);
    }
  }

private:
  /// Store `value` into the next value slot and point the next parameter at it.
  void appendValue(Value value) {
    Location loc = value.getLoc();
    assert(next < capacity && "kernel params storage too small");
    auto type = getStorageType(b, capacity);
    Value slot = LLVM::GEPOp::create(
        b, loc, storage.getType(), type, storage,
        ArrayRef<LLVM::GEPArg>{0, 1, static_cast<int32_t>(next)});
    LLVM::StoreOp::create(b, loc, value, slot);
    Value param = LLVM::GEPOp::create(
        b, loc, storage.getType(), type, storage,
        ArrayRef<LLVM::GEPArg>{0, 0, static_cast<int32_t>(next)});
    LLVM::StoreOp::create(b, loc, slot, param);
    ++next;
  }

  void appendMemRef(Value operand, MemRefType type) {
    Location loc = operand.getLoc();
    Type descriptorType = converter.convertType(type);
    Value lowered =
        UnrealizedConversionCastOp::create(b, loc, descriptorType, operand)
            ->getResult(0);
    MemRefDescriptor descriptor(lowered);
    Value base = descriptor.bufferPtr(b, loc, converter, type);
    appendValue(base);

    auto [strides, offset] = type.getStridesAndOffset();

    for (auto [idx, staticDimSize] : llvm::enumerate(type.getShape())) {
      if (!ShapedType::isDynamic(staticDimSize)) {
        continue;
      }
      Value size = descriptor.size(b, loc, idx);
      checkRange(b, loc, size, 0, INT32_MAX, -2);
      Value narrow = LLVM::TruncOp::create(b, loc, b.getI32Type(), size);
      appendValue(narrow);
    }
    if (type.getLayout().isIdentity()) {
      return;
    }
    for (auto [idx, staticStride] : llvm::enumerate(strides)) {
      if (!ShapedType::isDynamic(staticStride)) {
        continue;
      }
      Value stride = descriptor.stride(b, loc, idx);
      checkRange(b, loc, stride, INT32_MIN, INT32_MAX, -2);
      Value narrow = LLVM::TruncOp::create(b, loc, b.getI32Type(), stride);
      appendValue(narrow);
    }
  }

  void appendScalar(Value operand) {
    Location loc = operand.getLoc();
    if (operand.getType().isIndex()) {
      operand = arith::IndexCastOp::create(b, loc, b.getI64Type(), operand);
    } else if (Type converted = converter.convertType(operand.getType());
               converted && converted != operand.getType()) {
      // TensorIR scalar kernel arguments may carry signed/unsigned
      // IntegerType (e.g. si32); LLVM dialect ops only accept the signless
      // equivalent, and there is no dedicated arith op for this
      // conversion, so an explicit same-width reinterpret cast is
      // required.
      operand = UnrealizedConversionCastOp::create(b, loc, converted, operand)
                    ->getResult(0);
    }
    appendValue(operand);
  }

  OpBuilder &b;
  const LLVMTypeConverter &converter;
  Value storage;
  unsigned capacity;
  unsigned next = 0;
};
} // namespace

static Value createI64Constant(OpBuilder &b, Location loc, int64_t value) {
  return LLVM::ConstantOp::create(b, loc, b.getI64IntegerAttr(value));
}

static Value createI32Constant(OpBuilder &b, Location loc, int32_t value) {
  return arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(value));
}

/// Lower host launches to the TensorIRLaunchFnV1 callback:
///   int32_t status = runtime->launch(userData, kernelOrdinal, &config, params);
static LogicalResult
lowerGpuLaunchToCallback(IRRewriter &b, ModuleOp module,
                         const DataLayout &dataLayout,
                         const LLVMTypeConverter &converter) {

  SmallVector<gpu::LaunchFuncOp> launches;
  for (auto wrapper : module.getOps<FunctionOpInterface>()) {
    wrapper.walk([&](gpu::LaunchFuncOp launch) { launches.push_back(launch); });
  }

  // Size each function's kernel parameter storage for its largest launch.
  struct LaunchStorage {
    unsigned capacity = 0;
    Value config;
    Value params;
  };
  llvm::DenseMap<func::FuncOp, LaunchStorage> launchStorage;
  llvm::DenseMap<SymbolRefAttr, IntegerAttr> kernelOrdinals;
  int64_t nextOrdinal = 0;
  for (auto launch : launches) {
    if (auto ordinal = launch->getAttrOfType<IntegerAttr>(
            TensorIRDialect::getKernelOrdinalAttrName())) {
      auto [it, inserted] =
          kernelOrdinals.try_emplace(launch.getKernel(), ordinal);
      if (!inserted && it->second != ordinal) {
        return launch.emitError(
            "conflicting ordinals for the same device kernel");
      }
      nextOrdinal = std::max(nextOrdinal, ordinal.getInt() + 1);
    }
  }
  SymbolTableCollection symbolTables;
  for (auto launch : launches) {
    auto func = launch->getParentOfType<func::FuncOp>();
    if (!func || func->getParentOp() != module ||
        launch->getParentOp() != func) {
      return launch.emitError(
          "lower SCF to control flow before callback lowering "
          "in a top-level func.func");
    }
    if (launch.getAsyncToken() || !launch.getAsyncDependencies().empty() ||
        launch.getAsyncObject()) {
      return launch.emitError(
          "native host execution requires synchronous launches");
    }
    if (launch.getClusterSizeX() || launch.getClusterSizeY() ||
        launch.getClusterSizeZ()) {
      return launch.emitError(
          "native host ABI v1 does not support cluster dimensions");
    }
    if (func.getNumArguments() == 0 ||
        func.getArgument(0).getType() !=
            LLVM::LLVMPointerType::get(b.getContext()) ||
        !func.getArgAttr(0, TensorIRDialect::getRuntimeContextAttrName()) ||
        func.getNumResults() == 0 ||
        !func.getResultTypes().front().isInteger(32)) {
      return launch.emitError(
          "expected host function with runtime pointer argument "
          "and leading i32 result; run tir-host-abi-bridge first");
    }
    auto kernel = symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
        launch, launch.getKernel());
    if (!kernel || !llvm::equal(kernel.getArgumentTypes(),
                                launch.getKernelOperands().getTypes())) {
      return launch.emitError(
          "launch does not match the device kernel declaration");
    }
    auto [it, inserted] = kernelOrdinals.try_emplace(launch.getKernel());
    if (inserted) {
      it->second = b.getI32IntegerAttr(nextOrdinal++);
    }
    IntegerAttr ordinal = it->second;
    launch->setAttr(TensorIRDialect::getKernelOrdinalAttrName(), ordinal);
    kernel->setAttr(TensorIRDialect::getKernelOrdinalAttrName(), ordinal);
    FailureOr<unsigned> count = countDeviceParameters(launch);
    if (failed(count)) {
      return failure();
    }
    unsigned &capacity = launchStorage[func].capacity;
    capacity = std::max(capacity, *count);
  }

  LLVM::LLVMStructType launchConfigV1Type = LaunchConfigV1Builder::getType(b);
  uint64_t launchConfigV1Alignment =
      dataLayout.getTypePreferredAlignment(launchConfigV1Type);
  auto ptrType = LLVM::LLVMPointerType::get(module.getContext());

  // Allocate at function entry to prevent stack growth inside loops. Size for
  // the largest launch; synchronous callbacks allow reuse across launch sites.
  for (auto func : module.getOps<func::FuncOp>()) {
    auto it = launchStorage.find(func);
    if (it == launchStorage.end()) {
      continue;
    }
    LaunchStorage &storage = it->second;
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    Value one = createI64Constant(b, loc, 1);
    if (storage.capacity == 0) {
      storage.params = LLVM::ZeroOp::create(b, loc, ptrType);
    } else {
      auto type = KernelParamsBuilder::getStorageType(b, storage.capacity);
      uint64_t alignment = dataLayout.getTypePreferredAlignment(type);
      storage.params =
          LLVM::AllocaOp::create(b, loc, ptrType, type, one, alignment);
    }
    storage.config = LLVM::AllocaOp::create(b, loc, ptrType, launchConfigV1Type,
                                            one, launchConfigV1Alignment);
  }

  for (auto launch : launches) {
    auto func = launch->getParentOfType<func::FuncOp>();
    b.setInsertionPoint(launch);
    Location loc = launch.getLoc();
    Value runtime = func.getArgument(0);
    const LaunchStorage &storage = launchStorage.find(func)->second;

    // Check the original values before any high bits can be discarded.
    for (Value dim : {launch.getGridSizeX(), launch.getGridSizeY(),
                      launch.getGridSizeZ(), launch.getBlockSizeX(),
                      launch.getBlockSizeY(), launch.getBlockSizeZ()}) {
      checkRange(b, loc, dim, 0, UINT32_MAX, -5);
    }
    if (Value smem = launch.getDynamicSharedMemorySize()) {
      checkRange(b, loc, smem, 0, UINT32_MAX, -5);
    }

    auto launchConfig = LaunchConfigV1Builder::poison(b, loc);
    launchConfig.setGrid(b, loc, launch.getGridSizeX(), launch.getGridSizeY(),
                         launch.getGridSizeZ());
    launchConfig.setBlock(b, loc, launch.getBlockSizeX(),
                          launch.getBlockSizeY(), launch.getBlockSizeZ());
    Value smemBytes = launch.getDynamicSharedMemorySize();
    if (!smemBytes) {
      smemBytes = createI32Constant(b, loc, 0);
    }
    launchConfig.setDynamicSharedMemoryBytes(b, loc, smemBytes);

    Value launchConfigPtr = storage.config;
    LLVM::StoreOp::create(b, loc, Value(launchConfig), launchConfigPtr);

    // Rebuild the kernel parameter table from this launch's operands.
    Value params = storage.params;
    KernelParamsBuilder kernelParams(b, loc, converter, params,
                                     storage.capacity);
    for (Value operand : launch.getKernelOperands()) {
      kernelParams.append(operand);
    }

    Value runtimeValue =
        LLVM::LoadOp::create(b, loc, RuntimeV1Builder::getType(b), runtime);
    auto callbackStruct = RuntimeV1Builder(runtimeValue);
    Value callback = callbackStruct.getLaunchCallbackFn(b, loc);
    Value userData = callbackStruct.getUserData(b, loc);

    Value ordinal =
        createI32Constant(b, loc,
                          launch
                              ->getAttrOfType<IntegerAttr>(
                                  TensorIRDialect::getKernelOrdinalAttrName())
                              .getInt());
    auto callbackType = RuntimeV1Builder::getLaunchCallbackFnType(b);
    Value status = LLVM::CallOp::create(b, loc, callbackType,
                                        ValueRange{callback, userData, ordinal,
                                                   launchConfigPtr, params})
                       .getResult();

    Value zero = createI32Constant(b, loc, 0);
    Value ok =
        LLVM::ICmpOp::create(b, loc, LLVM::ICmpPredicate::eq, status, zero);
    require(b, loc, ok, -3);
    launch.erase();
  }
  return success();
}

namespace {
/// Older ArithToLLVM implementations convert index result types but retain
/// index-typed value attributes, preventing CSE with integer constants.
/// NOTE: Remove when all supported LLVM versions incorporate
/// LLVM-Project change https://github.com/llvm/llvm-project/commit/325a853c1264.
struct NormalizeIndexConstant : OpConversionPattern<LLVM::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = dyn_cast<IntegerAttr>(op.getValue());
    auto type = dyn_cast<IntegerType>(op.getType());
    if (!value || !value.getType().isIndex() || !type) {
      return rewriter.notifyMatchFailure(
          op, "expected index attribute and integer result");
    }
    auto converted =
        IntegerAttr::get(type, value.getValue().sextOrTrunc(type.getWidth()));
    rewriter.modifyOpInPlace(op, [&] { op.setValueAttr(converted); });
    return success();
  }
};

/// Converts the host portion of an outlined module to LLVM dialect while
/// preserving embedded device `gpu.module` operations for subsequent device
/// compilation passes.
struct HostToLLVMPass : nv_tensor_ir::impl::HostToLLVMPassBase<HostToLLVMPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    DataLayoutAnalysis &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    const DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(module);
    HostLLVMTypeConverter converter(&getContext(), dataLayoutAnalysis);

    // Replace host launches with runtime callbacks before the dialect
    // conversion lowers the arith, cf, func, and memref ops they introduce.
    IRRewriter b(module);
    if (failed(lowerGpuLaunchToCallback(b, module, dataLayout, converter))) {
      return signalPassFailure();
    }

    // Populate dialect-to-LLVM conversion patterns for host operations.
    RewritePatternSet patterns(&getContext());
    arith::populateArithToLLVMConversionPatterns(converter, patterns);
    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    ptr::populatePtrToLLVMConversionPatterns(converter, patterns);
    ub::populateUBToLLVMConversionPatterns(converter, patterns);
    patterns.add<NormalizeIndexConstant>(&getContext());

    // Mark gpu.module recursively legal to leave device kernels unlowered.
    LLVMConversionTarget target(getContext());
    target.addDynamicallyLegalOp<LLVM::ConstantOp>([](LLVM::ConstantOp op) {
      auto value = dyn_cast<IntegerAttr>(op.getValue());
      return !value || !value.getType().isIndex() ||
             !isa<IntegerType>(op.getType());
    });
    target.addLegalOp<ModuleOp, gpu::GPUModuleOp, UnrealizedConversionCastOp>();
    target.markOpRecursivelyLegal<gpu::GPUModuleOp>();
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Collect and reconcile unrealized conversion casts in the host module.
    // Casts inside device modules are skipped. Any unreconciled cast indicates
    // an unresolved type boundary error in host lowering.
    SmallVector<UnrealizedConversionCastOp> casts;
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<gpu::GPUModuleOp>(op)) {
        return WalkResult::skip();
      }
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(op)) {
        casts.push_back(cast);
      }
      return WalkResult::advance();
    });
    SmallVector<UnrealizedConversionCastOp> remaining;
    reconcileUnrealizedCasts(casts, &remaining);
    if (!remaining.empty()) {
      remaining.front().emitError("unresolved cast in native host lowering");
      return signalPassFailure();
    }
  }
};
} // namespace
