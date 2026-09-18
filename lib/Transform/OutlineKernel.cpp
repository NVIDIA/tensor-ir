// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- OutlineKernel.cpp - Outline tiled TensorIR kernels ------------------===//
//
// The tiled-program formation pipeline uses a mapped scf.forall as its
// inspectable launch model. This pass separates that model into host and device
// IR. The retained host wrapper replaces the forall with gpu.launch_func. The
// destination-passing device entry point is nested in a gpu.module, reads its
// launch coordinates through standard GPU operations, and contains the
// original per-tile computation.
//
//===----------------------------------------------------------------------===//

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <string>

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DEF_OUTLINEKERNELPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral dispatchSuffix = "_dispatch";

/// Return a collision-free dispatch-wrapper name derived from `kernelName`.
static std::string getDispatchName(SymbolTable &symbolTable,
                                   StringRef kernelName) {
  std::string base = (kernelName + dispatchSuffix).str();
  if (!symbolTable.lookup(base)) {
    return base;
  }

  for (unsigned suffix = 0;; ++suffix) {
    std::string candidate = base + "_" + std::to_string(suffix);
    if (!symbolTable.lookup(candidate)) {
      return candidate;
    }
  }
}

/// Clone the pure setup needed to materialize `value` in the kernel.
/// Wrapper arguments and launch-coordinate values must already be present in
/// `mapping`. Region-bearing operations are intentionally rejected: formation
/// is responsible for placing semantic control flow inside the forall body.
static LogicalResult mapCapture(Value value, func::FuncOp wrapper,
                                OpBuilder &builder, IRMapping &mapping,
                                SmallVectorImpl<Operation *> &setupOps) {
  if (mapping.contains(value)) {
    return success();
  }

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    return wrapper.emitError()
           << "cannot outline unexpected capture " << argument
           << "; only entry-block arguments may cross the kernel boundary";
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp || definingOp->getParentOfType<func::FuncOp>() != wrapper) {
    return wrapper.emitError("cannot outline a capture defined outside the "
                             "dispatch wrapper");
  }
  if (!isPure(definingOp) || definingOp->getNumRegions() != 0) {
    return definingOp->emitError()
           << "cannot outline capture because '"
           << definingOp->getName().getStringRef()
           << "' is not a pure, regionless setup operation";
  }

  for (Value operand : definingOp->getOperands()) {
    if (failed(mapCapture(operand, wrapper, builder, mapping, setupOps))) {
      return failure();
    }
  }

  Operation *clone = builder.clone(*definingOp, mapping);
  mapping.map(definingOp, clone);
  setupOps.push_back(definingOp);
  return success();
}

static LogicalResult verifyForall(func::FuncOp wrapper,
                                  scf::ForallOp forallOp) {
  if (!wrapper.getResultTypes().empty()) {
    return wrapper.emitError(
        "expected a void destination-passing dispatch wrapper");
  }
  if (!wrapper.getBody().hasOneBlock()) {
    return wrapper.emitError("expected a single-block dispatch wrapper");
  }
  if (forallOp->getParentOp() != wrapper) {
    return forallOp.emitError(
        "expected the mapped forall directly in the dispatch wrapper");
  }
  if (forallOp.getRank() != 1) {
    return forallOp.emitError("expected a rank-one flattened grid");
  }
  if (!forallOp.getOutputs().empty() || !forallOp.getResults().empty() ||
      forallOp.getTerminator().getYieldingOps().begin() !=
          forallOp.getTerminator().getYieldingOps().end()) {
    return forallOp.emitError(
        "expected destination passing through memref arguments, not "
        "scf.forall shared outputs");
  }

  ArrayAttr mapping = forallOp.getMappingAttr();
  auto blockMapping = mapping && mapping.size() == 1
                          ? dyn_cast<gpu::GPUBlockMappingAttr>(mapping[0])
                          : gpu::GPUBlockMappingAttr();
  if (!blockMapping || blockMapping.getBlock() != gpu::MappingId::DimX) {
    return forallOp.emitError(
        "expected mapping = [#gpu.block<x>] on the flattened grid");
  }

  if (!forallOp.getDynamicLowerBound().empty() ||
      forallOp.getStaticLowerBound().front() != 0 ||
      !forallOp.getDynamicStep().empty() ||
      forallOp.getStaticStep().front() != 1) {
    return forallOp.emitError(
        "expected a normalized forall with lower bound 0 and step 1");
  }
  return success();
}

static LogicalResult outlineForall(func::FuncOp wrapper, scf::ForallOp forallOp,
                                   SymbolTable &hostSymbolTable,
                                   gpu::GPUModuleOp gpuModule,
                                   SymbolTable &deviceSymbolTable) {
  if (failed(verifyForall(wrapper, forallOp))) {
    return failure();
  }

  MLIRContext *context = wrapper.getContext();
  Location loc = forallOp.getLoc();
  std::string oldName = wrapper.getSymName().str();
  std::string kernelName = oldName;

  if (StringRef(oldName).ends_with(dispatchSuffix)) {
    kernelName = StringRef(oldName).drop_back(dispatchSuffix.size()).str();
    if (kernelName.empty()) {
      return wrapper.emitError("cannot derive a kernel symbol from dispatch "
                               "wrapper name '_dispatch'");
    }
  } else {
    std::string dispatchName = getDispatchName(hostSymbolTable, oldName);
    if (failed(hostSymbolTable.rename(wrapper, dispatchName))) {
      return wrapper.emitError()
             << "failed to rename dispatch wrapper to @" << dispatchName;
    }
  }
  if (Operation *collision = deviceSymbolTable.lookup(kernelName)) {
    return wrapper.emitError() << "cannot reserve kernel symbol @" << kernelName
                               << "; it is already used by '"
                               << collision->getName().getStringRef() << "'";
  }

  FunctionType kernelType =
      FunctionType::get(context, wrapper.getArgumentTypes(), TypeRange{});
  func::FuncOp kernel = func::FuncOp::create(loc, kernelName, kernelType);
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  UnitAttr::get(context));
  deviceSymbolTable.insert(kernel);

  Block *kernelBody = kernel.addEntryBlock();
  OpBuilder kernelBuilder = OpBuilder::atBlockBegin(kernelBody);
  Value blockId = gpu::BlockIdOp::create(kernelBuilder, loc, gpu::Dimension::x);
  Value gridDim = gpu::GridDimOp::create(kernelBuilder, loc, gpu::Dimension::x);

  IRMapping mapping;
  OpFoldResult upperBound = forallOp.getMixedUpperBound().front();
  if (Value dynamicUpperBound = upperBound.dyn_cast<Value>()) {
    mapping.map(dynamicUpperBound, gridDim);
  }
  for (auto [wrapperArg, kernelArg] :
       llvm::zip(wrapper.getArguments(), kernel.getArguments())) {
    if (!mapping.contains(wrapperArg)) {
      mapping.map(wrapperArg, kernelArg);
    }
  }
  mapping.map(forallOp.getInductionVar(0), blockId);

  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(forallOp.getRegion(), forallOp.getRegion(),
                            captures);
  SmallVector<Operation *> setupOps;
  for (Value capture : captures) {
    if (failed(
            mapCapture(capture, wrapper, kernelBuilder, mapping, setupOps))) {
      kernel.erase();
      return failure();
    }
  }

  for (Operation &op : forallOp.getBody()->without_terminator()) {
    kernelBuilder.clone(op, mapping);
  }
  func::ReturnOp::create(kernelBuilder, loc);

  OpBuilder wrapperBuilder(forallOp);
  Value gridSize;
  if (!forallOp.getDynamicUpperBound().empty()) {
    gridSize = forallOp.getDynamicUpperBound().front();
  } else {
    gridSize = arith::ConstantIndexOp::create(
        wrapperBuilder, loc, forallOp.getStaticUpperBound().front());
  }
  Value one = arith::ConstantIndexOp::create(wrapperBuilder, loc, 1);
  gpu::KernelDim3 grid{gridSize, one, one};
  // CudaTile selects its worker-warp configuration later through entry-point
  // optimization hints. The unity thread dimensions only make this staging
  // launch structurally complete.
  gpu::KernelDim3 block{one, one, one};
  SymbolRefAttr callee = SymbolRefAttr::get(
      context, gpuModule.getSymName(),
      {FlatSymbolRefAttr::get(context, kernel.getSymName())});
  gpu::LaunchFuncOp::create(wrapperBuilder, loc, callee, grid, block,
                            /*dynamicSharedMemorySize=*/Value{},
                            wrapper.getArguments());
  forallOp.erase();

  // Setup cloned into the kernel should not remain as dead wrapper IR.
  for (Operation *setupOp : llvm::reverse(setupOps)) {
    if (setupOp->use_empty()) {
      setupOp->erase();
    }
  }
  return success();
}

struct OutlineKernelPass
    : public impl::OutlineKernelPassBase<OutlineKernelPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable hostSymbolTable(module);
    struct OutliningCandidate {
      func::FuncOp wrapper;
      scf::ForallOp forallOp;
    };
    SmallVector<OutliningCandidate> candidates;

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal() ||
          func->hasAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
        continue;
      }

      SmallVector<scf::ForallOp> foralls;
      func.walk([&](scf::ForallOp forallOp) {
        if (forallOp.getMappingAttr()) {
          foralls.push_back(forallOp);
        }
      });
      if (foralls.empty()) {
        continue;
      }
      // Downstream conversion currently erases each dispatch wrapper and
      // cannot preserve a sequence of host launches. Keep the one-kernel
      // restriction until host functions can be JIT-compiled.
      if (foralls.size() != 1) {
        func.emitError("expected exactly one mapped scf.forall to outline");
        signalPassFailure();
        return;
      }
      candidates.push_back({func, foralls.front()});
    }

    if (candidates.empty()) {
      return;
    }

    std::string gpuModuleName = "kernels";
    if (hostSymbolTable.lookup(gpuModuleName)) {
      for (unsigned suffix = 0;; ++suffix) {
        std::string candidate = "kernels_" + std::to_string(suffix);
        if (!hostSymbolTable.lookup(candidate)) {
          gpuModuleName = std::move(candidate);
          break;
        }
      }
    }
    OpBuilder moduleBuilder(module.getContext());
    moduleBuilder.setInsertionPointToEnd(module.getBody());
    gpu::GPUModuleOp gpuModule =
        gpu::GPUModuleOp::create(moduleBuilder, module.getLoc(), gpuModuleName);
    SymbolTable deviceSymbolTable(gpuModule);
    module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    UnitAttr::get(module.getContext()));

    for (auto [wrapper, forallOp] : candidates) {
      if (failed(outlineForall(wrapper, forallOp, hostSymbolTable, gpuModule,
                               deviceSymbolTable))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::nv_tensor_ir
