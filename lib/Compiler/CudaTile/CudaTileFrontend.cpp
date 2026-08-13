// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/CudaTileFrontend.h"

#include "tensor_ir/Compiler/CudaTile/KernelArgLayout.h"
#include "tensor_ir/Compiler/CudaTile/Pipelines.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

static bool isSupportedTensorIrOp(mlir::Operation *op) {
  using namespace mlir::nv_tensor_ir;
  namespace arith = mlir::arith;

  return mlir::isa<
      GraphOp, ResultsOp,
      // Data movement.
      TransposeOp, ConcatenateOp, SliceOp, ReshapeOp, BroadcastOp,
      // Type conversion.
      ConvertOp,
      // Unary pointwise.
      AbsOp, CeilOp, FloorOp, NegOp, SqrtOp, RsqrtOp, ExpOp, LogOp, SinOp,
      CosOp, TanOp, ErfOp,
      // Activations.
      TanhFwdOp, ReluFwdOp, SigmoidFwdOp, GeluFwdOp, GeluApproxTanhFwdOp,
      ReciprocalOp, SoftplusFwdOp, SwishFwdOp, EluFwdOp,
      // Binary pointwise.
      AddOp, SubOp, MulOp, DivOp, ModOp, RemOp, MaxOp, MinOp, PowOp, Atan2Op,
      AddSquareOp,
      // Constants and broadcast.
      ConstantOp, SplatOp, IotaOp,
      // Comparison and logical.
      CmpOp, BinarySelectOp, LogicalAndOp, LogicalOrOp, LogicalNotOp,
      // Reduction.
      ReduceOp, ReduceUDOp, YieldOp,
      // Matmul.
      MatmulOp,
      // Arith dialect ops used inside reductions.
      arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::RemFOp,
      arith::MaximumFOp, arith::MinimumFOp, arith::NegFOp, arith::AddIOp,
      arith::SubIOp, arith::MulIOp, arith::MaxSIOp, arith::MaxUIOp,
      arith::MinSIOp, arith::MinUIOp, arith::AndIOp, arith::OrIOp,
      arith::XOrIOp, arith::CmpFOp, arith::CmpIOp, arith::SelectOp,
      arith::ConstantOp>(op);
}

static bool runStaticPreChecks(
    mlir::nv_tensor_ir::GraphOp graphOp,
    const mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions &options) {
  bool supported = true;
  graphOp.walk([&](mlir::Operation *op) {
    if (!isSupportedTensorIrOp(op)) {
      supported = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!supported) {
    return false;
  }

  if (options.numCTAs < 1 || options.numCTAs > 8) {
    return false;
  }
  if (options.numWarps < 1) {
    return false;
  }
  if (options.maxCandidates < 0) {
    return false;
  }
  return true;
}

static bool usesRuntimeGrid(const ::tensor_ir::rt::KernelArgLayout &layout) {
  for (const auto &desc : layout.tensorDescs) {
    for (int64_t dim : desc.staticShape) {
      if (dim == ::tensor_ir::rt::TensorArgDesc::kDynamic) {
        return true;
      }
    }
  }
  return false;
}

static mlir::FailureOr<std::string>
getCudaTileEntryFunctionName(mlir::Operation *op,
                             mlir::cuda_tile::ModuleOp &cudaTileModule) {
  cudaTileModule = mlir::cuda_tile::extractCudaTileModuleOp(op);
  if (!cudaTileModule) {
    return mlir::failure();
  }

  auto entryOps = cudaTileModule.getOps<mlir::cuda_tile::EntryOp>();
  if (!llvm::hasSingleElement(entryOps)) {
    return mlir::failure();
  }
  return (*entryOps.begin()).getSymName().str();
}

static void configureDebugInstrumentation(
    mlir::ModuleOp module, mlir::PassManager &pm,
    const mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendDebugOptions
        &debug) {
  if (debug.printIRAfterAll || !debug.printIRTreeDir.empty()) {
    module->getContext()->disableMultithreading();
  }

  if (debug.printIRAfterAll && !debug.printIRTreeDir.empty()) {
    llvm::errs()
        << "Warning: both --print-ir-after-all and --print-ir-tree-dir are "
           "set; using --print-ir-tree-dir and skipping stderr IR dumps.\n";
  }

  if (!debug.printIRTreeDir.empty()) {
    if (!llvm::sys::fs::exists(debug.printIRTreeDir)) {
      std::error_code ec =
          llvm::sys::fs::create_directory(debug.printIRTreeDir);
      if (ec) {
        llvm::errs() << "Warning: cannot create IR tree directory '"
                     << debug.printIRTreeDir << "': " << ec.message() << "\n";
      }
    }
    pm.enableIRPrintingToFileTree(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true, /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false, debug.printIRTreeDir,
        mlir::OpPrintingFlags());
    return;
  }

  if (debug.printIRAfterAll) {
    pm.enableIRPrinting([](mlir::Pass *, mlir::Operation *) { return false; },
                        [](mlir::Pass *, mlir::Operation *) { return true; },
                        /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false,
                        /*printAfterOnlyOnFailure=*/false, llvm::errs(),
                        mlir::OpPrintingFlags());
  }
}

template <typename PopulatePipeline>
static mlir::LogicalResult runPipeline(
    mlir::ModuleOp module,
    const mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendOptions
        &options,
    llvm::StringRef timingName, PopulatePipeline &&populatePipeline) {
  mlir::TimingScope timing =
      options.timing ? options.timing->nest(timingName) : mlir::TimingScope();
  mlir::PassManager pm(module->getContext());
  populatePipeline(pm);
  configureDebugInstrumentation(module, pm, options.debug);
  if (options.timing) {
    pm.enableTiming(timing);
  } else if (options.debug.enableTiming) {
    pm.enableTiming();
  }
  return pm.run(module);
}

static mlir::FailureOr<mlir::nv_tensor_ir::GraphOp> getSupportedGraph(
    mlir::ModuleOp module,
    const mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendOptions
        &options) {
  llvm::SmallVector<mlir::nv_tensor_ir::GraphOp> graphOps;
  module.walk([&](mlir::nv_tensor_ir::GraphOp op) { graphOps.push_back(op); });
  if (graphOps.size() != 1 ||
      !runStaticPreChecks(graphOps.front(), options.pipelineOptions)) {
    return mlir::failure();
  }
  return graphOps.front();
}

static mlir::LogicalResult runLayoutPropagationLowering(
    mlir::ModuleOp module,
    const mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendOptions
        &options) {
  using namespace mlir::nv_tensor_ir;

  if (failed(runPipeline(module, options, "Analyze TensorIR graph",
                         [&](mlir::PassManager &pm) {
                           buildGraphAnalysisPipeline(pm,
                                                      options.pipelineOptions);
                         }))) {
    return mlir::failure();
  }
  if (failed(runPipeline(
          module, options, "Select tile", [&](mlir::PassManager &pm) {
            buildTileSelectionPipeline(pm, options.pipelineOptions);
          }))) {
    return mlir::failure();
  }

  if (options.onTensorIRReady) {
    options.onTensorIRReady(module);
  }
  return runPipeline(module, options, "Lower TensorIR to CudaTile",
                     [&](mlir::PassManager &pm) {
                       buildTensorToCudaTileConversionOnlyPipeline(
                           pm, options.pipelineOptions);
                     });
}

static mlir::LogicalResult runAffineMapLowering(
    mlir::ModuleOp module,
    const mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendOptions
        &options) {
  if (options.onTensorIRReady) {
    options.onTensorIRReady(module);
  }
  return runPipeline(
      module, options, "Lower TensorIR to CudaTile",
      [&](mlir::PassManager &pm) {
        mlir::nv_tensor_ir::buildTensorToCudaTileConversionPipeline(
            pm, options.pipelineOptions);
      });
}

static mlir::LogicalResult runTensorIRLowering(
    mlir::ModuleOp module,
    const mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendOptions
        &options) {
  if (options.pipelineOptions.codegenStrategy ==
      mlir::nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation) {
    return runLayoutPropagationLowering(module, options);
  }
  return runAffineMapLowering(module, options);
}

static std::string getRuntimeKernelName(mlir::ModuleOp module) {
  auto moduleName = module->getAttrOfType<mlir::StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  return moduleName ? moduleName.getValue().str() : "tensor_ir_rtk";
}

static void resolveFrontendMetadata(
    mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendResult &result) {
  using namespace mlir::nv_tensor_ir;

  if (result.argLayout.tileSizes.empty()) {
    if (auto tileSizeAttr = result.module->getOperation()
                                ->getAttrOfType<mlir::DenseI32ArrayAttr>(
                                    kResolvedTileSizeAttrName)) {
      llvm::ArrayRef<int32_t> tileSizes = tileSizeAttr.asArrayRef();
      result.argLayout.tileSizes.assign(tileSizes.begin(), tileSizes.end());
    }
  }

  if (auto shapeAttr =
          result.module->getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
              kResolvedIterationSpaceShapeAttrName)) {
    llvm::ArrayRef<int64_t> shape = shapeAttr.asArrayRef();
    result.resolvedIterationSpaceShape.assign(shape.begin(), shape.end());
  }

  result.module->getOperation()->removeAttr(kResolvedTileSizeAttrName);
  result.module->getOperation()->removeAttr(
      kResolvedIterationSpaceShapeAttrName);
}

static llvm::ArrayRef<int64_t> getStaticGridShape(
    const mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendResult
        &result) {
  if (!result.resolvedIterationSpaceShape.empty()) {
    return result.resolvedIterationSpaceShape;
  }

  int32_t shapeTensorIdx = result.argLayout.gridShapeTensorIdx;
  if (shapeTensorIdx < 0 ||
      shapeTensorIdx >=
          static_cast<int32_t>(result.argLayout.tensorDescs.size())) {
    return {};
  }
  return result.argLayout.tensorDescs[shapeTensorIdx].staticShape;
}

static mlir::LogicalResult validateStaticGridMetadata(
    mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendResult &result) {
  if (result.argLayout.tileSizes.empty()) {
    return result.module->emitError(
        "static-grid lowering did not produce tile-size metadata");
  }
  if (getStaticGridShape(result).empty()) {
    return result.module->emitError(
        "static-grid lowering did not produce iteration-space shape metadata");
  }
  return mlir::success();
}

static void dumpCudaTileIR(
    mlir::ModuleOp module,
    const mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendDebugOptions
        &debug) {
  if (debug.printCudaTileIR) {
    llvm::errs() << "// -----// Dumped CudaTile IR //----- //\n";
    module.print(llvm::errs());
    llvm::errs() << "\n";
  }

  if (debug.dumpCudaTileIRPath.empty()) {
    return;
  }

  std::error_code ec;
  llvm::raw_fd_ostream os(debug.dumpCudaTileIRPath, ec);
  if (ec) {
    llvm::errs() << "Warning: cannot dump CudaTile IR to '"
                 << debug.dumpCudaTileIRPath << "': " << ec.message() << "\n";
    return;
  }
  module.print(os);
}

namespace mlir::nv_tensor_ir::compiler::cuda_tile {

bool isCudaTileFrontendSupported(ModuleOp module,
                                 const CudaTileFrontendOptions &options) {
  return succeeded(getSupportedGraph(module, options));
}

FailureOr<CudaTileFrontendResult>
lowerTensorIRToCudaTile(ModuleOp module,
                        const CudaTileFrontendOptions &options) {
  FailureOr<GraphOp> graph = getSupportedGraph(module, options);
  if (failed(graph)) {
    return failure();
  }

  CudaTileFrontendResult result;
  result.runtimeKernelName = getRuntimeKernelName(module);
  result.argLayout = ::tensor_ir::extractKernelArgLayout(
      *graph, options.pipelineOptions.tileSize,
      options.pipelineOptions.uniformSignature);
  result.module = OwningOpRef<ModuleOp>(module.clone());
  if (failed(runTensorIRLowering(*result.module, options))) {
    return failure();
  }

  resolveFrontendMetadata(result);
  dumpCudaTileIR(*result.module, options.debug);
  result.useRuntimeGrid = usesRuntimeGrid(result.argLayout);
  if (!result.useRuntimeGrid && failed(validateStaticGridMetadata(result))) {
    return failure();
  }

  FailureOr<std::string> entryFunctionName = getCudaTileEntryFunctionName(
      result.module->getOperation(), result.cudaTileModule);
  if (failed(entryFunctionName)) {
    return failure();
  }

  result.entryFunctionName = *entryFunctionName;
  return result;
}

std::array<int32_t, 3>
computeStaticGridSize(const CudaTileFrontendResult &result) {
  std::array<int32_t, 3> gridSize = {1, 1, 1};
  ArrayRef<int64_t> gridShape = getStaticGridShape(result);

  if (gridShape.empty() || result.argLayout.tileSizes.empty()) {
    return gridSize;
  }

  size_t numDims =
      std::min(result.argLayout.tileSizes.size(), gridShape.size());
  int64_t totalTiles = 1;
  for (size_t i = 0; i < numDims; ++i) {
    int64_t dimSize = gridShape[i];
    int32_t tileSize = result.argLayout.tileSizes[i];
    if (tileSize > 0 && dimSize > 0) {
      totalTiles *= llvm::divideCeil(dimSize, tileSize);
    }
  }
  gridSize[0] = static_cast<int32_t>(totalTiles);
  return gridSize;
}

} // namespace mlir::nv_tensor_ir::compiler::cuda_tile
