// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/CudaTileFrontend.h"

#include "tensor_ir/Analysis/LayoutPropagation.h"
#include "tensor_ir/Compiler/CudaTile/KernelMetadataExtraction.h"
#include "tensor_ir/Compiler/CudaTile/Pipelines.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <limits>

namespace tcg = mlir::nv_tensor_ir::tcutegen;

static bool isSupportedTensorIrOp(mlir::Operation *op) {
  using namespace mlir::nv_tensor_ir;
  namespace arith = mlir::arith;

  return mlir::isa<
      GraphOp, ResultsOp,
      // Data movement.
      TransposeOp, ConcatenateOp, ScanSliceOp, SliceOp, ReshapeOp, BroadcastOp,
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

static bool
runStaticPreChecks(mlir::nv_tensor_ir::GraphOp graphOp,
                   const mlir::nv_tensor_ir::CompilerInvocation &options) {
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

  if (options.codegen.ctaCount < 1 || options.codegen.ctaCount > 8) {
    return false;
  }
  if (options.cudaTile.warpCount < 1) {
    return false;
  }
  if (options.cudaTile.maxTileCandidates < 0) {
    return false;
  }
  return true;
}

static bool usesRuntimeGrid(llvm::ArrayRef<::tensor_ir::rt::ArgDesc> args) {
  for (const auto &arg : args) {
    const auto *desc = std::get_if<::tensor_ir::rt::TensorArgDesc>(&arg);
    if (!desc) {
      continue;
    }
    for (int64_t dim : desc->shape()) {
      if (dim == ::tensor_ir::rt::TensorArgDesc::kDynamic) {
        return true;
      }
    }
  }
  return false;
}

static bool usesRuntimeGrid(const ::tensor_ir::rt::KernelMetadata &layout) {
  return usesRuntimeGrid(layout.args);
}

static bool
isStaticPersistenceActive(llvm::ArrayRef<int64_t> iterationShape,
                          llvm::ArrayRef<int32_t> tileSizes,
                          const mlir::nv_tensor_ir::CudaTileOptions &options) {
  using namespace mlir::nv_tensor_ir;

  if (options.persistenceMode != PersistenceMode::Static ||
      options.smCount <= 0 || options.occupancy <= 0 ||
      iterationShape.size() != tileSizes.size()) {
    return false;
  }
  if (llvm::any_of(iterationShape, [](int64_t dim) {
        return mlir::ShapedType::isDynamic(dim);
      })) {
    return true;
  }

  int64_t totalTiles = 1;
  constexpr int64_t maxI32 = std::numeric_limits<int32_t>::max();
  for (auto [dim, tile] : llvm::zip_equal(iterationShape, tileSizes)) {
    if (dim < 0 || tile <= 0) {
      return false;
    }
    if (dim == 0) {
      totalTiles = 0;
      break;
    }
    int64_t tilesInDim = llvm::divideCeil(dim, static_cast<int64_t>(tile));
    if (tilesInDim > maxI32 / totalTiles) {
      return false;
    }
    totalTiles *= tilesInDim;
  }

  int64_t persistentGridSize =
      static_cast<int64_t>(options.smCount) * options.occupancy;
  return persistentGridSize <= maxI32 && totalTiles > persistentGridSize;
}

static mlir::LogicalResult configureLayoutPropGridMetadata(
    mlir::nv_tensor_ir::GraphOp graphOp,
    ::tensor_ir::rt::KernelMetadataBuilder &argLayout,
    llvm::SmallVectorImpl<int64_t> &resolvedIterationSpaceShape,
    const mlir::nv_tensor_ir::CudaTileOptions &options) {
  using namespace mlir::nv_tensor_ir;
  using ::tensor_ir::rt::TensorArgDesc;

  auto tileSizeAttr = graphOp->getAttrOfType<mlir::DenseI32ArrayAttr>(
      TensorIRDialect::getTileSizeAttrName());
  if (!tileSizeAttr) {
    return graphOp.emitError(
        "layout-propagation lowering did not select a tile size");
  }
  llvm::ArrayRef<int32_t> tileSizes = tileSizeAttr.asArrayRef();
  argLayout.tileSizes.assign(tileSizes.begin(), tileSizes.end());

  mlir::FailureOr<LayoutNormalization> normalization =
      LayoutNormalization::compute(graphOp);
  if (mlir::failed(normalization)) {
    return mlir::failure();
  }
  llvm::SmallVector<int64_t> iterationShape =
      normalization->iterationSpace.getShape();
  if (iterationShape.size() != tileSizes.size()) {
    return graphOp.emitError(
        "iteration-space rank does not match the selected tile rank");
  }
  resolvedIterationSpaceShape.assign(iterationShape.begin(),
                                     iterationShape.end());

  if (isStaticPersistenceActive(iterationShape, tileSizes, options)) {
    argLayout.persistence = PersistenceMode::Static;
    argLayout.smCount = options.smCount;
    argLayout.occupancy = options.occupancy;
  }

  if (!usesRuntimeGrid(argLayout.args)) {
    return mlir::success();
  }

  auto resultDescriptors = getTensorDescriptors(graphOp.getResultTypes(),
                                                graphOp.getAllResultAttrs());
  if (mlir::failed(resultDescriptors)) {
    return graphOp.emitError(
        "failed to read output descriptors for runtime grid computation");
  }

  int32_t tensorDescIdx = argLayout.numInputs;
  mlir::RankedTensorType outputType;
  const TensorDescriptor *outputDesc = nullptr;
  for (auto [idx, type] : llvm::enumerate(graphOp.getResultTypes())) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensorType) {
      if (type.isIntOrFloat()) {
        ++tensorDescIdx;
      }
      continue;
    }
    if (tensorType.getRank() == 0) {
      continue;
    }
    outputType = tensorType;
    outputDesc = &(*resultDescriptors)[idx];
    break;
  }
  if (!outputType || !outputDesc ||
      tensorDescIdx >= static_cast<int32_t>(argLayout.args.size())) {
    return graphOp.emitError(
        "runtime grid computation requires a ranked output tensor");
  }

  tcg::Layout outputLayout(getShapeRef(outputType));
  if (!outputDesc->strides.empty()) {
    tcg::Stride outputStrides;
    for (const auto &stride : outputDesc->strides) {
      if (stride.staticValue == mlir::ShapedType::kDynamic) {
        outputStrides.appendDynamic();
      } else {
        outputStrides.append(stride.staticValue);
      }
    }
    outputLayout = tcg::Layout(getShapeRef(outputType), outputStrides);
  }

  auto outputSource = TensorSourceAttr::get(
      graphOp.getContext(), /*tensorId=*/-1, /*offset=*/0,
      outputLayout.toString(), getDynamicValueMapping(outputLayout));
  auto normalizedOutput = mlir::dyn_cast_if_present<TensorSourceAttr>(
      outputSource.reshape(iterationShape));
  if (!normalizedOutput) {
    return graphOp.emitError(
        "failed to map the output shape to the runtime iteration space");
  }

  const auto *shapeDesc =
      std::get_if<TensorArgDesc>(&argLayout.args[tensorDescIdx]);
  if (!shapeDesc) {
    return graphOp.emitError("grid shape requires a tensor argument");
  }
  llvm::SmallVector<int32_t> dynamicShapeDims;
  for (auto [dim, size] : llvm::enumerate(shapeDesc->shape())) {
    if (size == TensorArgDesc::kDynamic) {
      dynamicShapeDims.push_back(static_cast<int32_t>(dim));
    }
  }

  argLayout.gridShape = iterationShape;
  argLayout.gridShapeDimMapping.assign(argLayout.gridShape.size(), -1);
  auto dynamicMapping = normalizedOutput.getDynamicValueMapping();
  size_t dynamicShapeOrdinal = 0;
  for (auto [dim, size] : llvm::enumerate(argLayout.gridShape)) {
    if (size != TensorArgDesc::kDynamic) {
      if (size < 0) {
        return graphOp.emitError(
            "runtime iteration-space dimensions must be non-negative");
      }
      continue;
    }

    int32_t sourceDynamicOrdinal =
        dynamicMapping.empty() ? static_cast<int32_t>(dynamicShapeOrdinal)
        : dynamicShapeOrdinal < dynamicMapping.size()
            ? dynamicMapping[dynamicShapeOrdinal]
            : -1;
    if (sourceDynamicOrdinal < 0 ||
        static_cast<size_t>(sourceDynamicOrdinal) >= dynamicShapeDims.size()) {
      return graphOp.emitError()
             << "failed to map dynamic iteration-space dimension " << dim
             << " to the output tensor shape: mapping index "
             << sourceDynamicOrdinal << ", output dynamic shape count "
             << dynamicShapeDims.size();
    }
    argLayout.gridShapeDimMapping[dim] = dynamicShapeDims[sourceDynamicOrdinal];
    ++dynamicShapeOrdinal;
  }

  argLayout.gridShapeTensorIdx = tensorDescIdx;
  return mlir::success();
}

static mlir::FailureOr<std::string>
getCudaTileEntryFunctionName(mlir::ModuleOp module,
                             mlir::cuda_tile::ModuleOp &cudaTileModule) {
  auto deviceModules = module.getOps<mlir::cuda_tile::ModuleOp>();
  if (!llvm::hasSingleElement(deviceModules)) {
    return mlir::failure();
  }
  cudaTileModule = *deviceModules.begin();

  auto entryOps = cudaTileModule.getOps<mlir::cuda_tile::EntryOp>();
  if (!llvm::hasSingleElement(entryOps)) {
    return mlir::failure();
  }
  return (*entryOps.begin()).getSymName().str();
}

static void configureDebugInstrumentation(
    mlir::ModuleOp module, mlir::PassManager &pm,
    const ::mlir::nv_tensor_ir::CompilerDebugOptions &debug,
    llvm::StringRef pipelineName) {
  if (debug.printIRAfterAll || !debug.printIRTreeDir.empty()) {
    module->getContext()->disableMultithreading();
  }

  if (debug.printIRAfterAll && !debug.printIRTreeDir.empty()) {
    llvm::errs()
        << "Warning: both --print-ir-after-all and --print-ir-tree-dir are "
           "set; using --print-ir-tree-dir and skipping stderr IR dumps.\n";
  }

  auto ensureDirectoryExists = [](llvm::StringRef dir) {
    if (!llvm::sys::fs::exists(dir)) {
      std::error_code ec = llvm::sys::fs::create_directory(dir);
      if (ec) {
        llvm::errs() << "Warning: cannot create IR tree directory '" << dir
                     << "': " << ec.message() << "\n";
      }
    }
  };

  if (!debug.reproducerDir.empty()) {
    ensureDirectoryExists(debug.reproducerDir);
    std::string reproducerName(pipelineName);
    std::transform(pipelineName.begin(), pipelineName.end(),
                   reproducerName.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::replace(reproducerName.begin(), reproducerName.end(), ' ', '-');
    llvm::SmallString<128> reproducerFile;
    llvm::sys::path::append(reproducerFile, debug.reproducerDir,
                            reproducerName + ".mlir");

    mlir::makeReproducer(pm.getOpAnchorName(), pm.getPasses(),
                         module.getOperation(), reproducerFile, true);
  }

  if (!debug.printIRTreeDir.empty()) {
    ensureDirectoryExists(debug.printIRTreeDir);
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
static mlir::LogicalResult
runPipeline(mlir::ModuleOp module,
            const mlir::nv_tensor_ir::CompilerInvocation &options,
            mlir::TimingScope *timingParent, llvm::StringRef timingName,
            PopulatePipeline &&populatePipeline) {
  mlir::TimingScope timing =
      timingParent ? timingParent->nest(timingName) : mlir::TimingScope();
  mlir::PassManager pm(module->getContext());
  auto status = populatePipeline(pm);
  if (!status.ok()) {
    return module.emitError(status.message());
  }
  configureDebugInstrumentation(module, pm, options.debug, timingName);
  if (timingParent) {
    pm.enableTiming(timing);
  } else if (options.debug.enableTiming) {
    pm.enableTiming();
  }
  return pm.run(module);
}

static mlir::FailureOr<mlir::nv_tensor_ir::GraphOp>
getSupportedGraph(mlir::ModuleOp module,
                  const mlir::nv_tensor_ir::CompilerInvocation &options) {
  llvm::SmallVector<mlir::nv_tensor_ir::GraphOp> graphOps;
  module.walk([&](mlir::nv_tensor_ir::GraphOp op) { graphOps.push_back(op); });
  if (graphOps.size() != 1 || !runStaticPreChecks(graphOps.front(), options)) {
    return mlir::failure();
  }
  return graphOps.front();
}

static mlir::LogicalResult runLayoutPropagationLowering(
    mlir::ModuleOp module,
    const mlir::nv_tensor_ir::CompilerInvocation &options,
    mlir::TimingScope *timing,
    const std::function<void(mlir::ModuleOp)> &onTensorIRReady) {
  using namespace mlir::nv_tensor_ir;

  if (failed(runPipeline(module, options, timing, "Analyze TensorIR graph",
                         [&](mlir::PassManager &pm) {
                           return buildGraphAnalysisPipeline(pm, options);
                         }))) {
    return mlir::failure();
  }
  if (failed(runPipeline(module, options, timing, "Select tile",
                         [&](mlir::PassManager &pm) {
                           buildTileSelectionPipeline(pm, options);
                           return mlir::nv_tensor_ir::Status::Ok();
                         }))) {
    return mlir::failure();
  }

  if (onTensorIRReady) {
    onTensorIRReady(module);
  }
  if (failed(runPipeline(module, options, timing, "Form TensorIR tiled program",
                         [&](mlir::PassManager &pm) {
                           buildTensorIRTiledProgramFormationPipeline(pm,
                                                                      options);
                           return mlir::nv_tensor_ir::Status::Ok();
                         }))) {
    return mlir::failure();
  }

  return runPipeline(module, options, timing, "Lower TensorIR to CudaTile",
                     [&](mlir::PassManager &pm) {
                       return buildTensorToCudaTileConversionOnlyPipeline(
                           pm, options);
                     });
}

static mlir::LogicalResult runAffineMapLowering(
    mlir::ModuleOp module,
    const mlir::nv_tensor_ir::CompilerInvocation &options,
    mlir::TimingScope *timing,
    const std::function<void(mlir::ModuleOp)> &onTensorIRReady) {
  if (onTensorIRReady) {
    onTensorIRReady(module);
  }
  return runPipeline(
      module, options, timing, "Lower TensorIR to CudaTile",
      [&](mlir::PassManager &pm) {
        return mlir::nv_tensor_ir::buildTensorToCudaTileConversionPipeline(
            pm, options);
      });
}

static mlir::LogicalResult runTensorIRLowering(
    mlir::ModuleOp module,
    const mlir::nv_tensor_ir::CompilerInvocation &options,
    mlir::TimingScope *timing,
    const std::function<void(mlir::ModuleOp)> &onTensorIRReady) {
  if (options.cudaTile.codegenStrategy ==
      mlir::nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation) {
    return runLayoutPropagationLowering(module, options, timing,
                                        onTensorIRReady);
  }
  return runAffineMapLowering(module, options, timing, onTensorIRReady);
}

static std::string getRuntimeKernelName(mlir::ModuleOp module) {
  auto moduleName = module.getSymNameAttr();
  return moduleName ? moduleName.getValue().str() : "tensor_ir_rtk";
}

static void
resolveFrontendMetadata(mlir::ModuleOp module,
                        ::tensor_ir::rt::KernelMetadataBuilder &argLayout) {
  using namespace mlir::nv_tensor_ir;

  if (argLayout.tileSizes.empty()) {
    if (auto tileSizeAttr = module->getAttrOfType<mlir::DenseI32ArrayAttr>(
            kResolvedTileSizeAttrName)) {
      llvm::ArrayRef<int32_t> tileSizes = tileSizeAttr.asArrayRef();
      argLayout.tileSizes.assign(tileSizes.begin(), tileSizes.end());
    }
  }

  module->removeAttr(kResolvedTileSizeAttrName);
}

static llvm::ArrayRef<int64_t> getStaticGridShape(
    const mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendResult
        &result) {
  if (!result.resolvedIterationSpaceShape.empty()) {
    return result.resolvedIterationSpaceShape;
  }

  int32_t shapeTensorIdx = result.argLayout.gridShapeTensorIdx;
  if (shapeTensorIdx < 0 ||
      shapeTensorIdx >= static_cast<int32_t>(result.argLayout.args.size())) {
    return {};
  }
  const auto *desc = std::get_if<::tensor_ir::rt::TensorArgDesc>(
      &result.argLayout.args[shapeTensorIdx]);
  return desc ? desc->shape() : llvm::ArrayRef<int64_t>();
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

static void
dumpCudaTileIR(mlir::ModuleOp module,
               const ::mlir::nv_tensor_ir::CompilerDebugOptions &debug) {
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
                                 const CompilerInvocation &options) {
  return succeeded(getSupportedGraph(module, options));
}

FailureOr<CudaTileFrontendResult>
lowerTensorIRToCudaTile(ModuleOp module, const CompilerInvocation &options,
                        TimingScope *timing,
                        std::function<void(ModuleOp)> onTensorIRReady) {
  FailureOr<GraphOp> graph = getSupportedGraph(module, options);
  if (failed(graph)) {
    return failure();
  }

  FailureOr<::tensor_ir::rt::KernelMetadata> argLayout =
      extractKernelMetadata(*graph, options.codegen, options.cudaTile);
  if (failed(argLayout)) {
    return failure();
  }

  ::tensor_ir::rt::KernelMetadataBuilder layoutBuilder = argLayout->toBuilder();
  OwningOpRef<ModuleOp> frontendModule(module.clone());
  SmallVector<int64_t> resolvedIterationSpaceShape;

  LogicalResult gridMetadataStatus = success();
  if (options.cudaTile.codegenStrategy ==
      CudaTileCodegenStrategy::LayoutPropagation) {
    std::function<void(ModuleOp)> loweringHook = [&](ModuleOp analyzedModule) {
      auto graphOps = analyzedModule.getOps<GraphOp>();
      if (!llvm::hasSingleElement(graphOps)) {
        analyzedModule.emitError(
            "grid metadata requires exactly one TensorIR graph");
        gridMetadataStatus = failure();
      } else {
        gridMetadataStatus = configureLayoutPropGridMetadata(
            *graphOps.begin(), layoutBuilder, resolvedIterationSpaceShape,
            options.cudaTile);
      }
      if (onTensorIRReady) {
        onTensorIRReady(analyzedModule);
      }
    };
    if (failed(runTensorIRLowering(*frontendModule, options, timing,
                                   loweringHook))) {
      return failure();
    }
  } else if (failed(runTensorIRLowering(*frontendModule, options, timing,
                                        onTensorIRReady))) {
    return failure();
  }

  if (failed(gridMetadataStatus)) {
    return failure();
  }

  resolveFrontendMetadata(*frontendModule, layoutBuilder);
  auto validatedLayout =
      ::tensor_ir::rt::KernelMetadata::create(std::move(layoutBuilder));
  if (!validatedLayout.ok()) {
    frontendModule->emitError() << validatedLayout.status().message();
    return failure();
  }
  CudaTileFrontendResult result(std::move(*validatedLayout));
  result.runtimeKernelName = getRuntimeKernelName(module);
  result.module = std::move(frontendModule);
  result.resolvedIterationSpaceShape = std::move(resolvedIterationSpaceShape);
  dumpCudaTileIR(*result.module, options.debug);
  result.useRuntimeGrid = usesRuntimeGrid(result.argLayout);
  if (!result.useRuntimeGrid && failed(validateStaticGridMetadata(result))) {
    return failure();
  }

  FailureOr<std::string> entryFunctionName =
      getCudaTileEntryFunctionName(*result.module, result.cudaTileModule);
  if (failed(entryFunctionName)) {
    return failure();
  }

  result.entryFunctionName = *entryFunctionName;

  FailureOr<SmTarget> target = options.getKernelSmTarget();
  if (failed(target)) {
    return failure();
  }
  if (target->getComputeCapabilityVersion() >= 90) {
    result.cudaTileModule.walk([&](::mlir::cuda_tile::GdcWaitTkoOp) {
      result.usesPDL = true;
      return WalkResult::interrupt();
    });
  }
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
  constexpr int64_t maxI32 = std::numeric_limits<int32_t>::max();
  for (size_t i = 0; i < numDims; ++i) {
    int64_t dimSize = gridShape[i];
    int32_t tileSize = result.argLayout.tileSizes[i];
    if (tileSize <= 0 || dimSize < 0) {
      return gridSize;
    }
    if (dimSize == 0) {
      totalTiles = 0;
      break;
    }
    int64_t tilesInDim =
        llvm::divideCeil(dimSize, static_cast<int64_t>(tileSize));
    if (tilesInDim > maxI32 / totalTiles) {
      totalTiles = maxI32;
      break;
    }
    totalTiles *= tilesInDim;
  }
  if (result.argLayout.persistence == PersistenceMode::Static &&
      result.argLayout.smCount > 0 && result.argLayout.occupancy > 0) {
    int64_t persistentGridSize =
        static_cast<int64_t>(result.argLayout.smCount) *
        result.argLayout.occupancy;
    totalTiles = std::min(totalTiles, persistentGridSize);
  }
  gridSize[0] = static_cast<int32_t>(std::min(totalTiles, maxI32));
  return gridSize;
}

} // namespace mlir::nv_tensor_ir::compiler::cuda_tile
