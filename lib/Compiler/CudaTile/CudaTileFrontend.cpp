// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/CudaTileFrontend.h"

#include "tensor_ir/Compiler/CudaTile/KernelArgLayout.h"
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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>

namespace tcg = mlir::nv_tensor_ir::tcutegen;

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

static bool isStaticPersistenceActive(
    llvm::ArrayRef<int64_t> iterationShape, llvm::ArrayRef<int32_t> tileSizes,
    const mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions &options) {
  using namespace mlir::nv_tensor_ir;

  if (options.persistence != PersistenceMode::Static || options.smCount <= 0 ||
      options.occupancy <= 0 || iterationShape.size() != tileSizes.size()) {
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
    ::tensor_ir::rt::KernelArgLayout &argLayout,
    const mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions &options) {
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

  mlir::Operation *terminator = graphOp.getBody()->getTerminator();
  auto iterationSpace = mlir::dyn_cast_if_present<LayoutSourceAttrInterface>(
      terminator->getAttr(TensorIRDialect::getIterationSpaceAttrName()));
  if (!iterationSpace) {
    return graphOp.emitError(
        "grid computation requires an iteration-space layout");
  }
  llvm::SmallVector<int64_t> iterationShape = iterationSpace.getShape();
  if (iterationShape.size() != tileSizes.size()) {
    return graphOp.emitError(
        "iteration-space rank does not match the selected tile rank");
  }

  if (isStaticPersistenceActive(iterationShape, tileSizes, options)) {
    argLayout.persistence = PersistenceMode::Static;
    argLayout.smCount = options.smCount;
    argLayout.occupancy = options.occupancy;
  }

  if (!usesRuntimeGrid(argLayout)) {
    return mlir::success();
  }

  auto resultDescriptors = getTensorDescriptors(graphOp.getResultTypes(),
                                                graphOp.getAllResultAttrs());
  if (mlir::failed(resultDescriptors)) {
    return graphOp.emitError(
        "failed to read output descriptors for runtime grid computation");
  }

  int32_t tensorDescIdx = argLayout.outputTensorStartIdx();
  TensorType outputType;
  const TensorDescriptor *outputDesc = nullptr;
  for (auto [idx, type] : llvm::enumerate(graphOp.getResultTypes())) {
    auto tensorType = mlir::dyn_cast<TensorType>(type);
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
      tensorDescIdx >= static_cast<int32_t>(argLayout.tensorDescs.size())) {
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

  const auto &shapeDesc = argLayout.tensorDescs[tensorDescIdx];
  llvm::SmallVector<int32_t> dynamicShapeDims;
  for (auto [dim, size] : llvm::enumerate(shapeDesc.staticShape)) {
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
        &debug,
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
static mlir::LogicalResult runPipeline(
    mlir::ModuleOp module,
    const mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendOptions
        &options,
    llvm::StringRef pipelineName, PopulatePipeline &&populatePipeline) {
  mlir::TimingScope timing =
      options.timing ? options.timing->nest(pipelineName) : mlir::TimingScope();
  mlir::PassManager pm(module->getContext());
  populatePipeline(pm);

  configureDebugInstrumentation(module, pm, options.debug, pipelineName);
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
  result.argLayout =
      ::tensor_ir::extractKernelArgLayout(*graph, options.pipelineOptions);
  result.module = OwningOpRef<ModuleOp>(module.clone());

  CudaTileFrontendOptions loweringOptions = options;
  LogicalResult gridMetadataStatus = success();
  if (options.pipelineOptions.codegenStrategy ==
      CudaTileCodegenStrategy::LayoutPropagation) {
    loweringOptions.onTensorIRReady = [&](ModuleOp analyzedModule) {
      auto graphOps = analyzedModule.getOps<GraphOp>();
      if (!llvm::hasSingleElement(graphOps)) {
        analyzedModule.emitError(
            "grid metadata requires exactly one TensorIR graph");
        gridMetadataStatus = failure();
      } else {
        gridMetadataStatus = configureLayoutPropGridMetadata(
            *graphOps.begin(), result.argLayout, options.pipelineOptions);
      }
      if (options.onTensorIRReady) {
        options.onTensorIRReady(analyzedModule);
      }
    };
  }

  if (failed(runTensorIRLowering(*result.module, loweringOptions)) ||
      failed(gridMetadataStatus)) {
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
