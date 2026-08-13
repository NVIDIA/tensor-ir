// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/Compiler.h"
#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"
#include "tensor_ir/Reference/reference_graph.h"
#include "tensor_ir/Reference/tensor_memory.h"
#include "tensor_ir/Registration/Registration.h"
#include "tensor_ir/Runtime/CudaTile/CudaTileRuntimeKernel.h"
#include "tensor_ir/Support/CudaApi.h"
#include "tensor_ir/Support/Status.h"

#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "test_options.h"
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using mlir::nv_tensor_ir::Status;
using mlir::nv_tensor_ir::StatusOr;
using mlir::nv_tensor_ir::reference::DeviceRun;
using mlir::nv_tensor_ir::test::Options;
using namespace mlir::nv_tensor_ir;
using namespace tensor_ir::rt;

namespace {

using mlir::nv_tensor_ir::reference::ReferenceGraph;
using mlir::nv_tensor_ir::reference::SimplifiedTensor;

using OutputResults = std::vector<std::vector<std::byte>>;

using TileSize = llvm::SmallVector<int32_t>;

constexpr int kWarmupIterations = 5;

mlir::nv_tensor_ir::CudaTileCodegenStrategy
parseCodegenStrategy(llvm::StringRef strategy) {
  if (strategy == "layout-propagation" || strategy == "layout_propagation" ||
      strategy == "layout") {
    return mlir::nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation;
  }
  if (strategy != "affine-map" && strategy != "affine_map" &&
      strategy != "affine") {
    llvm::errs() << "Warning: unknown --codegen-strategy='" << strategy
                 << "', using layout-propagation\n";
    return mlir::nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation;
  }
  return mlir::nv_tensor_ir::CudaTileCodegenStrategy::AffineMap;
}

struct ParsedModule {
  ParsedModule(std::unique_ptr<mlir::MLIRContext> context,
               mlir::OwningOpRef<mlir::ModuleOp> module)
      : context(std::move(context)), module(std::move(module)) {}

  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

StatusOr<std::unique_ptr<ParsedModule>> parseModule(llvm::StringRef inputFile) {
  llvm::outs() << "Processing: " << inputFile << "\n";

  auto bufferOrErr = llvm::MemoryBuffer::getFile(inputFile);
  if (!bufferOrErr) {
    return Status::NotFound("Failed to read file: " + inputFile.str());
  }

  llvm::StringRef source = (*bufferOrErr)->getBuffer();
  llvm::outs() << "  Read " << source.size() << " bytes\n";

  mlir::DialectRegistry registry;
  mlir::nv_tensor_ir::registerDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  mlir::ParserConfig parserConfig(context.get());

  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, parserConfig);
  if (!module) {
    return Status::InvalidArgument("Failed to parse MLIR");
  }

  llvm::outs() << "  Parsed MLIR successfully\n";
  return std::make_unique<ParsedModule>(std::move(context), std::move(module));
}

backend::cuda_tile::CudaTileCompileOptions
createCompileOptions(const Options &options, const SmTarget &smTarget) {
  backend::cuda_tile::CudaTileCompileOptions compileOptions{smTarget};
  compileOptions.ctaCount = 1;
  compileOptions.warpCount = 4;
  compileOptions.tileSize.assign(options.parsedTileSize.begin(),
                                 options.parsedTileSize.end());
  compileOptions.uniformSignature = options.uniformSignature;
  compileOptions.codegenStrategy =
      parseCodegenStrategy(options.codegenStrategy.getValue());
  compileOptions.bytecodeVersion = options.getBytecodeVersion();
  compileOptions.artifactKind = options.artifactKind.getValue();

  // Environment variables can override these CLI defaults at compile time.
  compileOptions.irDebug.dumpCudaTileIRPath = options.dumpCudaTileIR.getValue();
  compileOptions.irDebug.dumpTileIRBCPath = options.dumpTileIRBC.getValue();
  compileOptions.irDebug.loadTileIRBCPath = options.loadTileIRBC.getValue();
  compileOptions.irDebug.printIRAfterAll = bool(options.printIrAfterAll);
  compileOptions.irDebug.printIRTreeDir = options.printIrTreeDir.getValue();
  compileOptions.irDebug.enableTiming = bool(options.timing);
  return compileOptions;
}

StatusOr<IRuntimeKernelPtr> compileModule(ICompiler &compiler,
                                          mlir::ModuleOp module,
                                          const Options &options,
                                          const TileSize &tileSize,
                                          const SmTarget &smTarget) {
  llvm::outs() << "  Compiling...\n";
  auto compileOptions = createCompileOptions(options, smTarget);
  compileOptions.tileSize = tileSize;
  TIR_RETURN_IF_ERROR(compileOptions.validate());
  TIR_ASSIGN_OR_RETURN(auto runtimeKernel,
                       compiler.compile(module, compileOptions));
  return runtimeKernel;
}

Status dumpArtifact(IRuntimeKernel &runtimeKernel, const Options &options) {
  if (options.dumpArtifactPath.empty()) {
    return Status::Ok();
  }

  auto *tileRuntimeKernel =
      static_cast<CudaTileRuntimeKernel *>(&runtimeKernel);
  if (!tileRuntimeKernel->hasDeviceCode()) {
    return Status::InvalidArgument(
        "Failed to dump artifact: runtime kernel has no CudaTile device code");
  }

  std::error_code error;
  llvm::raw_fd_ostream artifactOut(options.dumpArtifactPath, error,
                                   llvm::sys::fs::OF_None);
  if (error) {
    return Status::InvalidArgument("Failed to open artifact path " +
                                   options.dumpArtifactPath + ": " +
                                   error.message());
  }
  artifactOut.write(tileRuntimeKernel->deviceCode().data(),
                    tileRuntimeKernel->deviceCode().size());
  artifactOut.close();

  std::string metadataPath = options.dumpArtifactPath + ".meta";
  llvm::raw_fd_ostream metadataOut(metadataPath, error, llvm::sys::fs::OF_Text);
  if (error) {
    return Status::InvalidArgument("Failed to open artifact metadata path " +
                                   metadataPath + ": " + error.message());
  }
  metadataOut << "format="
              << (tileRuntimeKernel->hasTileIRBytecode() ? "cuda_tile_bytecode"
                                                         : "cubin")
              << "\n";
  metadataOut << "runtime_kernel_name=" << tileRuntimeKernel->name() << "\n";
  metadataOut << "kernel_name=" << tileRuntimeKernel->funcName() << "\n";
  metadataOut << "runtime_api=cuLibraryLoadData,cuLibraryGetKernel,"
                 "cuLaunchKernelEx\n";
  metadataOut.close();

  llvm::outs() << "  Wrote CudaTile artifact: " << options.dumpArtifactPath
               << "\n";
  llvm::outs() << "  Wrote CudaTile artifact metadata: " << metadataPath
               << "\n";
  return Status::Ok();
}

Status buildReferenceGraph(mlir::ModuleOp module, const Options &options,
                           ReferenceGraph &graph) {
  if (!options.parsedDynamicDims.empty()) {
    graph.setDynamicDims(options.parsedDynamicDims);
  }
  if (!options.parsedDynamicStrides.empty()) {
    graph.setDynamicStrides(options.parsedDynamicStrides);
  }

  Status buildStatus = graph.buildFromMLIR(module);
  if (!buildStatus.ok()) {
    std::string prefix = buildStatus.code() == StatusCode::kNotSupported
                             ? "Reference execution is unsupported: "
                             : "Failed to build reference graph: ";
    return Status(buildStatus.code(), prefix + buildStatus.message());
  }

  graph.print();
  llvm::outs() << " Reference graph built successfully\n";
  return Status::Ok();
}

template <typename PrintFn>
void printTensors(llvm::StringRef label,
                  llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> tensors,
                  PrintFn printFn) {
  llvm::outs() << label << "\n";
  for (const auto &tensor : tensors) {
    printFn(*tensor);
  }
}

Status initializeInputs(ReferenceGraph &graph, const Options &options) {
  TIR_RETURN_IF_ERROR(graph.fillInputTensorsRandom(-1.0, 1.0, options.seed));

  if (options.verbose) {
    llvm::outs() << "[Data check] After fillInputTensorsRandom min -1, max 1\n";
    printTensors("[Data check] Input host data:", graph.getInputTensors(),
                 [](SimplifiedTensor &tensor) { tensor.printHostData(); });
    printTensors("[Data check] Output host data:", graph.getOutputTensors(),
                 [](SimplifiedTensor &tensor) { tensor.printHostData(); });
    llvm::outs() << "--------------------------------\n\n\n";
  }
  return Status::Ok();
}

Status runOnHost(ReferenceGraph &graph, const Options &options) {
  TIR_RETURN_IF_ERROR(graph.execute());

  if (options.verbose) {
    llvm::outs() << "[Data check] After execute\n";
    for (const auto &outputTensor : graph.getOutputTensors()) {
      outputTensor->printHostData();
    }
    llvm::outs() << "--------------------------------\n\n\n";
  }
  return Status::Ok();
}

Status runWarmups(IRuntimeKernel &runtimeKernel, PackedArgs args) {
  for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
    TIR_RETURN_IF_ERROR(runtimeKernel.launch(args, Workspace(), Stream()));
  }
  return cuda::runtime::synchronizeDevice();
}

struct BenchmarkResults {
  float gpuTotalMs;
  double wallTotalUs;
};

StatusOr<BenchmarkResults> benchmark(IRuntimeKernel &runtimeKernel,
                                     PackedArgs args, int iterations) {
  TIR_ASSIGN_OR_RETURN(auto startEvent, cuda::runtime::createEvent());
  TIR_ASSIGN_OR_RETURN(auto stopEvent, cuda::runtime::createEvent());
  auto cleanup = llvm::scope_exit([&] {
    (void)cuda::runtime::destroyEvent(startEvent);
    (void)cuda::runtime::destroyEvent(stopEvent);
  });

  auto wallStart = std::chrono::high_resolution_clock::now();
  TIR_RETURN_IF_ERROR(cuda::runtime::recordEvent(startEvent));

  for (int iteration = 0; iteration < iterations; ++iteration) {
    TIR_RETURN_IF_ERROR(runtimeKernel.launch(args, Workspace(), Stream()));
  }

  TIR_RETURN_IF_ERROR(cuda::runtime::recordEvent(stopEvent));
  TIR_RETURN_IF_ERROR(cuda::runtime::synchronizeEvent(stopEvent));
  auto wallEnd = std::chrono::high_resolution_clock::now();

  float gpuTotalMs = 0;
  TIR_ASSIGN_OR_RETURN(gpuTotalMs,
                       cuda::runtime::getElapsedTime(startEvent, stopEvent));
  double wallTotalUs =
      std::chrono::duration<double, std::micro>(wallEnd - wallStart).count();
  return BenchmarkResults{gpuTotalMs, wallTotalUs};
}

void printDeviceTensors(const DeviceRun &deviceRun) {
  llvm::outs() << "[Data check] After cudaDeviceSynchronize\n";
  llvm::outs() << "[Data check] Input device data:\n";
  Status printStatus = deviceRun.printInputs();
  if (!printStatus.ok()) {
    llvm::errs() << "  Failed to print device inputs: " << printStatus.message()
                 << "\n";
  }
  llvm::outs() << "[Data check] Output device data:\n";
  printStatus = deviceRun.printOutputs();
  if (!printStatus.ok()) {
    llvm::errs() << "  Failed to print device outputs: "
                 << printStatus.message() << "\n";
  }
  llvm::outs() << "--------------------------------\n\n\n";
}

void printBenchmarkResults(const BenchmarkResults &results,
                           const Options &options) {
  double gpuAverageUs = (results.gpuTotalMs * 1000.0) / options.iterations;
  double wallAverageUs = results.wallTotalUs / options.iterations;
  double hostOverheadUs = wallAverageUs - gpuAverageUs;

  llvm::outs() << "  Benchmark (" << kWarmupIterations << " warmup + "
               << options.iterations << " timed iterations, uniformSignature="
               << (options.uniformSignature ? "true" : "false") << ")\n";
  llvm::outs() << "  GPU kernel time:   "
               << llvm::format("%.3f", results.gpuTotalMs) << " ms total, "
               << llvm::format("%.2f", gpuAverageUs) << " us/iter\n";
  llvm::outs() << "  Wall-clock time:   "
               << llvm::format("%.3f", results.wallTotalUs / 1000.0)
               << " ms total, " << llvm::format("%.2f", wallAverageUs)
               << " us/iter\n";
  llvm::outs() << "  Host overhead:     "
               << llvm::format("%.2f", hostOverheadUs)
               << " us/iter  (wall - GPU = arg packing + launch)\n";
}

StatusOr<OutputResults> runOnDevice(IRuntimeKernel &runtimeKernel,
                                    ReferenceGraph &graph,
                                    const Options &options) {
  TIR_ASSIGN_OR_RETURN(
      DeviceRun deviceRun,
      DeviceRun::create(graph.getInputTensors(), graph.getOutputTensors()));
  llvm::ArrayRef<Any> args = deviceRun.args();
  PackedArgs packedArgs(args.data(), args.size());

  TIR_RETURN_IF_ERROR(runtimeKernel.initializeRuntimeState());
  TIR_RETURN_IF_ERROR(runWarmups(runtimeKernel, packedArgs));
  TIR_ASSIGN_OR_RETURN(
      auto benchmarkResults,
      benchmark(runtimeKernel, packedArgs, options.iterations));

  OutputResults outputResults;
  if (options.verify) {
    TIR_RETURN_IF_ERROR(deviceRun.copyOutputsTo(outputResults));
  }
  if (options.verbose) {
    printDeviceTensors(deviceRun);
  }

  printBenchmarkResults(benchmarkResults, options);
  return outputResults;
}

Status compareResults(const ReferenceGraph &graph,
                      const OutputResults &outputResults, double tolerance) {
  llvm::outs() << "  Verifying results...\n";
  auto outputTensors = graph.getOutputTensors();
  if (outputResults.size() != outputTensors.size()) {
    return Status::ConstraintNotSatisfied("Verification result count mismatch");
  }
  for (size_t i = 0, e = outputTensors.size(); i != e; ++i) {
    if (!outputTensors[i]->checkRefAgainst(outputResults[i].data(), tolerance,
                                           tolerance, true)) {
      return Status::ConstraintNotSatisfied("Verification failed");
    }
  }

  llvm::outs() << "  ✓ Verification passed!\n";
  return Status::Ok();
}

Status processInputFile(llvm::StringRef inputFile, const Options &options,
                        const SmTarget &smTarget, ICompiler &compiler) {
  TIR_ASSIGN_OR_RETURN(auto parsedModule, parseModule(inputFile));
  TIR_ASSIGN_OR_RETURN(auto runtimeKernel,
                       compileModule(compiler, *parsedModule->module, options,
                                     options.parsedTileSize, smTarget));
  TIR_RETURN_IF_ERROR(dumpArtifact(*runtimeKernel, options));

  llvm::outs() << "  Compiled successfully\n";
  if (!options.launch && !options.verify) {
    llvm::outs() << "  Done: " << inputFile << "\n";
    return Status::Ok();
  }

  ReferenceGraph graph;
  TIR_RETURN_IF_ERROR(
      buildReferenceGraph(*parsedModule->module, options, graph));

  llvm::outs() << "  Initializing inputs...\n";
  TIR_RETURN_IF_ERROR(initializeInputs(graph, options));

  if (options.verify) {
    llvm::outs() << "  Running on host...\n";
    TIR_RETURN_IF_ERROR(runOnHost(graph, options));
  }

  llvm::outs() << "  Running on device...\n";
  TIR_ASSIGN_OR_RETURN(auto outputResults,
                       runOnDevice(*runtimeKernel, graph, options));

  if (options.verify) {
    TIR_RETURN_IF_ERROR(compareResults(graph, outputResults,
                                       static_cast<double>(options.tolerance)));
  }
  llvm::outs() << "  Done: " << inputFile << "\n";
  return Status::Ok();
}

} // namespace

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  Options options(argc, argv);
  auto smTarget = options.getTargetSM();
  if (failed(smTarget)) {
    llvm::errs() << "Error: invalid or unsupported --target-sm value '"
                 << options.targetSM
                 << "'; expected 'sm_<cc>', optionally suffixed with 'a' or "
                    "'f'\n";
    return 1;
  }

  if (options.verbose) {
    options.print();
  }
  if (options.inputFiles.empty()) {
    llvm::cl::PrintHelpMessage();
    return 0;
  }

  std::unique_ptr<ICompiler> compiler =
      ICompiler::create(CompilerBackend::CudaTile);
  if (!compiler) {
    llvm::errs() << "Failed to create CudaTile compiler\n";
    return 1;
  }
  if (options.iterations <= 0) {
    llvm::errs() << "Error: --iterations must be greater than 0\n";
    return 1;
  }

  int result = 0;
  for (const auto &inputFile : options.inputFiles) {
    Status status = processInputFile(inputFile, options, *smTarget, *compiler);
    if (!status.ok()) {
      llvm::errs() << "  " << status.message() << "\n";
      result = 1;
    }
  }
  return result;
}
