// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"

#include "tensor_ir/Compiler/CudaTile/CudaTileFrontend.h"
#include "tensor_ir/Compiler/CudaTile/TileIRAssembly.h"
#include "tensor_ir/Runtime/CudaTile/CudaTileRuntimeKernel.h"
#include "tensor_ir/Runtime/CudaTile/KernelLaunchHelpers.h"
#include "tensor_ir/Runtime/CudaTile/RuntimeOperandAccessor.h"
#include "tensor_ir/Runtime/IRuntimeKernel.h"
#include "tensor_ir/Support/Status.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include <cstdlib>
#include <memory>
#include <optional>

namespace {

static bool isEnvEnabledValue(llvm::StringRef value) {
  return value.equals_insensitive("1") || value.equals_insensitive("true") ||
         value.equals_insensitive("on") || value.equals_insensitive("yes");
}

static bool isEnvDisabledValue(llvm::StringRef value) {
  return value.equals_insensitive("0") || value.equals_insensitive("false") ||
         value.equals_insensitive("off") || value.equals_insensitive("no");
}

static void applyBoolEnvVar(const char *name, bool &option) {
  const char *value = std::getenv(name);
  if (!value || !*value) {
    return;
  }

  llvm::StringRef valueRef(value);
  if (isEnvEnabledValue(valueRef)) {
    option = true;
    return;
  }
  if (isEnvDisabledValue(valueRef)) {
    option = false;
    return;
  }

  llvm::errs() << "Warning: ignoring invalid boolean value '" << valueRef
               << "' for " << name
               << "; expected 0/1, false/true, off/on, or no/yes.\n";
}

static void applyPathEnvVar(const char *name, std::string &path) {
  const char *value = std::getenv(name);
  if (!value || !*value) {
    return;
  }

  llvm::StringRef valueRef(value);
  if (isEnvDisabledValue(valueRef)) {
    path.clear();
    return;
  }
  if (isEnvEnabledValue(valueRef)) {
    path.clear();
    llvm::errs() << "Warning: ignoring boolean value '" << valueRef << "' for "
                 << name << "; expected a file or directory path.\n";
    return;
  }

  path = value;
}

static mlir::nv_tensor_ir::CudaTileCodegenStrategy getCodegenStrategy(
    mlir::nv_tensor_ir::CudaTileCodegenStrategy defaultStrategy) {
  const char *strategy = std::getenv("TENSOR_IR_CODEGEN_STRATEGY");
  if (!strategy) {
    return defaultStrategy;
  }
  llvm::StringRef value(strategy);
  if (value == "layout-propagation" || value == "layout_propagation" ||
      value == "layout") {
    return mlir::nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation;
  }
  if (value == "affine-map" || value == "affine_map" || value == "affine") {
    return mlir::nv_tensor_ir::CudaTileCodegenStrategy::AffineMap;
  }
  return defaultStrategy;
}

static int getMaxTileCandidatesFromEnvOrDefault(int defaultValue) {
  const char *maxCandidates = std::getenv("TENSOR_IR_MAX_TILE_CANDIDATES");
  int parsed = defaultValue;
  if (maxCandidates &&
      !llvm::StringRef(maxCandidates).getAsInteger(10, parsed) && parsed > 0) {
    return parsed;
  }
  return defaultValue;
}

static mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions makePipelineOptions(
    const mlir::nv_tensor_ir::backend::cuda_tile::CudaTileCompileOptions
        &cudaTileOptions) {
  mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions pipelineOptions;
  pipelineOptions.computeCapability =
      cudaTileOptions.computeCapability.getComputeCapabilityVersion();
  pipelineOptions.numCTAs = cudaTileOptions.ctaCount;
  pipelineOptions.numWarps = cudaTileOptions.warpCount;
  pipelineOptions.tileSize.assign(cudaTileOptions.tileSize.begin(),
                                  cudaTileOptions.tileSize.end());
  pipelineOptions.uniformSignature = cudaTileOptions.uniformSignature;
  pipelineOptions.codegenStrategy =
      getCodegenStrategy(cudaTileOptions.codegenStrategy);
  pipelineOptions.maxCandidates =
      getMaxTileCandidatesFromEnvOrDefault(cudaTileOptions.maxTileCandidates);
  return pipelineOptions;
}

static mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendOptions
makeFrontendOptions(
    const mlir::nv_tensor_ir::backend::cuda_tile::CudaTileCompileOptions
        &cudaTileOptions,
    const mlir::nv_tensor_ir::backend::cuda_tile::IRDebugOptions &debug) {
  mlir::nv_tensor_ir::compiler::cuda_tile::CudaTileFrontendOptions options;
  options.pipelineOptions = makePipelineOptions(cudaTileOptions);
  options.debug.dumpCudaTileIRPath = debug.dumpCudaTileIRPath;
  options.debug.printIRAfterAll = debug.printIRAfterAll;
  options.debug.printIRTreeDir = debug.printIRTreeDir;
  options.debug.enableTiming = debug.enableTiming;
  applyBoolEnvVar("TENSOR_IR_PRINT_IR", options.debug.printCudaTileIR);
  return options;
}

} // namespace

namespace mlir::nv_tensor_ir::backend::cuda_tile {

bool CudaTileCompiler::canCompile(mlir::ModuleOp module,
                                  const CompileOptions &options) const {
  const auto *cudaTileOptions =
      llvm::dyn_cast<CudaTileCompileOptions>(&options);
  if (!cudaTileOptions || !options.validate().ok()) {
    return false;
  }

  IRDebugOptions debug;
  auto frontendOptions = makeFrontendOptions(*cudaTileOptions, debug);
  return compiler::cuda_tile::isCudaTileFrontendSupported(module,
                                                          frontendOptions);
}

StatusOr<::tensor_ir::rt::IRuntimeKernelPtr>
CudaTileCompiler::compile(mlir::ModuleOp module,
                          const CompileOptions &options) {
  const auto *cudaTileOptions =
      llvm::dyn_cast<CudaTileCompileOptions>(&options);
  if (!cudaTileOptions) {
    return Status::InvalidArgument("Invalid CudaTile compilation options");
  }

  IRDebugOptions debug = cudaTileOptions->irDebug;
  applyPathEnvVar("TENSOR_IR_DUMP_IR", debug.dumpCudaTileIRPath);
  applyPathEnvVar("TENSOR_IR_DUMP_TILEIR_BC", debug.dumpTileIRBCPath);
  applyPathEnvVar("TENSOR_IR_LOAD_TILEIR_BC", debug.loadTileIRBCPath);
  applyBoolEnvVar("TENSOR_IR_PRINT_IR_AFTER_ALL", debug.printIRAfterAll);
  applyPathEnvVar("TENSOR_IR_PRINT_IR_TREE_DIR", debug.printIRTreeDir);
  applyBoolEnvVar("TENSOR_IR_TIMING", debug.enableTiming);

  auto frontendOptions = makeFrontendOptions(*cudaTileOptions, debug);
  auto frontendOr =
      compiler::cuda_tile::lowerTensorIRToCudaTile(module, frontendOptions);
  if (failed(frontendOr)) {
    return Status::CompilationError(
        "Failed to lower TensorIR to CudaTile dialect");
  }
  compiler::cuda_tile::CudaTileFrontendResult frontend = std::move(*frontendOr);

  llvm::SmallVector<char, 0> tileirBytecode;
  std::optional<mlir::cuda_tile::BytecodeVersion> runtimeBytecodeVersion =
      cudaTileOptions->bytecodeVersion;
  llvm::raw_svector_ostream os(tileirBytecode);
  if (failed(mlir::cuda_tile::writeBytecode(
          os, frontend.cudaTileModule, cudaTileOptions->bytecodeVersion))) {
    return Status::CompilationError("Failed to write tile-ir bytecode");
  }

  if (!debug.dumpTileIRBCPath.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream bcOs(debug.dumpTileIRBCPath, ec,
                              llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Warning: cannot dump Tile IR bytecode to '"
                   << debug.dumpTileIRBCPath << "': " << ec.message() << "\n";
    } else {
      bcOs.write(tileirBytecode.data(), tileirBytecode.size());
    }
  }

  if (!debug.loadTileIRBCPath.empty()) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(debug.loadTileIRBCPath,
                                                /*IsText=*/false);
    if (!bufOrErr) {
      return Status::CompilationError("Cannot load Tile IR bytecode from '" +
                                      debug.loadTileIRBCPath +
                                      "': " + bufOrErr.getError().message());
    }
    llvm::StringRef bc = (*bufOrErr)->getBuffer();
    tileirBytecode.assign(bc.begin(), bc.end());
    runtimeBytecodeVersion = std::nullopt;
  }

  if (runtimeBytecodeVersion &&
      cudaTileOptions->artifactKind == CudaTileArtifactKind::Cubin) {
    TIR_ASSIGN_OR_RETURN(
        auto cubin, assembleTileIRToCubin(tileirBytecode,
                                          cudaTileOptions->computeCapability,
                                          *runtimeBytecodeVersion));
    if (cubin) {
      tileirBytecode = std::move(*cubin);
      runtimeBytecodeVersion = std::nullopt;
    }
  }

  ::tensor_ir::rt::IRuntimeKernelPtr rtk =
      std::make_unique<::tensor_ir::rt::CudaTileRuntimeKernel>(
          frontend.runtimeKernelName, frontend.entryFunctionName,
          std::move(tileirBytecode), runtimeBytecodeVersion);

  using RuntimeAcc = ::tensor_ir::rt::RuntimeOperandAccessor;
  auto *tileRtk =
      static_cast<::tensor_ir::rt::CudaTileRuntimeKernel *>(rtk.get());
  if (runtimeBytecodeVersion) {
    SmTarget compiledTarget = cudaTileOptions->computeCapability;
    mlir::cuda_tile::BytecodeVersion bytecodeVersion = *runtimeBytecodeVersion;
    tileRtk->setTileIRFallbackAssembler(
        [compiledTarget, bytecodeVersion](
            llvm::ArrayRef<char> bytecode,
            int deviceCc) -> StatusOr<llvm::SmallVector<char, 0>> {
          if (!compiledTarget.validateCodegenTargetCompatibility(deviceCc)) {
            return Status::NotSupported("TileIR target is incompatible with "
                                        "the current CUDA device");
          }
          FailureOr<SmTarget> deviceTarget =
              SmTarget::fromCc(deviceCc, ArchPortability::arch_conditional);
          if (failed(deviceTarget)) {
            return Status::NotSupported(
                "unrecognized CUDA device compute capability");
          }
          TIR_ASSIGN_OR_RETURN(
              auto cubin,
              assembleTileIRToCubin(bytecode, *deviceTarget, bytecodeVersion));
          if (!cubin) {
            return Status::NotSupported("no compatible TileIR assembler");
          }
          return std::move(*cubin);
        });
  }
  const auto &argLayout = frontend.argLayout;
  if (argLayout.uniformSignature || argLayout.hasDynamicShapes()) {
    tileRtk->setArgPacker(
        std::make_unique<::tensor_ir::rt::FlatArgPacker<RuntimeAcc>>(
            argLayout));
  } else {
    tileRtk->setArgPacker(
        std::make_unique<::tensor_ir::rt::PointerOnlyArgPacker<RuntimeAcc>>());
  }

  if (frontend.useRuntimeGrid) {
    tileRtk->setGridComputer(
        std::make_unique<::tensor_ir::rt::TileBasedGridComputer<RuntimeAcc>>(
            argLayout));
  } else {
    tileRtk->setGridComputer(
        std::make_unique<::tensor_ir::rt::StaticGridComputer<RuntimeAcc>>(
            compiler::cuda_tile::computeStaticGridSize(frontend)));
  }

  return rtk;
}

Status CudaTileCompileOptions::validateDerived() const {
  if (ctaCount <= 0) {
    return Status::InvalidArgument("CTA count must be greater than 0");
  }
  if (warpCount <= 0) {
    return Status::InvalidArgument("Warp count must be greater than 0");
  }
  switch (artifactKind) {
  case CudaTileArtifactKind::TileIR:
    break;
  case CudaTileArtifactKind::Cubin:
    if (computeCapability.getPortability() !=
        ArchPortability::arch_conditional) {
      return Status::InvalidArgument(
          "Cubin requires an arch-conditional target");
    }
    break;
  default:
    return Status::InvalidArgument("Unrecognized CUDA Tile artifact kind");
  }
  return Status::Ok();
}

std::string CudaTileCompileOptions::toStringDerived() const {
  std::string s;
  s += "\nCTA count: " + std::to_string(ctaCount);
  s += "\nWarp count: " + std::to_string(warpCount);
  if (!tileSize.empty()) {
    s += "\nTile size: ";
    for (size_t i = 0; i < tileSize.size(); ++i) {
      if (i > 0) {
        s += "x";
      }
      s += std::to_string(tileSize[i]);
    }
  }
  s += "\nTile IR bytecode version: " + bytecodeVersion.toString();
  s += "\nCUDA Tile artifact kind: ";
  s += artifactKind == CudaTileArtifactKind::TileIR ? "tileir" : "cubin";
  return s;
}

std::string CudaTileCompileOptions::toUniqueStringDerived() const {
  std::string s = llvm::join_items("_", std::to_string(ctaCount),
                                   std::to_string(warpCount));
  if (!tileSize.empty()) {
    s += "_tile";
    for (size_t i = 0; i < tileSize.size(); ++i) {
      s += "_" + std::to_string(tileSize[i]);
    }
  }
  s += "_bc";
  s += llvm::join_items(
      "_", std::to_string(static_cast<unsigned>(bytecodeVersion.getMajor())),
      std::to_string(static_cast<unsigned>(bytecodeVersion.getMinor())),
      std::to_string(bytecodeVersion.getTag()));
  s += artifactKind == CudaTileArtifactKind::TileIR ? "_tileir" : "_cubin";
  return s;
}

} // namespace mlir::nv_tensor_ir::backend::cuda_tile
