// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"

#include "tensor_ir/Artifact/TensorIRArtifact.h"
#include "tensor_ir/Compiler/CudaTile/CudaTileFrontend.h"
#include "tensor_ir/Compiler/CudaTile/TileIRAssembly.h"
#include "tensor_ir/Support/Status.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include <optional>

static std::string createUniqueReproducerDir(const std::string &dir) {
  auto stripTrailingSlash = [](llvm::SmallString<128> &path) {
    while (path.size() > 1 && llvm::sys::path::is_separator(path.back())) {
      path.pop_back();
    }
  };

  if (dir.empty()) {
    return {};
  }

  llvm::SmallString<128> reproducerPath(dir);
  if (auto ec = llvm::sys::fs::make_absolute(reproducerPath)) {
    llvm::errs() << "failed to read the absolute path for mlir reproducer: "
                 << ec.message() << "\n";
    return {};
  }

  stripTrailingSlash(reproducerPath);
  if (auto ec = llvm::sys::fs::create_directories(
          llvm::sys::path::parent_path(reproducerPath))) {
    llvm::errs() << "failed to create the parent directory for mlir "
                    "reproducer: "
                 << ec.message() << "\n";
    return {};
  }

  llvm::SmallString<128> uniqueReproducerDir;
  if (auto ec = llvm::sys::fs::createUniqueDirectory(reproducerPath,
                                                     uniqueReproducerDir)) {
    llvm::errs() << "failed to create a unique directory for mlir reproducer: "
                 << ec.message() << "\n";
    return {};
  }
  return std::string(uniqueReproducerDir);
}

namespace mlir::nv_tensor_ir::backend::cuda_tile {

bool CudaTileCompiler::canCompile(mlir::ModuleOp module,
                                  const CompilerInvocation &options) const {
  if (!options.validate().ok()) {
    return false;
  }
  return compiler::cuda_tile::isCudaTileFrontendSupported(module, options);
}

StatusOr<std::unique_ptr<::tensor_ir::rt::TensorIRArtifact>>
CudaTileCompiler::compile(mlir::ModuleOp module,
                          const CompilerInvocation &options) {
  TIR_RETURN_IF_ERROR(options.validate());

  CompilerInvocation effectiveInvocation = options;
  effectiveInvocation.debug.reproducerDir =
      createUniqueReproducerDir(options.debug.reproducerDir);

  const auto &debug = effectiveInvocation.debug;
  auto frontendOr =
      compiler::cuda_tile::lowerTensorIRToCudaTile(module, effectiveInvocation);
  if (failed(frontendOr)) {
    return Status::CompilationError(
        "Failed to lower TensorIR to CudaTile dialect");
  }
  compiler::cuda_tile::CudaTileFrontendResult frontend = std::move(*frontendOr);

  llvm::SmallVector<char, 0> tileirBytecode;
  TIR_ASSIGN_OR_RETURN(
      auto bytecodeVersion,
      resolveBytecodeVersion(options.cudaTile.bytecodeVersion));
  std::optional<mlir::cuda_tile::BytecodeVersion> runtimeBytecodeVersion =
      bytecodeVersion;
  llvm::raw_svector_ostream os(tileirBytecode);
  if (failed(mlir::cuda_tile::writeBytecode(os, frontend.cudaTileModule,
                                            *runtimeBytecodeVersion))) {
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
    // This file's bytecode version is unknown (it wasn't written by
    // writeBytecode() above); the assembler and the runtime fallback
    // assembler both still work without it -- they recover the version from
    // the bytecode's own contents -- but the version-aware driver
    // compatibility diagnostic on failure needs a known version, so that
    // diagnostic won't be available for this artifact.
    runtimeBytecodeVersion = std::nullopt;
  }

  // The binary starts out as the TileIR bytecode just produced above (with
  // whatever version -- possibly unknown, if the load-from-disk debug path
  // ran) -- unless artifactKind asks for a cubin, in which case it's
  // assembled into a real cubin below regardless of whether the version is
  // known.
  auto compiledTarget = options.getKernelSmTarget();
  if (failed(compiledTarget)) {
    return Status::InvalidArgument("Invalid target SM '" +
                                   options.deviceTarget.kernelTarget + "'");
  }
  ::tensor_ir::rt::Binary binary = ::tensor_ir::rt::TileIRBytecode{
      std::move(tileirBytecode), runtimeBytecodeVersion};
  if (options.cudaTile.artifactKind == ArtifactKind::Cubin) {
    const auto &tileirBinary =
        std::get<::tensor_ir::rt::TileIRBytecode>(binary);
    TIR_ASSIGN_OR_RETURN(
        auto cubin, assembleTileIRToCubin(tileirBinary.bytes, *compiledTarget,
                                          runtimeBytecodeVersion));
    if (cubin) {
      binary = ::tensor_ir::rt::Cubin{std::move(*cubin)};
    }
  }

  std::optional<std::array<int32_t, 3>> staticGrid;
  if (!frontend.useRuntimeGrid) {
    staticGrid = compiler::cuda_tile::computeStaticGridSize(frontend);
  }
  return ::tensor_ir::rt::TensorIRArtifact::create(
      frontend.runtimeKernelName, frontend.entryFunctionName, std::move(binary),
      *compiledTarget, std::move(frontend.argLayout), staticGrid);
}

} // namespace mlir::nv_tensor_ir::backend::cuda_tile
