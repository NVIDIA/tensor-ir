// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

#include "cuda_tile/Bytecode/Common/Version.h"
#include <string>

namespace mlir::nv_tensor_ir::test {

//===----------------------------------------------------------------------===//
// Bytecode Version Targets
//===----------------------------------------------------------------------===//

enum class BytecodeVersionTarget {
  Default,
  Current,
  Compatibility,
};

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

struct Options {
  std::string programName;

  //=========================================================================
  // Option Categories
  //=========================================================================

  llvm::cl::OptionCategory GeneralCat{
      "General Options", "Options for controlling overall program behavior"};
  llvm::cl::OptionCategory CompileCat{"Compilation Options",
                                      "Options for JIT compilation"};
  llvm::cl::OptionCategory TestCat{"Test Options",
                                   "Options for testing and verification"};

  //=========================================================================
  // General Options
  //=========================================================================

  llvm::cl::opt<bool> verbose{"verbose",
                              llvm::cl::desc("Enable verbose output"),
                              llvm::cl::cat(GeneralCat)};

  llvm::cl::opt<std::string> dumpArtifactPath{
      "dump-artifact",
      llvm::cl::desc("Write compiled CudaTile device code to this path and "
                     "metadata to <path>.meta"),
      llvm::cl::init(""), llvm::cl::cat(GeneralCat)};

  // Note: Don't define --help here; LLVM provides it automatically

  //=========================================================================
  // Compilation Options
  //=========================================================================

  llvm::cl::list<std::string> inputFiles{llvm::cl::Positional,
                                         llvm::cl::desc("<input mlir files>"),
                                         llvm::cl::cat(CompileCat)};

  llvm::cl::opt<std::string> targetSM{
      "target-sm",
      llvm::cl::desc("Target SM version (e.g., sm_100, sm_100a, or sm_100f; "
                     "default: sm_100f)"),
      llvm::cl::init("sm_100f"), llvm::cl::cat(CompileCat)};

  llvm::cl::opt<std::string> codegenStrategy{
      "codegen-strategy",
      llvm::cl::desc("TensorIR-to-CudaTile lowering strategy: affine-map or "
                     "layout-propagation (default: layout-propagation)"),
      llvm::cl::init("layout-propagation"), llvm::cl::cat(CompileCat)};

  llvm::cl::opt<BytecodeVersionTarget> bytecodeVersion{
      "bytecode-version",
      llvm::cl::desc("TileIR bytecode target version (default: default)"),
      llvm::cl::init(BytecodeVersionTarget::Default),
      llvm::cl::values(clEnumValN(BytecodeVersionTarget::Default, "default",
                                  "max(CUDA Tile compatibility version, 13.3)"),
                       clEnumValN(BytecodeVersionTarget::Current, "current",
                                  "CUDA Tile current bytecode version"),
                       clEnumValN(BytecodeVersionTarget::Compatibility,
                                  "compatibility",
                                  "CUDA Tile compatibility bytecode version")),
      llvm::cl::cat(CompileCat)};

  llvm::cl::opt<backend::cuda_tile::CudaTileArtifactKind> artifactKind{
      "artifact-kind",
      llvm::cl::desc("Requested device artifact (default: tileir)"),
      llvm::cl::init(backend::cuda_tile::CudaTileArtifactKind::TileIR),
      llvm::cl::values(
          clEnumValN(backend::cuda_tile::CudaTileArtifactKind::TileIR, "tileir",
                     "Return TileIR bytecode for driver JIT"),
          clEnumValN(backend::cuda_tile::CudaTileArtifactKind::Cubin, "cubin",
                     "Request an eagerly assembled cubin")),
      llvm::cl::cat(CompileCat)};

  llvm::cl::opt<bool> timing{
      "timing",
      llvm::cl::desc("Enable MLIR pass timing (default: false)\n"
                     "env TENSOR_IR_TIMING overrides this flag; it accepts\n"
                     "1/0, true/false, on/off, or yes/no"),
      llvm::cl::cat(CompileCat)};

  llvm::cl::opt<bool> printIrAfterAll{
      "print-ir-after-all",
      llvm::cl::desc(
          "Print IR after each pass (default: false); ignored when\n"
          "--print-ir-tree-dir is set\n"
          "env TENSOR_IR_PRINT_IR_AFTER_ALL overrides this flag; it\n"
          "accepts 1/0, true/false, on/off, or yes/no"),
      llvm::cl::cat(CompileCat)};

  llvm::cl::opt<std::string> printIrTreeDir{
      "print-ir-tree-dir",
      llvm::cl::desc("Directory for dumping IR tree after passes; takes\n"
                     "precedence over --print-ir-after-all\n"
                     "env TENSOR_IR_PRINT_IR_TREE_DIR overrides this flag; it\n"
                     "requires a path, and 0/false/off/no disables the dump"),
      llvm::cl::init(""), llvm::cl::cat(CompileCat)};

  //=========================================================================
  // Test Options
  //=========================================================================

  llvm::cl::opt<bool> launch{
      "launch", llvm::cl::desc("Launch compiled runtime kernel on GPU"),
      llvm::cl::cat(TestCat)};

  llvm::cl::opt<bool> verify{
      "verify", llvm::cl::desc("Launch and verify results against reference"),
      llvm::cl::cat(TestCat)};

  llvm::cl::opt<unsigned> seed{"seed",
                               llvm::cl::desc("Random seed for test data"),
                               llvm::cl::init(2024), llvm::cl::cat(TestCat)};

  llvm::cl::opt<int> iterations{"iterations",
                                llvm::cl::desc("Number of launch iterations"),
                                llvm::cl::init(1), llvm::cl::cat(TestCat)};

  llvm::cl::opt<float> tolerance{"tolerance",
                                 llvm::cl::desc("Tolerance for verification"),
                                 llvm::cl::init(1e-2f), llvm::cl::cat(TestCat)};

  llvm::cl::opt<std::string> dynamicDimsStr{
      "dynamic-dims",
      llvm::cl::desc("Runtime values for dynamic '?' dimensions, "
                     "comma-separated by position (e.g., 16 or 16,32)"),
      llvm::cl::init(""), llvm::cl::cat(TestCat)};

  llvm::cl::opt<std::string> dynamicStridesStr{
      "dynamic-strides",
      llvm::cl::desc("Runtime values for dynamic '?' strides, "
                     "comma-separated by position (e.g., 8 or 8,1)"),
      llvm::cl::init(""), llvm::cl::cat(TestCat)};

  //=========================================================================
  // Compilation Options (continued)
  //=========================================================================

  llvm::cl::opt<std::string> tileSizeStr{
      "tile-size",
      llvm::cl::desc("Tile sizes for kernel tiling, 'x'-separated "
                     "(e.g., 8x8 or 32x32x16)"),
      llvm::cl::init(""), llvm::cl::cat(CompileCat)};

  llvm::cl::opt<std::string> dumpCudaTileIR{
      "dump-ir",
      llvm::cl::desc("Dump lowered CudaTile dialect MLIR to this file\n"
                     "env TENSOR_IR_DUMP_IR overrides this flag; it requires\n"
                     "a path, and 0/false/off/no disables the dump"),
      llvm::cl::init(""), llvm::cl::cat(CompileCat)};

  llvm::cl::opt<std::string> dumpTileIRBC{
      "dump-tileir-bc",
      llvm::cl::desc("Dump Tile IR bytecode to this file after compilation\n"
                     "env TENSOR_IR_DUMP_TILEIR_BC overrides this flag; it\n"
                     "requires a path, and 0/false/off/no disables the dump"),
      llvm::cl::init(""), llvm::cl::cat(CompileCat)};

  llvm::cl::opt<std::string> loadTileIRBC{
      "load-tileir-bc",
      llvm::cl::desc("Load Tile IR bytecode from file instead of the compiled\n"
                     "result (MLIR passes still run for arg layout)\n"
                     "env TENSOR_IR_LOAD_TILEIR_BC overrides this flag; it\n"
                     "requires a path, and 0/false/off/no disables the load"),
      llvm::cl::init(""), llvm::cl::cat(CompileCat)};

  llvm::cl::opt<bool> uniformSignature{
      "uniform-signature",
      llvm::cl::desc("Always pass all sizes/strides as kernel args, even for\n"
                     "static dims (true=uniform, false=original behavior)\n"
                     "(default: false)"),
      llvm::cl::init(false), llvm::cl::cat(CompileCat)};

  //=========================================================================
  // Parsed values
  //=========================================================================

  /// Runtime values for dynamic dimensions, indexed by dim position.
  /// E.g., parsedDynamicDims[0] = 16 means dim-0 '?' resolves to 16.
  llvm::SmallVector<int64_t> parsedDynamicDims;

  /// Runtime values for dynamic strides, indexed by dim position.
  /// E.g., parsedDynamicStrides[0] = 8 means stride-0 '?' resolves to 8.
  /// If empty, dynamic strides are deduced from dims (contiguous layout).
  llvm::SmallVector<int64_t> parsedDynamicStrides;

  /// Tile sizes for kernel tiling, e.g., {8, 8}.
  llvm::SmallVector<int32_t> parsedTileSize;

  //=========================================================================
  // Methods
  //=========================================================================

  Options(int argc, char **argv);

  void print() const;

  /// Parse --target-sm using the canonical sm_<cc>[a|f] spelling.
  FailureOr<SmTarget> getTargetSM() const;

  /// Resolve the semantic --bytecode-version option to a CUDA Tile bytecode
  /// version. `default` resolves to getDefaultBytecodeVersion().
  mlir::cuda_tile::BytecodeVersion getBytecodeVersion() const;
};

} // namespace mlir::nv_tensor_ir::test
