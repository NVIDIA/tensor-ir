// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test_options.h"

#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace mlir::nv_tensor_ir::test {
namespace {

const char *toString(BytecodeVersionTarget target) {
  switch (target) {
  case BytecodeVersionTarget::Default:
    return "default";
  case BytecodeVersionTarget::Current:
    return "current";
  case BytecodeVersionTarget::Compatibility:
    return "compatibility";
  }
  llvm_unreachable("unknown TensorIR bytecode version target");
}

const char *toString(backend::cuda_tile::CudaTileArtifactKind kind) {
  switch (kind) {
  case backend::cuda_tile::CudaTileArtifactKind::TileIR:
    return "tileir";
  case backend::cuda_tile::CudaTileArtifactKind::Cubin:
    return "cubin";
  }
  llvm_unreachable("unknown CUDA Tile artifact kind");
}

} // namespace

namespace {

template <typename T>
void parseSeparatedIntsOrExit(llvm::StringRef value, char separator,
                              llvm::SmallVectorImpl<T> &out,
                              llvm::StringRef optionName) {
  llvm::SmallVector<llvm::StringRef, 8> parts;
  value.split(parts, separator);
  for (const auto &part : parts) {
    T val;
    if (part.trim().getAsInteger(10, val)) {
      llvm::errs() << "Error: Invalid " << optionName << " value: " << part
                   << "\n";
      std::exit(1);
    }
    out.push_back(val);
  }
}

} // namespace

Options::Options(int argc, char **argv) {
  programName = argv[0];

  llvm::cl::HideUnrelatedOptions({&GeneralCat, &CompileCat, &TestCat});

  // Parse command line
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TensorIR Compiler and Test Tool\n");

  // Parse --dynamic-dims (comma-separated int64 values)
  if (!dynamicDimsStr.empty()) {
    parseSeparatedIntsOrExit(dynamicDimsStr, ',', parsedDynamicDims,
                             "dynamic-dims");
  }

  // Parse --dynamic-strides (comma-separated int64 values)
  if (!dynamicStridesStr.empty()) {
    parseSeparatedIntsOrExit(dynamicStridesStr, ',', parsedDynamicStrides,
                             "dynamic-strides");
  }

  // Parse --tile-size ('x'-separated int32 values)
  if (!tileSizeStr.empty()) {
    parseSeparatedIntsOrExit(tileSizeStr, 'x', parsedTileSize, "tile-size");
  }
}

FailureOr<SmTarget> Options::getTargetSM() const {
  return SmTarget::fromString(targetSM);
}

mlir::cuda_tile::BytecodeVersion Options::getBytecodeVersion() const {
  switch (bytecodeVersion.getValue()) {
  case BytecodeVersionTarget::Default:
    return backend::cuda_tile::getDefaultBytecodeVersion();
  case BytecodeVersionTarget::Current:
    return mlir::cuda_tile::BytecodeVersion::kCurrentVersion;
  case BytecodeVersionTarget::Compatibility:
    return mlir::cuda_tile::BytecodeVersion::kCurrentCompatibilityVersion;
  }
  llvm_unreachable("unknown TensorIR bytecode version target");
}

void Options::print() const {
  llvm::outs() << "=== Options ===\n";
  llvm::outs() << "  Input files: ";
  for (const auto &f : inputFiles) {
    llvm::outs() << f << " ";
  }
  llvm::outs() << "\n";
  llvm::outs() << "  Target SM: " << targetSM << "\n";
  llvm::outs() << "  Codegen strategy: " << codegenStrategy.getValue() << "\n";
  llvm::outs() << "  Bytecode version: " << toString(bytecodeVersion.getValue())
               << " (" << getBytecodeVersion().toString() << ")\n";
  llvm::outs() << "  Artifact kind: " << toString(artifactKind.getValue())
               << "\n";
  if (!dumpArtifactPath.empty()) {
    llvm::outs() << "  Dump artifact: " << dumpArtifactPath << "\n";
  }
  llvm::outs() << "  Launch: " << ((launch || verify) ? "yes" : "no") << "\n";
  llvm::outs() << "  Verify: " << (verify ? "yes" : "no") << "\n";
  llvm::outs() << "  Seed: " << seed << "\n";
  llvm::outs() << "  Tolerance: " << tolerance << "\n";
  if (!parsedDynamicDims.empty()) {
    llvm::outs() << "  Dynamic dims: ";
    for (size_t i = 0; i < parsedDynamicDims.size(); ++i) {
      if (i > 0) {
        llvm::outs() << ",";
      }
      llvm::outs() << parsedDynamicDims[i];
    }
    llvm::outs() << "\n";
  }
  if (!parsedDynamicStrides.empty()) {
    llvm::outs() << "  Dynamic strides: ";
    for (size_t i = 0; i < parsedDynamicStrides.size(); ++i) {
      if (i > 0) {
        llvm::outs() << ",";
      }
      llvm::outs() << parsedDynamicStrides[i];
    }
    llvm::outs() << "\n";
  }
  if (!parsedTileSize.empty()) {
    llvm::outs() << "  Tile size: ";
    for (size_t i = 0; i < parsedTileSize.size(); ++i) {
      if (i > 0) {
        llvm::outs() << "x";
      }
      llvm::outs() << parsedTileSize[i];
    }
    llvm::outs() << "\n";
  }
  if (timing) {
    llvm::outs() << "  Timing: yes\n";
  }
  if (printIrAfterAll) {
    llvm::outs() << "  Print IR after all passes: yes\n";
  }
  if (!printIrTreeDir.empty()) {
    llvm::outs() << "  Print IR tree dir: " << printIrTreeDir << "\n";
  }
  if (!dumpCudaTileIR.empty()) {
    llvm::outs() << "  Dump CudaTile IR: " << dumpCudaTileIR << "\n";
  }
  if (!dumpTileIRBC.empty()) {
    llvm::outs() << "  Dump TileIR BC: " << dumpTileIRBC << "\n";
  }
  if (!loadTileIRBC.empty()) {
    llvm::outs() << "  Load TileIR BC: " << loadTileIRBC << "\n";
  }
}

} // namespace mlir::nv_tensor_ir::test
