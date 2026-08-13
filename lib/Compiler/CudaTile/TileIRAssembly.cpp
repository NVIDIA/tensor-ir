// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/TileIRAssembly.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace mlir::nv_tensor_ir::backend::cuda_tile {
namespace {

struct TemporaryFile {
  explicit TemporaryFile(llvm::StringRef path)
      : path(path.str()),
        remover(std::make_unique<llvm::FileRemover>(this->path)) {}

  std::string path;
  std::unique_ptr<llvm::FileRemover> remover;
};

struct ProcessResult {
  int exitCode;
  std::string error;
};

StatusOr<TemporaryFile> createTemporaryFile(llvm::StringRef suffix) {
  llvm::SmallString<128> path;
  if (std::error_code error =
          llvm::sys::fs::createTemporaryFile("tensor-ir", suffix, path)) {
    return Status::CompilationError("failed to create temporary file: " +
                                    error.message());
  }
  return TemporaryFile(path);
}

std::string readFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  return buffer ? (*buffer)->getBuffer().str() : std::string();
}

ProcessResult runProgram(llvm::StringRef program,
                         llvm::ArrayRef<llvm::StringRef> args,
                         llvm::StringRef stdoutPath,
                         llvm::StringRef stderrPath) {
  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects{
      llvm::StringRef(""), stdoutPath, stderrPath};
  ProcessResult result{};
  result.exitCode = llvm::sys::ExecuteAndWait(
      program, args, /*Env=*/std::nullopt, redirects, /*SecondsToWait=*/0,
      /*MemoryLimit=*/0, &result.error);
  return result;
}

Status processError(llvm::StringRef action, const ProcessResult &result,
                    llvm::StringRef stderrPath) {
  std::string message = action.str() + " failed with exit code " +
                        std::to_string(result.exitCode);
  if (!result.error.empty()) {
    message += ": " + result.error;
  }
  if (std::string output = readFile(stderrPath); !output.empty()) {
    message += "; stderr: " + output;
  }
  return Status::CompilationError(std::move(message));
}

[[maybe_unused]] StatusOr<llvm::SmallVector<char, 0>>
assembleWithExecutable(llvm::ArrayRef<char> bytecode, SmTarget target,
                       mlir::cuda_tile::BytecodeVersion bytecodeVersion) {
  auto program = llvm::sys::findProgramByName("tileiras");
  if (!program) {
    return Status::NotFound("tileiras was not found on PATH");
  }

  TIR_ASSIGN_OR_RETURN(auto stdoutFile, createTemporaryFile("stdout"));
  TIR_ASSIGN_OR_RETURN(auto stderrFile, createTemporaryFile("stderr"));
  llvm::SmallVector<llvm::StringRef, 2> versionArgs{*program,
                                                    "--list-versions"};
  ProcessResult query =
      runProgram(*program, versionArgs, stdoutFile.path, stderrFile.path);
  if (query.exitCode != 0) {
    return processError("tileiras --list-versions", query, stderrFile.path);
  }

  auto stdoutBuffer = llvm::MemoryBuffer::getFile(stdoutFile.path);
  std::string requested =
      std::to_string(static_cast<unsigned>(bytecodeVersion.getMajor())) + "." +
      std::to_string(static_cast<unsigned>(bytecodeVersion.getMinor()));
  if (!stdoutBuffer ||
      !llvm::is_contained(llvm::split((*stdoutBuffer)->getBuffer(), "\n"),
                          requested)) {
    return Status::NotSupported("tileiras does not support TileIR " +
                                requested);
  }

  TIR_ASSIGN_OR_RETURN(auto inputFile, createTemporaryFile("tilebc"));
  TIR_ASSIGN_OR_RETURN(auto outputFile, createTemporaryFile("cubin"));
  std::error_code error;
  llvm::raw_fd_ostream input(inputFile.path, error);
  if (error) {
    return Status::CompilationError("failed to open TileIR input: " +
                                    error.message());
  }
  input.write(bytecode.data(), bytecode.size());
  input.close();
  if (input.has_error()) {
    input.clear_error();
    return Status::CompilationError("failed to write TileIR input");
  }
  std::string gpuArg =
      "--gpu-name=sm_" + std::to_string(target.getComputeCapabilityVersion());
  std::string outputArg = "--output-file=" + outputFile.path;
  llvm::SmallVector<llvm::StringRef, 4> args{*program, gpuArg, outputArg,
                                             inputFile.path};
  ProcessResult compile =
      runProgram(*program, args, stdoutFile.path, stderrFile.path);
  if (compile.exitCode != 0) {
    return processError("tileiras compilation", compile, stderrFile.path);
  }

  auto cubin = llvm::MemoryBuffer::getFile(outputFile.path);
  if (!cubin || (*cubin)->getBufferSize() == 0) {
    return Status::CompilationError("tileiras produced no cubin");
  }
  llvm::StringRef buffer = (*cubin)->getBuffer();
  return llvm::SmallVector<char, 0>(buffer.begin(), buffer.end());
}

bool isUnavailable(const Status &status) {
  return status.code() == StatusCode::kNotFound ||
         status.code() == StatusCode::kNotSupported;
}

} // namespace

StatusOr<std::optional<llvm::SmallVector<char, 0>>>
assembleTileIRToCubin(llvm::ArrayRef<char> bytecode, SmTarget target,
                      mlir::cuda_tile::BytecodeVersion bytecodeVersion) {

  auto executable = assembleWithExecutable(bytecode, target, bytecodeVersion);
  if (executable.ok()) {
    return std::optional<llvm::SmallVector<char, 0>>(std::move(*executable));
  }
  if (isUnavailable(executable.status())) {
    return std::optional<llvm::SmallVector<char, 0>>();
  }
  return executable.status();
}

} // namespace mlir::nv_tensor_ir::backend::cuda_tile
