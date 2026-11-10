// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Support/TileIRAssembly.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir::nv_tensor_ir;
using namespace mlir::nv_tensor_ir::backend::cuda_tile;

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
} // namespace

static StatusOr<TemporaryFile> createTemporaryFile(llvm::StringRef suffix) {
  llvm::SmallString<128> path;
  if (std::error_code error =
          llvm::sys::fs::createTemporaryFile("tensor-ir", suffix, path)) {
    return Status::CompilationError("failed to create temporary file: " +
                                    error.message());
  }
  return TemporaryFile(path);
}

static std::string readFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  return buffer ? (*buffer)->getBuffer().str() : std::string();
}

static Status forwardOutput(llvm::StringRef path, llvm::StringRef destination,
                            llvm::raw_ostream *capture = nullptr) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    return Status::CompilationError("cannot read assembler output: " +
                                    buffer.getError().message());
  }
  std::unique_ptr<llvm::raw_fd_ostream> file;
  llvm::raw_fd_ostream *stream =
      destination == "-" ? &llvm::outs() : &llvm::errs();
  if (!destination.empty() && destination != "-") {
    std::error_code error;
    file = std::make_unique<llvm::raw_fd_ostream>(destination, error);
    if (error) {
      return Status::CompilationError("cannot open remark output '" +
                                      destination.str() +
                                      "': " + error.message());
    }
    stream = file.get();
  }
  *stream << (*buffer)->getBuffer();
  stream->flush();
  if (file) {
    file->close();
  }
  if (stream->has_error()) {
    auto error = stream->error();
    stream->clear_error();
    return Status::CompilationError("cannot write remark output: " +
                                    error.message());
  }
  if (capture) {
    *capture << (*buffer)->getBuffer();
  }
  return Status::Ok();
}

static void appendError(Status &status, const Status &extra) {
  if (extra.ok()) {
    return;
  }
  status =
      status.ok()
          ? extra
          : Status::CompilationError(status.message() + "; " + extra.message());
}

static ProcessResult runProgram(llvm::StringRef program,
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

static Status processError(llvm::StringRef action, const ProcessResult &result,
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

static StatusOr<std::string> findTileiras(const CompilerInvocation &options) {
  if (!options.codegen.tileirasPath.empty()) {
    if (!llvm::sys::fs::can_execute(options.codegen.tileirasPath)) {
      return Status::NotFound("configured --tileiras-path is unavailable: " +
                              options.codegen.tileirasPath);
    }
    return options.codegen.tileirasPath;
  }
  if (std::string toolkitPath = options.getCudaToolkitPath();
      !toolkitPath.empty()) {
    llvm::SmallString<128> binPath(toolkitPath);
    llvm::sys::path::append(binPath, "bin");
    if (auto program = llvm::sys::findProgramByName("tileiras", {binPath})) {
      return *program;
    }
  }
  auto program = llvm::sys::findProgramByName("tileiras");
  if (!program) {
    return Status::NotFound(
        "tileiras was not found in the CUDA toolkit or PATH");
  }
  return *program;
}

static StatusOr<llvm::SmallVector<char, 0>> assembleWithExecutable(
    llvm::ArrayRef<char> bytecode, SmTarget target,
    std::optional<mlir::cuda_tile::BytecodeVersion> bytecodeVersion,
    const CompilerInvocation &options, const RemarkFilters *remarkFilters,
    llvm::raw_ostream &remarksStream) {
  TIR_ASSIGN_OR_RETURN(auto program, findTileiras(options));

  TIR_ASSIGN_OR_RETURN(auto stdoutFile, createTemporaryFile("stdout"));
  TIR_ASSIGN_OR_RETURN(auto stderrFile, createTemporaryFile("stderr"));
  // `tileiras` itself takes no version argument -- it determines the
  // bytecode's version from the bytecode's own contents. When we know the
  // version up front, do a cheap pre-flight check so an unsupported version
  // fails fast with a clear message instead of via `tileiras`'s own error.
  if (bytecodeVersion) {
    llvm::SmallVector<llvm::StringRef, 2> versionArgs{program,
                                                      "--list-versions"};
    ProcessResult query =
        runProgram(program, versionArgs, stdoutFile.path, stderrFile.path);
    if (query.exitCode != 0) {
      return processError("tileiras --list-versions", query, stderrFile.path);
    }

    auto stdoutBuffer = llvm::MemoryBuffer::getFile(stdoutFile.path);
    std::string requested =
        std::to_string(static_cast<unsigned>(bytecodeVersion->getMajor())) +
        "." +
        std::to_string(static_cast<unsigned>(bytecodeVersion->getMinor()));
    if (!stdoutBuffer ||
        !llvm::is_contained(llvm::split((*stdoutBuffer)->getBuffer(), "\n"),
                            requested)) {
      return Status::NotSupported("tileiras does not support TileIR " +
                                  requested);
    }
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
  std::optional<TemporaryFile> remarksFile;
  std::vector<std::string> remarkArgs;
  if (remarkFilters) {
    TIR_ASSIGN_OR_RETURN(auto temp, createTemporaryFile("remarks.yaml"));
    remarksFile.emplace(std::move(temp));
    remarkArgs = {"--lineinfo", "--remark-format=yaml",
                  "--remarks-output-file=" + remarksFile->path};
    const char *kinds[] = {"passed", "missed", "failed", "analysis"};
    for (unsigned i = 0; i < remarkFilters->categories.size(); ++i) {
      remarkArgs.push_back(
          std::string("--remarks-") + kinds[i] + "=" +
          stringifyRemarkCategory(remarkFilters->categories[i]).str());
    }
  }
  std::string gpuArg =
      "--gpu-name=sm_" + std::to_string(target.getComputeCapabilityVersion());
  std::string outputArg = "--output-file=" + outputFile.path;
  llvm::SmallVector<llvm::StringRef, 4> args{program, gpuArg, outputArg,
                                             inputFile.path};
  for (const auto &arg : remarkArgs) {
    args.push_back(arg);
  }
  ProcessResult compile =
      runProgram(program, args, stdoutFile.path, stderrFile.path);
  Status status =
      compile.exitCode == 0
          ? Status::Ok()
          : processError("tileiras compilation", compile, stderrFile.path);
  if (remarkFilters) {
    appendError(status, forwardOutput(stdoutFile.path, "-"));
    // processError includes stderr on failure, avoiding duplicate diagnostics.
    if (compile.exitCode == 0) {
      appendError(status, forwardOutput(stderrFile.path, ""));
    }
    appendError(status,
                forwardOutput(remarksFile->path,
                              options.optimizationRemarks.remarksOutputFile,
                              &remarksStream));
  }
  TIR_RETURN_IF_ERROR(status);

  auto cubin = llvm::MemoryBuffer::getFile(outputFile.path);
  if (!cubin || (*cubin)->getBufferSize() == 0) {
    return Status::CompilationError("tileiras produced no cubin");
  }
  llvm::StringRef buffer = (*cubin)->getBuffer();
  return llvm::SmallVector<char, 0>(buffer.begin(), buffer.end());
}

static bool isUnavailable(const Status &status) {
  return status.code() == StatusCode::kNotFound ||
         status.code() == StatusCode::kNotSupported;
}

StatusOr<std::optional<llvm::SmallVector<char, 0>>>
mlir::nv_tensor_ir::backend::cuda_tile::assembleTileIRToCubin(
    llvm::ArrayRef<char> bytecode, SmTarget target,
    std::optional<mlir::cuda_tile::BytecodeVersion> bytecodeVersion,
    const CompilerInvocation &options, llvm::raw_ostream &remarksStream) {
  TIR_ASSIGN_OR_RETURN(auto filters,
                       resolveRemarkFilters(options.optimizationRemarks));
  if (filters.isEnabled() || !options.codegen.tileirasPath.empty()) {
    // TODO: Revisit subprocess assembly for remarks once libtileiras supports
    // remark configuration and capture.
    TIR_ASSIGN_OR_RETURN(
        auto binary,
        assembleWithExecutable(bytecode, target, bytecodeVersion, options,
                               filters.isEnabled() ? &filters : nullptr,
                               remarksStream));
    return std::optional<llvm::SmallVector<char, 0>>(std::move(binary));
  }

  auto executable = assembleWithExecutable(bytecode, target, bytecodeVersion,
                                           options, nullptr, remarksStream);
  if (executable.ok()) {
    return std::optional<llvm::SmallVector<char, 0>>(std::move(*executable));
  }
  if (isUnavailable(executable.status())) {
    return std::optional<llvm::SmallVector<char, 0>>();
  }
  return executable.status();
}
