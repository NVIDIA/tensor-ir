// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_CUDATILE_TILEIRASSEMBLY_H
#define TENSOR_IR_COMPILER_CUDATILE_TILEIRASSEMBLY_H

#include "tensor_ir/Options/Options.h"
#include "tensor_ir/Support/Status.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Bytecode/Common/Version.h"
#include <optional>

namespace mlir::nv_tensor_ir::backend::cuda_tile {

/// Assemble TileIR bytecode for the target, checking bytecodeVersion if given.
/// Returns std::nullopt when the assembler is unavailable or incompatible;
/// other assembly failures are errors. Remarks or an explicit tileirasPath
/// require executable assembly without fallback. Executable lookup uses
/// tileirasPath, getCudaToolkitPath()/bin, then PATH.
/// Append opaque YAML to remarksStream and write it to remarksOutputFile
/// (empty for stderr, "-" for stdout).
StatusOr<std::optional<llvm::SmallVector<char, 0>>> assembleTileIRToCubin(
    llvm::ArrayRef<char> bytecode, SmTarget target,
    std::optional<mlir::cuda_tile::BytecodeVersion> bytecodeVersion,
    const CompilerInvocation &options, llvm::raw_ostream &remarksStream);

} // namespace mlir::nv_tensor_ir::backend::cuda_tile

#endif // TENSOR_IR_COMPILER_CUDATILE_TILEIRASSEMBLY_H
