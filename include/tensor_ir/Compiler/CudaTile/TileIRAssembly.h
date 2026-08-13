// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_CUDATILE_TILEIRASSEMBLY_H
#define TENSOR_IR_COMPILER_CUDATILE_TILEIRASSEMBLY_H

#include "tensor_ir/Support/Status.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "cuda_tile/Bytecode/Common/Version.h"
#include <optional>

namespace mlir::nv_tensor_ir::backend::cuda_tile {

/// Assemble the TileIR bytecode into a cubin by invoking 'tileiras'
/// as a subprocess for the given target and bytecode version.
/// Returns std::nullopt when the assembler is unavailable or incompatible;
/// assembler failures are returned.
StatusOr<std::optional<llvm::SmallVector<char, 0>>>
assembleTileIRToCubin(llvm::ArrayRef<char> bytecode, SmTarget target,
                      mlir::cuda_tile::BytecodeVersion bytecodeVersion);

} // namespace mlir::nv_tensor_ir::backend::cuda_tile

#endif // TENSOR_IR_COMPILER_CUDATILE_TILEIRASSEMBLY_H
