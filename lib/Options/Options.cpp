// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Options/Options.h"

#include "tensor_ir/Options/BytecodeVersionOptions.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/StringSaver.h"

llvm::LogicalResult
mlir::nv_tensor_ir::opt_detail::forEachOption(llvm::StringRef str,
                                              OptionHandler handler) {
  llvm::BumpPtrAllocator allocator;
  llvm::StringSaver saver(allocator);
  llvm::SmallVector<const char *> tokens;
  llvm::cl::TokenizeGNUCommandLine(str, saver, tokens);
  for (llvm::StringRef token : tokens) {
    size_t separator = token.find('=');
    if (separator == llvm::StringRef::npos || separator == 0 ||
        llvm::failed(handler(token.take_front(separator),
                             token.drop_front(separator + 1)))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

#define GEN_OPTIONS_PARSER
#include "tensor_ir/Options/CompilerOptions.h.inc"
#include "tensor_ir/Options/OptionsEnums.cpp.inc"
