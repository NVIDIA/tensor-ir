// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-line support for code emitted by -gen-component-options.

#ifndef TENSORIR_OPTIONS_OPTIONSCLDETAIL_H
#define TENSORIR_OPTIONS_OPTIONSCLDETAIL_H

#include "tensor_ir/Options/OptionsDetail.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace mlir::nv_tensor_ir::opt_detail {

[[noreturn]] inline void reportCommandLineError(const llvm::Twine &message) {
  llvm::errs() << "error: " << message << '\n';
  std::exit(1);
}

template <typename T, typename ParseFn>
LogicalResult appendOccurrences(std::vector<T> &out,
                                llvm::ArrayRef<std::string> occurrences,
                                llvm::StringRef separators, ParseFn &&parse) {
  for (llvm::StringRef value : occurrences) {
    if (llvm::failed(appendElements(out, value, separators, parse))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

template <typename T>
void parseListOrFatal(std::vector<T> &out,
                      llvm::ArrayRef<std::string> occurrences, char delimiter,
                      llvm::StringRef flag) {
  out.clear();
  if (llvm::succeeded(appendOccurrences(out, occurrences,
                                        llvm::StringRef(&delimiter, 1),
                                        [](llvm::StringRef text, T &value) {
                                          return parseValue(text, value);
                                        }))) {
    return;
  }
  reportCommandLineError(llvm::Twine("--") + flag +
                         ": cannot parse as a list separated by '" +
                         llvm::Twine(delimiter) + "'");
}

inline void parseShapeOrFatal(std::vector<int64_t> &out, llvm::StringRef value,
                              llvm::StringRef flag) {
  if (llvm::succeeded(
          parseElements(out, value, kShapeSeparators, parseShapeExtent))) {
    return;
  }
  reportCommandLineError(
      llvm::Twine("--") + flag +
      ": cannot parse as a shape; expected non-negative extents or '?', "
      "separated by ',' or 'x'");
}

} // namespace mlir::nv_tensor_ir::opt_detail

#endif // TENSORIR_OPTIONS_OPTIONSCLDETAIL_H
