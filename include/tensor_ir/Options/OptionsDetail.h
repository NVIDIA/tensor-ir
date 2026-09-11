// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runtime support for code emitted by -gen-component-options.

#ifndef TENSORIR_OPTIONS_OPTIONSDETAIL_H
#define TENSORIR_OPTIONS_OPTIONSDETAIL_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir::nv_tensor_ir::opt_detail {

using llvm::LogicalResult;

/// Visits GNU-tokenized `key=value` pairs in `str`.
using OptionHandler =
    llvm::function_ref<LogicalResult(llvm::StringRef, llvm::StringRef)>;
LogicalResult forEachOption(llvm::StringRef str, OptionHandler handler);

/// Visits elements separated by any character in `separators`.
template <typename Fn>
LogicalResult forEachElement(llvm::StringRef value, llvm::StringRef separators,
                             Fn &&fn) {
  if (value.empty()) {
    return llvm::success();
  }
  while (true) {
    size_t end = value.find_first_of(separators);
    if (llvm::failed(fn(value.substr(0, end)))) {
      return llvm::failure();
    }
    if (end == llvm::StringRef::npos) {
      return llvm::success();
    }
    value = value.drop_front(end + 1);
  }
}

inline LogicalResult parseValue(llvm::StringRef value, std::string &out) {
  out = value.str();
  return llvm::success();
}
inline LogicalResult parseValue(llvm::StringRef value, bool &out) {
  if (value == "true" || value == "1") {
    out = true;
    return llvm::success();
  }
  if (value == "false" || value == "0") {
    out = false;
    return llvm::success();
  }
  return llvm::failure();
}
template <typename T,
          std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>,
                           int> = 0>
LogicalResult parseValue(llvm::StringRef value, T &out) {
  return llvm::failure(value.getAsInteger(/*Radix=*/0, out));
}
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
LogicalResult parseValue(llvm::StringRef value, T &out) {
  return llvm::failure(!llvm::to_float(value, out));
}

constexpr char kDynamicExtentToken = '?';
constexpr llvm::StringLiteral kShapeSeparators = ",x";
// Kept independent of MLIR IR headers; Options.cpp checks this value.
constexpr int64_t kDynamicExtent = std::numeric_limits<int64_t>::min();
inline LogicalResult parseShapeExtent(llvm::StringRef value, int64_t &out) {
  if (value == "?") {
    out = kDynamicExtent;
    return llvm::success();
  }
  if (value.empty() || !llvm::all_of(value, llvm::isDigit)) {
    return llvm::failure();
  }
  return llvm::failure(value.getAsInteger(/*Radix=*/10, out));
}
template <typename T, typename ParseFn>
LogicalResult appendElements(std::vector<T> &out, llvm::StringRef value,
                             llvm::StringRef separators, ParseFn &&parse) {
  return forEachElement(value, separators, [&](llvm::StringRef text) {
    T element{};
    if (llvm::failed(parse(text, element))) {
      return llvm::failure();
    }
    out.push_back(std::move(element));
    return llvm::success();
  });
}
template <typename T, typename ParseFn>
LogicalResult parseElements(std::vector<T> &out, llvm::StringRef value,
                            llvm::StringRef separators, ParseFn &&parse) {
  out.clear();
  return appendElements(out, value, separators, std::forward<ParseFn>(parse));
}
inline void printMaybeQuoted(llvm::raw_ostream &os, llvm::StringRef value) {
  bool quote = value.empty() || llvm::any_of(value, [](char c) {
                 return llvm::isSpace(c) || c == '"' || c == '\\';
               });
  if (!quote) {
    os << value;
    return;
  }
  os << '"';
  for (char c : value) {
    if (c == '"' || c == '\\') {
      os << '\\';
    }
    os << c;
  }
  os << '"';
}

inline void printScalar(llvm::raw_ostream &os, llvm::StringRef value) {
  printMaybeQuoted(os, value);
}
inline void printScalar(llvm::raw_ostream &os, bool value) {
  os << (value ? "true" : "false");
}
template <typename T,
          std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                           int> = 0>
void printScalar(llvm::raw_ostream &os, T value) {
  os << value;
}
template <typename T, typename PrintFn>
void printElements(llvm::raw_ostream &os, const std::vector<T> &values,
                   char delimiter, PrintFn &&print) {
  const char separator[] = {delimiter, '\0'};
  llvm::interleave(
      values, os, [&](const T &value) { print(os, value); }, separator);
}
inline void printShape(llvm::raw_ostream &os,
                       const std::vector<int64_t> &extents, char delimiter) {
  printElements(os, extents, delimiter,
                [](llvm::raw_ostream &stream, int64_t extent) {
                  if (extent == kDynamicExtent) {
                    stream << kDynamicExtentToken;
                  } else {
                    stream << extent;
                  }
                });
}

} // namespace mlir::nv_tensor_ir::opt_detail

#endif // TENSORIR_OPTIONS_OPTIONSDETAIL_H
