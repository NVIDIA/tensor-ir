// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Options/Options.h" // IWYU pragma: keep

#include "tensor_ir/Options/BytecodeVersionOptions.h" // IWYU pragma: keep
#include "tensor_ir/Options/OptionsDetail.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/StringSaver.h"

using namespace mlir::nv_tensor_ir;

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

bool mlir::nv_tensor_ir::RemarkFilters::isEnabled() const {
  return llvm::any_of(categories, [](RemarkCategory category) {
    return category != RemarkCategory::None;
  });
}

StatusOr<RemarkFilters> mlir::nv_tensor_ir::resolveRemarkFilters(
    const OptimizationRemarkOptions &options) {
  if (auto error = options.validate()) {
    return Status::InvalidArgument(*error);
  }
  if (options.remarksFilter == RemarkCategory::Inherit) {
    return Status::InvalidArgument("remarks cannot inherit a global category");
  }
  RemarkFilters result{
      {options.remarksPassedFilter, options.remarksMissedFilter,
       options.remarksFailedFilter, options.remarksAnalysisFilter}};
  for (unsigned i = 0; i < result.categories.size(); ++i) {
    RemarkCategory &category = result.categories[i];
    bool inherited = category == RemarkCategory::Inherit;
    if (inherited) {
      category = options.remarksFilter;
    }
  }
  return result;
}
