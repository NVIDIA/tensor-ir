// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Support/TIROverrideOptions.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace mlir::nv_tensor_ir {
namespace {

using ArgList = llvm::SmallVector<std::string, 32>;

llvm::StringRef optionName(llvm::StringRef arg) {
  if (arg.starts_with("--")) {
    arg = arg.drop_front(2);
  } else if (arg.starts_with('-')) {
    arg = arg.drop_front(1);
  }
  return arg.split('=').first;
}

bool isOptionArg(llvm::StringRef arg) { return arg.starts_with("-"); }

bool consumesNextArg(llvm::StringRef arg) {
  if (arg.contains('=')) {
    return false;
  }

  auto &registeredOptions = llvm::cl::getRegisteredOptions();
  auto it = registeredOptions.find(optionName(arg));
  return it != registeredOptions.end() &&
         it->second->getValueExpectedFlag() == llvm::cl::ValueRequired;
}

auto findFlag(llvm::ArrayRef<std::string> args, llvm::StringRef flag) {
  return llvm::find_if(args, [&](llvm::StringRef arg) {
    return isOptionArg(arg) && optionName(arg) == flag;
  });
}

void substituteFlagPrefix(ArgList &args, llvm::StringRef oldFlag,
                          llvm::StringRef newFlag) {
  std::string doubleDash = ("--" + oldFlag).str();
  std::string singleDash = ("-" + oldFlag).str();
  for (std::string &arg : args) {
    llvm::StringRef argRef(arg);
    if (argRef.starts_with(doubleDash)) {
      arg = ("--" + newFlag + arg.substr(doubleDash.size())).str();
    } else if (argRef.starts_with(singleDash)) {
      arg = ("-" + newFlag + arg.substr(singleDash.size())).str();
    }
  }
}

void applySubstitutionEdit(ArgList &args, llvm::StringRef edit) {
  if (!edit.starts_with("s/") || !edit.ends_with('/')) {
    return;
  }

  llvm::StringRef substitution = edit.drop_front(2).drop_back();
  if (!substitution.contains('/')) {
    return;
  }

  auto [oldFlag, newFlag] = substitution.split('/');
  substituteFlagPrefix(args, oldFlag, newFlag);
}

void removeAllPositionals(ArgList &args) {
  args.erase(llvm::remove_if(
                 args, [](llvm::StringRef arg) { return !isOptionArg(arg); }),
             args.end());
}

void removeFirstPositional(ArgList &args) {
  auto it = llvm::find_if(
      args, [](llvm::StringRef arg) { return !isOptionArg(arg); });
  if (it != args.end()) {
    args.erase(it);
  }
}

void removeFlag(ArgList &args, llvm::StringRef flag) {
  auto it = findFlag(args, flag);
  while (it != args.end()) {
    bool eraseValue = consumesNextArg(*it);
    it = args.erase(it);
    if (eraseValue && it != args.end()) {
      it = args.erase(it);
    }
    it = findFlag(args, flag);
  }
}

void moveFlagToFront(ArgList &args, llvm::StringRef flag) {
  auto found = findFlag(args, flag);
  if (found == args.end()) {
    return;
  }

  auto first = args.begin() + (found - args.data());
  auto last = first + 1;
  if (consumesNextArg(*first) && last != args.end()) {
    ++last;
  }
  std::rotate(args.begin(), first, last);
}

void appendFlagIfMissing(ArgList &args, llvm::StringRef flag,
                         llvm::StringRef optionSpec) {
  if (findFlag(args, flag) == args.end()) {
    args.push_back("--" + optionSpec.str());
  }
}

void applyEdit(ArgList &args, llvm::StringRef edit) {
  if (edit.empty()) {
    return;
  }

  switch (edit.front()) {
  case 's':
    applySubstitutionEdit(args, edit);
    return;
  case 'X':
    removeAllPositionals(args);
    return;
  case 'x':
    removeFirstPositional(args);
    return;
  case '#':
  case '^':
  case '+':
    break;
  default:
    return;
  }

  llvm::StringRef optionSpec = edit.drop_front();
  if (optionSpec.empty()) {
    return;
  }
  llvm::StringRef flag = optionSpec.split('=').first;
  switch (edit.front()) {
  case '#':
    removeFlag(args, flag);
    return;
  case '^':
    moveFlagToFront(args, flag);
    return;
  case '+':
    appendFlagIfMissing(args, flag, optionSpec);
    return;
  default:
    return;
  }
}

ArgList applyTIROverrideOptions(int argc, char **argv) {
  ArgList args;
  if (argc <= 0) {
    return args;
  }
  args.reserve(static_cast<size_t>(argc - 1));
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }

  const char *overrideSpec = std::getenv("TIR_OVERRIDE_OPTIONS");
  if (!overrideSpec || !*overrideSpec) {
    return args;
  }

  llvm::SmallVector<llvm::StringRef, 8> edits;
  llvm::SplitString(overrideSpec, edits, ",");
  for (llvm::StringRef edit : edits) {
    edit = edit.trim();
    if (edit.empty()) {
      continue;
    }
    applyEdit(args, edit);
  }
  return args;
}

} // namespace

TIRCommandLineOverride::TIRCommandLineOverride(int argc, char **argv)
    : args_(applyTIROverrideOptions(argc, argv)) {
  if (argc > 0) {
    argv_.push_back(argv[0]);
  }
  for (std::string &arg : args_) {
    argv_.push_back(arg.data());
  }
  argv_.push_back(nullptr);
}

} // namespace mlir::nv_tensor_ir
