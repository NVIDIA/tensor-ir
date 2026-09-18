// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Options/Options.h"
#include "tensor_ir/Options/OptionsCLDetail.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

#define GEN_OPTIONS_CL_BINDINGS
#include "tensor_ir/Options/CompilerOptions.h.inc"
