// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: tensor_ir-tblgen %s --gen-component-options \
// RUN:   -I %S/../../../include %mlir_include_dirs \
// RUN:   | FileCheck %s

include "tensor_ir/Options/OptionsBase.td"

def EmptyOptions : ComponentOptions<"Empty"> {
  let invocationMember = "empty";
}
def CompilerInvocation : CompilerInvocationOptions<[EmptyOptions]>;

// CHECK: friend bool operator==(const CompilerInvocationComponents &a,
// CHECK:   return true;
