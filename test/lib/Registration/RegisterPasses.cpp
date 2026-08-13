// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RegisterPasses.h"

namespace mlir::test {

void registerTestConvertOpBuilderPass();
void registerTestTileAnalyzerPass();

} // namespace mlir::test

namespace mlir::nv_tensor_ir::test {

void registerAllPasses() {
  ::mlir::test::registerTestConvertOpBuilderPass();
  ::mlir::test::registerTestTileAnalyzerPass();
}

} // namespace mlir::nv_tensor_ir::test
