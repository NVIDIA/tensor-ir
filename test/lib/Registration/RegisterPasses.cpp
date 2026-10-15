// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RegisterPasses.h"

namespace mlir::test {

void registerTestConvertOpBuilderPass();
void registerTestSideEffectsPass();
void registerTestControlFlowInterfacePass();
void registerTestLayoutPropagationPasses();
void registerTestTileAnalyzerPass();
void registerTestTilingEnumeratorPass();
void registerTestTilingEvaluatorPass();

} // namespace mlir::test

namespace mlir::nv_tensor_ir::test {

void registerAllPasses() {
  ::mlir::test::registerTestConvertOpBuilderPass();
  ::mlir::test::registerTestSideEffectsPass();
  ::mlir::test::registerTestControlFlowInterfacePass();
  ::mlir::test::registerTestLayoutPropagationPasses();
  ::mlir::test::registerTestTileAnalyzerPass();
  ::mlir::test::registerTestTilingEnumeratorPass();
  ::mlir::test::registerTestTilingEvaluatorPass();
}

} // namespace mlir::nv_tensor_ir::test
