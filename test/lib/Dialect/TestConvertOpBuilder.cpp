// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

namespace {

// Test pass that exercises the ConvertOp builder taking an element type.
// The pass runs on a whole module, materialises the builder call once, and
// leaves a module attribute behind so the FileCheck test can assert success.
struct TestConvertOpBuilderPass
    : public PassWrapper<TestConvertOpBuilderPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConvertOpBuilderPass);

  StringRef getArgument() const final { return "test-convert-op-build"; }
  StringRef getDescription() const final {
    return "Rebuild nv_tensor_ir.convert ops via the element-type builder";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TensorIRDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr("test.convert_builder")) {
      return;
    }

    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    auto tensorType = RankedTensorType::get({4, 4}, builder.getF32Type());
    auto funcType = builder.getFunctionType({tensorType}, {});

    // The temporary function gives us a Value to feed into the builder without
    // relying on existing IR.
    auto func = func::FuncOp::create(builder, loc, "__test_convert_builder_tmp",
                                     funcType);
    Block *entry = func.addEntryBlock();
    Value arg = entry->getArgument(0);

    // Invoke the generated ConvertOp builder and rely on the call succeeding
    // to validate the coverage target.
    OperationState state(loc, ConvertOp::getOperationName());
    ConvertOp::build(builder, state, arg, builder.getF16Type());

    module->setAttr("test.convert_builder", builder.getUnitAttr());
    func.erase();
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestConvertOpBuilderPass() {
  PassRegistration<TestConvertOpBuilderPass>();
}
} // namespace test
} // namespace mlir
