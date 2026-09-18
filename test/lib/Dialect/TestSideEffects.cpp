// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestSideEffectsPass
    : public PassWrapper<TestSideEffectsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSideEffectsPass);

  StringRef getArgument() const final { return "test-side-effects"; }
  StringRef getDescription() const final {
    return "Test memory side-effect interfaces";
  }

  void runOnOperation() override {
    SmallVector<MemoryEffects::EffectInstance, 8> effects;
    getOperation().walk([&](MemoryEffectOpInterface op) {
      effects.clear();
      op.getEffects(effects);

      if (op->hasTrait<OpTrait::IsTerminator>()) {
        return;
      }

      if (effects.empty()) {
        op.emitRemark() << "operation has no memory effects";
        return;
      }

      for (MemoryEffects::EffectInstance effect : effects) {
        auto diagnostic = op.emitRemark() << "found an instance of ";

        if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
          diagnostic << "'allocate'";
        } else if (isa<MemoryEffects::Free>(effect.getEffect())) {
          diagnostic << "'free'";
        } else if (isa<MemoryEffects::Read>(effect.getEffect())) {
          diagnostic << "'read'";
        } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
          diagnostic << "'write'";
        }

        if (effect.getValue()) {
          if (auto *operand = effect.getEffectValue<OpOperand *>()) {
            diagnostic << " on op operand " << operand->getOperandNumber()
                       << ",";
          } else if (auto result = effect.getEffectValue<OpResult>()) {
            diagnostic << " on op result " << result.getResultNumber() << ",";
          } else if (auto argument = effect.getEffectValue<BlockArgument>()) {
            diagnostic << " on block argument " << argument.getArgNumber()
                       << ",";
          }
        } else if (SymbolRefAttr symbol = effect.getSymbolRef()) {
          diagnostic << " on a symbol '" << symbol << "',";
        }

        diagnostic << " on resource '" << effect.getResource()->getName()
                   << "'";
      }
    });
  }
};

} // namespace

namespace mlir::test {

void registerTestSideEffectsPass() { PassRegistration<TestSideEffectsPass>(); }

} // namespace mlir::test
