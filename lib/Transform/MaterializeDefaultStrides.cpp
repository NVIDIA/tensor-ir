// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- MaterializeDefaultStrides.cpp - Row-major stride materialization ---===//
//
// Attaches a row-major stride attribute to every GraphOp argument and result
// of TensorType that does not already declare one. Running this pass at the
// front of every backend pipeline turns TensorIR's row-major default into an
// explicit IR fact, so downstream code can read strides from the graph
// attributes without needing to know the default-layout convention.
//
//===----------------------------------------------------------------------===//

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"
#include "tensor_ir/Transform/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DEF_MATERIALIZEDEFAULTSTRIDESPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace {

namespace tcg = mlir::nv_tensor_ir::tcutegen;

/// Build the row-major stride string for a tensor type's shape.
std::string makeRowMajorStrideString(TensorType tensorTy) {
  tcg::Shape shape = getShapeRef(tensorTy);
  return tcg::to_string(tcg::Layout(shape).stride());
}

/// If `existing` lacks a stride entry, return a new dictionary with one
/// inserted. Otherwise return `existing` unchanged. Skipped cases:
///   - Rank <= 1: trivially contiguous (stride is always 1); materializing
///     just clutters IR with `nv_tensor_ir.stride = "(1)"`.
///   - Any dynamic shape dim: materializing would introduce dynamic stride
///     values that the kernel launcher does not pass (the ABI only knows
///     about dynamic strides that were already explicit in the graph), so
///     the generated kernel would expect more args than it receives.
///     Dynamic-shape graphs fall back to the codegen-time row-major default.
DictionaryAttr ensureStrideEntry(MLIRContext *ctx, DictionaryAttr existing,
                                 TensorType tensorTy, StringRef strideName) {
  if (existing && existing.contains(strideName)) {
    return existing;
  }
  if (tensorTy.getRank() <= 1 || !tensorTy.hasStaticShape()) {
    return existing ? existing : DictionaryAttr::get(ctx, {});
  }
  SmallVector<NamedAttribute> attrs;
  if (existing) {
    attrs.append(existing.getValue().begin(), existing.getValue().end());
  }
  attrs.emplace_back(StringAttr::get(ctx, strideName),
                     StringAttr::get(ctx, makeRowMajorStrideString(tensorTy)));
  return DictionaryAttr::get(ctx, attrs);
}

/// Materialize stride entries on one attribute array (arg_attrs or res_attrs).
/// Returns the new ArrayAttr and sets `changed` to true if any slot was
/// updated. `types` provides the per-slot types so non-tensor slots are
/// passed through.
ArrayAttr materializeStrides(MLIRContext *ctx, ArrayAttr existing,
                             ArrayRef<Type> types, StringRef strideName,
                             bool &changed) {
  SmallVector<Attribute> updated;
  updated.reserve(types.size());
  for (size_t i = 0; i < types.size(); ++i) {
    auto tensorTy = dyn_cast<TensorType>(types[i]);
    DictionaryAttr slot;
    if (existing && i < existing.size()) {
      slot = dyn_cast_or_null<DictionaryAttr>(existing[i]);
    }
    if (!tensorTy) {
      updated.push_back(slot ? Attribute(slot) : DictionaryAttr::get(ctx, {}));
      continue;
    }
    DictionaryAttr next = ensureStrideEntry(ctx, slot, tensorTy, strideName);
    if (next != slot) {
      changed = true;
    }
    updated.push_back(next);
  }
  return ArrayAttr::get(ctx, updated);
}

} // namespace

struct MaterializeDefaultStridesPass
    : public impl::MaterializeDefaultStridesPassBase<
          MaterializeDefaultStridesPass> {
  using impl::MaterializeDefaultStridesPassBase<
      MaterializeDefaultStridesPass>::MaterializeDefaultStridesPassBase;

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    MLIRContext *ctx = graphOp.getContext();
    StringRef strideName = TensorIRDialect::getStrideAttrName();

    bool argsChanged = false;
    ArrayAttr newArgAttrs =
        materializeStrides(ctx, graphOp.getAllArgAttrs(),
                           graphOp.getArgumentTypes(), strideName, argsChanged);
    if (argsChanged) {
      graphOp.setAllArgAttrs(newArgAttrs);
    }

    bool resultsChanged = false;
    ArrayAttr newResAttrs = materializeStrides(ctx, graphOp.getAllResultAttrs(),
                                               graphOp.getResultTypes(),
                                               strideName, resultsChanged);
    if (resultsChanged) {
      graphOp.setAllResultAttrs(newResAttrs);
    }
  }
};

} // namespace mlir::nv_tensor_ir
