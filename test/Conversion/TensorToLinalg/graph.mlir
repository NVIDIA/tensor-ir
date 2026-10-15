// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg,convert-tensor-to-linalg)' | FileCheck %s
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)' \
// RUN:   | tensor_ir-opt --split-input-file --mlir-print-local-scope | FileCheck %s
// RUN: tensor_ir-opt %s --mlir-print-op-generic | tensor_ir-opt \
// RUN:   | FileCheck %s --check-prefix=ROUNDTRIP

// CHECK-LABEL: func.func private @attributes(
// CHECK-SAME: %[[A:.*]]: tensor<2xf32> {test.arg = 7 : i64}
// CHECK-SAME: -> (tensor<2xf32> {test.result = "kept"})
// CHECK-SAME: attributes {test.graph = "kept"}
// ROUNDTRIP-LABEL: nv_tensor_ir.graph private @attributes(
nv_tensor_ir.graph private @attributes(%a: tensor<2xf32> {test.arg = 7 : i64}) -> (tensor<2xf32> {test.result = "kept"}) attributes {test.graph = "kept"} {
  results %a : tensor<2xf32>
}
// CHECK: return %[[A]]

// -----

// CHECK-LABEL: func.func @public_graph(
// ROUNDTRIP-LABEL: nv_tensor_ir.graph @public_graph(
nv_tensor_ir.graph @public_graph(%a: tensor<2xf32>) -> tensor<2xf32> {
  results %a : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func nested @nested_graph(
// ROUNDTRIP-LABEL: nv_tensor_ir.graph nested @nested_graph(
nv_tensor_ir.graph nested @nested_graph(%a: tensor<2xf32>) -> tensor<2xf32> {
  results %a : tensor<2xf32>
}
