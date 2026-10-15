// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(canonicalize,convert-tensor-to-linalg)' \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.
// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)' \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg,convert-tensor-to-linalg)' \
// RUN:   | tensor_ir-opt --split-input-file --mlir-print-local-scope \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.

// CHECK-LABEL: func.func @dynamic_pointwise(
// CHECK-SAME: %[[A:.*]]: tensor<?x?xf32>, %{{.*}}: tensor<?x?xf32>
nv_tensor_ir.graph @dynamic_pointwise(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %r = add %a, %b : tensor<?x?xf32>
  results %r : tensor<?x?xf32>
}
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[A]], %[[C1]]
// CHECK: tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xf32>
// CHECK: linalg.generic

// -----

// CHECK-LABEL: func.func @dynamic_transpose(
// CHECK-SAME: %[[A:.*]]: tensor<?x4x?xf32>
nv_tensor_ir.graph @dynamic_transpose(%a: tensor<?x4x?xf32>) -> tensor<?x?x4xf32> {
  %r = transpose %a permutation = [2, 0, 1] : tensor<?x4x?xf32> -> tensor<?x?x4xf32>
  results %r : tensor<?x?x4xf32>
}
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D2:.*]] = tensor.dim %[[A]], %[[C2]]
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK: tensor.empty(%[[D2]], %[[D0]]) : tensor<?x?x4xf32>
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [2, 0, 1]

// -----

// CHECK-LABEL: func.func @dynamic_splat_iota(
// CHECK-SAME: %[[N:.*]]: index, %[[V:.*]]: f32
nv_tensor_ir.graph @dynamic_splat_iota(%n: index, %v: f32) -> (tensor<?x4xf32>, tensor<?x4xf32>) {
  %a = splat %v dynamic_dims(%n) : tensor<?x4xf32>
  %b = iota dimension = 1 dynamic_dims(%n) : tensor<?x4xf32>
  results %a, %b : tensor<?x4xf32>, tensor<?x4xf32>
}
// CHECK: tensor.empty(%[[N]]) : tensor<?x4xf32>
// CHECK: linalg.fill ins(%[[V]] : f32)
// CHECK: tensor.empty(%[[N]]) : tensor<?x4xf32>
// CHECK: %[[IDX:.*]] = linalg.index 1
// CHECK: %[[INT:.*]] = arith.index_cast %[[IDX]] : index to i64
// CHECK: arith.sitofp %[[INT]] : i64 to f32

// -----

// CHECK-LABEL: func.func @dynamic_reduce(
// CHECK-SAME: %[[A:.*]]: tensor<?x?xf32>
nv_tensor_ir.graph @dynamic_reduce(%a: tensor<?x?xf32>) -> tensor<?x1xf32> {
  %r = reduce(%a)<dimensions = [1], reduction_mode = <avg>> : tensor<?x?xf32> -> tensor<?x1xf32>
  results %r : tensor<?x1xf32>
}
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK: tensor.empty(%[[D0]]) : tensor<?xf32>
// CHECK: linalg.reduce
// CHECK: tensor.empty(%{{.*}}) : tensor<?x1xf32>
// CHECK: linalg.broadcast
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: tensor.dim %[[A]], %[[C1]]
// CHECK: arith.muli
// CHECK: arith.divf

// -----

// Equal tensor types do not imply equal runtime shapes.
// CHECK-LABEL: func.func @dynamic_reshape_same_type(
// CHECK-SAME: %[[A:.*]]: tensor<?x?xf32>, %[[M:.*]]: index, %[[N:.*]]: index
nv_tensor_ir.graph @dynamic_reshape_same_type(%a: tensor<?x?xf32>, %m: index, %n: index) -> tensor<?x?xf32> {
  %r = reshape %a dynamic_dims(%m, %n) : tensor<?x?xf32> -> tensor<?x?xf32>
  results %r : tensor<?x?xf32>
}
// CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[M]], %[[N]]
// CHECK: %[[RESULT:.*]] = tensor.reshape %[[A]](%[[SHAPE]])
// CHECK: return %[[RESULT]]
