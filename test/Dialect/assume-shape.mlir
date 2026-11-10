// RUN: tensor_ir-opt %s --split-input-file | tensor_ir-opt | FileCheck %s
// RUN: tensor_ir-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: nv_tensor_ir.graph @scaled_dimension
// CHECK-SAME: (%[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?x4xf32>)
// CHECK-DAG: %[[ROWS:.*]] = dim %[[B]] dimension = 0
// CHECK-DAG: %[[FOUR:.*]] = arith.constant 4 : index
// CHECK: %[[ELEMENTS:.*]] = arith.muli %[[ROWS]], %[[FOUR]] : index
// CHECK: assume_shape %[[A]], [%[[ELEMENTS]]] : tensor<?xf32>
nv_tensor_ir.graph @scaled_dimension(%a: tensor<?xf32>, %b: tensor<?x4xf32>) -> tensor<?xf32> {
  %rows = dim %b dimension = 0 : tensor<?x4xf32>
  %four = arith.constant 4 : index
  %elements = arith.muli %rows, %four : index
  assume_shape %a, [%elements] : tensor<?xf32>
  results %a : tensor<?xf32>
}

// -----

// CHECK-LABEL: func.func @mixed_extents
// CHECK-SAME: (%[[A:.*]]: tensor<4x?x?xf32>, %[[N:.*]]: index)
// CHECK: nv_tensor_ir.assume_shape %[[A]], [4, %[[N]], 0] : tensor<4x?x?xf32>
func.func @mixed_extents(%a: tensor<4x?x?xf32>, %n: index) {
  nv_tensor_ir.assume_shape %a, [4, %n, 0] : tensor<4x?x?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @rank_zero
// CHECK-SAME: (%[[A:.*]]: tensor<f32>)
// CHECK: nv_tensor_ir.assume_shape %[[A]], [] : tensor<f32>
func.func @rank_zero(%a: tensor<f32>) {
  nv_tensor_ir.assume_shape %a, [] : tensor<f32>
  return
}
