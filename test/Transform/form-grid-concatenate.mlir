// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,layout-propagation-annotation,layout-propagation-normalization,tile-selection,graph-splitting),tir-bufferize,func.func(tir-form-grid))' --split-input-file | FileCheck %s --implicit-check-not='nv_tensor_ir.concatenate' --implicit-check-not='nv_tensor_ir.iter_space'

// CHECK-LABEL: func.func @three_way_dispatch
//       CHECK: scf.forall
//       CHECK: %[[FIRST:[0-9]+]] = arith.cmpi ult
//       CHECK: %[[OUTER:[0-9]+]] = scf.if %[[FIRST]] -> (tensor<1xf32>) {
//       CHECK: %[[A:[0-9]+]] = nv_tensor_ir.load
//       CHECK: scf.yield %[[A]]
//       CHECK: } else {
//       CHECK: %[[SECOND:[0-9]+]] = arith.cmpi ult
//       CHECK: %[[INNER:[0-9]+]] = scf.if %[[SECOND]] -> (tensor<1xf32>) {
//       CHECK: %[[B:[0-9]+]] = nv_tensor_ir.load
//       CHECK: scf.yield %[[B]]
//       CHECK: } else {
//       CHECK: %[[C:[0-9]+]] = nv_tensor_ir.load
//       CHECK: scf.yield %[[C]]
//       CHECK: }
//       CHECK: scf.yield %[[INNER]]
//       CHECK: }
//       CHECK: nv_tensor_ir.store %[[OUTER]]
//   CHECK-NOT: nv_tensor_ir.iter_space_id
nv_tensor_ir.graph @three_way(
    %arg0: tensor<2xf32>, %arg1: tensor<3xf32>, %arg2: tensor<4xf32>)
    -> tensor<9xf32> attributes {tile_size = array<i32: 1>} {
  %result = concatenate %arg0, %arg1, %arg2 dimension = 0
      : (tensor<2xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<9xf32>
  results %result : tensor<9xf32>
}

// -----

// A slice may prune concat operands. Only the surviving operand positions are
// represented in ConcatSourceAttr::argumentIndex. Formation must neither load
// nor compute the discarded first input.

// CHECK-LABEL: func.func @pruned_dispatch
//       CHECK: %[[DISCARDED:.*]] = ptr.to_ptr
//       CHECK: %[[SECOND_PTR:.*]] = ptr.to_ptr
//       CHECK: %[[THIRD_PTR:.*]] = ptr.to_ptr
//       CHECK: scf.forall
//   CHECK-NOT: nv_tensor_ir.load %[[DISCARDED]]
//       CHECK: scf.if
//   CHECK-NOT: nv_tensor_ir.load %[[DISCARDED]]
//       CHECK: nv_tensor_ir.load %[[SECOND_PTR]]
//   CHECK-NOT: nv_tensor_ir.load %[[DISCARDED]]
//       CHECK: nv_tensor_ir.load %[[THIRD_PTR]]
//   CHECK-NOT: nv_tensor_ir.load %[[DISCARDED]]
//       CHECK: nv_tensor_ir.store
nv_tensor_ir.graph @pruned(
    %arg0: tensor<2xf32>, %arg1: tensor<4xf32>, %arg2: tensor<6xf32>)
    -> tensor<8xf32> attributes {tile_size = array<i32: 1, 4>} {
  %concat = concatenate %arg0, %arg1, %arg2 dimension = 0
      : (tensor<2xf32>, tensor<4xf32>, tensor<6xf32>) -> tensor<12xf32>
  %result = slice %concat starts = [2] limits = [10] strides = [1]
      : tensor<12xf32> -> tensor<8xf32>
  results %result : tensor<8xf32>
}

// -----

// CHECK-LABEL: func.func @nested_dispatch
//       CHECK: %[[OUTER:[0-9]+]] = scf.if
//       CHECK: %[[LEFT:[0-9]+]] = scf.if
//       CHECK: scf.yield %[[LEFT]]
//       CHECK: } else {
//       CHECK: %[[RIGHT:[0-9]+]] = scf.if
//       CHECK: scf.yield %[[RIGHT]]
//       CHECK: }
//       CHECK: nv_tensor_ir.store %[[OUTER]]
nv_tensor_ir.graph @nested(
    %arg0: tensor<2xf32>, %arg1: tensor<3xf32>,
    %arg2: tensor<4xf32>, %arg3: tensor<5xf32>)
    -> tensor<14xf32> attributes {tile_size = array<i32: 1>} {
  %left = concatenate %arg0, %arg1 dimension = 0
      : (tensor<2xf32>, tensor<3xf32>) -> tensor<5xf32>
  %right = concatenate %arg2, %arg3 dimension = 0
      : (tensor<4xf32>, tensor<5xf32>) -> tensor<9xf32>
  %result = concatenate %left, %right dimension = 0
      : (tensor<5xf32>, tensor<9xf32>) -> tensor<14xf32>
  results %result : tensor<14xf32>
}
