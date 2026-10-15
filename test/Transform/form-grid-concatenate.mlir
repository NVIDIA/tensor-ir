// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid))' --split-input-file | FileCheck %s --implicit-check-not='nv_tensor_ir.concatenate' --implicit-check-not='nv_tensor_ir.iter_space'

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

// CHECK-LABEL: func.func @pruned_dispatch(
//  CHECK-SAME:     %[[ARG0:[^:]+]]: memref<2xf32, #ptr.generic_space>
//  CHECK-SAME:     %[[ARG1:[^:]+]]: memref<4xf32, #ptr.generic_space>
//  CHECK-SAME:     %[[ARG2:[^:]+]]: memref<6xf32, #ptr.generic_space>
//  CHECK-SAME:     %[[ARG3:[^:]+]]: memref<8xf32, #ptr.generic_space>
//   CHECK-DAG:   %[[ARG1_PTR:.*]] = ptr.to_ptr %[[ARG1]]
//   CHECK-DAG:   %[[ARG2_PTR:.*]] = ptr.to_ptr %[[ARG2]]
//   CHECK-DAG:   %[[ARG3_PTR:.*]] = ptr.to_ptr %[[ARG3]]

//       CHECK:   scf.forall (%[[ROW:.*]]) in (2) {
nv_tensor_ir.graph @pruned(
    %arg0: tensor<2xf32>, %arg1: tensor<4xf32>, %arg2: tensor<6xf32>)
    -> tensor<8xf32> attributes {tile_size = array<i32: 1, 4>} {
  //       CHECK:     %[[ONE:.*]] = arith.constant 1 : index
  //       CHECK:     %[[IS_ARG1:.*]] = arith.cmpi ult, %[[ROW]], %[[ONE]] : index
  //       CHECK:     %[[TILE:.*]] = scf.if %[[IS_ARG1]] -> (tensor<1x4xf32>) {
  //       CHECK:       %[[ARG1_TILE:.*]] = nv_tensor_ir.load %[[ARG1_PTR]][0, 0]
  //  CHECK-SAME:         view offset: [0], sizes: [1, 4], strides: [0, 1], alignment: 4
  //  CHECK-SAME:         to tensor<1x4xf32>
  //       CHECK:       scf.yield %[[ARG1_TILE]] : tensor<1x4xf32>
  //       CHECK:     } else {
  //       CHECK:       %[[ARG2_TILE:.*]] = nv_tensor_ir.load %[[ARG2_PTR]][0, 0]
  //  CHECK-SAME:         view offset: [0], sizes: [1, 4], strides: [0, 1], alignment: 4
  //  CHECK-SAME:         to tensor<1x4xf32>
  //       CHECK:       scf.yield %[[ARG2_TILE]] : tensor<1x4xf32>
  //       CHECK:     }
  %concat = concatenate %arg0, %arg1, %arg2 dimension = 0
      : (tensor<2xf32>, tensor<4xf32>, tensor<6xf32>) -> tensor<12xf32>
  // The slice [2, 10) emits no ops. It covers all of %arg1 and the first four
  // elements of %arg2, so both loads above start at offset 0.
  %result = slice %concat starts = [2] limits = [10] strides = [1]
      : tensor<12xf32> -> tensor<8xf32>
  // The selected tile is stored at its grid coordinate.
  //       CHECK:     nv_tensor_ir.store %[[TILE]], %[[ARG3_PTR]][%[[ROW]], 0]
  //  CHECK-SAME:       view offset: [0], sizes: [2, 4], strides: [4, 1], alignment: 4
  //       CHECK:   } {mapping = [#gpu.block<x>]}
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
