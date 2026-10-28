// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' | FileCheck %s

// The same producer is needed in both branches and after the conditional.
// Branch-local loads, compute values, and scalar setup must not escape their
// regions. The verifier checks dominance; captures check reuse within a scope.
// CHECK-LABEL: func.func @shared_producer(
//  CHECK-SAME:   %[[INPUT:[^:]+]]: memref<1xf32>
//  CHECK-SAME:   %[[OUTPUT:[^:]+]]: memref<16xf32>
//  CHECK-SAME:   %[[SCALAR:[^:]+]]: f32
func.func @shared_producer(%input: memref<1xf32>,
    %output: memref<16xf32>, %scalar: f32) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 1, 8>
} {
  %input_tile = bufferization.to_tensor %input restrict
      : memref<1xf32> to tensor<1xf32>
  %twice = arith.addf %scalar, %scalar : f32
  %splat = nv_tensor_ir.splat %twice : tensor<1xf32>
  %shared = nv_tensor_ir.add %input_tile, %splat : tensor<1xf32>
  %half = nv_tensor_ir.broadcast %shared
      : tensor<1xf32> -> tensor<8xf32>
  //       CHECK:   scf.forall (%[[BID:.*]]) in (2) {
  //       CHECK:     %[[ONE:.*]] = arith.constant 1 : index
  //       CHECK:     %[[IS_FIRST:.*]] = arith.cmpi ult, %[[BID]], %[[ONE]] : index
  //       CHECK:     %[[BRANCH:.*]] = scf.if %[[IS_FIRST]] -> (tensor<1x8xf32>) {
  //       CHECK:       %[[THEN_VIEW:.*]] = memref.reinterpret_cast %[[INPUT]]
  //  CHECK-SAME:        to offset: [0], sizes: [1, 1], strides: [2, 2]
  //  CHECK-SAME:        : memref<1xf32> to memref<1x1xf32, strided<[2, 2]>>
  //       CHECK:       %[[THEN_LOAD:.*]] = nv_tensor_ir.load %[[THEN_VIEW]] tile_indices [0, 0]
  //       CHECK:       %[[THEN_BROADCAST:.*]] = nv_tensor_ir.broadcast %[[THEN_LOAD]] : tensor<1x1xf32> -> tensor<1x8xf32>
  //       CHECK:       %[[THEN_SCALAR:.*]] = arith.addf %[[SCALAR]], %[[SCALAR]] : f32
  //       CHECK:       %[[THEN_SPLAT:.*]] = nv_tensor_ir.splat %[[THEN_SCALAR]] : tensor<1x8xf32>
  //       CHECK:       %[[THEN_VALUE:.*]] = nv_tensor_ir.add %[[THEN_BROADCAST]], %[[THEN_SPLAT]] : tensor<1x8xf32>
  //       CHECK:       scf.yield %[[THEN_VALUE]] : tensor<1x8xf32>
  //       CHECK:     } else {
  //       CHECK:       %[[ELSE_VIEW:.*]] = memref.reinterpret_cast %[[INPUT]]
  //  CHECK-SAME:        to offset: [0], sizes: [1, 1], strides: [2, 2]
  //  CHECK-SAME:        : memref<1xf32> to memref<1x1xf32, strided<[2, 2]>>
  //       CHECK:       %[[ELSE_LOAD:.*]] = nv_tensor_ir.load %[[ELSE_VIEW]] tile_indices [0, 0]
  //       CHECK:       %[[ELSE_BROADCAST:.*]] = nv_tensor_ir.broadcast %[[ELSE_LOAD]] : tensor<1x1xf32> -> tensor<1x8xf32>
  //       CHECK:       %[[ELSE_SCALAR:.*]] = arith.addf %[[SCALAR]], %[[SCALAR]] : f32
  //       CHECK:       %[[ELSE_SPLAT:.*]] = nv_tensor_ir.splat %[[ELSE_SCALAR]] : tensor<1x8xf32>
  //       CHECK:       %[[ELSE_VALUE:.*]] = nv_tensor_ir.add %[[ELSE_BROADCAST]], %[[ELSE_SPLAT]] : tensor<1x8xf32>
  //       CHECK:       scf.yield %[[ELSE_VALUE]] : tensor<1x8xf32>
  //       CHECK:     }
  %concat = nv_tensor_ir.concatenate %half, %half dimension = 0
      : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  //       CHECK:     %[[OUTER_VIEW:.*]] = memref.reinterpret_cast %[[INPUT]]
  //  CHECK-SAME:      to offset: [0], sizes: [1, 1], strides: [2, 2]
  //  CHECK-SAME:      : memref<1xf32> to memref<1x1xf32, strided<[2, 2]>>
  //       CHECK:     %[[OUTER_LOAD:.*]] = nv_tensor_ir.load %[[OUTER_VIEW]] tile_indices [0, 0]
  //       CHECK:     %[[OUTER_BROADCAST:.*]] = nv_tensor_ir.broadcast %[[OUTER_LOAD]] : tensor<1x1xf32> -> tensor<1x8xf32>
  //       CHECK:     %[[OUTER_SCALAR:.*]] = arith.addf %[[SCALAR]], %[[SCALAR]] : f32
  //       CHECK:     %[[OUTER_SPLAT:.*]] = nv_tensor_ir.splat %[[OUTER_SCALAR]] : tensor<1x8xf32>
  //       CHECK:     %[[OUTER_VALUE:.*]] = nv_tensor_ir.add %[[OUTER_BROADCAST]], %[[OUTER_SPLAT]] : tensor<1x8xf32>
  %whole = nv_tensor_ir.broadcast %shared
      : tensor<1xf32> -> tensor<16xf32>
  //       CHECK:     %[[RESULT:.*]] = nv_tensor_ir.add %[[BRANCH]], %[[OUTER_VALUE]] : tensor<1x8xf32>
  %result = nv_tensor_ir.add %concat, %whole
      : tensor<16xf32>
  //       CHECK:     %[[OUT_VIEW:.*]] = memref.reinterpret_cast %[[OUTPUT]]
  //  CHECK-SAME:      to offset: [0], sizes: [2, 8], strides: [8, 1]
  //  CHECK-SAME:      : memref<16xf32> to memref<2x8xf32, strided<[8, 1]>>
  //       CHECK:     nv_tensor_ir.store %[[RESULT]], %[[OUT_VIEW]] tile_indices [%[[BID]], 0]
  //       CHECK:   } {mapping = [#gpu.block<x>]}
  bufferization.materialize_in_destination %result in restrict writable %output
      : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}
