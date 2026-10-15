// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' | FileCheck %s

// The same producer is needed in both branches and after the conditional.
// Branch-local loads, compute values, and scalar setup must not escape their
// regions. The verifier checks dominance; captures check reuse within a scope.
// CHECK-LABEL: func.func @shared_producer
//       CHECK: scf.forall
//       CHECK: %[[BRANCH:.*]] = scf.if
//       CHECK: %[[THEN_LOAD:.*]] = nv_tensor_ir.load
//       CHECK: %[[THEN_SCALAR:.*]] = arith.addf
//       CHECK: %[[THEN_SPLAT:.*]] = nv_tensor_ir.splat %[[THEN_SCALAR]]
//       CHECK: %[[THEN_VALUE:.*]] = nv_tensor_ir.add %[[THEN_LOAD]], %[[THEN_SPLAT]]
//       CHECK: scf.yield %[[THEN_VALUE]]
//       CHECK: } else {
//       CHECK: %[[ELSE_LOAD:.*]] = nv_tensor_ir.load
//       CHECK: %[[ELSE_SCALAR:.*]] = arith.addf
//       CHECK: %[[ELSE_SPLAT:.*]] = nv_tensor_ir.splat %[[ELSE_SCALAR]]
//       CHECK: %[[ELSE_VALUE:.*]] = nv_tensor_ir.add %[[ELSE_LOAD]], %[[ELSE_SPLAT]]
//       CHECK: scf.yield %[[ELSE_VALUE]]
//       CHECK: }
//       CHECK: %[[OUTER_LOAD:.*]] = nv_tensor_ir.load
//       CHECK: %[[OUTER_SCALAR:.*]] = arith.addf
//       CHECK: %[[OUTER_SPLAT:.*]] = nv_tensor_ir.splat %[[OUTER_SCALAR]]
//       CHECK: %[[OUTER_VALUE:.*]] = nv_tensor_ir.add %[[OUTER_LOAD]], %[[OUTER_SPLAT]]
//       CHECK: %[[RESULT:.*]] = nv_tensor_ir.add %[[BRANCH]], %[[OUTER_VALUE]]
//       CHECK: nv_tensor_ir.store %[[RESULT]]
func.func @shared_producer(%input: memref<1xf32, #ptr.generic_space>,
    %output: memref<16xf32, #ptr.generic_space>, %scalar: f32) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 1, 8>
} {
  %input_tile = bufferization.to_tensor %input restrict
      : memref<1xf32, #ptr.generic_space> to tensor<1xf32>
  %twice = arith.addf %scalar, %scalar : f32
  %splat = nv_tensor_ir.splat %twice : tensor<1xf32>
  %shared = nv_tensor_ir.add %input_tile, %splat : tensor<1xf32>
  %half = nv_tensor_ir.broadcast %shared
      : tensor<1xf32> -> tensor<8xf32>
  %concat = nv_tensor_ir.concatenate %half, %half dimension = 0
      : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  %whole = nv_tensor_ir.broadcast %shared
      : tensor<1xf32> -> tensor<16xf32>
  %result = nv_tensor_ir.add %concat, %whole
      : tensor<16xf32>
  bufferization.materialize_in_destination %result in restrict writable %output
      : (tensor<16xf32>, memref<16xf32, #ptr.generic_space>) -> ()
  return
}
