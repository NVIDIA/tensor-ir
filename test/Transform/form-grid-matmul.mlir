// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' | FileCheck %s

// The operand plan selects M and N from different output dimensions. The
// result plan restores the transposed layout and broadcasts the batch tile.
// CHECK-LABEL: func.func @transposed_broadcast_matmul
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (12)
//       CHECK: %[[BATCH:.*]] = arith.remui %[[LINEAR]],
//       CHECK: %[[REST:.*]] = arith.divui %[[LINEAR]],
//       CHECK: %[[N:.*]] = arith.remui %[[REST]],
//       CHECK: %[[M:.*]] = arith.divui %[[REST]],
//       CHECK: %[[LHS:.*]] = nv_tensor_ir.load {{.*}}[%[[M]], 0]
//  CHECK-SAME: to tensor<2x8xf32>
//       CHECK: %[[RHS:.*]] = nv_tensor_ir.load {{.*}}[0, %[[N]]]
//  CHECK-SAME: to tensor<8x2xf32>
//       CHECK: %[[PRODUCT:.*]] = nv_tensor_ir.matmul(%[[LHS]], %[[RHS]])
//  CHECK-SAME: nv_tensor_ir.lhs_contracting_dimensions = array<i64: 1>
//  CHECK-SAME: nv_tensor_ir.rhs_contracting_dimensions = array<i64: 0>
//       CHECK: %[[RESHAPED:.*]] = nv_tensor_ir.reshape %[[PRODUCT]] : tensor<2x2xf32> -> tensor<1x2x2xf32>
//       CHECK: %[[TRANSPOSED:.*]] = nv_tensor_ir.transpose %[[RESHAPED]] permutation = [0, 2, 1]
//       CHECK: %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[TRANSPOSED]] : tensor<1x2x2xf32> -> tensor<2x2x2xf32>
//       CHECK: nv_tensor_ir.store %[[BROADCAST]], {{.*}}[%[[BATCH]], %[[N]], %[[M]]]
func.func @transposed_broadcast_matmul(
    %lhs: memref<1x4x8xf32, #ptr.generic_space>,
    %rhs: memref<1x8x6xf32, #ptr.generic_space>,
    %output: memref<3x6x4xf32, #ptr.generic_space>)
    attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 2, 2>
} {
  %a = bufferization.to_tensor %lhs restrict
      : memref<1x4x8xf32, #ptr.generic_space> to tensor<1x4x8xf32>
  %b = bufferization.to_tensor %rhs restrict
      : memref<1x8x6xf32, #ptr.generic_space> to tensor<1x8x6xf32>
  %product = nv_tensor_ir.matmul(%a, %b)
      : (tensor<1x4x8xf32>, tensor<1x8x6xf32>) -> tensor<1x4x6xf32>
  %transposed = nv_tensor_ir.transpose %product permutation = [0, 2, 1]
      : tensor<1x4x6xf32> -> tensor<1x6x4xf32>
  %result = nv_tensor_ir.broadcast %transposed
      : tensor<1x6x4xf32> -> tensor<3x6x4xf32>
  bufferization.materialize_in_destination %result in restrict writable %output
      : (tensor<3x6x4xf32>, memref<3x6x4xf32, #ptr.generic_space>) -> ()
  return
}
