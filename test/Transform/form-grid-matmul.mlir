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
#matmul = #nv_tensor_ir.matmul_source<"(3,6,4,8):(0,8,48,1)", 1, 4, 6, 8,
    #nv_tensor_ir.tensor_source<0, 0, "(4,8):(8,1)">,
    #nv_tensor_ir.tensor_source<1, 0, "(8,6):(6,1)">>
func.func @transposed_broadcast_matmul(%lhs: !ptr.ptr<#ptr.generic_space>,
    %rhs: !ptr.ptr<#ptr.generic_space>, %output: !ptr.ptr<#ptr.generic_space>)
    attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #matmul,
  tile_size = array<i32: 2, 2, 2>
} {
  %zero = arith.constant 0 : index
  %a = nv_tensor_ir.load %lhs[%zero, %zero, %zero]
      view offset: [0], sizes: [1, 4, 8], strides: [32, 8, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<1x4x8xf32>
  %b = nv_tensor_ir.load %rhs[%zero, %zero, %zero]
      view offset: [0], sizes: [1, 8, 6], strides: [48, 6, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<1x8x6xf32>
  %product = nv_tensor_ir.matmul(%a, %b) {layout = #matmul}
      : (tensor<1x4x8xf32>, tensor<1x8x6xf32>) -> tensor<1x4x6xf32>
  %transposed = nv_tensor_ir.transpose %product permutation = [0, 2, 1]
      {layout = #matmul} : tensor<1x4x6xf32> -> tensor<1x6x4xf32>
  %result = nv_tensor_ir.broadcast %transposed {layout = #matmul}
      : tensor<1x6x4xf32> -> tensor<3x6x4xf32>
  nv_tensor_ir.store %result, %output[%zero, %zero, %zero]
      view offset: [0], sizes: [3, 6, 4], strides: [24, 4, 1], alignment: 4
      : tensor<3x6x4xf32>, !ptr.ptr<#ptr.generic_space>
  return
}
