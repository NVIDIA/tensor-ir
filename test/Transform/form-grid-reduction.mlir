// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' | FileCheck %s

// Both results of one user-defined reduction must share the tiled reduction.
// CHECK-LABEL: func.func @multi_result_reduction
//       CHECK: scf.forall
//       CHECK: %[[A:.*]] = nv_tensor_ir.load {{.*}} to tensor<2x8xf32>
//       CHECK: %[[B:.*]] = nv_tensor_ir.load {{.*}} to tensor<2x8xf16>
//       CHECK: %[[REDUCED:.*]]:2 = nv_tensor_ir.reduce_ud(%[[A]], %[[B]])
//  CHECK-SAME: dimensions = [1]
//       CHECK: nv_tensor_ir.yield
//       CHECK: nv_tensor_ir.reduction_extent = 8 : i64
//  CHECK-SAME: tensor<2x8xf32>, tensor<2x8xf16> -> tensor<2x1xf32>, tensor<2x1xf16>
//   CHECK-NOT: nv_tensor_ir.reduce_ud
//   CHECK-NOT: nv_tensor_ir.slice
//       CHECK: nv_tensor_ir.store %[[REDUCED]]#0
//  CHECK-SAME: tensor<2x1xf32>, memref<4x1xf32, strided<[1, 2]>>
//   CHECK-NOT: nv_tensor_ir.reduce_ud
//   CHECK-NOT: nv_tensor_ir.slice
//       CHECK: nv_tensor_ir.store %[[REDUCED]]#1
//  CHECK-SAME: tensor<2x1xf16>, memref<4x1xf16, strided<[1, 2]>>
//   CHECK-NOT: nv_tensor_ir.reduce_ud
//       CHECK: return
func.func @multi_result_reduction(%a: memref<4x8xf32>,
    %b: memref<4x8xf16>,
    %out0: memref<4x1xf32, strided<[1, 0]>>,
    %out1: memref<4x1xf16, strided<[1, 0]>>) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 1>
} {
  %a_tile = bufferization.to_tensor %a restrict
      : memref<4x8xf32> to tensor<4x8xf32>
  %b_tile = bufferization.to_tensor %b restrict
      : memref<4x8xf16> to tensor<4x8xf16>
  %r0, %r1 = nv_tensor_ir.reduce_ud(%a_tile, %b_tile)
      <dimensions = [1], identity = [0.0 : f32, 0.0 : f16]>
      (%acc0: f32, %acc1: f16, %v0: f32, %v1: f16) {
    %sum0 = arith.addf %acc0, %v0 : f32
    %sum1 = arith.addf %acc1, %v1 : f16
    nv_tensor_ir.yield %sum0, %sum1 : f32, f16
  }
      : tensor<4x8xf32>, tensor<4x8xf16> -> tensor<4x1xf32>, tensor<4x1xf16>
  bufferization.materialize_in_destination %r0 in restrict writable %out0
      : (tensor<4x1xf32>, memref<4x1xf32, strided<[1, 0]>>)
        -> ()
  bufferization.materialize_in_destination %r1 in restrict writable %out1
      : (tensor<4x1xf16>, memref<4x1xf16, strided<[1, 0]>>)
        -> ()
  return
}
