// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @nonunit
func.func @nonunit(%input: memref<4x6x8xf32>,
    %output: memref<4x6x8xf32, strided<[0, 2, 0]>> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.allow_tma = false}) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 3, 4>
} {
  %value = bufferization.to_tensor %input restrict : memref<4x6x8xf32> to tensor<4x6x8xf32>
  bufferization.materialize_in_destination %value in restrict writable %output
      : (tensor<4x6x8xf32>, memref<4x6x8xf32, strided<[0, 2, 0]>>) -> ()
  return
}
// CHECK: nv_tensor_ir.slice {{.*}} starts = [0, 0, 0] limits = [1, 3, 1]
// CHECK-NOT: nv_tensor_ir.reshape
// CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [1, 6, 1], strides: [2, 2, 2]
// CHECK: nv_tensor_ir.store {{.*}} tile_indices [0, %{{.*}}, 0], alignment 16 {allow_tma = false}
// CHECK-SAME: tensor<1x3x1xf32>, memref<1x6x1xf32, strided<[2, 2, 2]>>

// -----

// CHECK-LABEL: func.func @scalar
func.func @scalar(%input: memref<4x8xf32, strided<[16, 1]>>,
    %output: memref<4x8xf32, strided<[0, 0]>> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.allow_tma = false}) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 4>
} {
  %value = bufferization.to_tensor %input restrict : memref<4x8xf32, strided<[16, 1]>> to tensor<4x8xf32>
  bufferization.materialize_in_destination %value in restrict writable %output
      : (tensor<4x8xf32>, memref<4x8xf32, strided<[0, 0]>>) -> ()
  return
}
// CHECK: nv_tensor_ir.slice {{.*}} starts = [0, 0] limits = [1, 1]
// CHECK-NOT: nv_tensor_ir.reshape
// CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [1, 1], strides: [2, 2]
// CHECK: nv_tensor_ir.store {{.*}} tile_indices [0, 0], alignment 16 {allow_tma = false}
// CHECK-SAME: tensor<1x1xf32>, memref<1x1xf32, strided<[2, 2]>>
