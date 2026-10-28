// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid))' | FileCheck %s
// RUN: tensor_ir-opt %s -layout-propagation-pipeline | FileCheck %s --check-prefix=LAYOUT

// The transposed bias keeps the common iteration space split into (2,4).
// Reshaping the normalized reduction result into that space must preserve
// its two contracting axes; merging them produces a hierarchical load stride.
// CHECK-LABEL: func.func @reduction_reshape_dispatch
// CHECK: nv_tensor_ir.load
// CHECK-SAME: memref<2x4x4x4x{{.*}}, strided<[16, 4, 32, 1]
// CHECK-SAME: tensor<1x2x4x4xf32>
// CHECK: nv_tensor_ir.reduce
// CHECK-SAME: dimensions = [2, 3]
// CHECK: nv_tensor_ir.load
// CHECK-SAME: memref<2x4x{{.*}}, strided<[1, 2]
// CHECK: nv_tensor_ir.add
// LAYOUT-LABEL: entry @reduction_reshape(
// LAYOUT: reduce

nv_tensor_ir.graph @reduction_reshape(%seed: tensor<4x8x4xf32>, %bias: tensor<4x2xf32>) -> tensor<2x4xf32> attributes {tile_size = array<i32: 1, 2>} {
  %r = reduce(%seed) <dimensions = [0, 2], reduction_mode = <add>> : tensor<4x8x4xf32> -> tensor<1x8x1xf32>
  %v = reshape %r : tensor<1x8x1xf32> -> tensor<2x4xf32>
  %t = transpose %bias permutation = [1, 0] : tensor<4x2xf32> -> tensor<2x4xf32>
  %out = add %v, %t : tensor<2x4xf32>
  results %out : tensor<2x4xf32>
}
