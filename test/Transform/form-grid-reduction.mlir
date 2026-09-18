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
//       CHECK: nv_tensor_ir.store %[[REDUCED]]#0
//   CHECK-NOT: nv_tensor_ir.reduce_ud
//       CHECK: nv_tensor_ir.store %[[REDUCED]]#1
//   CHECK-NOT: nv_tensor_ir.reduce_ud
//       CHECK: return
#reduction = #nv_tensor_ir.reduction_source<"(4,1,(8)):(8,0,(1))",
    #nv_tensor_ir.composite_source<
      #nv_tensor_ir.tensor_source<0, 0, "(4,8):(8,1)">,
      #nv_tensor_ir.tensor_source<1, 0, "(4,8):(8,1)">>>
func.func @multi_result_reduction(%a: !ptr.ptr<#ptr.generic_space>,
    %b: !ptr.ptr<#ptr.generic_space>, %out0: !ptr.ptr<#ptr.generic_space>,
    %out1: !ptr.ptr<#ptr.generic_space>) attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #reduction,
  result_layouts = [#reduction, #reduction],
  result_views = [#nv_tensor_ir.tensor_source<2, 0, "(4,1):(1,0)">,
                  #nv_tensor_ir.tensor_source<3, 0, "(4,1):(1,0)">],
  tile_size = array<i32: 2, 1>
} {
  %zero = arith.constant 0 : index
  %a_tile = nv_tensor_ir.load %a[%zero, %zero]
      view offset: [0], sizes: [4, 8], strides: [8, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<4x8xf32>
  %b_tile = nv_tensor_ir.load %b[%zero, %zero]
      view offset: [0], sizes: [4, 8], strides: [8, 1], alignment: 2
      : !ptr.ptr<#ptr.generic_space> to tensor<4x8xf16>
  %r0, %r1 = nv_tensor_ir.reduce_ud(%a_tile, %b_tile)
      <dimensions = [1], identity = [0.0 : f32, 0.0 : f16]>
      (%acc0: f32, %acc1: f16, %v0: f32, %v1: f16) {
    %sum0 = arith.addf %acc0, %v0 : f32
    %sum1 = arith.addf %acc1, %v1 : f16
    nv_tensor_ir.yield %sum0, %sum1 : f32, f16
  } {layout = #reduction}
      : tensor<4x8xf32>, tensor<4x8xf16> -> tensor<4x1xf32>, tensor<4x1xf16>
  nv_tensor_ir.store %r0, %out0[%zero, %zero]
      view offset: [0], sizes: [4, 1], strides: [1, 0], alignment: 4
      : tensor<4x1xf32>, !ptr.ptr<#ptr.generic_space>
  nv_tensor_ir.store %r1, %out1[%zero, %zero]
      view offset: [0], sizes: [4, 1], strides: [1, 0], alignment: 2
      : tensor<4x1xf16>, !ptr.ptr<#ptr.generic_space>
  return
}
