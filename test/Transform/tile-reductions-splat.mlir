// RUN: tensor_ir-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(tir-tile-reductions{reduction-tile-size=128}))' | FileCheck %s

// CHECK-LABEL: func.func @splat_reduction(
// CHECK-SAME: %[[VALUE:[^:]+]]: f32
func.func @splat_reduction(%value: f32) -> tensor<1xf32> {
  %splat = nv_tensor_ir.splat %value : tensor<1024xf32>
  // CHECK: scf.for
  // CHECK:   %[[SPLAT:.*]] = nv_tensor_ir.splat %[[VALUE]] : tensor<128xf32>
  // CHECK:   nv_tensor_ir.add {{.*}}, %[[SPLAT]] : tensor<128xf32>
  // CHECK:   scf.yield
  // CHECK: nv_tensor_ir.reduce
  %result = nv_tensor_ir.reduce(%splat)
      <dimensions = [0], reduction_mode = <add>>
      : tensor<1024xf32> -> tensor<1xf32>
  return %result : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @splat_matmul(
// CHECK-SAME: %[[VALUE:[^:]+]]: f32
func.func @splat_matmul(%value: f32, %input: memref<1024x16xf32>) -> tensor<16x16xf32> {
  %splat = nv_tensor_ir.splat %value : tensor<16x1024xf32>
  %load = nv_tensor_ir.load %input tile_indices [0, 0], alignment 4
      : memref<1024x16xf32> to tensor<1024x16xf32>
  // CHECK: scf.for
  // CHECK:   %[[SPLAT:.*]] = nv_tensor_ir.splat %[[VALUE]] : tensor<16x128xf32>
  // CHECK:   %[[LOAD:.*]] = nv_tensor_ir.load {{.*}} to tensor<128x16xf32>
  // CHECK:   nv_tensor_ir.matmul(%[[SPLAT]], %[[LOAD]]) accum
  // CHECK:   scf.yield
  %result = nv_tensor_ir.matmul(%splat, %load) {
    nv_tensor_ir.contraction_shape = array<i64: 1024>,
    nv_tensor_ir.lhs_source_shape = array<i64: 16, 1024>,
    nv_tensor_ir.rhs_source_shape = array<i64: 1024, 16>,
    nv_tensor_ir.lhs_contracting_dimensions = array<i64: 1>,
    nv_tensor_ir.rhs_contracting_dimensions = array<i64: 0>
  } : (tensor<16x1024xf32>, tensor<1024x16xf32>) -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}
