// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{reduction-tile-size=128}))' --split-input-file | FileCheck %s
// RUN: tensor_ir-opt %s --layout-propagation-pipeline --split-input-file | FileCheck %s --check-prefix=LOWERED

// M tiles have 32 rows and K tiles have 128 columns. The row origin,
// rather than the M tile index, determines the first K tile.
// CHECK-LABEL: func.func @upper_dispatch
// CHECK: nv_tensor_ir.constant dense<0.000000e+00>
// CHECK: %[[FIRST:.*]] = arith.divui
// CHECK: %[[BEGIN:.*]] = arith.minui %[[FIRST]], %[[END:.*]] : index
// CHECK: scf.for %[[K:.*]] = %[[BEGIN]] to %[[END]]
// CHECK: nv_tensor_ir.load {{.*}}[%{{.*}}, %[[K]]]
// CHECK: nv_tensor_ir.matmul
// CHECK-NOT: trmm_tile_info
// CHECK-NOT: left_triangle = #nv_tensor_ir.triangle<upper>
// LOWERED-LABEL: entry @upper
// LOWERED: for %{{.*}} in
// LOWERED: mmaf
nv_tensor_ir.graph @upper(%a: tensor<512x512xf32>, %b: tensor<512x48xf32>)
    -> tensor<512x48xf32> attributes {tile_size = array<i32: 32, 16>} {
  %c = matmul(%a, %b) <left_triangle = upper>
      : (tensor<512x512xf32>, tensor<512x48xf32>) -> tensor<512x48xf32>
  results %c : tensor<512x48xf32>
}

// -----

// Partial M and K tails must keep the entire tile touching the diagonal.
// CHECK-LABEL: func.func @lower_dispatch
// CHECK: nv_tensor_ir.constant dense<0.000000e+00>
// CHECK: %[[ROWEND:.*]] = arith.addi
// CHECK: %[[LAST:.*]] = arith.ceildivui %[[ROWEND]],
// CHECK: %[[END:.*]] = arith.minui %[[LAST]],
// CHECK: scf.for {{.*}} to %[[END]]
// CHECK: nv_tensor_ir.matmul
// CHECK-NOT: trmm_tile_info
// CHECK-NOT: left_triangle = #nv_tensor_ir.triangle<lower>
// LOWERED-LABEL: entry @lower
// LOWERED: for %{{.*}} in
// LOWERED: mmaf
nv_tensor_ir.graph @lower(%a: tensor<513x513xf32>, %b: tensor<513x48xf32>)
    -> tensor<513x48xf32> attributes {tile_size = array<i32: 32, 16>} {
  %c = matmul(%a, %b) <left_triangle = lower>
      : (tensor<513x513xf32>, tensor<513x48xf32>) -> tensor<513x48xf32>
  results %c : tensor<513x48xf32>
}

// -----

// A single K tile uses the dense fast path, even for wholly zero row tiles.
// CHECK-LABEL: func.func @upper_single_k_dispatch
// CHECK: scf.forall
// CHECK-NOT: scf.for %
// CHECK-NOT: trmm_tile_info
// CHECK: nv_tensor_ir.matmul
// CHECK-NOT: left_triangle
// CHECK-NOT: scf.for %
// LOWERED-LABEL: entry @upper_single_k
// LOWERED-NOT: for %
// LOWERED: mmaf
nv_tensor_ir.graph @upper_single_k(%a: tensor<256x64xf32>, %b: tensor<64x48xf32>)
    -> tensor<256x48xf32> attributes {tile_size = array<i32: 32, 16>} {
  %c = matmul(%a, %b) <left_triangle = upper>
      : (tensor<256x64xf32>, tensor<64x48xf32>) -> tensor<256x48xf32>
  results %c : tensor<256x48xf32>
}

// -----

// Omitted triangle attributes preserve the dense loop range.
// CHECK-LABEL: func.func @dense_dispatch
// CHECK-NOT: arith.minui
// CHECK: scf.for
// CHECK: nv_tensor_ir.matmul
// LOWERED-LABEL: entry @dense
// LOWERED: mmaf
nv_tensor_ir.graph @dense(%a: tensor<512x512xf32>, %b: tensor<512x48xf32>)
    -> tensor<512x48xf32> attributes {tile_size = array<i32: 32, 16>} {
  %c = matmul(%a, %b)
      : (tensor<512x512xf32>, tensor<512x48xf32>) -> tensor<512x48xf32>
  results %c : tensor<512x48xf32>
}

// -----

// Batch coordinates must not contribute to the logical M row origin.
// CHECK-LABEL: func.func @batched_upper_dispatch
// CHECK: nv_tensor_ir.constant dense<0.000000e+00>
// CHECK: %[[FIRST:.*]] = arith.divui
// CHECK: %[[BEGIN:.*]] = arith.minui %[[FIRST]], %[[END:.*]] : index
// CHECK: scf.for {{.*}} = %[[BEGIN]] to %[[END]]
// CHECK: nv_tensor_ir.matmul
// LOWERED-LABEL: entry @batched_upper
// LOWERED: mmaf
nv_tensor_ir.graph @batched_upper(
    %a: tensor<2x257x257xf32>, %b: tensor<2x257x48xf32>)
    -> tensor<2x257x48xf32> attributes {tile_size = array<i32: 1, 32, 16>} {
  %c = matmul(%a, %b) <left_triangle = upper>
      : (tensor<2x257x257xf32>, tensor<2x257x48xf32>) -> tensor<2x257x48xf32>
  results %c : tensor<2x257x48xf32>
}
