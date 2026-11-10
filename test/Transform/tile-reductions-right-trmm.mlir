// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{reduction-tile-size=128}))' --split-input-file | FileCheck %s
// RUN: tensor_ir-opt %s --layout-propagation-pipeline --split-input-file | FileCheck %s --check-prefix=LOWERED

// Right upper means B[k,n] is zero when k > n: restrict the upper bound.
// BN=16 differs from both BM=32 and BK=128.
// CHECK-LABEL: func.func @right_upper_dispatch
// CHECK: %[[N:.*]] = arith.remui
// CHECK: %[[SCALE:.*]] = arith.constant 16 : index
// CHECK: arith.muli %[[N]], %[[SCALE]]
// CHECK: nv_tensor_ir.constant dense<0.000000e+00>
// CHECK: %[[END:.*]] = arith.addi
// CHECK: %[[LAST:.*]] = arith.ceildivui %[[END]],
// CHECK: %[[UB:.*]] = arith.minui %[[LAST]],
// CHECK: scf.for {{.*}} to %[[UB]]
// CHECK: nv_tensor_ir.matmul
// CHECK-NOT: trmm_tile_info
// LOWERED-LABEL: entry @right_upper
// LOWERED: mmaf
nv_tensor_ir.graph @right_upper(%a: tensor<96x513xf32>, %b: tensor<513x513xf32>)
    -> tensor<96x513xf32> attributes {tile_size = array<i32: 32, 16>} {
  %c = matmul(%a, %b) <right_triangle = upper>
      : (tensor<96x513xf32>, tensor<513x513xf32>) -> tensor<96x513xf32>
  results %c : tensor<96x513xf32>
}

// -----

// A single K tile uses the dense fast path, even for wholly zero column tiles.
// CHECK-LABEL: func.func @right_lower_single_k_dispatch
// CHECK: scf.forall
// CHECK-NOT: scf.for %
// CHECK-NOT: trmm_tile_info
// CHECK: nv_tensor_ir.matmul
// CHECK-NOT: right_triangle
// CHECK-NOT: scf.for %
// LOWERED-LABEL: entry @right_lower_single_k
// LOWERED-NOT: for %
// LOWERED: mmaf
nv_tensor_ir.graph @right_lower_single_k(
    %a: tensor<96x64xf32>, %b: tensor<64x256xf32>)
    -> tensor<96x256xf32> attributes {tile_size = array<i32: 32, 16>} {
  %c = matmul(%a, %b) <right_triangle = lower>
      : (tensor<96x64xf32>, tensor<64x256xf32>) -> tensor<96x256xf32>
  results %c : tensor<96x256xf32>
}

// -----

// Upper on both sides intersects k >= m with k <= n. Clamp disjoint
// intervals to an empty loop rather than constructing an inverted range.
// CHECK-LABEL: func.func @both_upper_dispatch
// CHECK: nv_tensor_ir.constant dense<0.000000e+00>
// CHECK: %[[FIRST:.*]] = arith.divui
// CHECK: %[[BEGIN:.*]] = arith.minui %[[FIRST]],
// CHECK: arith.ceildivui
// CHECK: %[[UB:.*]] = arith.minui
// CHECK: %[[LB:.*]] = arith.minui %[[BEGIN]], %[[UB]]
// CHECK: scf.for {{.*}} = %[[LB]] to %[[UB]]
// CHECK: nv_tensor_ir.matmul
// LOWERED-LABEL: entry @both_upper
// LOWERED: mmaf
nv_tensor_ir.graph @both_upper(
    %a: tensor<2x512x512xf32>, %b: tensor<2x512x512xf32>)
    -> tensor<2x512x512xf32> attributes {tile_size = array<i32: 1, 32, 16>} {
  %c = matmul(%a, %b) <left_triangle = upper, right_triangle = upper>
      : (tensor<2x512x512xf32>, tensor<2x512x512xf32>) -> tensor<2x512x512xf32>
  results %c : tensor<2x512x512xf32>
}

// -----

// Upper LHS and lower RHS both restrict the lower bound; take their max.
// CHECK-LABEL: func.func @both_lower_bounds_dispatch
// CHECK: nv_tensor_ir.constant dense<0.000000e+00>
// CHECK: %[[L:.*]] = arith.minui
// CHECK: %[[R:.*]] = arith.minui
// CHECK: %[[MAX:.*]] = arith.maxui %[[L]], %[[R]]
// CHECK: %[[LB:.*]] = arith.minui %[[MAX]], %[[UB:.*]] : index
// CHECK: scf.for {{.*}} = %[[LB]] to %[[UB]]
// LOWERED-LABEL: entry @both_lower_bounds
// LOWERED: mmaf
nv_tensor_ir.graph @both_lower_bounds(%a: tensor<512x512xf32>, %b: tensor<512x512xf32>)
    -> tensor<512x512xf32> attributes {tile_size = array<i32: 32, 16>} {
  %c = matmul(%a, %b) <left_triangle = upper, right_triangle = lower>
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  results %c : tensor<512x512xf32>
}

// -----

// Lower LHS and upper RHS both restrict the upper bound; take their min.
// CHECK-LABEL: func.func @both_upper_bounds_dispatch
// CHECK: nv_tensor_ir.constant dense<0.000000e+00>
// CHECK: %[[L:.*]] = arith.minui
// CHECK: %[[R:.*]] = arith.ceildivui
// CHECK: %[[UB:.*]] = arith.minui %[[R]], %[[L]]
// CHECK: %[[LB:.*]] = arith.minui {{.*}}, %[[UB]]
// CHECK: scf.for {{.*}} = %[[LB]] to %[[UB]]
// LOWERED-LABEL: entry @both_upper_bounds
// LOWERED: mmaf
nv_tensor_ir.graph @both_upper_bounds(%a: tensor<512x512xf32>, %b: tensor<512x512xf32>)
    -> tensor<512x512xf32> attributes {tile_size = array<i32: 32, 16>} {
  %c = matmul(%a, %b) <left_triangle = lower, right_triangle = upper>
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  results %c : tensor<512x512xf32>
}

// -----

// Lower on both sides intersects k <= m with k >= n.
// CHECK-LABEL: func.func @both_lower_dispatch
// CHECK: nv_tensor_ir.constant dense<0.000000e+00>
// CHECK: %[[UB:.*]] = arith.minui
// CHECK: %[[BEGIN:.*]] = arith.minui
// CHECK: %[[LB:.*]] = arith.minui %[[BEGIN]], %[[UB]]
// CHECK: scf.for {{.*}} = %[[LB]] to %[[UB]]
// LOWERED-LABEL: entry @both_lower
// LOWERED: mmaf
nv_tensor_ir.graph @both_lower(%a: tensor<512x512xf32>, %b: tensor<512x512xf32>)
    -> tensor<512x512xf32> attributes {tile_size = array<i32: 32, 16>} {
  %c = matmul(%a, %b) <left_triangle = lower, right_triangle = lower>
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  results %c : tensor<512x512xf32>
}
