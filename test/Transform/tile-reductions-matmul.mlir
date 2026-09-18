// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,layout-propagation-annotation,layout-propagation-normalization,tile-selection,graph-splitting),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{reduction-tile-size=128}),outline-tensor-ir-kernel,convert-tensor-to-cuda-tile{codegen-strategy=layout_propagation})' --split-input-file | FileCheck %s --check-prefix=LOWERED
// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,layout-propagation-annotation,layout-propagation-normalization,tile-selection,graph-splitting),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{reduction-tile-size=128}))' --split-input-file | FileCheck %s

// A contracting domain that fits in one tile stays loop-free, but grid
// formation's temporary matmul metadata is still consumed.
// CHECK-LABEL: func.func @small_matmul_dispatch
// CHECK: scf.forall
// CHECK-NOT: scf.for %
// CHECK: %[[LHS:.*]] = nv_tensor_ir.load {{.*}} to tensor<32x64xf32>
// CHECK: %[[RHS:.*]] = nv_tensor_ir.load {{.*}} to tensor<64x16xf32>
// CHECK: %[[RESULT:.*]] = nv_tensor_ir.matmul(%[[LHS]], %[[RHS]])
// CHECK-NOT: nv_tensor_ir.contraction_shape
// CHECK: nv_tensor_ir.store %[[RESULT]]
// LOWERED-LABEL: entry @small_matmul
// LOWERED: mmaf {{.*}} : tile<32x64xf32>, tile<64x16xf32>, tile<32x16xf32>
nv_tensor_ir.graph @small_matmul(
    %lhs: tensor<64x64xf32>, %rhs: tensor<64x48xf32>)
    -> tensor<64x48xf32> attributes {tile_size = array<i32: 32, 16>} {
  %result = matmul(%lhs, %rhs)
      : (tensor<64x64xf32>, tensor<64x48xf32>) -> tensor<64x48xf32>
  results %result : tensor<64x48xf32>
}

// -----

// A large K domain becomes an accumulator-carrying loop. The loop induction
// variable indexes the contracting dimension of both source views.
// CHECK-LABEL: func.func @large_matmul_dispatch
// CHECK: %[[INIT:.*]] = nv_tensor_ir.constant dense<0.000000e+00> : tensor<32x16xf32>
// CHECK: %[[LOOP:.*]] = scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]])
// CHECK: %[[LHS:.*]] = nv_tensor_ir.load {{.*}}[%{{.*}}, %[[K]]] {{.*}} to tensor<32x128xf32>
// CHECK: %[[RHS:.*]] = nv_tensor_ir.load {{.*}}[%[[K]], %{{.*}}] {{.*}} to tensor<128x16xf32>
// CHECK: %[[PARTIAL:.*]] = nv_tensor_ir.matmul(%[[LHS]], %[[RHS]]) accum(%[[ACC]] : tensor<32x16xf32>)
// CHECK: scf.yield %[[PARTIAL]] : tensor<32x16xf32>
// CHECK: nv_tensor_ir.store %[[LOOP]]
// LOWERED-LABEL: entry @large_matmul
// LOWERED: for %{{.*}} in
// LOWERED: mmaf {{.*}} : tile<32x128xf32>, tile<128x16xf32>, tile<32x16xf32>
nv_tensor_ir.graph @large_matmul(
    %lhs: tensor<64x1024xf32>, %rhs: tensor<1024x48xf32>)
    -> tensor<64x48xf32> attributes {tile_size = array<i32: 32, 16>} {
  %result = matmul(%lhs, %rhs)
      : (tensor<64x1024xf32>, tensor<1024x48xf32>) -> tensor<64x48xf32>
  results %result : tensor<64x48xf32>
}

// -----

// Fragmented contracting dimensions are strip-mined independently and then
// reshaped to the conventional [M, K] and [K, N] MMA operand contract.
// CHECK-LABEL: func.func @fragmented_matmul_dispatch
// CHECK: %[[OUTER:.*]] = scf.for %[[K0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[OUT_ACC:.*]] = %{{.*}})
// CHECK: %[[INNER:.*]] = scf.for %[[K1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[IN_ACC:.*]] = %[[OUT_ACC]])
// CHECK: %[[LHS_TILE:.*]] = nv_tensor_ir.load {{.*}}[%{{.*}}, %[[K0]], %[[K1]]] {{.*}} to tensor<32x16x8xf32>
// CHECK: %[[RHS_TILE:.*]] = nv_tensor_ir.load {{.*}}[%[[K0]], %[[K1]], %{{.*}}] {{.*}} to tensor<16x8x16xf32>
// CHECK: %[[LHS:.*]] = nv_tensor_ir.reshape %[[LHS_TILE]] : tensor<32x16x8xf32> -> tensor<32x128xf32>
// CHECK: %[[RHS:.*]] = nv_tensor_ir.reshape %[[RHS_TILE]] : tensor<16x8x16xf32> -> tensor<128x16xf32>
// CHECK: %[[PARTIAL:.*]] = nv_tensor_ir.matmul(%[[LHS]], %[[RHS]]) accum(%[[IN_ACC]] : tensor<32x16xf32>)
// CHECK: scf.yield %[[PARTIAL]] : tensor<32x16xf32>
// CHECK: scf.yield %[[INNER]] : tensor<32x16xf32>
// CHECK: nv_tensor_ir.store %[[OUTER]]
// LOWERED-LABEL: entry @fragmented_matmul
// LOWERED: for %{{.*}} in
// LOWERED: for %{{.*}} in
// LOWERED: mmaf {{.*}} : tile<32x128xf32>, tile<128x16xf32>, tile<32x16xf32>
nv_tensor_ir.graph @fragmented_matmul(
    %lhs: tensor<64x32x32xf32> {nv_tensor_ir.stride = "(1024,32,1)"},
    %rhs: tensor<1024x48xf32>)
    -> tensor<64x48xf32> attributes {tile_size = array<i32: 32, 16>} {
  %lhs_transposed = transpose %lhs permutation = [2, 1, 0]
      : tensor<64x32x32xf32> -> tensor<32x32x64xf32>
  %lhs_flat = reshape %lhs_transposed
      : tensor<32x32x64xf32> -> tensor<1024x64xf32>
  %lhs_matrix = transpose %lhs_flat permutation = [1, 0]
      : tensor<1024x64xf32> -> tensor<64x1024xf32>
  %result = matmul(%lhs_matrix, %rhs)
      : (tensor<64x1024xf32>, tensor<1024x48xf32>) -> tensor<64x48xf32>
  results %result : tensor<64x48xf32>
}
