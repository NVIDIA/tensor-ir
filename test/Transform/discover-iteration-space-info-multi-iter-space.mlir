// RUN: tensor_ir-opt -discover-iteration-space-info -mlir-print-local-scope -split-input-file %s | FileCheck %s

// Tests for multi-iteration space graphs with transitions.

// -----

// CHECK-LABEL: nv_tensor_ir.graph @broadcast_establishes_domain
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
nv_tensor_ir.graph @broadcast_establishes_domain(
    %inputA: tensor<16x1x64xf32>,
    %inputB: tensor<16x32x64xf32>
) -> tensor<16x32x64xf32> {

  // When a broadcast expands a unit dimension whose domain is Undef (was never
  // defined), it establishes the domain for the first time. No transition. 

  // CHECK: broadcast
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %0 = broadcast %inputA : tensor<16x1x64xf32> -> tensor<16x32x64xf32>

  // CHECK: add
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %1 = add %0, %inputB : tensor<16x32x64xf32>
  results %1 : tensor<16x32x64xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @reduce_broadcast_transition
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 1>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
nv_tensor_ir.graph @reduce_broadcast_transition(
    %input: tensor<16x32x64xf32>,
    %bias: tensor<16x48x64xf32>
) -> tensor<16x48x64xf32> {
  // CHECK: reduce
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %reduced = reduce(%input) <dimensions = [1], reduction_mode = <add>>
      : tensor<16x32x64xf32> -> tensor<16x1x64xf32>

  // When a broadcast expands a unit dimension whose domain is Def (was defined,
  // then contracted), it redefines the domain. This creates a transition.

  // CHECK: broadcast
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %broadcast = broadcast %reduced : tensor<16x1x64xf32> -> tensor<16x48x64xf32>

  // CHECK: add
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %result = add %broadcast, %bias : tensor<16x48x64xf32>
  results %result : tensor<16x48x64xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @broadcast_reduce_broadcast
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 1>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
nv_tensor_ir.graph @broadcast_reduce_broadcast(
    %inputA: tensor<16x1x64xf32>,
    %inputC: tensor<16x32x64xf32>,
    %inputB: tensor<16x48x64xf32>
) -> (tensor<16x48x64xf32>) {
  // CHECK: broadcast
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %broadcast1 = broadcast %inputA : tensor<16x1x64xf32> -> tensor<16x32x64xf32>

  // CHECK: mul
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %mul_result = mul %broadcast1, %inputC : tensor<16x32x64xf32>

  // CHECK: reduce
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %reduced = reduce(%mul_result) <dimensions = [1], reduction_mode = <add>>
      : tensor<16x32x64xf32> -> tensor<16x1x64xf32>

  // First iteration space transition: 32 -> 1 -> 48.

  // CHECK: broadcast
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %broadcast2 = broadcast %reduced : tensor<16x1x64xf32> -> tensor<16x48x64xf32>

  // CHECK: add
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %result = add %broadcast2, %inputB : tensor<16x48x64xf32>
  results %result : tensor<16x48x64xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @two_transitions
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 1>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 2>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
nv_tensor_ir.graph @two_transitions(
    %input: tensor<16x32x64xf32>,
    %bias48: tensor<16x48x64xf32>,
    %bias64: tensor<16x64x64xf32>
) -> tensor<16x64x64xf32> {
  // CHECK: reduce
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %reduced1 = reduce(%input) <dimensions = [1], reduction_mode = <add>>
      : tensor<16x32x64xf32> -> tensor<16x1x64xf32>

  // First iteration space transition: 32 -> 1 -> 48.

  // CHECK: broadcast
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %bcast1 = broadcast %reduced1 : tensor<16x1x64xf32> -> tensor<16x48x64xf32>

  // CHECK: add
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %add1 = add %bcast1, %bias48 : tensor<16x48x64xf32>

  // CHECK: reduce
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %reduced2 = reduce(%add1) <dimensions = [1], reduction_mode = <add>>
      : tensor<16x48x64xf32> -> tensor<16x1x64xf32>

  // Second iteration space transition: 48 -> 1 -> 64.

  // CHECK: broadcast
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 2 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %bcast2 = broadcast %reduced2 : tensor<16x1x64xf32> -> tensor<16x64x64xf32>

  // CHECK: add
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 2 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %result = add %bcast2, %bias64 : tensor<16x64x64xf32>
  results %result : tensor<16x64x64xf32>
}

// -----

// Test that 32 -> 1 -> 32 (same size) still creates a transition because
// the dimension domain was defined (Def) before being contracted.
// CHECK-LABEL: nv_tensor_ir.graph @same_size_transition
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 1>
// CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
nv_tensor_ir.graph @same_size_transition(
    %input: tensor<16x32x64xf32>,
    %bias: tensor<16x32x64xf32>
) -> tensor<16x32x64xf32> {
  // CHECK: reduce
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %reduced = reduce(%input) <dimensions = [1], reduction_mode = <add>>
      : tensor<16x32x64xf32> -> tensor<16x1x64xf32>

  // Even though we broadcast back to the same size (32), it's a transition
  // because dim 1's domain was defined (Def), then contracted, now redefined.

  // CHECK: broadcast
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %broadcast = broadcast %reduced : tensor<16x1x64xf32> -> tensor<16x32x64xf32>

  // CHECK: add
  // CHECK-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  // CHECK-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  %result = add %broadcast, %bias : tensor<16x32x64xf32>
  results %result : tensor<16x32x64xf32>
}

// -----

// RMSNorm pattern: reduce + broadcast with an input used in multiple iteration spaces.

// CHECK-LABEL: nv_tensor_ir.graph @rmsnorm
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0, 1>
// CHECK-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
nv_tensor_ir.graph @rmsnorm(
    %input: tensor<64x128xf32>,
    %scale: tensor<1x128xf32>
) -> tensor<64x128xf32> {

  // CHECK: mul
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  %sq = mul %input, %input : tensor<64x128xf32>

  // CHECK: reduce
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  %sum = reduce(%sq) <dimensions = [1], reduction_mode = <add>>
      : tensor<64x128xf32> -> tensor<64x1xf32>

  // CHECK: add
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  %eps = nv_tensor_ir.constant dense<1.0e-05> : tensor<64x1xf32>
  %sum_eps = add %sum, %eps : tensor<64x1xf32>

  // CHECK: rsqrt
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  %inv = rsqrt %sum_eps : tensor<64x1xf32>

  // Transition: broadcast re-expands the reduced dimension.

  // CHECK: broadcast
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  %inv_b = broadcast %inv
      : tensor<64x1xf32> -> tensor<64x128xf32>

  // CHECK: broadcast
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  %scale_b = broadcast %scale
      : tensor<1x128xf32> -> tensor<64x128xf32>

  // CHECK: mul
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 1 : i32
  %norm = mul %input, %inv_b : tensor<64x128xf32>

  // CHECK: mul
  // CHECK-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  %result = mul %norm, %scale_b : tensor<64x128xf32>
  results %result : tensor<64x128xf32>
}
