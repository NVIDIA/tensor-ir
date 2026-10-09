// RUN: tensor_ir-opt %s -split-input-file -mlir-print-local-scope | FileCheck %s
// RUN: tensor_ir-opt %s -split-input-file | tensor_ir-opt -split-input-file -mlir-print-local-scope | FileCheck %s
// RUN: tensor_ir-opt %s -split-input-file -mlir-print-op-generic | tensor_ir-opt -split-input-file -mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: nv_tensor_ir.graph @matmul
nv_tensor_ir.graph @matmul(
    %a: tensor<2x5xf16>, %b: tensor<5x7xf16>) -> tensor<2x7xf32> {
  // CHECK: contraction(%{{[^, )]+}}, %{{[^, )]+}})
  // CHECK-SAME: maps (d0, d1, d2) -> (d0, d2), (d2, d1), (d0, d1)
  // CHECK-SAME: {test.annotation = "discardable"}
  // CHECK-SAME: : (tensor<2x5xf16>, tensor<5x7xf16>) -> tensor<2x7xf32>
  %result = contraction(%a, %b) maps (m, n, k) -> (m, k), (k, n), (m, n) {test.annotation = "discardable"} : (tensor<2x5xf16>, tensor<5x7xf16>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @permuted_products_and_contractions
nv_tensor_ir.graph @permuted_products_and_contractions(
    %a: tensor<5x2x7x3xf16>, %b: tensor<11x5x13x7xf16>,
    %acc: tensor<3x2x13x11xf32>) -> tensor<3x2x13x11xf32> {
  // CHECK: contraction(%{{[^, )]+}}, %{{[^, )]+}}) accum(%{{[^ ]+}} : tensor<3x2x13x11xf32>)
  // CHECK-SAME: maps (d0, d1, d2, d3, d4, d5) -> (d5, d1, d4, d0), (d3, d5, d2, d4), (d0, d1, d2, d3)
  // CHECK-SAME: : (tensor<5x2x7x3xf16>, tensor<11x5x13x7xf16>) -> tensor<3x2x13x11xf32>
  %result = contraction(%a, %b) accum(%acc : tensor<3x2x13x11xf32>) maps (d0, d1, d2, d3, d4, d5) -> (d5, d1, d4, d0), (d3, d5, d2, d4), (d0, d1, d2, d3) : (tensor<5x2x7x3xf16>, tensor<11x5x13x7xf16>) -> tensor<3x2x13x11xf32>
  results %result : tensor<3x2x13x11xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @unequal_ranks
nv_tensor_ir.graph @unequal_ranks(
    %a: tensor<2x3x5x7xf32>, %b: tensor<5x7x11xf32>) -> tensor<2x3x11xf32> {
  // CHECK: contraction(%{{[^, )]+}}, %{{[^, )]+}})
  // CHECK-SAME: maps (d0, d1, d2, d3, d4) -> (d0, d1, d3, d4), (d3, d4, d2), (d0, d1, d2)
  // CHECK-SAME: : (tensor<2x3x5x7xf32>, tensor<5x7x11xf32>) -> tensor<2x3x11xf32>
  %result = contraction(%a, %b) maps (d0, d1, d2, d3, d4) -> (d0, d1, d3, d4), (d3, d4, d2), (d0, d1, d2) : (tensor<2x3x5x7xf32>, tensor<5x7x11xf32>) -> tensor<2x3x11xf32>
  results %result : tensor<2x3x11xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @permuted_batches
nv_tensor_ir.graph @permuted_batches(
    %a: tensor<3x1x5x2xf32>, %b: tensor<7x11x1x5xf32>) -> tensor<2x7x3x11xf32> {
  // CHECK: contraction(%{{[^, )]+}}, %{{[^, )]+}})
  // CHECK-SAME: maps (d0, d1, d2, d3, d4) -> (d2, d1, d4, d0), (d1, d3, d0, d4), (d0, d1, d2, d3)
  // CHECK-SAME: : (tensor<3x1x5x2xf32>, tensor<7x11x1x5xf32>) -> tensor<2x7x3x11xf32>
  %result = contraction(%a, %b) maps (d0, d1, d2, d3, d4) -> (d2, d1, d4, d0), (d1, d3, d0, d4), (d0, d1, d2, d3) : (tensor<3x1x5x2xf32>, tensor<7x11x1x5xf32>) -> tensor<2x7x3x11xf32>
  results %result : tensor<2x7x3x11xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @permuted_domain
nv_tensor_ir.graph @permuted_domain(
    %a: tensor<5x2xf32>, %b: tensor<7x5xf32>) -> tensor<2x7xf32> {
  // CHECK: contraction(%{{[^, )]+}}, %{{[^, )]+}})
  // CHECK-SAME: maps (d0, d1, d2) -> (d0, d2), (d1, d0), (d2, d1)
  // CHECK-SAME: : (tensor<5x2xf32>, tensor<7x5xf32>) -> tensor<2x7xf32>
  %result = contraction(%a, %b) maps (d0, d1, d2) -> (d0, d2), (d1, d0), (d2, d1) : (tensor<5x2xf32>, tensor<7x5xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @dynamic
nv_tensor_ir.graph @dynamic(
    %a: tensor<?x1x?x5x?xf32>, %b: tensor<2x?x5x7x11xf32>) -> tensor<2x?x?x11xf32> {
  // CHECK: contraction(%{{[^, )]+}}, %{{[^, )]+}})
  // CHECK-SAME: maps (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d5), (d0, d1, d4, d5, d3), (d0, d1, d2, d3)
  // CHECK-SAME: : (tensor<?x1x?x5x?xf32>, tensor<2x?x5x7x11xf32>) -> tensor<2x?x?x11xf32>
  %result = contraction(%a, %b) maps (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d5), (d0, d1, d4, d5, d3), (d0, d1, d2, d3) : (tensor<?x1x?x5x?xf32>, tensor<2x?x5x7x11xf32>) -> tensor<2x?x?x11xf32>
  results %result : tensor<2x?x?x11xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @dot
nv_tensor_ir.graph @dot(
    %a: tensor<5x7xf32>, %b: tensor<7x5xf32>) -> tensor<f32> {
  // CHECK: contraction(%{{[^, )]+}}, %{{[^, )]+}})
  // CHECK-SAME: maps (d0, d1) -> (d1, d0), (d0, d1), ()
  // CHECK-SAME: : (tensor<5x7xf32>, tensor<7x5xf32>) -> tensor<f32>
  %result = contraction(%a, %b) maps (d0, d1) -> (d1, d0), (d0, d1), () : (tensor<5x7xf32>, tensor<7x5xf32>) -> tensor<f32>
  results %result : tensor<f32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @outer
nv_tensor_ir.graph @outer(
    %a: tensor<2x3xf32>, %b: tensor<5x7xf32>) -> tensor<3x2x7x5xf32> {
  // CHECK: contraction(%{{[^, )]+}}, %{{[^, )]+}})
  // CHECK-SAME: maps (d0, d1, d2, d3) -> (d1, d0), (d3, d2), (d0, d1, d2, d3)
  // CHECK-SAME: : (tensor<2x3xf32>, tensor<5x7xf32>) -> tensor<3x2x7x5xf32>
  %result = contraction(%a, %b) maps (d0, d1, d2, d3) -> (d1, d0), (d3, d2), (d0, d1, d2, d3) : (tensor<2x3xf32>, tensor<5x7xf32>) -> tensor<3x2x7x5xf32>
  results %result : tensor<3x2x7x5xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @zero_extent
nv_tensor_ir.graph @zero_extent(
    %a: tensor<2x0x7xf32>, %b: tensor<0x7x3xf32>) -> tensor<2x3xf32> {
  // CHECK: contraction(%{{[^, )]+}}, %{{[^, )]+}})
  // CHECK-SAME: maps (d0, d1, d2, d3) -> (d0, d2, d3), (d2, d3, d1), (d0, d1)
  // CHECK-SAME: : (tensor<2x0x7xf32>, tensor<0x7x3xf32>) -> tensor<2x3xf32>
  %result = contraction(%a, %b) maps (d0, d1, d2, d3) -> (d0, d2, d3), (d2, d3, d1), (d0, d1) : (tensor<2x0x7xf32>, tensor<0x7x3xf32>) -> tensor<2x3xf32>
  results %result : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @scalar
nv_tensor_ir.graph @scalar(
    %a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
  // CHECK: contraction(%{{[^, )]+}}, %{{[^, )]+}})
  // CHECK-SAME: maps () -> (), (), ()
  // CHECK-SAME: : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %result = contraction(%a, %b) maps () -> (), (), () : (tensor<f32>, tensor<f32>) -> tensor<f32>
  results %result : tensor<f32>
}
