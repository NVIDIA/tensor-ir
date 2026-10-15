// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)' \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg,convert-tensor-to-linalg)' \
// RUN:   | tensor_ir-opt --split-input-file --mlir-print-local-scope \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.

// CHECK-LABEL: func.func @matmul(
nv_tensor_ir.graph @matmul(
    %a: tensor<2x5xf16>, %b: tensor<5x7xf16>) -> tensor<2x7xf32> {
  %result = contraction(%a, %b) maps (m, n, k) -> (m, k), (k, n), (m, n) {test.annotation = "discardable"} : (tensor<2x5xf16>, tensor<5x7xf16>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: %[[A:.*]] = arith.extf %{{.*}} : f16 to f32
// CHECK: %[[B:.*]] = arith.extf %{{.*}} : f16 to f32
// CHECK: %[[P:.*]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK: %[[S:.*]] = arith.addf %[[ACC:.*]], %[[P]] : f32
// CHECK: linalg.yield %[[S]] : f32

// -----

// CHECK-LABEL: func.func @permuted_products_and_contractions(
nv_tensor_ir.graph @permuted_products_and_contractions(
    %a: tensor<5x2x7x3xf16>, %b: tensor<11x5x13x7xf16>,
    %acc: tensor<3x2x13x11xf32>) -> tensor<3x2x13x11xf32> {
  %result = contraction(%a, %b) accum(%acc : tensor<3x2x13x11xf32>) maps (d0, d1, d2, d3, d4, d5) -> (d5, d1, d4, d0), (d3, d5, d2, d4), (d0, d1, d2, d3) : (tensor<5x2x7x3xf16>, tensor<11x5x13x7xf16>) -> tensor<3x2x13x11xf32>
  results %result : tensor<3x2x13x11xf32>
}
// CHECK-SAME: %[[A:.*]]: tensor<5x2x7x3xf16>, %[[B:.*]]: tensor<11x5x13x7xf16>, %[[INIT:.*]]: tensor<3x2x13x11xf32>
// CHECK-NOT: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d1, d4, d0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d5, d2, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME: outs(%[[INIT]] : tensor<3x2x13x11xf32>)
// CHECK: arith.extf
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @unequal_ranks(
nv_tensor_ir.graph @unequal_ranks(
    %a: tensor<2x3x5x7xf32>, %b: tensor<5x7x11xf32>) -> tensor<2x3x11xf32> {
  %result = contraction(%a, %b) maps (d0, d1, d2, d3, d4) -> (d0, d1, d3, d4), (d3, d4, d2), (d0, d1, d2) : (tensor<2x3x5x7xf32>, tensor<5x7x11xf32>) -> tensor<2x3x11xf32>
  results %result : tensor<2x3x11xf32>
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @permuted_batches(
nv_tensor_ir.graph @permuted_batches(
    %a: tensor<3x1x5x2xf32>, %b: tensor<7x11x1x5xf32>) -> tensor<2x7x3x11xf32> {
  %result = contraction(%a, %b) maps (d0, d1, d2, d3, d4) -> (d2, d1, d4, d0), (d1, d3, d0, d4), (d0, d1, d2, d3) : (tensor<3x1x5x2xf32>, tensor<7x11x1x5xf32>) -> tensor<2x7x3x11xf32>
  results %result : tensor<2x7x3x11xf32>
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d2, 0, d4, d0)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @permuted_domain(
nv_tensor_ir.graph @permuted_domain(
    %a: tensor<5x2xf32>, %b: tensor<7x5xf32>) -> tensor<2x7xf32> {
  %result = contraction(%a, %b) maps (d0, d1, d2) -> (d0, d2), (d1, d0), (d2, d1) : (tensor<5x2xf32>, tensor<7x5xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>]
// CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @dynamic(
nv_tensor_ir.graph @dynamic(
    %a: tensor<?x1x?x5x?xf32>, %b: tensor<2x?x5x7x11xf32>) -> tensor<2x?x?x11xf32> {
  %result = contraction(%a, %b) maps (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d5), (d0, d1, d4, d5, d3), (d0, d1, d2, d3) : (tensor<?x1x?x5x?xf32>, tensor<2x?x5x7x11xf32>) -> tensor<2x?x?x11xf32>
  results %result : tensor<2x?x?x11xf32>
}
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: tensor.generate
// CHECK: arith.select
// CHECK: tensor.extract
// CHECK: tensor.yield
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, 0, d2, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @dot(
nv_tensor_ir.graph @dot(
    %a: tensor<5x7xf32>, %b: tensor<7x5xf32>) -> tensor<f32> {
  %result = contraction(%a, %b) maps (d0, d1) -> (d1, d0), (d0, d1), () : (tensor<5x7xf32>, tensor<7x5xf32>) -> tensor<f32>
  results %result : tensor<f32>
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>]
// CHECK-SAME: iterator_types = ["reduction", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @outer(
nv_tensor_ir.graph @outer(
    %a: tensor<2x3xf32>, %b: tensor<5x7xf32>) -> tensor<3x2x7x5xf32> {
  %result = contraction(%a, %b) maps (d0, d1, d2, d3) -> (d1, d0), (d3, d2), (d0, d1, d2, d3) : (tensor<2x3xf32>, tensor<5x7xf32>) -> tensor<3x2x7x5xf32>
  results %result : tensor<3x2x7x5xf32>
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @zero_extent(
nv_tensor_ir.graph @zero_extent(
    %a: tensor<2x0x7xf32>, %b: tensor<0x7x3xf32>) -> tensor<2x3xf32> {
  %result = contraction(%a, %b) maps (d0, d1, d2, d3) -> (d0, d2, d3), (d2, d3, d1), (d0, d1) : (tensor<2x0x7xf32>, tensor<0x7x3xf32>) -> tensor<2x3xf32>
  results %result : tensor<2x3xf32>
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @scalar(
nv_tensor_ir.graph @scalar(
    %a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
  %result = contraction(%a, %b) maps () -> (), (), () : (tensor<f32>, tensor<f32>) -> tensor<f32>
  results %result : tensor<f32>
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>]
// CHECK-SAME: iterator_types = []
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

// Both named matmul forms promote f16 operands before multiplying.
// CHECK-LABEL: func.func @mixed_precision_matmul(
nv_tensor_ir.graph @mixed_precision_matmul(%a: tensor<1x2xf16>, %b: tensor<2x1xf16>) -> tensor<1x1xf32> {
  %r = matmul(%a, %b) : (tensor<1x2xf16>, tensor<2x1xf16>) -> tensor<1x1xf32>
  results %r : tensor<1x1xf32>
}
// CHECK: linalg.matmul
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<1x2xf16>, tensor<2x1xf16>)
// CHECK-SAME: outs(%{{.*}} : tensor<1x1xf32>)

// -----

// CHECK-LABEL: func.func @mixed_precision_batch_matmul(
nv_tensor_ir.graph @mixed_precision_batch_matmul(%a: tensor<1x1x2xf16>, %b: tensor<1x2x1xf16>) -> tensor<1x1x1xf32> {
  %r = matmul(%a, %b) : (tensor<1x1x2xf16>, tensor<1x2x1xf16>) -> tensor<1x1x1xf32>
  results %r : tensor<1x1x1xf32>
}
// CHECK: linalg.batch_matmul
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<1x1x2xf16>, tensor<1x2x1xf16>)
// CHECK-SAME: outs(%{{.*}} : tensor<1x1x1xf32>)

// -----

// CHECK-LABEL: func.func @matmul_rank4(
nv_tensor_ir.graph @matmul_rank4(%a: tensor<2x3x4x5xf32>, %b: tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
  %r = matmul(%a, %b) : (tensor<2x3x4x5xf32>, tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32>
  results %r : tensor<2x3x4x6xf32>
}
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @matmul_rank5_dynamic(
nv_tensor_ir.graph @matmul_rank5_dynamic(%a: tensor<?x2x3x4x5xf16>, %b: tensor<?x2x3x5x6xf16>) -> tensor<?x2x3x4x6xf32> {
  %r = matmul(%a, %b) : (tensor<?x2x3x4x5xf16>, tensor<?x2x3x5x6xf16>) -> tensor<?x2x3x4x6xf32>
  results %r : tensor<?x2x3x4x6xf32>
}
// CHECK: linalg.generic
// CHECK: arith.extf
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield
