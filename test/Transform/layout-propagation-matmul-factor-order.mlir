// RUN: tensor_ir-opt %s --split-input-file --materialize-default-strides \
// RUN:   --test-layout-normalization -o /dev/null | FileCheck %s --check-prefix=NORM
// RUN: tensor_ir-opt %s --split-input-file --pass-pipeline='builtin.module(\
// RUN:    nv_tensor_ir.graph(materialize-default-strides,tile-selection), \
// RUN:    tir-bufferize,func.func(tir-form-grid))' | FileCheck %s --check-prefix=GRID

// The unpermuted control keeps the physical batch factors (2,3) and both
// padded input strides. No layout change is needed for this case.
// NORM-LABEL: layout normalization for @batch_factor_order
//       NORM: iteration_space: #nv_tensor_ir.matmul_source<"(2,3,2,2,4):(48,16,8,4,1)"
// GRID-LABEL: func.func @batch_factor_order_dispatch
//       GRID: scf.forall (%[[CTA:.*]]) in (6)
//       GRID: %[[B0:.*]] = arith.remui %[[CTA]],
//       GRID: %[[B1:.*]] = arith.divui %[[CTA]],
//       GRID: %[[LHS:.*]] = nv_tensor_ir.load {{.*}}[%[[B0]], %[[B1]], 0, 0]
//  GRID-SAME: sizes: [2, 3, 2, 4], strides: [96, 16, 4, 1]
//       GRID: %[[RHS:.*]] = nv_tensor_ir.load {{.*}}[%[[B0]], %[[B1]], 0, 0]
//  GRID-SAME: sizes: [2, 3, 4, 2], strides: [96, 16, 2, 1]
//       GRID: %[[A:.*]] = nv_tensor_ir.reshape %[[LHS]]
//       GRID: %[[B:.*]] = nv_tensor_ir.reshape %[[RHS]]
//       GRID: %[[PRODUCT:.*]] = nv_tensor_ir.matmul(%[[A]], %[[B]])
//       GRID: %[[OUT:.*]] = nv_tensor_ir.reshape %[[PRODUCT]]
//       GRID: nv_tensor_ir.store %[[OUT]], {{.*}}[%[[B0]], %[[B1]], 0, 0]
nv_tensor_ir.graph @batch_factor_order(
    %lhs: tensor<2x3x2x4xbf16> {nv_tensor_ir.stride = "(96,16,4,1)"},
    %rhs: tensor<2x3x4x2xbf16> {nv_tensor_ir.stride = "(96,16,2,1)"})
    -> tensor<2x3x2x2xf32> attributes {tile_size = array<i32: 1, 1, 2, 2>} {
  %product = matmul(%lhs, %rhs) : (tensor<2x3x2x4xbf16>, tensor<2x3x4x2xbf16>) -> tensor<2x3x2x2xf32>
  results %product : tensor<2x3x2x2xf32>
}

// -----

// Reduction moves one batch factor to the end of the output view. It must
// not reshape the inputs from physical batch order (2,3) into view order
// (3,2). The two non-divisible extents expose the previously hidden mismatch.
// NORM-LABEL: layout normalization for @permuted_batch_reduction
//       NORM: iteration_space: #nv_tensor_ir.reduction_source<"(3,2,2,(2)):(8,4,2,(1))", #nv_tensor_ir.matmul_source<"(3,2,2,2,4):(16,8,4,48,1)"
// GRID-LABEL: func.func @permuted_batch_reduction_dispatch
//       GRID: scf.forall (%[[BATCH:.*]]) in (3)
//       GRID: %[[LHS:.*]] = nv_tensor_ir.load {{.*}}[0, %[[BATCH]], 0, 0]
//  GRID-SAME: sizes: [2, 3, 2, 4], strides: [96, 16, 4, 1]
//       GRID: %[[RHS:.*]] = nv_tensor_ir.load {{.*}}[0, %[[BATCH]], 0, 0]
//  GRID-SAME: sizes: [2, 3, 4, 2], strides: [96, 16, 2, 1]
//       GRID: %[[A:.*]] = nv_tensor_ir.reshape %[[LHS]]
//       GRID: %[[B:.*]] = nv_tensor_ir.reshape %[[RHS]]
//       GRID: %[[PRODUCT:.*]] = nv_tensor_ir.matmul(%[[A]], %[[B]])
//       GRID: %[[SHAPED:.*]] = nv_tensor_ir.reshape %[[PRODUCT]]
//       GRID: %[[PERMUTED:.*]] = nv_tensor_ir.transpose %[[SHAPED]] permutation = [1, 2, 3, 0]
//       GRID: %[[SUM:.*]] = nv_tensor_ir.reduce(%[[PERMUTED]]) <dimensions = [3], reduction_mode = <add>>
//       GRID: %[[OUT:.*]] = nv_tensor_ir.reshape %[[SUM]]
//       GRID: nv_tensor_ir.store %[[OUT]], {{.*}}[%[[BATCH]], 0, 0]
nv_tensor_ir.graph @permuted_batch_reduction(
    %lhs: tensor<2x3x2x4xbf16> {nv_tensor_ir.stride = "(96,16,4,1)"},
    %rhs: tensor<2x3x4x2xbf16> {nv_tensor_ir.stride = "(96,16,2,1)"})
    -> tensor<3x1x2x2xf32> attributes {tile_size = array<i32: 1, 2, 2>} {
  %product = matmul(%lhs, %rhs) : (tensor<2x3x2x4xbf16>, tensor<2x3x4x2xbf16>) -> tensor<2x3x2x2xf32>
  %permuted = transpose %product permutation = [1, 0, 2, 3] : tensor<2x3x2x2xf32> -> tensor<3x2x2x2xf32>
  %sum = reduce(%permuted) <dimensions = [1], reduction_mode = <add>> : tensor<3x2x2x2xf32> -> tensor<3x1x2x2xf32>
  results %sum : tensor<3x1x2x2xf32>
}

// -----

// Concatenation keeps the packed batch factor distinct even with contiguous
// inputs. Transposing the output must preserve the operand factor order.
// NORM-LABEL: layout normalization for @packed_batch_matmul
//       NORM: iteration_space: #nv_tensor_ir.matmul_source<"(3,2,2,2,4):(16,48,8,4,1)"
// GRID-LABEL: func.func @packed_batch_matmul_dispatch
//       GRID: scf.forall (%[[CTA:.*]]) in (6)
//       GRID: %[[PACKED:.*]] = arith.remui %[[CTA]],
//       GRID: %[[BATCH:.*]] = arith.divui %[[CTA]],
//       GRID: %[[PRODUCT:.*]] = nv_tensor_ir.matmul(
//       GRID: %[[SHAPED:.*]] = nv_tensor_ir.reshape %[[PRODUCT]]
//       GRID: %[[OUT:.*]] = nv_tensor_ir.transpose %[[SHAPED]] permutation = [1, 0, 2, 3]
//       GRID: nv_tensor_ir.store %[[OUT]], {{.*}}[%[[PACKED]], %[[BATCH]], 0, 0]
nv_tensor_ir.graph @packed_batch_matmul(
    %lhs: tensor<2x1x2x4xbf16>, %rhs: tensor<2x1x4x2xbf16>)
    -> tensor<3x2x2x2xf32> attributes {tile_size = array<i32: 1, 1, 2, 2>} {
  %a = concatenate %lhs, %lhs, %lhs dimension = 1 : (tensor<2x1x2x4xbf16>, tensor<2x1x2x4xbf16>, tensor<2x1x2x4xbf16>) -> tensor<2x3x2x4xbf16>
  %b = concatenate %rhs, %rhs, %rhs dimension = 1 : (tensor<2x1x4x2xbf16>, tensor<2x1x4x2xbf16>, tensor<2x1x4x2xbf16>) -> tensor<2x3x4x2xbf16>
  %product = matmul(%a, %b) : (tensor<2x3x2x4xbf16>, tensor<2x3x4x2xbf16>) -> tensor<2x3x2x2xf32>
  %transposed = transpose %product permutation = [1, 0, 2, 3] : tensor<2x3x2x2xf32> -> tensor<3x2x2x2xf32>
  results %transposed : tensor<3x2x2x2xf32>
}
