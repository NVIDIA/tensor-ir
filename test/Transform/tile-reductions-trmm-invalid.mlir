// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{reduction-tile-size=128}))' --split-input-file --verify-diagnostics

// The initial TRMM implementation rejects fragmented K instead of using
// a one-dimensional triangular bound on a multi-dimensional K loop nest.
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
  // expected-error@below {{left_triangle requires an unfragmented K dimension}}
  %result = matmul(%lhs_matrix, %rhs) <left_triangle = upper>
      : (tensor<64x1024xf32>, tensor<1024x48xf32>) -> tensor<64x48xf32>
  results %result : tensor<64x48xf32>
}

// -----

nv_tensor_ir.graph @fragmented_right_matmul(
    %lhs: tensor<64x32x32xf32> {nv_tensor_ir.stride = "(1024,32,1)"},
    %rhs: tensor<1024x48xf32>)
    -> tensor<64x48xf32> attributes {tile_size = array<i32: 32, 16>} {
  %lhs_transposed = transpose %lhs permutation = [2, 1, 0]
      : tensor<64x32x32xf32> -> tensor<32x32x64xf32>
  %lhs_flat = reshape %lhs_transposed
      : tensor<32x32x64xf32> -> tensor<1024x64xf32>
  %lhs_matrix = transpose %lhs_flat permutation = [1, 0]
      : tensor<1024x64xf32> -> tensor<64x1024xf32>
  // expected-error@below {{right_triangle requires an unfragmented K dimension}}
  %result = matmul(%lhs_matrix, %rhs) <right_triangle = upper>
      : (tensor<64x1024xf32>, tensor<1024x48xf32>) -> tensor<64x48xf32>
  results %result : tensor<64x48xf32>
}
