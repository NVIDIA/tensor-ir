// RUN: tensor_ir-opt %s \
// RUN:   --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides),test-layout-normalization)' \
// RUN:   --split-input-file --verify-diagnostics -o /dev/null

// The reshaped and transposed matmul output has the view layout
// (4,9,4):(4,16,1), whose mode (9,16) straddles the K/N group boundary (24)
// without being divisible by it. Normalization must fail gracefully instead of
// asserting, so that callers can skip the unsupported graph.
// expected-error@below {{failed to normalize layout}}
nv_tensor_ir.graph @matmul_reshape_transpose(
    %arg0: tensor<6x4xf32>, %arg1: tensor<4x6xf32>) -> tensor<4x9xf32> {
  %0 = matmul(%arg0, %arg1) : (tensor<6x4xf32>, tensor<4x6xf32>) -> tensor<6x6xf32>
  %1 = reshape %0 : tensor<6x6xf32> -> tensor<9x4xf32>
  %2 = transpose %1 permutation = [1, 0] : tensor<9x4xf32> -> tensor<4x9xf32>
  results %2 : tensor<4x9xf32>
}
