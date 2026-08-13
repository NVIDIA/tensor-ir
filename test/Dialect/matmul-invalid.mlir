// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics

nv_tensor_ir.graph @matmulRankTooSmall(
  %a: tensor<32xf32>,
  %b: tensor<32xf32>
) -> tensor<1xf32> {
  // expected-error@below {{expects rank of at least 2, but got 1}}
  %result = matmul(%a, %b)
      : (tensor<32xf32>, tensor<32xf32>) -> tensor<1xf32>
  results %result : tensor<1xf32>
}

// -----

nv_tensor_ir.graph @matmulRankMismatch(
  %a: tensor<16x32xf32>,
  %b: tensor<2x32x8xf32>
) -> tensor<16x8xf32> {
  // expected-error@below {{failed to verify that all of {a, b, c} have same rank}}
  %result = matmul(%a, %b)
      : (tensor<16x32xf32>, tensor<2x32x8xf32>) -> tensor<16x8xf32>
  results %result : tensor<16x8xf32>
}

// -----

nv_tensor_ir.graph @matmulBatchMismatch(
  %a: tensor<2x16x32xf32>,
  %b: tensor<3x32x8xf32>
) -> tensor<2x16x8xf32> {
  // expected-error@below {{expects C batch dimensions to be the broadcasted shape of A and B, but got 2, 3, and 2}}
  %result = matmul(%a, %b)
      : (tensor<2x16x32xf32>, tensor<3x32x8xf32>)
        -> tensor<2x16x8xf32>
  results %result : tensor<2x16x8xf32>
}

// -----

nv_tensor_ir.graph @matmulBroadcastResultMismatch(
  %a: tensor<2x16x32xf32>,
  %b: tensor<1x32x8xf32>
) -> tensor<1x16x8xf32> {
  // expected-error@below {{expects C batch dimensions to be the broadcasted shape of A and B, but got 2, 1, and 1}}
  %result = matmul(%a, %b)
      : (tensor<2x16x32xf32>, tensor<1x32x8xf32>)
        -> tensor<1x16x8xf32>
  results %result : tensor<1x16x8xf32>
}

// -----

nv_tensor_ir.graph @matmulContractingDimensionMismatch(
  %a: tensor<16x32xf32>,
  %b: tensor<64x8xf32>
) -> tensor<16x8xf32> {
  // expected-error@below {{expects the contracting dimensions of A and B to match, but got 32 and 64}}
  %result = matmul(%a, %b)
      : (tensor<16x32xf32>, tensor<64x8xf32>) -> tensor<16x8xf32>
  results %result : tensor<16x8xf32>
}

// -----

nv_tensor_ir.graph @matmulResultShapeMismatch(
  %a: tensor<16x32xf32>,
  %b: tensor<32x8xf32>
) -> tensor<15x8xf32> {
  // expected-error@below {{expects C's matrix dimensions to be inherited exactly from A's rows and B's columns, but got A, B, and C shapes}}
  %result = matmul(%a, %b)
      : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<15x8xf32>
  results %result : tensor<15x8xf32>
}

// -----

nv_tensor_ir.graph @matmulAccumulatorShapeMismatch(
  %a: tensor<16x32xf32>,
  %b: tensor<32x8xf32>,
  %acc: tensor<15x8xf32>
) -> tensor<16x8xf32> {
  // expected-error@below {{expects accumulator type to match result type}}
  %result = matmul(%a, %b) accum(%acc : tensor<15x8xf32>)
      : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
  results %result : tensor<16x8xf32>
}

// -----

nv_tensor_ir.graph @matmulAccumulatorElementTypeMismatch(
  %a: tensor<16x32xf16>,
  %b: tensor<32x8xf16>,
  %acc: tensor<16x8xf16>
) -> tensor<16x8xf32> {
  // expected-error@below {{expects accumulator type to match result type}}
  %result = matmul(%a, %b) accum(%acc : tensor<16x8xf16>)
      : (tensor<16x32xf16>, tensor<32x8xf16>) -> tensor<16x8xf32>
  results %result : tensor<16x8xf32>
}
