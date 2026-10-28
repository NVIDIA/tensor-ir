// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics

// Use generic assembly to express an empty variadic input list unambiguously.
nv_tensor_ir.graph @reduceUDRequiresInput() -> tensor<8x1xf32> {
  // expected-error@below {{'nv_tensor_ir.reduce_ud' op requires at least one input}}
  %result = "nv_tensor_ir.reduce_ud"() ({
  ^bb0(%acc: f32, %value: f32):
    nv_tensor_ir.yield %acc : f32
  }) {dimensions = array<i32: 1>, identity = [0.0 : f32]}
      : () -> tensor<8x1xf32>
  results %result : tensor<8x1xf32>
}

// -----

nv_tensor_ir.graph @reduceUDOperandResultCount(
  %input: tensor<8x32xf32>, %other: tensor<8x32xf32>
) -> tensor<8x1xf32> {
  // expected-error@below {{expects number of operands to match number of results, but got 2 operands and 1 results}}
  %result = reduce_ud(%input, %other)<dimensions = [1], identity = [0.0 : f32]>
      (%acc: f32, %value: f32) {
    nv_tensor_ir.yield %acc : f32
  } : tensor<8x32xf32>, tensor<8x32xf32> -> tensor<8x1xf32>
  results %result : tensor<8x1xf32>
}

// -----

// The first input is dynamic: merging the second input refines its first
// dimension to 8, so the third input's dimension 9 must be rejected.
nv_tensor_ir.graph @reduceUDConflictingRefinements(
  %input: tensor<?x32xf32>, %other: tensor<8x32xf32>,
  %conflict: tensor<9x32xf32>
) -> (tensor<8x1xf32>, tensor<8x1xf32>, tensor<8x1xf32>) {
  // expected-error@below {{expects all input dimensions to merge, but input 0 has type 'tensor<?x32xf32>' and input 2 has type 'tensor<9x32xf32>'}}
  %result:3 = reduce_ud(%input, %other, %conflict)<
      dimensions = [1], identity = [0.0 : f32, 0.0 : f32, 0.0 : f32]>
      (%acc0: f32, %acc1: f32, %acc2: f32,
       %value0: f32, %value1: f32, %value2: f32) {
    nv_tensor_ir.yield %acc0, %acc1, %acc2 : f32, f32, f32
  } : tensor<?x32xf32>, tensor<8x32xf32>, tensor<9x32xf32>
      -> tensor<8x1xf32>, tensor<8x1xf32>, tensor<8x1xf32>
  results %result#0, %result#1, %result#2
      : tensor<8x1xf32>, tensor<8x1xf32>, tensor<8x1xf32>
}

// -----

// A matching first identity must not hide a mismatch at the second result.
nv_tensor_ir.graph @reduceUDSecondIdentityType(
  %input: tensor<8x32xf32>, %other: tensor<8x32xf16>
) -> (tensor<8x1xf32>, tensor<8x1xf16>) {
  // expected-error@below {{expects identity type to match result element type for index 1, but got identity type: 'f32' and result element type: 'f16'}}
  %result:2 = reduce_ud(%input, %other)<
      dimensions = [1], identity = [0.0 : f32, 0.0 : f32]>
      (%acc0: f32, %acc1: f32, %value0: f32, %value1: f32) {
    nv_tensor_ir.yield %acc0, %acc1 : f32, f32
  } : tensor<8x32xf32>, tensor<8x32xf16>
      -> tensor<8x1xf32>, tensor<8x1xf16>
  results %result#0, %result#1 : tensor<8x1xf32>, tensor<8x1xf16>
}

// -----

nv_tensor_ir.graph @reduceUDYieldCount(
  %input: tensor<8x32xf32>, %other: tensor<8x32xf32>
) -> (tensor<8x1xf32>, tensor<8x1xf32>) {
  // expected-error@below {{expects body region to yield exactly 2 values, but got 1 values}}
  %result:2 = reduce_ud(%input, %other)<
      dimensions = [1], identity = [0.0 : f32, 0.0 : f32]>
      (%acc0: f32, %acc1: f32, %value0: f32, %value1: f32) {
    nv_tensor_ir.yield %acc0 : f32
  } : tensor<8x32xf32>, tensor<8x32xf32>
      -> tensor<8x1xf32>, tensor<8x1xf32>
  results %result#0, %result#1 : tensor<8x1xf32>, tensor<8x1xf32>
}

// -----

nv_tensor_ir.graph @reduceUDSecondAccumulatorType(
  %input: tensor<8x32xf32>, %other: tensor<8x32xf16>
) -> (tensor<8x1xf32>, tensor<8x1xf16>) {
  // expected-error@below {{expects block argument 1 type to match identity type, but got argument type: 'f32' and identity type: 'f16'}}
  %result:2 = reduce_ud(%input, %other)<
      dimensions = [1], identity = [0.0 : f32, 0.0 : f16]>
      (%acc0: f32, %acc1: f32, %value0: f32, %value1: f16) {
    nv_tensor_ir.yield %acc0, %value1 : f32, f16
  } : tensor<8x32xf32>, tensor<8x32xf16>
      -> tensor<8x1xf32>, tensor<8x1xf16>
  results %result#0, %result#1 : tensor<8x1xf32>, tensor<8x1xf16>
}

// -----

nv_tensor_ir.graph @reduceUDSecondCurrentValueType(
  %input: tensor<8x32xf32>, %other: tensor<8x32xf16>
) -> (tensor<8x1xf32>, tensor<8x1xf16>) {
  // expected-error@below {{expects block argument 3 type to match identity type, but got argument type: 'f32' and identity type: 'f16'}}
  %result:2 = reduce_ud(%input, %other)<
      dimensions = [1], identity = [0.0 : f32, 0.0 : f16]>
      (%acc0: f32, %acc1: f16, %value0: f32, %value1: f32) {
    nv_tensor_ir.yield %acc0, %acc1 : f32, f16
  } : tensor<8x32xf32>, tensor<8x32xf16>
      -> tensor<8x1xf32>, tensor<8x1xf16>
  results %result#0, %result#1 : tensor<8x1xf32>, tensor<8x1xf16>
}
