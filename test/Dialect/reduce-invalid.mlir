// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics

nv_tensor_ir.graph @emptyDimensions(
  %input: tensor<8x32xf32>
) -> (tensor<8x32xf32>) {
  // expected-error@below {{attribute 'dimensions' failed to satisfy constraint: i32 dense array attribute with at least 1 elements}}
  %result = reduce(%input)<
      dimensions = [], reduction_mode = <add>>
      : tensor<8x32xf32> -> tensor<8x32xf32>
  results %result : tensor<8x32xf32>
}

// -----

nv_tensor_ir.graph @dimensionOutOfBounds(
  %input: tensor<8x32xf32>
) -> (tensor<8x1xf32>) {
  // expected-error@below {{reduction dimension 2 is out of bounds for tensor rank 2}}
  %result = reduce(%input)<
      dimensions = [2], reduction_mode = <add>>
      : tensor<8x32xf32> -> tensor<8x1xf32>
  results %result : tensor<8x1xf32>
}

// -----

nv_tensor_ir.graph @duplicateDimension(
  %input: tensor<8x32xf32>
) -> (tensor<1x1xf32>) {
  // expected-error@below {{reduction dimension 1 is duplicated}}
  %result = reduce(%input)<
      dimensions = [1, 1], reduction_mode = <add>>
      : tensor<8x32xf32> -> tensor<1x1xf32>
  results %result : tensor<1x1xf32>
}

// -----

nv_tensor_ir.graph @nonReducedDimensionMustBeInherited(
  %input: tensor<?x32xf32>
) -> (tensor<4x1xf32>) {
  // expected-error@below {{expects output dimension 0 to remain dynamic because the corresponding input dimension is dynamic}}
  %result = reduce(%input)<
      dimensions = [1], reduction_mode = <add>>
      : tensor<?x32xf32> -> tensor<4x1xf32>
  results %result : tensor<4x1xf32>
}

// -----

nv_tensor_ir.graph @reducedDimensionMustBeUnit(
  %input: tensor<8x32xf32>
) -> (tensor<8x2xf32>) {
  // expected-error@below {{expects output dimension 1 to be 1, but got 2}}
  %result = reduce(%input)<
      dimensions = [1], reduction_mode = <add>>
      : tensor<8x32xf32> -> tensor<8x2xf32>
  results %result : tensor<8x2xf32>
}

// -----

func.func @staticNonReducedDimensionCannotBeErased(
    %input: tensor<8x32xf32>) {
  // expected-error@below {{expects output dimension 0 to be 8}}
  %result = nv_tensor_ir.reduce(%input)<
      dimensions = [1], reduction_mode = <add>>
      : tensor<8x32xf32> -> tensor<?x1xf32>
  return
}

// -----

func.func @staticReducedDimensionCannotBeErased(
    %input: tensor<8x32xf32>) {
  // expected-error@below {{expects output dimension 1 to be 1}}
  %result = nv_tensor_ir.reduce(%input)<
      dimensions = [1], reduction_mode = <add>>
      : tensor<8x32xf32> -> tensor<8x?xf32>
  return
}

// -----

func.func @reduceUDStaticNonReducedDimensionCannotBeErased(
    %input: tensor<8x32xf32>) {
  // expected-error@below {{expects output dimension 0 to be 8}}
  %result = nv_tensor_ir.reduce_ud(%input)
      <dimensions = [1], identity = [0.0 : f32]>
      (%lhs: f32, %rhs: f32) {
        nv_tensor_ir.yield %lhs : f32
      } : tensor<8x32xf32> -> tensor<?x1xf32>
  return
}

// -----

func.func @reduceUDStaticReducedDimensionCannotBeErased(
    %input: tensor<8x32xf32>) {
  // expected-error@below {{expects output dimension 1 to be 1}}
  %result = nv_tensor_ir.reduce_ud(%input)
      <dimensions = [1], identity = [0.0 : f32]>
      (%lhs: f32, %rhs: f32) {
        nv_tensor_ir.yield %lhs : f32
      } : tensor<8x32xf32> -> tensor<8x?xf32>
  return
}

// -----

func.func @reducedDimensionsMustBeAllRetainedOrAllDropped(
    %input: tensor<2x3x4xf32>) {
  // expected-error@below {{expects output rank to be 3 when retaining reduced dimensions or 1 when dropping them, but got 2}}
  %result = nv_tensor_ir.reduce(%input)<
      dimensions = [0, 2], reduction_mode = <add>>
      : tensor<2x3x4xf32> -> tensor<3x1xf32>
  return
}

// -----

func.func @reduceUDResultsMustUseSameConvention(
    %input0: tensor<8x32xf32>, %input1: tensor<8x32xf32>) {
  // expected-error@below {{expects all results to use the same reduced-dimension convention}}
  %result:2 = nv_tensor_ir.reduce_ud(%input0, %input1)
      <dimensions = [1], identity = [0.0 : f32, 0.0 : f32]>
      (%lhs0: f32, %lhs1: f32, %rhs0: f32, %rhs1: f32) {
        nv_tensor_ir.yield %lhs0, %lhs1 : f32, f32
      } : tensor<8x32xf32>, tensor<8x32xf32>
          -> tensor<8x1xf32>, tensor<8xf32>
  return
}

// -----

func.func @reduceUDInputsMustHaveSameRank(
    %input0: tensor<8x32xf32>, %input1: tensor<8x32x64xf32>) {
  // expected-error@below {{expects all inputs to have the same rank, but input 0 has type 'tensor<8x32xf32>' and input 1 has type 'tensor<8x32x64xf32>'}}
  %result:2 = nv_tensor_ir.reduce_ud(%input0, %input1)
      <dimensions = [1], identity = [0.0 : f32, 0.0 : f32]>
      (%lhs0: f32, %lhs1: f32, %rhs0: f32, %rhs1: f32) {
        nv_tensor_ir.yield %lhs0, %lhs1 : f32, f32
      } : tensor<8x32xf32>, tensor<8x32x64xf32>
          -> tensor<8xf32>, tensor<8x64xf32>
  return
}
