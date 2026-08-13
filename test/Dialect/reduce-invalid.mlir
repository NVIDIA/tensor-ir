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
