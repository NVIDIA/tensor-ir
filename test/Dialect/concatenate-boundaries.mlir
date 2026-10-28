// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics

// Shape-only verification must reject an extent sum larger than INT64_MAX.
// No tensor storage is allocated by these tests.
nv_tensor_ir.graph @concatenateStaticExtentOverflow(
  %large: tensor<9223372036854775807xf32>, %one: tensor<1xf32>
) -> tensor<?xf32> {
  // expected-error@below {{'nv_tensor_ir.concatenate' op concatenation dimension size overflows i64}}
  %result = concatenate %large, %one dimension = 0
      : (tensor<9223372036854775807xf32>, tensor<1xf32>) -> tensor<?xf32>
  results %result : tensor<?xf32>
}

// -----

// Seeing a dynamic extent must not bypass overflow checking of later static
// extents, even though the result extent is dynamic.
nv_tensor_ir.graph @concatenateOverflowAfterDynamicExtent(
  %dynamic: tensor<?xf32>, %large: tensor<9223372036854775807xf32>,
  %one: tensor<1xf32>
) -> tensor<?xf32> {
  // expected-error@below {{'nv_tensor_ir.concatenate' op concatenation dimension size overflows i64}}
  %result = concatenate %dynamic, %large, %one dimension = 0
      : (tensor<?xf32>, tensor<9223372036854775807xf32>, tensor<1xf32>) -> tensor<?xf32>
  results %result : tensor<?xf32>
}

// -----

// The largest representable sum is valid: the overflow comparison is strict.
// With --verify-diagnostics, these valid cases must emit no diagnostics.
nv_tensor_ir.graph @concatenateMaximumStaticExtent(
  %large: tensor<9223372036854775806xf32>, %one: tensor<1xf32>
) -> tensor<9223372036854775807xf32> {
  %result = concatenate %large, %one dimension = 0
      : (tensor<9223372036854775806xf32>, tensor<1xf32>) -> tensor<9223372036854775807xf32>
  results %result : tensor<9223372036854775807xf32>
}

// -----

// A dynamic extent still requires a dynamic result when the static subtotal
// reaches the largest representable value without overflowing.
nv_tensor_ir.graph @concatenateMaximumStaticSubtotalWithDynamicExtent(
  %dynamic: tensor<?xf32>, %large: tensor<9223372036854775806xf32>,
  %one: tensor<1xf32>
) -> tensor<?xf32> {
  %result = concatenate %dynamic, %large, %one dimension = 0
      : (tensor<?xf32>, tensor<9223372036854775806xf32>, tensor<1xf32>) -> tensor<?xf32>
  results %result : tensor<?xf32>
}
