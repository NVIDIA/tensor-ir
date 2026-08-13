// RUN: tensor_ir-opt -verify-layout-prop-lowerable -verify-diagnostics -split-input-file %s

// Verify that the layout-propagation lowerability check rejects pointwise ops
// whose element type is not supported by the lowering. The legality check is
// shared between the conversion pattern and this pre-flight feasibility check,
// so the same diagnostics cover the canCompile path.

// `pow` only supports a floating-point result; an integer result is rejected.
nv_tensor_ir.graph @negative_pow_integer_result(
    %arg0: tensor<128xsi32>,
    %arg1: tensor<128xsi32>) -> (tensor<128xsi32>)
    attributes {tile_size = array<i32: 128>} {
  // expected-error@below {{unsupported element type}}
  %out = pow %arg0, %arg1 : (tensor<128xsi32>, tensor<128xsi32>) -> tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// `not` only supports the boolean (i1) element type.
nv_tensor_ir.graph @negative_not_non_boolean(
    %arg0: tensor<128xsi32>) -> (tensor<128xsi32>)
    attributes {tile_size = array<i32: 128>} {
  // expected-error@below {{unsupported element type}}
  %out = not %arg0 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// Exact GELU depends on cuda_tile.experimental$erf, which only supports f32
// and f64 element types.
nv_tensor_ir.graph @negative_gelu_fwd_f16(
    %arg0: tensor<128xf16>) -> (tensor<128xf16>)
    attributes {tile_size = array<i32: 128>} {
  // expected-error@below {{unsupported element type}}
  %out = gelu_fwd %arg0 : tensor<128xf16>
  results %out : tensor<128xf16>
}

// -----

// A tensor-typed constant must carry a splat literal; a non-splat dense literal
// is unsupported by the lowering.
nv_tensor_ir.graph @negative_non_splat_constant(
    %arg0: tensor<4xf32>) -> (tensor<4xf32>)
    attributes {tile_size = array<i32: 4>} {
  // expected-error@below {{unsupported constant value}}
  %c = nv_tensor_ir.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  %out = add %arg0, %c : tensor<4xf32>
  results %out : tensor<4xf32>
}

// -----

// Accept case (the inverse of the negatives above): a pointwise op on a
// supported element type is lowerable, so the check must emit no diagnostic and
// the tool must exit 0. `abs` on f32 is the accept-direction analogue used by
// the C++ unit test (TestCanCompileAcceptsSupportedElementType). No
// `// expected-error` here is exactly the accept assertion under
// -verify-diagnostics.
nv_tensor_ir.graph @accept_supported_pointwise(
    %arg0: tensor<128xf32>) -> (tensor<128xf32>)
    attributes {tile_size = array<i32: 128>} {
  %out = abs %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// `reduce` with the `customize` reduction mode carries a user-defined combiner
// that the tile lowering cannot synthesize, so the mode is rejected.
nv_tensor_ir.graph @negative_reduce_customize(
    %arg0: tensor<128xf32>) -> (tensor<1xf32>)
    attributes {tile_size = array<i32: 1>} {
  // expected-error@below {{Unsupported reduction mode}}
  %out = nv_tensor_ir.reduce(%arg0)<
      dimensions = [0],
      reduction_mode = <customize> >
      : tensor<128xf32> -> tensor<1xf32>
  results %out : tensor<1xf32>
}
