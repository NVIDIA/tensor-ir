// RUN: tensor_ir-opt -layout-propagation-pipeline -verify-diagnostics -split-input-file %s

// Verify failures at each relevant stage of layout-propagation lowering:
// tensor constants are validated during grid formation, while compute ops are
// validated during outlined-kernel conversion. The accept case confirms that a
// supported pointwise op completes the full pipeline.

// `pow` only supports a floating-point result; an integer result is rejected.
nv_tensor_ir.graph @negative_pow_integer_result(
    %arg0: tensor<128xsi32>,
    %arg1: tensor<128xsi32>) -> (tensor<128xsi32>)
    attributes {tile_size = array<i32: 128>} {
  // expected-error @below {{unsupported element type}}
  // expected-error @below {{failed to legalize operation 'nv_tensor_ir.pow'}}
  %out = pow %arg0, %arg1 : (tensor<128xsi32>, tensor<128xsi32>) -> tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// `not` only supports the boolean (i1) element type.
nv_tensor_ir.graph @negative_not_non_boolean(
    %arg0: tensor<128xsi32>) -> (tensor<128xsi32>)
    attributes {tile_size = array<i32: 128>} {
  // expected-error @below {{unsupported element type}}
  // expected-error @below {{failed to legalize operation 'nv_tensor_ir.not'}}
  %out = not %arg0 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// Exact GELU depends on cuda_tile.experimental$erf, which only supports f32
// and f64 element types.
nv_tensor_ir.graph @negative_gelu_fwd_f16(
    %arg0: tensor<128xf16>) -> (tensor<128xf16>)
    attributes {tile_size = array<i32: 128>} {
  // expected-error @below {{unsupported element type}}
  // expected-error @below {{failed to legalize operation 'nv_tensor_ir.gelu_fwd'}}
  %out = gelu_fwd %arg0 : tensor<128xf16>
  results %out : tensor<128xf16>
}

// -----

// A tensor-typed constant must carry a splat literal; a non-splat dense literal
// is unsupported by the lowering.
nv_tensor_ir.graph @negative_non_splat_constant(
    %arg0: tensor<4xf32>) -> (tensor<4xf32>)
    attributes {tile_size = array<i32: 4>} {
  // expected-error @below {{grid formation requires tensor constants to be splats}}
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
// The staged converter currently groups this with unsupported element types
// under "Unsupported reduction type".
nv_tensor_ir.graph @negative_reduce_customize(
    %arg0: tensor<128xf32>) -> (tensor<1xf32>)
    attributes {tile_size = array<i32: 1>} {
  // expected-error @below {{Unsupported reduction type}}
  // expected-error @below {{failed to legalize operation 'nv_tensor_ir.reduce'}}
  %out = nv_tensor_ir.reduce(%arg0)<
      dimensions = [0],
      reduction_mode = <customize> >
      : tensor<128xf32> -> tensor<1xf32>
  results %out : tensor<1xf32>
}

// -----

// Metadata-driven single-output lowering requires both entries to use the
// normalized carrier shape, not merely to have one entry each.
// expected-error @below {{single-output result metadata is not expressed in normalized carrier shape [8]}}
nv_tensor_ir.graph @invalid_single_output_metadata_shape(
    %arg0: tensor<8xf32>) -> (tensor<8xf32>)
    attributes {tile_size = array<i32: 8>} {
  %out = abs %arg0 : tensor<8xf32>
  results %out attributes {
    iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(8):(1)">,
    result_layouts = [#nv_tensor_ir.tensor_source<0, 0, "(4):(1)">],
    result_views = [#nv_tensor_ir.tensor_source<1, 0, "(4):(1)">]
  } : tensor<8xf32>
}
