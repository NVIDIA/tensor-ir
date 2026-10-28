// RUN: split-file %s %t
// RUN: tensor_ir-opt %t/missing-map.mlir \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   -verify-diagnostics
// RUN: tensor_ir-opt %t/missing-map-with-strides.mlir \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   -verify-diagnostics
// RUN: tensor_ir-opt %t/reductions.mlir -split-input-file \
// RUN:   -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   -verify-diagnostics

//--- missing-map.mlir

// An operation map establishes the iteration-space rank, but cannot replace
// the input descriptor map that the affine lowering requires.
// expected-error @+2 {{Iteration space map not found for affine map codegen}}
// expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @missing_map(%a: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %r = abs %a {
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
    : tensor<8x8xf32>
  results %r : tensor<8x8xf32>
}

//--- missing-map-with-strides.mlir

// Having other descriptor attributes does not supply an iteration-space map.
// expected-error @+2 {{Iteration space map not found for affine map codegen}}
// expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @missing_map_with_strides(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8, 1)"})
    -> tensor<8x8xf32> {
  %r = abs %a {
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
    : tensor<8x8xf32>
  results %r : tensor<8x8xf32>
}

//--- reductions.mlir

nv_tensor_ir.graph @integer_reduction(%a: tensor<8x8xsi32>)
    -> tensor<8x1xsi32> {
  // expected-error @+2 {{Unsupported element type for reduction operation}}
  // expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.reduce'}}
  %r = reduce(%a) <dimensions = [1], reduction_mode = <add>>
    : tensor<8x8xsi32> -> tensor<8x1xsi32>
  results %r : tensor<8x1xsi32>
}

// -----

nv_tensor_ir.graph @multi_reduction(%a: tensor<8x8xf32>)
    -> tensor<1x1xf32> {
  // expected-error @+2 {{Multiple dimensions are not supported for reduction}}
  // expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.reduce'}}
  %r = reduce(%a) <dimensions = [0, 1], reduction_mode = <add>>
    : tensor<8x8xf32> -> tensor<1x1xf32>
  results %r : tensor<1x1xf32>
}

// -----

nv_tensor_ir.graph @customize_reduction(%a: tensor<8xf32>)
    -> tensor<1xf32> {
  // expected-error @+2 {{Customize reduction mode is not supported}}
  // expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.reduce'}}
  %r = reduce(%a) <dimensions = [0], reduction_mode = <customize>>
    : tensor<8xf32> -> tensor<1xf32>
  results %r : tensor<1xf32>
}
