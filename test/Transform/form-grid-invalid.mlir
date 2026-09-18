// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(tir-bufferize,func.func(tir-form-grid))' --split-input-file --verify-diagnostics

// expected-error @below {{missing iteration_space attribute on bufferized program}}
nv_tensor_ir.graph @missing_iteration_space(%input: tensor<16xf32>)
    -> tensor<16xf32> attributes {tile_size = array<i32: 8>} {
  %result = cos %input {layout = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">,
                       nv_tensor_ir.iter_space_id = 0 : i32}
      : tensor<16xf32>
  results %result : tensor<16xf32>
}

// -----

// expected-error @below {{missing tile_size attribute on bufferized program}}
nv_tensor_ir.graph @missing_tile(%input: tensor<16xf32>) -> tensor<16xf32> {
  %result = cos %input {layout = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">,
                       nv_tensor_ir.iter_space_id = 0 : i32}
      : tensor<16xf32>
  results %result attributes {
    iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">
  } : tensor<16xf32>
}

// -----

nv_tensor_ir.graph @bad_concat_tile(
    %lhs: tensor<2xf32>, %rhs: tensor<2xf32>) -> tensor<4xf32>
    attributes {tile_size = array<i32: 2>} {
  // expected-error @below {{concatenation dimension 0 must have tile size 1, but got 2}}
  %result = concatenate %lhs, %rhs dimension = 0 {
    layout = #nv_tensor_ir.concat_source<dim = 0,
        #nv_tensor_ir.tensor_source<0, 0, "(2):(1)">,
        #nv_tensor_ir.tensor_source<1, 0, "(2):(1)">>,
    nv_tensor_ir.iter_space_id = 0 : i32
  } : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
  results %result attributes {
    iteration_space = #nv_tensor_ir.concat_source<dim = 0,
        #nv_tensor_ir.tensor_source<0, 0, "(2):(1)">,
        #nv_tensor_ir.tensor_source<1, 0, "(2):(1)">>
  } : tensor<4xf32>
}
