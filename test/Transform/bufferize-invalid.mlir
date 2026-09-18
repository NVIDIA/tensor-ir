// RUN: tensor_ir-opt %s --tir-bufferize --verify-diagnostics

func.func @collision_dispatch() {
  return
}

// expected-error @below {{cannot create bufferized function @collision_dispatch; symbol already exists}}
nv_tensor_ir.graph @collision(%input: tensor<16xf32>)
    -> tensor<16xf32> attributes {tile_size = array<i32: 8>} {
  %result = cos %input {layout = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">,
                       nv_tensor_ir.iter_space_id = 0 : i32}
      : tensor<16xf32>
  results %result attributes {
    iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">
  } : tensor<16xf32>
}
