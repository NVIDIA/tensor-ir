// RUN: tensor_ir-opt %s --tir-bufferize --verify-diagnostics

func.func @collision_dispatch() {
  return
}

// expected-error @below {{cannot create bufferized function @collision_dispatch; symbol already exists}}
nv_tensor_ir.graph @collision(%input: tensor<16xf32>)
    -> tensor<16xf32> attributes {tile_size = array<i32: 8>} {
  %result = cos %input
      : tensor<16xf32>
  results %result : tensor<16xf32>
}
