// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --split-input-file --verify-diagnostics

func.func @non_argument_boundary(%output: memref<16xf32, #ptr.generic_space>)
    attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 4>
} {
  %buffer = memref.alloca() : memref<16xf32, #ptr.generic_space>
  // expected-error @below {{bufferized TensorIR boundary must be a memref function argument}}
  %load = bufferization.to_tensor %buffer restrict
      : memref<16xf32, #ptr.generic_space> to tensor<16xf32>
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<16xf32>, memref<16xf32, #ptr.generic_space>) -> ()
  return
}
