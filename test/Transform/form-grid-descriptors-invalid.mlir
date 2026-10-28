// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --split-input-file --verify-diagnostics

func.func @static_input_offset(%input: memref<16xf32, strided<[1], offset: 3>>,
                               %output: memref<16xf32>)
    attributes {nv_tensor_ir.bufferized_program, tile_size = array<i32: 4>} {
  // expected-error @below {{bufferized TensorIR boundary must have a statically zero offset}}
  %load = bufferization.to_tensor %input restrict
      : memref<16xf32, strided<[1], offset: 3>> to tensor<16xf32>
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

func.func @dynamic_output_offset(%input: memref<16xf32>,
                                 %output: memref<16xf32, strided<[1], offset: ?>>)
    attributes {nv_tensor_ir.bufferized_program, tile_size = array<i32: 4>} {
  %load = bufferization.to_tensor %input restrict
      : memref<16xf32> to tensor<16xf32>
  // expected-error @below {{bufferized TensorIR boundary must have a statically zero offset}}
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<16xf32>, memref<16xf32, strided<[1], offset: ?>>) -> ()
  return
}

// -----

func.func @non_argument_boundary(%output: memref<16xf32>)
    attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 4>
} {
  %buffer = memref.alloca() : memref<16xf32>
  // expected-error @below {{bufferized TensorIR boundary must be a memref function argument}}
  %load = bufferization.to_tensor %buffer restrict
      : memref<16xf32> to tensor<16xf32>
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

func.func @non_default_memory_space(%input: memref<16xf32, 1>,
                                    %output: memref<16xf32>)
    attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 4>
} {
  // expected-error @below {{bufferized TensorIR boundary must be in the default memory space}}
  %load = bufferization.to_tensor %input restrict
      : memref<16xf32, 1> to tensor<16xf32>
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}
