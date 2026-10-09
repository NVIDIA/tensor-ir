// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --split-input-file --verify-diagnostics

// Reject an overflowing reduction plan before attempting to load operands
// using its malformed descriptor metadata.
#reduction = #nv_tensor_ir.reduction_source<
    "(4,1,(3037000500,3037000500)):(1,0,(3037000500,1))",
    #nv_tensor_ir.tensor_source<0, 0, "(?,?):(?,1)", [0, 1, 2]>>
func.func @overflow(%input: memref<4x8xf32, #ptr.generic_space>,
    %output: memref<4x1xf32, strided<[1, 0]>, #ptr.generic_space>) attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(4,1):(1,0)">,
  tile_size = array<i32: 2, 1>
} {
  %input_tile = bufferization.to_tensor %input restrict
      : memref<4x8xf32, #ptr.generic_space> to tensor<4x8xf32>
  // expected-error @below {{reduction extent overflows the index range}}
  %result = nv_tensor_ir.reduce(%input_tile)
      <dimensions = [1], reduction_mode = <add>> {layout = #reduction}
      : tensor<4x8xf32> -> tensor<4x1xf32>
  bufferization.materialize_in_destination %result in restrict writable %output
      : (tensor<4x1xf32>, memref<4x1xf32, strided<[1, 0]>, #ptr.generic_space>)
        -> ()
  return
}
