// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --split-input-file --verify-diagnostics

// Reject an overflowing reduction plan before attempting to load operands
// using its malformed descriptor metadata.
#reduction = #nv_tensor_ir.reduction_source<
    "(4,1,(3037000500,3037000500)):(1,0,(3037000500,1))",
    #nv_tensor_ir.tensor_source<0, 0, "(?,?):(?,1)", [0, 1, 2]>>
func.func @overflow(%input: !ptr.ptr<#ptr.generic_space>,
    %output: !ptr.ptr<#ptr.generic_space>) attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(4,1):(1,0)">,
  tile_size = array<i32: 2, 1>
} {
  %zero = arith.constant 0 : index
  %input_tile = nv_tensor_ir.load %input[%zero, %zero]
      view offset: [0], sizes: [4, 8], strides: [8, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<4x8xf32>
  // expected-error @below {{reduction extent overflows the index range}}
  %result = nv_tensor_ir.reduce(%input_tile)
      <dimensions = [1], reduction_mode = <add>> {layout = #reduction}
      : tensor<4x8xf32> -> tensor<4x1xf32>
  nv_tensor_ir.store %result, %output[%zero, %zero]
      view offset: [0], sizes: [4, 1], strides: [1, 0], alignment: 4
      : tensor<4x1xf32>, !ptr.ptr<#ptr.generic_space>
  return
}
