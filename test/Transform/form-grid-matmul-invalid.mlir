// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --verify-diagnostics

// Validate the view rank before interpreting its dimension map against the
// output tile; a contracting dimension must not be read as an output axis.
#matmul = #nv_tensor_ir.matmul_source<"(3,6,4,8):(0,8,48,1)", 1, 4, 6, 8,
    #nv_tensor_ir.tensor_source<0, 0, "(4,8):(8,1)">,
    #nv_tensor_ir.tensor_source<1, 0, "(8,6):(6,1)">>
#bad = #nv_tensor_ir.matmul_source<"(4,6,8):(48,8,1)", 1, 4, 6, 8,
    #nv_tensor_ir.tensor_source<0, 0, "(4,8):(8,1)">,
    #nv_tensor_ir.tensor_source<1, 0, "(8,6):(6,1)">>
func.func @mismatched_matmul_rank(%lhs: !ptr.ptr<#ptr.generic_space>,
    %rhs: !ptr.ptr<#ptr.generic_space>, %output: !ptr.ptr<#ptr.generic_space>)
    attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #matmul,
  tile_size = array<i32: 2, 2, 2>
} {
  %zero = arith.constant 0 : index
  %a = nv_tensor_ir.load %lhs[%zero, %zero, %zero]
      view offset: [0], sizes: [1, 4, 8], strides: [32, 8, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<1x4x8xf32>
  %b = nv_tensor_ir.load %rhs[%zero, %zero, %zero]
      view offset: [0], sizes: [1, 8, 6], strides: [48, 6, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<1x8x6xf32>
  // expected-error @below {{matmul view rank does not match its tile}}
  %product = nv_tensor_ir.matmul(%a, %b) {layout = #bad}
      : (tensor<1x4x8xf32>, tensor<1x8x6xf32>) -> tensor<1x4x6xf32>
  %transposed = nv_tensor_ir.transpose %product permutation = [0, 2, 1]
      {layout = #matmul} : tensor<1x4x6xf32> -> tensor<1x6x4xf32>
  %result = nv_tensor_ir.broadcast %transposed {layout = #matmul}
      : tensor<1x6x4xf32> -> tensor<3x6x4xf32>
  nv_tensor_ir.store %result, %output[%zero, %zero, %zero]
      view offset: [0], sizes: [3, 6, 4], strides: [24, 4, 1], alignment: 4
      : tensor<3x6x4xf32>, !ptr.ptr<#ptr.generic_space>
  return
}
