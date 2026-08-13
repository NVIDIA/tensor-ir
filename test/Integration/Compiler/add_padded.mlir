// RUN: tensor_ir-compiler %s --verbose --launch --verify
//
// Padded row-major layout: shape 8x8 but row stride 16.
// Logical elements: 8*8 = 64; footprint: (7*16 + 7*1) + 1 = 120 elements.
// So num_elements * sizeof(f32) < buffer bytes required for the strided tensor.

module {
  nv_tensor_ir.graph @add_op_padded(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(16,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(16,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(16,1)"}) {
    %add = add %a, %b : tensor<8x8xf32>
    results %add : tensor<8x8xf32>
  }
}
