// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @constant_splat_op_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %constant = nv_tensor_ir.constant 2.5 : f32
    %splat = nv_tensor_ir.splat %constant : tensor<8x8xf32>
    %output = add %input, %splat : tensor<8x8xf32>
    results %output : tensor<8x8xf32>
  }
}
