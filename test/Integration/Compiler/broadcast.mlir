// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @broadcast_op_simple(
    %input: tensor<1x3xf32> {nv_tensor_ir.stride = "(3,1)"}
  ) -> (tensor<2x3xf32> {nv_tensor_ir.stride = "(3,1)"}) {
    %broadcast = broadcast %input : tensor<1x3xf32> -> tensor<2x3xf32>
    results %broadcast : tensor<2x3xf32>
  }
}
