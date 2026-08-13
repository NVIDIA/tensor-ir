// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @reshape_op_simple(
    %input: tensor<4x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x4xf32> {nv_tensor_ir.stride = "(4,1)"}) {
    %reshape = reshape %input : tensor<4x8xf32> -> tensor<8x4xf32>
    results %reshape : tensor<8x4xf32>
  }
}
