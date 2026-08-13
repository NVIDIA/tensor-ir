// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @reduce_add(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x1xf32> {nv_tensor_ir.stride = "(1,1)"}) {
    %sum = reduce(%input)<
      dimensions = [1], reduction_mode = <add>>
      : tensor<8x8xf32> -> tensor<8x1xf32>
    results %sum : tensor<8x1xf32>
  }
}
