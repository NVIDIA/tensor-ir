// RUN: tensor_ir-compiler %s --tile-size=4x4 --verbose --launch --verify

module {
  nv_tensor_ir.graph @slice_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<4x4xf32> {nv_tensor_ir.stride = "(4,1)"}) {
    %slice = slice %input starts = [2, 1] limits = [6, 5] strides = [1, 1] : tensor<8x8xf32> -> tensor<4x4xf32>
    results %slice : tensor<4x4xf32>
  }
}
