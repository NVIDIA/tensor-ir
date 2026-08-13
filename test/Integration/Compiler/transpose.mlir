// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @transpose_op_simple(
    %input: tensor<2x4x4xf32> {nv_tensor_ir.stride = "(16,4,1)"}
  ) -> (tensor<4x2x4xf32> {nv_tensor_ir.stride = "(8,4,1)"}) {
    %transpose = transpose %input permutation = [2, 0, 1] : tensor<2x4x4xf32> -> tensor<4x2x4xf32>
    results %transpose : tensor<4x2x4xf32>
  }
}
