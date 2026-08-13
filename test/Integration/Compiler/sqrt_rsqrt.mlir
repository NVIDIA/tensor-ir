// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @sqrt_rsqrt_op_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %abs = abs %input : tensor<8x8xf32>
    %sqrt = sqrt %abs : tensor<8x8xf32>
    %rsqrt = rsqrt %sqrt : tensor<8x8xf32>
    results %rsqrt : tensor<8x8xf32>
  }
}
