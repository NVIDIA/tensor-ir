// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @abs_ceil_cos_floor_op_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %abs = abs %input : tensor<8x8xf32>
    %ceil = ceil %abs : tensor<8x8xf32>
    %cos = cos %ceil : tensor<8x8xf32>
    %floor = floor %cos : tensor<8x8xf32>
    results %floor : tensor<8x8xf32>
  }
}
