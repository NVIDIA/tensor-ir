// RUN: tensor_ir-compiler %s --codegen-strategy=layout-propagation --verbose --launch --verify

module {
  nv_tensor_ir.graph @log_sin_tan_reciprocal_op_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %abs = abs %input : tensor<8x8xf32>
    %constant = nv_tensor_ir.constant 1.25 : f32
    %splat = nv_tensor_ir.splat %constant : tensor<8x8xf32>
    %shifted = add %abs, %splat : tensor<8x8xf32>
    %log = log %shifted : tensor<8x8xf32>
    %sin = sin %log : tensor<8x8xf32>
    %tan = tan %sin : tensor<8x8xf32>
    %reciprocal = reciprocal %tan : tensor<8x8xf32>
    results %reciprocal : tensor<8x8xf32>
  }
}
