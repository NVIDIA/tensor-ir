// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @convert_supported_float_types(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    // This chain covers every supported ordered pair over f32/f16/bf16/f64
    // exactly once while keeping layout-propagation lowering single-output.
    %f32_to_f32 = convert %input : tensor<8x8xf32> -> tensor<8x8xf32>
    %f32_to_f16 = convert %f32_to_f32 : tensor<8x8xf32> -> tensor<8x8xf16>
    %f16_to_f32 = convert %f32_to_f16 : tensor<8x8xf16> -> tensor<8x8xf32>
    %f32_to_bf16 = convert %f16_to_f32 : tensor<8x8xf32> -> tensor<8x8xbf16>
    %bf16_to_f32 = convert %f32_to_bf16 : tensor<8x8xbf16> -> tensor<8x8xf32>
    %f32_to_f64 = convert %bf16_to_f32 : tensor<8x8xf32> -> tensor<8x8xf64>
    %f64_to_f16 = convert %f32_to_f64 : tensor<8x8xf64> -> tensor<8x8xf16>
    %f16_to_f16 = convert %f64_to_f16 : tensor<8x8xf16> -> tensor<8x8xf16>
    %f16_to_bf16 = convert %f16_to_f16 : tensor<8x8xf16> -> tensor<8x8xbf16>
    %bf16_to_f16 = convert %f16_to_bf16 : tensor<8x8xbf16> -> tensor<8x8xf16>
    %f16_to_f64 = convert %bf16_to_f16 : tensor<8x8xf16> -> tensor<8x8xf64>
    %f64_to_bf16 = convert %f16_to_f64 : tensor<8x8xf64> -> tensor<8x8xbf16>
    %bf16_to_bf16 = convert %f64_to_bf16 : tensor<8x8xbf16> -> tensor<8x8xbf16>
    %bf16_to_f64 = convert %bf16_to_bf16 : tensor<8x8xbf16> -> tensor<8x8xf64>
    %f64_to_f64 = convert %bf16_to_f64 : tensor<8x8xf64> -> tensor<8x8xf64>
    %f64_to_f32 = convert %f64_to_f64 : tensor<8x8xf64> -> tensor<8x8xf32>

    results %f64_to_f32 : tensor<8x8xf32>
  }
}
