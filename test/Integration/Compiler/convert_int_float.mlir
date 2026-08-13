// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @convert_int_float(
    %input: tensor<8x8xsi32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xsi32> {nv_tensor_ir.stride = "(8,1)"}) {
    // This chain covers the supported ordered pairs that cross between si32
    // and the floating-point reference types. Float-to-float conversions are
    // covered by convert_supported_float_types.mlir.
    %i32_to_i32 = convert %input : tensor<8x8xsi32> -> tensor<8x8xsi32>
    %i32_to_f32 = convert %i32_to_i32 : tensor<8x8xsi32> -> tensor<8x8xf32>
    %f32_to_i32 = convert %i32_to_f32 : tensor<8x8xf32> -> tensor<8x8xsi32>
    %i32_to_f16 = convert %f32_to_i32 : tensor<8x8xsi32> -> tensor<8x8xf16>
    %f16_to_i32 = convert %i32_to_f16 : tensor<8x8xf16> -> tensor<8x8xsi32>
    %i32_to_bf16 = convert %f16_to_i32 : tensor<8x8xsi32> -> tensor<8x8xbf16>
    %bf16_to_i32 = convert %i32_to_bf16 : tensor<8x8xbf16> -> tensor<8x8xsi32>
    %i32_to_f64 = convert %bf16_to_i32 : tensor<8x8xsi32> -> tensor<8x8xf64>
    %f64_to_i32 = convert %i32_to_f64 : tensor<8x8xf64> -> tensor<8x8xsi32>

    results %f64_to_i32 : tensor<8x8xsi32>
  }
}
