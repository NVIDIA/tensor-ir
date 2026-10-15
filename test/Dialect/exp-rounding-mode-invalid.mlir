// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics

nv_tensor_ir.graph @approx_f64(%x: tensor<128xf64>) -> tensor<128xf64> {
  // expected-error @below {{approx rounding_mode requires f16, bf16, or f32}}
  %y = exp %x <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<approx>}> : tensor<128xf64>
  results %y : tensor<128xf64>
}

// -----

nv_tensor_ir.graph @explicit_fp8(%x: tensor<128xf8E4M3FN>) -> tensor<128xf8E4M3FN> {
  // expected-error @below {{rounding_mode requires f16, bf16, f32, or f64}}
  %y = exp %x <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<full>}> : tensor<128xf8E4M3FN>
  results %y : tensor<128xf8E4M3FN>
}

// -----

nv_tensor_ir.graph @wrong_attribute_type(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @below {{Invalid attribute `rounding_mode` in property conversion}}
  %y = exp %x <{rounding_mode = "approx"}> : tensor<128xf32>
  results %y : tensor<128xf32>
}

// -----

nv_tensor_ir.graph @unsupported_rounding(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+2 {{expected ::mlir::nv_tensor_ir::ExpRoundingMode to be one of: full, approx}}
  // expected-error @+1 {{failed to parse ExpRoundingModeAttr parameter 'value'}}
  %y = exp %x <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<nearest_even>}> : tensor<128xf32>
  results %y : tensor<128xf32>
}
