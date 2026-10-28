// RUN: tensor_ir-opt -materialize-default-strides --tir-rewrite-unsupported-ops -split-input-file -verify-diagnostics %s

nv_tensor_ir.graph @matmul_invalid_mixed_f16_bf16(
  %a: tensor<128x64xf16>,
  %b: tensor<64x128xbf16>) -> (
    tensor<128x128xf32>) {
  // expected-error@below {{unsupported input element types}}
  %c = matmul(%a, %b) : (tensor<128x64xf16>, tensor<64x128xbf16>) -> tensor<128x128xf32>
  results %c : tensor<128x128xf32>
}

// -----

nv_tensor_ir.graph @matmul_invalid_i16(
  %a: tensor<128x64xsi16>,
  %b: tensor<64x128xsi16>) -> (
    tensor<128x128xsi32>) {
  // expected-error@below {{unsupported input element types}}
  %c = matmul(%a, %b) : (tensor<128x64xsi16>, tensor<64x128xsi16>) -> tensor<128x128xsi32>
  results %c : tensor<128x128xsi32>
}

// -----

nv_tensor_ir.graph @matmul_invalid_mixed_i8_f16(
  %a: tensor<128x64xsi8>,
  %b: tensor<64x128xf16>) -> (
    tensor<128x128xf32>) {
  // expected-error@below {{unsupported input element types}}
  %c = matmul(%a, %b) : (tensor<128x64xsi8>, tensor<64x128xf16>) -> tensor<128x128xf32>
  results %c : tensor<128x128xf32>
}
