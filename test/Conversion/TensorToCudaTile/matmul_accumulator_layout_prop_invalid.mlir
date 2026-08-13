// RUN: tensor_ir-opt %s -layout-propagation-pipeline -verify-diagnostics

nv_tensor_ir.graph @matmul_accumulator_layout_prop_unsupported(
    %a: tensor<32x128xf32>,
    %b: tensor<128x16xf32>,
    %acc: tensor<32x16xf32>
    ) -> tensor<32x16xf32> {
  // expected-error@below {{custom accumulator is not supported by layout propagation}}
  %out = matmul(%a, %b) accum(%acc : tensor<32x16xf32>)
    : (tensor<32x128xf32>, tensor<128x16xf32>) -> tensor<32x16xf32>
  results %out : tensor<32x16xf32>
}
