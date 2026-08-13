// RUN: tensor_ir-opt %s -materialize-default-strides \
// RUN:   -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   -verify-diagnostics

nv_tensor_ir.graph @matmul_accumulator_affine_unsupported(
    %a: tensor<32x128xf32>,
    %b: tensor<128x16xf32>,
    %acc: tensor<32x16xf32>
    ) -> tensor<32x16xf32> {
  // expected-error@below {{custom accumulator is not supported by TensorToCudaTile}}
  %out = matmul(%a, %b) accum(%acc : tensor<32x16xf32>)
    : (tensor<32x128xf32>, tensor<128x16xf32>) -> tensor<32x16xf32>
  results %out : tensor<32x16xf32>
}
