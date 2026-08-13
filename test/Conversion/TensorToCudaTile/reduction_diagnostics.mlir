// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file -verify-diagnostics=only-expected %s

nv_tensor_ir.graph @test_reduce_norm2_int(
    %arg0: tensor<64x16xsi32>
    ) -> (tensor<64x1xsi32>) {
  // expected-error @+1 {{Unsupported reduction mode}}
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <norm2>>
    : tensor<64x16xsi32> -> tensor<64x1xsi32>
  results %out : tensor<64x1xsi32>
}
