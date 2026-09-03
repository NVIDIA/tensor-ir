// RUN: tensor_ir-opt -materialize-default-strides \
// RUN:   -layout-propagation-annotation -layout-propagation-normalization \
// RUN:   -tile-analyzer -tile-selection -verify-diagnostics=only-expected %s

// expected-error @+1 {{Selected tile [2, 8] partitions result-projected iteration-space dimension 1; expected full tile size 16}}
nv_tensor_ir.graph @partitioned_projected_result(
    %x: tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"},
          tensor<64x1xf32> {nv_tensor_ir.stride = "(1,1)"})
    attributes {tile_size = array<i32: 2, 8>} {
  %sq = mul %x, %x : tensor<64x16xf32>
  %sum = reduce(%sq)<dimensions = [1], reduction_mode = <add>>
      : tensor<64x16xf32> -> tensor<64x1xf32>
  %wide = broadcast %sum : tensor<64x1xf32> -> tensor<64x16xf32>
  %y = mul %x, %wide : tensor<64x16xf32>
  results %y, %sum : tensor<64x16xf32>, tensor<64x1xf32>
}
