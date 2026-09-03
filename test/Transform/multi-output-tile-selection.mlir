// RUN: tensor_ir-opt -materialize-default-strides \
// RUN:   -layout-propagation-annotation -layout-propagation-normalization \
// RUN:   -tile-analyzer -tile-selection %s | FileCheck %s

// The compact result projects the D carrier dimension. Tile analysis must keep
// all of D in one CTA so exactly one tile owns each physical inv_rms value.
// CHECK-LABEL: nv_tensor_ir.graph @rmsnorm_auto_tile
// CHECK-SAME: attributes {tile_size = array<i32: 2, 16>}
// CHECK: results {{.*}} result_views = [
// CHECK-SAME: #nv_tensor_ir.tensor_source<1, 0, "(64,16):(16,1)">
// CHECK-SAME: #nv_tensor_ir.tensor_source<2, 0, "(64,16):(1,0)">
nv_tensor_ir.graph @rmsnorm_auto_tile(
    %x: tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"},
          tensor<64x1xf32> {nv_tensor_ir.stride = "(1,1)"}) {
  %sq = mul %x, %x : tensor<64x16xf32>
  %sum = reduce(%sq)<dimensions = [1], reduction_mode = <add>>
      : tensor<64x16xf32> -> tensor<64x1xf32>
  %wide = broadcast %sum : tensor<64x1xf32> -> tensor<64x16xf32>
  %y = mul %x, %wide : tensor<64x16xf32>
  results %y, %sum : tensor<64x16xf32>, tensor<64x1xf32>
}

// The fixed full-coverage constraint is not capped by the analyzer's ordinary
// fallback extent. A realistic larger D still selects one root tile over all
// of D; reduction lowering may iterate over smaller child tiles later.
// CHECK-LABEL: nv_tensor_ir.graph @rmsnorm_large_auto_tile
// CHECK-SAME: attributes {tile_size = array<i32: 1, 512>}
nv_tensor_ir.graph @rmsnorm_large_auto_tile(
    %x: tensor<8x512xf32> {nv_tensor_ir.stride = "(512,1)"}
    ) -> (tensor<8x512xf32> {nv_tensor_ir.stride = "(512,1)"},
          tensor<8x1xf32> {nv_tensor_ir.stride = "(1,1)"}) {
  %sq = mul %x, %x : tensor<8x512xf32>
  %sum = reduce(%sq)<dimensions = [1], reduction_mode = <add>>
      : tensor<8x512xf32> -> tensor<8x1xf32>
  %wide = broadcast %sum : tensor<8x1xf32> -> tensor<8x512xf32>
  %y = mul %x, %wide : tensor<8x512xf32>
  results %y, %sum : tensor<8x512xf32>, tensor<8x1xf32>
}
