// RUN: tensor_ir-opt "-layout-propagation-pipeline=tile-size=4 tile-size=1024 reduction-tile-size=256" %s | FileCheck %s

module {
  nv_tensor_ir.graph @rmsnorm_reduction_tile(
    %input: tensor<16x1024xf32> {nv_tensor_ir.stride = "(1024,1)"}
  ) -> (tensor<16x1024xf32> {nv_tensor_ir.stride = "(1024,1)"}) {
    %input_squared = nv_tensor_ir.mul %input, %input : tensor<16x1024xf32>
    %sum_x2 = nv_tensor_ir.reduce(%input_squared) <
      dimensions = [1],
      reduction_mode = <add>> : tensor<16x1024xf32> -> tensor<16x1xf32>
    %broadcast = nv_tensor_ir.broadcast %sum_x2 : tensor<16x1xf32> -> tensor<16x1024xf32>
    nv_tensor_ir.results %broadcast : tensor<16x1024xf32>
  }
}

// CHECK: entry @rmsnorm_reduction_tile
// CHECK: constant <f32: 0.000000e+00> : tile<4x256xf32>
// CHECK: %[[FOR:.*]] = for %{{.*}} in (%{{.*}} to %{{.*}}, step %{{.*}}) : tile<i32> iter_values(%{{.*}}) -> (tile<4x256xf32>)
// CHECK: partition_view<tile=(4x256), tensor_view<16x1024xf32, strides=[1024,1]>>
// CHECK: reduce %[[FOR]] dim=1 identities=[0.000000e+00 : f32] : tile<4x256xf32> -> tile<4xf32>
// CHECK: broadcast %{{.*}} : tile<4x1xf32> -> tile<4x1024xf32>
