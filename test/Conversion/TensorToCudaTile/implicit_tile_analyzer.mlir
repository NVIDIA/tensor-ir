// RUN: tensor_ir-opt -materialize-default-strides -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s

// CHECK: tensor_ir.resolved_tile_size = array<i32: 1, 512>
// CHECK-LABEL: entry @implicit_row_major_tile_analyzer
// CHECK: make_partition_view {{.*}} : partition_view<tile=(1x512), tensor_view<1024x1024xf32, strides=[1024,1]>>
module {
  nv_tensor_ir.graph @implicit_row_major_tile_analyzer(
      %arg0: tensor<1024x1024xf32>)
      -> tensor<1024x1024xf32> {
    %abs = abs %arg0 : tensor<1024x1024xf32>
    results %abs : tensor<1024x1024xf32>
  }
}
