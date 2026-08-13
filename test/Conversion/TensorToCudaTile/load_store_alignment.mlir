// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// CHECK-LABEL: @alignment_explicit
// CHECK-SAME: %[[PTR_IN:.*]]: tile<ptr<f32>>, %[[PTR_OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[PTR_IN_ALIGNED:.*]] = assume div_by<128>, %[[PTR_IN]]
// CHECK: make_tensor_view %[[PTR_IN_ALIGNED]]
// CHECK: %[[PTR_OUT_ALIGNED:.*]] = assume div_by<128>, %[[PTR_OUT]]
// CHECK: make_tensor_view %[[PTR_OUT_ALIGNED]]
nv_tensor_ir.graph @alignment_explicit(
    %arg0: tensor<64xf32> {nv_tensor_ir.alignment = 128}) ->
    (tensor<64xf32> {nv_tensor_ir.alignment = 128})
    attributes {tile_size = array<i32: 64>} {
  %out = neg %arg0 : tensor<64xf32>
  results %out : tensor<64xf32>
}

// -----

// CHECK-LABEL: @alignment_implicit
// CHECK-SAME: %[[PTR_IN:.*]]: tile<ptr<f32>>, %[[PTR_OUT:.*]]: tile<ptr<f32>>
// CHECK: make_tensor_view %[[PTR_IN]]
// CHECK: make_tensor_view %[[PTR_OUT]]
nv_tensor_ir.graph @alignment_implicit(
    %arg0: tensor<64xf32>) ->
    (tensor<64xf32>)
    attributes {tile_size = array<i32: 64>} {
  %out = neg %arg0 : tensor<64xf32>
  results %out : tensor<64xf32>
}

// -----

// CHECK-LABEL: @alignment_smaller_than_element_size
// CHECK-SAME: %[[PTR_IN:.*]]: tile<ptr<f32>>, %[[PTR_OUT:.*]]: tile<ptr<f32>>
// CHECK: make_tensor_view %[[PTR_IN]]
// CHECK: make_tensor_view %[[PTR_OUT]]
nv_tensor_ir.graph @alignment_smaller_than_element_size(
    %arg0: tensor<64xf32> {nv_tensor_ir.alignment = 2}) ->
    (tensor<64xf32> {nv_tensor_ir.alignment = 4})
    attributes {tile_size = array<i32: 64>} {
  %out = neg %arg0 : tensor<64xf32>
  results %out : tensor<64xf32>
}

// -----

// CHECK-LABEL: @alignment_updated_by_offset
// CHECK-SAME: %[[PTR_IN:.*]]: tile<ptr<f32>>, %[[PTR_OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[PTR_IN_OFFSET:.*]] = offset %[[PTR_IN]], %{{.*}}
// CHECK: %[[PTR_IN_ALIGNED:.*]] = assume div_by<16>, %[[PTR_IN_OFFSET]]
// CHECK: make_tensor_view %[[PTR_IN_ALIGNED]]
// CHECK: %[[PTR_OUT_ALIGNED:.*]] = assume div_by<128>, %[[PTR_OUT]]
// CHECK: make_tensor_view %[[PTR_OUT_ALIGNED]]
nv_tensor_ir.graph @alignment_updated_by_offset(
    %arg0: tensor<64xf32> {nv_tensor_ir.alignment = 128}) ->
    (tensor<32xf32> {nv_tensor_ir.alignment = 128})
    attributes {tile_size = array<i32: 32>} {
  %out = slice %arg0 starts = [4] limits = [36] strides = [1] : tensor<64xf32> -> tensor<32xf32>
  results %out : tensor<32xf32>
}
