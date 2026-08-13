// RUN: tensor_ir-opt --layout-propagation-pipeline -split-input-file %s | FileCheck %s

// CHECK-LABEL: entry @broadcast_one_dimension
// CHECK: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK: %[[TVIEW1:.*]] = make_tensor_view {{.*}}, shape = [32, 1], strides = [1, 1] : tensor_view<32x1xf32, strides=[1,1]>
// CHECK: %[[PVIEW1:.*]] = make_partition_view %[[TVIEW1]] : partition_view<tile=(1x1), tensor_view<32x1xf32, strides=[1,1]>>
// CHECK: %[[LOAD1:.*]], {{.*}} = load_view_tko weak %[[PVIEW1]][{{.*}}, %[[ZERO]]]
// CHECK: %[[TILE1:.*]] = broadcast %[[LOAD1]] : tile<1x1xf32> -> tile<1x32xf32>
// CHECK: %[[TVIEW2:.*]] = make_tensor_view {{.*}}, shape = [32, 32], strides = [32, 1] : tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[PVIEW2:.*]] = make_partition_view %[[TVIEW2]] : partition_view<tile=(1x32), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[TILE2:.*]], {{.*}} = load_view_tko weak %[[PVIEW2]][{{.*}}, {{.*}}]
// CHECK: %[[ADD:.*]] = addf %[[TILE1]], %[[TILE2]]  : tile<1x32xf32>
// CHECK: store_view_tko weak %[[ADD]]
module {
  nv_tensor_ir.graph @broadcast_one_dimension(
      %arg0: tensor<32x1xf32>,
      %arg1: tensor<32x32xf32>)
      -> (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 1, 32>} {
    %bcast = broadcast %arg0 : tensor<32x1xf32> -> tensor<32x32xf32>
    %add = add %bcast, %arg1 : tensor<32x32xf32>
    results %add : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: entry @broadcast_two_dimensions
// CHECK: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK: %[[TVIEW1:.*]] = make_tensor_view {{.*}}, shape = [1, 1], strides = [1, 1] : tensor_view<1x1xf32, strides=[1,1]>
// CHECK: %[[PVIEW1:.*]] = make_partition_view %[[TVIEW1]] : partition_view<tile=(1x1), tensor_view<1x1xf32, strides=[1,1]>>
// CHECK: %[[LOAD1:.*]], {{.*}} = load_view_tko weak %[[PVIEW1]][%[[ZERO]], %[[ZERO]]]
// CHECK: %[[TILE1:.*]] = broadcast %[[LOAD1]] : tile<1x1xf32> -> tile<32x1xf32>
// CHECK: %[[TVIEW2:.*]] = make_tensor_view {{.*}}, shape = [32, 32], strides = [1, 32] : tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[PVIEW2:.*]] = make_partition_view %[[TVIEW2]] : partition_view<tile=(32x1), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[TILE2:.*]], {{.*}} = load_view_tko weak %[[PVIEW2]][{{.*}}, {{.*}}]
// CHECK: %[[ADD:.*]] = addf %[[TILE1]], %[[TILE2]]  : tile<32x1xf32>
// CHECK: store_view_tko weak %[[ADD]]
module {
  nv_tensor_ir.graph @broadcast_two_dimensions(
      %arg0: tensor<1x1xf32>,
      %arg1: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"})
      -> (tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"})
      attributes {tile_size = array<i32: 32, 1>} {
    %bcast = broadcast %arg0 : tensor<1x1xf32> -> tensor<32x32xf32>
    %add = add %bcast, %arg1 : tensor<32x32xf32>
    results %add : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: entry @broadcast_two_dimensions_col_major_output
// CHECK: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK: %[[TVIEW1:.*]] = make_tensor_view {{.*}}, shape = [1], strides = [1] : tensor_view<1xf32, strides=[1]>
// CHECK: %[[PVIEW1:.*]] = make_partition_view %[[TVIEW1]] : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>
// CHECK: %[[LOAD1:.*]], {{.*}} = load_view_tko weak %[[PVIEW1]][%[[ZERO]]]
// CHECK: %[[TILE1:.*]] = broadcast %[[LOAD1]] : tile<1xf32> -> tile<32xf32>
// CHECK: %[[TVIEW2:.*]] = make_tensor_view {{.*}}, shape = [1024], strides = [1] : tensor_view<1024xf32, strides=[1]>
// CHECK: %[[PVIEW2:.*]] = make_partition_view %[[TVIEW2]] : partition_view<tile=(32), tensor_view<1024xf32, strides=[1]>>
// CHECK: %[[TILE2:.*]], {{.*}} = load_view_tko weak %[[PVIEW2]][{{.*}}]
// CHECK: %[[ADD:.*]] = addf %[[TILE1]], %[[TILE2]]  : tile<32xf32>
// CHECK: store_view_tko weak %[[ADD]]
nv_tensor_ir.graph @broadcast_two_dimensions_col_major_output(
    %arg0: tensor<1x1xf32>,
    %arg1: tensor<32x32xf32>)
    -> (tensor<32x32xf32>)
    attributes {tile_size = array<i32: 32>} {
  %bcast = broadcast %arg0 : tensor<1x1xf32> -> tensor<32x32xf32>
  %add = add %bcast, %arg1 : tensor<32x32xf32>
  results %add : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: entry @broadcast_same_input
// CHECK: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK: %[[TVIEW1:.*]] = make_tensor_view {{.*}}, shape = [16, 1], strides = [1, 1] : tensor_view<16x1xf32, strides=[1,1]>
// CHECK: %[[PVIEW1:.*]] = make_partition_view %[[TVIEW1]] : partition_view<tile=(1x1), tensor_view<16x1xf32, strides=[1,1]>>
// CHECK: %[[LOAD1:.*]], {{.*}} = load_view_tko weak %[[PVIEW1]][{{.*}}, %[[ZERO]]]
// CHECK: %[[TILE1:.*]] = broadcast %[[LOAD1]] : tile<1x1xf32> -> tile<1x16xf32>
// CHECK: %[[TVIEW2:.*]] = make_tensor_view {{.*}}, shape = [1, 16], strides = [1, 1] : tensor_view<1x16xf32, strides=[1,1]>
// CHECK: %[[PVIEW2:.*]] = make_partition_view %[[TVIEW2]] : partition_view<tile=(1x16), tensor_view<1x16xf32, strides=[1,1]>>
// CHECK: %[[TILE2:.*]], {{.*}} = load_view_tko weak %[[PVIEW2]][%[[ZERO]], {{.*}}]
// CHECK: %[[ADD:.*]] = addf %[[TILE1]], %[[TILE2]] : tile<1x16xf32>
// CHECK: store_view_tko weak %[[ADD]]
module {
  nv_tensor_ir.graph @broadcast_same_input(
      %arg0: tensor<16xf32>)
      -> (tensor<16x16xf32>)
      attributes {tile_size = array<i32: 1, 16>} {
    %r1 = reshape %arg0 : tensor<16xf32> -> tensor<16x1xf32>
    %r2 = reshape %arg0 : tensor<16xf32> -> tensor<1x16xf32>
    %b1 = broadcast %r1 : tensor<16x1xf32> -> tensor<16x16xf32>
    %b2 = broadcast %r2 : tensor<1x16xf32> -> tensor<16x16xf32>
    %add = add %b1, %b2 : tensor<16x16xf32>
    results %add : tensor<16x16xf32>
  }
}
