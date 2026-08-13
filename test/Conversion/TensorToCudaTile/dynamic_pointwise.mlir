// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// TEST 1: Dynamic shape / static stride, rank 1
// ============================================================================
// CHECK-LABEL: entry @dynamic_shape_static_stride_rank_1
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^,]*]]: tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [%[[M0]]], strides = [1]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[BLOCK]]]
// CHECK: make_tensor_view %[[OUT]], shape = [%[[MR]]], strides = [1]
module {
  nv_tensor_ir.graph @dynamic_shape_static_stride_rank_1(
      %arg0: tensor<?xf32> {nv_tensor_ir.stride = "(1)"}) ->
      (tensor<?xf32> {nv_tensor_ir.stride = "(1)"})
      attributes {tile_size = array<i32: 32>} {
    %res = cos %arg0 : tensor<?xf32>
    results %res : tensor<?xf32>
  }
}

// -----

// ============================================================================
// TEST 2: Dynamic shape / static stride, rank 3
// ============================================================================
// CHECK-LABEL: entry @dynamic_shape_static_stride_rank_3
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>, %[[N0:[^,]*]]: tile<i32>, %[[K0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^,]*]]: tile<i32>, %[[NR:[^,]*]]: tile<i32>, %[[KR:[^,]*]]: tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]], %[[KR]]]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: %[[INDEX:.*]]:3 = get_index_space_shape %[[PVIEW]]
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[TEMP:.*]] = divi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[IDX1:.*]] = remi %[[TEMP]], %[[INDEX]]#1 unsigned
// CHECK: %[[IDX2:.*]] = divi %[[TEMP]], %[[INDEX]]#1 unsigned
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [%[[M0]], %[[N0]], %[[K0]]], strides = [16384, 128, 1]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[IDX0]], %[[IDX1]], %[[IDX2]]]
// CHECK: make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]], %[[KR]]], strides = [16384, 128, 1]
module {
  nv_tensor_ir.graph @dynamic_shape_static_stride_rank_3(
      %arg0: tensor<?x?x?xf32> {nv_tensor_ir.stride = "(16384,128,1)"}) ->
      (tensor<?x?x?xf32> {nv_tensor_ir.stride = "(16384,128,1)"})
      attributes {tile_size = array<i32: 8, 8, 8>} {
    %res = cos %arg0 : tensor<?x?x?xf32>
    results %res : tensor<?x?x?xf32>
  }
}

// -----

// ============================================================================
// TEST 3: Static shape / dynamic stride, rank 1
// ============================================================================
// CHECK-LABEL: entry @static_shape_dynamic_stride_rank_1
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[S0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[SR:[^,]*]]: tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [1024], strides = [%[[S0]]]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[BLOCK]]]
// CHECK: make_tensor_view %[[OUT]], shape = [1024], strides = [%[[SR]]]
module {
  nv_tensor_ir.graph @static_shape_dynamic_stride_rank_1(
      %arg0: tensor<1024xf32> {nv_tensor_ir.stride = "(?)"}) ->
      (tensor<1024xf32> {nv_tensor_ir.stride = "(?)"})
      attributes {tile_size = array<i32: 32>} {
    %res = cos %arg0 : tensor<1024xf32>
    results %res : tensor<1024xf32>
  }
}

// -----

// ============================================================================
// TEST 4: Static shape / dynamic stride, rank 3
// ============================================================================
// CHECK-LABEL: entry @static_shape_dynamic_stride_rank_3
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[S0:[^,]*]]: tile<i32>, %[[T0:[^,]*]]: tile<i32>, %[[U0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[SR:[^,]*]]: tile<i32>, %[[TR:[^,]*]]: tile<i32>, %[[UR:[^,]*]]: tile<i32>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK-DAG: %[[C4:.*]] = constant <i32: 4>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[C4]] unsigned
// CHECK: %[[TEMP:.*]] = divi %[[BLOCK]], %[[C4]] unsigned
// CHECK: %[[IDX1:.*]] = remi %[[TEMP]], %[[C8]] unsigned
// CHECK: %[[IDX2:.*]] = divi %[[TEMP]], %[[C8]] unsigned
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [32, 64, 128], strides = [%[[S0]], %[[T0]], %[[U0]]]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[IDX0]], %[[IDX1]], %[[IDX2]]]
// CHECK: make_tensor_view %[[OUT]], shape = [32, 64, 128], strides = [%[[SR]], %[[TR]], %[[UR]]]
module {
  nv_tensor_ir.graph @static_shape_dynamic_stride_rank_3(
      %arg0: tensor<32x64x128xf32> {nv_tensor_ir.stride = "(?,?,?)"}) ->
      (tensor<32x64x128xf32> {nv_tensor_ir.stride = "(?,?,?)"})
      attributes {tile_size = array<i32: 8, 8, 8>} {
    %res = cos %arg0 : tensor<32x64x128xf32>
    results %res : tensor<32x64x128xf32>
  }
}

// -----

// ============================================================================
// TEST 5: Dynamic shape / implicit stride, rank 1
// ============================================================================
// CHECK-LABEL: entry @dynamic_shape_implicit_stride_rank_1
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^,]*]]: tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [%[[M0]]], strides = [1]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[BLOCK]]]
// CHECK: make_tensor_view %[[OUT]], shape = [%[[MR]]], strides = [1]
module {
  nv_tensor_ir.graph @dynamic_shape_implicit_stride_rank_1(
      %arg0: tensor<?xf32>) ->
      (tensor<?xf32>)
      attributes {tile_size = array<i32: 32>} {
    %res = cos %arg0 : tensor<?xf32>
    results %res : tensor<?xf32>
  }
}

// -----

// ============================================================================
// TEST 6: Dynamic shape / implicit stride, rank 3
// ============================================================================
// CHECK-LABEL: entry @dynamic_shape_implicit_stride_rank_3
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>, %[[N0:[^,]*]]: tile<i32>, %[[K0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^,]*]]: tile<i32>, %[[NR:[^,]*]]: tile<i32>, %[[KR:[^,]*]]: tile<i32>
// CHECK-DAG: %[[S0:.*]] = muli %[[K0]], %[[N0]]
// CHECK-DAG: %[[SR:.*]] = muli %[[KR]], %[[NR]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]], %[[KR]]]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: %[[INDEX:.*]]:3 = get_index_space_shape %[[PVIEW]]
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[TEMP:.*]] = divi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[IDX1:.*]] = remi %[[TEMP]], %[[INDEX]]#1 unsigned
// CHECK: %[[IDX2:.*]] = divi %[[TEMP]], %[[INDEX]]#1 unsigned
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [%[[M0]], %[[N0]], %[[K0]]], strides = [%[[S0]], %[[K0]], 1]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[IDX0]], %[[IDX1]], %[[IDX2]]]
// CHECK: make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]], %[[KR]]], strides = [%[[SR]], %[[KR]], 1]
module {
  nv_tensor_ir.graph @dynamic_shape_implicit_stride_rank_3(
      %arg0: tensor<?x?x?xf32>) ->
      (tensor<?x?x?xf32>)
      attributes {tile_size = array<i32: 8, 8, 8>} {
    %res = cos %arg0 : tensor<?x?x?xf32>
    results %res : tensor<?x?x?xf32>
  }
}

// -----

// ============================================================================
// TEST 7: Dynamic shape and stride, rank 1
// ============================================================================
// CHECK-LABEL: entry @dynamic_shape_and_stride_rank_1
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>, %[[S0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^,]*]]: tile<i32>, %[[SR:[^,]*]]: tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [%[[M0]]], strides = [%[[S0]]]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[BLOCK]]]
// CHECK: make_tensor_view %[[OUT]], shape = [%[[MR]]], strides = [%[[SR]]]
module {
  nv_tensor_ir.graph @dynamic_shape_and_stride_rank_1(
      %arg0: tensor<?xf32> {nv_tensor_ir.stride = "(?)"}) ->
      (tensor<?xf32> {nv_tensor_ir.stride = "(?)"})
      attributes {tile_size = array<i32: 32>} {
    %res = cos %arg0 : tensor<?xf32>
    results %res : tensor<?xf32>
  }
}

// -----

// ============================================================================
// TEST 8: Dynamic shape and stride, rank 3
// ============================================================================
// CHECK-LABEL: entry @dynamic_shape_and_stride_rank_3
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>, %[[N0:[^,]*]]: tile<i32>, %[[K0:[^,]*]]: tile<i32>, %[[S0:[^,]*]]: tile<i32>, %[[T0:[^,]*]]: tile<i32>, %[[U0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^,]*]]: tile<i32>, %[[NR:[^,]*]]: tile<i32>, %[[KR:[^,]*]]: tile<i32>, %[[SR:[^,]*]]: tile<i32>, %[[TR:[^,]*]]: tile<i32>, %[[UR:[^,]*]]: tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]], %[[KR]]]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: %[[INDEX:.*]]:3 = get_index_space_shape %[[PVIEW]]
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[TEMP:.*]] = divi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[IDX1:.*]] = remi %[[TEMP]], %[[INDEX]]#1 unsigned
// CHECK: %[[IDX2:.*]] = divi %[[TEMP]], %[[INDEX]]#1 unsigned
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [%[[M0]], %[[N0]], %[[K0]]], strides = [%[[S0]], %[[T0]], %[[U0]]]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[IDX0]], %[[IDX1]], %[[IDX2]]]
// CHECK: make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]], %[[KR]]], strides = [%[[SR]], %[[TR]], %[[UR]]]
module {
  nv_tensor_ir.graph @dynamic_shape_and_stride_rank_3(
      %arg0: tensor<?x?x?xf32> {nv_tensor_ir.stride = "(?,?,?)"}) ->
      (tensor<?x?x?xf32> {nv_tensor_ir.stride = "(?,?,?)"})
      attributes {tile_size = array<i32: 8, 8, 8>} {
    %res = cos %arg0 : tensor<?x?x?xf32>
    results %res : tensor<?x?x?xf32>
  }
}

// -----

// ============================================================================
// TEST 9: Mixed dynamic shape and stride
// ============================================================================
// CHECK-LABEL: entry @mixed_dynamic_shape_and_stride
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>, %[[N0:[^,]*]]: tile<i32>, %[[S0:[^,]*]]: tile<i32>, %[[T0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^,]*]]: tile<i32>, %[[NR:[^,]*]]: tile<i32>, %[[SR:[^,]*]]: tile<i32>, %[[TR:[^,]*]]: tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]], 32]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: %[[INDEX:.*]]:3 = get_index_space_shape %[[PVIEW]]
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[TEMP:.*]] = divi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[IDX1:.*]] = remi %[[TEMP]], %[[INDEX]]#1 unsigned
// CHECK: %[[IDX2:.*]] = divi %[[TEMP]], %[[INDEX]]#1 unsigned
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [%[[M0]], %[[N0]], 32], strides = [%[[S0]], %[[T0]], 1]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[IDX0]], %[[IDX1]], %[[IDX2]]]
// CHECK: make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]], 32], strides = [%[[SR]], %[[TR]], 1]
module {
  nv_tensor_ir.graph @mixed_dynamic_shape_and_stride(
      %arg0: tensor<?x?x32xf32> {nv_tensor_ir.stride = "(?,?,1)"}) ->
      (tensor<?x?x32xf32> {nv_tensor_ir.stride = "(?,?,1)"})
      attributes {tile_size = array<i32: 8, 8, 8>} {
    %res = cos %arg0 : tensor<?x?x32xf32>
    results %res : tensor<?x?x32xf32>
  }
}

// -----

// ============================================================================
// TEST 10: Binary pointwise (simple)
// ============================================================================
// CHECK-LABEL: entry @dynamic_binary_pointwise_simple
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>, %[[N0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[ARG1:[^,]*]]: tile<ptr<f32>>, %[[M1:[^,]*]]: tile<i32>, %[[N1:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^,]*]]: tile<i32>, %[[NR:[^,]*]]: tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]]]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: %[[INDEX:.*]]:2 = get_index_space_shape %[[PVIEW]]
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[IDX1:.*]] = divi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [%[[M0]], %[[N0]]], strides = [%[[N0]], 1]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[IDX0]], %[[IDX1]]]
// CHECK: %[[TVIEW1:.*]] = make_tensor_view %[[ARG1]], shape = [%[[M1]], %[[N1]]], strides = [%[[N1]], 1]
// CHECK: %[[PVIEW1:.*]] = make_partition_view %[[TVIEW1]]
// CHECK: load_view_tko weak %[[PVIEW1]][%[[IDX0]], %[[IDX1]]]
// CHECK: make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]]], strides = [%[[NR]], 1]
module {
  nv_tensor_ir.graph @dynamic_binary_pointwise_simple(
      %arg0: tensor<?x?xf32>,
      %arg1: tensor<?x?xf32>) ->
      (tensor<?x?xf32>)
      attributes {tile_size = array<i32: 8, 8>} {
    %res = add %arg0, %arg1 : tensor<?x?xf32>
    results %res : tensor<?x?xf32>
  }
}

// -----

// ============================================================================
// TEST 11: Binary pointwise (with reshape)
// ============================================================================
// CHECK-LABEL: entry @dynamic_binary_pointwise_reshape
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[ARG1:[^,]*]]: tile<ptr<f32>>, %[[M1:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^,]*]]: tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [%[[MR]], 128]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: %[[INDEX:.*]]:2 = get_index_space_shape %[[PVIEW]]
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[IDX1:.*]] = divi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [%[[M0]], 128], strides = [128, 1]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[IDX0]], %[[IDX1]]]
// CHECK: %[[TVIEW1:.*]] = make_tensor_view %[[ARG1]], shape = [%[[M1]], 128], strides = [128, 1]
// CHECK: %[[PVIEW1:.*]] = make_partition_view %[[TVIEW1]]
// CHECK: load_view_tko weak %[[PVIEW1]][%[[IDX0]], %[[IDX1]]]
// CHECK: make_tensor_view %[[OUT]], shape = [%[[MR]], 128], strides = [128, 1]
module {
  nv_tensor_ir.graph @dynamic_binary_pointwise_reshape(
      %arg0: tensor<?x32x4xf32>,
      %arg1: tensor<?x128xf32>) ->
      (tensor<?x128xf32>)
      attributes {tile_size = array<i32: 8, 8>} {
    %reshaped = reshape %arg0 : tensor<?x32x4xf32> -> tensor<?x128xf32>
    %res = add %reshaped, %arg1 : tensor<?x128xf32>
    results %res : tensor<?x128xf32>
  }
}

// -----

// ============================================================================
// TEST 12: Binary pointwise (with transpose)
// ============================================================================
// CHECK-LABEL: entry @dynamic_binary_pointwise_reshape
// CHECK-SAME: %[[ARG0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>, %[[N0:[^,]*]]: tile<i32>
// CHECK-SAME: %[[ARG1:[^,]*]]: tile<ptr<f32>>, %[[M1:[^,]*]]: tile<i32>, %[[N1:[^,]*]]: tile<i32>
// CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^,]*]]: tile<i32>, %[[NR:[^,]*]]: tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]]]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: %[[INDEX:.*]]:2 = get_index_space_shape %[[PVIEW]]
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[IDX1:.*]] = divi %[[BLOCK]], %[[INDEX]]#0 unsigned
// CHECK: %[[TVIEW0:.*]] = make_tensor_view %[[ARG0]], shape = [%[[N0]], %[[M0]]], strides = [1, %[[N0]]]
// CHECK: %[[PVIEW0:.*]] = make_partition_view %[[TVIEW0]]
// CHECK: load_view_tko weak %[[PVIEW0]][%[[IDX0]], %[[IDX1]]]
// CHECK: %[[TVIEW1:.*]] = make_tensor_view %[[ARG1]], shape = [%[[M1]], %[[N1]]], strides = [%[[N1]], 1]
// CHECK: %[[PVIEW1:.*]] = make_partition_view %[[TVIEW1]]
// CHECK: load_view_tko weak %[[PVIEW1]][%[[IDX0]], %[[IDX1]]]
// CHECK: make_tensor_view %[[OUT]], shape = [%[[MR]], %[[NR]]], strides = [%[[NR]], 1]
module {
  nv_tensor_ir.graph @dynamic_binary_pointwise_reshape(
      %arg0: tensor<?x?xf32>,
      %arg1: tensor<?x?xf32>) ->
      (tensor<?x?xf32>)
      attributes {tile_size = array<i32: 8, 8>} {
    %transposed = transpose %arg0 permutation = [1, 0] : tensor<?x?xf32> -> tensor<?x?xf32>
    %res = add %transposed, %arg1 : tensor<?x?xf32>
    results %res : tensor<?x?xf32>
  }
}
