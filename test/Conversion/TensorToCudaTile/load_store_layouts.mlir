// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// Load emission tests for graphs with diamond patterns, where a single input
// is read with different layouts.
// ============================================================================

// CHECK-LABEL: @loads_unary_elementwise
// CHECK-SAME: (%[[INPTR:.+]]: tile<ptr<f32>>, %[[OUTPTR:.+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK-DAG: %[[ROW:.+]] = remi %[[BID_X]],
// CHECK-DAG: %[[COL:.+]] = divi %[[BID_X]],
// CHECK: %[[VIEW1:.+]] = make_tensor_view %[[INPTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW1:.+]] = make_partition_view %[[VIEW1]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW1]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[NEG1:.+]] = negf %[[TILE1]]
// CHECK: %[[VIEW2:.+]] = make_tensor_view %[[INPTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW2:.+]] = make_partition_view %[[VIEW2]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW2]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[NEG2:.+]] = negf %[[TILE2]]
// CHECK: %[[SUM:.+]] = addf %[[NEG1]], %[[NEG2]] : tile<16x16xf32>
// CHECK: %[[OUTVIEW:.+]] = make_tensor_view %[[OUTPTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[OUTPVIEW:.+]] = make_partition_view %[[OUTVIEW]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[SUM]], %[[OUTPVIEW]][%[[ROW]], %[[COL]]]
module {
  nv_tensor_ir.graph @loads_unary_elementwise(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %neg = neg %arg0 : tensor<32x32xf32>
    %trans = transpose %neg permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %neg, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_binary_elementwise
// CHECK-SAME: (%[[IN0_PTR:.+]]: tile<ptr<f32>>, %[[IN1_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK-DAG: %[[ROW:.+]] = remi %[[BID_X]],
// CHECK-DAG: %[[COL:.+]] = divi %[[BID_X]],
// CHECK: %[[TVIEW01:.+]] = make_tensor_view %[[IN0_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW01:.+]] = make_partition_view %[[TVIEW01]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[ARG0_TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW01]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[TVIEW11:.+]] = make_tensor_view %[[IN1_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW11:.+]] = make_partition_view %[[TVIEW11]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[ARG1_TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW11]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[MUL1:.+]] = mulf %[[ARG0_TILE1]], %[[ARG1_TILE1]] : tile<16x16xf32>
// CHECK: %[[TVIEW02:.+]] = make_tensor_view %[[IN0_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW02:.+]] = make_partition_view %[[TVIEW02]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[ARG0_TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW02]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[TVIEW12:.+]] = make_tensor_view %[[IN1_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW12:.+]] = make_partition_view %[[TVIEW12]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[ARG1_TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW12]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[MUL2:.+]] = mulf %[[ARG0_TILE2]], %[[ARG1_TILE2]] : tile<16x16xf32>
// CHECK: %[[SUM:.+]] = addf %[[MUL1]], %[[MUL2]] : tile<16x16xf32>
// CHECK: %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[SUM]], %[[PVIEW_OUT]][%[[ROW]], %[[COL]]]
module {
  nv_tensor_ir.graph @loads_binary_elementwise(
      %arg0: tensor<32x32xf32>,
      %arg1: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %mul = mul %arg0, %arg1 : tensor<32x32xf32>
    %trans = transpose %mul permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %mul, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_binary_elementwise_single_input
// CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK-DAG: %[[ROW:.+]] = remi %[[BID_X]],
// CHECK-DAG: %[[COL:.+]] = divi %[[BID_X]],
// CHECK: %[[TVIEW1:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW1:.+]] = make_partition_view %[[TVIEW1]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW1]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[MUL1:.+]] = mulf %[[TILE1]], %[[TILE1]] : tile<16x16xf32>
// CHECK: %[[TVIEW2:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW2:.+]] = make_partition_view %[[TVIEW2]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW2]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[MUL2:.+]] = mulf %[[TILE2]], %[[TILE2]] : tile<16x16xf32>
// CHECK: %[[SUM:.+]] = addf %[[MUL1]], %[[MUL2]] : tile<16x16xf32>
// CHECK: %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[SUM]], %[[PVIEW_OUT]][%[[ROW]], %[[COL]]]
module {
  nv_tensor_ir.graph @loads_binary_elementwise_single_input(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %mul = mul %arg0, %arg0 : tensor<32x32xf32>
    %trans = transpose %mul permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %mul, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_binary_elementwise_lhs_input
// CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK-DAG: %[[ROW:.+]] = remi %[[BID_X]],
// CHECK-DAG: %[[COL:.+]] = divi %[[BID_X]],
// CHECK: %[[TVIEW1:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW1:.+]] = make_partition_view %[[TVIEW1]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW1]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[ABS1:.+]] = absf %[[TILE1]]
// CHECK: %[[MUL1:.+]] = mulf %[[TILE1]], %[[ABS1]] : tile<16x16xf32>
// CHECK: %[[TVIEW2:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW2:.+]] = make_partition_view %[[TVIEW2]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW2]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[ABS2:.+]] = absf %[[TILE2]]
// CHECK: %[[MUL2:.+]] = mulf %[[TILE2]], %[[ABS2]] : tile<16x16xf32>
// CHECK: %[[SUM:.+]] = addf %[[MUL1]], %[[MUL2]] : tile<16x16xf32>
// CHECK: %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[SUM]], %[[PVIEW_OUT]][%[[ROW]], %[[COL]]]
module {
  nv_tensor_ir.graph @loads_binary_elementwise_lhs_input(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %abs = abs %arg0 : tensor<32x32xf32>
    %mul = mul %arg0, %abs : tensor<32x32xf32>
    %trans = transpose %mul permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %mul, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_binary_elementwise_rhs_input
// CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK-DAG: %[[ROW:.+]] = remi %[[BID_X]],
// CHECK-DAG: %[[COL:.+]] = divi %[[BID_X]],
// CHECK: %[[TVIEW1:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW1:.+]] = make_partition_view %[[TVIEW1]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW1]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[ABS1:.+]] = absf %[[TILE1]]
// CHECK: %[[MUL1:.+]] = mulf %[[ABS1]], %[[TILE1]] : tile<16x16xf32>
// CHECK: %[[TVIEW2:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW2:.+]] = make_partition_view %[[TVIEW2]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW2]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[ABS2:.+]] = absf %[[TILE2]]
// CHECK: %[[MUL2:.+]] = mulf %[[ABS2]], %[[TILE2]] : tile<16x16xf32>
// CHECK: %[[SUM:.+]] = addf %[[MUL1]], %[[MUL2]] : tile<16x16xf32>
// CHECK: %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[SUM]], %[[PVIEW_OUT]][%[[ROW]], %[[COL]]]
module {
  nv_tensor_ir.graph @loads_binary_elementwise_rhs_input(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %abs = abs %arg0 : tensor<32x32xf32>
    %mul = mul %abs, %arg0 : tensor<32x32xf32>
    %trans = transpose %mul permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %mul, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_ternary_elementwise_select
// CHECK-SAME: (%[[IN0_PTR:.+]]: tile<ptr<i1>>, %[[IN1_PTR:.+]]: tile<ptr<f32>>, %[[IN2_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK-DAG: %[[ROW:.+]] = remi %[[BID_X]],
// CHECK-DAG: %[[COL:.+]] = divi %[[BID_X]],
// CHECK: %[[TVIEW01:.+]] = make_tensor_view %[[IN0_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW01:.+]] = make_partition_view %[[TVIEW01]] : partition_view<tile=(16x16), tensor_view<32x32xi1, strides=[32,1]>>
// CHECK: %[[ARG0_TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW01]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xi1>, token
// CHECK: %[[TVIEW11:.+]] = make_tensor_view %[[IN1_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW11:.+]] = make_partition_view %[[TVIEW11]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[ARG1_TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW11]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[TVIEW21:.+]] = make_tensor_view %[[IN2_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW21:.+]] = make_partition_view %[[TVIEW21]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[ARG2_TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW21]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[SELECT1:.+]] = select %[[ARG0_TILE1]], %[[ARG1_TILE1]], %[[ARG2_TILE1]]
// CHECK: %[[TVIEW02:.+]] = make_tensor_view %[[IN0_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW02:.+]] = make_partition_view %[[TVIEW02]] : partition_view<tile=(16x16), tensor_view<32x32xi1, strides=[1,32]>>
// CHECK: %[[ARG0_TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW02]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xi1>, token
// CHECK: %[[TVIEW12:.+]] = make_tensor_view %[[IN1_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW12:.+]] = make_partition_view %[[TVIEW12]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[ARG1_TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW12]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[TVIEW22:.+]] = make_tensor_view %[[IN2_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW22:.+]] = make_partition_view %[[TVIEW22]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[ARG2_TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW22]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[SELECT2:.+]] = select %[[ARG0_TILE2]], %[[ARG1_TILE2]], %[[ARG2_TILE2]]
// CHECK: %[[SUM:.+]] = addf %[[SELECT1]], %[[SELECT2]] : tile<16x16xf32>
// CHECK: %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[SUM]], %[[PVIEW_OUT]][%[[ROW]], %[[COL]]]
module {
  nv_tensor_ir.graph @loads_ternary_elementwise_select(
      %arg0: tensor<32x32xi1>,
      %arg1: tensor<32x32xf32>,
      %arg2: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %select = binary_select %arg0, %arg1, %arg2 : tensor<32x32xf32>
    %trans = transpose %select permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %select, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_transpose_kernel
// CHECK-SAME: (%[[INPTR:.+]]: tile<ptr<f32>>, %[[OUTPTR:.+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK-DAG: %[[ROW:.+]] = remi %[[BID_X]],
// CHECK-DAG: %[[COL:.+]] = divi %[[BID_X]],
// CHECK: %[[VIEW:.+]] = make_tensor_view %[[INPTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW:.+]] = make_partition_view %[[VIEW]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[TILE:.+]], %{{.+}} = load_view_tko weak %[[PVIEW]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[OUTVIEW:.+]] = make_tensor_view %[[OUTPTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[OUTPVIEW:.+]] = make_partition_view %[[OUTVIEW]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[TILE]], %[[OUTPVIEW]][%[[ROW]], %[[COL]]]
module {
  nv_tensor_ir.graph @loads_transpose_kernel(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %trans = transpose %arg0 permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    results %trans : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_multiple_convert
// CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f16>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK-DAG: %[[ROW:.+]] = remi %[[BID_X]],
// CHECK-DAG: %[[COL:.+]] = divi %[[BID_X]],
// CHECK: %[[TVIEW1:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW1:.+]] = make_partition_view %[[TVIEW1]] : partition_view<tile=(16x16), tensor_view<32x32xf16, strides=[32,1]>>
// CHECK: %[[TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW1]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf16>, token
// CHECK: %[[CONVERT1:.+]] = ftof %[[TILE1]]
// CHECK: %[[TVIEW2:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW2:.+]] = make_partition_view %[[TVIEW2]] : partition_view<tile=(16x16), tensor_view<32x32xf16, strides=[1,32]>>
// CHECK: %[[TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW2]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf16>, token
// CHECK: %[[CONVERT2:.+]] = ftof %[[TILE2]]
// CHECK: %[[SUM:.+]] = addf %[[CONVERT1]], %[[CONVERT2]] : tile<16x16xf32>
// CHECK: %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[SUM]], %[[PVIEW_OUT]][%[[ROW]], %[[COL]]]
module {
  nv_tensor_ir.graph @loads_multiple_convert(
      %arg0: tensor<32x32xf16>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %convert = convert %arg0 : tensor<32x32xf16> -> tensor<32x32xf32>
    %trans = transpose %convert permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %convert, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_multiple_compare
// CHECK-SAME: (%[[IN0_PTR:.+]]: tile<ptr<f32>>, %[[IN1_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<i1>>)
// CHECK-DAG: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK-DAG: %[[ROW:.+]] = remi %[[BID_X]],
// CHECK-DAG: %[[COL:.+]] = divi %[[BID_X]],
// CHECK: %[[TVIEW01:.+]] = make_tensor_view %[[IN0_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW01:.+]] = make_partition_view %[[TVIEW01]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[ARG0_TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW01]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[TVIEW11:.+]] = make_tensor_view %[[IN1_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW11:.+]] = make_partition_view %[[TVIEW11]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[32,1]>>
// CHECK: %[[ARG1_TILE1:.+]], %{{.+}} = load_view_tko weak %[[PVIEW11]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[COMPARE1:.+]] = cmpf less_than unordered %[[ARG0_TILE1]], %[[ARG1_TILE1]]
// CHECK: %[[TVIEW02:.+]] = make_tensor_view %[[IN0_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW02:.+]] = make_partition_view %[[TVIEW02]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[ARG0_TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW02]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[TVIEW12:.+]] = make_tensor_view %[[IN1_PTR]], shape = [32, 32], strides = [1, 32]
// CHECK: %[[PVIEW12:.+]] = make_partition_view %[[TVIEW12]] : partition_view<tile=(16x16), tensor_view<32x32xf32, strides=[1,32]>>
// CHECK: %[[ARG1_TILE2:.+]], %{{.+}} = load_view_tko weak %[[PVIEW12]][%[[ROW]], %[[COL]]] {{.+}} tile<16x16xf32>, token
// CHECK: %[[COMPARE2:.+]] = cmpf less_than unordered %[[ARG0_TILE2]], %[[ARG1_TILE2]]
// CHECK: %[[BOTH:.+]] = andi %[[COMPARE1]], %[[COMPARE2]] : tile<16x16xi1>
// CHECK: %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [32, 32], strides = [32, 1]
// CHECK: %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(16x16), tensor_view<32x32xi1, strides=[32,1]>>
// CHECK: store_view_tko weak %[[BOTH]], %[[PVIEW_OUT]][%[[ROW]], %[[COL]]]
module {
  nv_tensor_ir.graph @loads_multiple_compare(
      %arg0: tensor<32x32xf32>,
      %arg1: tensor<32x32xf32>) ->
      (tensor<32x32xi1>)
      attributes {tile_size = array<i32: 16, 16>} {
    %compare = cmp %arg0 ult %arg1 : tensor<32x32xf32>
    %trans = transpose %compare permutation = [1, 0] : tensor<32x32xi1> -> tensor<32x32xi1>
    %out = and %compare, %trans : tensor<32x32xi1>
    results %out : tensor<32x32xi1>
  }
}

// -----

// ============================================================================
// Store emission tests to verify the explicit strides are not lost when the
// graph iteration space doesn't match the tensor rank.
// ============================================================================

// CHECK-LABEL: @store_output_rank_increase
// CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK-DAG: %[[ZERO:.+]] = constant <i32: 0> : tile<i32>
// CHECK-DAG: %[[ROW:.+]] = remi %[[BID_X]],
// CHECK-DAG: %[[COL:.+]] = divi %[[BID_X]],
// CHECK: %[[TVIEW_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [4, 8, 4, 8], strides = [1, 4, 32, 128]
// CHECK: %[[PVIEW_IN:.+]] = make_partition_view %[[TVIEW_IN]] : partition_view<tile=(4x8x2x1), tensor_view<4x8x4x8xf32, strides=[1,4,32,128]>>
// CHECK: %[[TILE:.+]], %{{.+}} = load_view_tko weak %[[PVIEW_IN]][%[[ZERO]], %[[ZERO]], %[[ROW]], %[[COL]]] {{.+}} tile<4x8x2x1xf32>, token
// CHECK: %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [4, 8, 4, 8], strides = [8, 1, 512, 64]
// CHECK: %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(4x8x2x1), tensor_view<4x8x4x8xf32, strides=[8,1,512,64]>>
// CHECK: store_view_tko weak %[[TILE]], %[[PVIEW_OUT]][%[[ZERO]], %[[ZERO]], %[[ROW]], %[[COL]]]
module {
  nv_tensor_ir.graph @store_output_rank_increase(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32> {nv_tensor_ir.stride = "(1,64)"})
      attributes {tile_size = array<i32: 4, 8, 2, 1>} {
    %reshape1_input_reversed = transpose %arg0 permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %reshape1_reversed = reshape %reshape1_input_reversed : tensor<32x32xf32> -> tensor<8x4x8x4xf32>
    %reshape2_input_reversed = transpose %reshape1_reversed permutation = [3, 2, 1, 0] : tensor<8x4x8x4xf32> -> tensor<4x8x4x8xf32>
    %reshape2_reversed = reshape %reshape2_input_reversed : tensor<4x8x4x8xf32> -> tensor<32x32xf32>
    %reshape2 = transpose %reshape2_reversed permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    results %reshape2 : tensor<32x32xf32>
  }
}

// -----


// CHECK-LABEL: @store_output_rank_decrease
// CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
// CHECK: %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[TVIEW_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [1024], strides = [1]
// CHECK: %[[PVIEW_IN:.+]] = make_partition_view %[[TVIEW_IN]] : partition_view<tile=(32), tensor_view<1024xf32, strides=[1]>>
// CHECK: %[[TILE:.+]], %{{.+}} = load_view_tko weak %[[PVIEW_IN]][%[[BID_X]]] {{.+}} tile<32xf32>, token
// CHECK: %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [1024], strides = [1]
// CHECK: %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(32), tensor_view<1024xf32, strides=[1]>>
// CHECK: store_view_tko weak %[[TILE]], %[[PVIEW_OUT]][%[[BID_X]]] : tile<32xf32>,
module {
  nv_tensor_ir.graph @store_output_rank_decrease(
      %arg0: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}) ->
      (tensor<32x32xf32> {nv_tensor_ir.stride = "(32,1)"})
      attributes {tile_size = array<i32: 32>} {
    %trans = transpose %arg0 permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    results %trans : tensor<32x32xf32>
  }
}
