// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// Load/Store emission tests for elementwise add graphs.
// With layout-based codegen + normalize, contiguous row-major layouts are
// coalesced to 1D.
// ============================================================================

// CHECK-LABEL: entry @add_self_single_input
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
//       CHECK:   %[[TVIEW_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   %[[PVIEW_SHAPE:.+]] = make_partition_view %[[TVIEW_IN]] : partition_view<tile=({{[0-9]+}}), tensor_view<2048xf32, strides=[1]>>
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko weak %[[PVIEW_SHAPE]][{{.*}}] :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE0]], %[[TILE0]] : tile<{{[0-9]+}}xf32>
//       CHECK:   %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   %[[PVIEW_ST:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=({{[0-9]+}}), tensor_view<2048xf32, strides=[1]>>
//       CHECK:   store_view_tko weak %[[ADD]], %[[PVIEW_ST]][{{.*}}] :{{.*}}-> token
//       CHECK:   return
nv_tensor_ir.graph @add_self_single_input(
    %arg0: tensor<32x64xf32> ) ->
    (tensor<32x64xf32>)
    attributes {tile_size = array<i32: 32>} {
  %add = add %arg0, %arg0 : tensor<32x64xf32>
  results %add : tensor<32x64xf32>
}

// -----

// CHECK-LABEL: entry @add_self_single_input_col_major
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//   CHECK-DAG:   %[[BID_X:.+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
//   CHECK-DAG:   %[[ZERO:.+]] = constant <i32: 0> : tile<i32>
//       CHECK:   %[[TVIEW_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 64], strides = [1, 32] : tensor_view<32x64xf32, strides=[1,32]>
//       CHECK:   %[[PVIEW_SHAPE:.+]] = make_partition_view %[[TVIEW_IN]] : partition_view<tile=(32x1), tensor_view<32x64xf32, strides=[1,32]>>
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko weak %[[PVIEW_SHAPE]][%[[ZERO]], %[[BID_X]]] :{{.*}}-> tile<32x1xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE0]], %[[TILE0]] : tile<32x1xf32>
//       CHECK:   %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [32, 64], strides = [1, 32] : tensor_view<32x64xf32, strides=[1,32]>
//       CHECK:   %[[PVIEW_ST:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(32x1), tensor_view<32x64xf32, strides=[1,32]>>
//       CHECK:   store_view_tko weak %[[ADD]], %[[PVIEW_ST]][%[[ZERO]], %[[BID_X]]] :{{.*}}-> token
//       CHECK:   return
nv_tensor_ir.graph @add_self_single_input_col_major(
    %arg0: tensor<32x64xf32> {nv_tensor_ir.stride = "(1,32)"}) ->
    (tensor<32x64xf32> {nv_tensor_ir.stride = "(1,32)"})
    attributes {tile_size = array<i32: 32, 1>} {
  %add = add %arg0, %arg0 : tensor<32x64xf32>
  results %add : tensor<32x64xf32>
}

// -----

// CHECK-LABEL: entry @add_two_inputs
//  CHECK-SAME: (%[[IN0_PTR:.+]]: tile<ptr<f32>>, %[[IN1_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TVIEW_IN0:.+]] = make_tensor_view %[[IN0_PTR]], shape = [8192], strides = [1] : tensor_view<8192xf32, strides=[1]>
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[TVIEW_IN1:.+]] = make_tensor_view %[[IN1_PTR]], shape = [8192], strides = [1] : tensor_view<8192xf32, strides=[1]>
//       CHECK:   %[[TILE1:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE0]], %[[TILE1]] : tile<{{[0-9]+}}xf32>
//       CHECK:   %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [8192], strides = [1] : tensor_view<8192xf32, strides=[1]>
//       CHECK:   store_view_tko weak %[[ADD]], {{.*}} :{{.*}}-> token
//       CHECK:   return
module {
  nv_tensor_ir.graph @add_two_inputs(
      %arg0: tensor<64x128xf32>,
      %arg1: tensor<64x128xf32>) ->
      (tensor<64x128xf32>)
      attributes {tile_size = array<i32: 64>} {
    %add = add %arg0, %arg1 : tensor<64x128xf32>
    results %add : tensor<64x128xf32>
  }
}

// -----

// CHECK-LABEL: entry @add_chain_two_inputs
//  CHECK-SAME: (%[[IN0_PTR:.+]]: tile<ptr<f32>>, %[[IN1_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   make_tensor_view %[[IN0_PTR]], shape = [2048], strides = [1]
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko
//       CHECK:   make_tensor_view %[[IN1_PTR]], shape = [2048], strides = [1]
//       CHECK:   %[[TILE1:.+]], {{.*}} = load_view_tko
//       CHECK:   %[[ADD0:.+]] = addf %[[TILE0]], %[[TILE1]] : tile<{{[0-9]+}}xf32>
//       CHECK:   %[[ADD1:.+]] = addf %[[ADD0]], %[[TILE0]] : tile<{{[0-9]+}}xf32>
//       CHECK:   make_tensor_view %[[OUT_PTR]], shape = [2048], strides = [1]
//       CHECK:   store_view_tko weak %[[ADD1]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @add_chain_two_inputs(
      %arg0: tensor<32x64xf32>,
      %arg1: tensor<32x64xf32>) ->
      (tensor<32x64xf32>)
      attributes {tile_size = array<i32: 32>} {
    %add0 = add %arg0, %arg1 : tensor<32x64xf32>
    %add1 = add %add0, %arg0 : tensor<32x64xf32>
    results %add1 : tensor<32x64xf32>
  }
}

// -----

// CHECK-LABEL: entry @add_three_inputs
//       CHECK:   make_tensor_view {{.*}}, shape = [512], strides = [1]
//       CHECK:   %[[T0:.+]], {{.*}} = load_view_tko
//       CHECK:   make_tensor_view {{.*}}, shape = [512], strides = [1]
//       CHECK:   %[[T1:.+]], {{.*}} = load_view_tko
//       CHECK:   make_tensor_view {{.*}}, shape = [512], strides = [1]
//       CHECK:   %[[T2:.+]], {{.*}} = load_view_tko
//       CHECK:   %[[A0:.+]] = addf %[[T0]], %[[T1]] : tile<{{[0-9]+}}xf32>
//       CHECK:   %[[A1:.+]] = addf %[[A0]], %[[T2]] : tile<{{[0-9]+}}xf32>
//       CHECK:   make_tensor_view {{.*}}, shape = [512], strides = [1]
//       CHECK:   store_view_tko weak %[[A1]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @add_three_inputs(
      %arg0: tensor<16x32xf32>,
      %arg1: tensor<16x32xf32>,
      %arg2: tensor<16x32xf32>) ->
      (tensor<16x32xf32>)
      attributes {tile_size = array<i32: 32>} {
    %add0 = add %arg0, %arg1 : tensor<16x32xf32>
    %add1 = add %add0, %arg2 : tensor<16x32xf32>
    results %add1 : tensor<16x32xf32>
  }
}

// -----

// CHECK-LABEL: entry @add_chain_single_input
//       CHECK:   make_tensor_view {{.*}}, shape = [4096], strides = [1]
//       CHECK:   %[[T:.+]], {{.*}} = load_view_tko
//       CHECK:   %[[A0:.+]] = addf %[[T]], %[[T]] : tile<{{[0-9]+}}xf32>
//       CHECK:   %[[A1:.+]] = addf %[[A0]], %[[T]] : tile<{{[0-9]+}}xf32>
//       CHECK:   %[[A2:.+]] = addf %[[A1]], %[[A0]] : tile<{{[0-9]+}}xf32>
//       CHECK:   make_tensor_view {{.*}}, shape = [4096], strides = [1]
//       CHECK:   store_view_tko weak %[[A2]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @add_chain_single_input(
      %arg0: tensor<64x64xf32>) ->
      (tensor<64x64xf32>)
      attributes {tile_size = array<i32: 32>} {
    %add0 = add %arg0, %arg0 : tensor<64x64xf32>
    %add1 = add %add0, %arg0 : tensor<64x64xf32>
    %add2 = add %add1, %add0 : tensor<64x64xf32>
    results %add2 : tensor<64x64xf32>
  }
}
