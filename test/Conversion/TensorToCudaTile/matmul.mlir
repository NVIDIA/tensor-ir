// RUN: tensor_ir-opt -materialize-default-strides -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s

module {
  nv_tensor_ir.graph @matmul_f32_static(
    %a: tensor<128x64xf32>,
    %b: tensor<64x128xf32>) -> (
      tensor<128x128xf32>) {
    %c = "nv_tensor_ir.matmul"(%a, %b) : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    results %c : tensor<128x128xf32>
  }
}

// CHECK-LABEL: entry @matmul_f32_static
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f32>>, %[[B_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:    %[[TVIEW_A:.+]] = make_tensor_view %[[A_PTR]], shape = [128, 64], strides = [64, 1] : tensor_view<128x64xf32, strides=[64,1]>
//       CHECK:    %[[PVIEW_A_PADDED:.+]] = make_partition_view %[[TVIEW_A]] : partition_view<tile=(2x2), padding_value = zero, tensor_view<128x64xf32, strides=[64,1]>>
//       CHECK:    %[[TVIEW_B:.+]] = make_tensor_view %[[B_PTR]], shape = [64, 128], strides = [128, 1] : tensor_view<64x128xf32, strides=[128,1]>
//       CHECK:    %[[PVIEW_B_PADDED:.+]] = make_partition_view %[[TVIEW_B]] : partition_view<tile=(2x2), padding_value = zero, tensor_view<64x128xf32, strides=[128,1]>>
//       CHECK:    %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
//       CHECK:    %[[BID_X:.+]], %[[BID_Y:.+]], %[[BID_Z:.+]] = get_tile_block_id : tile<i32>
//       CHECK:    %[[PVIEW_A_SHAPE:.+]] = make_partition_view %[[TVIEW_A]] : partition_view<tile=(2x2), tensor_view<128x64xf32, strides=[64,1]>>
//       CHECK:    %[[SHAPE_A:.+]]:2 = get_index_space_shape %[[PVIEW_A_SHAPE]] : partition_view<tile=(2x2), tensor_view<128x64xf32, strides=[64,1]>> -> tile<i32>
//       CHECK:    %[[PVIEW_B_SHAPE:.+]] = make_partition_view %[[TVIEW_B]] : partition_view<tile=(2x2), tensor_view<64x128xf32, strides=[128,1]>>
//       CHECK:    %[[SHAPE_B:.+]]:2 = get_index_space_shape %[[PVIEW_B_SHAPE]] : partition_view<tile=(2x2), tensor_view<64x128xf32, strides=[128,1]>> -> tile<i32>
//       CHECK:    %[[LB:.+]] = constant <i32: 0> : tile<i32>
//       CHECK:    %[[STEP:.+]] = constant <i32: 1> : tile<i32>
//       CHECK:    %[[ACC_INIT:.+]] = constant <f32: 0.000000e+00> : tile<2x2xf32>
//       CHECK:    %[[FOR:.+]] = for %[[K_IDX:.+]] in (%[[LB]] to %[[SHAPE_B]]#0, step %[[STEP]]) : tile<i32> iter_values(%[[ACC:.+]] = %[[ACC_INIT]]) -> (tile<2x2xf32>) {
//       CHECK:      %[[BID_N:.+]] = remi %[[BID_X]], %[[SHAPE_B]]#1 unsigned : tile<i32>
//       CHECK:      %[[BID_M_DIV:.+]] = divi %[[BID_X]], %[[SHAPE_B]]#1 unsigned : tile<i32>
//       CHECK:      %[[BID_M:.+]] = remi %[[BID_M_DIV]], %[[SHAPE_A]]#0 unsigned : tile<i32>
//       CHECK:      %[[TILE_A:.+]], {{.*}} = load_view_tko weak %[[PVIEW_A_PADDED]][%[[BID_M]], %[[K_IDX]]] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[BID_N_B:.+]] = remi %[[BID_X]], %[[SHAPE_B]]#1 unsigned : tile<i32>
//       CHECK:      %[[BID_M_DIV_B:.+]] = divi %[[BID_X]], %[[SHAPE_B]]#1 unsigned : tile<i32>
//       CHECK:      %[[BID_M_B:.+]] = remi %[[BID_M_DIV_B]], %[[SHAPE_A]]#0 unsigned : tile<i32>
//       CHECK:      %[[TILE_B:.+]], {{.*}} = load_view_tko weak %[[PVIEW_B_PADDED]][%[[K_IDX]], %[[BID_N_B]]] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[MMA:.+]] = mmaf %[[TILE_A]], %[[TILE_B]], %[[ACC]] : tile<2x2xf32>, tile<2x2xf32>, tile<2x2xf32>
//       CHECK:      continue %[[MMA]] : tile<2x2xf32>
//       CHECK:    }
//       CHECK:    %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(2x2), tensor_view<128x128xf32, strides=[128,1]>>
//       CHECK:    %[[OUT_N:.+]] = remi %[[BID_X]], %[[SHAPE_B]]#1 unsigned : tile<i32>
//       CHECK:    %[[OUT_M_DIV:.+]] = divi %[[BID_X]], %[[SHAPE_B]]#1 unsigned : tile<i32>
//       CHECK:    %[[OUT_M:.+]] = remi %[[OUT_M_DIV]], %[[SHAPE_A]]#0 unsigned : tile<i32>
//       CHECK:    store_view_tko weak %[[FOR]], %[[PVIEW_OUT]][%[[OUT_M]], %[[OUT_N]]] :{{.*}}-> token
//       CHECK:    return

// Reference CuTile IR for a matmul kernel:
//
// module attributes {gpu.container_module} {
//   cuda_tile.module @kernels {
//     entry @mma_kernel_0(%arg0: !cuda_tile.tile<ptr<f32>>, %arg1: !cuda_tile.tile<ptr<f32>>, %arg2: !cuda_tile.tile<ptr<f32>>, %arg3: !cuda_tile.tile<ptr<f32>>, %arg4: !cuda_tile.tile<i32>, %arg5: !cuda_tile.tile<i32>, %arg6: !cuda_tile.tile<i32>, %arg7: !cuda_tile.tile<i32>, %arg8: !cuda_tile.tile<i32>, %arg9: !cuda_tile.tile<i32>, %arg10: !cuda_tile.tile<i32>, %arg11: !cuda_tile.tile<i32>, %arg12: !cuda_tile.tile<i32>, %arg13: !cuda_tile.tile<i32>, %arg14: !cuda_tile.tile<i32>, %arg15: !cuda_tile.tile<i32>, %arg16: !cuda_tile.tile<i32>, %arg17: !cuda_tile.tile<i32>, %arg18: !cuda_tile.tile<i32>, %arg19: !cuda_tile.tile<i32>, %arg20: !cuda_tile.tile<i32>) {
//       %0 = constant dense<1> : !cuda_tile.tile<i32>
//       %1 = constant dense<0> : !cuda_tile.tile<i32>
//       %2 = assume #cuda_tile.div_by<16>, %arg2 : tile<ptr<f32>>
//       %3 = make_tensor_view %2, shape = [%arg11, %arg12], strides = [%arg18, 1] : tensor_view<?x?xf32, strides=[?,1]>
//       %4 = assume #cuda_tile.div_by<16>, %arg1 : tile<ptr<f32>>
//       %5 = make_tensor_view %4, shape = [%arg9, %arg10], strides = [%arg16, 1] : tensor_view<?x?xf32, strides=[?,1]>
//       %6 = assume #cuda_tile.div_by<16>, %arg0 : tile<ptr<f32>>
//       %7 = assume #cuda_tile.div_by<16>, %arg14 : tile<i32>
//       %8 = make_tensor_view %6, shape = [%arg7, %arg8], strides = [%7, 1] : tensor_view<?x?xf32, strides=[?,1]>
//       %result_x, %result_y, %result_z = get_tile_block_id : tile<i32>
//       %result_x_0, %result_y_1, %result_z_2 = get_tile_block_id : tile<i32>
//       %9 = make_partition_view %8 : partition_view<masked tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//       %10 = get_dim_size %9 1 : partition_view<masked tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>> -> tile<i32>
//       %11 = reshape %1 : tile<i32> -> tile<1x1xi32>
//       %12 = broadcast %11 : tile<1x1xi32> -> tile<2x2xi32>
//       %13 = itof %12 : tile<2x2xi32> -> tile<2x2xf32>
//       %14 = for %arg21 in (%1 to %10, step %0) : tile<i32> iter_values(%arg22 = %13) -> (tile<2x2xf32>) {
//         %17 = make_partition_view %8 : partition_view<masked tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//         %tile, %result_token = load_from_view_unordered weak %17[%result_x, %arg21] : partition_view<masked tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>> -> tile<2x2xf32>, token
//         %18 = make_partition_view %5 : partition_view<masked tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//         %tile_3, %result_token_4 = load_from_view_unordered weak %18[%arg21, %result_y_1] : partition_view<masked tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>> -> tile<2x2xf32>, token
//         %19 = mma %tile, %tile_3, %arg22 : tile<2x2xf32>, tile<2x2xf32>, tile<2x2xf32>
//         continue %19 : tile<2x2xf32>
//       }
//       %15 = make_partition_view %3 : partition_view<masked tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//       %16 = store_to_view_unordered weak %14, %15[%result_x, %result_y_1] : tile<2x2xf32>, partition_view<masked tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>> -> token
//       return
//     }
//   }
// }

// -----

module {
  nv_tensor_ir.graph @matmul_f16_static(
    %a: tensor<128x64xf16>,
    %b: tensor<64x128xf16>) -> (
      tensor<128x128xf16>) {
    %c = "nv_tensor_ir.matmul"(%a, %b) : (tensor<128x64xf16>, tensor<64x128xf16>) -> tensor<128x128xf16>
    results %c : tensor<128x128xf16>
  }
}

// CHECK-LABEL: entry @matmul_f16_static
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f16>>, %[[B_PTR:.+]]: tile<ptr<f16>>, %[[OUT_PTR:.+]]: tile<ptr<f16>>)
//       CHECK:    %[[TVIEW_A:.+]] = make_tensor_view %[[A_PTR]], shape = [128, 64], strides = [64, 1] : tensor_view<128x64xf16, strides=[64,1]>
//       CHECK:    %[[TVIEW_B:.+]] = make_tensor_view %[[B_PTR]], shape = [64, 128], strides = [128, 1] : tensor_view<64x128xf16, strides=[128,1]>
//       CHECK:    %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf16, strides=[128,1]>
//       CHECK:    %[[BID_X:.+]], %[[BID_Y:.+]], %[[BID_Z:.+]] = get_tile_block_id : tile<i32>
//       CHECK:    %[[PVIEW_A_SHAPE:.+]] = make_partition_view %[[TVIEW_A]] : partition_view<tile=(2x2), tensor_view<128x64xf16, strides=[64,1]>>
//       CHECK:    %[[SHAPE_A:.+]]:2 = get_index_space_shape %[[PVIEW_A_SHAPE]] : partition_view<tile=(2x2), tensor_view<128x64xf16, strides=[64,1]>> -> tile<i32>
//       CHECK:    %[[PVIEW_B_SHAPE:.+]] = make_partition_view %[[TVIEW_B]] : partition_view<tile=(2x2), tensor_view<64x128xf16, strides=[128,1]>>
//       CHECK:    %[[SHAPE_B:.+]]:2 = get_index_space_shape %[[PVIEW_B_SHAPE]] : partition_view<tile=(2x2), tensor_view<64x128xf16, strides=[128,1]>> -> tile<i32>
//       CHECK:    %[[LB:.+]] = constant <i32: 0> : tile<i32>
//       CHECK:    %[[STEP:.+]] = constant <i32: 1> : tile<i32>
//       CHECK:    %[[ACC_INIT:.+]] = constant <f16: 0.000000e+00> : tile<2x2xf16>
//       CHECK:    %[[FOR:.+]] = for %[[K_IDX:.+]] in (%[[LB]] to %[[SHAPE_B]]#0, step %[[STEP]]) : tile<i32> iter_values(%[[ACC:.+]] = %[[ACC_INIT]]) -> (tile<2x2xf16>) {
//       CHECK:      %[[TILE_A:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %[[K_IDX]]] :{{.*}}-> tile<2x2xf16>, token
//       CHECK:      %[[TILE_B:.+]], {{.*}} = load_view_tko weak %{{.+}}[%[[K_IDX]], %{{.+}}] :{{.*}}-> tile<2x2xf16>, token
//       CHECK:      %[[MMA:.+]] = mmaf %[[TILE_A]], %[[TILE_B]], %[[ACC]] : tile<2x2xf16>, tile<2x2xf16>, tile<2x2xf16>
//       CHECK:      continue %[[MMA]] : tile<2x2xf16>
//       CHECK:    }
//       CHECK:    store_view_tko weak %[[FOR]], %{{.+}}[%{{.+}}, %{{.+}}] :{{.*}}-> token
//       CHECK:    return

// -----

module {
  nv_tensor_ir.graph @matmul_f32_dynamic(
    %a: tensor<?x?xf32>,
    %b: tensor<?x?xf32>) -> (
      tensor<?x?xf32>) {
    %c = "nv_tensor_ir.matmul"(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    results %c : tensor<?x?xf32>
  }
}

// CHECK-LABEL: entry @matmul_f32_dynamic
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f32>>,{{.*}}%[[B_PTR:.+]]: tile<ptr<f32>>,{{.*}}%[[OUT_PTR:.+]]: tile<ptr<f32>>,{{.*}})
//       CHECK:    %[[TVIEW_A:.+]] = make_tensor_view %[[A_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
//       CHECK:    %[[TVIEW_B:.+]] = make_tensor_view %[[B_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
//       CHECK:    %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
//       CHECK:    %[[BID_X:.+]], %[[BID_Y:.+]], %[[BID_Z:.+]] = get_tile_block_id : tile<i32>
//       CHECK:    %[[PVIEW_A_SHAPE:.+]] = make_partition_view %[[TVIEW_A]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//       CHECK:    %[[SHAPE_A:.+]]:2 = get_index_space_shape %[[PVIEW_A_SHAPE]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>> -> tile<i32>
//       CHECK:    %[[PVIEW_B_SHAPE:.+]] = make_partition_view %[[TVIEW_B]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//       CHECK:    %[[SHAPE_B:.+]]:2 = get_index_space_shape %[[PVIEW_B_SHAPE]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>> -> tile<i32>
//       CHECK:    %[[LB:.+]] = constant <i32: 0> : tile<i32>
//       CHECK:    %[[STEP:.+]] = constant <i32: 1> : tile<i32>
//       CHECK:    %[[ACC_INIT:.+]] = constant <f32: 0.000000e+00> : tile<2x2xf32>
//       CHECK:    %[[FOR:.+]] = for %[[K_IDX:.+]] in (%[[LB]] to %[[SHAPE_B]]#0, step %[[STEP]]) : tile<i32> iter_values(%[[ACC:.+]] = %[[ACC_INIT]]) -> (tile<2x2xf32>) {
//       CHECK:      %[[TILE_A:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %[[K_IDX]]] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[TILE_B:.+]], {{.*}} = load_view_tko weak %{{.+}}[%[[K_IDX]], %{{.+}}] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[MMA:.+]] = mmaf %[[TILE_A]], %[[TILE_B]], %[[ACC]] : tile<2x2xf32>, tile<2x2xf32>, tile<2x2xf32>
//       CHECK:      continue %[[MMA]] : tile<2x2xf32>
//       CHECK:    }
//       CHECK:    store_view_tko weak %[[FOR]], %{{.+}}[%{{.+}}, %{{.+}}] :{{.*}}-> token
//       CHECK:    return

// -----

module {
  nv_tensor_ir.graph @matmul_f16_dynamic(
    %a: tensor<?x?xf16>,
    %b: tensor<?x?xf16>) -> (
      tensor<?x?xf16>) {
    %c = "nv_tensor_ir.matmul"(%a, %b) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
    results %c : tensor<?x?xf16>
  }
}

// CHECK-LABEL: entry @matmul_f16_dynamic
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f16>>,{{.*}}%[[B_PTR:.+]]: tile<ptr<f16>>,{{.*}}%[[OUT_PTR:.+]]: tile<ptr<f16>>,{{.*}})
//       CHECK:    %[[TVIEW_A:.+]] = make_tensor_view %[[A_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
//       CHECK:    %[[TVIEW_B:.+]] = make_tensor_view %[[B_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
//       CHECK:    %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
//       CHECK:    %[[BID_X:.+]], %[[BID_Y:.+]], %[[BID_Z:.+]] = get_tile_block_id : tile<i32>
//       CHECK:    %[[PVIEW_A_SHAPE:.+]] = make_partition_view %[[TVIEW_A]] : partition_view<tile=(2x2), tensor_view<?x?xf16, strides=[?,1]>>
//       CHECK:    %[[SHAPE_A:.+]]:2 = get_index_space_shape %[[PVIEW_A_SHAPE]] : partition_view<tile=(2x2), tensor_view<?x?xf16, strides=[?,1]>> -> tile<i32>
//       CHECK:    %[[PVIEW_B_SHAPE:.+]] = make_partition_view %[[TVIEW_B]] : partition_view<tile=(2x2), tensor_view<?x?xf16, strides=[?,1]>>
//       CHECK:    %[[SHAPE_B:.+]]:2 = get_index_space_shape %[[PVIEW_B_SHAPE]] : partition_view<tile=(2x2), tensor_view<?x?xf16, strides=[?,1]>> -> tile<i32>
//       CHECK:    %[[LB:.+]] = constant <i32: 0> : tile<i32>
//       CHECK:    %[[STEP:.+]] = constant <i32: 1> : tile<i32>
//       CHECK:    %[[ACC_INIT:.+]] = constant <f16: 0.000000e+00> : tile<2x2xf16>
//       CHECK:    %[[FOR:.+]] = for %[[K_IDX:.+]] in (%[[LB]] to %[[SHAPE_B]]#0, step %[[STEP]]) : tile<i32> iter_values(%[[ACC:.+]] = %[[ACC_INIT]]) -> (tile<2x2xf16>) {
//       CHECK:      %[[TILE_A:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %[[K_IDX]]] :{{.*}}-> tile<2x2xf16>, token
//       CHECK:      %[[TILE_B:.+]], {{.*}} = load_view_tko weak %{{.+}}[%[[K_IDX]], %{{.+}}] :{{.*}}-> tile<2x2xf16>, token
//       CHECK:      %[[MMA:.+]] = mmaf %[[TILE_A]], %[[TILE_B]], %[[ACC]] : tile<2x2xf16>, tile<2x2xf16>, tile<2x2xf16>
//       CHECK:      continue %[[MMA]] : tile<2x2xf16>
//       CHECK:    }
//       CHECK:    store_view_tko weak %[[FOR]], %{{.+}}[%{{.+}}, %{{.+}}] :{{.*}}-> token
//       CHECK:    return
