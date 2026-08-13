// RUN: tensor_ir-opt -materialize-default-strides -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s

// This file tests ops that create high-level compiler scopes, like modules and graphs.

module @moduloX {
  nv_tensor_ir.graph @test_module_name() {
    results
  }
}

// CHECK-LABEL: module @moduloX {
//  CHECK-NEXT:   cuda_tile.module @cuda_tile_moduloX {
//  CHECK-NEXT:     entry @test_module_name() {
//  CHECK-NEXT:       return
//  CHECK-NEXT:     }
//  CHECK-NEXT:   }
//  CHECK-NEXT: }

// -----

module {
  nv_tensor_ir.graph @test_module_no_name() {
      results
  }
}

//        CHECK: module {
//  CHECK-LABEL:   cuda_tile.module @cuda_tile_module {
//   CHECK-NEXT:     entry @test_module_no_name() {
//   CHECK-NEXT:       return
//   CHECK-NEXT:     }
//   CHECK-NEXT:   }
//   CHECK-NEXT: }

// -----

module {
  nv_tensor_ir.graph @test_static_inputs_outputs(
    %a: tensor<8x16x32xf16>,
    %b: tensor<64x128xf32>) -> (
      tensor<8x16x32xf16>,
      tensor<64x128xf32>) {
      results %a, %b : tensor<8x16x32xf16>, tensor<64x128xf32>
  }
}

// CHECK-LABEL: entry @test_static_inputs_outputs
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f16>>, %[[B_PTR:.+]]: tile<ptr<f32>>, %[[OUT0_PTR:.+]]: tile<ptr<f16>>, %[[OUT1_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TVIEW:.+]] = make_tensor_view %[[A_PTR]], shape = [8, 16, 32], strides = [512, 32, 1] : tensor_view<8x16x32xf16, strides=[512,32,1]>
//       CHECK:   %[[TVIEW_0:.+]] = make_tensor_view %[[B_PTR]], shape = [64, 128], strides = [128, 1] : tensor_view<64x128xf32, strides=[128,1]>
//       CHECK:   %[[TVIEW_1:.+]] = make_tensor_view %[[OUT0_PTR]], shape = [8, 16, 32], strides = [512, 32, 1] : tensor_view<8x16x32xf16, strides=[512,32,1]>
//       CHECK:   %[[TVIEW_2:.+]] = make_tensor_view %[[OUT1_PTR]], shape = [64, 128], strides = [128, 1] : tensor_view<64x128xf32, strides=[128,1]>
//       CHECK:   %[[BID_X:.+]], {{.*}}, {{.*}} = get_tile_block_id : tile<i32>
//       CHECK:   %[[PVIEW:.+]] = make_partition_view %[[TVIEW]] : partition_view<tile=(2x2x2), tensor_view<8x16x32xf16, strides=[512,32,1]>>
//       CHECK:   %[[SHAPE:.+]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(2x2x2), tensor_view<8x16x32xf16, strides=[512,32,1]>> -> tile<i32>
//       CHECK:   %[[IDX0:.+]] = remi %[[BID_X]], %[[SHAPE]]#2 unsigned : tile<i32>
//       CHECK:   %[[TMP0:.+]] = divi %[[BID_X]], %[[SHAPE]]#2 unsigned : tile<i32>
//       CHECK:   %[[IDX1:.+]] = remi %[[TMP0]], %[[SHAPE]]#1 unsigned : tile<i32>
//       CHECK:   %[[TMP1:.+]] = divi %[[TMP0]], %[[SHAPE]]#1 unsigned : tile<i32>
//       CHECK:   %[[IDX2:.+]] = remi %[[TMP1]], %[[SHAPE]]#0 unsigned : tile<i32>
//       CHECK:   %[[TILE:.+]], {{.*}} = load_view_tko weak %{{.*}}[%[[IDX2]], %[[IDX1]], %[[IDX0]]] :{{.*}}-> tile<2x2x2xf16>, token
//       CHECK:   %[[PVIEW_3:.+]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(2x2), tensor_view<64x128xf32, strides=[128,1]>>
//       CHECK:   %[[TILE_4:.+]], {{.*}} = load_view_tko weak %[[PVIEW_3]][{{.*}}] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:   %[[PVIEW_6:.+]] = make_partition_view %[[TVIEW_1]] : partition_view<tile=(2x2x2), tensor_view<8x16x32xf16, strides=[512,32,1]>>
//       CHECK:   store_view_tko weak %[[TILE]], %[[PVIEW_6]][{{.*}}] :{{.*}}-> token
//       CHECK:   %[[PVIEW_7:.+]] = make_partition_view %[[TVIEW_2]] : partition_view<tile=(2x2), tensor_view<64x128xf32, strides=[128,1]>>
//       CHECK:   store_view_tko weak %[[TILE_4]], %[[PVIEW_7]][{{.*}}] :{{.*}}-> token
//       CHECK:   return

// -----

module {
  nv_tensor_ir.graph @test_dynamic_inputs_outputs(
    %a: tensor<?x?x?xf16> {nv_tensor_ir.stride = "(?,?,1)"},
    %b: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"}) -> (
      tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"},
      tensor<?x?x?xf16> {nv_tensor_ir.stride = "(?,?,1)"}) {
      results %b, %a : tensor<?x?xf32>, tensor<?x?x?xf16>
  }
}

// With explicit stride attributes, only dynamic stride components are passed as args.
// stride (?,?,1) has 2 dynamic strides, stride (?,1) has 1 dynamic stride.
// Tensor A: ptr + 3 sizes + 2 strides = 6 args
// Tensor B: ptr + 2 sizes + 1 stride = 4 args
// Output 0: ptr + 2 sizes + 1 stride = 4 args
// Output 1: ptr + 3 sizes + 2 strides = 6 args
// Total: 20 args

// CHECK-LABEL: entry @test_dynamic_inputs_outputs
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f16>>,{{.*}}%[[B_PTR:.+]]: tile<ptr<f32>>,{{.*}}%[[OUT0_PTR:.+]]: tile<ptr<f32>>,{{.*}}%[[OUT1_PTR:.+]]: tile<ptr<f16>>,{{.*}})
//       CHECK:   %[[TVIEW:.+]] = make_tensor_view %[[A_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?x?xf16, strides=[?,?,1]>
//       CHECK:   %[[TVIEW_0:.+]] = make_tensor_view %[[B_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
//       CHECK:   %[[TVIEW_1:.+]] = make_tensor_view %[[OUT0_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
//       CHECK:   %[[TVIEW_2:.+]] = make_tensor_view %[[OUT1_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?x?xf16, strides=[?,?,1]>
//       CHECK:   %[[BID_X:.+]], {{.*}}, {{.*}} = get_tile_block_id : tile<i32>
//       CHECK:   %[[PVIEW:.+]] = make_partition_view %[[TVIEW]] : partition_view<tile=(2x2x2), tensor_view<?x?x?xf16, strides=[?,?,1]>>
//       CHECK:   %[[SHAPE:.+]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(2x2x2), tensor_view<?x?x?xf16, strides=[?,?,1]>> -> tile<i32>
//       CHECK:   %[[IDX0:.+]] = remi %[[BID_X]], %[[SHAPE]]#2 unsigned : tile<i32>
//       CHECK:   %[[TMP0:.+]] = divi %[[BID_X]], %[[SHAPE]]#2 unsigned : tile<i32>
//       CHECK:   %[[IDX1:.+]] = remi %[[TMP0]], %[[SHAPE]]#1 unsigned : tile<i32>
//       CHECK:   %[[TMP1:.+]] = divi %[[TMP0]], %[[SHAPE]]#1 unsigned : tile<i32>
//       CHECK:   %[[IDX2:.+]] = remi %[[TMP1]], %[[SHAPE]]#0 unsigned : tile<i32>
//       CHECK:   %[[TILE:.+]], {{.*}} = load_view_tko weak %{{.*}}[%[[IDX2]], %[[IDX1]], %[[IDX0]]] :{{.*}}-> tile<2x2x2xf16>, token
//       CHECK:   %[[PVIEW_3:.+]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//       CHECK:   %[[TILE_4:.+]], {{.*}} = load_view_tko weak %[[PVIEW_3]][{{.*}}] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:   %[[PVIEW_6:.+]] = make_partition_view %[[TVIEW_1]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//       CHECK:   store_view_tko weak %[[TILE_4]], %[[PVIEW_6]][{{.*}}] :{{.*}}-> token
//       CHECK:   %[[PVIEW_7:.+]] = make_partition_view %[[TVIEW_2]] : partition_view<tile=(2x2x2), tensor_view<?x?x?xf16, strides=[?,?,1]>>
//       CHECK:   store_view_tko weak %[[TILE]], %[[PVIEW_7]][{{.*}}] :{{.*}}-> token
//       CHECK:   return
