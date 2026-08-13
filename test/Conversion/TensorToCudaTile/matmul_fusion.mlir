// RUN: tensor_ir-opt -materialize-default-strides -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s

module {
  nv_tensor_ir.graph @matmul_prologue_static(
    %a: tensor<128x64xf32>,
    %b: tensor<64x128xf32>) -> (
      tensor<128x128xf32>) {
    %neg_a = "nv_tensor_ir.neg"(%a) : (tensor<128x64xf32>) -> tensor<128x64xf32>
    %c = "nv_tensor_ir.matmul"(%neg_a, %b) : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    results %c : tensor<128x128xf32>
  }
}

// CHECK-LABEL: entry @matmul_prologue_static
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f32>>, %[[B_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:    %[[TVIEW_A:.+]] = make_tensor_view %[[A_PTR]], shape = [128, 64], strides = [64, 1] : tensor_view<128x64xf32, strides=[64,1]>
//       CHECK:    %[[TVIEW_B:.+]] = make_tensor_view %[[B_PTR]], shape = [64, 128], strides = [128, 1] : tensor_view<64x128xf32, strides=[128,1]>
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
//       CHECK:      %[[TILE_A:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %[[K_IDX]]] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[TILE_B:.+]], {{.*}} = load_view_tko weak %{{.+}}[%[[K_IDX]], %{{.+}}] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[NEG_A:.+]] = negf %[[TILE_A]] : tile<2x2xf32>
//       CHECK:      %[[MMA:.+]] = mmaf %[[NEG_A]], %[[TILE_B]], %[[ACC]] : tile<2x2xf32>, tile<2x2xf32>, tile<2x2xf32>
//       CHECK:      continue %[[MMA]] : tile<2x2xf32>
//       CHECK:    }
//       CHECK:    %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(2x2), tensor_view<128x128xf32, strides=[128,1]>>
//       CHECK:    store_view_tko weak %[[FOR]], %[[PVIEW_OUT]][%{{.+}}, %{{.+}}] :{{.*}}-> token
//       CHECK:    return

// -----

module {
  nv_tensor_ir.graph @matmul_epilogue_static(
    %a: tensor<128x64xf32>,
    %b: tensor<64x128xf32>,
    %bias: tensor<128x128xf32>) -> (
      tensor<128x128xf32>) {
    %c = "nv_tensor_ir.matmul"(%a, %b) : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    %result = "nv_tensor_ir.mul"(%c, %bias) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    results %result : tensor<128x128xf32>
  }
}

// CHECK-LABEL: entry @matmul_epilogue_static
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f32>>, %[[B_PTR:.+]]: tile<ptr<f32>>, %[[BIAS_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:    %[[TVIEW_A:.+]] = make_tensor_view %[[A_PTR]], shape = [128, 64], strides = [64, 1] : tensor_view<128x64xf32, strides=[64,1]>
//       CHECK:    %[[TVIEW_B:.+]] = make_tensor_view %[[B_PTR]], shape = [64, 128], strides = [128, 1] : tensor_view<64x128xf32, strides=[128,1]>
//       CHECK:    %[[TVIEW_BIAS:.+]] = make_tensor_view %[[BIAS_PTR]], shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
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
//       CHECK:      %[[TILE_A:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %[[K_IDX]]] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[TILE_B:.+]], {{.*}} = load_view_tko weak %{{.+}}[%[[K_IDX]], %{{.+}}] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[MMA:.+]] = mmaf %[[TILE_A]], %[[TILE_B]], %[[ACC]] : tile<2x2xf32>, tile<2x2xf32>, tile<2x2xf32>
//       CHECK:      continue %[[MMA]] : tile<2x2xf32>
//       CHECK:    }
//       CHECK:    %[[TILE_BIAS:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %{{.+}}] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:    %[[MUL:.+]] = mulf %[[FOR]], %[[TILE_BIAS]] : tile<2x2xf32>
//       CHECK:    %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(2x2), tensor_view<128x128xf32, strides=[128,1]>>
//       CHECK:    store_view_tko weak %[[MUL]], %[[PVIEW_OUT]][%{{.+}}, %{{.+}}] :{{.*}}-> token
//       CHECK:    return

// -----

module {
  nv_tensor_ir.graph @matmul_prologue_epilogue_static(
    %a: tensor<128x64xf16>,
    %b: tensor<64x128xf16>,
    %bias: tensor<128x128xf16>) -> (
      tensor<128x128xf16>) {
    %a_f32 = "nv_tensor_ir.convert"(%a) : (tensor<128x64xf16>) -> tensor<128x64xf32>
    %b_f32 = "nv_tensor_ir.convert"(%b) : (tensor<64x128xf16>) -> tensor<64x128xf32>
    %c = "nv_tensor_ir.matmul"(%a_f32, %b_f32) : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    %c_f16 = "nv_tensor_ir.convert"(%c) : (tensor<128x128xf32>) -> tensor<128x128xf16>
    %result = "nv_tensor_ir.add"(%c_f16, %bias) : (tensor<128x128xf16>, tensor<128x128xf16>) -> tensor<128x128xf16>
    results %result : tensor<128x128xf16>
  }
}

// CHECK-LABEL: entry @matmul_prologue_epilogue_static
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f16>>, %[[B_PTR:.+]]: tile<ptr<f16>>, %[[BIAS_PTR:.+]]: tile<ptr<f16>>, %[[OUT_PTR:.+]]: tile<ptr<f16>>)
//       CHECK:    %[[TVIEW_A:.+]] = make_tensor_view %[[A_PTR]], shape = [128, 64], strides = [64, 1] : tensor_view<128x64xf16, strides=[64,1]>
//       CHECK:    %[[TVIEW_B:.+]] = make_tensor_view %[[B_PTR]], shape = [64, 128], strides = [128, 1] : tensor_view<64x128xf16, strides=[128,1]>
//       CHECK:    %[[TVIEW_BIAS:.+]] = make_tensor_view %[[BIAS_PTR]], shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf16, strides=[128,1]>
//       CHECK:    %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf16, strides=[128,1]>
//       CHECK:    %[[BID_X:.+]], %[[BID_Y:.+]], %[[BID_Z:.+]] = get_tile_block_id : tile<i32>
//       CHECK:    %[[PVIEW_A_SHAPE:.+]] = make_partition_view %[[TVIEW_A]] : partition_view<tile=(2x2), tensor_view<128x64xf16, strides=[64,1]>>
//       CHECK:    %[[SHAPE_A:.+]]:2 = get_index_space_shape %[[PVIEW_A_SHAPE]] : partition_view<tile=(2x2), tensor_view<128x64xf16, strides=[64,1]>> -> tile<i32>
//       CHECK:    %[[PVIEW_B_SHAPE:.+]] = make_partition_view %[[TVIEW_B]] : partition_view<tile=(2x2), tensor_view<64x128xf16, strides=[128,1]>>
//       CHECK:    %[[SHAPE_B:.+]]:2 = get_index_space_shape %[[PVIEW_B_SHAPE]] : partition_view<tile=(2x2), tensor_view<64x128xf16, strides=[128,1]>> -> tile<i32>
//       CHECK:    %[[LB:.+]] = constant <i32: 0> : tile<i32>
//       CHECK:    %[[STEP:.+]] = constant <i32: 1> : tile<i32>
//       CHECK:    %[[ACC_INIT:.+]] = constant <f32: 0.000000e+00> : tile<2x2xf32>
//       CHECK:    %[[FOR:.+]] = for %[[IV:.+]] in (%[[LB]] to %[[SHAPE_B]]#0, step %[[STEP]]) : tile<i32> iter_values(%[[ACC:.+]] = %[[ACC_INIT]]) -> (tile<2x2xf32>) {
//       CHECK:      %[[TILE_A_F16:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %[[IV]]] :{{.*}}-> tile<2x2xf16>, token
//       CHECK:      %[[TILE_B_F16:.+]], {{.*}} = load_view_tko weak %{{.+}}[%[[IV]], %{{.+}}] :{{.*}}-> tile<2x2xf16>, token
//       CHECK:      %[[TILE_A_F32:.+]] = ftof %[[TILE_A_F16]] : tile<2x2xf16> -> tile<2x2xf32>
//       CHECK:      %[[TILE_B_F32:.+]] = ftof %[[TILE_B_F16]] : tile<2x2xf16> -> tile<2x2xf32>
//       CHECK:      %[[MMA:.+]] = mmaf %[[TILE_A_F32]], %[[TILE_B_F32]], %[[ACC]] : tile<2x2xf32>, tile<2x2xf32>, tile<2x2xf32>
//       CHECK:      continue %[[MMA]] : tile<2x2xf32>
//       CHECK:    }
//       CHECK:    %[[TILE_BIAS:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %{{.+}}] :{{.*}}-> tile<2x2xf16>, token
//       CHECK:    %[[FOR_F16:.+]] = ftof %[[FOR]] : tile<2x2xf32> -> tile<2x2xf16>
//       CHECK:    %[[ADD:.+]] = addf %[[FOR_F16]], %[[TILE_BIAS]] : tile<2x2xf16>
//       CHECK:    %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(2x2), tensor_view<128x128xf16, strides=[128,1]>>
//       CHECK:    store_view_tko weak %[[ADD]], %[[PVIEW_OUT]][%{{.+}}, %{{.+}}] :{{.*}}-> token
//       CHECK:    return

// -----

module {
  nv_tensor_ir.graph @matmul_prologue_dynamic(
    %a: tensor<?x?xf32>,
    %b: tensor<?x?xf32>) -> (
      tensor<?x?xf32>) {
    %neg_a = "nv_tensor_ir.neg"(%a) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %c = "nv_tensor_ir.matmul"(%neg_a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    results %c : tensor<?x?xf32>
  }
}

// CHECK-LABEL: entry @matmul_prologue_dynamic
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
//       CHECK:    %[[FOR:.+]] = for %[[IV:.+]] in (%[[LB]] to %[[SHAPE_B]]#0, step %[[STEP]]) : tile<i32> iter_values(%[[ACC:.+]] = %[[ACC_INIT]]) -> (tile<2x2xf32>) {
//       CHECK:      %[[TILE_A:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %[[IV]]] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[TILE_B:.+]], {{.*}} = load_view_tko weak %{{.+}}[%[[IV]], %{{.+}}] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[NEG_A:.+]] = negf %[[TILE_A]] : tile<2x2xf32>
//       CHECK:      %[[MMA:.+]] = mmaf %[[NEG_A]], %[[TILE_B]], %[[ACC]] : tile<2x2xf32>, tile<2x2xf32>, tile<2x2xf32>
//       CHECK:      continue %[[MMA]] : tile<2x2xf32>
//       CHECK:    }
//       CHECK:    %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//       CHECK:    store_view_tko weak %[[FOR]], %[[PVIEW_OUT]][%{{.+}}, %{{.+}}] :{{.*}}-> token
//       CHECK:    return

// -----

module {
  nv_tensor_ir.graph @matmul_epilogue_dynamic(
    %a: tensor<?x?xf32>,
    %b: tensor<?x?xf32>,
    %bias: tensor<?x?xf32>) -> (
      tensor<?x?xf32>) {
    %c = "nv_tensor_ir.matmul"(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %result = "nv_tensor_ir.mul"(%c, %bias) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    results %result : tensor<?x?xf32>
  }
}

// CHECK-LABEL: entry @matmul_epilogue_dynamic
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f32>>,{{.*}}%[[B_PTR:.+]]: tile<ptr<f32>>,{{.*}}%[[BIAS_PTR:.+]]: tile<ptr<f32>>,{{.*}}%[[OUT_PTR:.+]]: tile<ptr<f32>>,{{.*}})
//       CHECK:    %[[TVIEW_A:.+]] = make_tensor_view %[[A_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
//       CHECK:    %[[TVIEW_B:.+]] = make_tensor_view %[[B_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
//       CHECK:    %[[TVIEW_BIAS:.+]] = make_tensor_view %[[BIAS_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
//       CHECK:    %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
//       CHECK:    %[[BID_X:.+]], %[[BID_Y:.+]], %[[BID_Z:.+]] = get_tile_block_id : tile<i32>
//       CHECK:    %[[PVIEW_A_SHAPE:.+]] = make_partition_view %[[TVIEW_A]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//       CHECK:    %[[SHAPE_A:.+]]:2 = get_index_space_shape %[[PVIEW_A_SHAPE]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>> -> tile<i32>
//       CHECK:    %[[PVIEW_B_SHAPE:.+]] = make_partition_view %[[TVIEW_B]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//       CHECK:    %[[SHAPE_B:.+]]:2 = get_index_space_shape %[[PVIEW_B_SHAPE]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>> -> tile<i32>
//       CHECK:    %[[LB:.+]] = constant <i32: 0> : tile<i32>
//       CHECK:    %[[STEP:.+]] = constant <i32: 1> : tile<i32>
//       CHECK:    %[[ACC_INIT:.+]] = constant <f32: 0.000000e+00> : tile<2x2xf32>
//       CHECK:    %[[FOR:.+]] = for %[[IV:.+]] in (%[[LB]] to %[[SHAPE_B]]#0, step %[[STEP]]) : tile<i32> iter_values(%[[ACC:.+]] = %[[ACC_INIT]]) -> (tile<2x2xf32>) {
//       CHECK:      %[[TILE_A:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %[[IV]]] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[TILE_B:.+]], {{.*}} = load_view_tko weak %{{.+}}[%[[IV]], %{{.+}}] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:      %[[MMA:.+]] = mmaf %[[TILE_A]], %[[TILE_B]], %[[ACC]] : tile<2x2xf32>, tile<2x2xf32>, tile<2x2xf32>
//       CHECK:      continue %[[MMA]] : tile<2x2xf32>
//       CHECK:    }
//       CHECK:    %[[TILE_BIAS:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %{{.+}}] :{{.*}}-> tile<2x2xf32>, token
//       CHECK:    %[[MUL:.+]] = mulf %[[FOR]], %[[TILE_BIAS]] : tile<2x2xf32>
//       CHECK:    %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(2x2), tensor_view<?x?xf32, strides=[?,1]>>
//       CHECK:    store_view_tko weak %[[MUL]], %[[PVIEW_OUT]][%{{.+}}, %{{.+}}] :{{.*}}-> token
//       CHECK:    return

// -----

module {
  nv_tensor_ir.graph @matmul_prologue_epilogue_dynamic(
    %a: tensor<?x?xf16>,
    %b: tensor<?x?xf16>,
    %bias: tensor<?x?xf16>) -> (
      tensor<?x?xf16>) {
    %a_f32 = "nv_tensor_ir.convert"(%a) : (tensor<?x?xf16>) -> tensor<?x?xf32>
    %b_f32 = "nv_tensor_ir.convert"(%b) : (tensor<?x?xf16>) -> tensor<?x?xf32>
    %c = "nv_tensor_ir.matmul"(%a_f32, %b_f32) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %c_f16 = "nv_tensor_ir.convert"(%c) : (tensor<?x?xf32>) -> tensor<?x?xf16>
    %result = "nv_tensor_ir.add"(%c_f16, %bias) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
    results %result : tensor<?x?xf16>
  }
}

// CHECK-LABEL: entry @matmul_prologue_epilogue_dynamic
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f16>>,{{.*}}%[[B_PTR:.+]]: tile<ptr<f16>>,{{.*}}%[[BIAS_PTR:.+]]: tile<ptr<f16>>,{{.*}}%[[OUT_PTR:.+]]: tile<ptr<f16>>,{{.*}})
//       CHECK:    %[[TVIEW_A:.+]] = make_tensor_view %[[A_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
//       CHECK:    %[[TVIEW_B:.+]] = make_tensor_view %[[B_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
//       CHECK:    %[[TVIEW_BIAS:.+]] = make_tensor_view %[[BIAS_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
//       CHECK:    %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]],{{.*}}: tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
//       CHECK:    %[[BID_X:.+]], %[[BID_Y:.+]], %[[BID_Z:.+]] = get_tile_block_id : tile<i32>
//       CHECK:    %[[PVIEW_A_SHAPE:.+]] = make_partition_view %[[TVIEW_A]] : partition_view<tile=(2x2), tensor_view<?x?xf16, strides=[?,1]>>
//       CHECK:    %[[SHAPE_A:.+]]:2 = get_index_space_shape %[[PVIEW_A_SHAPE]] : partition_view<tile=(2x2), tensor_view<?x?xf16, strides=[?,1]>> -> tile<i32>
//       CHECK:    %[[PVIEW_B_SHAPE:.+]] = make_partition_view %[[TVIEW_B]] : partition_view<tile=(2x2), tensor_view<?x?xf16, strides=[?,1]>>
//       CHECK:    %[[SHAPE_B:.+]]:2 = get_index_space_shape %[[PVIEW_B_SHAPE]] : partition_view<tile=(2x2), tensor_view<?x?xf16, strides=[?,1]>> -> tile<i32>
//       CHECK:    %[[LB:.+]] = constant <i32: 0> : tile<i32>
//       CHECK:    %[[STEP:.+]] = constant <i32: 1> : tile<i32>
//       CHECK:    %[[ACC_INIT:.+]] = constant <f32: 0.000000e+00> : tile<2x2xf32>
//       CHECK:    %[[FOR:.+]] = for %[[IV:.+]] in (%[[LB]] to %[[SHAPE_B]]#0, step %[[STEP]]) : tile<i32> iter_values(%[[ACC:.+]] = %[[ACC_INIT]]) -> (tile<2x2xf32>) {
//       CHECK:      %[[TILE_A_F16:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %[[IV]]] :{{.*}}-> tile<2x2xf16>, token
//       CHECK:      %[[TILE_B_F16:.+]], {{.*}} = load_view_tko weak %{{.+}}[%[[IV]], %{{.+}}] :{{.*}}-> tile<2x2xf16>, token
//       CHECK:      %[[TILE_A_F32:.+]] = ftof %[[TILE_A_F16]] : tile<2x2xf16> -> tile<2x2xf32>
//       CHECK:      %[[TILE_B_F32:.+]] = ftof %[[TILE_B_F16]] : tile<2x2xf16> -> tile<2x2xf32>
//       CHECK:      %[[MMA:.+]] = mmaf %[[TILE_A_F32]], %[[TILE_B_F32]], %[[ACC]] : tile<2x2xf32>, tile<2x2xf32>, tile<2x2xf32>
//       CHECK:      continue %[[MMA]] : tile<2x2xf32>
//       CHECK:    }
//       CHECK:    %[[TILE_BIAS:.+]], {{.*}} = load_view_tko weak %{{.+}}[%{{.+}}, %{{.+}}] :{{.*}}-> tile<2x2xf16>, token
//       CHECK:    %[[FOR_F16:.+]] = ftof %[[FOR]] : tile<2x2xf32> -> tile<2x2xf16>
//       CHECK:    %[[ADD:.+]] = addf %[[FOR_F16]], %[[TILE_BIAS]] : tile<2x2xf16>
//       CHECK:    %[[PVIEW_OUT:.+]] = make_partition_view %[[TVIEW_OUT]] : partition_view<tile=(2x2), tensor_view<?x?xf16, strides=[?,1]>>
//       CHECK:    store_view_tko weak %[[ADD]], %[[PVIEW_OUT]][%{{.+}}, %{{.+}}] :{{.*}}-> token
//       CHECK:    return
