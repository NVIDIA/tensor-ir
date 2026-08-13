// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s

// Test ReduceOp with add mode (sum reduction)

module {
// CHECK-LABEL:   cuda_tile.module @cuda_tile_module {
// CHECK:           entry @reduction_op_add(%[[ARG0:.*]]: tile<ptr<f32>>, %[[ARG1:.*]]: tile<ptr<f32>>) {
// CHECK:             %[[TVIEW:.*]] = make_tensor_view %[[ARG0]], shape = [16, 32, 64], strides = [2048, 64, 1] : tensor_view<16x32x64xf32, strides=[2048,64,1]>
// CHECK:             %[[TVIEW_0:.*]] = make_tensor_view %[[ARG1]], shape = [16, 1, 64], strides = [64, 64, 1] : tensor_view<16x1x64xf32, strides=[64,64,1]>
// CHECK:             %[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]] = get_tile_block_id : tile<i32>
// CHECK:             %[[PVIEW:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<16x32x64xf32, strides=[2048,64,1]>>
// CHECK:             %[[IDX_SPACE:.*]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(8x32x64), tensor_view<16x32x64xf32, strides=[2048,64,1]>> -> tile<i32>
// CHECK:             %[[PVIEW_1:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<16x32x64xf32, strides=[2048,64,1]>>
// CHECK:             %[[COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[COORD_1:.*]] = remi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[TMP_1:.*]] = divi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[COORD_0:.*]] = remi %[[TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[TILE:.*]], %[[RESULT_TOKEN:.*]] = load_view_tko weak %[[PVIEW_1]]{{\[}}%[[COORD_0]], %[[COORD_1]], %[[COORD_2]]] : partition_view<tile=(8x32x64), tensor_view<16x32x64xf32, strides=[2048,64,1]>>, tile<i32> -> tile<8x32x64xf32>, token
// CHECK:             %[[REDUCE:.*]] = reduce %[[TILE]] dim=1 identities=[0.000000e+00 : f32] : tile<8x32x64xf32> -> tile<8x64xf32>
// CHECK:             (%[[REDUCE_LHS:.*]]: tile<f32>, %[[REDUCE_RHS:.*]]: tile<f32>) {
// CHECK:               %[[ADD:.*]] = addf %[[REDUCE_LHS]], %[[REDUCE_RHS]]  : tile<f32>
// CHECK:               yield %[[ADD]] : tile<f32>
// CHECK:             }
// CHECK:             %[[RESHAPE:.*]] = reshape %[[REDUCE]] : tile<8x64xf32> -> tile<8x1x64xf32>
// CHECK:             %[[PVIEW_2:.*]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(8x1x64), tensor_view<16x1x64xf32, strides=[64,64,1]>>
// CHECK:             %[[OUT_COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_1:.*]] = remi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_1:.*]] = divi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_0:.*]] = remi %[[OUT_TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[STORE_TOKEN:.*]] = store_view_tko weak %[[RESHAPE]], %[[PVIEW_2]]{{\[}}%[[OUT_COORD_0]], %[[OUT_COORD_1]], %[[OUT_COORD_2]]] : tile<8x1x64xf32>, partition_view<tile=(8x1x64), tensor_view<16x1x64xf32, strides=[64,64,1]>>, tile<i32> -> token
// CHECK:             return
// CHECK:           }
// CHECK:         }
  nv_tensor_ir.graph @reduction_op_add(%input: tensor<16x32x64xf32>{nv_tensor_ir.stride = "(2048,64,1)"}) -> (tensor<16x1x64xf32> {nv_tensor_ir.stride = "(64,64,1)"}) {
    %result = reduce(%input) <
      dimensions = [1],
      reduction_mode = <add>> : tensor<16x32x64xf32> -> tensor<16x1x64xf32>
    results %result : tensor<16x1x64xf32>
  }
}

// -----
// Test ReduceOp with mul mode

module {
// CHECK-LABEL:   cuda_tile.module @cuda_tile_module {
// CHECK:           entry @reduction_op_mul(%[[ARG0:.*]]: tile<ptr<f32>>, %[[ARG1:.*]]: tile<ptr<f32>>) {
// CHECK:             %[[TVIEW:.*]] = make_tensor_view %[[ARG0]], shape = [8, 32, 128], strides = [1, 8, 256] : tensor_view<8x32x128xf32, strides=[1,8,256]>
// CHECK:             %[[TVIEW_0:.*]] = make_tensor_view %[[ARG1]], shape = [8, 1, 128], strides = [1, 1, 8] : tensor_view<8x1x128xf32, strides=[1,1,8]>
// CHECK:             %[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]] = get_tile_block_id : tile<i32>
// CHECK:             %[[PVIEW:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[IDX_SPACE:.*]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>> -> tile<i32>
// CHECK:             %[[PVIEW_1:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[COORD_1:.*]] = remi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[TMP_1:.*]] = divi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[COORD_0:.*]] = remi %[[TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[TILE:.*]], %[[RESULT_TOKEN:.*]] = load_view_tko weak %[[PVIEW_1]]{{\[}}%[[COORD_0]], %[[COORD_1]], %[[COORD_2]]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>, tile<i32> -> tile<8x32x64xf32>, token
// CHECK:             %[[REDUCE:.*]] = reduce %[[TILE]] dim=1 identities=[1.000000e+00 : f32] : tile<8x32x64xf32> -> tile<8x64xf32>
// CHECK:             (%[[REDUCE_LHS:.*]]: tile<f32>, %[[REDUCE_RHS:.*]]: tile<f32>) {
// CHECK:               %[[MUL:.*]] = mulf %[[REDUCE_LHS]], %[[REDUCE_RHS]]  : tile<f32>
// CHECK:               yield %[[MUL]] : tile<f32>
// CHECK:             }
// CHECK:             %[[RESHAPE:.*]] = reshape %[[REDUCE]] : tile<8x64xf32> -> tile<8x1x64xf32>
// CHECK:             %[[PVIEW_2:.*]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>
// CHECK:             %[[OUT_COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_1:.*]] = remi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_1:.*]] = divi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_0:.*]] = remi %[[OUT_TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[STORE_TOKEN:.*]] = store_view_tko weak %[[RESHAPE]], %[[PVIEW_2]]{{\[}}%[[OUT_COORD_0]], %[[OUT_COORD_1]], %[[OUT_COORD_2]]] : tile<8x1x64xf32>, partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>, tile<i32> -> token
// CHECK:             return
// CHECK:           }
// CHECK:         }
  nv_tensor_ir.graph @reduction_op_mul(%input: tensor<8x32x128xf32>{nv_tensor_ir.stride = "(1,8,256)"}) -> (tensor<8x1x128xf32> {nv_tensor_ir.stride = "(1,1,8)"}) {
    %result = reduce(%input) <
      dimensions = [1],
      reduction_mode = <mul>> : tensor<8x32x128xf32> -> tensor<8x1x128xf32>
    results %result : tensor<8x1x128xf32>
  }
}

// -----
// Test ReduceOp with avg mode (average)

module {
// CHECK-LABEL:   cuda_tile.module @cuda_tile_module {
// CHECK:           entry @reduction_op_avg(%[[ARG0:.*]]: tile<ptr<f32>>, %[[ARG1:.*]]: tile<ptr<f32>>) {
// CHECK:             %[[TVIEW:.*]] = make_tensor_view %[[ARG0]], shape = [8, 32, 128], strides = [1, 8, 256] : tensor_view<8x32x128xf32, strides=[1,8,256]>
// CHECK:             %[[TVIEW_0:.*]] = make_tensor_view %[[ARG1]], shape = [8, 1, 128], strides = [1, 1, 8] : tensor_view<8x1x128xf32, strides=[1,1,8]>
// CHECK:             %[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]] = get_tile_block_id : tile<i32>
// CHECK:             %[[PVIEW:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[IDX_SPACE:.*]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>> -> tile<i32>
// CHECK:             %[[PVIEW_1:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[COORD_1:.*]] = remi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[TMP_1:.*]] = divi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[COORD_0:.*]] = remi %[[TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[TILE:.*]], %[[RESULT_TOKEN:.*]] = load_view_tko weak %[[PVIEW_1]]{{\[}}%[[COORD_0]], %[[COORD_1]], %[[COORD_2]]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>, tile<i32> -> tile<8x32x64xf32>, token
// CHECK:             %[[REDUCE:.*]] = reduce %[[TILE]] dim=1 identities=[0.000000e+00 : f32] : tile<8x32x64xf32> -> tile<8x64xf32>
// CHECK:             (%[[REDUCE_LHS:.*]]: tile<f32>, %[[REDUCE_RHS:.*]]: tile<f32>) {
// CHECK:               %[[ADD:.*]] = addf %[[REDUCE_LHS]], %[[REDUCE_RHS]]  : tile<f32>
// CHECK:               yield %[[ADD]] : tile<f32>
// CHECK:             }
// CHECK:             %[[DIVISOR:.*]] = constant <f32: 3.200000e+01> : tile<8x64xf32>
// CHECK:             %[[AVG:.*]] = divf %[[REDUCE]], %[[DIVISOR]]  : tile<8x64xf32>
// CHECK:             %[[RESHAPE:.*]] = reshape %[[AVG]] : tile<8x64xf32> -> tile<8x1x64xf32>
// CHECK:             %[[PVIEW_2:.*]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>
// CHECK:             %[[OUT_COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_1:.*]] = remi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_1:.*]] = divi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_0:.*]] = remi %[[OUT_TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[STORE_TOKEN:.*]] = store_view_tko weak %[[RESHAPE]], %[[PVIEW_2]]{{\[}}%[[OUT_COORD_0]], %[[OUT_COORD_1]], %[[OUT_COORD_2]]] : tile<8x1x64xf32>, partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>, tile<i32> -> token
// CHECK:             return
// CHECK:           }
// CHECK:         }
  nv_tensor_ir.graph @reduction_op_avg(%input: tensor<8x32x128xf32>{nv_tensor_ir.stride = "(1,8,256)"}) -> (tensor<8x1x128xf32> {nv_tensor_ir.stride = "(1,1,8)"}) {
    %result = reduce(%input) <
      dimensions = [1],
      reduction_mode = <avg>> : tensor<8x32x128xf32> -> tensor<8x1x128xf32>
    results %result : tensor<8x1x128xf32>
  }
}

// -----
// Test ReduceOp with min mode

module {
// CHECK-LABEL:   cuda_tile.module @cuda_tile_module {
// CHECK:           entry @reduction_op_min(%[[ARG0:.*]]: tile<ptr<f32>>, %[[ARG1:.*]]: tile<ptr<f32>>) {
// CHECK:             %[[TVIEW:.*]] = make_tensor_view %[[ARG0]], shape = [8, 32, 128], strides = [1, 8, 256] : tensor_view<8x32x128xf32, strides=[1,8,256]>
// CHECK:             %[[TVIEW_0:.*]] = make_tensor_view %[[ARG1]], shape = [8, 1, 128], strides = [1, 1, 8] : tensor_view<8x1x128xf32, strides=[1,1,8]>
// CHECK:             %[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]] = get_tile_block_id : tile<i32>
// CHECK:             %[[PVIEW:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[IDX_SPACE:.*]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>> -> tile<i32>
// CHECK:             %[[PVIEW_1:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[COORD_1:.*]] = remi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[TMP_1:.*]] = divi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[COORD_0:.*]] = remi %[[TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[TILE:.*]], %[[RESULT_TOKEN:.*]] = load_view_tko weak %[[PVIEW_1]]{{\[}}%[[COORD_0]], %[[COORD_1]], %[[COORD_2]]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>, tile<i32> -> tile<8x32x64xf32>, token
// CHECK:             %[[REDUCE:.*]] = reduce %[[TILE]] dim=1 identities=[0x7F800000 : f32] : tile<8x32x64xf32> -> tile<8x64xf32>
// CHECK:             (%[[REDUCE_LHS:.*]]: tile<f32>, %[[REDUCE_RHS:.*]]: tile<f32>) {
// CHECK:               %[[MIN:.*]] = minf %[[REDUCE_LHS]], %[[REDUCE_RHS]] propagate_nan : tile<f32>
// CHECK:               yield %[[MIN]] : tile<f32>
// CHECK:             }
// CHECK:             %[[RESHAPE:.*]] = reshape %[[REDUCE]] : tile<8x64xf32> -> tile<8x1x64xf32>
// CHECK:             %[[PVIEW_2:.*]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>
// CHECK:             %[[OUT_COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_1:.*]] = remi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_1:.*]] = divi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_0:.*]] = remi %[[OUT_TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[STORE_TOKEN:.*]] = store_view_tko weak %[[RESHAPE]], %[[PVIEW_2]]{{\[}}%[[OUT_COORD_0]], %[[OUT_COORD_1]], %[[OUT_COORD_2]]] : tile<8x1x64xf32>, partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>, tile<i32> -> token
// CHECK:             return
// CHECK:           }
// CHECK:         }
  nv_tensor_ir.graph @reduction_op_min(%input: tensor<8x32x128xf32>{nv_tensor_ir.stride = "(1,8,256)"}) -> (tensor<8x1x128xf32> {nv_tensor_ir.stride = "(1,1,8)"}) {
    %result = reduce(%input) <
      dimensions = [1],
      reduction_mode = <min>> : tensor<8x32x128xf32> -> tensor<8x1x128xf32>
    results %result : tensor<8x1x128xf32>
  }
}

// -----
// Test ReduceOp with max mode

module {
// CHECK-LABEL:   cuda_tile.module @cuda_tile_module {
// CHECK:           entry @reduction_op_max(%[[ARG0:.*]]: tile<ptr<f32>>, %[[ARG1:.*]]: tile<ptr<f32>>) {
// CHECK:             %[[TVIEW:.*]] = make_tensor_view %[[ARG0]], shape = [8, 32, 128], strides = [1, 8, 256] : tensor_view<8x32x128xf32, strides=[1,8,256]>
// CHECK:             %[[TVIEW_0:.*]] = make_tensor_view %[[ARG1]], shape = [8, 1, 128], strides = [1, 1, 8] : tensor_view<8x1x128xf32, strides=[1,1,8]>
// CHECK:             %[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]] = get_tile_block_id : tile<i32>
// CHECK:             %[[PVIEW:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[IDX_SPACE:.*]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>> -> tile<i32>
// CHECK:             %[[PVIEW_1:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[COORD_1:.*]] = remi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[TMP_1:.*]] = divi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[COORD_0:.*]] = remi %[[TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[TILE:.*]], %[[RESULT_TOKEN:.*]] = load_view_tko weak %[[PVIEW_1]]{{\[}}%[[COORD_0]], %[[COORD_1]], %[[COORD_2]]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>, tile<i32> -> tile<8x32x64xf32>, token
// CHECK:             %[[REDUCE:.*]] = reduce %[[TILE]] dim=1 identities=[0xFF800000 : f32] : tile<8x32x64xf32> -> tile<8x64xf32>
// CHECK:             (%[[REDUCE_LHS:.*]]: tile<f32>, %[[REDUCE_RHS:.*]]: tile<f32>) {
// CHECK:               %[[MAX:.*]] = maxf %[[REDUCE_LHS]], %[[REDUCE_RHS]] propagate_nan : tile<f32>
// CHECK:               yield %[[MAX]] : tile<f32>
// CHECK:             }
// CHECK:             %[[RESHAPE:.*]] = reshape %[[REDUCE]] : tile<8x64xf32> -> tile<8x1x64xf32>
// CHECK:             %[[PVIEW_2:.*]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>
// CHECK:             %[[OUT_COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_1:.*]] = remi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_1:.*]] = divi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_0:.*]] = remi %[[OUT_TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[STORE_TOKEN:.*]] = store_view_tko weak %[[RESHAPE]], %[[PVIEW_2]]{{\[}}%[[OUT_COORD_0]], %[[OUT_COORD_1]], %[[OUT_COORD_2]]] : tile<8x1x64xf32>, partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>, tile<i32> -> token
// CHECK:             return
// CHECK:           }
// CHECK:         }
  nv_tensor_ir.graph @reduction_op_max(%input: tensor<8x32x128xf32>{nv_tensor_ir.stride = "(1,8,256)"}) -> (tensor<8x1x128xf32> {nv_tensor_ir.stride = "(1,1,8)"}) {
    %result = reduce(%input) <
      dimensions = [1],
      reduction_mode = <max>> : tensor<8x32x128xf32> -> tensor<8x1x128xf32>
    results %result : tensor<8x1x128xf32>
  }
}

// -----
// Test ReduceOp with amax mode (absolute maximum)

module {
// CHECK-LABEL:   cuda_tile.module @cuda_tile_module {
// CHECK:           entry @reduction_op_amax(%[[ARG0:.*]]: tile<ptr<f32>>, %[[ARG1:.*]]: tile<ptr<f32>>) {
// CHECK:             %[[TVIEW:.*]] = make_tensor_view %[[ARG0]], shape = [16, 32, 64], strides = [2048, 64, 1] : tensor_view<16x32x64xf32, strides=[2048,64,1]>
// CHECK:             %[[TVIEW_0:.*]] = make_tensor_view %[[ARG1]], shape = [16, 1, 64], strides = [64, 64, 1] : tensor_view<16x1x64xf32, strides=[64,64,1]>
// CHECK:             %[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]] = get_tile_block_id : tile<i32>
// CHECK:             %[[PVIEW:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<16x32x64xf32, strides=[2048,64,1]>>
// CHECK:             %[[IDX_SPACE:.*]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(8x32x64), tensor_view<16x32x64xf32, strides=[2048,64,1]>> -> tile<i32>
// CHECK:             %[[PVIEW_1:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<16x32x64xf32, strides=[2048,64,1]>>
// CHECK:             %[[COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[COORD_1:.*]] = remi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[TMP_1:.*]] = divi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[COORD_0:.*]] = remi %[[TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[TILE:.*]], %[[RESULT_TOKEN:.*]] = load_view_tko weak %[[PVIEW_1]]{{\[}}%[[COORD_0]], %[[COORD_1]], %[[COORD_2]]] : partition_view<tile=(8x32x64), tensor_view<16x32x64xf32, strides=[2048,64,1]>>, tile<i32> -> tile<8x32x64xf32>, token
// CHECK:             %[[ABS:.*]] = absf %[[TILE]] : tile<8x32x64xf32>
// CHECK:             %[[REDUCE:.*]] = reduce %[[ABS]] dim=1 identities=[0.000000e+00 : f32] : tile<8x32x64xf32> -> tile<8x64xf32>
// CHECK:             (%[[REDUCE_LHS:.*]]: tile<f32>, %[[REDUCE_RHS:.*]]: tile<f32>) {
// CHECK:               %[[MAX:.*]] = maxf %[[REDUCE_LHS]], %[[REDUCE_RHS]] propagate_nan : tile<f32>
// CHECK:               yield %[[MAX]] : tile<f32>
// CHECK:             }
// CHECK:             %[[RESHAPE:.*]] = reshape %[[REDUCE]] : tile<8x64xf32> -> tile<8x1x64xf32>
// CHECK:             %[[PVIEW_2:.*]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(8x1x64), tensor_view<16x1x64xf32, strides=[64,64,1]>>
// CHECK:             %[[OUT_COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_1:.*]] = remi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_1:.*]] = divi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_0:.*]] = remi %[[OUT_TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[STORE_TOKEN:.*]] = store_view_tko weak %[[RESHAPE]], %[[PVIEW_2]]{{\[}}%[[OUT_COORD_0]], %[[OUT_COORD_1]], %[[OUT_COORD_2]]] : tile<8x1x64xf32>, partition_view<tile=(8x1x64), tensor_view<16x1x64xf32, strides=[64,64,1]>>, tile<i32> -> token
// CHECK:             return
// CHECK:           }
// CHECK:         }
  nv_tensor_ir.graph @reduction_op_amax(%input: tensor<16x32x64xf32>{nv_tensor_ir.stride = "(2048,64,1)"}) -> (tensor<16x1x64xf32> {nv_tensor_ir.stride = "(64,64,1)"}) {
    %result = reduce(%input) <
      dimensions = [1],
      reduction_mode = <amax>> : tensor<16x32x64xf32> -> tensor<16x1x64xf32>
    results %result : tensor<16x1x64xf32>
  }
}

// -----
// Test ReduceOp with norm1 mode (L1 norm)

module {
// CHECK-LABEL:   cuda_tile.module @cuda_tile_module {
// CHECK:           entry @reduction_op_norm1(%[[ARG0:.*]]: tile<ptr<f32>>, %[[ARG1:.*]]: tile<ptr<f32>>) {
// CHECK:             %[[TVIEW:.*]] = make_tensor_view %[[ARG0]], shape = [8, 32, 128], strides = [1, 8, 256] : tensor_view<8x32x128xf32, strides=[1,8,256]>
// CHECK:             %[[TVIEW_0:.*]] = make_tensor_view %[[ARG1]], shape = [8, 1, 128], strides = [1, 1, 8] : tensor_view<8x1x128xf32, strides=[1,1,8]>
// CHECK:             %[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]] = get_tile_block_id : tile<i32>
// CHECK:             %[[PVIEW:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[IDX_SPACE:.*]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>> -> tile<i32>
// CHECK:             %[[PVIEW_1:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[COORD_1:.*]] = remi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[TMP_1:.*]] = divi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[COORD_0:.*]] = remi %[[TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[TILE:.*]], %[[RESULT_TOKEN:.*]] = load_view_tko weak %[[PVIEW_1]]{{\[}}%[[COORD_0]], %[[COORD_1]], %[[COORD_2]]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>, tile<i32> -> tile<8x32x64xf32>, token
// CHECK:             %[[ABS:.*]] = absf %[[TILE]] : tile<8x32x64xf32>
// CHECK:             %[[REDUCE:.*]] = reduce %[[ABS]] dim=1 identities=[0.000000e+00 : f32] : tile<8x32x64xf32> -> tile<8x64xf32>
// CHECK:             (%[[REDUCE_LHS:.*]]: tile<f32>, %[[REDUCE_RHS:.*]]: tile<f32>) {
// CHECK:               %[[ADD:.*]] = addf %[[REDUCE_LHS]], %[[REDUCE_RHS]]  : tile<f32>
// CHECK:               yield %[[ADD]] : tile<f32>
// CHECK:             }
// CHECK:             %[[RESHAPE:.*]] = reshape %[[REDUCE]] : tile<8x64xf32> -> tile<8x1x64xf32>
// CHECK:             %[[PVIEW_2:.*]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>
// CHECK:             %[[OUT_COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_1:.*]] = remi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_1:.*]] = divi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_0:.*]] = remi %[[OUT_TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[STORE_TOKEN:.*]] = store_view_tko weak %[[RESHAPE]], %[[PVIEW_2]]{{\[}}%[[OUT_COORD_0]], %[[OUT_COORD_1]], %[[OUT_COORD_2]]] : tile<8x1x64xf32>, partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>, tile<i32> -> token
// CHECK:             return
// CHECK:           }
// CHECK:         }
  nv_tensor_ir.graph @reduction_op_norm1(%input: tensor<8x32x128xf32>{nv_tensor_ir.stride = "(1,8,256)"}) -> (tensor<8x1x128xf32> {nv_tensor_ir.stride = "(1,1,8)"}) {
    %result = reduce(%input) <
      dimensions = [1],
      reduction_mode = <norm1>> : tensor<8x32x128xf32> -> tensor<8x1x128xf32>
    results %result : tensor<8x1x128xf32>
  }
}

// -----
// Test ReduceOp with norm2 mode (L2 norm)

module {
// CHECK-LABEL:   cuda_tile.module @cuda_tile_module {
// CHECK:           entry @reduction_op_norm2(%[[ARG0:.*]]: tile<ptr<f32>>, %[[ARG1:.*]]: tile<ptr<f32>>) {
// CHECK:             %[[TVIEW:.*]] = make_tensor_view %[[ARG0]], shape = [8, 32, 128], strides = [1, 8, 256] : tensor_view<8x32x128xf32, strides=[1,8,256]>
// CHECK:             %[[TVIEW_0:.*]] = make_tensor_view %[[ARG1]], shape = [8, 1, 128], strides = [1, 1, 8] : tensor_view<8x1x128xf32, strides=[1,1,8]>
// CHECK:             %[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]] = get_tile_block_id : tile<i32>
// CHECK:             %[[PVIEW:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[IDX_SPACE:.*]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>> -> tile<i32>
// CHECK:             %[[PVIEW_1:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[COORD_1:.*]] = remi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[TMP_1:.*]] = divi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[COORD_0:.*]] = remi %[[TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[TILE:.*]], %[[RESULT_TOKEN:.*]] = load_view_tko weak %[[PVIEW_1]]{{\[}}%[[COORD_0]], %[[COORD_1]], %[[COORD_2]]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>, tile<i32> -> tile<8x32x64xf32>, token
// CHECK:             %[[SQUARED:.*]] = mulf %[[TILE]], %[[TILE]]  : tile<8x32x64xf32>
// CHECK:             %[[REDUCE:.*]] = reduce %[[SQUARED]] dim=1 identities=[0.000000e+00 : f32] : tile<8x32x64xf32> -> tile<8x64xf32>
// CHECK:             (%[[REDUCE_LHS:.*]]: tile<f32>, %[[REDUCE_RHS:.*]]: tile<f32>) {
// CHECK:               %[[ADD:.*]] = addf %[[REDUCE_LHS]], %[[REDUCE_RHS]]  : tile<f32>
// CHECK:               yield %[[ADD]] : tile<f32>
// CHECK:             }
// CHECK:             %[[SQRT:.*]] = sqrt %[[REDUCE]]  : tile<8x64xf32>
// CHECK:             %[[RESHAPE:.*]] = reshape %[[SQRT]] : tile<8x64xf32> -> tile<8x1x64xf32>
// CHECK:             %[[PVIEW_2:.*]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>
// CHECK:             %[[OUT_COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_1:.*]] = remi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_1:.*]] = divi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_0:.*]] = remi %[[OUT_TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[STORE_TOKEN:.*]] = store_view_tko weak %[[RESHAPE]], %[[PVIEW_2]]{{\[}}%[[OUT_COORD_0]], %[[OUT_COORD_1]], %[[OUT_COORD_2]]] : tile<8x1x64xf32>, partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>, tile<i32> -> token
// CHECK:             return
// CHECK:           }
// CHECK:         }
  nv_tensor_ir.graph @reduction_op_norm2(%input: tensor<8x32x128xf32>{nv_tensor_ir.stride = "(1,8,256)"}) -> (tensor<8x1x128xf32> {nv_tensor_ir.stride = "(1,1,8)"}) {
    %result = reduce(%input) <
      dimensions = [1],
      reduction_mode = <norm2>> : tensor<8x32x128xf32> -> tensor<8x1x128xf32>
    results %result : tensor<8x1x128xf32>
  }
}

// -----
// Test ReduceOp with mul_no_zeros mode

module {
// CHECK-LABEL:   cuda_tile.module @cuda_tile_module {
// CHECK:           entry @reduction_op_mul_no_zeros(%[[ARG0:.*]]: tile<ptr<f32>>, %[[ARG1:.*]]: tile<ptr<f32>>) {
// CHECK:             %[[TVIEW:.*]] = make_tensor_view %[[ARG0]], shape = [8, 32, 128], strides = [1, 8, 256] : tensor_view<8x32x128xf32, strides=[1,8,256]>
// CHECK:             %[[TVIEW_0:.*]] = make_tensor_view %[[ARG1]], shape = [8, 1, 128], strides = [1, 1, 8] : tensor_view<8x1x128xf32, strides=[1,1,8]>
// CHECK:             %[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]] = get_tile_block_id : tile<i32>
// CHECK:             %[[PVIEW:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[IDX_SPACE:.*]]:3 = get_index_space_shape %[[PVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>> -> tile<i32>
// CHECK:             %[[PVIEW_1:.*]] = make_partition_view %[[TVIEW]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>
// CHECK:             %[[COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[COORD_1:.*]] = remi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[TMP_1:.*]] = divi %[[TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[COORD_0:.*]] = remi %[[TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[TILE:.*]], %[[RESULT_TOKEN:.*]] = load_view_tko weak %[[PVIEW_1]]{{\[}}%[[COORD_0]], %[[COORD_1]], %[[COORD_2]]] : partition_view<tile=(8x32x64), tensor_view<8x32x128xf32, strides=[1,8,256]>>, tile<i32> -> tile<8x32x64xf32>, token
// CHECK:             %[[ZERO:.*]] = constant <f32: 0.000000e+00> : tile<8x32x64xf32>
// CHECK:             %[[ONE:.*]] = constant <f32: 1.000000e+00> : tile<8x32x64xf32>
// CHECK:             %[[IS_ZERO:.*]] = cmpf equal ordered %[[TILE]], %[[ZERO]] : tile<8x32x64xf32> -> tile<8x32x64xi1>
// CHECK:             %[[REPLACED:.*]] = select %[[IS_ZERO]], %[[ONE]], %[[TILE]] : tile<8x32x64xi1>, tile<8x32x64xf32>
// CHECK:             %[[REDUCE:.*]] = reduce %[[REPLACED]] dim=1 identities=[1.000000e+00 : f32] : tile<8x32x64xf32> -> tile<8x64xf32>
// CHECK:             (%[[REDUCE_LHS:.*]]: tile<f32>, %[[REDUCE_RHS:.*]]: tile<f32>) {
// CHECK:               %[[MUL:.*]] = mulf %[[REDUCE_LHS]], %[[REDUCE_RHS]]  : tile<f32>
// CHECK:               yield %[[MUL]] : tile<f32>
// CHECK:             }
// CHECK:             %[[RESHAPE:.*]] = reshape %[[REDUCE]] : tile<8x64xf32> -> tile<8x1x64xf32>
// CHECK:             %[[PVIEW_2:.*]] = make_partition_view %[[TVIEW_0]] : partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>
// CHECK:             %[[OUT_COORD_2:.*]] = remi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_0:.*]] = divi %[[BLOCK_X]], %[[IDX_SPACE]]#2 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_1:.*]] = remi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_TMP_1:.*]] = divi %[[OUT_TMP_0]], %[[IDX_SPACE]]#1 unsigned : tile<i32>
// CHECK:             %[[OUT_COORD_0:.*]] = remi %[[OUT_TMP_1]], %[[IDX_SPACE]]#0 unsigned : tile<i32>
// CHECK:             %[[STORE_TOKEN:.*]] = store_view_tko weak %[[RESHAPE]], %[[PVIEW_2]]{{\[}}%[[OUT_COORD_0]], %[[OUT_COORD_1]], %[[OUT_COORD_2]]] : tile<8x1x64xf32>, partition_view<tile=(8x1x64), tensor_view<8x1x128xf32, strides=[1,1,8]>>, tile<i32> -> token
// CHECK:             return
// CHECK:           }
// CHECK:         }
  nv_tensor_ir.graph @reduction_op_mul_no_zeros(%input: tensor<8x32x128xf32>{nv_tensor_ir.stride = "(1,8,256)"}) -> (tensor<8x1x128xf32> {nv_tensor_ir.stride = "(1,1,8)"}) {
    %result = reduce(%input) <
      dimensions = [1],
      reduction_mode = <mul_no_zeros>> : tensor<8x32x128xf32> -> tensor<8x1x128xf32>
    results %result : tensor<8x1x128xf32>
  }
}
