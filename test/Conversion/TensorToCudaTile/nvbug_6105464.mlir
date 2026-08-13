// RUN: tensor_ir-opt --layout-propagation-pipeline -split-input-file %s | FileCheck %s

// Regression test: a graph taking a scalar (non-tensor) argument that is
// splatted against a tensor input. This exercises:
//   * `setDefaultStrides` with an empty tensor descriptor (scalar argument)
//   * `SplatOpConversion` with a splat operand that is a block argument
//     (no defining op), which must be reshape+broadcast, not a constant.

// CHECK-LABEL: entry @graph_jpzyTXor4mF8Q1ND
// CHECK-SAME: (%[[INPTR:.*]]: tile<ptr<f16>>, %[[SCALAR:.*]]: tile<f32>, %[[OUTPTR:.*]]: tile<ptr<f16>>)
// CHECK: %[[TVIEW_IN:.*]] = make_tensor_view %{{.*}}, shape = [64], strides = [1] : tensor_view<64xf16, strides=[1]>
// CHECK: %[[PVIEW_IN:.*]] = make_partition_view %[[TVIEW_IN]]
// CHECK: %[[TILE_IN:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_IN]]
// CHECK: %[[CONV_UP:.*]] = ftof %[[TILE_IN]]  : tile<64xf16> -> tile<64xf32>
// CHECK: %[[RESHAPE:.*]] = reshape %[[SCALAR]] : tile<f32> -> tile<1xf32>
// CHECK: %[[BCAST:.*]] = broadcast %[[RESHAPE]] : tile<1xf32> -> tile<64xf32>
// CHECK: %[[ADD:.*]] = addf %[[CONV_UP]], %[[BCAST]]  : tile<64xf32>
// CHECK: %[[CONV_DOWN:.*]] = ftof %[[ADD]]  : tile<64xf32> -> tile<64xf16>
// CHECK: %[[TVIEW_OUT:.*]] = make_tensor_view %{{.*}}, shape = [64], strides = [1] : tensor_view<64xf16, strides=[1]>
// CHECK: %[[PVIEW_OUT:.*]] = make_partition_view %[[TVIEW_OUT]]
// CHECK: store_view_tko weak %[[CONV_DOWN]], %[[PVIEW_OUT]]

module {
  nv_tensor_ir.graph @graph_jpzyTXor4mF8Q1ND(
      %in: tensor<4x4x4xf16> {nv_tensor_ir.alignment = 16 : i64},
      %in_0: f32)
      -> (tensor<4x4x4xf16> {nv_tensor_ir.alignment = 16 : i64}) {
    %convert = convert %in : tensor<4x4x4xf16> -> tensor<4x4x4xf32>
    %splat = splat %in_0 : tensor<4x4x4xf32>
    %add = add %convert, %splat : tensor<4x4x4xf32>
    %convert_1 = convert %add : tensor<4x4x4xf32> -> tensor<4x4x4xf16>
    results %convert_1 : tensor<4x4x4xf16>
  }
}

// -----

// Regression test: a graph with two f8 tensor inputs and two f32 scalar
// arguments. The first scalar is splatted twice (once against each tensor),
// which exercises CSE of the reshape+broadcast lowering of `splat` so the
// shared scalar produces a single broadcast that feeds both multiplies.
// Also exercises the swish/sigmoid expansion and a second independent splat
// (using the second scalar) at the end of the graph.

// CHECK-LABEL: entry @graph_SfjksTBQx5ehL6Cl
// CHECK-SAME: (%[[INPTR0:.*]]: tile<ptr<f8E4M3FN>>, %[[SCALAR0:.*]]: tile<f32>, %[[INPTR1:.*]]: tile<ptr<f8E4M3FN>>, %[[SCALAR1:.*]]: tile<f32>, %[[OUTPTR:.*]]: tile<ptr<f8E4M3FN>>)
// CHECK: %[[ONE:.*]] = constant <f32: 1.000000e+00> : tile<64xf32>
// CHECK: %[[TVIEW_IN0:.*]] = make_tensor_view %{{.*}}, shape = [4096], strides = [1] : tensor_view<4096xf8E4M3FN, strides=[1]>
// CHECK: %[[PVIEW_IN0:.*]] = make_partition_view %[[TVIEW_IN0]]
// CHECK: %[[TILE_IN0:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_IN0]]
// CHECK: %[[TVIEW_IN1:.*]] = make_tensor_view %{{.*}}, shape = [4096], strides = [1] : tensor_view<4096xf8E4M3FN, strides=[1]>
// CHECK: %[[PVIEW_IN1:.*]] = make_partition_view %[[TVIEW_IN1]]
// CHECK: %[[TILE_IN1:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_IN1]]
// CHECK: %[[CONV0_UP:.*]] = ftof %[[TILE_IN0]]  : tile<64xf8E4M3FN> -> tile<64xf32>
// CHECK: %[[RESHAPE0:.*]] = reshape %[[SCALAR0]] : tile<f32> -> tile<1xf32>
// CHECK: %[[BCAST0:.*]] = broadcast %[[RESHAPE0]] : tile<1xf32> -> tile<64xf32>
// CHECK: %[[MUL0:.*]] = mulf %[[CONV0_UP]], %[[BCAST0]]  : tile<64xf32>
// CHECK: %[[CONV1_UP:.*]] = ftof %[[TILE_IN1]]  : tile<64xf8E4M3FN> -> tile<64xf32>
// CHECK: %[[MUL1:.*]] = mulf %[[CONV1_UP]], %[[BCAST0]]  : tile<64xf32>
// CHECK: %[[NEG:.*]] = negf %[[MUL1]] : tile<64xf32>
// CHECK: %[[EXP:.*]] = exp %[[NEG]]  : tile<64xf32>
// CHECK: %[[ONEPLUS:.*]] = addf %[[ONE]], %[[EXP]]  : tile<64xf32>
// CHECK: %[[SIGMOID:.*]] = divf %[[ONE]], %[[ONEPLUS]]  : tile<64xf32>
// CHECK: %[[SWISH:.*]] = mulf %[[SIGMOID]], %[[MUL1]]  : tile<64xf32>
// CHECK: %[[PROD:.*]] = mulf %[[MUL0]], %[[SWISH]]  : tile<64xf32>
// CHECK: %[[RESHAPE1:.*]] = reshape %[[SCALAR1]] : tile<f32> -> tile<1xf32>
// CHECK: %[[BCAST1:.*]] = broadcast %[[RESHAPE1]] : tile<1xf32> -> tile<64xf32>
// CHECK: %[[SCALED:.*]] = mulf %[[PROD]], %[[BCAST1]]  : tile<64xf32>
// CHECK: %[[CONV_DOWN:.*]] = ftof %[[SCALED]]  : tile<64xf32> -> tile<64xf8E4M3FN>
// CHECK: %[[TVIEW_OUT:.*]] = make_tensor_view %{{.*}}, shape = [4096], strides = [1] : tensor_view<4096xf8E4M3FN, strides=[1]>
// CHECK: %[[PVIEW_OUT:.*]] = make_partition_view %[[TVIEW_OUT]]
// CHECK: store_view_tko weak %[[CONV_DOWN]], %[[PVIEW_OUT]]

module {
  nv_tensor_ir.graph @graph_SfjksTBQx5ehL6Cl(
      %in: tensor<1x64x64xf8E4M3FN> {
        nv_tensor_ir.alignment = 16
      },
      %in_0: f32,
      %in_1: tensor<1x64x64xf8E4M3FN> {
        nv_tensor_ir.alignment = 16 : i64
      }, %in_2: f32)
      -> (tensor<1x64x64xf8E4M3FN> {nv_tensor_ir.alignment = 16 : i64}) {
    %convert = convert %in : tensor<1x64x64xf8E4M3FN> -> tensor<1x64x64xf32>
    %splat = splat %in_0 : tensor<1x64x64xf32>
    %mul = mul %convert, %splat : tensor<1x64x64xf32>
    %convert_3 = convert %in_1 : tensor<1x64x64xf8E4M3FN> -> tensor<1x64x64xf32>
    %splat_4 = splat %in_0 : tensor<1x64x64xf32>
    %mul_5 = mul %convert_3, %splat_4 : tensor<1x64x64xf32>
    %sigmoid_fwd = sigmoid_fwd %mul_5 : tensor<1x64x64xf32>
    %mul_6 = mul %sigmoid_fwd, %mul_5 : tensor<1x64x64xf32>
    %mul_7 = mul %mul, %mul_6 : tensor<1x64x64xf32>
    %splat_8 = splat %in_2 : tensor<1x64x64xf32>
    %mul_9 = mul %mul_7, %splat_8 : tensor<1x64x64xf32>
    %convert_10 = convert %mul_9 : tensor<1x64x64xf32> -> tensor<1x64x64xf8E4M3FN>
    results %convert_10 : tensor<1x64x64xf8E4M3FN>
  }
}
