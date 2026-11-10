// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid))' --split-input-file | FileCheck %s

// FormGrid decodes the linear CTA index back to front, so the last output
// dimension varies fastest. Operand layouts, broadcasts, dynamic strides, the
// number of outputs, and the op kind do not change the order.

// A rank-3 grid decodes dimension 2, then 1, then 0.
// CHECK-LABEL: func.func @outer_transpose_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (1024)
//       CHECK: %[[C4:.*]] = arith.constant 4 : index
//       CHECK: %[[D2:.*]] = arith.remui %[[LINEAR]], %[[C4]]
//       CHECK: %[[Q2:.*]] = arith.divui %[[LINEAR]], %[[C4]]
//       CHECK: %[[C8:.*]] = arith.constant 8 : index
//       CHECK: %[[D1:.*]] = arith.remui %[[Q2]], %[[C8]]
//       CHECK: %[[D0:.*]] = arith.divui %[[Q2]], %[[C8]]
//       CHECK: nv_tensor_ir.load {{.*}}[%[[D0]], %[[D1]], %[[D2]]]
nv_tensor_ir.graph @outer_transpose(%input: tensor<8x32x1024xf32>)
    -> tensor<32x8x1024xf32>
    attributes {tile_size = array<i32: 1, 1, 256>} {
  %result = transpose %input permutation = [1, 0, 2]
      : tensor<8x32x1024xf32> -> tensor<32x8x1024xf32>
  results %result : tensor<32x8x1024xf32>
}

// -----

// Dimensions with one tile use index 0, and the first tiled dimension takes
// the remaining linear index without another division.
// CHECK-LABEL: func.func @unit_tile_count_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (64)
//       CHECK: %[[C8:.*]] = arith.constant 8 : index
//       CHECK: %[[D3:.*]] = arith.remui %[[LINEAR]], %[[C8]]
//       CHECK: %[[D1:.*]] = arith.divui %[[LINEAR]], %[[C8]]
//   CHECK-NOT: arith.remui
//   CHECK-NOT: arith.divui
//       CHECK: nv_tensor_ir.load {{.*}}[0, %[[D1]], 0, %[[D3]]]
nv_tensor_ir.graph @unit_tile_count(%input: tensor<4x8x64x128xf32>)
    -> tensor<4x64x8x128xf32>
    attributes {tile_size = array<i32: 4, 8, 8, 16>} {
  %result = transpose %input permutation = [0, 2, 1, 3]
      : tensor<4x8x64x128xf32> -> tensor<4x64x8x128xf32>
  results %result : tensor<4x64x8x128xf32>
}

// -----

// A transpose that moves the fastest input dimension keeps the output order.
// CHECK-LABEL: func.func @minor_transpose_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (32768)
//       CHECK: %[[C32:.*]] = arith.constant 32 : index
//       CHECK: %[[D1:.*]] = arith.remui %[[LINEAR]], %[[C32]]
//       CHECK: %[[D0:.*]] = arith.divui %[[LINEAR]], %[[C32]]
//       CHECK: nv_tensor_ir.load {{.*}}[%[[D0]], %[[D1]]]
nv_tensor_ir.graph @minor_transpose(%input: tensor<2048x32768xf32>)
    -> tensor<32768x2048xf32>
    attributes {tile_size = array<i32: 32, 64>} {
  %result = transpose %input permutation = [1, 0]
      : tensor<2048x32768xf32> -> tensor<32768x2048xf32>
  results %result : tensor<32768x2048xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_input_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (32768)
//       CHECK: %[[C32:.*]] = arith.constant 32 : index
//       CHECK: %[[D1:.*]] = arith.remui %[[LINEAR]], %[[C32]]
//       CHECK: %[[D0:.*]] = arith.divui %[[LINEAR]], %[[C32]]
//       CHECK: nv_tensor_ir.store {{.*}}[%[[D0]], %[[D1]]]
nv_tensor_ir.graph @broadcast_input(
    %input: tensor<32768x1xf32> {nv_tensor_ir.stride = "(1,1)"})
    -> tensor<32768x2048xf32>
    attributes {tile_size = array<i32: 32, 64>} {
  %result = broadcast %input
      : tensor<32768x1xf32> -> tensor<32768x2048xf32>
  results %result : tensor<32768x2048xf32>
}

// -----

// CHECK-LABEL: func.func @disagreeing_inputs_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (512)
//       CHECK: %[[C16:.*]] = arith.constant 16 : index
//       CHECK: %[[D2:.*]] = arith.remui %[[LINEAR]], %[[C16]]
//       CHECK: %[[Q2:.*]] = arith.divui %[[LINEAR]], %[[C16]]
//       CHECK: %[[C8:.*]] = arith.constant 8 : index
//       CHECK: %[[D1:.*]] = arith.remui %[[Q2]], %[[C8]]
//       CHECK: %[[D0:.*]] = arith.divui %[[Q2]], %[[C8]]
//       CHECK: nv_tensor_ir.store {{.*}}[%[[D0]], %[[D1]], %[[D2]]]
nv_tensor_ir.graph @disagreeing_inputs(
    %lhs: tensor<16x32x64xf32> {nv_tensor_ir.stride = "(1,1024,16)"},
    %rhs: tensor<16x32x64xf32> {nv_tensor_ir.stride = "(32,1,512)"})
    -> tensor<16x32x64xf32>
    attributes {tile_size = array<i32: 4, 4, 4>} {
  %result = add %lhs, %rhs : tensor<16x32x64xf32>
  results %result : tensor<16x32x64xf32>
}

// -----

// CHECK-LABEL: func.func @dynamic_output_stride_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (64)
//       CHECK: %[[C8:.*]] = arith.constant 8 : index
//       CHECK: %[[D1:.*]] = arith.remui %[[LINEAR]], %[[C8]]
//       CHECK: %[[D0:.*]] = arith.divui %[[LINEAR]], %[[C8]]
//       CHECK: nv_tensor_ir.store {{.*}}[%[[D0]], %[[D1]]]
nv_tensor_ir.graph @dynamic_output_stride(%input: tensor<64x128xf32>)
    -> (tensor<64x128xf32> {nv_tensor_ir.stride = "(?,1)"})
    attributes {tile_size = array<i32: 8, 16>} {
  %result = cos %input : tensor<64x128xf32>
  results %result : tensor<64x128xf32>
}

// -----

// CHECK-LABEL: func.func @multi_output_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (64)
//       CHECK: %[[C8:.*]] = arith.constant 8 : index
//       CHECK: %[[D1:.*]] = arith.remui %[[LINEAR]], %[[C8]]
//       CHECK: %[[D0:.*]] = arith.divui %[[LINEAR]], %[[C8]]
//       CHECK: nv_tensor_ir.store {{.*}}[%[[D0]], %[[D1]]]
//       CHECK: nv_tensor_ir.store {{.*}}[%[[D0]], %[[D1]]]
nv_tensor_ir.graph @multi_output(%input: tensor<64x128xf32>)
    -> (tensor<64x128xf32> {nv_tensor_ir.stride = "(256,1)"},
        tensor<64x128xf32> {nv_tensor_ir.stride = "(256,1)"})
    attributes {tile_size = array<i32: 8, 16>} {
  %cos = cos %input : tensor<64x128xf32>
  %sin = sin %input : tensor<64x128xf32>
  results %cos, %sin : tensor<64x128xf32>, tensor<64x128xf32>
}

// -----

// CHECK-LABEL: func.func @dynamic_stride_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (64)
//       CHECK: %[[C8:.*]] = arith.constant 8 : index
//       CHECK: %[[D1:.*]] = arith.remui %[[LINEAR]], %[[C8]]
//       CHECK: %[[D0:.*]] = arith.divui %[[LINEAR]], %[[C8]]
//       CHECK: nv_tensor_ir.store {{.*}}[%[[D0]], %[[D1]]]
nv_tensor_ir.graph @dynamic_stride(
    %input: tensor<64x128xf32> {nv_tensor_ir.stride = "(1,?)"})
    -> tensor<64x128xf32>
    attributes {tile_size = array<i32: 8, 16>} {
  %result = cos %input : tensor<64x128xf32>
  results %result : tensor<64x128xf32>
}

// -----

// CHECK-LABEL: func.func @permuted_three_dim_input_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (512)
//       CHECK: %[[C16:.*]] = arith.constant 16 : index
//       CHECK: %[[D2:.*]] = arith.remui %[[LINEAR]], %[[C16]]
//       CHECK: %[[Q2:.*]] = arith.divui %[[LINEAR]], %[[C16]]
//       CHECK: %[[C8:.*]] = arith.constant 8 : index
//       CHECK: %[[D1:.*]] = arith.remui %[[Q2]], %[[C8]]
//       CHECK: %[[D0:.*]] = arith.divui %[[Q2]], %[[C8]]
//       CHECK: nv_tensor_ir.load {{.*}}[%[[D0]], %[[D1]], %[[D2]]]
nv_tensor_ir.graph @permuted_three_dim_input(
    %input: tensor<16x32x64xf32> {nv_tensor_ir.stride = "(2048,1,32)"})
    -> tensor<16x32x64xf32>
    attributes {tile_size = array<i32: 4, 4, 4>} {
  %result = cos %input : tensor<16x32x64xf32>
  results %result : tensor<16x32x64xf32>
}

// -----

// CHECK-LABEL: func.func @mixed_inputs_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (64)
//       CHECK: %[[C8:.*]] = arith.constant 8 : index
//       CHECK: %[[D1:.*]] = arith.remui %[[LINEAR]], %[[C8]]
//       CHECK: %[[D0:.*]] = arith.divui %[[LINEAR]], %[[C8]]
//       CHECK: nv_tensor_ir.load {{.*}}[%[[D0]], %[[D1]]]
//       CHECK: nv_tensor_ir.load {{.*}}[%[[D0]], %[[D1]]]
nv_tensor_ir.graph @mixed_inputs(
    %lhs: tensor<256x256xf32> {nv_tensor_ir.stride = "(1,256)"},
    %rhs: tensor<256x256xf32>)
    -> tensor<256x256xf32>
    attributes {tile_size = array<i32: 32, 32>} {
  %result = add %lhs, %rhs : tensor<256x256xf32>
  results %result : tensor<256x256xf32>
}

// -----

// Matmul decodes N fastest.
// CHECK-LABEL: func.func @matmul_operands_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (64)
//       CHECK: %[[C8:.*]] = arith.constant 8 : index
//       CHECK: %[[N:.*]] = arith.remui %[[LINEAR]], %[[C8]]
//       CHECK: %[[M:.*]] = arith.divui %[[LINEAR]], %[[C8]]
//       CHECK: nv_tensor_ir.load {{.*}}[%[[M]], 0]
//       CHECK: nv_tensor_ir.load {{.*}}[0, %[[N]]]
//       CHECK: nv_tensor_ir.store {{.*}}[%[[M]], %[[N]]]
nv_tensor_ir.graph @matmul_operands(
    %lhs: tensor<256x64xf32> {nv_tensor_ir.stride = "(1,256)"},
    %rhs: tensor<64x256xf32>)
    -> tensor<256x256xf32>
    attributes {tile_size = array<i32: 32, 32>} {
  %result = matmul(%lhs, %rhs)
      : (tensor<256x64xf32>, tensor<64x256xf32>) -> tensor<256x256xf32>
  results %result : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func.func @reduction_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (64)
//       CHECK: %[[C8:.*]] = arith.constant 8 : index
//       CHECK: %[[D1:.*]] = arith.remui %[[LINEAR]], %[[C8]]
//       CHECK: %[[D0:.*]] = arith.divui %[[LINEAR]], %[[C8]]
//       CHECK: nv_tensor_ir.store {{.*}}[%[[D0]], %[[D1]]]
nv_tensor_ir.graph @reduction(%input: tensor<64x128xf32>)
    -> tensor<64x128xf32>
    attributes {tile_size = array<i32: 8, 16>} {
  %sum = reduce(%input) <dimensions = [1], reduction_mode = <add>>
      : tensor<64x128xf32> -> tensor<64x1xf32>
  %wide = broadcast %sum : tensor<64x1xf32> -> tensor<64x128xf32>
  %result = add %input, %wide : tensor<64x128xf32>
  results %result : tensor<64x128xf32>
}
