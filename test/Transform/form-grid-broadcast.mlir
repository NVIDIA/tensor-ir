// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --split-input-file | FileCheck %s

// Load the varying dimension once, preserving leading and trailing unit dimensions with stride 2 before broadcasting the tile.
// CHECK-LABEL: func.func @leading_and_trailing(
//  CHECK-SAME:   %[[INPUT:[^:]+]]: memref<6xf32>
//  CHECK-SAME:   %[[OUTPUT:[^:]+]]: memref<4x6x8xf32, strided<[128, 16, 1]>>
func.func @leading_and_trailing(%input: memref<6xf32>,
    %output: memref<4x6x8xf32, strided<[128, 16, 1]>>) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 3, 4>
} {
  %input_tensor = bufferization.to_tensor %input restrict
      : memref<6xf32> to tensor<6xf32>
  %expanded = nv_tensor_ir.reshape %input_tensor
      : tensor<6xf32> -> tensor<1x6x1xf32>
  //       CHECK:   scf.forall (%[[BID:.*]]) in (8) {
  //       CHECK:     %[[TWO:.*]] = arith.constant 2 : index
  //       CHECK:     %[[ROW:.*]] = arith.remui %[[BID]], %[[TWO]] : index
  //       CHECK:     %[[REST:.*]] = arith.divui %[[BID]], %[[TWO]] : index
  //       CHECK:     %[[TWO_AGAIN:.*]] = arith.constant 2 : index
  //       CHECK:     %[[COLUMN:.*]] = arith.remui %[[REST]], %[[TWO_AGAIN]] : index
  //       CHECK:     %[[DEPTH:.*]] = arith.divui %[[REST]], %[[TWO_AGAIN]] : index
  //       CHECK:     %[[VIEW:.*]] = memref.reinterpret_cast %[[INPUT]] to offset: [0], sizes: [1, 6, 1], strides: [2, 1, 2]
  //       CHECK:     %[[LOAD:.*]] = nv_tensor_ir.load %[[VIEW]] tile_indices [0, %[[COLUMN]], 0]
  //  CHECK-SAME:      : memref<1x6x1xf32, strided<[2, 1, 2]>> to tensor<1x3x1xf32>
  //       CHECK:     %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[LOAD]] : tensor<1x3x1xf32> -> tensor<2x3x4xf32>
  %result = nv_tensor_ir.broadcast %expanded
      : tensor<1x6x1xf32> -> tensor<4x6x8xf32>
  //       CHECK:     nv_tensor_ir.store %[[BROADCAST]], %[[OUTPUT]] tile_indices [%[[ROW]], %[[COLUMN]], %[[DEPTH]]]
  bufferization.materialize_in_destination %result in restrict writable %output
      : (tensor<4x6x8xf32>, memref<4x6x8xf32, strided<[128, 16, 1]>>) -> ()
  return
}

// -----

// Unit broadcast dimensions use stride 2 while varying dimensions preserve their dynamic extents and strides.
// CHECK-LABEL: func.func @dynamic_broadcast(
//  CHECK-SAME:   %[[INPUT:[^:]+]]: memref<?xf32, strided<[?]>>
//  CHECK-SAME:   %[[OUTPUT:[^:]+]]: memref<4x?xf32, strided<[?, 1]>>
func.func @dynamic_broadcast(%input: memref<?xf32, strided<[?]>>,
    %output: memref<4x?xf32, strided<[?, 1]>>) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 8>
} {
  %input_tensor = bufferization.to_tensor %input restrict
      : memref<?xf32, strided<[?]>> to tensor<?xf32>
  %expanded = nv_tensor_ir.reshape %input_tensor
      : tensor<?xf32> -> tensor<1x?xf32>
  //       CHECK:   scf.forall (%[[BID:.*]]) in (%{{.*}}) {
  //       CHECK:     %[[TWO:.*]] = arith.constant 2 : index
  //       CHECK:     %[[ROW:.*]] = arith.remui %[[BID]], %[[TWO]] : index
  //       CHECK:     %[[COLUMN:.*]] = arith.divui %[[BID]], %[[TWO]] : index
  //       CHECK:     %[[VIEW:.*]] = memref.reinterpret_cast %[[INPUT]] to offset: [0], sizes: [1, %{{.*}}], strides: [2, %{{.*}}]
  //       CHECK:     %[[LOAD:.*]] = nv_tensor_ir.load %[[VIEW]] tile_indices [0, %[[COLUMN]]]
  //  CHECK-SAME:      : memref<1x?xf32, strided<[2, ?]>> to tensor<1x8xf32>
  //       CHECK:     %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[LOAD]] : tensor<1x8xf32> -> tensor<2x8xf32>
  %result = nv_tensor_ir.broadcast %expanded
      : tensor<1x?xf32> -> tensor<4x?xf32>
  //       CHECK:     nv_tensor_ir.store %[[BROADCAST]], %[[OUTPUT]] tile_indices [%[[ROW]], %[[COLUMN]]]
  bufferization.materialize_in_destination %result in restrict writable %output
      : (tensor<4x?xf32>, memref<4x?xf32, strided<[?, 1]>>) -> ()
  return
}
