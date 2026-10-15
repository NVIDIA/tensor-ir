// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --mlir-print-debuginfo --split-input-file | FileCheck %s

// Sizes and strides are permuted together with the transposed input view. The
// output store retains its source location.
// CHECK-LABEL: func.func @descriptor(
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<?x?xf32, strided<[?, 1], offset: 3>, #ptr.generic_space>
//  CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<?x?xf32, strided<[?, 1], offset: 7>, #ptr.generic_space>
//       CHECK: %{{[^,]*}}, %{{[^,]*}}, %[[IN_SIZES:[^:]+]]:2, %[[IN_STRIDES:[^:]+]]:2 = memref.extract_strided_metadata %[[INPUT]]
//       CHECK: %[[IN_PTR:[^ ]+]] = ptr.to_ptr %[[INPUT]]
//       CHECK: %{{[^,]*}}, %{{[^,]*}}, %[[OUT_SIZES:[^:]+]]:2, %[[OUT_STRIDES:[^:]+]]:2 = memref.extract_strided_metadata %[[OUTPUT]]
//       CHECK: %[[OUT_PTR:[^ ]+]] = ptr.to_ptr %[[OUTPUT]]
//       CHECK: arith.ceildivui %[[OUT_SIZES]]#0,
//       CHECK: arith.ceildivui %[[OUT_SIZES]]#1,
//       CHECK: scf.forall
//       CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[IN_PTR]]
//  CHECK-SAME: view offset: [3], sizes: [%[[IN_SIZES]]#1, %[[IN_SIZES]]#0], strides: [1, %[[IN_STRIDES]]#0], alignment: 16
//       CHECK: nv_tensor_ir.store %[[TILE]], %[[OUT_PTR]]
//  CHECK-SAME: view offset: [7], sizes: [%[[OUT_SIZES]]#0, %[[OUT_SIZES]]#1], strides: [%[[OUT_STRIDES]]#0, 1], alignment: 16
//  CHECK-SAME: loc([[OUTPUT_LOC:#[a-zA-Z0-9]+]])
//       CHECK: [[OUTPUT_LOC]] = loc("output")
func.func @descriptor(
    %input: memref<?x?xf32, strided<[?, 1], offset: 3>, #ptr.generic_space>
        {nv_tensor_ir.alignment = 16 : i64},
    %output: memref<?x?xf32, strided<[?, 1], offset: 7>, #ptr.generic_space>
        {nv_tensor_ir.alignment = 16 : i64}) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 4, 8>
} {
  %load = bufferization.to_tensor %input restrict
      : memref<?x?xf32, strided<[?, 1], offset: 3>, #ptr.generic_space>
        to tensor<?x?xf32>
  %transposed = nv_tensor_ir.transpose %load permutation = [1, 0]
      : tensor<?x?xf32> -> tensor<?x?xf32>
  bufferization.materialize_in_destination %transposed in restrict writable
      %output : (tensor<?x?xf32>,
                 memref<?x?xf32, strided<[?, 1], offset: 7>, #ptr.generic_space>)
                -> ()
      loc("output")
  return
}

// -----

// Dynamic physical offsets remain SSA values when logical offsets are added.
// CHECK-LABEL: func.func @dynamic_offsets(
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<21xf32, strided<[1], offset: ?>, #ptr.generic_space>
//  CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<16xf32, strided<[1], offset: ?>, #ptr.generic_space>
//       CHECK: %{{[^,]*}}, %[[IN_OFFSET:[^,]*]], %{{[^,]*}}, %{{[^ ]*}} = memref.extract_strided_metadata %[[INPUT]]
//       CHECK: %[[IN_PTR:[^ ]+]] = ptr.to_ptr %[[INPUT]]
//       CHECK: %{{[^,]*}}, %[[OUT_OFFSET:[^,]*]], %{{[^,]*}}, %{{[^ ]*}} = memref.extract_strided_metadata %[[OUTPUT]]
//       CHECK: %[[OUT_PTR:[^ ]+]] = ptr.to_ptr %[[OUTPUT]]
//       CHECK: scf.forall (%[[I:.*]]) in (4)
//       CHECK: %[[FIVE:.*]] = arith.constant 5 : index
//       CHECK: %[[LOAD_OFFSET:.*]] = arith.addi %[[IN_OFFSET]], %[[FIVE]]
//       CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[IN_PTR]][%[[I]]]
//  CHECK-SAME: view offset: [%[[LOAD_OFFSET]]], sizes: [16], strides: [1], alignment: 4
//   CHECK-NOT: arith.addi
//       CHECK: nv_tensor_ir.store %[[TILE]], %[[OUT_PTR]][%[[I]]]
//  CHECK-SAME: view offset: [%[[OUT_OFFSET]]], sizes: [16], strides: [1], alignment: 16
func.func @dynamic_offsets(
    %input: memref<21xf32, strided<[1], offset: ?>, #ptr.generic_space>
        {nv_tensor_ir.alignment = 16 : i64},
    %output: memref<16xf32, strided<[1], offset: ?>, #ptr.generic_space>
        {nv_tensor_ir.alignment = 16 : i64}) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 4>
} {
  %load = bufferization.to_tensor %input restrict
      : memref<21xf32, strided<[1], offset: ?>, #ptr.generic_space>
        to tensor<21xf32>
  %slice = nv_tensor_ir.slice %load starts = [5] limits = [21] strides = [1]
      : tensor<21xf32> -> tensor<16xf32>
  bufferization.materialize_in_destination %slice in restrict writable %output
      : (tensor<16xf32>,
         memref<16xf32, strided<[1], offset: ?>, #ptr.generic_space>) -> ()
  return
}

// -----

// A zero logical offset preserves each dynamic physical offset directly.
// CHECK-LABEL: func.func @unchanged_dynamic_offsets(
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<16xf32, strided<[1], offset: ?>, #ptr.generic_space>
//  CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<16xf32, strided<[1], offset: ?>, #ptr.generic_space>
//       CHECK: %{{[^,]*}}, %[[IN_OFFSET:[^,]*]], %{{[^,]*}}, %{{[^ ]*}} = memref.extract_strided_metadata %[[INPUT]]
//       CHECK: %[[IN_PTR:[^ ]+]] = ptr.to_ptr %[[INPUT]]
//       CHECK: %{{[^,]*}}, %[[OUT_OFFSET:[^,]*]], %{{[^,]*}}, %{{[^ ]*}} = memref.extract_strided_metadata %[[OUTPUT]]
//       CHECK: %[[OUT_PTR:[^ ]+]] = ptr.to_ptr %[[OUTPUT]]
//       CHECK: scf.forall
//   CHECK-NOT: arith.addi
//       CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[IN_PTR]]
//  CHECK-SAME: view offset: [%[[IN_OFFSET]]], sizes: [16], strides: [1], alignment: 16
//   CHECK-NOT: arith.addi
//       CHECK: nv_tensor_ir.store %[[TILE]], %[[OUT_PTR]]
//  CHECK-SAME: view offset: [%[[OUT_OFFSET]]], sizes: [16], strides: [1], alignment: 16
func.func @unchanged_dynamic_offsets(
    %input: memref<16xf32, strided<[1], offset: ?>, #ptr.generic_space>
        {nv_tensor_ir.alignment = 16 : i64},
    %output: memref<16xf32, strided<[1], offset: ?>, #ptr.generic_space>
        {nv_tensor_ir.alignment = 16 : i64}) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 4>
} {
  %load = bufferization.to_tensor %input restrict
      : memref<16xf32, strided<[1], offset: ?>, #ptr.generic_space>
        to tensor<16xf32>
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<16xf32>,
         memref<16xf32, strided<[1], offset: ?>, #ptr.generic_space>) -> ()
  return
}
