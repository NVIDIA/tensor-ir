// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --mlir-print-debuginfo --split-input-file | FileCheck %s

// Sizes and strides are permuted together with the transposed input view. The
// output store retains its source location.
// CHECK-LABEL: func.func @descriptor(
// CHECK-SAME: %[[INPUT:[^:]+]]: memref<?x?xf32, strided<[?, 1]>>
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<?x?xf32, strided<[?, 1]>>
// CHECK: %{{[^,]*}}, %{{[^,]*}}, %{{[^:]+}}:2, %[[STRIDES:[^:]+]]:2 = memref.extract_strided_metadata %[[INPUT]]
// CHECK: %[[M:.*]] = memref.dim %[[INPUT]]
// CHECK: %[[N:.*]] = memref.dim %[[INPUT]]
// CHECK: %[[OM:.*]] = memref.dim %[[OUTPUT]]
// CHECK: %[[ON:.*]] = memref.dim %[[OUTPUT]]
// CHECK: arith.ceildivui %[[OM]]
// CHECK: arith.ceildivui %[[ON]]
// CHECK: scf.forall
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[INPUT]] to offset: [0], sizes: [%[[N]], %[[M]]], strides: [1, %[[STRIDES]]#0]
// CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[VIEW]]
// CHECK-SAME: , alignment 16
// CHECK: nv_tensor_ir.store %[[TILE]], %[[OUTPUT]]
// CHECK-SAME: , alignment 16
// CHECK-SAME: loc([[OUTPUT_LOC:#[a-zA-Z0-9]+]])
// CHECK: [[OUTPUT_LOC]] = loc("output")

func.func @descriptor(
    %input: memref<?x?xf32, strided<[?, 1]>>
        {nv_tensor_ir.alignment = 16 : i64},
    %output: memref<?x?xf32, strided<[?, 1]>>
        {nv_tensor_ir.alignment = 16 : i64}) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 4, 8>
} {
  %load = bufferization.to_tensor %input restrict
      : memref<?x?xf32, strided<[?, 1]>>
        to tensor<?x?xf32>
  %transposed = nv_tensor_ir.transpose %load permutation = [1, 0]
      : tensor<?x?xf32> -> tensor<?x?xf32>
  bufferization.materialize_in_destination %transposed in restrict writable
      %output : (tensor<?x?xf32>,
                 memref<?x?xf32, strided<[?, 1]>>)
                -> ()
      loc("output")
  return
}

// -----

// Logical slice offsets are relative to zero-offset boundary buffers.
// CHECK-LABEL: func.func @logical_offset(
// CHECK-SAME: %[[INPUT:[^:]+]]: memref<21xf32>
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<16xf32>
// CHECK-NOT: memref.extract_strided_metadata
// CHECK: scf.forall (%[[I:.*]]) in (4)
// CHECK-NOT: arith.addi
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[INPUT]] to offset: [5], sizes: [16], strides: [1]
// CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[VIEW]] tile_indices [%[[I]]], alignment 4
// CHECK-NOT: arith.addi
// CHECK: nv_tensor_ir.store %[[TILE]], %[[OUTPUT]] tile_indices [%[[I]]], alignment 16

func.func @logical_offset(
    %input: memref<21xf32>
        {nv_tensor_ir.alignment = 16 : i64},
    %output: memref<16xf32>
        {nv_tensor_ir.alignment = 16 : i64}) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 4>
} {
  %load = bufferization.to_tensor %input restrict
      : memref<21xf32>
        to tensor<21xf32>
  %slice = nv_tensor_ir.slice %load starts = [5] limits = [21] strides = [1]
      : tensor<21xf32> -> tensor<16xf32>
  bufferization.materialize_in_destination %slice in restrict writable %output
      : (tensor<16xf32>,
         memref<16xf32>) -> ()
  return
}

// -----

// A zero logical offset reuses the boundary buffers directly.
// CHECK-LABEL: func.func @unchanged_views(
// CHECK-SAME: %[[INPUT:[^:]+]]: memref<16xf32>
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<16xf32>
// CHECK-NOT: memref.extract_strided_metadata
// CHECK-NOT: memref.reinterpret_cast
// CHECK: scf.forall
// CHECK-NOT: arith.addi
// CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[INPUT]]
// CHECK-SAME: , alignment 16
// CHECK: nv_tensor_ir.store %[[TILE]], %[[OUTPUT]]
// CHECK-SAME: , alignment 16

func.func @unchanged_views(
    %input: memref<16xf32>
        {nv_tensor_ir.alignment = 16 : i64},
    %output: memref<16xf32>
        {nv_tensor_ir.alignment = 16 : i64}) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 4>
} {
  %load = bufferization.to_tensor %input restrict
      : memref<16xf32>
        to tensor<16xf32>
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<16xf32>,
         memref<16xf32>) -> ()
  return
}
