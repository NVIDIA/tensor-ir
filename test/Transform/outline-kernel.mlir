// RUN: tensor_ir-opt %s --outline-tensor-ir-kernel | FileCheck %s

func.func @copy(
    %input: memref<?xf32, strided<[?], offset: ?>, #ptr.generic_space>,
    %output: memref<?xf32, strided<[?], offset: ?>, #ptr.generic_space>,
    %size: index, %inputStride: index, %outputStride: index,
    %inputOffset: index, %outputOffset: index, %tiles: index) {
  %inputPtr = ptr.to_ptr %input
      : memref<?xf32, strided<[?], offset: ?>, #ptr.generic_space>
        -> !ptr.ptr<#ptr.generic_space>
  %outputPtr = ptr.to_ptr %output
      : memref<?xf32, strided<[?], offset: ?>, #ptr.generic_space>
        -> !ptr.ptr<#ptr.generic_space>
  scf.forall (%block) in (%tiles) {
    %inGrid = arith.cmpi ult, %block, %tiles : index
    %tile = nv_tensor_ir.load %inputPtr[%block]
        view offset: [%inputOffset], sizes: [%size],
             strides: [%inputStride], alignment: 16 padding = <zero>
        : !ptr.ptr<#ptr.generic_space> to tensor<32xf32>
    nv_tensor_ir.store %tile, %outputPtr[%block]
        view offset: [%outputOffset], sizes: [%size],
             strides: [%outputStride], alignment: 16
        : tensor<32xf32>, !ptr.ptr<#ptr.generic_space>
  } {mapping = [#gpu.block<x>]}
  return
}

// CHECK-LABEL: module attributes {gpu.container_module}

// CHECK-LABEL: func.func @copy_dispatch(
// CHECK-SAME: %[[INPUT:[^:]+]]: memref<?xf32, strided<[?], offset: ?>, #ptr.generic_space>
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<?xf32, strided<[?], offset: ?>, #ptr.generic_space>
// CHECK-SAME: %[[SIZE:[^:]+]]: index, %[[ISTRIDE:[^:]+]]: index, %[[OSTRIDE:[^:]+]]: index
// CHECK-SAME: %[[IOFFSET:[^:]+]]: index, %[[OOFFSET:[^:]+]]: index, %[[TILES:[^:)]+]]: index)
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: gpu.launch_func @kernels_0::@copy blocks in (%[[TILES]], %[[ONE]], %[[ONE]]) threads in (%[[ONE]], %[[ONE]], %[[ONE]]) args(%[[INPUT]] : memref<?xf32, strided<[?], offset: ?>, #ptr.generic_space>, %[[OUTPUT]] : memref<?xf32, strided<[?], offset: ?>, #ptr.generic_space>, %[[SIZE]] : index, %[[ISTRIDE]] : index, %[[OSTRIDE]] : index, %[[IOFFSET]] : index, %[[OOFFSET]] : index, %[[TILES]] : index)
// CHECK: return
// CHECK-NOT: scf.forall
// CHECK-NOT: nv_tensor_ir.load

// A pre-existing host symbol forces a deterministic gpu.module suffix.
func.func private @kernels()

// A pre-existing dispatch-like symbol forces a deterministic unique suffix.
func.func private @two_outputs_dispatch()

func.func @two_outputs(%input: memref<32xf32>, %output0: memref<32xf32>,
                       %output1: memref<32xf32>) {
  scf.forall (%block) in (1) {
    %value = arith.constant 0.0 : f32
    memref.store %value, %output0[%block] : memref<32xf32>
  } {mapping = [#gpu.block<x>]}
  return
}

// CHECK-LABEL: func.func @two_outputs_dispatch_0(
// CHECK-SAME: %[[TWO_INPUT:[^:]+]]: memref<32xf32>, %[[OUT0:[^:]+]]: memref<32xf32>, %[[OUT1:[^:]+]]: memref<32xf32>)
// CHECK: %[[GRID_ONE:.*]] = arith.constant 1 : index
// CHECK: %[[THREAD_ONE:.*]] = arith.constant 1 : index
// CHECK: gpu.launch_func @kernels_0::@two_outputs blocks in (%[[GRID_ONE]], %[[THREAD_ONE]], %[[THREAD_ONE]]) threads in (%[[THREAD_ONE]], %[[THREAD_ONE]], %[[THREAD_ONE]]) args(%[[TWO_INPUT]] : memref<32xf32>, %[[OUT0]] : memref<32xf32>, %[[OUT1]] : memref<32xf32>)
// CHECK: return
// CHECK-NOT: scf.forall

// CHECK-LABEL: gpu.module @kernels_0 {
// CHECK-LABEL: func.func @copy(
// CHECK-SAME: %[[KINPUT:[^:]+]]: memref<?xf32, strided<[?], offset: ?>, #ptr.generic_space>
// CHECK-SAME: %[[KOUTPUT:[^:]+]]: memref<?xf32, strided<[?], offset: ?>, #ptr.generic_space>
// CHECK-SAME: %[[SIZE:[^:]+]]: index, %[[ISTRIDE:[^:]+]]: index, %[[OSTRIDE:[^:]+]]: index
// CHECK-SAME: %[[IOFFSET:[^:]+]]: index, %[[OOFFSET:[^:]+]]: index, %{{.*}}: index
// CHECK-SAME: attributes {gpu.kernel}
// CHECK: %[[BLOCK:.*]] = gpu.block_id x
// CHECK: %[[GRID:.*]] = gpu.grid_dim x
// CHECK: %[[IPTR:.*]] = ptr.to_ptr %[[KINPUT]]
// CHECK: %[[OPTR:.*]] = ptr.to_ptr %[[KOUTPUT]]
// CHECK: arith.cmpi ult, %[[BLOCK]], %[[GRID]] : index
// CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[IPTR]][%[[BLOCK]]] view offset: [%[[IOFFSET]]], sizes: [%[[SIZE]]], strides: [%[[ISTRIDE]]], alignment: 16
// CHECK: nv_tensor_ir.store %[[TILE]], %[[OPTR]][%[[BLOCK]]] view offset: [%[[OOFFSET]]], sizes: [%[[SIZE]]], strides: [%[[OSTRIDE]]], alignment: 16
// CHECK: return

// CHECK-LABEL: func.func @two_outputs(
// CHECK-SAME: %{{.*}}: memref<32xf32>, %[[KOUT0:[^:]+]]: memref<32xf32>, %{{.*}}: memref<32xf32>
// CHECK-SAME: attributes {gpu.kernel}
// CHECK: %[[KVALUE:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: memref.store %[[KVALUE]], %[[KOUT0]][%{{.*}}] : memref<32xf32>
// CHECK: return
