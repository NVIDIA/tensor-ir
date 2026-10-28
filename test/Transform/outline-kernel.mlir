// RUN: tensor_ir-opt %s --outline-tensor-ir-kernel | FileCheck %s --implicit-check-not=scf.forall

func.func @copy(
    %input: memref<?xf32, strided<[?]>>,
    %output: memref<?xf32, strided<[?]>>,
    %size: index, %inputStride: index, %outputStride: index,
    %inputOffset: index, %outputOffset: index, %tiles: index) {
  scf.forall (%block) in (%tiles) {
    %inGrid = arith.cmpi ult, %block, %tiles : index
    %inView = memref.reinterpret_cast %input to offset: [%inputOffset], sizes: [%size], strides: [%inputStride]
        : memref<?xf32, strided<[?]>> to memref<?xf32, strided<[?], offset: ?>>
    %tile = nv_tensor_ir.load %inView tile_indices [%block], alignment 16, padding <zero>
        : memref<?xf32, strided<[?], offset: ?>> to tensor<32xf32>
    %outView = memref.reinterpret_cast %output to offset: [%outputOffset], sizes: [%size], strides: [%outputStride]
        : memref<?xf32, strided<[?]>> to memref<?xf32, strided<[?], offset: ?>>
    nv_tensor_ir.store %tile, %outView tile_indices [%block], alignment 16
        : tensor<32xf32>, memref<?xf32, strided<[?], offset: ?>>
  } {mapping = [#gpu.block<x>]}
  return
}

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

// CHECK-LABEL: module attributes {gpu.container_module}

// CHECK-LABEL: func.func @copy_dispatch(
// CHECK-SAME: %[[INPUT:[^:]+]]: memref<?xf32, strided<[?]>>
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<?xf32, strided<[?]>>
// CHECK-SAME: %[[SIZE:[^:]+]]: index
// CHECK-SAME: %[[ISTRIDE:[^:]+]]: index
// CHECK-SAME: %[[OSTRIDE:[^:]+]]: index
// CHECK-SAME: %[[IOFFSET:[^:]+]]: index
// CHECK-SAME: %[[OOFFSET:[^:]+]]: index
// CHECK-SAME: %[[TILES:[^:)]+]]: index)
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func @kernels_0::@copy
// CHECK-SAME: blocks in (%[[TILES]], %[[ONE]], %[[ONE]])
// CHECK-SAME: threads in (%[[ONE]], %[[ONE]], %[[ONE]])
// CHECK-SAME: args(%[[INPUT]] : memref<?xf32, strided<[?]>>,
// CHECK-SAME: %[[OUTPUT]] : memref<?xf32, strided<[?]>>,
// CHECK-SAME: %[[SIZE]] : index, %[[ISTRIDE]] : index, %[[OSTRIDE]] : index,
// CHECK-SAME: %[[IOFFSET]] : index, %[[OOFFSET]] : index, %[[TILES]] : index)
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func private @kernels()
// CHECK-LABEL: func.func private @two_outputs_dispatch()

// CHECK-LABEL: func.func @two_outputs_dispatch_0(
// CHECK-SAME: %[[INPUT:[^:]+]]: memref<32xf32>
// CHECK-SAME: %[[OUT0:[^:]+]]: memref<32xf32>
// CHECK-SAME: %[[OUT1:[^:]+]]: memref<32xf32>)
// CHECK-NEXT: %[[GRID_ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[THREAD_ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func @kernels_0::@two_outputs
// CHECK-SAME: blocks in (%[[GRID_ONE]], %[[THREAD_ONE]], %[[THREAD_ONE]])
// CHECK-SAME: threads in (%[[THREAD_ONE]], %[[THREAD_ONE]], %[[THREAD_ONE]])
// CHECK-SAME: args(%[[INPUT]] : memref<32xf32>, %[[OUT0]] : memref<32xf32>, %[[OUT1]] : memref<32xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: gpu.module @kernels_0 {
// CHECK-LABEL: func.func @copy(
// CHECK-SAME: %[[INPUT:[^:]+]]: memref<?xf32, strided<[?]>>
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<?xf32, strided<[?]>>
// CHECK-SAME: %[[SIZE:[^:]+]]: index
// CHECK-SAME: %[[ISTRIDE:[^:]+]]: index
// CHECK-SAME: %[[OSTRIDE:[^:]+]]: index
// CHECK-SAME: %[[IOFFSET:[^:]+]]: index
// CHECK-SAME: %[[OOFFSET:[^:]+]]: index
// CHECK-SAME: %{{[^:]+}}: index) attributes {gpu.kernel}
// CHECK: %[[BLOCK:.*]] = gpu.block_id x
// CHECK: %[[GRID:.*]] = gpu.grid_dim x
// CHECK: arith.cmpi ult, %[[BLOCK]], %[[GRID]] : index
// CHECK: %[[IVIEW:.*]] = memref.reinterpret_cast %[[INPUT]]
// CHECK-SAME: to offset: [%[[IOFFSET]]], sizes: [%[[SIZE]]], strides: [%[[ISTRIDE]]]
// CHECK-SAME: : memref<?xf32, strided<[?]>>
// CHECK-SAME: to memref<?xf32, strided<[?], offset: ?>>
// CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[IVIEW]]
// CHECK-SAME: tile_indices [%[[BLOCK]]], alignment 16, padding <zero>
// CHECK-SAME: : memref<?xf32, strided<[?], offset: ?>> to tensor<32xf32>
// CHECK: %[[OVIEW:.*]] = memref.reinterpret_cast %[[OUTPUT]]
// CHECK-SAME: to offset: [%[[OOFFSET]]], sizes: [%[[SIZE]]], strides: [%[[OSTRIDE]]]
// CHECK-SAME: : memref<?xf32, strided<[?]>>
// CHECK-SAME: to memref<?xf32, strided<[?], offset: ?>>
// CHECK: nv_tensor_ir.store %[[TILE]], %[[OVIEW]] tile_indices [%[[BLOCK]]], alignment 16
// CHECK-SAME: : tensor<32xf32>, memref<?xf32, strided<[?], offset: ?>>
// CHECK: return

// CHECK-LABEL: func.func @two_outputs(
// CHECK-SAME: %{{[^:]+}}: memref<32xf32>
// CHECK-SAME: %[[OUT0:[^:]+]]: memref<32xf32>
// CHECK-SAME: %{{[^:]+}}: memref<32xf32>) attributes {gpu.kernel}
// CHECK: %[[BLOCK:.*]] = gpu.block_id x
// CHECK: %[[VALUE:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: memref.store %[[VALUE]], %[[OUT0]][%[[BLOCK]]] : memref<32xf32>
// CHECK: return
