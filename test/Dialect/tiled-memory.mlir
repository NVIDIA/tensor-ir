// RUN: tensor_ir-opt %s | tensor_ir-opt | FileCheck %s

func.func @load(
    %base: memref<f32>,
    %m: index, %stride: index, %offset: index,
    %row: index, %column: index) -> tensor<32x64xf32> {
  // CHECK-LABEL: func.func @load(
  // CHECK-SAME: %[[BASE:[^:]+]]: memref<f32>
  // CHECK-SAME: %[[M:[^:]+]]: index, %[[STRIDE:[^:]+]]: index, %[[OFFSET:[^:]+]]: index
  // CHECK-SAME: %[[ROW:[^:]+]]: index, %[[COLUMN:[^:]+]]: index
  // CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[OFFSET]]], sizes: [%[[M]], 128], strides: [%[[STRIDE]], 1]
  // CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[VIEW]] tile_indices [%[[ROW]], %[[COLUMN]]], alignment 16, padding <zero>
  // CHECK: return %[[TILE]]
  %access_view_0 = memref.reinterpret_cast %base to offset: [%offset], sizes: [%m, 128], strides: [%stride, 1]
      : memref<f32> to memref<?x128xf32, strided<[?, 1], offset: ?>>
  %tile = nv_tensor_ir.load %access_view_0 tile_indices [%row, %column], alignment 16,
      padding <zero>
      : memref<?x128xf32, strided<[?, 1], offset: ?>> to tensor<32x64xf32>
  return %tile : tensor<32x64xf32>
}

func.func @store(
    %base: memref<si32>, %tile: tensor<16x16xsi32>,
    %row: index, %column: index) {
  // CHECK-LABEL: func.func @store(
  // CHECK-SAME: %[[BASE:[^:]+]]: memref<si32>, %[[TILE:[^:]+]]: tensor<16x16xsi32>
  // CHECK-SAME: %[[ROW:[^:]+]]: index, %[[COLUMN:[^:]+]]: index
  // CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [0], sizes: [64, 64], strides: [64, 1]
  // CHECK: nv_tensor_ir.store %[[TILE]], %[[VIEW]] tile_indices [%[[ROW]], %[[COLUMN]]], alignment 4
  %access_view_0 = memref.reinterpret_cast %base to offset: [0], sizes: [64, 64], strides: [64, 1]
      : memref<si32> to memref<64x64xsi32, strided<[64, 1], offset: 0>>
  nv_tensor_ir.store %tile, %access_view_0 tile_indices [%row, %column], alignment 4
      : tensor<16x16xsi32>, memref<64x64xsi32, strided<[64, 1], offset: 0>>
  return
}

// A logical access may change rank and static layout while retaining the same
// underlying buffer. The reinterpret cast owns the complete logical view.
// CHECK-LABEL: func.func @logical_reinterpretation(
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [128], sizes: [32, 64], strides: [1, 32]
// CHECK: nv_tensor_ir.load %[[VIEW]]{{.*}}, alignment 16
func.func @logical_reinterpretation(
    %base: memref<f32>, %row: index, %column: index)
    -> tensor<8x8xf32> {
  %access_view_0 = memref.reinterpret_cast %base to offset: [128], sizes: [32, 64], strides: [1, 32]
      : memref<f32> to memref<32x64xf32, strided<[1, 32], offset: 128>>
  %tile = nv_tensor_ir.load %access_view_0 tile_indices [%row, %column], alignment 16
      : memref<32x64xf32, strided<[1, 32], offset: 128>> to tensor<8x8xf32>
  return %tile : tensor<8x8xf32>
}

// Empty logical views remain representable so zero-extent slices can pass
// through tiled-program formation.
// CHECK-LABEL: func.func @empty_view(
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[TILE:.*]] = nv_tensor_ir.load {{.*}} : memref<0xf32,
// CHECK: %[[EMPTY:.*]] = memref.reinterpret_cast {{.*}} sizes: [%[[ZERO]]]
// CHECK: nv_tensor_ir.store %[[TILE]], %[[EMPTY]]
func.func @empty_view(%base: memref<f32>) {
  %zero = arith.constant 0 : index
  %access_view_0 = memref.reinterpret_cast %base to offset: [0], sizes: [0], strides: [1]
      : memref<f32> to memref<0xf32, strided<[1], offset: 0>>
  %tile = nv_tensor_ir.load %access_view_0 tile_indices [%zero], alignment 4,
      padding <zero>
      : memref<0xf32, strided<[1], offset: 0>> to tensor<1xf32>
  %access_view_1 = memref.reinterpret_cast %base to offset: [0], sizes: [%zero], strides: [1]
      : memref<f32> to memref<?xf32, strided<[1], offset: 0>>
  nv_tensor_ir.store %tile, %access_view_1 tile_indices [%zero], alignment 4
      : tensor<1xf32>, memref<?xf32, strided<[1], offset: 0>>
  return
}

// CHECK-LABEL: func.func @mixed_coordinates(
func.func @mixed_coordinates(%base: memref<f32>, %column: index) {
  // CHECK: nv_tensor_ir.load {{.*}} tile_indices [2, %{{.*}}, 4]
  %access_view_0 = memref.reinterpret_cast %base to offset: [7], sizes: [64, 128, 256], strides: [32768, 256, 1]
      : memref<f32> to memref<64x128x256xf32, strided<[32768, 256, 1], offset: 7>>
  %tile = nv_tensor_ir.load %access_view_0 tile_indices [2, %column, 4], alignment 4
      : memref<64x128x256xf32, strided<[32768, 256, 1], offset: 7>> to tensor<8x8x8xf32>
  // CHECK: nv_tensor_ir.store {{.*}} tile_indices [%{{.*}}, 3, %{{.*}}]
  %access_view_1 = memref.reinterpret_cast %base to offset: [7], sizes: [64, 128, 256], strides: [32768, 256, 1]
      : memref<f32> to memref<64x128x256xf32, strided<[32768, 256, 1], offset: 7>>
  nv_tensor_ir.store %tile, %access_view_1 tile_indices [%column, 3, %column], alignment 4
      : tensor<8x8x8xf32>, memref<64x128x256xf32, strided<[32768, 256, 1], offset: 7>>
  // CHECK: nv_tensor_ir.load {{.*}} tile_indices [0, 1, 2]
  %access_view_2 = memref.reinterpret_cast %base to offset: [0], sizes: [64, 128, 256], strides: [32768, 256, 1]
      : memref<f32> to memref<64x128x256xf32, strided<[32768, 256, 1], offset: 0>>
  %static = nv_tensor_ir.load %access_view_2 tile_indices [0, 1, 2], alignment 4
      : memref<64x128x256xf32, strided<[32768, 256, 1], offset: 0>> to tensor<8x8x8xf32>
  // CHECK: nv_tensor_ir.store {{.*}} tile_indices [0, 1, 2]
  %access_view_3 = memref.reinterpret_cast %base to offset: [0], sizes: [64, 128, 256], strides: [32768, 256, 1]
      : memref<f32> to memref<64x128x256xf32, strided<[32768, 256, 1], offset: 0>>
  nv_tensor_ir.store %static, %access_view_3 tile_indices [0, 1, 2], alignment 4
      : tensor<8x8x8xf32>, memref<64x128x256xf32, strided<[32768, 256, 1], offset: 0>>
  return
}
