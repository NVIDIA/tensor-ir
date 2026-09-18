// RUN: tensor_ir-opt %s | tensor_ir-opt | FileCheck %s

func.func @load(
    %base: !ptr.ptr<#ptr.generic_space>,
    %m: index, %stride: index, %offset: index,
    %row: index, %column: index) -> tensor<32x64xf32> {
  // CHECK-LABEL: func.func @load(
  // CHECK-SAME: %[[BASE:[^:]+]]: !ptr.ptr<#ptr.generic_space>
  // CHECK-SAME: %[[M:[^:]+]]: index, %[[STRIDE:[^:]+]]: index, %[[OFFSET:[^:]+]]: index
  // CHECK-SAME: %[[ROW:[^:]+]]: index, %[[COLUMN:[^:]+]]: index
  // CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[BASE]][%[[ROW]], %[[COLUMN]]] view offset: [%[[OFFSET]]], sizes: [%[[M]], 128], strides: [%[[STRIDE]], 1], alignment: 16 padding = <zero>
  // CHECK: return %[[TILE]]
  %tile = nv_tensor_ir.load %base[%row, %column]
      view offset: [%offset], sizes: [%m, 128],
           strides: [%stride, 1], alignment: 16
      padding = <zero>
      : !ptr.ptr<#ptr.generic_space> to tensor<32x64xf32>
  return %tile : tensor<32x64xf32>
}

func.func @store(
    %base: !ptr.ptr<#ptr.generic_space>, %tile: tensor<16x16xsi32>,
    %row: index, %column: index) {
  // CHECK-LABEL: func.func @store(
  // CHECK-SAME: %[[BASE:[^:]+]]: !ptr.ptr<#ptr.generic_space>, %[[TILE:[^:]+]]: tensor<16x16xsi32>
  // CHECK-SAME: %[[ROW:[^:]+]]: index, %[[COLUMN:[^:]+]]: index
  // CHECK: nv_tensor_ir.store %[[TILE]], %[[BASE]][%[[ROW]], %[[COLUMN]]] view offset: [0], sizes: [64, 64], strides: [64, 1], alignment: 4
  nv_tensor_ir.store %tile, %base[%row, %column]
      view offset: [0], sizes: [64, 64], strides: [64, 1], alignment: 4
      : tensor<16x16xsi32>, !ptr.ptr<#ptr.generic_space>
  return
}

// A logical access may change rank and static layout while retaining the same
// opaque pointer. This supports reshape, transpose, broadcast, and slice
// addressing without reconstructing a memref descriptor.
// CHECK-LABEL: func.func @logical_reinterpretation(
// CHECK: nv_tensor_ir.load {{.*}} view offset: [128], sizes: [32, 64], strides: [1, 32], alignment: 16
func.func @logical_reinterpretation(
    %base: !ptr.ptr<#ptr.generic_space>, %row: index, %column: index)
    -> tensor<8x8xf32> {
  %tile = nv_tensor_ir.load %base[%row, %column]
      view offset: [128], sizes: [32, 64], strides: [1, 32], alignment: 16
      : !ptr.ptr<#ptr.generic_space> to tensor<8x8xf32>
  return %tile : tensor<8x8xf32>
}

// Empty logical views remain representable so zero-extent slices can pass
// through tiled-program formation.
// CHECK-LABEL: func.func @empty_view(
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[TILE:.*]] = nv_tensor_ir.load {{.*}} sizes: [0]
// CHECK: nv_tensor_ir.store %[[TILE]], {{.*}} sizes: [%[[ZERO]]]
func.func @empty_view(%base: !ptr.ptr<#ptr.generic_space>) {
  %zero = arith.constant 0 : index
  %tile = nv_tensor_ir.load %base[%zero]
      view offset: [0], sizes: [0], strides: [1], alignment: 4
      padding = <zero>
      : !ptr.ptr<#ptr.generic_space> to tensor<1xf32>
  nv_tensor_ir.store %tile, %base[%zero]
      view offset: [0], sizes: [%zero], strides: [1], alignment: 4
      : tensor<1xf32>, !ptr.ptr<#ptr.generic_space>
  return
}

// CHECK-LABEL: func.func @mixed_coordinates(
func.func @mixed_coordinates(%base: !ptr.ptr<#ptr.generic_space>, %column: index) {
  // CHECK: nv_tensor_ir.load {{.*}}[2, %{{.*}}, 4]
  %tile = nv_tensor_ir.load %base[2, %column, 4]
      view offset: [7], sizes: [64, 128, 256], strides: [32768, 256, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8x8x8xf32>
  // CHECK: nv_tensor_ir.store {{.*}}[%{{.*}}, 3, %{{.*}}]
  nv_tensor_ir.store %tile, %base[%column, 3, %column]
      view offset: [7], sizes: [64, 128, 256], strides: [32768, 256, 1], alignment: 4
      : tensor<8x8x8xf32>, !ptr.ptr<#ptr.generic_space>
  // CHECK: nv_tensor_ir.load {{.*}}[0, 1, 2]
  %static = nv_tensor_ir.load %base[0, 1, 2]
      view offset: [0], sizes: [64, 128, 256], strides: [32768, 256, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8x8x8xf32>
  // CHECK: nv_tensor_ir.store {{.*}}[0, 1, 2]
  nv_tensor_ir.store %static, %base[0, 1, 2]
      view offset: [0], sizes: [64, 128, 256], strides: [32768, 256, 1], alignment: 4
      : tensor<8x8x8xf32>, !ptr.ptr<#ptr.generic_space>
  return
}
