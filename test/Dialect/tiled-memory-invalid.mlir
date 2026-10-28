// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics

// -----

func.func @wrong_tile_rank(%view: memref<16x16xf32>, %index: index) {
  // expected-error @below {{expected view and tile to have the same rank}}
  %tile = nv_tensor_ir.load %view tile_indices [%index, %index], alignment 4
      : memref<16x16xf32> to tensor<8xf32>
  return
}

// -----

func.func @wrong_coordinate_count(%view: memref<16x16xf32>, %index: index) {
  // expected-error @below {{expected 2 tile coordinate(s), but got 1}}
  %tile = nv_tensor_ir.load %view tile_indices [%index], alignment 4
      : memref<16x16xf32> to tensor<8x8xf32>
  return
}

// -----

func.func @negative_coordinate(%view: memref<16xf32>, %index: index) {
  %negative = arith.constant -1 : index
  // expected-error @below {{expected nonnegative tile coordinate #0, but got -1}}
  %tile = nv_tensor_ir.load %view tile_indices [%negative], alignment 4
      : memref<16xf32> to tensor<8xf32>
  return
}

// -----

func.func @integer_nan_padding(%view: memref<16xsi32>, %index: index) {
  // expected-error @below {{padding nan requires a floating-point element type, got 'si32'}}
  %tile = nv_tensor_ir.load %view tile_indices [%index], alignment 4, padding <nan>
      : memref<16xsi32> to tensor<8xsi32>
  return
}

// -----

func.func @invalid_alignment(%view: memref<16xf32>, %index: index) {
  // expected-error @below {{attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  %tile = nv_tensor_ir.load %view tile_indices [%index], alignment 3
      : memref<16xf32> to tensor<8xf32>
  return
}

// -----

func.func @negative_static_coordinate(%view: memref<16xf32>, %index: index) {
  // expected-error @below {{expected nonnegative tile coordinate #0, but got -1}}
  %tile = nv_tensor_ir.load %view tile_indices [-1], alignment 4
      : memref<16xf32> to tensor<8xf32>
  return
}

// -----

func.func @negative_static_store_coordinate(%view: memref<16xf32>, %index: index, %tile: tensor<8xf32>) {
  // expected-error @below {{expected nonnegative tile coordinate #0, but got -2}}
  nv_tensor_ir.store %tile, %view tile_indices [-2], alignment 4
      : tensor<8xf32>, memref<16xf32>
  return
}

// -----

func.func @mismatched_element_type(%view: memref<16xi32>, %index: index) {
  // expected-error @below {{expected view and tile to have the same element type}}
  %tile = nv_tensor_ir.load %view tile_indices [%index], alignment 4
      : memref<16xi32> to tensor<8xf32>
  return
}

// -----

func.func @negative_stride(%view: memref<16xf32, strided<[-1], offset: 15>>, %index: index) {
  // expected-error @below {{expected nonnegative or dynamic offset and positive or dynamic strides}}
  %tile = nv_tensor_ir.load %view tile_indices [%index], alignment 4
      : memref<16xf32, strided<[-1], offset: 15>> to tensor<8xf32>
  return
}

// -----

func.func @negative_offset(%view: memref<16xf32, strided<[1], offset: -1>>, %index: index, %tile: tensor<8xf32>) {
  // expected-error @below {{expected nonnegative or dynamic offset and positive or dynamic strides}}
  nv_tensor_ir.store %tile, %view tile_indices [%index], alignment 4
      : tensor<8xf32>, memref<16xf32, strided<[1], offset: -1>>
  return
}

// -----

func.func @non_strided(%view: memref<16x16xf32, affine_map<(d0, d1) -> (d0 * 16 + d1 floordiv 4, d1 mod 4)>>, %index: index) {
  // expected-error @below {{expected a strided memref view}}
  %tile = nv_tensor_ir.load %view tile_indices [%index, %index], alignment 4
      : memref<16x16xf32, affine_map<(d0, d1) -> (d0 * 16 + d1 floordiv 4, d1 mod 4)>> to tensor<8x8xf32>
  return
}

// -----

func.func @missing_dynamic_index(%view: memref<16xf32>) {
  // expected-error @below {{expected 1 dynamic indices values}}
  %tile = "nv_tensor_ir.load"(%view)
      <{static_indices = array<i64: -9223372036854775808>, alignment = 4 : i64}>
      : (memref<16xf32>) -> tensor<8xf32>
  return
}

// -----

func.func @extra_dynamic_index(%view: memref<16xf32>, %tile: tensor<8xf32>, %index: index) {
  // expected-error @below {{expected 0 dynamic indices values}}
  "nv_tensor_ir.store"(%tile, %view, %index)
      <{static_indices = array<i64: 0>, alignment = 4 : i64}>
      : (tensor<8xf32>, memref<16xf32>, index) -> ()
  return
}

// -----

func.func @zero_stride_load_1(%view: memref<1xf32, strided<[0]>>, %tile: tensor<1xf32>) {
  // expected-error @below {{expected nonnegative or dynamic offset and positive or dynamic strides}}
  %result = nv_tensor_ir.load %view tile_indices [0], alignment 4 : memref<1xf32, strided<[0]>> to tensor<1xf32>
  return
}

// -----

func.func @zero_stride_load_8(%view: memref<8xf32, strided<[0]>>, %tile: tensor<8xf32>) {
  // expected-error @below {{expected nonnegative or dynamic offset and positive or dynamic strides}}
  %result = nv_tensor_ir.load %view tile_indices [0], alignment 4 : memref<8xf32, strided<[0]>> to tensor<8xf32>
  return
}

// -----

func.func @zero_stride_store_1(%view: memref<1xf32, strided<[0]>>, %tile: tensor<1xf32>) {
  // expected-error @below {{expected nonnegative or dynamic offset and positive or dynamic strides}}
  nv_tensor_ir.store %tile, %view tile_indices [0], alignment 4 : tensor<1xf32>, memref<1xf32, strided<[0]>>
  return
}

// -----

func.func @zero_stride_store_8(%view: memref<8xf32, strided<[0]>>, %tile: tensor<8xf32>) {
  // expected-error @below {{expected nonnegative or dynamic offset and positive or dynamic strides}}
  nv_tensor_ir.store %tile, %view tile_indices [0], alignment 4 : tensor<8xf32>, memref<8xf32, strided<[0]>>
  return
}
