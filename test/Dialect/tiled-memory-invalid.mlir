// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics

func.func @missing_dynamic_size(
    %base: !ptr.ptr<#ptr.generic_space>, %stride: index, %index: index)
    -> tensor<8x8xf32> {
  // expected-error @below {{expected the number of 'sizes' to match the number of dynamic entries in 'static_sizes'}}
  %tile = "nv_tensor_ir.load"(%base, %stride, %index, %index)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 1, 2>,
        static_offsets = array<i64: 0>,
        static_indices = array<i64: -9223372036854775808, -9223372036854775808>,
        static_sizes = array<i64: -9223372036854775808, 16>,
        static_strides = array<i64: -9223372036854775808, 1>,
        alignment = 4 : i64}>
      : (!ptr.ptr<#ptr.generic_space>, index, index, index) -> tensor<8x8xf32>
  return %tile : tensor<8x8xf32>
}

// -----

func.func @missing_dynamic_stride(
    %base: !ptr.ptr<#ptr.generic_space>, %size: index, %index: index)
    -> tensor<8x8xf32> {
  // expected-error @below {{expected the number of 'strides' to match the number of dynamic entries in 'static_strides'}}
  %tile = "nv_tensor_ir.load"(%base, %size, %index, %index)
      <{operandSegmentSizes = array<i32: 1, 0, 1, 0, 2>,
        static_offsets = array<i64: 0>,
        static_indices = array<i64: -9223372036854775808, -9223372036854775808>,
        static_sizes = array<i64: -9223372036854775808, 16>,
        static_strides = array<i64: -9223372036854775808, 1>,
        alignment = 4 : i64}>
      : (!ptr.ptr<#ptr.generic_space>, index, index, index) -> tensor<8x8xf32>
  return %tile : tensor<8x8xf32>
}

// -----

func.func @missing_dynamic_offset(
    %base: !ptr.ptr<#ptr.generic_space>, %index: index) -> tensor<8xf32> {
  // expected-error @below {{expected the number of 'offsets' to match the number of dynamic entries in 'static_offsets'}}
  %tile = "nv_tensor_ir.load"(%base, %index)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 1>,
        static_offsets = array<i64: -9223372036854775808>,
        static_indices = array<i64: -9223372036854775808>,
        static_sizes = array<i64: 16>, static_strides = array<i64: 1>,
        alignment = 4 : i64}>
      : (!ptr.ptr<#ptr.generic_space>, index) -> tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

func.func @wrong_tile_rank(
    %base: !ptr.ptr<#ptr.generic_space>, %index: index) -> tensor<8xf32> {
  // expected-error @below {{expected 1 size values, got 2}}
  %tile = nv_tensor_ir.load %base[%index, %index]
      view offset: [0], sizes: [16, 16], strides: [16, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

func.func @wrong_coordinate_count(
    %base: !ptr.ptr<#ptr.generic_space>, %index: index) -> tensor<8x8xf32> {
  // expected-error @below {{expected 2 tile coordinate(s), but got 1}}
  %tile = nv_tensor_ir.load %base[%index]
      view offset: [0], sizes: [16, 16], strides: [16, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8x8xf32>
  return %tile : tensor<8x8xf32>
}

// -----

func.func @negative_coordinate(%base: !ptr.ptr<#ptr.generic_space>)
    -> tensor<8xf32> {
  %negative = arith.constant -1 : index
  // expected-error @below {{expected nonnegative tile coordinate #0, but got -1}}
  %tile = nv_tensor_ir.load %base[%negative]
      view offset: [0], sizes: [16], strides: [1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

func.func @integer_nan_padding(
    %base: !ptr.ptr<#ptr.generic_space>, %index: index) -> tensor<8xsi32> {
  // expected-error @below {{padding nan requires a floating-point element type, got 'si32'}}
  %tile = nv_tensor_ir.load %base[%index]
      view offset: [0], sizes: [16], strides: [1], alignment: 4
      padding = <nan>
      : !ptr.ptr<#ptr.generic_space> to tensor<8xsi32>
  return %tile : tensor<8xsi32>
}

// -----

func.func @mismatched_size_and_stride_ranks(%base: !ptr.ptr<#ptr.generic_space>, %index: index) -> tensor<8xf32> {
  // expected-error @below {{expected mixed sizes rank to match mixed strides rank}}
  %tile = nv_tensor_ir.load %base[%index]
      view offset: [0], sizes: [8, 8], strides: [1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

func.func @invalid_alignment(%base: !ptr.ptr<#ptr.generic_space>, %index: index) -> tensor<8xf32> {
  // expected-error @below {{attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  %tile = nv_tensor_ir.load %base[%index]
      view offset: [0], sizes: [8], strides: [1], alignment: 3
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

func.func @empty_offset_view(
    %base: !ptr.ptr<#ptr.generic_space>, %index: index) -> tensor<8xf32> {
  // expected-error @below {{expected mixed offsets rank to match mixed sizes rank}}
  %tile = nv_tensor_ir.load %base[%index]
      view offset: [], sizes: [16], strides: [1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

func.func @multiple_offset_view(
    %base: !ptr.ptr<#ptr.generic_space>, %index: index) -> tensor<8xf32> {
  // expected-error @below {{expected mixed offsets rank to match mixed sizes rank}}
  %tile = nv_tensor_ir.load %base[%index]
      view offset: [0, 1], sizes: [16], strides: [1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

func.func @negative_dynamic_load_size(
    %base: !ptr.ptr<#ptr.generic_space>, %index: index) {
  %negative = arith.constant -1 : index
  // expected-error @below {{expected dynamic size #0 to be nonnegative, but got -1}}
  %tile = nv_tensor_ir.load %base[%index]
      view offset: [0], sizes: [%negative], strides: [1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  return
}

// -----

func.func @negative_dynamic_load_stride(
    %base: !ptr.ptr<#ptr.generic_space>, %index: index) {
  %negative = arith.constant -1 : index
  // expected-error @below {{expected dynamic stride #0 to be nonnegative, but got -1}}
  %tile = nv_tensor_ir.load %base[%index]
      view offset: [0], sizes: [16], strides: [%negative], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  return
}

// -----

func.func @negative_dynamic_load_offset(
    %base: !ptr.ptr<#ptr.generic_space>, %index: index) {
  %negative = arith.constant -1 : index
  // expected-error @below {{expected dynamic offset #0 to be nonnegative, but got -1}}
  %tile = nv_tensor_ir.load %base[%index]
      view offset: [%negative], sizes: [16], strides: [1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  return
}

// -----

// -----

func.func @negative_static_coordinate(%base: !ptr.ptr<#ptr.generic_space>) -> tensor<8xf32> {
  // expected-error @below {{expected nonnegative tile coordinate #0, but got -1}}
  %tile = nv_tensor_ir.load %base[-1]
      view offset: [0], sizes: [16], strides: [1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

func.func @negative_static_store_coordinate(%base: !ptr.ptr<#ptr.generic_space>, %tile: tensor<8xf32>) {
  // expected-error @below {{expected nonnegative tile coordinate #0, but got -2}}
  nv_tensor_ir.store %tile, %base[-2]
      view offset: [0], sizes: [16], strides: [1], alignment: 4
      : tensor<8xf32>, !ptr.ptr<#ptr.generic_space>
  return
}

// -----

func.func @missing_dynamic_index(%base: !ptr.ptr<#ptr.generic_space>) -> tensor<8xf32> {
  // expected-error @below {{expected 1 dynamic indices values}}
  %tile = "nv_tensor_ir.load"(%base)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>,
        static_offsets = array<i64: 0>, static_sizes = array<i64: 16>,
        static_strides = array<i64: 1>,
        static_indices = array<i64: -9223372036854775808>, alignment = 4 : i64}>
      : (!ptr.ptr<#ptr.generic_space>) -> tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

func.func @extra_dynamic_index(%base: !ptr.ptr<#ptr.generic_space>, %tile: tensor<8xf32>, %index: index) {
  // expected-error @below {{expected 0 dynamic indices values}}
  "nv_tensor_ir.store"(%tile, %base, %index)
      <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>,
        static_offsets = array<i64: 0>, static_sizes = array<i64: 16>,
        static_strides = array<i64: 1>, static_indices = array<i64: 0>,
        alignment = 4 : i64}>
      : (tensor<8xf32>, !ptr.ptr<#ptr.generic_space>, index) -> ()
  return
}
