// RUN: tensor_ir-opt %s --split-input-file -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation" -canonicalize -cse | FileCheck %s

// An internal view applies its static offset exactly once.
module @static_offset {
  gpu.module @kernels {
    func.func @static_offset(%input: memref<64xf32>) attributes {gpu.kernel} {
      %view = memref.reinterpret_cast %input to offset: [7], sizes: [32], strides: [1]
          : memref<64xf32> to memref<32xf32, strided<[1], offset: 7>>
      %tile = nv_tensor_ir.load %view tile_indices [0], alignment 4
          : memref<32xf32, strided<[1], offset: 7>> to tensor<8xf32>
      nv_tensor_ir.store %tile, %view tile_indices [1], alignment 4
          : tensor<8xf32>, memref<32xf32, strided<[1], offset: 7>>
      return
    }
  }
}
// CHECK-LABEL: entry @static_offset(
// CHECK-SAME: %[[BASE:[^:]+]]: tile<ptr<f32>>)
// CHECK: %[[SEVEN:.*]] = constant <i32: 7>
// CHECK: %[[ORIGIN:.*]] = offset %[[BASE]], %[[SEVEN]]
// CHECK: %[[VIEW:.*]] = make_tensor_view %[[ORIGIN]], shape = [32], strides = [1]
// CHECK: load_view_tko
// CHECK: store_view_tko

// -----

// Internal dynamic offsets do not add fields to the memref argument ABI.
// Extracting a view's metadata preserves its offset for subsequent slicing.
module @dynamic_offset {
  gpu.module @kernels {
    func.func @dynamic_offset(%input: memref<?xf32, strided<[?]>>) attributes {gpu.kernel} {
      %input_base, %input_offset, %input_size, %input_stride = memref.extract_strided_metadata %input
          : memref<?xf32, strided<[?]>> -> memref<f32>, index, index, index
      %block = gpu.block_id x
      %view = memref.reinterpret_cast %input to offset: [%block], sizes: [%input_size], strides: [%input_stride]
          : memref<?xf32, strided<[?]>> to memref<?xf32, strided<[?], offset: ?>>
      %base, %offset, %size, %stride = memref.extract_strided_metadata %view
          : memref<?xf32, strided<[?], offset: ?>> -> memref<f32>, index, index, index
      %five = arith.constant 5 : index
      %shift = arith.addi %offset, %five : index
      %slice = memref.reinterpret_cast %view to offset: [%shift], sizes: [16], strides: [%stride]
          : memref<?xf32, strided<[?], offset: ?>> to memref<16xf32, strided<[?], offset: ?>>
      %tile = nv_tensor_ir.load %slice tile_indices [0], alignment 4
          : memref<16xf32, strided<[?], offset: ?>> to tensor<8xf32>
      nv_tensor_ir.store %tile, %view tile_indices [0], alignment 4
          : tensor<8xf32>, memref<?xf32, strided<[?], offset: ?>>
      return
    }
  }
}
// CHECK-LABEL: entry @dynamic_offset(
// CHECK-SAME: %[[BASE:[^:]+]]: tile<ptr<f32>>, %[[SIZE:[^:]+]]: tile<i32>, %[[STRIDE:[^:]+]]: tile<i32>)
// CHECK: %[[FIVE:.*]] = constant <i32: 5>
// CHECK: %[[OFFSET:[^,]+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[SHIFT:.*]] = addi %[[OFFSET]], %[[FIVE]]
// CHECK: %[[SLICE:.*]] = offset %[[BASE]], %[[SHIFT]]
// CHECK: make_tensor_view %[[SLICE]], shape = [16], strides = [%[[STRIDE]]]
// CHECK: load_view_tko
// CHECK: %[[ORIGIN:.*]] = offset %[[BASE]], %[[OFFSET]]
// CHECK: make_tensor_view %[[ORIGIN]], shape = [%[[SIZE]]], strides = [%[[STRIDE]]]
// CHECK: store_view_tko

// -----

// Reinterpret casts replace offsets rather than accumulating them, even when
// the source already has a nonzero offset or the result changes rank.
module @replace_offset {
  gpu.module @kernels {
    func.func @replace_offset(%input: memref<64xf32>) attributes {gpu.kernel} {
      %initial = memref.reinterpret_cast %input to offset: [7], sizes: [64], strides: [1]
          : memref<64xf32> to memref<64xf32, strided<[1], offset: 7>>
      %first = memref.reinterpret_cast %initial to offset: [3], sizes: [32], strides: [1]
          : memref<64xf32, strided<[1], offset: 7>> to memref<32xf32, strided<[1], offset: 3>>
      %second = memref.reinterpret_cast %first to offset: [0], sizes: [4, 8], strides: [8, 1]
          : memref<32xf32, strided<[1], offset: 3>> to memref<4x8xf32, strided<[8, 1]>>
      %tile = nv_tensor_ir.load %second tile_indices [0, 0], alignment 4
          : memref<4x8xf32, strided<[8, 1]>> to tensor<4x8xf32>
      nv_tensor_ir.store %tile, %second tile_indices [0, 0], alignment 4
          : tensor<4x8xf32>, memref<4x8xf32, strided<[8, 1]>>
      return
    }
  }
}
// CHECK-LABEL: entry @replace_offset(
// CHECK-SAME: %[[BASE:[^:]+]]: tile<ptr<f32>>)
// CHECK-NOT: offset
// CHECK: make_tensor_view %[[BASE]], shape = [4, 8], strides = [8, 1]
// CHECK: load_view_tko
// CHECK: store_view_tko
