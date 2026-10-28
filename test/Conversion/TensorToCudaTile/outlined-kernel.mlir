// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation" -split-input-file | FileCheck %s

module @copy_module {
  gpu.module @kernels {
  func.func @copy(
      %input: memref<65xf32>,
      %output: memref<65xf32>) attributes {gpu.kernel} {
    %block = gpu.block_id x
    %tile = nv_tensor_ir.load %input tile_indices [%block], alignment 16, padding <zero>
        : memref<65xf32> to tensor<32xf32>
    nv_tensor_ir.store %tile, %output tile_indices [%block], alignment 16
        : tensor<32xf32>, memref<65xf32>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_copy_module {
// CHECK: entry @copy(%[[INPUT:[^:]+]]: tile<ptr<f32>>, %[[OUTPUT:[^:]+]]: tile<ptr<f32>>)
// CHECK: %[[BLOCK:[^,]+]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[INPUT_ALIGNED:[^ ]+]] = assume div_by<16>, %[[INPUT]] : tile<ptr<f32>>
// CHECK: %[[INPUT_VIEW:[^ ]+]] = make_tensor_view %[[INPUT_ALIGNED]], shape = [65], strides = [1]
// CHECK: %[[INPUT_PARTITION:[^ ]+]] = make_partition_view %[[INPUT_VIEW]]
// CHECK: %[[TILE:[^,]+]], %{{.*}} = load_view_tko weak %[[INPUT_PARTITION]][%[[BLOCK]]]
// CHECK: %[[OUTPUT_ALIGNED:[^ ]+]] = assume div_by<16>, %[[OUTPUT]] : tile<ptr<f32>>
// CHECK: %[[OUTPUT_VIEW:[^ ]+]] = make_tensor_view %[[OUTPUT_ALIGNED]], shape = [65], strides = [1]
// CHECK: %[[OUTPUT_PARTITION:[^ ]+]] = make_partition_view %[[OUTPUT_VIEW]]
// CHECK: store_view_tko weak %[[TILE]], %[[OUTPUT_PARTITION]][%[[BLOCK]]]
// CHECK: return
// CHECK: }


// -----

module @padding_values_module {
  gpu.module @kernels {
  func.func @padding_values(
      %input: memref<32xf32>) attributes {gpu.kernel} {
    %c0 = arith.constant 0 : index
    %0 = nv_tensor_ir.load %input tile_indices [%c0], alignment 4, padding <zero>
        : memref<32xf32> to tensor<16xf32>
    %1 = nv_tensor_ir.load %input tile_indices [%c0], alignment 4, padding <neg_zero>
        : memref<32xf32> to tensor<16xf32>
    %2 = nv_tensor_ir.load %input tile_indices [%c0], alignment 4, padding <nan>
        : memref<32xf32> to tensor<16xf32>
    %3 = nv_tensor_ir.load %input tile_indices [%c0], alignment 4, padding <pos_inf>
        : memref<32xf32> to tensor<16xf32>
    %4 = nv_tensor_ir.load %input tile_indices [%c0], alignment 4, padding <neg_inf>
        : memref<32xf32> to tensor<16xf32>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_padding_values_module {
// CHECK: make_partition_view %{{.*}} : partition_view<tile=(16), padding_value = zero
// CHECK: make_partition_view %{{.*}} : partition_view<tile=(16), padding_value = neg_zero
// CHECK: make_partition_view %{{.*}} : partition_view<tile=(16), padding_value = nan
// CHECK: make_partition_view %{{.*}} : partition_view<tile=(16), padding_value = pos_inf
// CHECK: make_partition_view %{{.*}} : partition_view<tile=(16), padding_value = neg_inf

// -----

module @dynamic_module {
  gpu.module @kernels {
  func.func @dynamic(
      %input: memref<?x128xf32, strided<[?, 1]>>,
      %output: memref<?x128xf32, strided<[?, 1]>>) attributes {gpu.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %input_base, %input_offset, %input_sizes:2, %input_strides:2 =
        memref.extract_strided_metadata %input
        : memref<?x128xf32, strided<[?, 1]>>
          -> memref<f32>, index, index, index, index, index
    %access_view_0 = memref.reinterpret_cast %input to offset: [0], sizes: [%input_sizes#0, 128], strides: [%input_strides#0, 1]
        : memref<?x128xf32, strided<[?, 1]>> to memref<?x128xf32, strided<[?, 1], offset: 0>>
    %tile = nv_tensor_ir.load %access_view_0 tile_indices [%c0, %c1], alignment 4
        : memref<?x128xf32, strided<[?, 1], offset: 0>> to tensor<16x32xf32>
    %output_base, %output_offset, %output_sizes:2, %output_strides:2 =
        memref.extract_strided_metadata %output
        : memref<?x128xf32, strided<[?, 1]>>
          -> memref<f32>, index, index, index, index, index
    %access_view_1 = memref.reinterpret_cast %output to offset: [0], sizes: [%output_sizes#0, 128], strides: [%output_strides#0, 1]
        : memref<?x128xf32, strided<[?, 1]>> to memref<?x128xf32, strided<[?, 1], offset: 0>>
    nv_tensor_ir.store %tile, %access_view_1 tile_indices [%c0, %c1], alignment 4
        : tensor<16x32xf32>, memref<?x128xf32, strided<[?, 1], offset: 0>>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_dynamic_module {
// CHECK: entry @dynamic(
// CHECK-SAME: %[[INPUT:[^:]+]]: tile<ptr<f32>>, %[[M:[^:]+]]: tile<i32>, %[[INPUT_STRIDE:[^:]+]]: tile<i32>
// CHECK-SAME: %[[OUTPUT:[^:]+]]: tile<ptr<f32>>, %[[N:[^:]+]]: tile<i32>, %[[OUTPUT_STRIDE:[^:]+]]: tile<i32>)
// CHECK: make_tensor_view %[[INPUT]], shape = [%[[M]], 128], strides = [%[[INPUT_STRIDE]], 1]
// CHECK: make_tensor_view %[[OUTPUT]], shape = [%[[N]], 128], strides = [%[[OUTPUT_STRIDE]], 1]

// -----

module @broadcast_view_module {
  gpu.module @kernels {
    func.func @broadcast_view(%input: memref<16xf32>,
        %output: memref<16xf32>) attributes {gpu.kernel} {
      %input_view = memref.reinterpret_cast %input to offset: [0], sizes: [16, 1], strides: [1, 2]
          : memref<16xf32> to memref<16x1xf32, strided<[1, 2]>>
      %loaded = nv_tensor_ir.load %input_view tile_indices [1, 0], alignment 4
          : memref<16x1xf32, strided<[1, 2]>> to tensor<16x1xf32>
      %tile = nv_tensor_ir.broadcast %loaded : tensor<16x1xf32> -> tensor<16x8xf32>
      %slice = nv_tensor_ir.slice %tile starts = [0, 0] limits = [16, 1] strides = [1, 1]
          : tensor<16x8xf32> -> tensor<16x1xf32>
      %output_view = memref.reinterpret_cast %output to offset: [0], sizes: [16, 1], strides: [1, 2]
          : memref<16xf32> to memref<16x1xf32, strided<[1, 2]>>
      nv_tensor_ir.store %slice, %output_view tile_indices [1, 0], alignment 4
          : tensor<16x1xf32>, memref<16x1xf32, strided<[1, 2]>>
      return
    }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_broadcast_view_module
// CHECK: %[[INPUT_VIEW:.*]] = make_tensor_view %{{.*}}, shape = [16, 1], strides = [1, 2]
// CHECK: %[[INPUT_PARTITION:.*]] = make_partition_view %[[INPUT_VIEW]]
// CHECK: %[[LOADED:.*]], %{{.*}} = load_view_tko weak %[[INPUT_PARTITION]]
// CHECK-SAME: -> tile<16x1xf32>
// CHECK: %[[BROADCAST:.*]] = broadcast %[[LOADED]] : tile<16x1xf32> -> tile<16x8xf32>
// CHECK: %[[EXTRACTED:.*]] = extract %[[BROADCAST]]{{.*}} : tile<16x8xf32> -> tile<16x1xf32>
// CHECK: %[[OUTPUT_VIEW:.*]] = make_tensor_view %{{.*}}, shape = [16, 1], strides = [1, 2]
// CHECK: %[[OUTPUT_PARTITION:.*]] = make_partition_view %[[OUTPUT_VIEW]]
// CHECK: store_view_tko weak %[[EXTRACTED]], %[[OUTPUT_PARTITION]]

// -----

// F4 accesses require exactly one unit stride to identify the packing dimension. Unit broadcast dimensions use stride 2 for both load and store.
module @f4_broadcast_view_module {
  gpu.module @kernels {
    func.func @f4_broadcast_view(%input: memref<16xf4E2M1FN>,
        %output: memref<16xf4E2M1FN>) attributes {gpu.kernel} {
      %input_view = memref.reinterpret_cast %input to offset: [0], sizes: [16, 1, 1], strides: [1, 2, 2]
          : memref<16xf4E2M1FN> to memref<16x1x1xf4E2M1FN, strided<[1, 2, 2]>>
      %loaded = nv_tensor_ir.load %input_view tile_indices [1, 0, 0], alignment 16
          : memref<16x1x1xf4E2M1FN, strided<[1, 2, 2]>> to tensor<16x1x1xf4E2M1FN>
      %tile = nv_tensor_ir.broadcast %loaded : tensor<16x1x1xf4E2M1FN> -> tensor<16x8x4xf4E2M1FN>
      %slice = nv_tensor_ir.slice %tile starts = [0, 0, 0] limits = [16, 1, 1] strides = [1, 1, 1]
          : tensor<16x8x4xf4E2M1FN> -> tensor<16x1x1xf4E2M1FN>
      %output_view = memref.reinterpret_cast %output to offset: [0], sizes: [16, 1, 1], strides: [1, 2, 2]
          : memref<16xf4E2M1FN> to memref<16x1x1xf4E2M1FN, strided<[1, 2, 2]>>
      nv_tensor_ir.store %slice, %output_view tile_indices [1, 0, 0], alignment 16
          : tensor<16x1x1xf4E2M1FN>, memref<16x1x1xf4E2M1FN, strided<[1, 2, 2]>>
      return
    }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_f4_broadcast_view_module
// CHECK: entry @f4_broadcast_view(%[[INPUT:.*]]: tile<ptr<f4E2M1FN>>, %[[OUTPUT:.*]]: tile<ptr<f4E2M1FN>>)
// CHECK: %[[INPUT_ALIGNED:.*]] = assume div_by<16>, %[[INPUT]] : tile<ptr<f4E2M1FN>>
// CHECK: %[[INPUT_VIEW:.*]] = make_tensor_view %[[INPUT_ALIGNED]], shape = [16, 1, 1], strides = [1, 2, 2]
// CHECK: %[[INPUT_PARTITION:.*]] = make_partition_view %[[INPUT_VIEW]]
// CHECK: %[[LOADED:.*]], %{{.*}} = load_view_tko weak %[[INPUT_PARTITION]]
// CHECK-SAME: -> tile<16x1x1xf4E2M1FN>
// CHECK: %[[BROADCAST:.*]] = broadcast %[[LOADED]] : tile<16x1x1xf4E2M1FN> -> tile<16x8x4xf4E2M1FN>
// CHECK: %[[EXTRACTED:.*]] = extract %[[BROADCAST]]{{.*}} : tile<16x8x4xf4E2M1FN> -> tile<16x1x1xf4E2M1FN>
// CHECK: %[[OUTPUT_ALIGNED:.*]] = assume div_by<16>, %[[OUTPUT]] : tile<ptr<f4E2M1FN>>
// CHECK: %[[OUTPUT_VIEW:.*]] = make_tensor_view %[[OUTPUT_ALIGNED]], shape = [16, 1, 1], strides = [1, 2, 2]
// CHECK: %[[OUTPUT_PARTITION:.*]] = make_partition_view %[[OUTPUT_VIEW]]
// CHECK: store_view_tko weak %[[EXTRACTED]], %[[OUTPUT_PARTITION]]

// -----

module @dynamic_broadcast_view_module {
  gpu.module @kernels {
    func.func @dynamic_broadcast_view(%input: memref<?x16xf32, strided<[16, 1]>>,
        %output: memref<4x16xf32>) attributes {gpu.kernel} {
      %view = memref.reinterpret_cast %input to offset: [0], sizes: [1, 16], strides: [2, 1]
          : memref<?x16xf32, strided<[16, 1]>> to memref<1x16xf32, strided<[2, 1]>>
      %loaded = nv_tensor_ir.load %view tile_indices [0, 0], alignment 4
          : memref<1x16xf32, strided<[2, 1]>> to tensor<1x16xf32>
      %tile = nv_tensor_ir.broadcast %loaded : tensor<1x16xf32> -> tensor<4x16xf32>
      nv_tensor_ir.store %tile, %output tile_indices [0, 0], alignment 4
          : tensor<4x16xf32>, memref<4x16xf32>
      return
    }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_dynamic_broadcast_view_module
// CHECK: %[[INPUT_VIEW:.*]] = make_tensor_view %{{.*}}, shape = [1, 16], strides = [2, 1]
// CHECK: %[[INPUT_PARTITION:.*]] = make_partition_view %[[INPUT_VIEW]]
// CHECK: %[[INPUT_TILE:.*]], %{{.*}} = load_view_tko weak %[[INPUT_PARTITION]]
// CHECK-SAME: -> tile<1x16xf32>
// CHECK: %[[BROADCAST:.*]] = broadcast %[[INPUT_TILE]] : tile<1x16xf32> -> tile<4x16xf32>
// CHECK: store_view_tko weak %[[BROADCAST]]

// -----

module @control_module {
  gpu.module @kernels {
  func.func @control(%limit: index, %condition: i1) attributes {gpu.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %grid = gpu.grid_dim x
    %upper = arith.addi %limit, %grid : index
    %result = scf.for %iv = %c0 to %upper step %c1
        iter_args(%accumulator = %c0) -> index {
      %in_range = arith.cmpi ult, %iv, %limit : index
      %execute = arith.select %condition, %in_range, %condition : i1
      %selected = scf.if %execute -> index {
        scf.yield %iv : index
      } else {
        scf.yield %limit : index
      }
      %next = arith.addi %accumulator, %selected : index
      scf.yield %next : index
    }
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_control_module {
// CHECK: entry @control(%[[LIMIT:.*]]: tile<i32>, %[[CONDITION:.*]]: tile<i1>)
// CHECK: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK: %[[ONE:.*]] = constant <i32: 1> : tile<i32>
// CHECK: %[[GRID:.*]], %{{.*}}, %{{.*}} = get_num_tile_blocks : tile<i32>
// CHECK: %[[UPPER:.*]] = addi %[[LIMIT]], %[[GRID]] : tile<i32>
// CHECK: %[[RESULT:.*]] = for %[[IV:.*]] in (%[[ZERO]] to %[[UPPER]], step %[[ONE]]) : tile<i32>
// CHECK-SAME: iter_values(%[[ACC:.*]] = %[[ZERO]]) -> (tile<i32>) {
// CHECK: %[[IN_RANGE:.*]] = cmpi less_than %[[IV]], %[[LIMIT]], unsigned
// CHECK: %[[EXECUTE:.*]] = select %[[CONDITION]], %[[IN_RANGE]], %[[CONDITION]]
// CHECK: %[[SELECTED:.*]] = if %[[EXECUTE]] -> (tile<i32>) {
// CHECK: yield %[[IV]] : tile<i32>
// CHECK: } else {
// CHECK: yield %[[LIMIT]] : tile<i32>
// CHECK: }
// CHECK: %[[NEXT:.*]] = addi %[[ACC]], %[[SELECTED]] : tile<i32>
// CHECK: continue %[[NEXT]] : tile<i32>
// CHECK: }
// CHECK: return
