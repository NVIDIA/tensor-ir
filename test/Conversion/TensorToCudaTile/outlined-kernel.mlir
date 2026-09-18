// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation" -split-input-file | FileCheck %s
// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation uniform-signature=true" -split-input-file | FileCheck %s --check-prefix=UNIFORM

module @copy_module {
  gpu.module @kernels {
  func.func @copy(
      %input: memref<65xf32, #ptr.generic_space>,
      %output: memref<65xf32, #ptr.generic_space>) attributes {gpu.kernel} {
    %block = gpu.block_id x
    %input_ptr = ptr.to_ptr %input
        : memref<65xf32, #ptr.generic_space>
          -> !ptr.ptr<#ptr.generic_space>
    %tile = nv_tensor_ir.load %input_ptr[%block]
        view offset: [0], sizes: [65], strides: [1], alignment: 16 padding = <zero>
        : !ptr.ptr<#ptr.generic_space> to tensor<32xf32>
    %output_ptr = ptr.to_ptr %output
        : memref<65xf32, #ptr.generic_space>
          -> !ptr.ptr<#ptr.generic_space>
    nv_tensor_ir.store %tile, %output_ptr[%block]
        view offset: [0], sizes: [65], strides: [1], alignment: 16
        : tensor<32xf32>, !ptr.ptr<#ptr.generic_space>
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

// UNIFORM-LABEL: entry @copy(
// UNIFORM-SAME: %{{[^:]+}}: tile<ptr<f32>>, %{{[^:]+}}: tile<i32>, %{{[^:]+}}: tile<i32>
// UNIFORM-SAME: %{{[^:]+}}: tile<ptr<f32>>, %{{[^:]+}}: tile<i32>, %{{[^:]+}}: tile<i32>

// -----

module @padding_values_module {
  gpu.module @kernels {
  func.func @padding_values(
      %input: memref<32xf32, #ptr.generic_space>) attributes {gpu.kernel} {
    %c0 = arith.constant 0 : index
    %ptr = ptr.to_ptr %input : memref<32xf32, #ptr.generic_space>
        -> !ptr.ptr<#ptr.generic_space>
    %0 = nv_tensor_ir.load %ptr[%c0]
        view offset: [0], sizes: [32], strides: [1], alignment: 4 padding = <zero>
        : !ptr.ptr<#ptr.generic_space> to tensor<16xf32>
    %1 = nv_tensor_ir.load %ptr[%c0]
        view offset: [0], sizes: [32], strides: [1], alignment: 4 padding = <neg_zero>
        : !ptr.ptr<#ptr.generic_space> to tensor<16xf32>
    %2 = nv_tensor_ir.load %ptr[%c0]
        view offset: [0], sizes: [32], strides: [1], alignment: 4 padding = <nan>
        : !ptr.ptr<#ptr.generic_space> to tensor<16xf32>
    %3 = nv_tensor_ir.load %ptr[%c0]
        view offset: [0], sizes: [32], strides: [1], alignment: 4 padding = <pos_inf>
        : !ptr.ptr<#ptr.generic_space> to tensor<16xf32>
    %4 = nv_tensor_ir.load %ptr[%c0]
        view offset: [0], sizes: [32], strides: [1], alignment: 4 padding = <neg_inf>
        : !ptr.ptr<#ptr.generic_space> to tensor<16xf32>
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
      %input: memref<?x128xf32, strided<[?, 1], offset: ?>, #ptr.generic_space>,
      %output: memref<?x128xf32, strided<[?, 1], offset: ?>, #ptr.generic_space>) attributes {gpu.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %input_base, %input_offset, %input_sizes:2, %input_strides:2 =
        memref.extract_strided_metadata %input
        : memref<?x128xf32, strided<[?, 1], offset: ?>, #ptr.generic_space>
          -> memref<f32, #ptr.generic_space>, index, index, index, index, index
    %input_ptr = ptr.to_ptr %input
        : memref<?x128xf32, strided<[?, 1], offset: ?>, #ptr.generic_space>
          -> !ptr.ptr<#ptr.generic_space>
    %tile = nv_tensor_ir.load %input_ptr[%c0, %c1]
        view offset: [0], sizes: [%input_sizes#0, 128],
             strides: [%input_strides#0, 1], alignment: 4
        : !ptr.ptr<#ptr.generic_space> to tensor<16x32xf32>
    %output_base, %output_offset, %output_sizes:2, %output_strides:2 =
        memref.extract_strided_metadata %output
        : memref<?x128xf32, strided<[?, 1], offset: ?>, #ptr.generic_space>
          -> memref<f32, #ptr.generic_space>, index, index, index, index, index
    %output_ptr = ptr.to_ptr %output
        : memref<?x128xf32, strided<[?, 1], offset: ?>, #ptr.generic_space>
          -> !ptr.ptr<#ptr.generic_space>
    nv_tensor_ir.store %tile, %output_ptr[%c0, %c1]
        view offset: [0], sizes: [%output_sizes#0, 128],
             strides: [%output_strides#0, 1], alignment: 4
        : tensor<16x32xf32>, !ptr.ptr<#ptr.generic_space>
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
  func.func @broadcast_view(
      %input: memref<16xf32, #ptr.generic_space>,
      %output: memref<16xf32, #ptr.generic_space>) attributes {gpu.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %input_ptr = ptr.to_ptr %input : memref<16xf32, #ptr.generic_space>
        -> !ptr.ptr<#ptr.generic_space>
    %tile = nv_tensor_ir.load %input_ptr[%c1, %c1]
        view offset: [0], sizes: [16, 8], strides: [1, 0], alignment: 4
        : !ptr.ptr<#ptr.generic_space> to tensor<16x8xf32>
    %output_ptr = ptr.to_ptr %output : memref<16xf32, #ptr.generic_space>
        -> !ptr.ptr<#ptr.generic_space>
    nv_tensor_ir.store %tile, %output_ptr[%c1, %c1]
        view offset: [0], sizes: [16, 8], strides: [1, 0], alignment: 4
        : tensor<16x8xf32>, !ptr.ptr<#ptr.generic_space>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_broadcast_view_module {
// CHECK: entry @broadcast_view
// CHECK: %[[ONE:.*]] = constant <i32: 1> : tile<i32>
// CHECK: %[[INPUT_VIEW:.*]] = make_tensor_view %{{.*}}, shape = [16, 1], strides = [1, 2]
// CHECK: %[[INPUT_PARTITION:.*]] = make_partition_view %[[INPUT_VIEW]]
// CHECK: %[[LOAD_ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK: %[[LOADED:.*]], %{{.*}} = load_view_tko weak %[[INPUT_PARTITION]][%[[ONE]], %[[LOAD_ZERO]]]
// CHECK-SAME: -> tile<16x1xf32>
// CHECK: %[[BROADCAST:.*]] = broadcast %[[LOADED]] : tile<16x1xf32> -> tile<16x8xf32>
// CHECK: %[[OUTPUT_VIEW:.*]] = make_tensor_view %{{.*}}, shape = [16, 1], strides = [1, 2]
// CHECK: %[[OUTPUT_PARTITION:.*]] = make_partition_view %[[OUTPUT_VIEW]]
// CHECK: %[[STORE_ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK: %[[EXTRACTED:.*]] = extract %[[BROADCAST]][%[[STORE_ZERO]], %[[STORE_ZERO]]] : tile<16x8xf32> -> tile<16x1xf32>
// CHECK: store_view_tko weak %[[EXTRACTED]], %[[OUTPUT_PARTITION]][%[[ONE]], %[[STORE_ZERO]]]

// -----

// F4 view accesses require exactly one unit stride to identify the packing
// dimension. Collapsing two broadcast dimensions must therefore preserve the
// first dimension as the only unit-stride dimension for both load and store.
module @f4_broadcast_view_module {
  gpu.module @kernels {
  func.func @f4_broadcast_view(
      %input: memref<16xf4E2M1FN, #ptr.generic_space>,
      %output: memref<16xf4E2M1FN, #ptr.generic_space>) attributes {gpu.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %input_ptr = ptr.to_ptr %input
        : memref<16xf4E2M1FN, #ptr.generic_space>
          -> !ptr.ptr<#ptr.generic_space>
    %tile = nv_tensor_ir.load %input_ptr[%c1, %c1, %c1]
        view offset: [0], sizes: [16, 8, 4], strides: [1, 0, 0], alignment: 16
        : !ptr.ptr<#ptr.generic_space> to tensor<16x8x4xf4E2M1FN>
    %output_ptr = ptr.to_ptr %output
        : memref<16xf4E2M1FN, #ptr.generic_space>
          -> !ptr.ptr<#ptr.generic_space>
    nv_tensor_ir.store %tile, %output_ptr[%c1, %c1, %c1]
        view offset: [0], sizes: [16, 8, 4], strides: [1, 0, 0], alignment: 16
        : tensor<16x8x4xf4E2M1FN>, !ptr.ptr<#ptr.generic_space>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_f4_broadcast_view_module {
// CHECK: entry @f4_broadcast_view(%[[INPUT:.*]]: tile<ptr<f4E2M1FN>>, %[[OUTPUT:.*]]: tile<ptr<f4E2M1FN>>)
// CHECK: %[[INPUT_ALIGNED:.*]] = assume div_by<16>, %[[INPUT]] : tile<ptr<f4E2M1FN>>
// CHECK: %[[INPUT_VIEW:.*]] = make_tensor_view %[[INPUT_ALIGNED]], shape = [16, 1, 1], strides = [1, 2, 2]
// CHECK: %[[INPUT_PARTITION:.*]] = make_partition_view %[[INPUT_VIEW]]
// CHECK: %[[LOADED:.*]], %{{.*}} = load_view_tko weak %[[INPUT_PARTITION]]
// CHECK-SAME: -> tile<16x1x1xf4E2M1FN>
// CHECK: %[[BROADCAST:.*]] = broadcast %[[LOADED]] : tile<16x1x1xf4E2M1FN> -> tile<16x8x4xf4E2M1FN>
// CHECK: %[[OUTPUT_ALIGNED:.*]] = assume div_by<16>, %[[OUTPUT]] : tile<ptr<f4E2M1FN>>
// CHECK: %[[OUTPUT_VIEW:.*]] = make_tensor_view %[[OUTPUT_ALIGNED]], shape = [16, 1, 1], strides = [1, 2, 2]
// CHECK: %[[OUTPUT_PARTITION:.*]] = make_partition_view %[[OUTPUT_VIEW]]
// CHECK: %[[EXTRACTED:.*]] = extract %[[BROADCAST]]{{.*}} : tile<16x8x4xf4E2M1FN> -> tile<16x1x1xf4E2M1FN>
// CHECK: store_view_tko weak %[[EXTRACTED]], %[[OUTPUT_PARTITION]]

// -----

module @dynamic_broadcast_view_module {
  gpu.module @kernels {
  func.func @dynamic_broadcast_view(
      %input: memref<?x16xf32, strided<[16, 1]>, #ptr.generic_space>,
      %output: memref<4x16xf32, #ptr.generic_space>) attributes {gpu.kernel} {
    %c0 = arith.constant 0 : index
    %input_base, %input_offset, %input_sizes:2, %input_strides:2 =
        memref.extract_strided_metadata %input
        : memref<?x16xf32, strided<[16, 1]>, #ptr.generic_space>
          -> memref<f32, #ptr.generic_space>, index, index, index, index, index
    %input_ptr = ptr.to_ptr %input
        : memref<?x16xf32, strided<[16, 1]>, #ptr.generic_space>
          -> !ptr.ptr<#ptr.generic_space>
    %tile = nv_tensor_ir.load %input_ptr[%c0, %c0]
        view offset: [0], sizes: [%input_sizes#0, 16], strides: [0, 1], alignment: 4
        : !ptr.ptr<#ptr.generic_space> to tensor<4x16xf32>
    %output_ptr = ptr.to_ptr %output
        : memref<4x16xf32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
    nv_tensor_ir.store %tile, %output_ptr[%c0, %c0]
        view offset: [0], sizes: [4, 16], strides: [16, 1], alignment: 4
        : tensor<4x16xf32>, !ptr.ptr<#ptr.generic_space>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_dynamic_broadcast_view_module {
// CHECK: entry @dynamic_broadcast_view
// CHECK: %[[INPUT_VIEW:.*]] = make_tensor_view %{{.*}}, shape = [1, 16], strides = [2, 1] : tensor_view<1x16xf32, strides=[2,1]>
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
