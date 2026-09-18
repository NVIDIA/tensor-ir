// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,layout-propagation-annotation,layout-propagation-normalization,tile-selection,graph-splitting),tir-bufferize,func.func(tir-form-grid))' --split-input-file | FileCheck %s --implicit-check-not='layout =' --implicit-check-not='iteration_space' --implicit-check-not='tile_size' --implicit-check-not='result_views' --implicit-check-not='result_layouts' --implicit-check-not='nv_tensor_ir.iter_space'

// CHECK-LABEL: func.func @pointwise_dispatch(
//  CHECK-SAME: %[[LHS:[^:]+]]: memref<32x64xf32, strided<[64, 1]>, #ptr.generic_space>
//  CHECK-SAME: %[[RHS:[^:]+]]: memref<32x64xf32, strided<[64, 1]>, #ptr.generic_space>
//  CHECK-SAME: %[[OUT:[^:]+]]: memref<32x64xf32, strided<[64, 1]>, #ptr.generic_space>
//       CHECK: %[[LPTR:.*]] = ptr.to_ptr %[[LHS]]
//       CHECK: %[[RPTR:.*]] = ptr.to_ptr %[[RHS]]
//       CHECK: %[[OPTR:.*]] = ptr.to_ptr %[[OUT]]
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (32)
//       CHECK: %[[LTILE:.*]] = nv_tensor_ir.load %[[LPTR]][%[[LINEAR]]] view offset: [0], sizes: [2048], strides: [1], alignment: 4
//       CHECK: %[[RTILE:.*]] = nv_tensor_ir.load %[[RPTR]][%[[LINEAR]]] view offset: [0], sizes: [2048], strides: [1], alignment: 4
//       CHECK: %[[SUM:.*]] = nv_tensor_ir.add %[[LTILE]], %[[RTILE]] : tensor<64xf32>
//       CHECK: nv_tensor_ir.store %[[SUM]], %[[OPTR]][%[[LINEAR]]] view offset: [0], sizes: [2048], strides: [1], alignment: 4
//       CHECK: } {mapping = [#gpu.block<x>]}
nv_tensor_ir.graph @pointwise(
    %lhs: tensor<32x64xf32>, %rhs: tensor<32x64xf32>)
    -> tensor<32x64xf32> attributes {tile_size = array<i32: 64>} {
  %sum = add %lhs, %rhs : tensor<32x64xf32>
  results %sum : tensor<32x64xf32>
}

// -----

// A statically unit outer tile count maps directly to coordinate zero. Grid
// formation must not emit remainder or division by one for that dimension.
// CHECK-LABEL: func.func @unit_outer_tile_count_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (4) {
//       CHECK: %[[ZERO:.*]] = arith.constant 0 : index
//   CHECK-NOT: arith.remui
//   CHECK-NOT: arith.divui
//       CHECK: nv_tensor_ir.load {{.*}}[0, %[[LINEAR]]]
//       CHECK: nv_tensor_ir.store {{.*}}[0, %[[LINEAR]]]
nv_tensor_ir.graph @unit_outer_tile_count(
    %input: tensor<4x64xf32> {nv_tensor_ir.stride = "(128,1)"})
    -> (tensor<4x64xf32> {nv_tensor_ir.stride = "(128,1)"})
    attributes {tile_size = array<i32: 4, 16>} {
  %result = cos %input : tensor<4x64xf32>
  results %result : tensor<4x64xf32>
}

// -----

// Dynamic extents and explicit dynamic strides remain descriptor values in
// the staged memref ABI. The output extent determines the runtime grid bound.
// CHECK-LABEL: func.func @dynamic_dispatch(
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space>
//  CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space>
//       CHECK: %{{.*}}, %{{.*}}, %[[ISIZES:.*]]:2, %[[ISTRIDES:.*]]:2 = memref.extract_strided_metadata %[[INPUT]]
//       CHECK: %[[IPTR:.*]] = ptr.to_ptr %[[INPUT]]
//       CHECK: %{{.*}}, %{{.*}}, %[[OSIZES:.*]]:2, %[[OSTRIDES:.*]]:2 = memref.extract_strided_metadata %[[OUTPUT]]
//       CHECK: %[[OPTR:.*]] = ptr.to_ptr %[[OUTPUT]]
//       CHECK: arith.addi %[[OSIZES]]#0
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (%{{.*}})
//       CHECK: nv_tensor_ir.load %[[IPTR]][%{{.*}}, %{{.*}}] view offset: [0], sizes: [%[[ISIZES]]#0, 32], strides: [%[[ISTRIDES]]#0, 1], alignment: 4
//       CHECK: nv_tensor_ir.cos
//       CHECK: nv_tensor_ir.store {{.*}}, %[[OPTR]][%{{.*}}, %{{.*}}] view offset: [0], sizes: [%[[OSIZES]]#0, 32], strides: [%[[OSTRIDES]]#0, 1], alignment: 4
//       CHECK: } {mapping = [#gpu.block<x>]}
nv_tensor_ir.graph @dynamic(
    %input: tensor<?x32xf32> {nv_tensor_ir.stride = "(?,1)"})
    -> (tensor<?x32xf32> {nv_tensor_ir.stride = "(?,1)"})
    attributes {tile_size = array<i32: 4, 32>} {
  %result = cos %input : tensor<?x32xf32>
  results %result : tensor<?x32xf32>
}

// -----

// The layout chain is represented by the logical input view. None of the
// layout-only TensorIR operations survive into the tiled computation.
// CHECK-LABEL: func.func @layout_chain_dispatch(
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<1024xf32, #ptr.generic_space>
//       CHECK: %[[IPTR:.*]] = ptr.to_ptr %[[INPUT]]
//       CHECK: scf.forall
//       CHECK: nv_tensor_ir.load %[[IPTR]][%{{.*}}, %{{.*}}] view offset: [768], sizes: [16, 16], strides: [1, 16], alignment: 4
//   CHECK-NOT: nv_tensor_ir.slice
//   CHECK-NOT: nv_tensor_ir.reshape
//   CHECK-NOT: nv_tensor_ir.transpose
//       CHECK: nv_tensor_ir.store
nv_tensor_ir.graph @layout_chain(%input: tensor<1024xf32>)
    -> tensor<16x16xf32> attributes {tile_size = array<i32: 8, 8>} {
  %slice = slice %input starts = [768] limits = [1024] strides = [1]
      : tensor<1024xf32> -> tensor<256xf32>
  %reshape = reshape %slice : tensor<256xf32> -> tensor<16x16xf32>
  %transpose = transpose %reshape permutation = [1, 0]
      : tensor<16x16xf32> -> tensor<16x16xf32>
  results %transpose : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @multi_output_dispatch(
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<64xf32, #ptr.generic_space>
//  CHECK-SAME: %[[SUM_OUT:[^:]+]]: memref<64xf32, #ptr.generic_space>
//  CHECK-SAME: %[[PROD_OUT:[^:]+]]: memref<64xf32, #ptr.generic_space>
//       CHECK: %[[TILE:.*]] = nv_tensor_ir.load
//       CHECK: %[[SUM:.*]] = nv_tensor_ir.add %[[TILE]], %[[TILE]]
//       CHECK: nv_tensor_ir.store %[[SUM]]
//       CHECK: %[[PROD:.*]] = nv_tensor_ir.mul %[[TILE]], %[[TILE]]
//       CHECK: nv_tensor_ir.store %[[PROD]]
nv_tensor_ir.graph @multi_output(%input: tensor<64xf32>)
    -> (tensor<64xf32>, tensor<64xf32>)
    attributes {tile_size = array<i32: 16>} {
  %sum = add %input, %input : tensor<64xf32>
  %product = mul %input, %input : tensor<64xf32>
  results %sum, %product : tensor<64xf32>, tensor<64xf32>
}

// -----

// Floating-point iotas use an unsigned integer tile until the final uitofp.
// CHECK-LABEL: func.func @float_iota_dispatch(
//  CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<128xf32, #ptr.generic_space>
//       CHECK: %[[OUTPUT_PTR:.*]] = ptr.to_ptr %[[OUTPUT]]
//       CHECK: scf.forall
//       CHECK: %[[IOTA:.*]] = nv_tensor_ir.iota dimension = 0 : tensor<64xui32>
//       CHECK: %[[OFFSET:.*]] = builtin.unrealized_conversion_cast %{{.*}} : i32 to ui32
//       CHECK: %[[SPLAT:.*]] = nv_tensor_ir.splat %[[OFFSET]] : tensor<64xui32>
//       CHECK: %[[INDEX:.*]] = nv_tensor_ir.add %[[IOTA]], %[[SPLAT]] : tensor<64xui32>
//       CHECK: %[[SIGNLESS:.*]] = builtin.unrealized_conversion_cast %[[INDEX]]
//  CHECK-SAME: : tensor<64xui32> to tensor<64xi32>
//       CHECK: %[[FLOAT:.*]] = arith.uitofp %[[SIGNLESS]]
//  CHECK-SAME: : tensor<64xi32> to tensor<64xf32>
//       CHECK: nv_tensor_ir.store %[[FLOAT]], %[[OUTPUT_PTR]]
nv_tensor_ir.graph @float_iota() -> tensor<128xf32>
    attributes {tile_size = array<i32: 64>} {
  %result = iota dimension = 0 : tensor<128xf32>
  results %result : tensor<128xf32>
}

// -----

// An input used by both the root tile and a reduction tile must be loaded at
// both requested shapes. The reduction's single operand retains its composite
// pointwise source as one layout instead of treating each leaf as an operand.
// CHECK-LABEL: func.func @reduction_tile_cache_dispatch(
//       CHECK: %[[ROOT_TILE:.*]] = nv_tensor_ir.load
//  CHECK-SAME: to tensor<4x4xf32>
//       CHECK: %[[REDUCE_TILE:.*]] = nv_tensor_ir.load
//  CHECK-SAME: to tensor<4x128xf32>
//       CHECK: %[[SQUARED:.*]] = nv_tensor_ir.mul %[[REDUCE_TILE]], %[[REDUCE_TILE]] : tensor<4x128xf32>
//       CHECK: %[[SUM:.*]] = nv_tensor_ir.reduce(%[[SQUARED]])
//  CHECK-SAME: tensor<4x128xf32> -> tensor<4x1xf32>
//       CHECK: %[[RESULT:.*]] = nv_tensor_ir.mul %[[ROOT_TILE]], {{.*}} : tensor<4x4xf32>
//       CHECK: nv_tensor_ir.store %[[RESULT]]
nv_tensor_ir.graph @reduction_tile_cache(%input: tensor<64x128xf32>)
    -> tensor<64x128xf32> attributes {tile_size = array<i32: 4, 4>} {
  %squared = mul %input, %input : tensor<64x128xf32>
  %sum = reduce(%squared)<dimensions = [1], reduction_mode = <add>>
      : tensor<64x128xf32> -> tensor<64x1xf32>
  %count = constant dense<128.0> : tensor<64x1xf32>
  %mean = div %sum, %count : tensor<64x1xf32>
  %wide = broadcast %mean : tensor<64x1xf32> -> tensor<64x128xf32>
  %result = mul %input, %wide : tensor<64x128xf32>
  results %result : tensor<64x128xf32>
}

// -----

// A matmul dimension whose static extent fits in one output tile must be read
// at coordinate zero. The broadcast batch leaves multiple grid tiles, making
// the otherwise-unit N coordinate a runtime expression before this fold.
// CHECK-LABEL: func.func @matmul_full_tile_coordinate_dispatch(
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (2)
//       CHECK: %[[BATCH:.*]] = arith.remui %[[LINEAR]]
//       CHECK: %[[UNIT_N_COORD:.*]] = arith.divui %[[LINEAR]]
//       CHECK: nv_tensor_ir.load
//       CHECK: nv_tensor_ir.load {{.*}}[0, 0]
//       CHECK: nv_tensor_ir.matmul
//       CHECK: nv_tensor_ir.store {{.*}}[%[[BATCH]], 0, %[[UNIT_N_COORD]]]
nv_tensor_ir.graph @matmul_full_tile_coordinate(
    %lhs: tensor<1x8x32xf32> {nv_tensor_ir.stride = "(0,32,1)"},
    %rhs: tensor<1x32x16xf32> {nv_tensor_ir.stride = "(0,16,1)"})
    -> (tensor<8x8x16xf32> {nv_tensor_ir.stride = "(128,16,1)"})
    attributes {tile_size = array<i32: 4, 8, 16>} {
  %product = matmul(%lhs, %rhs)
      : (tensor<1x8x32xf32>, tensor<1x32x16xf32>) -> tensor<1x8x16xf32>
  %result = broadcast %product
      : tensor<1x8x16xf32> -> tensor<8x8x16xf32>
  results %result : tensor<8x8x16xf32>
}

// -----

// The first output projects the dynamic root dimension with zero stride, so it
// cannot provide the runtime grid extent. FormGrid must obtain that extent from
// the later, non-broadcast output.
// CHECK-LABEL: func.func @dynamic_extent_from_later_output(
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>
//  CHECK-SAME: %[[PROJECTED:[^:]+]]: memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>
//  CHECK-SAME: %[[FULL:[^:]+]]: memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>
//       CHECK: %{{.*}}, %{{.*}}, %[[FULL_SIZES:.*]]:2, %{{.*}}:2 = memref.extract_strided_metadata %[[FULL]]
//       CHECK: arith.addi %[[FULL_SIZES]]#0
//       CHECK: scf.forall
func.func @dynamic_extent_from_later_output(
    %input: memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>,
    %projected: memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>,
    %full: memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>)
    attributes {
      iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(?,32):(32,1)", [0]>,
      nv_tensor_ir.bufferized_program,
      result_layouts = [
        #nv_tensor_ir.tensor_source<0, 0, "(?,32):(32,1)", [0]>,
        #nv_tensor_ir.tensor_source<0, 0, "(?,32):(32,1)", [0]>],
      result_views = [
        #nv_tensor_ir.tensor_source<1, 0, "(?,32):(0,1)">,
        #nv_tensor_ir.tensor_source<2, 0, "(?,32):(32,1)", [0]>],
      tile_size = array<i32: 4, 32>
    } {
  %base_buffer, %offset, %sizes:2, %strides:2 =
      memref.extract_strided_metadata %input
      : memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>
      -> memref<f32, #ptr.generic_space>, index, index, index, index, index
  %input_ptr = ptr.to_ptr %input
      : memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>
      -> <#ptr.generic_space>
  %c0 = arith.constant 0 : index
  %tile = nv_tensor_ir.load %input_ptr[%c0, %c0]
      view offset: [0], sizes: [%sizes#0, 32], strides: [32, 1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<?x32xf32>
  %result = nv_tensor_ir.cos %tile {
      layout = #nv_tensor_ir.tensor_source<0, 0, "(?,32):(32,1)", [0]>,
      nv_tensor_ir.iter_space_id = 0 : i32} : tensor<?x32xf32>
  %projected_base, %projected_offset, %projected_sizes:2,
      %projected_strides:2 = memref.extract_strided_metadata %projected
      : memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>
      -> memref<f32, #ptr.generic_space>, index, index, index, index, index
  %projected_ptr = ptr.to_ptr %projected
      : memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>
      -> <#ptr.generic_space>
  nv_tensor_ir.store %result, %projected_ptr[%c0, %c0]
      view offset: [0], sizes: [%projected_sizes#0, 32],
      strides: [32, 1], alignment: 4
      : tensor<?x32xf32>, !ptr.ptr<#ptr.generic_space>
  %full_base, %full_offset, %full_sizes:2, %full_strides:2 =
      memref.extract_strided_metadata %full
      : memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>
      -> memref<f32, #ptr.generic_space>, index, index, index, index, index
  %full_ptr = ptr.to_ptr %full
      : memref<?x32xf32, strided<[32, 1]>, #ptr.generic_space>
      -> <#ptr.generic_space>
  nv_tensor_ir.store %result, %full_ptr[%c0, %c0]
      view offset: [0], sizes: [%full_sizes#0, 32], strides: [32, 1],
      alignment: 4 : tensor<?x32xf32>, !ptr.ptr<#ptr.generic_space>
  return
}
