// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid))' --split-input-file | FileCheck %s --implicit-check-not='layout =' --implicit-check-not='iteration_space' --implicit-check-not='tile_size' --implicit-check-not='result_views' --implicit-check-not='result_layouts' --implicit-check-not='nv_tensor_ir.iter_space' --implicit-check-not='bufferization.' --implicit-check-not='ptr.to_ptr'

// CHECK-LABEL: func.func @pointwise_dispatch(
//  CHECK-SAME: %[[LHS:[^:]+]]: memref<32x64xf32, strided<[64, 1]>>
//  CHECK-SAME: %[[RHS:[^:]+]]: memref<32x64xf32, strided<[64, 1]>>
//  CHECK-SAME: %[[OUT:[^:]+]]: memref<32x64xf32, strided<[64, 1]>>
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (32)
//       CHECK: %[[VIEW0:.*]] = memref.reinterpret_cast %[[LHS]] to offset: [0], sizes: [2048], strides: [1]
//       CHECK: %[[LTILE:.*]] = nv_tensor_ir.load %[[VIEW0]] tile_indices [%[[LINEAR]]], alignment 4
//       CHECK: %[[VIEW1:.*]] = memref.reinterpret_cast %[[RHS]] to offset: [0], sizes: [2048], strides: [1]
//       CHECK: %[[RTILE:.*]] = nv_tensor_ir.load %[[VIEW1]] tile_indices [%[[LINEAR]]], alignment 4
//       CHECK: %[[SUM:.*]] = nv_tensor_ir.add %[[LTILE]], %[[RTILE]] : tensor<64xf32>
//       CHECK: %[[VIEW2:.*]] = memref.reinterpret_cast %[[OUT]] to offset: [0], sizes: [2048], strides: [1]
//       CHECK: nv_tensor_ir.store %[[SUM]], %[[VIEW2]] tile_indices [%[[LINEAR]]], alignment 4
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
//       CHECK: nv_tensor_ir.load {{.*}} tile_indices [0, %[[LINEAR]]]
//       CHECK: nv_tensor_ir.store {{.*}} tile_indices [0, %[[LINEAR]]]
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
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<?x32xf32, strided<[?, 1]>>
//  CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<?x32xf32, strided<[?, 1]>>
//   CHECK-NOT: memref.extract_strided_metadata
//       CHECK: %[[SIZE:.*]] = memref.dim %[[OUTPUT]]
//       CHECK: arith.ceildivui %[[SIZE]]
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (%{{.*}})
//       CHECK: nv_tensor_ir.load %[[INPUT]] tile_indices [%[[LINEAR]], 0], alignment 4
//       CHECK: nv_tensor_ir.cos
//       CHECK: nv_tensor_ir.store {{.*}}, %[[OUTPUT]] tile_indices [%[[LINEAR]], 0], alignment 4
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
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<1024xf32>
//       CHECK: scf.forall
//       CHECK: %[[VIEW3:.*]] = memref.reinterpret_cast %[[INPUT]] to offset: [768], sizes: [16, 16], strides: [1, 16]
//       CHECK: nv_tensor_ir.load %[[VIEW3]] tile_indices [%{{.*}}, %{{.*}}], alignment 4
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
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<64xf32>
//  CHECK-SAME: %[[SUM_OUT:[^:]+]]: memref<64xf32>
//  CHECK-SAME: %[[PROD_OUT:[^:]+]]: memref<64xf32>
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
//  CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<128xf32>
//       CHECK: scf.forall
//       CHECK: %[[IOTA:.*]] = nv_tensor_ir.iota dimension = 0 : tensor<64xui32>
//       CHECK: %[[OFFSET:.*]] = builtin.unrealized_conversion_cast %{{.*}} : i32 to ui32
//       CHECK: %[[SPLAT:.*]] = nv_tensor_ir.splat %[[OFFSET]] : tensor<64xui32>
//       CHECK: %[[INDEX:.*]] = nv_tensor_ir.add %[[IOTA]], %[[SPLAT]] : tensor<64xui32>
//       CHECK: %[[SIGNLESS:.*]] = builtin.unrealized_conversion_cast %[[INDEX]]
//  CHECK-SAME: : tensor<64xui32> to tensor<64xi32>
//       CHECK: %[[FLOAT:.*]] = arith.uitofp %[[SIGNLESS]]
//  CHECK-SAME: : tensor<64xi32> to tensor<64xf32>
//       CHECK: nv_tensor_ir.store %[[FLOAT]], %[[OUTPUT]]
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
// at coordinate zero, even when the broadcast batch leaves multiple grid tiles.
// CHECK-LABEL: func.func @matmul_full_tile_coordinate_dispatch(
//       CHECK: scf.forall (%[[BATCH:.*]]) in (2)
//       CHECK: nv_tensor_ir.load
//       CHECK: nv_tensor_ir.load {{.*}} tile_indices [0, 0]
//       CHECK: nv_tensor_ir.matmul
//       CHECK: nv_tensor_ir.store {{.*}} tile_indices [%[[BATCH]], 0, 0]
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

// Load the varying dimension once, preserving leading and trailing unit dimensions with stride 2 before broadcasting the tile.
// CHECK-LABEL: func.func @leading_and_trailing(
//  CHECK-SAME:   %[[INPUT:[^:]+]]: memref<6xf32>
//  CHECK-SAME:   %[[OUTPUT:[^:]+]]: memref<4x6x8xf32, strided<[128, 16, 1]>>
func.func @leading_and_trailing(%input: memref<6xf32>,
    %output: memref<4x6x8xf32, strided<[128, 16, 1]>>) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 3, 4>
} {
  %input_tensor = bufferization.to_tensor %input restrict
      : memref<6xf32> to tensor<6xf32>
  %expanded = nv_tensor_ir.reshape %input_tensor
      : tensor<6xf32> -> tensor<1x6x1xf32>
  //       CHECK:   scf.forall (%[[BID:.*]]) in (8) {
  //       CHECK:     %[[TWO:.*]] = arith.constant 2 : index
  //       CHECK:     %[[DEPTH:.*]] = arith.remui %[[BID]], %[[TWO]] : index
  //       CHECK:     %[[REST:.*]] = arith.divui %[[BID]], %[[TWO]] : index
  //       CHECK:     %[[TWO_AGAIN:.*]] = arith.constant 2 : index
  //       CHECK:     %[[COLUMN:.*]] = arith.remui %[[REST]], %[[TWO_AGAIN]] : index
  //       CHECK:     %[[ROW:.*]] = arith.divui %[[REST]], %[[TWO_AGAIN]] : index
  //       CHECK:     %[[VIEW:.*]] = memref.reinterpret_cast %[[INPUT]] to offset: [0], sizes: [1, 6, 1], strides: [2, 1, 2]
  //       CHECK:     %[[LOAD:.*]] = nv_tensor_ir.load %[[VIEW]] tile_indices [0, %[[COLUMN]], 0]
  //  CHECK-SAME:      : memref<1x6x1xf32, strided<[2, 1, 2]>> to tensor<1x3x1xf32>
  //       CHECK:     %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[LOAD]] : tensor<1x3x1xf32> -> tensor<2x3x4xf32>
  %result = nv_tensor_ir.broadcast %expanded
      : tensor<1x6x1xf32> -> tensor<4x6x8xf32>
  //       CHECK:     nv_tensor_ir.store %[[BROADCAST]], %[[OUTPUT]] tile_indices [%[[ROW]], %[[COLUMN]], %[[DEPTH]]]
  bufferization.materialize_in_destination %result in restrict writable %output
      : (tensor<4x6x8xf32>, memref<4x6x8xf32, strided<[128, 16, 1]>>) -> ()
  return
}

// -----

// Unit broadcast dimensions use stride 2 while varying dimensions preserve their dynamic extents and strides.
// CHECK-LABEL: func.func @dynamic_broadcast(
//  CHECK-SAME:   %[[INPUT:[^:]+]]: memref<?xf32, strided<[?]>>
//  CHECK-SAME:   %[[OUTPUT:[^:]+]]: memref<4x?xf32, strided<[?, 1]>>
func.func @dynamic_broadcast(%input: memref<?xf32, strided<[?]>>,
    %output: memref<4x?xf32, strided<[?, 1]>>) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 8>
} {
  %input_tensor = bufferization.to_tensor %input restrict
      : memref<?xf32, strided<[?]>> to tensor<?xf32>
  %expanded = nv_tensor_ir.reshape %input_tensor
      : tensor<?xf32> -> tensor<1x?xf32>
  //       CHECK:   %[[COUNT:.*]] = arith.ceildivui
  //       CHECK:   scf.forall (%[[BID:.*]]) in (%{{.*}}) {
  //       CHECK:     %[[COLUMN:.*]] = arith.remui %[[BID]], %[[COUNT]] : index
  //       CHECK:     %[[ROW:.*]] = arith.divui %[[BID]], %[[COUNT]] : index
  //       CHECK:     %[[VIEW:.*]] = memref.reinterpret_cast %[[INPUT]] to offset: [0], sizes: [1, %{{.*}}], strides: [2, %{{.*}}]
  //       CHECK:     %[[LOAD:.*]] = nv_tensor_ir.load %[[VIEW]] tile_indices [0, %[[COLUMN]]]
  //  CHECK-SAME:      : memref<1x?xf32, strided<[2, ?]>> to tensor<1x8xf32>
  //       CHECK:     %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[LOAD]] : tensor<1x8xf32> -> tensor<2x8xf32>
  %result = nv_tensor_ir.broadcast %expanded
      : tensor<1x?xf32> -> tensor<4x?xf32>
  //       CHECK:     nv_tensor_ir.store %[[BROADCAST]], %[[OUTPUT]] tile_indices [%[[ROW]], %[[COLUMN]]]
  bufferization.materialize_in_destination %result in restrict writable %output
      : (tensor<4x?xf32>, memref<4x?xf32, strided<[?, 1]>>) -> ()
  return
}

// -----

// The same producer is needed in both branches and after the conditional.
// Branch-local loads, compute values, and scalar setup must not escape their
// regions. The verifier checks dominance; captures check reuse within a scope.
// CHECK-LABEL: func.func @shared_producer(
//  CHECK-SAME:   %[[INPUT:[^:]+]]: memref<1xf32>
//  CHECK-SAME:   %[[OUTPUT:[^:]+]]: memref<16xf32>
//  CHECK-SAME:   %[[SCALAR:[^:]+]]: f32
func.func @shared_producer(%input: memref<1xf32>,
    %output: memref<16xf32>, %scalar: f32) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 1, 8>
} {
  %input_tile = bufferization.to_tensor %input restrict
      : memref<1xf32> to tensor<1xf32>
  %twice = arith.addf %scalar, %scalar : f32
  %splat = nv_tensor_ir.splat %twice : tensor<1xf32>
  %shared = nv_tensor_ir.add %input_tile, %splat : tensor<1xf32>
  %half = nv_tensor_ir.broadcast %shared
      : tensor<1xf32> -> tensor<8xf32>
  //       CHECK:   scf.forall (%[[BID:.*]]) in (2) {
  //       CHECK:     %[[ONE:.*]] = arith.constant 1 : index
  //       CHECK:     %[[IS_FIRST:.*]] = arith.cmpi ult, %[[BID]], %[[ONE]] : index
  //       CHECK:     %[[BRANCH:.*]] = scf.if %[[IS_FIRST]] -> (tensor<1x8xf32>) {
  //       CHECK:       %[[THEN_VIEW:.*]] = memref.reinterpret_cast %[[INPUT]]
  //  CHECK-SAME:        to offset: [0], sizes: [1, 1], strides: [2, 2]
  //  CHECK-SAME:        : memref<1xf32> to memref<1x1xf32, strided<[2, 2]>>
  //       CHECK:       %[[THEN_LOAD:.*]] = nv_tensor_ir.load %[[THEN_VIEW]] tile_indices [0, 0]
  //       CHECK:       %[[THEN_BROADCAST:.*]] = nv_tensor_ir.broadcast %[[THEN_LOAD]] : tensor<1x1xf32> -> tensor<1x8xf32>
  //       CHECK:       %[[THEN_SCALAR:.*]] = arith.addf %[[SCALAR]], %[[SCALAR]] : f32
  //       CHECK:       %[[THEN_SPLAT:.*]] = nv_tensor_ir.splat %[[THEN_SCALAR]] : tensor<1x8xf32>
  //       CHECK:       %[[THEN_VALUE:.*]] = nv_tensor_ir.add %[[THEN_BROADCAST]], %[[THEN_SPLAT]] : tensor<1x8xf32>
  //       CHECK:       scf.yield %[[THEN_VALUE]] : tensor<1x8xf32>
  //       CHECK:     } else {
  //       CHECK:       %[[ELSE_VIEW:.*]] = memref.reinterpret_cast %[[INPUT]]
  //  CHECK-SAME:        to offset: [0], sizes: [1, 1], strides: [2, 2]
  //  CHECK-SAME:        : memref<1xf32> to memref<1x1xf32, strided<[2, 2]>>
  //       CHECK:       %[[ELSE_LOAD:.*]] = nv_tensor_ir.load %[[ELSE_VIEW]] tile_indices [0, 0]
  //       CHECK:       %[[ELSE_BROADCAST:.*]] = nv_tensor_ir.broadcast %[[ELSE_LOAD]] : tensor<1x1xf32> -> tensor<1x8xf32>
  //       CHECK:       %[[ELSE_SCALAR:.*]] = arith.addf %[[SCALAR]], %[[SCALAR]] : f32
  //       CHECK:       %[[ELSE_SPLAT:.*]] = nv_tensor_ir.splat %[[ELSE_SCALAR]] : tensor<1x8xf32>
  //       CHECK:       %[[ELSE_VALUE:.*]] = nv_tensor_ir.add %[[ELSE_BROADCAST]], %[[ELSE_SPLAT]] : tensor<1x8xf32>
  //       CHECK:       scf.yield %[[ELSE_VALUE]] : tensor<1x8xf32>
  //       CHECK:     }
  %concat = nv_tensor_ir.concatenate %half, %half dimension = 0
      : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  //       CHECK:     %[[OUTER_VIEW:.*]] = memref.reinterpret_cast %[[INPUT]]
  //  CHECK-SAME:      to offset: [0], sizes: [1, 1], strides: [2, 2]
  //  CHECK-SAME:      : memref<1xf32> to memref<1x1xf32, strided<[2, 2]>>
  //       CHECK:     %[[OUTER_LOAD:.*]] = nv_tensor_ir.load %[[OUTER_VIEW]] tile_indices [0, 0]
  //       CHECK:     %[[OUTER_BROADCAST:.*]] = nv_tensor_ir.broadcast %[[OUTER_LOAD]] : tensor<1x1xf32> -> tensor<1x8xf32>
  //       CHECK:     %[[OUTER_SCALAR:.*]] = arith.addf %[[SCALAR]], %[[SCALAR]] : f32
  //       CHECK:     %[[OUTER_SPLAT:.*]] = nv_tensor_ir.splat %[[OUTER_SCALAR]] : tensor<1x8xf32>
  //       CHECK:     %[[OUTER_VALUE:.*]] = nv_tensor_ir.add %[[OUTER_BROADCAST]], %[[OUTER_SPLAT]] : tensor<1x8xf32>
  %whole = nv_tensor_ir.broadcast %shared
      : tensor<1xf32> -> tensor<16xf32>
  //       CHECK:     %[[RESULT:.*]] = nv_tensor_ir.add %[[BRANCH]], %[[OUTER_VALUE]] : tensor<1x8xf32>
  %result = nv_tensor_ir.add %concat, %whole
      : tensor<16xf32>
  //       CHECK:     %[[OUT_VIEW:.*]] = memref.reinterpret_cast %[[OUTPUT]]
  //  CHECK-SAME:      to offset: [0], sizes: [2, 8], strides: [8, 1]
  //  CHECK-SAME:      : memref<16xf32> to memref<2x8xf32, strided<[8, 1]>>
  //       CHECK:     nv_tensor_ir.store %[[RESULT]], %[[OUT_VIEW]] tile_indices [%[[BID]], 0]
  //       CHECK:   } {mapping = [#gpu.block<x>]}
  bufferization.materialize_in_destination %result in restrict writable %output
      : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @nonunit
func.func @nonunit(%input: memref<4x6x8xf32>,
    %output: memref<4x6x8xf32, strided<[0, 2, 0]>> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.allow_tma = false}) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 3, 4>
} {
  %value = bufferization.to_tensor %input restrict : memref<4x6x8xf32> to tensor<4x6x8xf32>
  bufferization.materialize_in_destination %value in restrict writable %output
      : (tensor<4x6x8xf32>, memref<4x6x8xf32, strided<[0, 2, 0]>>) -> ()
  return
}
// CHECK: nv_tensor_ir.slice {{.*}} starts = [0, 0, 0] limits = [1, 3, 1]
// CHECK-NOT: nv_tensor_ir.reshape
// CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [1, 6, 1], strides: [2, 2, 2]
// CHECK: nv_tensor_ir.store {{.*}} tile_indices [0, %{{.*}}, 0], alignment 16 allow_tma false
// CHECK-SAME: tensor<1x3x1xf32>, memref<1x6x1xf32, strided<[2, 2, 2]>>

// -----

// CHECK-LABEL: func.func @scalar
func.func @scalar(%input: memref<4x8xf32, strided<[16, 1]>>,
    %output: memref<4x8xf32, strided<[0, 0]>> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.allow_tma = false}) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 4>
} {
  %value = bufferization.to_tensor %input restrict : memref<4x8xf32, strided<[16, 1]>> to tensor<4x8xf32>
  bufferization.materialize_in_destination %value in restrict writable %output
      : (tensor<4x8xf32>, memref<4x8xf32, strided<[0, 0]>>) -> ()
  return
}
// CHECK: nv_tensor_ir.slice {{.*}} starts = [0, 0] limits = [1, 1]
// CHECK-NOT: nv_tensor_ir.reshape
// CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [1, 1], strides: [2, 2]
// CHECK: nv_tensor_ir.store {{.*}} tile_indices [0, 0], alignment 16 allow_tma false
// CHECK-SAME: tensor<1x1xf32>, memref<1x1xf32, strided<[2, 2]>>

// -----

// The operand plan selects M and N from different output dimensions. The
// result plan restores the transposed layout and broadcasts the batch tile.
// CHECK-LABEL: func.func @transposed_broadcast_matmul
//       CHECK: scf.forall (%[[LINEAR:.*]]) in (12)
//       CHECK: %[[M:.*]] = arith.remui %[[LINEAR]],
//       CHECK: %[[REST:.*]] = arith.divui %[[LINEAR]],
//       CHECK: %[[N:.*]] = arith.remui %[[REST]],
//       CHECK: %[[BATCH:.*]] = arith.divui %[[REST]],
//       CHECK: %[[LHS:.*]] = nv_tensor_ir.load {{.*}} tile_indices [%[[M]], 0]
//  CHECK-SAME: to tensor<2x8xf32>
//       CHECK: %[[RHS:.*]] = nv_tensor_ir.load {{.*}} tile_indices [0, %[[N]]]
//  CHECK-SAME: to tensor<8x2xf32>
//       CHECK: %[[PRODUCT:.*]] = nv_tensor_ir.matmul(%[[LHS]], %[[RHS]])
//  CHECK-SAME: nv_tensor_ir.lhs_contracting_dimensions = array<i64: 1>
//  CHECK-SAME: nv_tensor_ir.rhs_contracting_dimensions = array<i64: 0>
//       CHECK: %[[RESHAPED:.*]] = nv_tensor_ir.reshape %[[PRODUCT]] : tensor<2x2xf32> -> tensor<1x2x2xf32>
//       CHECK: %[[TRANSPOSED:.*]] = nv_tensor_ir.transpose %[[RESHAPED]] permutation = [0, 2, 1]
//       CHECK: %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[TRANSPOSED]] : tensor<1x2x2xf32> -> tensor<2x2x2xf32>
//       CHECK: nv_tensor_ir.store %[[BROADCAST]], {{.*}} tile_indices [%[[BATCH]], %[[N]], %[[M]]]
func.func @transposed_broadcast_matmul(
    %lhs: memref<1x4x8xf32>,
    %rhs: memref<1x8x6xf32>,
    %output: memref<3x6x4xf32>)
    attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 2, 2>
} {
  %a = bufferization.to_tensor %lhs restrict
      : memref<1x4x8xf32> to tensor<1x4x8xf32>
  %b = bufferization.to_tensor %rhs restrict
      : memref<1x8x6xf32> to tensor<1x8x6xf32>
  %product = nv_tensor_ir.matmul(%a, %b)
      : (tensor<1x4x8xf32>, tensor<1x8x6xf32>) -> tensor<1x4x6xf32>
  %transposed = nv_tensor_ir.transpose %product permutation = [0, 2, 1]
      : tensor<1x4x6xf32> -> tensor<1x6x4xf32>
  %result = nv_tensor_ir.broadcast %transposed
      : tensor<1x6x4xf32> -> tensor<3x6x4xf32>
  bufferization.materialize_in_destination %result in restrict writable %output
      : (tensor<3x6x4xf32>, memref<3x6x4xf32>) -> ()
  return
}

// -----

// Both results of one user-defined reduction must share the tiled reduction.
// CHECK-LABEL: func.func @multi_result_reduction
//       CHECK: scf.forall
//       CHECK: %[[A:.*]] = nv_tensor_ir.load {{.*}} to tensor<2x8xf32>
//       CHECK: %[[B:.*]] = nv_tensor_ir.load {{.*}} to tensor<2x8xf16>
//       CHECK: %[[REDUCED:.*]]:2 = nv_tensor_ir.reduce_ud(%[[A]], %[[B]])
//  CHECK-SAME: dimensions = [1]
//       CHECK: nv_tensor_ir.yield
//       CHECK: nv_tensor_ir.reduction_extent = 8 : i64
//  CHECK-SAME: tensor<2x8xf32>, tensor<2x8xf16> -> tensor<2x1xf32>, tensor<2x1xf16>
//   CHECK-NOT: nv_tensor_ir.reduce_ud
//   CHECK-NOT: nv_tensor_ir.slice
//       CHECK: nv_tensor_ir.store %[[REDUCED]]#0
//  CHECK-SAME: tensor<2x1xf32>, memref<4x1xf32, strided<[1, 2]>>
//   CHECK-NOT: nv_tensor_ir.reduce_ud
//   CHECK-NOT: nv_tensor_ir.slice
//       CHECK: nv_tensor_ir.store %[[REDUCED]]#1
//  CHECK-SAME: tensor<2x1xf16>, memref<4x1xf16, strided<[1, 2]>>
//   CHECK-NOT: nv_tensor_ir.reduce_ud
//       CHECK: return
func.func @multi_result_reduction(%a: memref<4x8xf32>,
    %b: memref<4x8xf16>,
    %out0: memref<4x1xf32, strided<[1, 0]>>,
    %out1: memref<4x1xf16, strided<[1, 0]>>) attributes {
  nv_tensor_ir.bufferized_program,
  tile_size = array<i32: 2, 1>
} {
  %a_tile = bufferization.to_tensor %a restrict
      : memref<4x8xf32> to tensor<4x8xf32>
  %b_tile = bufferization.to_tensor %b restrict
      : memref<4x8xf16> to tensor<4x8xf16>
  %r0, %r1 = nv_tensor_ir.reduce_ud(%a_tile, %b_tile)
      <dimensions = [1], identity = [0.0 : f32, 0.0 : f16]>
      (%acc0: f32, %acc1: f16, %v0: f32, %v1: f16) {
    %sum0 = arith.addf %acc0, %v0 : f32
    %sum1 = arith.addf %acc1, %v1 : f16
    nv_tensor_ir.yield %sum0, %sum1 : f32, f16
  }
      : tensor<4x8xf32>, tensor<4x8xf16> -> tensor<4x1xf32>, tensor<4x1xf16>
  bufferization.materialize_in_destination %r0 in restrict writable %out0
      : (tensor<4x1xf32>, memref<4x1xf32, strided<[1, 0]>>)
        -> ()
  bufferization.materialize_in_destination %r1 in restrict writable %out1
      : (tensor<4x1xf16>, memref<4x1xf16, strided<[1, 0]>>)
        -> ()
  return
}
