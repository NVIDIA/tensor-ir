// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid))' --split-input-file | FileCheck %s --implicit-check-not='layout =' --implicit-check-not='bufferization.'

// Reconvergent graphs request one value in several iteration-space layouts.
// Grid formation materializes one tile per requested layout, shares tiles
// whose layouts agree, and transposes a tile whose layout, tile shape, and
// coordinates agree up to a permutation of dimensions.

// neg(a) is consumed directly and transposed, so it is computed twice.
// CHECK-LABEL: func.func @transpose_diamond_dispatch
//       CHECK: scf.forall
//       CHECK: %[[DIRECT:.*]] = nv_tensor_ir.load {{.*}} strided<[1, 8]
//       CHECK: %[[NEG:.*]] = nv_tensor_ir.neg %[[DIRECT]]
//       CHECK: %[[TRANSPOSED:.*]] = nv_tensor_ir.load {{.*}} strided<[8, 1]
//       CHECK: %[[NEG_T:.*]] = nv_tensor_ir.neg %[[TRANSPOSED]]
//   CHECK-NOT: nv_tensor_ir.transpose
//       CHECK: %[[ADD:.*]] = nv_tensor_ir.add %[[NEG]], %[[NEG_T]]
//       CHECK: nv_tensor_ir.store %[[ADD]]
nv_tensor_ir.graph @transpose_diamond(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"})
    -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"})
    attributes {tile_size = array<i32: 8, 1>} {
  %neg = neg %a : tensor<8x8xf32>
  %transpose = transpose %neg permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
  %add = add %neg, %transpose : tensor<8x8xf32>
  results %add : tensor<8x8xf32>
}

// -----

// With a whole-tensor tile, the transposed request reuses the direct tile.
// CHECK-LABEL: func.func @transpose_diamond_whole_tile_dispatch
//       CHECK: %[[LOAD:.*]] = nv_tensor_ir.load {{.*}} strided<[1, 8]
//   CHECK-NOT: nv_tensor_ir.load
//       CHECK: %[[NEG:.*]] = nv_tensor_ir.neg %[[LOAD]]
//   CHECK-NOT: nv_tensor_ir.neg
//       CHECK: %[[NEG_T:.*]] = nv_tensor_ir.transpose %[[NEG]] permutation = [1, 0]
//       CHECK: %[[ADD:.*]] = nv_tensor_ir.add %[[NEG]], %[[NEG_T]]
//       CHECK: nv_tensor_ir.store %[[ADD]]
nv_tensor_ir.graph @transpose_diamond_whole_tile(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"})
    -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"})
    attributes {tile_size = array<i32: 8, 8>} {
  %neg = neg %a : tensor<8x8xf32>
  %transpose = transpose %neg permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
  %add = add %neg, %transpose : tensor<8x8xf32>
  results %add : tensor<8x8xf32>
}

// -----

// The direct and transposed loads of %a agree up to a permutation.
// CHECK-LABEL: func.func @transposed_load_dispatch
//       CHECK: %[[LOAD:.*]] = nv_tensor_ir.load {{.*}} strided<[1, 8]
//   CHECK-NOT: nv_tensor_ir.load
//       CHECK: %[[LOAD_T:.*]] = nv_tensor_ir.transpose %[[LOAD]] permutation = [1, 0]
//       CHECK: %[[ADD:.*]] = nv_tensor_ir.add %[[LOAD]], %[[LOAD_T]]
//       CHECK: nv_tensor_ir.store %[[ADD]]
nv_tensor_ir.graph @transposed_load(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"})
    -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"})
    attributes {tile_size = array<i32: 8, 8>} {
  %transpose = transpose %a permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
  %add = add %a, %transpose : tensor<8x8xf32>
  results %add : tensor<8x8xf32>
}

// -----

// The reduction requests %shared with its reduced dimension last. That tile
// is a transpose of the tile already computed for %dx, so %dout and %z are
// loaded once.
// CHECK-LABEL: func.func @shared_producer_reduction_dispatch
//       CHECK: %[[DOUT:.*]] = nv_tensor_ir.load {{.*}} to tensor<1x8x16xf32>
//       CHECK: %[[Z:.*]] = nv_tensor_ir.load {{.*}} to tensor<1x8x16xf32>
//       CHECK: %[[EXP:.*]] = nv_tensor_ir.exp %[[Z]]
//       CHECK: %[[SHARED:.*]] = nv_tensor_ir.mul %[[DOUT]], %[[EXP]]
//       CHECK: %[[DX:.*]] = nv_tensor_ir.neg %[[SHARED]]
//       CHECK: nv_tensor_ir.store %[[DX]]
//   CHECK-NOT: nv_tensor_ir.exp
//       CHECK: %[[SHARED_T:.*]] = nv_tensor_ir.transpose %[[SHARED]] permutation = [0, 2, 1] : tensor<1x8x16xf32> -> tensor<1x16x8xf32>
//       CHECK: %[[X:.*]] = nv_tensor_ir.load {{.*}} to tensor<1x16x8xf32>
//   CHECK-NOT: nv_tensor_ir.load
//       CHECK: %[[CONTRIBUTION:.*]] = nv_tensor_ir.mul %[[SHARED_T]], %[[X]]
//       CHECK: nv_tensor_ir.reduce(%[[CONTRIBUTION]])
nv_tensor_ir.graph @shared_producer_reduction(
    %z: tensor<4x8x16xf32>, %dout: tensor<4x8x16xf32>, %x: tensor<4x8x16xf32>)
    -> (tensor<4x8x16xf32>, tensor<4x1x16xf32>)
    attributes {tile_size = array<i32: 1, 8, 16>} {
  %exp = exp %z : tensor<4x8x16xf32>
  %shared = mul %dout, %exp : tensor<4x8x16xf32>
  %dx = neg %shared : tensor<4x8x16xf32>
  %contribution = mul %shared, %x : tensor<4x8x16xf32>
  %partial = reduce(%contribution) <dimensions = [1], reduction_mode = <add>>
      : tensor<4x8x16xf32> -> tensor<4x1x16xf32>
  results %dx, %partial : tensor<4x8x16xf32>, tensor<4x1x16xf32>
}

// -----

// Both uses of neg(a) request the same layout and share one tile.
// CHECK-LABEL: func.func @same_layout_multi_consumer_dispatch
//       CHECK: %[[LOAD:.*]] = nv_tensor_ir.load
//       CHECK: %[[NEG:.*]] = nv_tensor_ir.neg %[[LOAD]]
//   CHECK-NOT: nv_tensor_ir.neg
//       CHECK: %[[MUL:.*]] = nv_tensor_ir.mul %[[NEG]], %[[NEG]]
//       CHECK: nv_tensor_ir.store %[[MUL]]
nv_tensor_ir.graph @same_layout_multi_consumer(
    %a: tensor<4x4xf32> {nv_tensor_ir.stride = "(1,4)"})
    -> (tensor<4x4xf32> {nv_tensor_ir.stride = "(1,4)"})
    attributes {tile_size = array<i32: 4, 4>} {
  %neg = neg %a : tensor<4x4xf32>
  %mul = mul %neg, %neg : tensor<4x4xf32>
  results %mul : tensor<4x4xf32>
}

// -----

// Slices with different offsets specialize the shared producer per offset.
// CHECK-LABEL: func.func @slice_diamond_dispatch
// CHECK: %[[A0_VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [0]
//       CHECK: %[[A0:.*]] = nv_tensor_ir.load %[[A0_VIEW]]
// CHECK: %[[B0_VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [0]
//       CHECK: %[[B0:.*]] = nv_tensor_ir.load %[[B0_VIEW]]
//       CHECK: %[[ADD0:.*]] = nv_tensor_ir.add %[[A0]], %[[B0]]
// CHECK: %[[A1_VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [32]
//       CHECK: %[[A1:.*]] = nv_tensor_ir.load %[[A1_VIEW]]
// CHECK: %[[B1_VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [32]
//       CHECK: %[[B1:.*]] = nv_tensor_ir.load %[[B1_VIEW]]
//       CHECK: %[[ADD1:.*]] = nv_tensor_ir.add %[[A1]], %[[B1]]
//       CHECK: %[[SUB:.*]] = nv_tensor_ir.sub %[[ADD0]], %[[ADD1]]
//       CHECK: nv_tensor_ir.store %[[SUB]]
nv_tensor_ir.graph @slice_diamond(
    %in0: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"},
    %in1: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"})
    -> (tensor<8x4xf32> {nv_tensor_ir.stride = "(1,8)"})
    attributes {tile_size = array<i32: 8, 4>} {
  %add = add %in0, %in1 : tensor<8x8xf32>
  %slice0 = slice %add starts = [0, 0] limits = [8, 4] strides = [1, 1] : tensor<8x8xf32> -> tensor<8x4xf32>
  %slice1 = slice %add starts = [0, 4] limits = [8, 8] strides = [1, 1] : tensor<8x8xf32> -> tensor<8x4xf32>
  %sub = sub %slice0, %slice1 : tensor<8x4xf32>
  results %sub : tensor<8x4xf32>
}

// -----

// Each concatenation branch materializes the shared producer in its own layout.
// CHECK-LABEL: func.func @concat_diamond_input_dispatch
//       CHECK: scf.if
//       CHECK: %[[DIRECT:.*]] = nv_tensor_ir.load {{.*}} strided<[1, 2, 32]
//       CHECK: %[[ADD:.*]] = nv_tensor_ir.add %[[DIRECT]], %[[DIRECT]]
//       CHECK: scf.yield %[[ADD]]
//       CHECK: } else {
//       CHECK: %[[TRANSPOSED:.*]] = nv_tensor_ir.load {{.*}} strided<[32, 2, 1]
//       CHECK: %[[ADD_T:.*]] = nv_tensor_ir.add %[[TRANSPOSED]], %[[TRANSPOSED]]
//       CHECK: scf.yield %[[ADD_T]]
nv_tensor_ir.graph @concat_diamond_input(
    %in: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"})
    -> (tensor<32x64xf32> {nv_tensor_ir.stride = "(1,32)"})
    attributes {tile_size = array<i32: 1, 1, 32>} {
  %add = add %in, %in : tensor<32x32xf32>
  %trans = transpose %add permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
  %concat = concatenate %add, %trans dimension = 1
      : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x64xf32>
  results %concat : tensor<32x64xf32>
}

// -----

// A pruned concatenation operand is never loaded.
// CHECK-LABEL: func.func @concat_pruned_first_operand_dispatch
//       CHECK: %[[LOAD:.*]] = nv_tensor_ir.load {{.*}} memref<10x1x5x{{.*}}, strided<[16, 2, 1], offset: 4>
//   CHECK-NOT: nv_tensor_ir.load
//       CHECK: nv_tensor_ir.store %[[LOAD]]
nv_tensor_ir.graph @concat_pruned_first_operand(
    %in1: tensor<32x16xf32>, %in2: tensor<32x16xf32>) -> tensor<10x5xf32>
    attributes {tile_size = array<i32: 8, 1, 4>} {
  %concat = concatenate %in1, %in2 dimension = 1
      : (tensor<32x16xf32>, tensor<32x16xf32>) -> tensor<32x32xf32>
  %slice = slice %concat starts = [0, 20] limits = [10, 25] strides = [1, 1] : tensor<32x32xf32> -> tensor<10x5xf32>
  results %slice : tensor<10x5xf32>
}

// -----

// The reduction is consumed directly and transposed, so it is reduced once per
// iteration space.
// CHECK-LABEL: func.func @reduce_diamond_output_dispatch
//       CHECK: %[[FULL:.*]] = nv_tensor_ir.load {{.*}} : memref<{{.*}}> to tensor<32x32xf32>
//       CHECK: %[[REDUCE:.*]] = nv_tensor_ir.reduce(%[[FULL]])
//       CHECK: %[[ROW:.*]] = nv_tensor_ir.load {{.*}} : memref<{{.*}}> to tensor<1x32xf32>
//       CHECK: %[[REDUCE_T:.*]] = nv_tensor_ir.reduce(%[[ROW]])
//       CHECK: %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[REDUCE_T]]
//       CHECK: %[[ADD:.*]] = nv_tensor_ir.add %[[REDUCE]], %[[BROADCAST]]
//       CHECK: nv_tensor_ir.store %[[ADD]]
nv_tensor_ir.graph @reduce_diamond_output(
    %in: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"})
    -> (tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"})
    attributes {tile_size = array<i32: 32, 1>} {
  %reduce = reduce(%in)<dimensions = [1], reduction_mode = <add>> : tensor<32x32xf32> -> tensor<32x1xf32>
  %bcast = broadcast %reduce : tensor<32x1xf32> -> tensor<32x32xf32>
  %trans = transpose %bcast permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
  %add = add %bcast, %trans : tensor<32x32xf32>
  results %add : tensor<32x32xf32>
}

// -----

// The matmul operands request the shared producer in two layouts.
// CHECK-LABEL: func.func @matmul_diamond_input_dispatch
//       CHECK: %[[LHS_LOAD:.*]] = nv_tensor_ir.load {{.*}} strided<[1, 32]
//       CHECK: %[[LHS:.*]] = nv_tensor_ir.add %[[LHS_LOAD]], %[[LHS_LOAD]]
//       CHECK: %[[RHS_LOAD:.*]] = nv_tensor_ir.load {{.*}} strided<[32, 1]
//       CHECK: %[[RHS:.*]] = nv_tensor_ir.add %[[RHS_LOAD]], %[[RHS_LOAD]]
//       CHECK: %[[MATMUL:.*]] = nv_tensor_ir.matmul(%[[LHS]], %[[RHS]])
//       CHECK: nv_tensor_ir.store %[[MATMUL]]
nv_tensor_ir.graph @matmul_diamond_input(
    %in: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"})
    -> (tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"})
    attributes {tile_size = array<i32: 32, 1>} {
  %add = add %in, %in : tensor<32x32xf32>
  %trans = transpose %add permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
  %matmul = matmul(%add, %trans) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  results %matmul : tensor<32x32xf32>
}

// -----

// All extents and coordinates agree, but strides distinguish the axes. Matching
// must not enumerate the 10! permutations before finding the reversed layout.
// CHECK-LABEL: func.func @reverse_equal_extents_dispatch
//       CHECK: %[[LOAD:.*]] = nv_tensor_ir.load
//   CHECK-NOT: nv_tensor_ir.load
//       CHECK: %[[NEG:.*]] = nv_tensor_ir.neg %[[LOAD]]
//   CHECK-NOT: nv_tensor_ir.neg
//       CHECK: %[[REVERSED:.*]] = nv_tensor_ir.transpose %[[NEG]] permutation = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
//       CHECK: %[[ADD:.*]] = nv_tensor_ir.add %[[NEG]], %[[REVERSED]]
//       CHECK: nv_tensor_ir.store %[[ADD]]
nv_tensor_ir.graph @reverse_equal_extents(%a: tensor<2x2x2x2x2x2x2x2x2x2xf32>) -> tensor<2x2x2x2x2x2x2x2x2x2xf32>
    attributes {tile_size = array<i32: 2, 2, 2, 2, 2, 2, 2, 2, 2, 2>} {
  %neg = neg %a : tensor<2x2x2x2x2x2x2x2x2x2xf32>
  %t = transpose %neg permutation = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : tensor<2x2x2x2x2x2x2x2x2x2xf32> -> tensor<2x2x2x2x2x2x2x2x2x2xf32>
  %add = add %neg, %t : tensor<2x2x2x2x2x2x2x2x2x2xf32>
  results %add : tensor<2x2x2x2x2x2x2x2x2x2xf32>
}
