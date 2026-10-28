// RUN: tensor_ir-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(tir-tile-reductions{reduction-tile-size=4}))' | FileCheck %s --check-prefixes=BASE,BASE4
// RUN: tensor_ir-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(tir-tile-reductions{reduction-tile-size=1}))' | FileCheck %s --check-prefixes=BASE,BASE1
// RUN: tensor_ir-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(tir-tile-reductions{reduction-tile-size=2}))' | FileCheck %s --check-prefixes=COMMON,TILE2
// RUN: tensor_ir-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(tir-tile-reductions{reduction-tile-size=4}))' | FileCheck %s --check-prefixes=CHECK,COMMON,TILE4

// BASE-LABEL: func.func @splat_producer(
// BASE-SAME: %[[SCALAR:.*]]: f32
func.func @splat_producer(%scalar: f32) -> tensor<4x1xf32> {
  // BASE: scf.for
  // BASE4: %[[SPLAT:.*]] = nv_tensor_ir.splat %[[SCALAR]] : tensor<4x4xf32>
  // BASE1: %[[SPLAT:.*]] = nv_tensor_ir.splat %[[SCALAR]] : tensor<4x1xf32>
  // BASE: nv_tensor_ir.add {{.*}}, %[[SPLAT]]
  %input = nv_tensor_ir.splat %scalar : tensor<4x8xf32>
  %result = nv_tensor_ir.reduce(%input)
      <dimensions = [1], reduction_mode = <add>>
      : tensor<4x8xf32> -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// -----

// A broadcast source remains singleton along the contraction dimension, and
// each contraction chunk reads the same source element.
// BASE-LABEL: func.func @broadcast_contraction_axis(
func.func @broadcast_contraction_axis(%view: memref<4x1xf32>)
    -> tensor<4x1xf32> {
  // BASE: scf.for
  // BASE: %[[LOAD:.*]] = nv_tensor_ir.load {{.*}} tile_indices [0, 0]{{.*}} to tensor<4x1xf32>
  // BASE4: %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[LOAD]] : tensor<4x1xf32> -> tensor<4x4xf32>
  // BASE4: nv_tensor_ir.add {{.*}}, %[[BROADCAST]]
  // BASE1-NOT: nv_tensor_ir.broadcast
  // BASE1: nv_tensor_ir.add {{.*}}, %[[LOAD]]
  // BASE: scf.yield
  %input = nv_tensor_ir.load %view tile_indices [0, 0], alignment 4
      : memref<4x1xf32> to tensor<4x1xf32>
  %wide = nv_tensor_ir.broadcast %input
      : tensor<4x1xf32> -> tensor<4x8xf32>
  %result = nv_tensor_ir.reduce(%wide)
      <dimensions = [1], reduction_mode = <add>>
      : tensor<4x8xf32> -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// -----

// A broadcast along a parallel dimension still advances the source along the
// contraction dimension.
// BASE-LABEL: func.func @broadcast_parallel_axis(
func.func @broadcast_parallel_axis(%view: memref<1x8xf32>)
    -> tensor<4x1xf32> {
  // BASE: scf.for %[[K:[^ ]+]] =
  // BASE4: %[[LOAD:.*]] = nv_tensor_ir.load {{.*}} tile_indices [0, %[[K]]]{{.*}} to tensor<1x4xf32>
  // BASE1: %[[LOAD:.*]] = nv_tensor_ir.load {{.*}} tile_indices [0, %[[K]]]{{.*}} to tensor<1x1xf32>
  // BASE4: %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[LOAD]] : tensor<1x4xf32> -> tensor<4x4xf32>
  // BASE1: %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[LOAD]] : tensor<1x1xf32> -> tensor<4x1xf32>
  // BASE: nv_tensor_ir.add {{.*}}, %[[BROADCAST]]
  %input = nv_tensor_ir.load %view tile_indices [0, 0], alignment 4
      : memref<1x8xf32> to tensor<1x8xf32>
  %wide = nv_tensor_ir.broadcast %input
      : tensor<1x8xf32> -> tensor<4x8xf32>
  %result = nv_tensor_ir.reduce(%wide)
      <dimensions = [1], reduction_mode = <add>>
      : tensor<4x8xf32> -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// -----

// Broadcasting a parallel dimension preserves neutral hardware load padding.
// CHECK-LABEL: func.func @parallel_broadcast(
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<1x14xf32, strided<[2, 1]>>
func.func @parallel_broadcast(%input: memref<1x14xf32, strided<[2, 1]>>) -> tensor<2x1xf32> {
  %load = nv_tensor_ir.load %input tile_indices [0, 0], alignment 4
      : memref<1x14xf32, strided<[2, 1]>> to tensor<1x14xf32>
  %broadcast = nv_tensor_ir.broadcast %load
      : tensor<1x14xf32> -> tensor<2x14xf32>
  //       CHECK: %[[LOOP:.*]] = scf.for %[[K:[^ ]+]] =
  //       CHECK:   %[[LOAD:.*]] = nv_tensor_ir.load %[[INPUT]] tile_indices [0, %[[K]]], alignment 4, padding <zero>
  //  CHECK-SAME:    : memref<1x14xf32, strided<[2, 1]>> to tensor<1x4xf32>
  //       CHECK:   %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[LOAD]] : tensor<1x4xf32> -> tensor<2x4xf32>
  //       CHECK:   nv_tensor_ir.add {{.*}}, %[[BROADCAST]]
  //   CHECK-NOT:   nv_tensor_ir.binary_select
  //       CHECK:   scf.yield
  //       CHECK: nv_tensor_ir.reduce(%[[LOOP]])
  %result = nv_tensor_ir.reduce(%broadcast)
      <dimensions = [1], reduction_mode = <add>>
      : tensor<2x14xf32> -> tensor<2x1xf32>
  return %result : tensor<2x1xf32>
}

// -----

// A broadcast contracting dimension does not advance the scalar load. The
// final partial tile needs an explicit mask, since the load is always in bounds.
// CHECK-LABEL: func.func @contracting_broadcast(
//  CHECK-SAME: %[[INPUT:[^:]+]]: memref<1x1xf32, strided<[2, 2]>>
func.func @contracting_broadcast(%input: memref<1x1xf32, strided<[2, 2]>>) -> tensor<1x1xf32> {
  %load = nv_tensor_ir.load %input tile_indices [0, 0], alignment 4
      : memref<1x1xf32, strided<[2, 2]>> to tensor<1x1xf32>
  %broadcast = nv_tensor_ir.broadcast %load
      : tensor<1x1xf32> -> tensor<1x14xf32>
  //       CHECK: %[[LOOP:.*]] = scf.for %[[K:[^ ]+]] =
  //       CHECK:   arith.muli %[[K]],
  //       CHECK:   nv_tensor_ir.iota
  //       CHECK:   %[[LOAD:.*]] = nv_tensor_ir.load %[[INPUT]] tile_indices [0, 0], alignment 4
  //  CHECK-SAME:    : memref<1x1xf32, strided<[2, 2]>> to tensor<1x1xf32>
  //       CHECK:   %[[BROADCAST:.*]] = nv_tensor_ir.broadcast %[[LOAD]] : tensor<1x1xf32> -> tensor<1x4xf32>
  //       CHECK:   %[[MASKED:.*]] = nv_tensor_ir.binary_select {{.*}}, %[[BROADCAST]],
  //       CHECK:   nv_tensor_ir.add {{.*}}, %[[MASKED]]
  //       CHECK:   scf.yield
  //       CHECK: nv_tensor_ir.reduce(%[[LOOP]])
  %result = nv_tensor_ir.reduce(%broadcast)
      <dimensions = [1], reduction_mode = <add>>
      : tensor<1x14xf32> -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

// -----

// Mixed varying and broadcast contracting axes require explicit masking.
// This load/broadcast chain is produced by FormGrid.
// With a [2, 1] tile, reshaping the predicate already gives the mask shape;
// with a [2, 2] tile, the predicate must also be broadcast.
// COMMON-LABEL: func.func @mixed_axis_broadcast(
func.func @mixed_axis_broadcast(%input: memref<1x13x1xf32, strided<[2, 1, 2]>>) -> tensor<1x1x1xf32> {
  %load = nv_tensor_ir.load %input tile_indices [0, 0, 0], alignment 4
      : memref<1x13x1xf32, strided<[2, 1, 2]>> to tensor<1x13x1xf32>
  %broadcast = nv_tensor_ir.broadcast %load
      : tensor<1x13x1xf32> -> tensor<1x13x5xf32>
  // COMMON: %[[PRED:.*]] = nv_tensor_ir.reshape {{.*}} : tensor<2xi1> -> tensor<2x1xi1>
  // TILE2-NOT: nv_tensor_ir.broadcast %[[PRED]]
  // TILE2: %[[FULL_MASK:.*]] = nv_tensor_ir.reshape %[[PRED]] : tensor<2x1xi1> -> tensor<1x2x1xi1>
  // TILE2: nv_tensor_ir.binary_select %[[FULL_MASK]], {{.*}} : tensor<1x2x1xf32>
  // TILE4: %[[MASK:.*]] = nv_tensor_ir.broadcast %[[PRED]] : tensor<2x1xi1> -> tensor<2x2xi1>
  %result = nv_tensor_ir.reduce(%broadcast)
      <dimensions = [1, 2], reduction_mode = <add>>
      : tensor<1x13x5xf32> -> tensor<1x1x1xf32>
  return %result : tensor<1x1x1xf32>
}
