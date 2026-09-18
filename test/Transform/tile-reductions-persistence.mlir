// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,layout-propagation-annotation,layout-propagation-normalization,tile-selection,graph-splitting),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{persistence=static sm-count=2 occupancy=1}),outline-tensor-ir-kernel,convert-tensor-to-cuda-tile{codegen-strategy=layout_propagation})' --split-input-file | FileCheck %s --check-prefix=LOWERED
// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,layout-propagation-annotation,layout-propagation-normalization,tile-selection,graph-splitting),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{persistence=static sm-count=2 occupancy=1}))' --split-input-file | FileCheck %s

// Static persistence caps the physical grid and visits logical tile IDs with
// a grid-stride loop.
// CHECK-LABEL: func.func @persistent_dispatch
// CHECK-DAG: %[[TOTAL:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[GRID:.*]] = arith.constant 2 : index
// CHECK: scf.forall (%[[BLOCK:.*]]) in (2) {
// CHECK: scf.for %[[TILE:.*]] = %[[BLOCK]] to %[[TOTAL]] step %[[GRID]] {
// CHECK: nv_tensor_ir.load {{.*}}[%[[TILE]]]
// CHECK: nv_tensor_ir.store {{.*}}[%[[TILE]]]
// LOWERED-LABEL: entry @persistent
// LOWERED: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// LOWERED: %[[TOTAL:.*]] = constant <i32: 4>
// LOWERED: %[[GRID:.*]] = constant <i32: 2>
// LOWERED: for %{{.*}} in (%[[BLOCK]] to %[[TOTAL]], step %[[GRID]])
nv_tensor_ir.graph @persistent(%input: tensor<64xf32>)
    -> tensor<64xf32> attributes {tile_size = array<i32: 16>} {
  %result = cos %input : tensor<64xf32>
  results %result : tensor<64xf32>
}

// -----

// When every logical tile already has a resident block, persistence is a
// no-op and no serial loop is introduced.
// CHECK-LABEL: func.func @fits_grid_dispatch
// CHECK: scf.forall (%[[BLOCK:.*]]) in (2) {
// CHECK-NOT: scf.for %
// CHECK: nv_tensor_ir.load {{.*}}[%[[BLOCK]]]
// LOWERED-LABEL: entry @fits_grid
// LOWERED-NOT: for %
// LOWERED: load_view_tko
nv_tensor_ir.graph @fits_grid(%input: tensor<32xf32>)
    -> tensor<32xf32> attributes {tile_size = array<i32: 16>} {
  %result = cos %input : tensor<32xf32>
  results %result : tensor<32xf32>
}

// -----

// Runtime-shaped grids use max(1, min(totalTiles, smCount * occupancy)) for
// the physical grid while retaining the uncapped total as the loop bound.
// CHECK-LABEL: func.func @dynamic_grid_dispatch
// CHECK: %[[PARTIAL:.*]] = arith.muli
// CHECK: %[[TOTAL:.*]] = arith.muli %[[PARTIAL]],
// CHECK: %[[LIMIT:.*]] = arith.constant 2 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[MIN:.*]] = arith.minui %[[TOTAL]], %[[LIMIT]] : index
// CHECK: %[[GRID:.*]] = arith.maxui %[[MIN]], %[[ONE]] : index
// CHECK: scf.forall (%[[BLOCK:.*]]) in (%[[GRID]]) {
// CHECK: scf.for %[[TILE:.*]] = %[[BLOCK]] to %[[TOTAL]] step %[[GRID]] {
// CHECK: %[[ROW:.*]] = arith.remui %[[TILE]],
// CHECK: %[[COLUMN:.*]] = arith.divui %[[TILE]],
// CHECK: nv_tensor_ir.load {{.*}}[%[[ROW]], %[[COLUMN]]]
// LOWERED-LABEL: entry @dynamic_grid
// LOWERED: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// LOWERED: %[[GRID:.*]], %{{.*}}, %{{.*}} = get_num_tile_blocks
// LOWERED: for %{{.*}} in (%[[BLOCK]] to %{{.*}}, step %[[GRID]])
nv_tensor_ir.graph @dynamic_grid(
    %input: tensor<?x32xf32> {nv_tensor_ir.stride = "(?,1)"})
    -> (tensor<?x32xf32> {nv_tensor_ir.stride = "(?,1)"})
    attributes {tile_size = array<i32: 4, 32>} {
  %result = cos %input : tensor<?x32xf32>
  results %result : tensor<?x32xf32>
}
