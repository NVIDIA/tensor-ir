// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,layout-propagation-annotation,layout-propagation-normalization,tile-selection,graph-splitting),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{persistence=static sm-count=2 occupancy=1}))' \
// RUN:   | FileCheck %s
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,layout-propagation-annotation,layout-propagation-normalization,tile-selection,graph-splitting),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{persistence=static sm-count=2 occupancy=1}),outline-tensor-ir-kernel,convert-tensor-to-cuda-tile{codegen-strategy=layout_propagation})' \
// RUN:   | FileCheck %s --check-prefix=LOWERED

// Static persistence caps the physical grid and visits logical tile IDs with
// a grid-stride loop.
// CHECK-LABEL: func.func @persistent_dispatch(
//  CHECK-SAME:     %[[INPUT:[^:]+]]: memref<64xf32, #ptr.generic_space>
//  CHECK-SAME:     %[[OUTPUT:[^:]+]]: memref<64xf32, #ptr.generic_space>
//       CHECK:   %[[IPTR:.*]] = ptr.to_ptr %[[INPUT]]
//       CHECK:   %[[OPTR:.*]] = ptr.to_ptr %[[OUTPUT]]
//   CHECK-DAG:   %[[TOTAL:.*]] = arith.constant 4 : index
//   CHECK-DAG:   %[[GRID:.*]] = arith.constant 2 : index
//       CHECK:   scf.forall (%[[BLOCK:.*]]) in (2) {
//       CHECK:     scf.for %[[TILE:.*]] = %[[BLOCK]] to %[[TOTAL]] step %[[GRID]] {
//       CHECK:       %[[LOADED:.*]] = nv_tensor_ir.load %[[IPTR]][%[[TILE]]] view offset: [0], sizes: [64], strides: [1], alignment: 4
//  CHECK-SAME:         to tensor<16xf32>
//       CHECK:       %[[COS:.*]] = nv_tensor_ir.cos %[[LOADED]] : tensor<16xf32>
//       CHECK:       nv_tensor_ir.store %[[COS]], %[[OPTR]][%[[TILE]]] view offset: [0], sizes: [64], strides: [1], alignment: 4
//  CHECK-NEXT:     }
//  CHECK-NEXT:   } {mapping = [#gpu.block<x>]}

// LOWERED-LABEL: entry @persistent(
//  LOWERED-SAME: %[[INPUT:[^:]+]]: tile<ptr<f32>>
//  LOWERED-SAME: %[[OUTPUT:[^:]+]]: tile<ptr<f32>>
//       LOWERED: %[[BLOCK:[^,]+]], %{{[^,]+}}, %{{[^,]+}} = get_tile_block_id
//       LOWERED: %[[TOTAL:.*]] = constant <i32: 4>
//       LOWERED: %[[GRID:.*]] = constant <i32: 2>
//       LOWERED: for %[[TILE:.*]] in (%[[BLOCK]] to %[[TOTAL]], step %[[GRID]])
//       LOWERED:   %[[ITVIEW:.*]] = make_tensor_view %[[INPUT]], shape = [64], strides = [1]
//       LOWERED:   %[[IPVIEW:.*]] = make_partition_view %[[ITVIEW]] : partition_view<tile=(16)
//       LOWERED:   %[[LOADED:.*]], %{{.*}} = load_view_tko weak %[[IPVIEW]][%[[TILE]]]
//       LOWERED:   %[[COS:.*]] = cos %[[LOADED]] : tile<16xf32>
//       LOWERED:   %[[OTVIEW:.*]] = make_tensor_view %[[OUTPUT]], shape = [64], strides = [1]
//       LOWERED:   %[[OPVIEW:.*]] = make_partition_view %[[OTVIEW]] : partition_view<tile=(16)
//       LOWERED: store_view_tko weak %[[COS]], %[[OPVIEW]][%[[TILE]]]
//  LOWERED-NEXT: }
nv_tensor_ir.graph @persistent(%input: tensor<64xf32>)
    -> tensor<64xf32> attributes {tile_size = array<i32: 16>} {
  %result = cos %input : tensor<64xf32>
  results %result : tensor<64xf32>
}

// -----

// When every logical tile already has a resident block, persistence is a
// no-op and no serial loop is introduced.
// CHECK-LABEL: func.func @fits_grid_dispatch(
//  CHECK-SAME:     %[[INPUT:[^:]+]]: memref<32xf32, #ptr.generic_space>
//  CHECK-SAME:     %[[OUTPUT:[^:]+]]: memref<32xf32, #ptr.generic_space>
//       CHECK:   %[[IPTR:.*]] = ptr.to_ptr %[[INPUT]]
//       CHECK:   %[[OPTR:.*]] = ptr.to_ptr %[[OUTPUT]]
//       CHECK:   scf.forall (%[[BLOCK:.*]]) in (2) {
//   CHECK-NOT:     scf.for
//       CHECK:     %[[LOADED:.*]] = nv_tensor_ir.load %[[IPTR]][%[[BLOCK]]] view offset: [0], sizes: [32], strides: [1], alignment: 4
//  CHECK-SAME:       to tensor<16xf32>
//       CHECK:     %[[COS:.*]] = nv_tensor_ir.cos %[[LOADED]] : tensor<16xf32>
//       CHECK:     nv_tensor_ir.store %[[COS]], %[[OPTR]][%[[BLOCK]]] view offset: [0], sizes: [32], strides: [1], alignment: 4
//  CHECK-NEXT:   } {mapping = [#gpu.block<x>]}

// LOWERED-LABEL: entry @fits_grid(
//    LOWERED-SAME: %[[INPUT:[^:]+]]: tile<ptr<f32>>
//    LOWERED-SAME: %[[OUTPUT:[^:]+]]: tile<ptr<f32>>
//         LOWERED: %[[BLOCK:[^,]+]], %{{[^,]+}}, %{{[^,]+}} = get_tile_block_id
//     LOWERED-NOT: for %
//         LOWERED: %[[ITVIEW:.*]] = make_tensor_view %[[INPUT]], shape = [32], strides = [1]
//         LOWERED: %[[IPVIEW:.*]] = make_partition_view %[[ITVIEW]] : partition_view<tile=(16)
//         LOWERED: %[[LOADED:.*]], %{{.*}} = load_view_tko weak %[[IPVIEW]][%[[BLOCK]]]
//         LOWERED: %[[COS:.*]] = cos %[[LOADED]] : tile<16xf32>
//         LOWERED: %[[OTVIEW:.*]] = make_tensor_view %[[OUTPUT]], shape = [32], strides = [1]
//         LOWERED: %[[OPVIEW:.*]] = make_partition_view %[[OTVIEW]] : partition_view<tile=(16)
//         LOWERED: store_view_tko weak %[[COS]], %[[OPVIEW]][%[[BLOCK]]]
//    LOWERED-NEXT: return
nv_tensor_ir.graph @fits_grid(%input: tensor<32xf32>)
    -> tensor<32xf32> attributes {tile_size = array<i32: 16>} {
  %result = cos %input : tensor<32xf32>
  results %result : tensor<32xf32>
}

// -----

// Runtime-shaped grids use max(1, min(totalTiles, smCount * occupancy)) for
// the physical grid while retaining the uncapped total as the loop bound. The
// total is ceildiv(rows, 4) row tiles times one column tile; the logical tile
// ID is delinearized with the row-tile count.
// CHECK-LABEL: func.func @dynamic_grid_dispatch(
//  CHECK-SAME:     %[[INPUT:[^:]+]]: memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space>
//  CHECK-SAME:     %[[OUTPUT:[^:]+]]: memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space>
//       CHECK:   %{{[^,]+}}, %{{[^,]+}}, %[[ISIZES:[^:]+]]:2, %[[ISTRIDES:[^:]+]]:2 = memref.extract_strided_metadata %[[INPUT]]
//       CHECK:   %[[IPTR:.*]] = ptr.to_ptr %[[INPUT]]
//       CHECK:   %{{[^,]+}}, %{{[^,]+}}, %[[OSIZES:[^:]+]]:2, %[[OSTRIDES:[^:]+]]:2 = memref.extract_strided_metadata %[[OUTPUT]]
//       CHECK:   %[[OPTR:.*]] = ptr.to_ptr %[[OUTPUT]]
//       CHECK:   %[[C3:.*]] = arith.constant 3 : index
//       CHECK:   %[[ROWS_PLUS_3:.*]] = arith.addi %[[OSIZES]]#0, %[[C3]] : index
//       CHECK:   %[[C4:.*]] = arith.constant 4 : index
//       CHECK:   %[[ROW_TILES:.*]] = arith.divui %[[ROWS_PLUS_3]], %[[C4]] : index
//       CHECK:   %[[C1:.*]] = arith.constant 1 : index
//       CHECK:   %[[PARTIAL:.*]] = arith.muli %[[C1]], %[[ROW_TILES]] : index
//       CHECK:   %[[COLUMN_TILES:.*]] = arith.constant 1 : index
//       CHECK:   %[[TOTAL:.*]] = arith.muli %[[PARTIAL]], %[[COLUMN_TILES]] : index
//       CHECK:   %[[LIMIT:.*]] = arith.constant 2 : index
//       CHECK:   %[[ONE:.*]] = arith.constant 1 : index
//       CHECK:   %[[MIN:.*]] = arith.minui %[[TOTAL]], %[[LIMIT]] : index
//       CHECK:   %[[GRID:.*]] = arith.maxui %[[MIN]], %[[ONE]] : index
//       CHECK:   scf.forall (%[[BLOCK:.*]]) in (%[[GRID]]) {
//       CHECK:     scf.for %[[TILE:.*]] = %[[BLOCK]] to %[[TOTAL]] step %[[GRID]] {
//       CHECK:       %[[ROW:.*]] = arith.remui %[[TILE]], %[[ROW_TILES]] : index
//       CHECK:       %[[COLUMN:.*]] = arith.divui %[[TILE]], %[[ROW_TILES]] : index
//       CHECK:       %[[LOADED:.*]] = nv_tensor_ir.load %[[IPTR]][%[[ROW]], %[[COLUMN]]] view offset: [0], sizes: [%[[ISIZES]]#0, 32], strides: [%[[ISTRIDES]]#0, 1], alignment: 4
//  CHECK-SAME:         to tensor<4x32xf32>
//       CHECK:       %[[COS:.*]] = nv_tensor_ir.cos %[[LOADED]] : tensor<4x32xf32>
//       CHECK:       nv_tensor_ir.store %[[COS]], %[[OPTR]][%[[ROW]], %[[COLUMN]]] view offset: [0], sizes: [%[[OSIZES]]#0, 32], strides: [%[[OSTRIDES]]#0, 1], alignment: 4
//  CHECK-NEXT:     }
//  CHECK-NEXT:   } {mapping = [#gpu.block<x>]}

// LOWERED-LABEL: entry @dynamic_grid(
//    LOWERED-SAME: %[[INPUT:[^:]+]]: tile<ptr<f32>>
//    LOWERED-SAME: %[[IROWS:[^:]+]]: tile<i32>
//    LOWERED-SAME: %[[ISTRIDE:[^:]+]]: tile<i32>
//    LOWERED-SAME: %[[OUTPUT:[^:]+]]: tile<ptr<f32>>
//    LOWERED-SAME: %[[OROWS:[^:]+]]: tile<i32>
//    LOWERED-SAME: %[[OSTRIDE:[^:]+]]: tile<i32>
//         LOWERED: %[[BLOCK:[^,]+]], %{{[^,]+}}, %{{[^,]+}} = get_tile_block_id
//         LOWERED: %[[GRID:[^,]+]], %{{[^,]+}}, %{{[^,]+}} = get_num_tile_blocks
//         LOWERED: %[[C3:.*]] = constant <i32: 3>
//         LOWERED: %[[ROWS_PLUS_3:.*]] = addi %[[OROWS]], %[[C3]]
//         LOWERED: %[[C4:.*]] = constant <i32: 4>
//         LOWERED: %[[TOTAL:.*]] = divi %[[ROWS_PLUS_3]], %[[C4]] unsigned
//         LOWERED: for %[[TILE:.*]] in (%[[BLOCK]] to %[[TOTAL]], step %[[GRID]])
//         LOWERED:   %[[ROW:.*]] = remi %[[TILE]], %[[TOTAL]] unsigned
//         LOWERED:   %[[COLUMN:.*]] = divi %[[TILE]], %[[TOTAL]] unsigned
//         LOWERED:   %[[ITVIEW:.*]] = make_tensor_view %[[INPUT]], shape = [%[[IROWS]], 32], strides = [%[[ISTRIDE]], 1]
//         LOWERED:   %[[IPVIEW:.*]] = make_partition_view %[[ITVIEW]] : partition_view<tile=(4x32)
//         LOWERED:   %[[LOADED:.*]], %{{.*}} = load_view_tko weak %[[IPVIEW]][%[[ROW]], %[[COLUMN]]]
//         LOWERED:   %[[COS:.*]] = cos %[[LOADED]] : tile<4x32xf32>
//         LOWERED:   %[[OTVIEW:.*]] = make_tensor_view %[[OUTPUT]], shape = [%[[OROWS]], 32], strides = [%[[OSTRIDE]], 1]
//         LOWERED:   %[[OPVIEW:.*]] = make_partition_view %[[OTVIEW]] : partition_view<tile=(4x32)
//         LOWERED: store_view_tko weak %[[COS]], %[[OPVIEW]][%[[ROW]], %[[COLUMN]]]
//    LOWERED-NEXT: }
nv_tensor_ir.graph @dynamic_grid(
    %input: tensor<?x32xf32> {nv_tensor_ir.stride = "(?,1)"})
    -> (tensor<?x32xf32> {nv_tensor_ir.stride = "(?,1)"})
    attributes {tile_size = array<i32: 4, 32>} {
  %result = cos %input : tensor<?x32xf32>
  results %result : tensor<?x32xf32>
}
