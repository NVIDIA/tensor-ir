// RUN: tensor_ir-opt %s -layout-propagation-pipeline -split-input-file | FileCheck %s
// RUN: tensor_ir-opt %s \
// RUN:   --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides),test-layout-normalization)' \
// RUN:   --split-input-file -o /dev/null \
// RUN:   | FileCheck %s --check-prefix=LAYOUT

// RUN: tensor_ir-opt %s \
// RUN:   --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-analyzer,tile-selection),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{reduction-tile-size=4}))' \
// RUN:   --split-input-file \
// RUN:   | FileCheck %s --check-prefix=GRID \
// RUN:   --implicit-check-not='nv_tensor_ir.scan_slice' --implicit-check-not='scan_reduction'

// Strided streaming input, interleaved ordinary output, and a downstream
// pointwise operation whose composite layout contains the scan_slice.
// LAYOUT-LABEL: layout normalization for @generic_add_scan
//  LAYOUT-NEXT: iteration_space: #nv_tensor_ir.composite_source<
//  LAYOUT-SAME: #nv_tensor_ir.scan_slice_source<
//  LAYOUT-SAME: inputs = [#nv_tensor_ir.scan_slice_input<source =
//  LAYOUT-SAME: #nv_tensor_ir.tensor_source<0, 0, "(2,8):(16,2)">, dim = 0, extent = 5, stride = 64>]

//  CHECK-LABEL: entry @generic_add_scan(
//   CHECK-SAME:  %[[VALUES:[^ ,:]+]]: tile<ptr<f32>>,
//   CHECK-SAME:  %[[INITIAL:[^ ,:]+]]: tile<ptr<f32>>,
//   CHECK-SAME:  %[[INITIAL_OUT:[^ ,:]+]]: tile<ptr<f32>>,
//   CHECK-SAME:  %[[HISTORY:[^ ,:]+]]: tile<ptr<f32>>,
//   CHECK-SAME:  %[[OUT:[^ ,:]+]]: tile<ptr<f32>>)
//    CHECK-DAG: %[[C16:.+]] = constant <i32: 16> : tile<i32>
//    CHECK-DAG: %[[C64:.+]] = constant <i32: 64> : tile<i32>
//    CHECK-DAG: %[[C1:.+]] = constant <i32: 1> : tile<i32>
//    CHECK-DAG: %[[C5:.+]] = constant <i32: 5> : tile<i32>
//    CHECK-DAG: %[[C0:.+]] = constant <i32: 0> : tile<i32>
//    CHECK-NOT: get_tile_block_id
//        CHECK: %[[INITIAL_VIEW:.+]] = make_tensor_view %[[INITIAL]], shape = [2, 8], strides = [8, 1]
//        CHECK: %[[INITIAL_PARTITION:.+]] = make_partition_view %[[INITIAL_VIEW]]
//        CHECK: %[[INITIAL_CARRY:.+]], %{{[^ ,]+}} = load_view_tko weak %[[INITIAL_PARTITION]][%[[C0]], %[[C0]]]
//        CHECK: %[[INITIAL_OUT_VIEW:.+]] = make_tensor_view %[[INITIAL_OUT]], shape = [2, 8], strides = [8, 1]
//        CHECK: %[[INITIAL_OUT_PARTITION:.+]] = make_partition_view %[[INITIAL_OUT_VIEW]]
//        CHECK: store_view_tko weak %[[INITIAL_CARRY]], %[[INITIAL_OUT_PARTITION]][%[[C0]], %[[C0]]]
//        CHECK: %[[FINAL_CARRY:.+]] = for %[[IV:.+]] in (%[[C0]] to %[[C5]], step %[[C1]]) : tile<i32>
//   CHECK-SAME:     iter_values(%[[CARRY:.+]] = %[[INITIAL_CARRY]]) -> (tile<2x8xf32>) {
//        CHECK:   %[[INPUT_OFFSET:.+]] = muli %[[IV]], %[[C64]]
//        CHECK:   %[[VALUES_PTR:.+]] = offset %[[VALUES]], %[[INPUT_OFFSET]]
//        CHECK:   %[[VALUES_VIEW:.+]] = make_tensor_view %[[VALUES_PTR]], shape = [2, 8], strides = [16, 2]
//        CHECK:   %[[VALUES_PARTITION:.+]] = make_partition_view %[[VALUES_VIEW]]
//        CHECK:   %[[READ_VALUES:.+]], %{{[^ ,]+}} = load_view_tko weak %[[VALUES_PARTITION]][%[[C0]], %[[C0]]]
//        CHECK:   %[[NEXT:.+]] = addf %[[CARRY]], %[[READ_VALUES]] : tile<2x8xf32>
//        CHECK:   %[[HISTORY_OFFSET:.+]] = muli %[[IV]], %[[C16]] : tile<i32>
//        CHECK:   %[[HISTORY_PTR:.+]] = offset %[[HISTORY]], %[[HISTORY_OFFSET]]
//        CHECK:   %[[HISTORY_VIEW:.+]] = make_tensor_view %[[HISTORY_PTR]], shape = [2, 8], strides = [8, 1]
//        CHECK:   %[[HISTORY_PARTITION:.+]] = make_partition_view %[[HISTORY_VIEW]]
//        CHECK:   store_view_tko weak %[[NEXT]], %[[HISTORY_PARTITION]][%[[C0]], %[[C0]]]
//        CHECK:   continue %[[NEXT]] : tile<2x8xf32>
//        CHECK: %[[OUTPUT:.+]] = addf %[[FINAL_CARRY]], %[[INITIAL_CARRY]] : tile<2x8xf32>
//        CHECK: %[[OUT_VIEW:.+]] = make_tensor_view %[[OUT]], shape = [2, 8], strides = [8, 1]
//        CHECK: %[[OUT_PARTITION:.+]] = make_partition_view %[[OUT_VIEW]]
//        CHECK: store_view_tko weak %[[OUTPUT]], %[[OUT_PARTITION]][%[[C0]], %[[C0]]]
//        CHECK: return

//   GRID-LABEL: func.func @generic_add_scan_dispatch(
//    GRID-SAME: %[[VALUES:[^ ,:]+]]: memref<5x2x8xf32, strided<[64, 16, 2]>
//    GRID-SAME: %[[INITIAL:[^ ,:]+]]: memref<2x8xf32, strided<[8, 1]>
//    GRID-SAME: %[[INITIAL_OUT:[^ ,:]+]]: memref<2x8xf32, strided<[8, 1]>
//    GRID-SAME: %[[HISTORY:[^ ,:]+]]: memref<5x2x8xf32, strided<[16, 8, 1]>
//    GRID-SAME: %[[OUT:[^ ,:]+]]: memref<2x8xf32, strided<[8, 1]>
//         GRID: scf.forall (%{{.+}}) in (1) {
//         GRID:   %[[INITIAL_CARRY:.+]] = nv_tensor_ir.load %[[INITIAL]] tile_indices [0, 0], alignment 4 : memref<{{.*}}> to tensor<2x8xf32>
//         GRID:   nv_tensor_ir.store %[[INITIAL_CARRY]], %[[INITIAL_OUT]] tile_indices [0, 0], alignment 4 : tensor<2x8xf32>, memref<{{.*}}>
//     GRID-DAG:   %[[C0:.+]] = arith.constant 0 : index
//     GRID-DAG:   %[[C5:.+]] = arith.constant 5 : index
//     GRID-DAG:   %[[C1:.+]] = arith.constant 1 : index
//         GRID:   %[[FINAL_CARRY:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C5]] step %[[C1]]
//    GRID-SAME:    iter_args(%[[CARRY:.+]] = %[[INITIAL_CARRY]]) -> (tensor<2x8xf32>) {
//         GRID:     %[[C64:.+]] = arith.constant 64 : index
//         GRID:     %[[MULI:.+]] = arith.muli %[[IV]], %[[C64]] : index
//         GRID:     %[[C0_2:.+]] = arith.constant 0 : index
//         GRID:     %[[ADDI:.+]] = arith.addi %[[C0_2]], %[[MULI]] : index
//         GRID: %[[VIEW0:.*]] = memref.reinterpret_cast %[[VALUES]] to offset: [%[[ADDI]]], sizes: [2, 8], strides: [16, 2]
//         GRID:     %[[READ_VALUES:.+]] = nv_tensor_ir.load %[[VIEW0]] tile_indices [0, 0], alignment 4 : memref<{{.*}}> to tensor<2x8xf32>
//         GRID:     %[[NEXT:.+]] = nv_tensor_ir.add %[[CARRY]], %[[READ_VALUES]] : tensor<2x8xf32>
//         GRID:     %[[C16:.+]] = arith.constant 16 : index
//         GRID:     %[[MULI_2:.+]] = arith.muli %[[IV]], %[[C16]] : index
//         GRID:     %[[C0_3:.+]] = arith.constant 0 : index
//         GRID:     %[[ADDI_2:.+]] = arith.addi %[[C0_3]], %[[MULI_2]] : index
//         GRID: %[[VIEW1:.*]] = memref.reinterpret_cast %[[HISTORY]] to offset: [%[[ADDI_2]]], sizes: [2, 8], strides: [8, 1]
//         GRID:     nv_tensor_ir.store %[[NEXT]], %[[VIEW1]] tile_indices [0, 0], alignment 1 : tensor<2x8xf32>, memref<{{.*}}>
//         GRID:     scf.yield %[[NEXT]] : tensor<2x8xf32>
//         GRID:   %[[OUTPUT:.+]] = nv_tensor_ir.add %[[FINAL_CARRY]], %[[INITIAL_CARRY]] : tensor<2x8xf32>
//         GRID:   nv_tensor_ir.store %[[OUTPUT]], %[[OUT]] tile_indices [0, 0], alignment 4 : tensor<2x8xf32>, memref<{{.*}}>
//         GRID: return

nv_tensor_ir.graph @generic_add_scan(
    %values: tensor<5x2x8xf32> {nv_tensor_ir.stride = "(64,16,2)"},
    %initial: tensor<2x8xf32>
  ) -> (tensor<2x8xf32>, tensor<5x2x8xf32>, tensor<2x8xf32>) {
  %final, %prefixes = "nv_tensor_ir.scan_slice"(%initial, %values) <{
      output_dimensions = array<i64: 0>, reverse = false,
      scan_dimensions = array<i64: 0>
    }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    %next = nv_tensor_ir.add %carry, %value : tensor<2x8xf32>
    nv_tensor_ir.yield %next, %next : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>)
       -> (tensor<2x8xf32>, tensor<5x2x8xf32>)
  %out = nv_tensor_ir.add %final, %initial : tensor<2x8xf32>
  results %initial, %prefixes, %out
      : tensor<2x8xf32>, tensor<5x2x8xf32>, tensor<2x8xf32>
}

// -----

// Composite carry, no streaming output, and ordinary downstream pointwise.
// LAYOUT-LABEL: layout normalization for @pointwise_carry_scan
//  LAYOUT-NEXT: iteration_space: #nv_tensor_ir.scan_slice_source<carry = #nv_tensor_ir.composite_source<

//  CHECK-LABEL: entry @pointwise_carry_scan(
//   CHECK-SAME:  %[[VALUES:[^ ,:]+]]: tile<ptr<f32>>,
//   CHECK-SAME:  %[[INITIAL_A:[^ ,:]+]]: tile<ptr<f32>>,
//   CHECK-SAME:  %[[INITIAL_B:[^ ,:]+]]: tile<ptr<f32>>,
//   CHECK-SAME:  %[[OUT:[^ ,:]+]]: tile<ptr<f32>>)
//    CHECK-DAG: %[[C16:.+]] = constant <i32: 16> : tile<i32>
//    CHECK-DAG: %[[C1:.+]] = constant <i32: 1> : tile<i32>
//    CHECK-DAG: %[[C5:.+]] = constant <i32: 5> : tile<i32>
//    CHECK-DAG: %[[C0:.+]] = constant <i32: 0> : tile<i32>
//    CHECK-NOT: get_tile_block_id
//        CHECK: %[[INITIAL_A_VIEW:.+]] = make_tensor_view %[[INITIAL_A]], shape = [2, 8], strides = [8, 1]
//        CHECK: %[[INITIAL_A_PARTITION:.+]] = make_partition_view %[[INITIAL_A_VIEW]]
//        CHECK: %[[READ_INITIAL_A:.+]], %{{.+}} = load_view_tko weak %[[INITIAL_A_PARTITION]][%[[C0]], %[[C0]]]
//        CHECK: %[[INITIAL_B_VIEW:.+]] = make_tensor_view %[[INITIAL_B]], shape = [2, 8], strides = [8, 1]
//        CHECK: %[[INITIAL_B_PARTITION:.+]] = make_partition_view %[[INITIAL_B_VIEW]]
//        CHECK: %[[READ_INITIAL_B:.+]], %{{.+}} = load_view_tko weak %[[INITIAL_B_PARTITION]][%[[C0]], %[[C0]]]
//        CHECK: %[[INITIAL_CARRY:.+]] = addf %[[READ_INITIAL_A]], %[[READ_INITIAL_B]] : tile<2x8xf32>
//        CHECK: %[[FINAL_CARRY:.+]] = for %[[IV:.+]] in (%[[C0]] to %[[C5]], step %[[C1]]) : tile<i32>
//   CHECK-SAME:  iter_values(%[[CARRY:.+]] = %[[INITIAL_CARRY]]) -> (tile<2x8xf32>) {
//        CHECK:   %[[MULI:.+]] = muli %[[IV]], %[[C16]] : tile<i32>
//        CHECK:   %[[VALUES_PTR:.+]] = offset %[[VALUES]], %[[MULI]] : tile<ptr<f32>>, tile<i32> -> tile<ptr<f32>>
//        CHECK:   %[[VALUES_VIEW:.+]] = make_tensor_view %[[VALUES_PTR]], shape = [2, 8], strides = [8, 1]
//        CHECK:   %[[VALUES_PARTITION:.+]] = make_partition_view %[[VALUES_VIEW]]
//        CHECK:   %[[READ_VALUES:.+]], %{{.+}} = load_view_tko weak %[[VALUES_PARTITION]][%[[C0]], %[[C0]]]
//        CHECK:   %[[NEXT:.+]] = addf %[[CARRY]], %[[READ_VALUES]] : tile<2x8xf32>
//        CHECK:   continue %[[NEXT]] : tile<2x8xf32>
//        CHECK: %[[OUTPUT:.+]] = exp %[[FINAL_CARRY]] : tile<2x8xf32>
//        CHECK: %[[OUT_VIEW:.+]] = make_tensor_view %[[OUT]], shape = [2, 8], strides = [8, 1]
//        CHECK: %[[OUT_PARTITION:.+]] = make_partition_view %[[OUT_VIEW]]
//        CHECK: store_view_tko weak %[[OUTPUT]], %[[OUT_PARTITION]][%[[C0]], %[[C0]]]
//        CHECK: return

//   GRID-LABEL: func.func @pointwise_carry_scan_dispatch(
//    GRID-SAME:  %[[VALUES:[^ ,:]+]]: memref<5x2x8xf32, strided<[16, 8, 1]>
//    GRID-SAME:  %[[INITIAL_A:[^ ,:]+]]: memref<2x8xf32, strided<[8, 1]>
//    GRID-SAME:  %[[INITIAL_B:[^ ,:]+]]: memref<2x8xf32, strided<[8, 1]>
//    GRID-SAME:  %[[OUT:[^ ,:]+]]: memref<2x8xf32, strided<[8, 1]>
//         GRID: scf.forall (%{{.+}}) in (1) {
//         GRID:   %[[READ_INITIAL_A:.+]] = nv_tensor_ir.load %[[INITIAL_A]] tile_indices [0, 0], alignment 4 : memref<{{.*}}> to tensor<2x8xf32>
//         GRID:   %[[READ_INITIAL_B:.+]] = nv_tensor_ir.load %[[INITIAL_B]] tile_indices [0, 0], alignment 4 : memref<{{.*}}> to tensor<2x8xf32>
//         GRID:   %[[INITIAL_CARRY:.+]] = nv_tensor_ir.add %[[READ_INITIAL_A]], %[[READ_INITIAL_B]] : tensor<2x8xf32>
//     GRID-DAG:   %[[C0:.+]] = arith.constant 0 : index
//     GRID-DAG:   %[[C5:.+]] = arith.constant 5 : index
//     GRID-DAG:   %[[C1:.+]] = arith.constant 1 : index
//         GRID:   %[[FINAL_CARRY:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C5]] step %[[C1]]
//    GRID-SAME:      iter_args(%[[CARRY:.+]] = %[[INITIAL_CARRY]]) -> (tensor<2x8xf32>)
//         GRID:     %[[C16:.+]] = arith.constant 16 : index
//         GRID:     %[[MULI:.+]] = arith.muli %[[IV]], %[[C16]] : index
//         GRID:     %[[C0_2:.+]] = arith.constant 0 : index
//         GRID:     %[[ADDI:.+]] = arith.addi %[[C0_2]], %[[MULI]] : index
//         GRID: %[[VIEW2:.*]] = memref.reinterpret_cast %[[VALUES]] to offset: [%[[ADDI]]], sizes: [2, 8], strides: [8, 1]
//         GRID:     %[[READ_VALUES:.+]] = nv_tensor_ir.load %[[VIEW2]] tile_indices [0, 0], alignment 4 : memref<{{.*}}> to tensor<2x8xf32>
//         GRID:     %[[NEXT:.+]] = nv_tensor_ir.add %[[CARRY]], %[[READ_VALUES]] : tensor<2x8xf32>
//         GRID:     scf.yield %[[NEXT]] : tensor<2x8xf32>
//         GRID:   %[[OUTPUT:.+]] = nv_tensor_ir.exp %[[FINAL_CARRY]] : tensor<2x8xf32>
//         GRID:   nv_tensor_ir.store %[[OUTPUT]], %[[OUT]] tile_indices [0, 0], alignment 4 : tensor<2x8xf32>, memref<{{.*}}>
//         GRID: return

nv_tensor_ir.graph @pointwise_carry_scan(
    %values: tensor<5x2x8xf32>, %initial0: tensor<2x8xf32>,
    %initial1: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %initial = nv_tensor_ir.add %initial0, %initial1 : tensor<2x8xf32>
  %final = "nv_tensor_ir.scan_slice"(%initial, %values) <{
      output_dimensions = array<i64>, reverse = false,
      scan_dimensions = array<i64: 0>
    }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    %next = nv_tensor_ir.add %carry, %value : tensor<2x8xf32>
    nv_tensor_ir.yield %next : tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> tensor<2x8xf32>
  %out = nv_tensor_ir.exp %final : tensor<2x8xf32>
  results %out : tensor<2x8xf32>
}

// -----

// Backward: reverse order, prefix broadcast, full innermost reduction,
// mixed output dtypes and unit carry axes. Cast the final carry after scan_slice;
// callers reshape it outside the kernel.
// A scan_slice reduction uses the complete carry tile even when the ordinary
// reduction strip-mining option is smaller than that tile.
// LAYOUT-LABEL: layout normalization for @state_passing_scan_slice_reverse
//  LAYOUT-NEXT: iteration_space: #nv_tensor_ir.scan_slice_source<
//  LAYOUT-SAME: "(1,2,4,8):(0,32,8,1)"
//  LAYOUT-SAME: output_dims = [2, 2],

//  CHECK-LABEL: entry @state_passing_scan_slice_reverse(
//   CHECK-SAME:  %[[PREVIOUS:[^ ,:]+]]: tile<ptr<bf16>>,
//   CHECK-SAME:  %[[DA:[^ ,:]+]]: tile<ptr<f32>>,
//   CHECK-SAME:  %[[DPREVIOUS:[^ ,:]+]]: tile<ptr<bf16>>,
//   CHECK-SAME:  %[[DFINAL:[^ ,:]+]]: tile<ptr<bf16>>,
//   CHECK-SAME:  %[[DCHUNK:[^ ,:]+]]: tile<ptr<bf16>>,
//   CHECK-SAME:  %[[DDA:[^ ,:]+]]: tile<ptr<f32>>,
//   CHECK-SAME:  %[[FINAL_OUT:[^ ,:]+]]: tile<ptr<bf16>>)
//    CHECK-DAG: %[[C32:.+]] = constant <i32: 32> : tile<i32>
//    CHECK-DAG: %[[C64:.+]] = constant <i32: 64> : tile<i32>
//    CHECK-DAG: %[[C3:.+]] = constant <i32: 3> : tile<i32>
//    CHECK-DAG: %[[C1:.+]] = constant <i32: 1> : tile<i32>
//    CHECK-DAG: %[[C4:.+]] = constant <i32: 4> : tile<i32>
//    CHECK-DAG: %[[C0:.+]] = constant <i32: 0> : tile<i32>
//    CHECK-NOT: get_tile_block_id
//        CHECK: %[[DFINAL_VIEW:.+]] = make_tensor_view
//   CHECK-SAME: %[[DFINAL]], shape = [1, 2, 4, 8], strides = [{{[0-9]+}}, 32, 8, 1]
//        CHECK: %[[DFINAL_PARTITION:.+]] = make_partition_view %[[DFINAL_VIEW]]
//        CHECK: %[[READ_DFINAL:.+]], %{{[^ ,]+}} = load_view_tko weak %[[DFINAL_PARTITION]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
//        CHECK: %[[INITIAL_CARRY:.+]] = ftof %[[READ_DFINAL]] : tile<1x2x4x8xbf16> -> tile<1x2x4x8xf32>
//        CHECK: %[[FINAL_CARRY:.+]] = for %[[IV:.+]] in (%[[C0]] to %[[C4]], step %[[C1]]) : tile<i32>
//   CHECK-SAME:    iter_values(%[[CARRY:.+]] = %[[INITIAL_CARRY]]) -> (tile<1x2x4x8xf32>) {
//        CHECK:   %[[STEP:.+]] = subi %[[C3]], %[[IV]] : tile<i32>
//        CHECK:   %[[MULI:.+]] = muli %[[STEP]], %[[C64]] : tile<i32>
//        CHECK:   %[[PREVIOUS_PTR:.+]] = offset %[[PREVIOUS]], %[[MULI]]
//        CHECK:   %[[PREVIOUS_VIEW:.+]] = make_tensor_view %[[PREVIOUS_PTR]], shape = [1, 2, 4, 8], strides = [{{[0-9]+}}, 32, 8, 1]
//        CHECK:   %[[PREVIOUS_PARTITION:.+]] = make_partition_view %[[PREVIOUS_VIEW]]
//        CHECK:   %[[READ_PREVIOUS:.+]], %{{.+}} = load_view_tko weak %[[PREVIOUS_PARTITION]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
//        CHECK:   %[[DA_PTR:.+]] = offset %[[DA]], %[[STEP]]
//        CHECK:   %[[DA_VIEW:.+]] = make_tensor_view %[[DA_PTR]], shape = [1, 2], strides = [8, 4]
//        CHECK:   %[[DA_PARTITION:.+]] = make_partition_view %[[DA_VIEW]]
//        CHECK:   %[[READ_DA:.+]], %{{.+}} = load_view_tko weak %[[DA_PARTITION]][%[[C0]], %[[C0]]]
//        CHECK:   %[[DPREVIOUS_PTR:.+]] = offset %[[DPREVIOUS]], %[[MULI]]
//        CHECK:   %[[DPREVIOUS_VIEW:.+]] = make_tensor_view %[[DPREVIOUS_PTR]], shape = [1, 2, 4, 8], strides = [{{[0-9]+}}, 32, 8, 1]
//        CHECK:   %[[DPREVIOUS_PARTITION:.+]] = make_partition_view %[[DPREVIOUS_VIEW]]
//        CHECK:   %[[READ_DPREVIOUS:.+]], %{{.+}} = load_view_tko weak %[[DPREVIOUS_PARTITION]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
//        CHECK:   %[[PREV_F32:.+]] = ftof %[[READ_PREVIOUS]] : tile<1x2x4x8xbf16> -> tile<1x2x4x8xf32>
//        CHECK:   %[[DPREV_F32:.+]] = ftof %[[READ_DPREVIOUS]] : tile<1x2x4x8xbf16> -> tile<1x2x4x8xf32>
//        CHECK:   %[[DA_EXPANDED:.+]] = reshape %[[READ_DA]] : tile<1x2xf32> -> tile<1x2x1x1xf32>
//        CHECK:   %[[BROADCAST:.+]] = broadcast %[[DA_EXPANDED]] : tile<1x2x1x1xf32> -> tile<1x2x4x8xf32>
//        CHECK:   %[[DECAY:.+]] = exp %[[BROADCAST]] : tile<1x2x4x8xf32>
//        CHECK:   %[[DCHUNK_STEP:.+]] = ftof %[[CARRY]] : tile<1x2x4x8xf32> -> tile<1x2x4x8xbf16>
//        CHECK:   %[[CARRY_PREV:.+]] = mulf %[[CARRY]], %[[PREV_F32]] : tile<1x2x4x8xf32>
//        CHECK:   %[[DDA_PRODUCT:.+]] = mulf %[[CARRY_PREV]], %[[DECAY]] : tile<1x2x4x8xf32>
//        CHECK:   %[[REDUCE:.+]] = reduce %[[DDA_PRODUCT]] dim=3 identities=[0.000000e+00 : f32] : tile<1x2x4x8xf32> -> tile<1x2x4xf32>
//        CHECK:      (%[[REDUCE_LHS:.+]]: tile<f32>, %[[REDUCE_RHS:.+]]: tile<f32>) {
//        CHECK:     %[[REDUCE_SUM:.+]] = addf %[[REDUCE_RHS]], %[[REDUCE_LHS]] : tile<f32>
//        CHECK:     yield %[[REDUCE_SUM]] : tile<f32>
//        CHECK:   %[[DDA_STEP:.+]] = reshape %[[REDUCE]] : tile<1x2x4xf32> -> tile<1x2x4x1xf32>
//        CHECK:   %[[DECAYED_CARRY:.+]] = mulf %[[CARRY]], %[[DECAY]] : tile<1x2x4x8xf32>
//        CHECK:   %[[NEXT:.+]] = addf %[[DECAYED_CARRY]], %[[DPREV_F32]] : tile<1x2x4x8xf32>
//        CHECK:   %[[MULI_2:.+]] = muli %[[STEP]], %[[C32]] : tile<i32>
//        CHECK:   %[[DCHUNK_PTR:.+]] = offset %[[DCHUNK]], %[[MULI_2]]
//        CHECK:   %[[DCHUNK_VIEW:.+]] = make_tensor_view %[[DCHUNK_PTR]], shape = [1, 2, 4, 8], strides = [{{[0-9]+}}, 128, 8, 1]
//        CHECK:   %[[DCHUNK_PARTITION:.+]] = make_partition_view %[[DCHUNK_VIEW]]
//        CHECK:   store_view_tko weak %[[DCHUNK_STEP]], %[[DCHUNK_PARTITION]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
//        CHECK:   %[[MULI_3:.+]] = muli %[[STEP]], %[[C4]] : tile<i32>
//        CHECK:   %[[DDA_PTR:.+]] = offset %[[DDA]], %[[MULI_3]] : tile<ptr<f32>>, tile<i32> -> tile<ptr<f32>>
//        CHECK:   %[[DDA_VIEW:.+]] = make_tensor_view %[[DDA_PTR]], shape = [1, 2, 4, 1], strides = [{{[0-9]+}}, 16, 1, {{[0-9]+}}]
//        CHECK:   %[[DDA_PARTITION:.+]] = make_partition_view %[[DDA_VIEW]]
//        CHECK:   store_view_tko weak %[[DDA_STEP]], %[[DDA_PARTITION]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
//        CHECK:   continue %[[NEXT]] : tile<1x2x4x8xf32>
//        CHECK: %[[FINAL_BF16:.+]] = ftof %[[FINAL_CARRY]] : tile<1x2x4x8xf32> -> tile<1x2x4x8xbf16>
//        CHECK: %[[FINAL_OUT_VIEW:.+]] = make_tensor_view %[[FINAL_OUT]], shape = [1, 2, 4, 8], strides = [{{[0-9]+}}, 32, 8, 1]
//        CHECK: %[[FINAL_OUT_PARTITION:.+]] = make_partition_view %[[FINAL_OUT_VIEW]]
//        CHECK: store_view_tko weak %[[FINAL_BF16]], %[[FINAL_OUT_PARTITION]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
//        CHECK: return

//   GRID-LABEL: func.func @state_passing_scan_slice_reverse_dispatch(
//    GRID-SAME:  %[[PREVIOUS:[^ ,:]+]]: memref<1x4x2x4x8xbf16, strided<[256, 64, 32, 8, 1]>
//    GRID-SAME:  %[[DA:[^ ,:]+]]: memref<1x2x4xf32, strided<[8, 4, 1]>
//    GRID-SAME:  %[[DPREVIOUS:[^ ,:]+]]: memref<1x4x2x4x8xbf16, strided<[256, 64, 32, 8, 1]>
//    GRID-SAME:  %[[DFINAL:[^ ,:]+]]: memref<1x2x4x8xbf16, strided<[64, 32, 8, 1]>
//    GRID-SAME:  %[[DCHUNK:[^ ,:]+]]: memref<1x2x4x4x8xbf16, strided<[256, 128, 32, 8, 1]>
//    GRID-SAME:  %[[DDA:[^ ,:]+]]: memref<1x2x4x4x1xf32, strided<[32, 16, 4, 1, 1]>
//    GRID-SAME:  %[[FINAL_OUT:[^ ,:]+]]: memref<1x2x4x8xbf16, strided<[64, 32, 8, 1]>
//         GRID: scf.forall (%{{.+}}) in (1) {
//         GRID: %[[VIEW3:.*]] = memref.reinterpret_cast %[[DFINAL]] to offset: [0], sizes: [1, 2, 4, 8], strides: [2, 32, 8, 1]
//         GRID:   %[[READ_DFINAL:.+]] = nv_tensor_ir.load %[[VIEW3]] tile_indices [0, 0, 0, 0], alignment 2
//         GRID:   %[[INITIAL_CARRY:.+]] = nv_tensor_ir.convert %[[READ_DFINAL]] : tensor<1x2x4x8xbf16> -> tensor<1x2x4x8xf32>
//     GRID-DAG:   %[[C0:.+]] = arith.constant 0 : index
//     GRID-DAG:   %[[C4:.+]] = arith.constant 4 : index
//     GRID-DAG:   %[[C1:.+]] = arith.constant 1 : index
//         GRID:   %[[FINAL_CARRY:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
//    GRID-SAME:      iter_args(%[[CARRY:.+]] = %[[INITIAL_CARRY]]) -> (tensor<1x2x4x8xf32>) {
//         GRID:     %[[C3:.+]] = arith.constant 3 : index
//         GRID:     %[[STEP:.+]] = arith.subi %[[C3]], %[[IV]] : index
//         GRID:     %[[C64:.+]] = arith.constant 64 : index
//         GRID:     %[[MULI:.+]] = arith.muli %[[STEP]], %[[C64]] : index
//         GRID:     %[[C0_2:.+]] = arith.constant 0 : index
//         GRID:     %[[ADDI:.+]] = arith.addi %[[C0_2]], %[[MULI]] : index
//         GRID: %[[PREVIOUS_VIEW:.*]] = memref.reinterpret_cast %[[PREVIOUS]] to offset: [%[[ADDI]]], sizes: [1, 2, 4, 8], strides: [256, 32, 8, 1]
//         GRID: %[[READ_PREVIOUS:.+]] = nv_tensor_ir.load %[[PREVIOUS_VIEW]] tile_indices [0, 0, 0, 0], alignment 2
//         GRID:     %[[C1_2:.+]] = arith.constant 1 : index
//         GRID:     %[[MULI_2:.+]] = arith.muli %[[STEP]], %[[C1_2]] : index
//         GRID:     %[[C0_3:.+]] = arith.constant 0 : index
//         GRID:     %[[ADDI_2:.+]] = arith.addi %[[C0_3]], %[[MULI_2]] : index
//         GRID: %[[VIEW4:.*]] = memref.reinterpret_cast %[[DA]] to offset: [%[[ADDI_2]]], sizes: [1, 2], strides: [8, 4]
//         GRID:     %[[READ_DA:.+]] = nv_tensor_ir.load %[[VIEW4]] tile_indices [0, 0], alignment 4
//         GRID:     %[[C64_2:.+]] = arith.constant 64 : index
//         GRID:     %[[MULI_3:.+]] = arith.muli %[[STEP]], %[[C64_2]] : index
//         GRID:     %[[C0_4:.+]] = arith.constant 0 : index
//         GRID:     %[[ADDI_3:.+]] = arith.addi %[[C0_4]], %[[MULI_3]] : index
//         GRID: %[[DPREVIOUS_VIEW:.*]] = memref.reinterpret_cast %[[DPREVIOUS]] to offset: [%[[ADDI_3]]], sizes: [1, 2, 4, 8], strides: [256, 32, 8, 1]
//         GRID: %[[READ_DPREVIOUS:.+]] = nv_tensor_ir.load %[[DPREVIOUS_VIEW]] tile_indices [0, 0, 0, 0], alignment 2
//         GRID:     %[[PREV_F32:.+]] = nv_tensor_ir.convert %[[READ_PREVIOUS]] : tensor<1x2x4x8xbf16> -> tensor<1x2x4x8xf32>
//         GRID:     %[[DPREV_F32:.+]] = nv_tensor_ir.convert %[[READ_DPREVIOUS]] : tensor<1x2x4x8xbf16> -> tensor<1x2x4x8xf32>
//         GRID:     %[[RESHAPE:.+]] = nv_tensor_ir.reshape %[[READ_DA]] : tensor<1x2xf32> -> tensor<1x2x1x1xf32>
//         GRID:     %[[BROADCAST:.+]] = nv_tensor_ir.broadcast %[[RESHAPE]] : tensor<1x2x1x1xf32> -> tensor<1x2x4x8xf32>
//         GRID:     %[[DECAY:.+]] = nv_tensor_ir.exp %[[BROADCAST]] : tensor<1x2x4x8xf32>
//         GRID:     %[[DCHUNK_STEP:.+]] = nv_tensor_ir.convert %[[CARRY]] : tensor<1x2x4x8xf32> -> tensor<1x2x4x8xbf16>
//         GRID:     %[[CARRY_PREV:.+]] = nv_tensor_ir.mul %[[CARRY]], %[[PREV_F32]] : tensor<1x2x4x8xf32>
//         GRID:     %[[DDA_PRODUCT:.+]] = nv_tensor_ir.mul %[[CARRY_PREV]], %[[DECAY]] : tensor<1x2x4x8xf32>
//         GRID:     %[[REDUCE__MUL_41_:.+]] = nv_tensor_ir.reduce(%[[DDA_PRODUCT]]) <dimensions = [3], reduction_mode = <add>> : tensor<1x2x4x8xf32> -> tensor<1x2x4x1xf32>
//         GRID:     %[[DECAYED_CARRY:.+]] = nv_tensor_ir.mul %[[CARRY]], %[[DECAY]] : tensor<1x2x4x8xf32>
//         GRID:     %[[NEXT:.+]] = nv_tensor_ir.add %[[DPREV_F32]], %[[DECAYED_CARRY]] : tensor<1x2x4x8xf32>
//         GRID:     %[[C32:.+]] = arith.constant 32 : index
//         GRID:     %[[MULI_4:.+]] = arith.muli %[[STEP]], %[[C32]] : index
//         GRID:     %[[C0_5:.+]] = arith.constant 0 : index
//         GRID:     %[[ADDI_4:.+]] = arith.addi %[[C0_5]], %[[MULI_4]] : index
//         GRID: %[[VIEW5:.*]] = memref.reinterpret_cast %[[DCHUNK]] to offset: [%[[ADDI_4]]], sizes: [1, 2, 4, 8], strides: [256, 128, 8, 1]
//         GRID:     nv_tensor_ir.store %[[DCHUNK_STEP]], %[[VIEW5]] tile_indices [0, 0, 0, 0], alignment 1
//         GRID:     %[[C4_2:.+]] = arith.constant 4 : index
//         GRID:     %[[MULI_5:.+]] = arith.muli %[[STEP]], %[[C4_2]] : index
//         GRID:     %[[C0_6:.+]] = arith.constant 0 : index
//         GRID:     %[[ADDI_5:.+]] = arith.addi %[[C0_6]], %[[MULI_5]] : index
//         GRID: %[[VIEW6:.*]] = memref.reinterpret_cast %[[DDA]] to offset: [%[[ADDI_5]]], sizes: [1, 2, 4, 1], strides: [32, 16, 1, 1]
//         GRID:     nv_tensor_ir.store %[[REDUCE__MUL_41_]], %[[VIEW6]] tile_indices [0, 0, 0, 0], alignment 1
//         GRID:     scf.yield %[[NEXT]] : tensor<1x2x4x8xf32>
//         GRID:   %[[FINAL_BF16:.+]] = nv_tensor_ir.convert %[[FINAL_CARRY]] : tensor<1x2x4x8xf32> -> tensor<1x2x4x8xbf16>
//         GRID: %[[VIEW7:.*]] = memref.reinterpret_cast %[[FINAL_OUT]] to offset: [0], sizes: [1, 2, 4, 8], strides: [2, 32, 8, 1]
//         GRID:   nv_tensor_ir.store %[[FINAL_BF16]], %[[VIEW7]] tile_indices [0, 0, 0, 0], alignment 2
//         GRID: return

nv_tensor_ir.graph @state_passing_scan_slice_reverse(
    %previous: tensor<1x4x2x4x8xbf16>, %dA: tensor<1x2x4xf32>,
    %dprevious: tensor<1x4x2x4x8xbf16>, %dfinal: tensor<1x2x4x8xbf16>
  ) -> (tensor<1x2x4x4x8xbf16>, tensor<1x2x4x4x1xf32>,
        tensor<1x2x4x8xbf16>) {
  %carry = convert %dfinal
      : tensor<1x2x4x8xbf16> -> tensor<1x2x4x8xf32>
  %final, %dchunk, %ddA = "nv_tensor_ir.scan_slice"(
      %carry, %previous, %dA, %dprevious) <{
        output_dimensions = array<i64: 2, 2>, reverse = true,
        scan_dimensions = array<i64: 1, 2, 1>
      }> ({
  ^bb0(%state: tensor<1x2x4x8xf32>, %prev: tensor<1x2x4x8xbf16>,
       %chunk_dA: tensor<1x2xf32>, %dprev: tensor<1x2x4x8xbf16>):
    %prev_f32 = nv_tensor_ir.convert %prev
        : tensor<1x2x4x8xbf16> -> tensor<1x2x4x8xf32>
    %dprev_f32 = nv_tensor_ir.convert %dprev
        : tensor<1x2x4x8xbf16> -> tensor<1x2x4x8xf32>
    %decay_scalar = nv_tensor_ir.reshape %chunk_dA
        : tensor<1x2xf32> -> tensor<1x2x1x1xf32>
    %decay_wide = nv_tensor_ir.broadcast %decay_scalar
        : tensor<1x2x1x1xf32> -> tensor<1x2x4x8xf32>
    %decay = nv_tensor_ir.exp %decay_wide : tensor<1x2x4x8xf32>
    %dchunk_step = nv_tensor_ir.convert %state
        : tensor<1x2x4x8xf32> -> tensor<1x2x4x8xbf16>
    %product0 = nv_tensor_ir.mul %state, %prev_f32
        : tensor<1x2x4x8xf32>
    %product1 = nv_tensor_ir.mul %product0, %decay
        : tensor<1x2x4x8xf32>
    %ddA_step = nv_tensor_ir.reduce(%product1)
        <dimensions = [3], reduction_mode = <add>>
        : tensor<1x2x4x8xf32> -> tensor<1x2x4x1xf32>
    %decayed = nv_tensor_ir.mul %state, %decay
        : tensor<1x2x4x8xf32>
    %next = nv_tensor_ir.add %dprev_f32, %decayed
        : tensor<1x2x4x8xf32>
    nv_tensor_ir.yield %next, %dchunk_step, %ddA_step
        : tensor<1x2x4x8xf32>, tensor<1x2x4x8xbf16>,
          tensor<1x2x4x1xf32>
  }) : (tensor<1x2x4x8xf32>, tensor<1x4x2x4x8xbf16>,
        tensor<1x2x4xf32>, tensor<1x4x2x4x8xbf16>)
       -> (tensor<1x2x4x8xf32>, tensor<1x2x4x4x8xbf16>,
           tensor<1x2x4x4x1xf32>)
  %final_bf16 = nv_tensor_ir.convert %final
      : tensor<1x2x4x8xf32> -> tensor<1x2x4x8xbf16>
  results %dchunk, %ddA, %final_bf16
      : tensor<1x2x4x4x8xbf16>, tensor<1x2x4x4x1xf32>, tensor<1x2x4x8xbf16>
}
