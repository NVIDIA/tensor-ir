// RUN: tensor_ir-opt %s \
// RUN:   --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid))' \
// RUN:   --split-input-file \
// RUN:   | FileCheck %s \
// RUN:   --implicit-check-not="= nv_tensor_ir.scan_slice(" --implicit-check-not="layout =" --implicit-check-not="nv_tensor_ir.iter_space_id"

// Expand the recurrence into a carry loop with per-step input/output accesses.
// CHECK-LABEL: func.func @streaming_metadata_dispatch(
//  CHECK-SAME: %[[INITIAL:[^:]+]]: memref<2x8xf32,
//  CHECK-SAME: %[[VALUES:[^:]+]]: memref<5x2x8xf32,
//  CHECK-SAME: %[[SCALES:[^:]+]]: memref<2x5xf32,
//  CHECK-SAME: %[[HISTORY:[^:]+]]: memref<2x5x8xbf16,
//  CHECK-SAME: %[[TOTALS:[^:]+]]: memref<5x2x1xf32,
//  CHECK-SAME: %[[FINAL:[^:]+]]: memref<2x8xf32,
//       CHECK: %[[IPTR:.*]] = ptr.to_ptr %[[INITIAL]]
//       CHECK: %[[VPTR:.*]] = ptr.to_ptr %[[VALUES]]
//       CHECK: %[[SPTR:.*]] = ptr.to_ptr %[[SCALES]]
//       CHECK: %[[HPTR:.*]] = ptr.to_ptr %[[HISTORY]]
//       CHECK: %[[TPTR:.*]] = ptr.to_ptr %[[TOTALS]]
//       CHECK: %[[FPTR:.*]] = ptr.to_ptr %[[FINAL]]
//       CHECK: scf.forall (%[[ROW:.*]]) in (2) {
//       CHECK: %[[INIT:.*]] = nv_tensor_ir.load %[[IPTR]][%[[ROW]], 0] view offset: [0], sizes: [2, 8],
//  CHECK-SAME: strides: [8, 1]
//  CHECK-SAME: to tensor<1x8xf32>
//       CHECK: %[[NEG:.*]] = nv_tensor_ir.neg %[[INIT]]
//       CHECK: %[[ZERO:.*]] = arith.constant 0 : index
//       CHECK: %[[FIVE:.*]] = arith.constant 5 : index
//       CHECK: %[[ONE:.*]] = arith.constant 1 : index
//       CHECK: %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[ZERO]] to %[[FIVE]] step %[[ONE]] iter_args(%[[CARRY:.*]] =
//  CHECK-SAME: %[[NEG]]) -> (tensor<1x8xf32>) {
//       CHECK: %[[LAST:.*]] = arith.constant 4 : index
//       CHECK: %[[STEP:.*]] = arith.subi %[[LAST]], %[[IV]] : index
//       CHECK: %[[STRIDE:.*]] = arith.constant 64 : index
//       CHECK: %[[VOFF:.*]] = arith.muli %[[STEP]], %[[STRIDE]] : index
//       CHECK: %[[VZERO:.*]] = arith.constant 0 : index
//       CHECK: %[[VBASE:.*]] = arith.addi %[[VZERO]], %[[VOFF]] : index
//       CHECK: %[[VALUE:.*]] = nv_tensor_ir.load %[[VPTR]][%[[ROW]], 0] view offset: [%[[VBASE]]],
//  CHECK-SAME: sizes: [2, 8], strides: [16, 2], alignment: 256
//  CHECK-SAME: to tensor<1x8xf32>
//       CHECK: %[[SSTRIDE:.*]] = arith.constant 1 : index
//       CHECK: %[[SOFF:.*]] = arith.muli %[[STEP]], %[[SSTRIDE]] : index
//       CHECK: %[[SZERO:.*]] = arith.constant 0 : index
//       CHECK: %[[SBASE:.*]] = arith.addi %[[SZERO]], %[[SOFF]] : index
//       CHECK: %[[SCALE:.*]] = nv_tensor_ir.load %[[SPTR]][%[[ROW]]] view offset: [%[[SBASE]]], sizes: [2],
//  CHECK-SAME: strides: [5], alignment: 4
//  CHECK-SAME: to tensor<1xf32>
//       CHECK: %[[COLUMN:.*]] = nv_tensor_ir.reshape %[[SCALE]] : tensor<1xf32> -> tensor<1x1xf32>
//       CHECK: %[[WIDE:.*]] = nv_tensor_ir.broadcast %[[COLUMN]] : tensor<1x1xf32> -> tensor<1x8xf32>
//       CHECK: %[[SCALED:.*]] = nv_tensor_ir.mul %[[VALUE]], %[[WIDE]]
//       CHECK: %[[NEXT:.*]] = nv_tensor_ir.add %[[CARRY]], %[[SCALED]]
//       CHECK: %[[LOW:.*]] = nv_tensor_ir.convert %[[NEXT]] : tensor<1x8xf32> -> tensor<1x8xbf16>
//       CHECK: %[[TOTAL:.*]] = nv_tensor_ir.reduce(%[[NEXT]]) <dimensions = [1], reduction_mode = <add>> {nv_tensor_ir.scan_slice_reduction}
//  CHECK-SAME: : tensor<1x8xf32> -> tensor<1x1xf32>
//       CHECK: %[[HSTRIDE:.*]] = arith.constant 8 : index
//       CHECK: %[[HOFF:.*]] = arith.muli %[[STEP]], %[[HSTRIDE]] : index
//       CHECK: %[[HZERO:.*]] = arith.constant 0 : index
//       CHECK: %[[HBASE:.*]] = arith.addi %[[HZERO]], %[[HOFF]] : index
//       CHECK: nv_tensor_ir.store %[[LOW]], %[[HPTR]][%[[ROW]], 0] view offset: [%[[HBASE]]], sizes: [2, 8],
//  CHECK-SAME: strides: [40, 1]
//       CHECK: %[[TSTRIDE:.*]] = arith.constant 2 : index
//       CHECK: %[[TOFF:.*]] = arith.muli %[[STEP]], %[[TSTRIDE]] : index
//       CHECK: %[[TZERO:.*]] = arith.constant 0 : index
//       CHECK: %[[TBASE:.*]] = arith.addi %[[TZERO]], %[[TOFF]] : index
//       CHECK: nv_tensor_ir.store %[[TOTAL]], %[[TPTR]][%[[ROW]], 0] view offset: [%[[TBASE]]], sizes: [2, 1],
//  CHECK-SAME: strides: [1, 1]
//  CHECK-NEXT: scf.yield %[[NEXT]] : tensor<1x8xf32>
//  CHECK-NEXT: }
//   CHECK-NOT: nv_tensor_ir.store
//       CHECK: %[[OUT:.*]] = nv_tensor_ir.exp %[[LOOP]]
//       CHECK: nv_tensor_ir.store %[[OUT]], %[[FPTR]][%[[ROW]], 0] view offset: [0], sizes: [2, 8],
//  CHECK-SAME: strides: [8, 1]
//   CHECK-NOT: nv_tensor_ir.store
//       CHECK: return

nv_tensor_ir.graph @streaming_metadata(
    %initial: tensor<2x8xf32>,
    %values: tensor<5x2x8xf32> {nv_tensor_ir.stride = "(64,16,2)", nv_tensor_ir.alignment = 256 : i64},
    %scales: tensor<2x5xf32> {nv_tensor_ir.alignment = 128 : i64})
    -> (tensor<2x5x8xbf16>, tensor<5x2x1xf32>, tensor<2x8xf32>)
    attributes {tile_size = array<i32: 1, 8>} {
  %initial_carry = neg %initial : tensor<2x8xf32>
  %final, %history, %totals = scan_slice(%initial_carry, %values, %scales)
      scan_dimensions = [0, 1] output_dimensions = [1, 0] {reverse = true}
      : (tensor<2x8xf32>, tensor<5x2x8xf32>, tensor<2x5xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>, %scale: tensor<2xf32>):
      %column = reshape %scale : tensor<2xf32> -> tensor<2x1xf32>
      %wide = broadcast %column : tensor<2x1xf32> -> tensor<2x8xf32>
      %scaled = mul %value, %wide : tensor<2x8xf32>
      %next = add %carry, %scaled : tensor<2x8xf32>
      %low = convert %next : tensor<2x8xf32> -> tensor<2x8xbf16>
      %total = reduce(%next) <dimensions = [1], reduction_mode = <add>>
          : tensor<2x8xf32> -> tensor<2x1xf32>
      yield %next, %low, %total : tensor<2x8xf32>, tensor<2x8xbf16>, tensor<2x1xf32>
  }
  %out = exp %final : tensor<2x8xf32>
  results %history, %totals, %out : tensor<2x5x8xbf16>, tensor<5x2x1xf32>, tensor<2x8xf32>
}

// -----

// A composite initial carry, no history, and a shared final carry consumer.
// CHECK-LABEL: func.func @composite_carry_dispatch(
//  CHECK-SAME: %[[ABUF:[^:]+]]: memref<2x8xf32,
//  CHECK-SAME: %[[BBUF:[^:]+]]: memref<2x8xf32,
//  CHECK-SAME: %[[VALUES:[^:]+]]: memref<5x2x8xf32,
//  CHECK-SAME: %[[FINAL:[^:]+]]: memref<2x8xf32,
//       CHECK: %[[APTR:.*]] = ptr.to_ptr %[[ABUF]]
//       CHECK: %[[BPTR:.*]] = ptr.to_ptr %[[BBUF]]
//       CHECK: %[[VPTR:.*]] = ptr.to_ptr %[[VALUES]]
//       CHECK: %[[FPTR:.*]] = ptr.to_ptr %[[FINAL]]
//       CHECK: scf.forall (%[[ROW:.*]]) in (2) {
//       CHECK: %[[A:.*]] = nv_tensor_ir.load %[[APTR]][%[[ROW]], 0] view offset: [0], sizes: [2, 8],
//  CHECK-SAME: strides: [8, 1]
//       CHECK: %[[B:.*]] = nv_tensor_ir.load %[[BPTR]][%[[ROW]], 0] view offset: [0], sizes: [2, 8],
//  CHECK-SAME: strides: [8, 1]
//       CHECK: %[[INITIAL:.*]] = nv_tensor_ir.add %[[A]], %[[B]]
//       CHECK: %[[ZERO:.*]] = arith.constant 0 : index
//       CHECK: %[[FIVE:.*]] = arith.constant 5 : index
//       CHECK: %[[ONE:.*]] = arith.constant 1 : index
//       CHECK: %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[ZERO]] to %[[FIVE]] step %[[ONE]] iter_args(%[[CARRY:.*]] =
//  CHECK-SAME: %[[INITIAL]]) -> (tensor<1x8xf32>) {
//   CHECK-NOT: arith.subi
//       CHECK: %[[STRIDE:.*]] = arith.constant 16 : index
//       CHECK: %[[OFF:.*]] = arith.muli %[[IV]], %[[STRIDE]] : index
//       CHECK: %[[ZERO:.*]] = arith.constant 0 : index
//       CHECK: %[[BASE:.*]] = arith.addi %[[ZERO]], %[[OFF]] : index
//       CHECK: %[[VALUE:.*]] = nv_tensor_ir.load %[[VPTR]][%[[ROW]], 0] view offset: [%[[BASE]]], sizes: [2, 8],
//  CHECK-SAME: strides: [8, 1]
//  CHECK-SAME: to tensor<1x8xf32>
//       CHECK: %[[NEXT:.*]] = nv_tensor_ir.add %[[CARRY]], %[[VALUE]]
//  CHECK-NEXT: scf.yield %[[NEXT]] : tensor<1x8xf32>
//  CHECK-NEXT: }
//   CHECK-NOT: = scf.for
//       CHECK: %[[OUT:.*]] = nv_tensor_ir.add %[[LOOP]], %[[LOOP]]
//       CHECK: nv_tensor_ir.store %[[OUT]], %[[FPTR]][%[[ROW]], 0] view offset: [0], sizes: [2, 8],
//  CHECK-SAME: strides: [8, 1]
//   CHECK-NOT: nv_tensor_ir.store
//       CHECK: return

nv_tensor_ir.graph @composite_carry(%a: tensor<2x8xf32>,
    %b: tensor<2x8xf32>, %values: tensor<5x2x8xf32>) -> tensor<2x8xf32>
    attributes {tile_size = array<i32: 1, 8>} {
  %initial = add %a, %b : tensor<2x8xf32>
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %next = add %carry, %value : tensor<2x8xf32>
      yield %next : tensor<2x8xf32>
  }
  %out = add %final, %final : tensor<2x8xf32>
  results %out : tensor<2x8xf32>
}

// -----

// Keep logical unit axes while tiling: the first non-unit dimension has tile
// size one. Both the next carry and the history reduction reuse the same add.
// CHECK-LABEL: func.func @unit_reshape_scan_slice_dispatch(
//       CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[CARRY:.*]] = {{.*}}) -> (tensor<1x8xf32>) {
//       CHECK: %[[VALUE:.*]] = nv_tensor_ir.load
//  CHECK-SAME: to tensor<1x8xf32>
//       CHECK: %[[EXPANDED_CARRY:.*]] = nv_tensor_ir.reshape %[[CARRY]] : tensor<1x8xf32> -> tensor<1x1x8xf32>
//       CHECK: %[[EXPANDED_VALUE:.*]] = nv_tensor_ir.reshape %[[VALUE]] : tensor<1x8xf32> -> tensor<1x1x8xf32>
//       CHECK: %[[SUM:.*]] = nv_tensor_ir.add %[[EXPANDED_CARRY]], %[[EXPANDED_VALUE]] : tensor<1x1x8xf32>
//   CHECK-NOT: nv_tensor_ir.add
//       CHECK: %[[NEXT:.*]] = nv_tensor_ir.reshape %[[SUM]] : tensor<1x1x8xf32> -> tensor<1x8xf32>
//       CHECK: %[[TOTAL:.*]] = nv_tensor_ir.reduce(%[[SUM]]) <dimensions = [2], reduction_mode = <add>> {nv_tensor_ir.scan_slice_reduction}
//  CHECK-SAME: : tensor<1x1x8xf32> -> tensor<1x1x1xf32>
//       CHECK: %[[ROW:.*]] = nv_tensor_ir.reshape %[[TOTAL]] : tensor<1x1x1xf32> -> tensor<1x1xf32>
//   CHECK-NOT: nv_tensor_ir.add
//       CHECK: nv_tensor_ir.store %[[ROW]],
//       CHECK: scf.yield %[[NEXT]] : tensor<1x8xf32>
//       CHECK: nv_tensor_ir.store %[[LOOP]],
nv_tensor_ir.graph @unit_reshape_scan_slice(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x1xf32>)
    attributes {tile_size = array<i32: 1, 8>} {
  %final, %history = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = [0]
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %expanded_carry = reshape %carry : tensor<2x8xf32> -> tensor<2x1x8xf32>
      %expanded_value = reshape %value : tensor<2x8xf32> -> tensor<2x1x8xf32>
      %sum = add %expanded_carry, %expanded_value : tensor<2x1x8xf32>
      %next = reshape %sum : tensor<2x1x8xf32> -> tensor<2x8xf32>
      %total = reduce(%sum) <dimensions = [2], reduction_mode = <add>>
          : tensor<2x1x8xf32> -> tensor<2x1x1xf32>
      %row = reshape %total : tensor<2x1x1xf32> -> tensor<2x1xf32>
      yield %next, %row : tensor<2x8xf32>, tensor<2x1xf32>
  }
  results %final, %history : tensor<2x8xf32>, tensor<5x2x1xf32>
}

// -----

// The native input slice has a different rank from carry. Load that rank
// directly; only the reshape explicitly present in the body removes its unit axis.
// CHECK-LABEL: func.func @native_slice_rank_dispatch(
//  CHECK-SAME: %[[INITIAL:[^:]+]]: memref<2x8xf32,
//  CHECK-SAME: %[[VALUES:[^:]+]]: memref<5x2x1x8xf32,
//       CHECK: %[[VPTR:.*]] = ptr.to_ptr %[[VALUES]]
//       CHECK: scf.forall (%[[ROW:.*]]) in (2) {
//       CHECK: %[[LOOP:.*]] = scf.for %[[STEP:.*]] = {{.*}} iter_args(%[[CARRY:.*]] = {{.*}}) -> (tensor<1x8xf32>) {
//       CHECK: %[[STRIDE:.*]] = arith.constant 16 : index
//       CHECK: %[[DELTA:.*]] = arith.muli %[[STEP]], %[[STRIDE]] : index
//       CHECK: %[[ZERO:.*]] = arith.constant 0 : index
//       CHECK: %[[OFFSET:.*]] = arith.addi %[[ZERO]], %[[DELTA]] : index
//       CHECK: %[[VALUE:.*]] = nv_tensor_ir.load %[[VPTR]][%[[ROW]], 0, 0] view offset: [%[[OFFSET]]],
//  CHECK-SAME: sizes: [2, 1, 8], strides: [8, 8, 1]
//  CHECK-SAME: to tensor<1x1x8xf32>
//       CHECK: %[[FLAT:.*]] = nv_tensor_ir.reshape %[[VALUE]] : tensor<1x1x8xf32> -> tensor<1x8xf32>
//   CHECK-NOT: nv_tensor_ir.reshape
//       CHECK: %[[NEXT:.*]] = nv_tensor_ir.add %[[CARRY]], %[[FLAT]] : tensor<1x8xf32>
//       CHECK: scf.yield %[[NEXT]] : tensor<1x8xf32>
//       CHECK: nv_tensor_ir.store %[[LOOP]],
nv_tensor_ir.graph @native_slice_rank(%initial: tensor<2x8xf32>, %values: tensor<5x2x1x8xf32>)
    -> tensor<2x8xf32> attributes {tile_size = array<i32: 1, 8>} {
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<2x8xf32>, tensor<5x2x1x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x1x8xf32>):
      %flat = reshape %value : tensor<2x1x8xf32> -> tensor<2x8xf32>
      %next = add %carry, %flat : tensor<2x8xf32>
      yield %next : tensor<2x8xf32>
  }
  results %final : tensor<2x8xf32>
}

// -----

// The native input slice has a different rank from carry. Load that rank
// directly; only the reshape explicitly present in the body removes its unit axis.
// CHECK-LABEL: func.func @native_slice_last_axis_dispatch(
//  CHECK-SAME: %[[INITIAL:[^:]+]]: memref<2x8xf32,
//  CHECK-SAME: %[[VALUES:[^:]+]]: memref<2x1x8x5xf32,
//       CHECK: %[[VPTR:.*]] = ptr.to_ptr %[[VALUES]]
//       CHECK: scf.forall (%[[ROW:.*]]) in (2) {
//       CHECK: %[[LOOP:.*]] = scf.for %[[STEP:.*]] = {{.*}} iter_args(%[[CARRY:.*]] = {{.*}}) -> (tensor<1x8xf32>) {
//       CHECK: %[[STRIDE:.*]] = arith.constant 1 : index
//       CHECK: %[[DELTA:.*]] = arith.muli %[[STEP]], %[[STRIDE]] : index
//       CHECK: %[[ZERO:.*]] = arith.constant 0 : index
//       CHECK: %[[OFFSET:.*]] = arith.addi %[[ZERO]], %[[DELTA]] : index
//       CHECK: %[[VALUE:.*]] = nv_tensor_ir.load %[[VPTR]][%[[ROW]], 0, 0] view offset: [%[[OFFSET]]],
//  CHECK-SAME: sizes: [2, 1, 8], strides: [40, 40, 5]
//  CHECK-SAME: to tensor<1x1x8xf32>
//       CHECK: %[[FLAT:.*]] = nv_tensor_ir.reshape %[[VALUE]] : tensor<1x1x8xf32> -> tensor<1x8xf32>
//   CHECK-NOT: nv_tensor_ir.reshape
//       CHECK: %[[NEXT:.*]] = nv_tensor_ir.add %[[CARRY]], %[[FLAT]] : tensor<1x8xf32>
//       CHECK: scf.yield %[[NEXT]] : tensor<1x8xf32>
//       CHECK: nv_tensor_ir.store %[[LOOP]],
nv_tensor_ir.graph @native_slice_last_axis(%initial: tensor<2x8xf32>, %values: tensor<2x1x8x5xf32>)
    -> tensor<2x8xf32> attributes {tile_size = array<i32: 1, 8>} {
  %final = scan_slice(%initial, %values) scan_dimensions = [3] output_dimensions = []
      : (tensor<2x8xf32>, tensor<2x1x8x5xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x1x8xf32>):
      %flat = reshape %value : tensor<2x1x8xf32> -> tensor<2x8xf32>
      %next = add %carry, %flat : tensor<2x8xf32>
      yield %next : tensor<2x8xf32>
  }
  results %final : tensor<2x8xf32>
}
