// RUN: tensor_ir-opt %s | FileCheck %s --check-prefixes=FLOW,ASM
// RUN: tensor_ir-opt %s | tensor_ir-opt | FileCheck %s --check-prefixes=FLOW,ASM
// RUN: tensor_ir-opt %s --split-input-file --test-control-flow-interface -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEAD
// RUN: tensor_ir-opt %s --mlir-print-op-generic | tensor_ir-opt | FileCheck %s --check-prefixes=FLOW,ASM

// Multiple inputs with distinct scan axes, multiple histories, and mixed dtypes.
//  FLOW-LABEL: graph @forward(
//   FLOW-SAME: %[[INITIAL:[^:]+]]: tensor<2x8xf32>
//   FLOW-SAME: %[[VALUES:[^:]+]]: tensor<5x2x8xf32>
//   FLOW-SAME: %[[OTHER:[^:]+]]: tensor<2x5x8xf32>
//        FLOW: %[[SCAN_SLICE:.*]]:3 = scan_slice(%[[INITIAL]], %[[VALUES]], %[[OTHER]]) scan_dimensions = [0, 1]
//   FLOW-SAME: output_dimensions = [0, 1]
//    ASM-SAME: {tag = "forward"} : (tensor<2x8xf32>, tensor<5x2x8xf32>, tensor<2x5x8xf32>) {
//        FLOW: ^bb0(%[[CARRY:.*]]: tensor<2x8xf32>, %[[VALUE:.*]]: tensor<2x8xf32>, %[[EXTRA:.*]]: tensor<2x8xf32>):
//        FLOW: %[[SUM:.*]] = add %[[CARRY]], %[[VALUE]] : tensor<2x8xf32>
//        FLOW: %[[NEXT:.*]] = add %[[SUM]], %[[EXTRA]] : tensor<2x8xf32>
//        FLOW: %[[CAST:.*]] = convert %[[NEXT]] : tensor<2x8xf32> -> tensor<2x8xbf16>
//        FLOW: yield %[[NEXT]], %[[NEXT]], %[[CAST]] : tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xbf16>
//        FLOW: results %[[SCAN_SLICE]]#0, %[[SCAN_SLICE]]#1, %[[SCAN_SLICE]]#2 : tensor<2x8xf32>, tensor<5x2x8xf32>, tensor<2x5x8xbf16>

nv_tensor_ir.graph @forward(%initial: tensor<2x8xf32>,
    %values: tensor<5x2x8xf32>, %other: tensor<2x5x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32>, tensor<2x5x8xbf16>) {
  // The body is entered from scan_slice and revisited from yield; only yield exits.
  // DEAD-LABEL: forward:
  //  DEAD-NEXT: region #0
  //  DEAD-NEXT: ^{{bb[0-9]+}} = live
  //  DEAD-NEXT: region_preds: (all) predecessors:
  //  DEAD-NEXT: %{{[^ ]+}} = nv_tensor_ir.scan_slice(
  //  DEAD-SAME: scan_dimensions = [0, 1] output_dimensions = [0, 1]
  //  DEAD-NEXT: nv_tensor_ir.yield %[[NEXT:[^ ,]+]], %[[NEXT]], %[[LOW:[^ ,]+]] : tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xbf16>
  //  DEAD-NEXT: op_preds: (all) predecessors:
  //  DEAD-NEXT: nv_tensor_ir.yield %[[NEXT]], %[[NEXT]], %[[LOW]] : tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xbf16>
  %final, %history, %low = nv_tensor_ir.scan_slice (%initial, %values, %other)
      scan_dimensions = [0, 1] output_dimensions = [0, 1] {tag = "forward"}
      : (tensor<2x8xf32>, tensor<5x2x8xf32>, tensor<2x5x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>, %extra: tensor<2x8xf32>):
      %sum = add %carry, %value : tensor<2x8xf32>
      %next = add %sum, %extra : tensor<2x8xf32>
      %cast = convert %next : tensor<2x8xf32> -> tensor<2x8xbf16>
      yield %next, %next, %cast : tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xbf16>
  }
  results %final, %history, %low : tensor<2x8xf32>, tensor<5x2x8xf32>, tensor<2x5x8xbf16>
}

// Singleton scans have no backedge, including reverse scans; no histories.
//  FLOW-LABEL: graph @singleton(
//   FLOW-SAME: %[[INITIAL:[^:]+]]: tensor<8xsi32>
//   FLOW-SAME: %[[VALUES:[^:]+]]: tensor<1x8xsi32>
//        FLOW: %[[SCAN_SLICE:.*]] = scan_slice(%[[INITIAL]], %[[VALUES]]) scan_dimensions = [0] output_dimensions = []
//    ASM-SAME: {reverse = true, tag = "singleton"} : (tensor<8xsi32>, tensor<1x8xsi32>) {
//        FLOW: ^bb0(%[[CARRY:.*]]: tensor<8xsi32>, %[[VALUE:.*]]: tensor<8xsi32>):
//        FLOW: %[[NEXT:.*]] = add %[[CARRY]], %[[VALUE]] : tensor<8xsi32>
//        FLOW: yield %[[NEXT]] : tensor<8xsi32>
//        FLOW: results %[[SCAN_SLICE]] : tensor<8xsi32>

nv_tensor_ir.graph @singleton(%initial: tensor<8xsi32>,
    %values: tensor<1x8xsi32>) -> tensor<8xsi32> {
  // The singleton body has only the entry predecessor, with no yield backedge.
  // DEAD-LABEL: singleton:
  //  DEAD-NEXT: region #0
  //  DEAD-NEXT: ^{{bb[0-9]+}} = live
  //  DEAD-NEXT: region_preds: (all) predecessors:
  //  DEAD-NEXT: %{{[^ ]+}} = nv_tensor_ir.scan_slice(
  //  DEAD-SAME: scan_dimensions = [0] output_dimensions = [] {reverse = true,
  //  DEAD-NEXT: op_preds: (all) predecessors:
  //  DEAD-NEXT: nv_tensor_ir.yield %{{[^ ,]+}} : tensor<8xsi32>
  //   DEAD-NOT: predecessors:
  %final = nv_tensor_ir.scan_slice (%initial, %values)
      scan_dimensions = [0] output_dimensions = [] {reverse = true, tag = "singleton"}
      : (tensor<8xsi32>, tensor<1x8xsi32>) {
    ^bb0(%carry: tensor<8xsi32>, %value: tensor<8xsi32>):
      %next = add %carry, %value : tensor<8xsi32>
      yield %next : tensor<8xsi32>
  }
  results %final : tensor<8xsi32>
}
