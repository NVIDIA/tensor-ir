// Test tile validation via TileSelectionPass (validation moved from TileAnalyzerPass).

// Valid tile (power-of-2, within bounds for 1D iteration space):
// RUN: tensor_ir-opt -layout-propagation-annotation -layout-propagation-normalization "-tile-analyzer=tile-size=32" "-tile-selection=tile-size=32" -mlir-print-local-scope %s | FileCheck %s --check-prefix=VALID
// VALID-LABEL: nv_tensor_ir.graph @pw_validate_target
// VALID-SAME: {tile_size = array<i32: 32>}
// VALID-NOT: tile_candidates

// Invalid tile (not power-of-2):
// RUN: not tensor_ir-opt -layout-propagation-annotation -layout-propagation-normalization "-tile-analyzer=tile-size=7" "-tile-selection=tile-size=7" %s 2>&1 | FileCheck %s --check-prefix=INVALID
// INVALID: is not valid for iteration space

nv_tensor_ir.graph @pw_validate_target(%in: tensor<1048576xf32>) -> tensor<1048576xf32> {
    %out = abs %in : tensor<1048576xf32>
    results %out : tensor<1048576xf32>
}
