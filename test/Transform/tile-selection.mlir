// Test TileSelectionPass: resolves final tile_size from multiple sources.

//===----------------------------------------------------------------------===//
// Source 3: Pick tile_candidates[0] from TileAnalyzerPass
//===----------------------------------------------------------------------===//

// RUN: tensor_ir-opt -tile-analyzer -tile-selection -mlir-print-local-scope -split-input-file %s | FileCheck %s --check-prefix=FROM_CANDIDATES

// FROM_CANDIDATES-LABEL: nv_tensor_ir.graph @pick_from_candidates
// FROM_CANDIDATES-SAME: {tile_size = array<i32: 2048>}
// FROM_CANDIDATES-NOT: tile_candidates

nv_tensor_ir.graph @pick_from_candidates(%in: tensor<1048576xf32>) -> tensor<1048576xf32> {
    %out = abs %in : tensor<1048576xf32>
    results %out : tensor<1048576xf32>
}

// -----

// Iteration-space dimensions may exceed the signed i32 range even though tile
// dimensions remain small.
// RUN: tensor_ir-opt -tile-analyzer -tile-selection -mlir-print-local-scope -split-input-file %s | FileCheck %s --check-prefix=LARGE_ITER_SPACE

// LARGE_ITER_SPACE-LABEL: nv_tensor_ir.graph @large_iteration_space
// LARGE_ITER_SPACE-SAME: {tile_size = array<i32: {{[1-9][0-9]*}}>}

nv_tensor_ir.graph @large_iteration_space(%in: tensor<2818572288xf32>) -> tensor<2818572288xf32> {
    %out = abs %in : tensor<2818572288xf32>
    results %out : tensor<2818572288xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Source 1: Explicit tile via pass option (compilation options)
//===----------------------------------------------------------------------===//

// RUN: tensor_ir-opt "-tile-analyzer=tile-size=32" "-tile-selection=tile-size=32" -mlir-print-local-scope -split-input-file %s | FileCheck %s --check-prefix=FROM_OPTION

// FROM_OPTION-LABEL: nv_tensor_ir.graph @pick_from_option
// FROM_OPTION-SAME: {tile_size = array<i32: 32>}
// FROM_OPTION-NOT: tile_candidates

nv_tensor_ir.graph @pick_from_option(%in: tensor<1048576xf32>) -> tensor<1048576xf32> {
    %out = abs %in : tensor<1048576xf32>
    results %out : tensor<1048576xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Source 2: tile_size attr on GraphOp (from tensor-ir)
//===----------------------------------------------------------------------===//

// RUN: tensor_ir-opt -tile-analyzer -tile-selection -mlir-print-local-scope -split-input-file %s | FileCheck %s --check-prefix=FROM_IR

// FROM_IR-LABEL: nv_tensor_ir.graph @pick_from_ir
// FROM_IR-SAME: {tile_size = array<i32: 32>}

nv_tensor_ir.graph @pick_from_ir(%in: tensor<1048576xf32>) -> tensor<1048576xf32>
    attributes {tile_size = array<i32: 32>} {
    %out = abs %in : tensor<1048576xf32>
    results %out : tensor<1048576xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Precedence: pass option overrides tile_size attr on GraphOp
//===----------------------------------------------------------------------===//

// RUN: tensor_ir-opt -tile-analyzer "-tile-selection=tile-size=64" -mlir-print-local-scope -split-input-file %s | FileCheck %s --check-prefix=OPTION_OVER_IR

// OPTION_OVER_IR-LABEL: nv_tensor_ir.graph @option_over_ir
// OPTION_OVER_IR-SAME: {tile_size = array<i32: 64>}

nv_tensor_ir.graph @option_over_ir(%in: tensor<1048576xf32>) -> tensor<1048576xf32>
    attributes {tile_size = array<i32: 32>} {
    %out = abs %in : tensor<1048576xf32>
    results %out : tensor<1048576xf32>
}
