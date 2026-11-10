// RUN: tensor_ir-opt %s -tile-selection -split-input-file -verify-diagnostics

// expected-error @below {{Selected tile [128, 4] is not valid for iteration space [1048576] — rank mismatch (tile rank 2 vs iteration space rank 1)}}
nv_tensor_ir.graph @invalid_rank_mismatch(%in: tensor<1048576xf32>) -> tensor<1048576xf32>
    attributes {tile_size = array<i32: 128, 4>} {
    %out = abs %in : tensor<1048576xf32>
    results %out : tensor<1048576xf32>
}

// -----

// expected-error @below {{Selected tile [7] is not valid for iteration space [1048576]}}
nv_tensor_ir.graph @invalid_tile(%in: tensor<1048576xf32>) -> tensor<1048576xf32>
    attributes {tile_size = array<i32: 7>} {
    %out = abs %in : tensor<1048576xf32>
    results %out : tensor<1048576xf32>
}

// -----

// No tile_size attr, no pass option, and no tile_candidates.
// expected-error @below {{No tile shape source available}}
nv_tensor_ir.graph @missing_tile_source(%in: tensor<1048576xf32>) -> tensor<1048576xf32> {
    %out = abs %in : tensor<1048576xf32>
    results %out : tensor<1048576xf32>
}
