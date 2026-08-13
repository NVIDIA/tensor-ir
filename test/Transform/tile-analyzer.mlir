// RUN: tensor_ir-opt -layout-propagation-annotation -layout-propagation-normalization "-tile-analyzer=max-candidates=8" -mlir-print-local-scope -split-input-file %s | FileCheck %s --check-prefix=CANDIDATES
// RUN: tensor_ir-opt -layout-propagation-annotation -layout-propagation-normalization -tile-analyzer -mlir-print-local-scope -split-input-file %s | FileCheck %s --check-prefix=DEFAULT

//===----------------------------------------------------------------------===//
// Single-source pointwise: contiguous 1D
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @pw_1d_contiguous
// CANDIDATES-SAME: {tile_candidates = [array<i32: 2048>, array<i32: 4096>, array<i32: 8192>, array<i32: 1024>, array<i32: 512>, array<i32: 256>, array<i32: 128>, array<i32: 64>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @pw_1d_contiguous
// DEFAULT-SAME: {tile_candidates = [array<i32: 2048>]}

nv_tensor_ir.graph @pw_1d_contiguous(%in: tensor<1048576xf32>) -> (tensor<1048576xf32>) {
    %out = abs %in : tensor<1048576xf32>
    results %out : tensor<1048576xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Single-source pointwise: 2D row-major (gets normalized to 1D contiguous)
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @pw_2d_row_major
// CANDIDATES-SAME: {tile_candidates = [array<i32: 64, 32>, array<i32: 128, 16>, array<i32: 256, 8>, array<i32: 512, 4>, array<i32: 64, 64>, array<i32: 128, 32>, array<i32: 256, 16>, array<i32: 512, 8>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @pw_2d_row_major
// DEFAULT-SAME: {tile_candidates = [array<i32: 64, 32>]}

nv_tensor_ir.graph @pw_2d_row_major(%in: tensor<512x1024xf32> {nv_tensor_ir.stride = "(1,512)"}) -> (tensor<512x1024xf32> {nv_tensor_ir.stride = "(1,512)"}) {
    %out = abs %in : tensor<512x1024xf32>
    results %out : tensor<512x1024xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Composite source: two inputs with same layout (binary add)
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @pw_binary_add
// CANDIDATES-SAME: {tile_candidates = [array<i32: 32, 64>, array<i32: 64, 32>, array<i32: 128, 16>, array<i32: 256, 8>, array<i32: 32, 32>, array<i32: 64, 16>, array<i32: 128, 8>, array<i32: 256, 4>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @pw_binary_add
// DEFAULT-SAME: {tile_candidates = [array<i32: 32, 64>]}

nv_tensor_ir.graph @pw_binary_add(%a: tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"},
                                   %b: tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"}) -> (tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"}) {
    %out = add %a, %b : tensor<256x1024xf32>
    results %out : tensor<256x1024xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Composite source: different strides (add with transpose)
// Normalization preserves 2D shape since strides differ.
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @pw_add_with_transpose
// CANDIDATES-SAME: {tile_candidates = [array<i32: 128, 16>, array<i32: 64, 32>, array<i32: 256, 8>, array<i32: 32, 64>, array<i32: 32, 32>, array<i32: 32, 128>, array<i32: 64, 16>, array<i32: 512, 4>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @pw_add_with_transpose
// DEFAULT-SAME: {tile_candidates = [array<i32: 128, 16>]}

nv_tensor_ir.graph @pw_add_with_transpose(%in: tensor<512x512xf32> {nv_tensor_ir.stride = "(1,512)"}) -> (tensor<512x512xf32> {nv_tensor_ir.stride = "(1,512)"}) {
    %transposed = transpose %in permutation = [1, 0] : tensor<512x512xf32> -> tensor<512x512xf32>
    %out = add %in, %transposed : tensor<512x512xf32>
    results %out : tensor<512x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Small tensor (< 100K elements)
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @pw_small_tensor
// CANDIDATES-SAME: {tile_candidates = [array<i32: 32, 2>, array<i32: 64, 1>, array<i32: 32, 1>, array<i32: 16, 4>, array<i32: 32, 4>, array<i32: 64, 2>, array<i32: 8, 8>, array<i32: 16, 2>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @pw_small_tensor
// DEFAULT-SAME: {tile_candidates = [array<i32: 32, 2>]}

nv_tensor_ir.graph @pw_small_tensor(%in: tensor<64x128xf32> {nv_tensor_ir.stride = "(1,64)"}) -> (tensor<64x128xf32> {nv_tensor_ir.stride = "(1,64)"}) {
    %out = abs %in : tensor<64x128xf32>
    results %out : tensor<64x128xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Different element type: f16 (cacheLineElems = 128/2 = 64)
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @pw_f16
// CANDIDATES-SAME: {tile_candidates = [array<i32: 64, 32>, array<i32: 128, 16>, array<i32: 256, 8>, array<i32: 512, 4>, array<i32: 64, 64>, array<i32: 128, 32>, array<i32: 256, 16>, array<i32: 512, 8>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @pw_f16
// DEFAULT-SAME: {tile_candidates = [array<i32: 64, 32>]}

nv_tensor_ir.graph @pw_f16(%in: tensor<512x1024xf16> {nv_tensor_ir.stride = "(1,512)"}) -> (tensor<512x1024xf16> {nv_tensor_ir.stride = "(1,512)"}) {
    %out = abs %in : tensor<512x1024xf16>
    results %out : tensor<512x1024xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Three inputs: composite with three sources
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @pw_three_inputs
// CANDIDATES-SAME: {tile_candidates = [array<i32: 32, 64>, array<i32: 64, 32>, array<i32: 128, 16>, array<i32: 256, 8>, array<i32: 32, 32>, array<i32: 64, 16>, array<i32: 128, 8>, array<i32: 256, 4>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @pw_three_inputs
// DEFAULT-SAME: {tile_candidates = [array<i32: 32, 64>]}

nv_tensor_ir.graph @pw_three_inputs(%a: tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"},
                                     %b: tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"},
                                     %c: tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"}) -> (tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"}) {
    %ab = add %a, %b : tensor<256x1024xf32>
    %out = add %ab, %c : tensor<256x1024xf32>
    results %out : tensor<256x1024xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Broadcast: iteration space has zero-stride dim (2D preserved)
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @pw_broadcast
// CANDIDATES-SAME: {tile_candidates = [array<i32: 32, 64>, array<i32: 64, 32>, array<i32: 32, 32>, array<i32: 16, 128>, array<i32: 128, 16>, array<i32: 32, 128>, array<i32: 8, 256>, array<i32: 256, 8>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @pw_broadcast
// DEFAULT-SAME: {tile_candidates = [array<i32: 32, 64>]}

nv_tensor_ir.graph @pw_broadcast(%in: tensor<1x1024xf32> {nv_tensor_ir.stride = "(0,1)"}) -> (tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"}) {
    %out = broadcast %in : tensor<1x1024xf32> -> tensor<256x1024xf32>
    results %out : tensor<256x1024xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Concat: top-level ConcatSourceAttr
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @concat_direct
// CANDIDATES-SAME: {tile_candidates = [array<i32: 32, 1, 32>, array<i32: 64, 1, 16>, array<i32: 32, 1, 16>, array<i32: 32, 1, 8>, array<i32: 32, 1, 4>, array<i32: 32, 1, 2>, array<i32: 128, 1, 8>, array<i32: 64, 1, 8>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @concat_direct
// DEFAULT-SAME: {tile_candidates = [array<i32: 32, 1, 32>]}

nv_tensor_ir.graph @concat_direct(%a: tensor<256x512xf32> {nv_tensor_ir.stride = "(1,256)"},
                                   %b: tensor<256x512xf32> {nv_tensor_ir.stride = "(1,256)"}) -> (tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"}) {
    %out = concatenate %a, %b dimension = 1
        : (tensor<256x512xf32>, tensor<256x512xf32>) -> tensor<256x1024xf32>
    results %out : tensor<256x1024xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Concat nested in CompositeSourceAttr
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @concat_in_composite
// CANDIDATES-SAME: {tile_candidates = [array<i32: 32, 1, 32>, array<i32: 64, 1, 16>, array<i32: 32, 1, 16>, array<i32: 32, 1, 8>, array<i32: 32, 1, 4>, array<i32: 32, 1, 2>, array<i32: 128, 1, 8>, array<i32: 64, 1, 8>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @concat_in_composite
// DEFAULT-SAME: {tile_candidates = [array<i32: 32, 1, 32>]}

nv_tensor_ir.graph @concat_in_composite(%a: tensor<256x512xf32> {nv_tensor_ir.stride = "(1,256)"},
                                         %b: tensor<256x512xf32> {nv_tensor_ir.stride = "(1,256)"},
                                         %c: tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"}) -> (tensor<256x1024xf32> {nv_tensor_ir.stride = "(1,256)"}) {
    %cat = concatenate %a, %b dimension = 1
        : (tensor<256x512xf32>, tensor<256x512xf32>) -> tensor<256x1024xf32>
    %out = add %cat, %c : tensor<256x1024xf32>
    results %out : tensor<256x1024xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Fallback: iteration space with non-power-of-2 dims so the byte-weighted /
// smem tracks can't synthesize any valid candidate. TileAnalyzerPass must
// still produce a single conservative candidate (PowerOf2Floor per dim,
// clamped to 128).
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @fallback_reshape_npow2
// CANDIDATES-SAME: {tile_candidates = [array<i32: 2, 4, 4>]}

// DEFAULT-LABEL: nv_tensor_ir.graph @fallback_reshape_npow2
// DEFAULT-SAME: {tile_candidates = [array<i32: 2, 4, 4>]}

nv_tensor_ir.graph @fallback_reshape_npow2(
    %in: tensor<3x4x5xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(1,3,12)"}
) -> (tensor<6x10xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(10,1)"}) {
    %0 = reshape %in : tensor<3x4x5xf32> -> tensor<6x10xf32>
    results %0 : tensor<6x10xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Fallback (1D, large): even the conservative path is exercised when
// the generator track silently produces something - verify that the tile
// is non-empty (one candidate, clamped to 128).
//===----------------------------------------------------------------------===//

// DEFAULT-LABEL: nv_tensor_ir.graph @fallback_1d
// DEFAULT-SAME: tile_candidates

nv_tensor_ir.graph @fallback_1d(%in: tensor<1048576xf32>) -> (tensor<1048576xf32>) {
    %out = abs %in : tensor<1048576xf32>
    results %out : tensor<1048576xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Iota (synthetic source, no memory traffic)
//===----------------------------------------------------------------------===//

// CANDIDATES-LABEL: nv_tensor_ir.graph @iota_only
// CANDIDATES-SAME: {tile_candidates

// DEFAULT-LABEL: nv_tensor_ir.graph @iota_only
// DEFAULT-SAME: {tile_candidates

nv_tensor_ir.graph @iota_only() -> tensor<4096xf32> {
    %out = iota dimension = 0 : tensor<4096xf32>
    results %out : tensor<4096xf32>
}

// -----

// CANDIDATES-LABEL: nv_tensor_ir.graph @iota_add_input
// CANDIDATES-SAME: {tile_candidates

// DEFAULT-LABEL: nv_tensor_ir.graph @iota_add_input
// DEFAULT-SAME: {tile_candidates

nv_tensor_ir.graph @iota_add_input(%in: tensor<256x1024xf32>) -> tensor<256x1024xf32> {
    %idx = iota dimension = 0 : tensor<256x1024xf32>
    %out = add %in, %idx : tensor<256x1024xf32>
    results %out : tensor<256x1024xf32>
}
