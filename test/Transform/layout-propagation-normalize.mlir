// RUN: tensor_ir-opt -layout-propagation-annotation -layout-propagation-normalization -mlir-print-local-scope -split-input-file %s | FileCheck %s

// Tests for the normalize() utility function.

//===----------------------------------------------------------------------===//
// TensorSourceAttr normalization
//===----------------------------------------------------------------------===//

// Unit dimension with zero stride: (1) -> (1):(0)
// CHECK-LABEL: nv_tensor_ir.graph @normalize_unit_dimension
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(1):(0)">}
nv_tensor_ir.graph @normalize_unit_dimension(%in: tensor<1xf32> {nv_tensor_ir.stride = "(0)"}) -> (tensor<1xf32> {nv_tensor_ir.stride = "(0)"}) {
    %out = abs %in : tensor<1xf32>
    results %out : tensor<1xf32>
}

// -----

// Join contiguous dimensions: (2,4) -> (8):(1)
// CHECK-LABEL: nv_tensor_ir.graph @normalize_join_dims
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(2,4):(1,2)">}
nv_tensor_ir.graph @normalize_join_dims(%in: tensor<2x4xf32> {nv_tensor_ir.stride = "(1,2)"}) -> (tensor<2x4xf32> {nv_tensor_ir.stride = "(1,2)"}) {
    %out = abs %in : tensor<2x4xf32>
    results %out : tensor<2x4xf32>
}

// -----

// 3D contiguous: (2,4,8) -> (64):(1)
// CHECK-LABEL: nv_tensor_ir.graph @normalize_3d_contiguous
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(2,4,8):(1,2,8)">}
nv_tensor_ir.graph @normalize_3d_contiguous(%in: tensor<2x4x8xf32> {nv_tensor_ir.stride = "(1,2,8)"}) -> (tensor<2x4x8xf32> {nv_tensor_ir.stride = "(1,2,8)"}) {
    %out = abs %in : tensor<2x4x8xf32>
    results %out : tensor<2x4x8xf32>
}

// -----

// Transpose result normalized: (8,4) transposed from (4,8) -> normalized to contiguous
// The input (4,8):(1,4) becomes (8,4):(4,1) after transpose, which is contiguous
// CHECK-LABEL: nv_tensor_ir.graph @normalize_after_transpose
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(8,4):(4,1)">}
nv_tensor_ir.graph @normalize_after_transpose(%in: tensor<4x8xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<8x4xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %out = transpose %in permutation = [1, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
    results %out : tensor<8x4xf32>
}

// -----

// Broadcast dimensions preserved with zero stride
// CHECK-LABEL: nv_tensor_ir.graph @normalize_after_broadcast
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(8,16):(0,1)">}
nv_tensor_ir.graph @normalize_after_broadcast(%in: tensor<1x16xf32> {nv_tensor_ir.stride = "(0,1)"}) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %out = broadcast %in : tensor<1x16xf32> -> tensor<8x16xf32>
    results %out : tensor<8x16xf32>
}

// -----

// Slice with stride: non-unit stride preserved
// CHECK-LABEL: nv_tensor_ir.graph @normalize_after_slice_stride
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(8):(2)">}
nv_tensor_ir.graph @normalize_after_slice_stride(%in: tensor<16xf32>) -> (tensor<8xf32>) {
    %out = slice %in starts = [0] limits = [16] strides = [2] : tensor<16xf32> -> tensor<8xf32>
    results %out : tensor<8xf32>
}

// -----

// Offset preserved through normalization
// CHECK-LABEL: nv_tensor_ir.graph @normalize_preserves_offset
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<0, 4, "(8):(1)">}
nv_tensor_ir.graph @normalize_preserves_offset(%in: tensor<16xf32>) -> (tensor<8xf32>) {
    %out = slice %in starts = [4] limits = [12] strides = [1] : tensor<16xf32> -> tensor<8xf32>
    results %out : tensor<8xf32>
}

// -----

//===----------------------------------------------------------------------===//
// CompositeSourceAttr normalization
//===----------------------------------------------------------------------===//

// Composite with same contiguous layouts: both normalized to same shape
// CHECK-LABEL: nv_tensor_ir.graph @normalize_composite_same_layout
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">, #nv_tensor_ir.tensor_source<1, 0, "(4,16):(1,4)">>}
nv_tensor_ir.graph @normalize_composite_same_layout(%a: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"},
                                                     %b: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %out = add %a, %b : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

// Composite with different strides (from transpose): coalesced where possible
// CHECK-LABEL: nv_tensor_ir.graph @normalize_composite_different_strides
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,8):(1,8)">, #nv_tensor_ir.tensor_source<0, 0, "(8,8):(8,1)">>}
nv_tensor_ir.graph @normalize_composite_different_strides(%in: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %transposed = transpose %in permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
    %out = add %in, %transposed : tensor<8x8xf32>
    results %out : tensor<8x8xf32>
}

// -----

// Composite through reshape: both sources reshaped and normalized
// CHECK-LABEL: nv_tensor_ir.graph @normalize_composite_through_reshape
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(64):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(64):(1)">>}
nv_tensor_ir.graph @normalize_composite_through_reshape(%a: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"},
                                                         %b: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<64xf32>) {
    %sum = add %a, %b : tensor<4x16xf32>
    %transposed = transpose %sum permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    %out = reshape %transposed : tensor<16x4xf32> -> tensor<64xf32>
    results %out : tensor<64xf32>
}

// -----

// Composite with broadcast: unit dimension gets coalesced in add, broadcast preserves shape
// CHECK-LABEL: nv_tensor_ir.graph @normalize_composite_with_broadcast
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,16):(0,1)">, #nv_tensor_ir.tensor_source<1, 0, "(8,16):(0,1)">>}
nv_tensor_ir.graph @normalize_composite_with_broadcast(%a: tensor<1x16xf32> {nv_tensor_ir.stride = "(0,1)"},
                                                        %b: tensor<1x16xf32> {nv_tensor_ir.stride = "(0,1)"}) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %sum = add %a, %b : tensor<1x16xf32>
    %out = broadcast %sum : tensor<1x16xf32> -> tensor<8x16xf32>
    results %out : tensor<8x16xf32>
}

// -----

// Three inputs: all normalized together
// CHECK-LABEL: nv_tensor_ir.graph @normalize_three_inputs
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">, #nv_tensor_ir.tensor_source<1, 0, "(4,16):(1,4)">, #nv_tensor_ir.tensor_source<2, 0, "(4,16):(1,4)">>}
nv_tensor_ir.graph @normalize_three_inputs(%a: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"},
                                            %b: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"},
                                            %c: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %ab = add %a, %b : tensor<4x16xf32>
    %out = add %ab, %c : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

// Splat with regular input: splat stride (0) preserved, regular normalized
// CHECK-LABEL: nv_tensor_ir.graph @normalize_splat_and_input
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">, #nv_tensor_ir.tensor_source<-1, 0, "(4,16):(0,0)">>}
nv_tensor_ir.graph @normalize_splat_and_input(%in: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}, %val: f32) -> (tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %splatted = splat %val : tensor<4x16xf32>
    %out = add %in, %splatted : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

// Matmul with scalar splat: matmul source and zero-stride scalar source are both preserved.
// CHECK-LABEL: nv_tensor_ir.graph @normalize_matmul_scalar_add
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.composite_source<#nv_tensor_ir.matmul_source<"(64,32,64,64):(131072,4096,64,1)", 64, 32, 64, 64, #nv_tensor_ir.tensor_source<0, 0, "(64,32,64):(2048,64,1)">, #nv_tensor_ir.tensor_source<1, 0, "(64,64,64):(4096,64,1)">>, #nv_tensor_ir.tensor_source<-1, 0, "(64,32,64):(0,0,0)">>}
nv_tensor_ir.graph @normalize_matmul_scalar_add(
    %a: tensor<64x32x64xbf16> {nv_tensor_ir.stride = "(2048,64,1)"},
    %b: tensor<64x64x64xbf16> {nv_tensor_ir.stride = "(4096,64,1)"},
    %bias: bf16
) -> (tensor<64x32x64xbf16> {nv_tensor_ir.stride = "(2048,64,1)"}) {
    %matmul = matmul(%a, %b) : (tensor<64x32x64xbf16>, tensor<64x64x64xbf16>) -> tensor<64x32x64xf32>
    %splatted = splat %bias : tensor<64x32x64xbf16>
    %bias_f32 = convert %splatted : tensor<64x32x64xbf16> -> tensor<64x32x64xf32>
    %out_f32 = add %matmul, %bias_f32 : tensor<64x32x64xf32>
    %out = convert %out_f32 : tensor<64x32x64xf32> -> tensor<64x32x64xbf16>
    results %out : tensor<64x32x64xbf16>
}

// -----

// Iota with regular input: iota dim-0 stride (1,0) is preserved; input stays row-major.
// Unlike splat, iota's broadcast stride-0 dim prevents coalescing to 1D, so the
// common iteration space stays (4,16) rather than flat (64).
// CHECK-LABEL: nv_tensor_ir.graph @normalize_iota_and_input
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(4,16):(16,1)">, #nv_tensor_ir.tensor_source<-1, 0, "(4,16):(1,0)">>}
nv_tensor_ir.graph @normalize_iota_and_input(%in: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %idx = iota dimension = 0 : tensor<4x16xf32>
    %out = add %in, %idx : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

// Iota-only graph: synthetic source normalized to 1D iteration space
// CHECK-LABEL: nv_tensor_ir.graph @normalize_iota_only
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<-1, 0, "(128):(1)">}
nv_tensor_ir.graph @normalize_iota_only() -> tensor<128xf32> {
    %out = iota dimension = 0 : tensor<128xf32>
    results %out : tensor<128xf32>
}

// -----

//===----------------------------------------------------------------------===//
// ConcatSourceAttr normalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @normalize_concat_prime_left_dim
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(3,4,16):(1,3,12)">, #nv_tensor_ir.tensor_source<1, 0, "(5,4,16):(1,5,20)">>}
nv_tensor_ir.graph @normalize_concat_prime_left_dim(
        %in0: tensor<3x4x16xf32> {nv_tensor_ir.stride = "(1,3,12)"},
        %in1: tensor<5x4x16xf32> {nv_tensor_ir.stride = "(1,5,20)"}) -> (tensor<8x4x16xf32> {nv_tensor_ir.stride = "(1,8,32)"}) {
    %out = concatenate %in0, %in1 dimension = 0
        : (tensor<3x4x16xf32>, tensor<5x4x16xf32>) -> tensor<8x4x16xf32>
    results %out : tensor<8x4x16xf32>
}

// -----

// CHECK-LABEL: @normalize_concat_prime_middle_dim
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.concat_source<dim = 2, #nv_tensor_ir.tensor_source<0, 0, "(2,4,3,16):(1,2,8,24)">, #nv_tensor_ir.tensor_source<1, 0, "(2,4,5,16):(1,2,8,40)">>}
nv_tensor_ir.graph @normalize_concat_prime_middle_dim(
        %in0: tensor<2x4x3x16xf32> {nv_tensor_ir.stride = "(1,2,8,24)"},
        %in1: tensor<2x4x5x16xf32> {nv_tensor_ir.stride = "(1,2,8,40)"}) -> (tensor<2x4x8x16xf32> {nv_tensor_ir.stride = "(1,2,8,64)"}) {
    %out = concatenate %in0, %in1 dimension = 2
        : (tensor<2x4x3x16xf32>, tensor<2x4x5x16xf32>) -> tensor<2x4x8x16xf32>
    results %out : tensor<2x4x8x16xf32>
}

// -----

// CHECK-LABEL: @normalize_concat_prime_right_dim
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.concat_source<dim = 2, #nv_tensor_ir.tensor_source<0, 0, "(4,16,3):(1,4,64)">, #nv_tensor_ir.tensor_source<1, 0, "(4,16,5):(1,4,64)">>}
nv_tensor_ir.graph @normalize_concat_prime_right_dim(
        %in0: tensor<4x16x3xf32> {nv_tensor_ir.stride = "(1,4,64)"},
        %in1: tensor<4x16x5xf32> {nv_tensor_ir.stride = "(1,4,64)"}) -> (tensor<4x16x8xf32> {nv_tensor_ir.stride = "(1,4,64)"}) {
    %out = concatenate %in0, %in1 dimension = 2
        : (tensor<4x16x3xf32>, tensor<4x16x5xf32>) -> tensor<4x16x8xf32>
    results %out : tensor<4x16x8xf32>
}

// -----

// CHECK-LABEL: @normalize_concat_prime_not_coalesced
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.concat_source<dim = 3, #nv_tensor_ir.tensor_source<0, 0, "(4,2,4,5):(1,80,20,4)">, #nv_tensor_ir.tensor_source<1, 0, "(4,2,4,7):(2,1,56,8)">>}
nv_tensor_ir.graph @normalize_concat_prime_not_coalesced(
        %in0: tensor<4x5x8xf32> {nv_tensor_ir.stride = "(1,4,20)"},
        %in1: tensor<8x7x4xf32> {nv_tensor_ir.stride = "(1,8,56)"}) -> (tensor<32x12xf32> {nv_tensor_ir.stride = "(1,32)"}) {
    %transpose0 = transpose %in0 permutation = [1, 0, 2] : tensor<4x5x8xf32> -> tensor<5x4x8xf32>
    %reshaped0 = reshape %transpose0 : tensor<5x4x8xf32> -> tensor<5x32xf32>
    %reshape0 = transpose %reshaped0 permutation = [1, 0] : tensor<5x32xf32> -> tensor<32x5xf32>
    %transpose1 = transpose %in1 permutation = [1, 0, 2] : tensor<8x7x4xf32> -> tensor<7x8x4xf32>
    %reshaped1 = reshape %transpose1 : tensor<7x8x4xf32> -> tensor<7x32xf32>
    %reshape1 = transpose %reshaped1 permutation = [1, 0] : tensor<7x32xf32> -> tensor<32x7xf32>
    %out = concatenate %reshape0, %reshape1 dimension = 1
        : (tensor<32x5xf32>, tensor<32x7xf32>) -> tensor<32x12xf32>
    results %out : tensor<32x12xf32>
}

// -----

// CHECK-LABEL: @normalize_concat_divisible_left_dim
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(1,4,4,16):(0,1,4,16)">, #nv_tensor_ir.tensor_source<1, 0, "(3,4,4,16):(4,1,12,48)">>}
nv_tensor_ir.graph @normalize_concat_divisible_left_dim(
        %in0: tensor<4x4x16xf32> {nv_tensor_ir.stride = "(1,4,16)"},
        %in1: tensor<12x4x16xf32> {nv_tensor_ir.stride = "(1,12,48)"}) -> (tensor<16x4x16xf32> {nv_tensor_ir.stride = "(1,16,64)"}) {
    %out = concatenate %in0, %in1 dimension = 0
        : (tensor<4x4x16xf32>, tensor<12x4x16xf32>) -> tensor<16x4x16xf32>
    results %out : tensor<16x4x16xf32>
}

// -----

// CHECK-LABEL: @normalize_concat_divisible_middle_dim
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(8,1,4,16):(1,0,8,32)">, #nv_tensor_ir.tensor_source<1, 0, "(8,3,4,16):(1,32,8,96)">>}
nv_tensor_ir.graph @normalize_concat_divisible_middle_dim(
        %in0: tensor<8x4x16xf32> {nv_tensor_ir.stride = "(1,8,32)"},
        %in1: tensor<8x12x16xf32> {nv_tensor_ir.stride = "(1,8,96)"}) -> (tensor<8x16x16xf32> {nv_tensor_ir.stride = "(1,8,128)"}) {
    %out = concatenate %in0, %in1 dimension = 1
        : (tensor<8x4x16xf32>, tensor<8x12x16xf32>) -> tensor<8x16x16xf32>
    results %out : tensor<8x16x16xf32>
}

// -----

// CHECK-LABEL: @normalize_concat_divisible_right_dim
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.concat_source<dim = 2, #nv_tensor_ir.tensor_source<0, 0, "(4,16,1,4):(1,4,0,64)">, #nv_tensor_ir.tensor_source<1, 0, "(4,16,3,4):(1,4,256,64)">>}
nv_tensor_ir.graph @normalize_concat_divisible_right_dim(
        %in0: tensor<4x16x4xf32> {nv_tensor_ir.stride = "(1,4,64)"},
        %in1: tensor<4x16x12xf32> {nv_tensor_ir.stride = "(1,4,64)"}) -> (tensor<4x16x16xf32> {nv_tensor_ir.stride = "(1,4,64)"}) {
    %out = concatenate %in0, %in1 dimension = 2
        : (tensor<4x16x4xf32>, tensor<4x16x12xf32>) -> tensor<4x16x16xf32>
    results %out : tensor<4x16x16xf32>
}

// -----

// CHECK-LABEL: @normalize_concat_divisible_fragmented_dim
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(8,1,2,2):(1,0,64,8)">, #nv_tensor_ir.tensor_source<1, 0, "(8,3,2,2):(1,64,32,8)">>}
nv_tensor_ir.graph @normalize_concat_divisible_fragmented_dim(
        %in0: tensor<16x8xf32> {nv_tensor_ir.stride = "(1,16)"},
        %in1: tensor<16x12xf32> {nv_tensor_ir.stride = "(1,16)"}) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %slice0 = slice %in0 starts = [0, 0] limits = [16, 8] strides = [1, 4] : tensor<16x8xf32> -> tensor<16x2xf32>
    %transposed0 = transpose %slice0 permutation = [1, 0] : tensor<16x2xf32> -> tensor<2x16xf32>
    %reshaped0 = reshape %transposed0 : tensor<2x16xf32> -> tensor<4x8xf32>
    %reshape0 = transpose %reshaped0 permutation = [1, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
    %slice1 = slice %in1 starts = [0, 0] limits = [16, 12] strides = [1, 2] : tensor<16x12xf32> -> tensor<16x6xf32>
    %transposed1 = transpose %slice1 permutation = [1, 0] : tensor<16x6xf32> -> tensor<6x16xf32>
    %reshaped1 = reshape %transposed1 : tensor<6x16xf32> -> tensor<12x8xf32>
    %reshape1 = transpose %reshaped1 permutation = [1, 0] : tensor<12x8xf32> -> tensor<8x12xf32>
    %out = concatenate %reshape0, %reshape1 dimension = 1
        : (tensor<8x4xf32>, tensor<8x12xf32>) -> tensor<8x16xf32>
    results %out : tensor<8x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// ReductionSourceAttr normalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @normalize_reduce_single_dim_first
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(8,16,(4)):(64,4,(1))", #nv_tensor_ir.tensor_source<0, 0, "(8,16,4):(4,32,1)">>}
nv_tensor_ir.graph @normalize_reduce_single_dim_first(
        %in: tensor<4x8x16xf32> {nv_tensor_ir.stride = "(1,4,32)"}) -> (tensor<1x8x16xf32> {nv_tensor_ir.stride = "(0,1,8)"}) {
    %out = reduce(%in)<dimensions = [0], reduction_mode = <add>> : tensor<4x8x16xf32> -> tensor<1x8x16xf32>
    results %out : tensor<1x8x16xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_single_dim_middle
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(4,16,(8)):(128,8,(1))", #nv_tensor_ir.tensor_source<0, 0, "(4,16,8):(1,32,4)">>}
nv_tensor_ir.graph @normalize_reduce_single_dim_middle(
        %in: tensor<4x8x16xf32> {nv_tensor_ir.stride = "(1,4,32)"}) -> (tensor<4x1x16xf32> {nv_tensor_ir.stride = "(1,0,4)"}) {
    %out = reduce(%in)<dimensions = [1], reduction_mode = <add>> : tensor<4x8x16xf32> -> tensor<4x1x16xf32>
    results %out : tensor<4x1x16xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_single_dim_last
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(4,8,(16)):(128,16,(1))", #nv_tensor_ir.tensor_source<0, 0, "(4,8,16):(1,4,32)">>}
nv_tensor_ir.graph @normalize_reduce_single_dim_last(
        %in: tensor<4x8x16xf32> {nv_tensor_ir.stride = "(1,4,32)"}) -> (tensor<4x8x1xf32> {nv_tensor_ir.stride = "(1,4,0)"}) {
    %out = reduce(%in)<dimensions = [2], reduction_mode = <add>> : tensor<4x8x16xf32> -> tensor<4x8x1xf32>
    results %out : tensor<4x8x1xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_multiple_dims_adjacent
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(16,32,64,(4,8)):(65536,2048,32,(8,1))", #nv_tensor_ir.tensor_source<0, 0, "(16,32,64,4,8):(32,512,16384,1,4)">>}
nv_tensor_ir.graph @normalize_reduce_multiple_dims_adjacent(
        %in: tensor<4x8x16x32x64xf32> {nv_tensor_ir.stride = "(1,4,32,512,16384)"}) -> (tensor<1x1x16x32x64xf32> {nv_tensor_ir.stride = "(0,0,1,16,512)"}) {
    %out = reduce(%in)<dimensions = [0, 1], reduction_mode = <add>> : tensor<4x8x16x32x64xf32> -> tensor<1x1x16x32x64xf32>
    results %out : tensor<1x1x16x32x64xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_multiple_dims_two
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(4,16,64,(8,32)):(262144,16384,256,(32,1))", #nv_tensor_ir.tensor_source<0, 0, "(4,16,64,8,32):(1,32,16384,4,512)">>}
nv_tensor_ir.graph @normalize_reduce_multiple_dims_two(
        %in: tensor<4x8x16x32x64xf32> {nv_tensor_ir.stride = "(1,4,32,512,16384)"}) -> (tensor<4x1x16x1x64xf32> {nv_tensor_ir.stride = "(1,0,4,0,64)"}) {
    %out = reduce(%in)<dimensions = [1, 3], reduction_mode = <add>> : tensor<4x8x16x32x64xf32> -> tensor<4x1x16x1x64xf32>
    results %out : tensor<4x1x16x1x64xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_multiple_dims_three
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(8,32,(4,16,64)):(131072,4096,(1024,64,1))", #nv_tensor_ir.tensor_source<0, 0, "(8,32,4,16,64):(4,512,1,32,16384)">>}
nv_tensor_ir.graph @normalize_reduce_multiple_dims_three(
        %in: tensor<4x8x16x32x64xf32> {nv_tensor_ir.stride = "(1,4,32,512,16384)"}) -> (tensor<1x8x1x32x1xf32> {nv_tensor_ir.stride = "(0,1,0,8,0)"}) {
    %out = reduce(%in)<dimensions = [0, 2, 4], reduction_mode = <add>> : tensor<4x8x16x32x64xf32> -> tensor<1x8x1x32x1xf32>
    results %out : tensor<1x8x1x32x1xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_broadcast_dim_first
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(64,32,8,2,(4,16)):(0,1024,128,64,(16,1))", #nv_tensor_ir.tensor_source<0, 0, "(32,8,2,4,16):(1024,8,1,2,64)">>}
nv_tensor_ir.graph @normalize_reduce_broadcast_dim_first(
        %in: tensor<2x4x8x16x32xf32> {nv_tensor_ir.stride = "(1,2,8,64,1024)"}) -> (tensor<64x512xf32> {nv_tensor_ir.stride = "(1,64)"}) {
    %reduce = reduce(%in)<dimensions = [1, 3], reduction_mode = <add>> : tensor<2x4x8x16x32xf32> -> tensor<2x1x8x1x32xf32>
    %transposed = transpose %reduce permutation = [4, 3, 2, 1, 0] : tensor<2x1x8x1x32xf32> -> tensor<32x1x8x1x2xf32>
    %reshaped = reshape %transposed : tensor<32x1x8x1x2xf32> -> tensor<512x1xf32>
    %reshape = transpose %reshaped permutation = [1, 0] : tensor<512x1xf32> -> tensor<1x512xf32>
    %broadcast = broadcast %reshape : tensor<1x512xf32> -> tensor<64x512xf32>
    results %broadcast : tensor<64x512xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_broadcast_dim_last
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(32,8,2,64,(4,16)):(1024,128,64,0,(16,1))", #nv_tensor_ir.tensor_source<0, 0, "(32,8,2,4,16):(1024,8,1,2,64)">>}
nv_tensor_ir.graph @normalize_reduce_broadcast_dim_last(
        %in: tensor<2x4x8x16x32xf32> {nv_tensor_ir.stride = "(1,2,8,64,1024)"}) -> (tensor<512x64xf32> {nv_tensor_ir.stride = "(1,512)"}) {
    %reduce = reduce(%in)<dimensions = [1, 3], reduction_mode = <add>> : tensor<2x4x8x16x32xf32> -> tensor<2x1x8x1x32xf32>
    %transposed = transpose %reduce permutation = [4, 3, 2, 1, 0] : tensor<2x1x8x1x32xf32> -> tensor<32x1x8x1x2xf32>
    %reshaped = reshape %transposed : tensor<32x1x8x1x2xf32> -> tensor<1x512xf32>
    %reshape = transpose %reshaped permutation = [1, 0] : tensor<1x512xf32> -> tensor<512x1xf32>
    %broadcast = broadcast %reshape : tensor<512x1xf32> -> tensor<512x64xf32>
    results %broadcast : tensor<512x64xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_broadcast_dim_boundary
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(2,4,8,16,32,(4,16)):(16384,0,2048,0,64,(16,1))", #nv_tensor_ir.tensor_source<0, 0, "(2,8,32,4,16):(1,8,1024,2,64)">>}
nv_tensor_ir.graph @normalize_reduce_broadcast_dim_boundary(
        %in: tensor<2x4x8x16x32xf32> {nv_tensor_ir.stride = "(1,2,8,64,1024)"}) -> (tensor<2x4x8x16x32xf32> {nv_tensor_ir.stride = "(1,2,8,64,1024)"}) {
    %reduce = reduce(%in)<dimensions = [1, 3], reduction_mode = <add>> : tensor<2x4x8x16x32xf32> -> tensor<2x1x8x1x32xf32>
    %broadcast = broadcast %reduce : tensor<2x1x8x1x32xf32> -> tensor<2x4x8x16x32xf32>
    results %broadcast : tensor<2x4x8x16x32xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_broadcast_split_one
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(4,2,16,32,2,(4,16)):(8192,4096,0,128,64,(16,1))", #nv_tensor_ir.tensor_source<0, 0, "(4,2,32,2,4,16):(8,1,1024,32,2,64)">>}
nv_tensor_ir.graph @normalize_reduce_broadcast_split_one(
        %in: tensor<2x4x8x16x32xf32> {nv_tensor_ir.stride = "(1,2,8,64,1024)"}) -> (tensor<8x16x64xf32> {nv_tensor_ir.stride = "(1,8,128)"}) {
    %reduce = reduce(%in)<dimensions = [1, 3], reduction_mode = <add>> : tensor<2x4x8x16x32xf32> -> tensor<2x1x8x1x32xf32>
    %transposed = transpose %reduce permutation = [4, 3, 2, 1, 0] : tensor<2x1x8x1x32xf32> -> tensor<32x1x8x1x2xf32>
    %reshaped = reshape %transposed : tensor<32x1x8x1x2xf32> -> tensor<64x1x8xf32>
    %reshape = transpose %reshaped permutation = [2, 1, 0] : tensor<64x1x8xf32> -> tensor<8x1x64xf32>
    %broadcast = broadcast %reshape : tensor<8x1x64xf32> -> tensor<8x16x64xf32>
    results %broadcast : tensor<8x16x64xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_broadcast_split_many
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(3,2,2,5,4,4,7,8,9,(4,16)):(0,16384,8192,0,2048,512,0,64,0,(16,1))", #nv_tensor_ir.tensor_source<0, 0, "(2,2,4,4,8,4,16):(8,1,1024,16,4096,2,64)">>}
nv_tensor_ir.graph @normalize_reduce_broadcast_split_many(
        %in: tensor<2x4x8x16x32xf32> {nv_tensor_ir.stride = "(1,2,8,64,1024)"}) -> (tensor<3x4x5x16x7x8x9xf32> {nv_tensor_ir.stride = "(1,3,12,60,960,6720,53760)"}) {
    %reduce = reduce(%in)<dimensions = [1, 3], reduction_mode = <add>> : tensor<2x4x8x16x32xf32> -> tensor<2x1x8x1x32xf32>
    %transposed = transpose %reduce permutation = [4, 3, 2, 1, 0] : tensor<2x1x8x1x32xf32> -> tensor<32x1x8x1x2xf32>
    %reshaped = reshape %transposed : tensor<32x1x8x1x2xf32> -> tensor<1x8x1x16x1x4x1xf32>
    %reshape = transpose %reshaped permutation = [6, 5, 4, 3, 2, 1, 0] : tensor<1x8x1x16x1x4x1xf32> -> tensor<1x4x1x16x1x8x1xf32>
    %broadcast = broadcast %reshape : tensor<1x4x1x16x1x8x1xf32> -> tensor<3x4x5x16x7x8x9xf32>
    results %broadcast : tensor<3x4x5x16x7x8x9xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_unit_full
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(1,(64)):(64,(1))", #nv_tensor_ir.tensor_source<0, 0, "(1,64):(0,1)">>}
nv_tensor_ir.graph @normalize_reduce_unit_full(
        %in: tensor<1x64xf32> {nv_tensor_ir.stride = "(0,1)"}) -> (tensor<1x1xf32> {nv_tensor_ir.stride = "(0,0)"}) {
    %out = reduce(%in)<dimensions = [1], reduction_mode = <add>> : tensor<1x64xf32> -> tensor<1x1xf32>
    results %out : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: @normalize_reduce_unit_trivial
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.reduction_source<"(32,(1)):(1,(1))", #nv_tensor_ir.tensor_source<0, 0, "(32,1):(1,0)">>}
nv_tensor_ir.graph @normalize_reduce_unit_trivial(
        %in: tensor<32x1xf32> {nv_tensor_ir.stride = "(1,0)"}) -> (tensor<32x1xf32> {nv_tensor_ir.stride = "(1,0)"}) {
    %out = reduce(%in)<dimensions = [1], reduction_mode = <add>> : tensor<32x1xf32> -> tensor<32x1xf32>
    results %out : tensor<32x1xf32>
}

// -----

//===----------------------------------------------------------------------===//
// MatmulSourceAttr normalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @normalize_matmul_no_batch
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(4,8,16):(128,16,1)", 1, 4, 8, 16, #nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">, #nv_tensor_ir.tensor_source<1, 0, "(16,8):(1,16)">>}
nv_tensor_ir.graph @normalize_matmul_no_batch(
        %in0: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"},
        %in1: tensor<16x8xf32> {nv_tensor_ir.stride = "(1,16)"}) -> (tensor<4x8xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
    results %matmul : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_batch_one
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(2,4,8,16):(512,128,16,1)", 2, 4, 8, 16, #nv_tensor_ir.tensor_source<0, 0, "(2,4,16):(1,2,8)">, #nv_tensor_ir.tensor_source<1, 0, "(2,16,8):(1,2,32)">>}
nv_tensor_ir.graph @normalize_matmul_batch_one(
        %in0: tensor<2x4x16xf32> {nv_tensor_ir.stride = "(1,2,8)"},
        %in1: tensor<2x16x8xf32> {nv_tensor_ir.stride = "(1,2,32)"}) -> (tensor<2x4x8xf32> {nv_tensor_ir.stride = "(1,2,8)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<2x4x16xf32>, tensor<2x16x8xf32>) -> tensor<2x4x8xf32>
    results %matmul : tensor<2x4x8xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_batch_two
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(2,3,4,8,16):(1536,512,128,16,1)", 6, 4, 8, 16, #nv_tensor_ir.tensor_source<0, 0, "(2,3,4,16):(1,2,6,24)">, #nv_tensor_ir.tensor_source<1, 0, "(2,3,16,8):(1,2,6,96)">>}
nv_tensor_ir.graph @normalize_matmul_batch_two(
        %in0: tensor<2x3x4x16xf32> {nv_tensor_ir.stride = "(1,2,6,24)"},
        %in1: tensor<2x3x16x8xf32> {nv_tensor_ir.stride = "(1,2,6,96)"}) -> (tensor<2x3x4x8xf32> {nv_tensor_ir.stride = "(1,2,6,24)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<2x3x4x16xf32>, tensor<2x3x16x8xf32>) -> tensor<2x3x4x8xf32>
    results %matmul : tensor<2x3x4x8xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_lhs_defines_with_batch
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(3,2,5,4,30,(7,6)):(50400,25200,5040,1260,42,(6,1))", 6, 20, 30, 42, #nv_tensor_ir.tensor_source<0, 0, "(3,2,5,4,7,6):(1,3,6,30,120,840)">, #nv_tensor_ir.tensor_source<1, 0, "(3,2,7,6,30):(2,1,36,6,252)">>}
nv_tensor_ir.graph @normalize_matmul_lhs_defines_with_batch(
        %in0: tensor<3x2x5x4x7x6xf32> {nv_tensor_ir.stride = "(1,3,6,30,120,840)"},
        %in1: tensor<6x42x30xf32> {nv_tensor_ir.stride = "(1,6,252)"}) -> (tensor<6x20x30xf32> {nv_tensor_ir.stride = "(1,6,120)"}) {
    %lhs_transpose = transpose %in0 permutation = [4, 5, 2, 3, 0, 1] : tensor<3x2x5x4x7x6xf32> -> tensor<7x6x5x4x3x2xf32>
    %lhs_reshaped = reshape %lhs_transpose : tensor<7x6x5x4x3x2xf32> -> tensor<42x20x6xf32>
    %lhs_reshape = transpose %lhs_reshaped permutation = [2, 1, 0] : tensor<42x20x6xf32> -> tensor<6x20x42xf32>
    %matmul = matmul(%lhs_reshape, %in1) : (tensor<6x20x42xf32>, tensor<6x42x30xf32>) -> tensor<6x20x30xf32>
    results %matmul : tensor<6x20x30xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_lhs_defines_no_batch
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(4,3,2,32,(7,6,5)):(40320,13440,6720,210,(30,5,1))", 1, 24, 32, 210, #nv_tensor_ir.tensor_source<0, 0, "(4,3,2,7,6,5):(1,4,12,24,168,1008)">, #nv_tensor_ir.tensor_source<1, 0, "(7,6,5,32):(30,5,1,210)">>}
nv_tensor_ir.graph @normalize_matmul_lhs_defines_no_batch(
        %in0: tensor<4x3x2x7x6x5xf32> {nv_tensor_ir.stride = "(1,4,12,24,168,1008)"},
        %in1: tensor<210x32xf32> {nv_tensor_ir.stride = "(1,210)"}) -> (tensor<24x32xf32> {nv_tensor_ir.stride = "(1,24)"}) {
    %lhs_transpose = transpose %in0 permutation = [3, 4, 5, 0, 1, 2] : tensor<4x3x2x7x6x5xf32> -> tensor<7x6x5x4x3x2xf32>
    %lhs_reshaped = reshape %lhs_transpose : tensor<7x6x5x4x3x2xf32> -> tensor<210x24xf32>
    %lhs_reshape = transpose %lhs_reshaped permutation = [1, 0] : tensor<210x24xf32> -> tensor<24x210xf32>
    %matmul = matmul(%lhs_reshape, %in1) : (tensor<24x210xf32>, tensor<210x32xf32>) -> tensor<24x32xf32>
    results %matmul : tensor<24x32xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_rhs_defines_with_batch
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(2,3,10,6,5,(4,5)):(18000,6000,600,100,20,(5,1))", 6, 10, 30, 20, #nv_tensor_ir.tensor_source<0, 0, "(2,3,10,4,5):(3,1,6,300,60)">, #nv_tensor_ir.tensor_source<1, 0, "(2,3,4,5,6,5):(1,2,6,24,120,720)">>}
nv_tensor_ir.graph @normalize_matmul_rhs_defines_with_batch(
        %in0: tensor<6x10x20xf32> {nv_tensor_ir.stride = "(1,6,60)"},
        %in1: tensor<2x3x4x5x6x5xf32> {nv_tensor_ir.stride = "(1,2,6,24,120,720)"}) -> (tensor<6x10x30xf32> {nv_tensor_ir.stride = "(1,6,60)"}) {
    %rhs_transpose = transpose %in1 permutation = [4, 5, 2, 3, 0, 1] : tensor<2x3x4x5x6x5xf32> -> tensor<6x5x4x5x2x3xf32>
    %rhs_reshaped = reshape %rhs_transpose : tensor<6x5x4x5x2x3xf32> -> tensor<30x20x6xf32>
    %rhs_reshape = transpose %rhs_reshaped permutation = [2, 1, 0] : tensor<30x20x6xf32> -> tensor<6x20x30xf32>
    %matmul = matmul(%in0, %rhs_reshape) : (tensor<6x10x20xf32>, tensor<6x20x30xf32>) -> tensor<6x10x30xf32>
    results %matmul : tensor<6x10x30xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_rhs_defines_no_batch
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(16,2,3,4,(5,6,7)):(5040,2520,840,210,(42,7,1))", 1, 16, 24, 210, #nv_tensor_ir.tensor_source<0, 0, "(16,5,6,7):(1,672,112,16)">, #nv_tensor_ir.tensor_source<1, 0, "(5,6,7,2,3,4):(24,120,720,1,2,6)">>}
nv_tensor_ir.graph @normalize_matmul_rhs_defines_no_batch(
        %in0: tensor<16x210xf32> {nv_tensor_ir.stride = "(1,16)"},
        %in1: tensor<2x3x4x5x6x7xf32> {nv_tensor_ir.stride = "(1,2,6,24,120,720)"}) -> (tensor<16x24xf32> {nv_tensor_ir.stride = "(1,16)"}) {
    %rhs_reshaped = reshape %in1 : tensor<2x3x4x5x6x7xf32> -> tensor<24x210xf32>
    %rhs_reshape = transpose %rhs_reshaped permutation = [1, 0] : tensor<24x210xf32> -> tensor<210x24xf32>
    %matmul = matmul(%in0, %rhs_reshape) : (tensor<16x210xf32>, tensor<210x24xf32>) -> tensor<16x24xf32>
    results %matmul : tensor<16x24xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_output_defines_with_batch
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(2,10,4,2,10,8,4,10,8,32):(32768,0,2048,65536,0,32,8192,0,256,1)", 4, 16, 64, 32, #nv_tensor_ir.tensor_source<0, 0, "(2,2,4,4,32):(2,1,16,4,64)">, #nv_tensor_ir.tensor_source<1, 0, "(2,2,32,8,8):(2,1,4,1024,128)">>}
nv_tensor_ir.graph @normalize_matmul_output_defines_with_batch(
        %in0: tensor<4x16x32xf32> {nv_tensor_ir.stride = "(1,4,64)"},
        %in1: tensor<4x32x64xf32> {nv_tensor_ir.stride = "(1,4,128)"}) -> (tensor<2x10x8x10x32x10x8xf32> {nv_tensor_ir.stride = "(1,2,20,160,1600,51200,512000)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<4x16x32xf32>, tensor<4x32x64xf32>) -> tensor<4x16x64xf32>
    %transposed = transpose %matmul permutation = [2, 1, 0] : tensor<4x16x64xf32> -> tensor<64x16x4xf32>
    %reshaped = reshape %transposed : tensor<64x16x4xf32> -> tensor<8x1x32x1x8x1x2xf32>
    %reshape = transpose %reshaped permutation = [6, 5, 4, 3, 2, 1, 0] : tensor<8x1x32x1x8x1x2xf32> -> tensor<2x1x8x1x32x1x8xf32>
    %broadcast = broadcast %reshape : tensor<2x1x8x1x32x1x8xf32> -> tensor<2x10x8x10x32x10x8xf32>
    results %broadcast : tensor<2x10x8x10x32x10x8xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_output_defines_no_batch
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(4,4,8,8,32):(128,32,4096,512,1)", 1, 64, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(64,32):(1,64)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(1,32)">>}
nv_tensor_ir.graph @normalize_matmul_output_defines_no_batch(
        %in0: tensor<64x32xf32> {nv_tensor_ir.stride = "(1,64)"},
        %in1: tensor<32x16xf32> {nv_tensor_ir.stride = "(1,32)"}) -> (tensor<4x32x8xf32> {nv_tensor_ir.stride = "(1,4,128)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<64x32xf32>, tensor<32x16xf32>) -> tensor<64x16xf32>
    %transposed = transpose %matmul permutation = [1, 0] : tensor<64x16xf32> -> tensor<16x64xf32>
    %reshape = reshape %transposed : tensor<16x64xf32> -> tensor<4x32x8xf32>
    results %reshape : tensor<4x32x8xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_unit_m
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(8,4):(4,1)", 1, 1, 8, 4, #nv_tensor_ir.tensor_source<0, 0, "(4):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(4,8):(1,4)">>}
nv_tensor_ir.graph @normalize_matmul_unit_m(
        %in0: tensor<1x4xf32> {nv_tensor_ir.stride = "(0,1)"},
        %in1: tensor<4x8xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<1x8xf32> {nv_tensor_ir.stride = "(0,1)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<1x4xf32>, tensor<4x8xf32>) -> tensor<1x8xf32>
    results %matmul : tensor<1x8xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_unit_n
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(2,4):(4,1)", 1, 2, 1, 4, #nv_tensor_ir.tensor_source<0, 0, "(2,4):(1,2)">, #nv_tensor_ir.tensor_source<1, 0, "(4):(1)">>}
nv_tensor_ir.graph @normalize_matmul_unit_n(
        %in0: tensor<2x4xf32> {nv_tensor_ir.stride = "(1,2)"},
        %in1: tensor<4x1xf32> {nv_tensor_ir.stride = "(1,0)"}) -> (tensor<2x1xf32> {nv_tensor_ir.stride = "(1,0)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<2x4xf32>, tensor<4x1xf32>) -> tensor<2x1xf32>
    results %matmul : tensor<2x1xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_unit_k
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(2,8,1):(8,1,0)", 1, 2, 8, 1, #nv_tensor_ir.tensor_source<0, 0, "(2):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(8):(1)">>}
nv_tensor_ir.graph @normalize_matmul_unit_k(
        %in0: tensor<2x1xf32> {nv_tensor_ir.stride = "(1,0)"},
        %in1: tensor<1x8xf32> {nv_tensor_ir.stride = "(0,1)"}) -> (tensor<2x8xf32> {nv_tensor_ir.stride = "(1,2)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<2x1xf32>, tensor<1x8xf32>) -> tensor<2x8xf32>
    results %matmul : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: @normalize_matmul_unit_all
// CHECK: results {{.*}} {iteration_space = #nv_tensor_ir.matmul_source<"(1,1):(0,0)", 1, 1, 1, 1, #nv_tensor_ir.tensor_source<0, 0, "(1):(0)">, #nv_tensor_ir.tensor_source<1, 0, "(1):(0)">>}
nv_tensor_ir.graph @normalize_matmul_unit_all(
        %in0: tensor<1x1xf32> {nv_tensor_ir.stride = "(0,0)"},
        %in1: tensor<1x1xf32> {nv_tensor_ir.stride = "(0,0)"}) -> (tensor<1x1xf32> {nv_tensor_ir.stride = "(0,0)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    results %matmul : tensor<1x1xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Output normalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: nv_tensor_ir.graph @normalize_output_strides
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(8,16):(1,8)">}
nv_tensor_ir.graph @normalize_output_strides(
    %a: tensor<16x8xf32> {nv_tensor_ir.stride = "(8,1)"})
    -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(16,1)"}) {
    %t = transpose %a permutation = [1, 0] : tensor<16x8xf32> -> tensor<8x16xf32>
    results %t : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @normalize_output_coalesce
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(2,4,8,16):(1,2,8,64)">}
nv_tensor_ir.graph @normalize_output_coalesce(
    %a: tensor<16x8x4x2xf32> {nv_tensor_ir.stride = "(64,8,2,1)"})
    -> (tensor<2x4x8x16xf32> {nv_tensor_ir.stride = "(128,256,1,8)"}) {
    %t = transpose %a permutation = [3, 2, 1, 0] : tensor<16x8x4x2xf32> -> tensor<2x4x8x16xf32>
    results %t : tensor<2x4x8x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Dynamic shapes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: nv_tensor_ir.graph @normalize_dynamic_tensor_source
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(?,128):(128,1)", [0]>}
nv_tensor_ir.graph @normalize_dynamic_tensor_source(
    %a: tensor<?x8x16xf32>)
    -> (tensor<?x8x16xf32>) {
    %out = abs %a : tensor<?x8x16xf32>
    results %out : tensor<?x8x16xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @normalize_dynamic_composite_source
// CHECK: results %{{.*}} {iteration_space = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(?,8,16):(128,16,1)", [0]>, #nv_tensor_ir.tensor_source<1, 0, "(?,8,16):(128,8,1)", [0]>>}
nv_tensor_ir.graph @normalize_dynamic_composite_source(
    %a: tensor<?x8x16xf32>,
    %b: tensor<16x8x?xf32> {nv_tensor_ir.stride = "(1,8,128)"})
    -> (tensor<?x8x16xf32>) {
    %b_trans = transpose %b permutation = [2, 1, 0] : tensor<16x8x?xf32> -> tensor<?x8x16xf32>
    %out = add %a, %b_trans : tensor<?x8x16xf32>
    results %out : tensor<?x8x16xf32>
}
