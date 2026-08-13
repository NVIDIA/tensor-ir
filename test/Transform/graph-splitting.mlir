// RUN: tensor_ir-opt -layout-propagation-annotation -layout-propagation-normalization -graph-splitting -split-input-file %s | FileCheck %s

// Diamond / reconvergent pattern that requires layout specialization.
// The `%mul` feeds two different slices which reconverge at `%div`.

// CHECK-LABEL: nv_tensor_ir.graph @diamond_slice_transpose
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(1,32)">>,
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,16):(32,1)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(32,1)">>,
// CHECK: transpose %{{.*}} permutation = [1, 0] {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,16):(32,1)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(32,1)">>,
// CHECK: mul %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(1,32)">>, #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,16):(32,1)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(32,1)">>>,
// CHECK: slice %{{.*}} starts = [0, 0] limits = [32, 16] strides = [1, 1] {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,16):(32,1)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(32,1)">>,
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 512, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<1, 512, "(32,16):(1,32)">>,
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 16, "(32,16):(32,1)">, #nv_tensor_ir.tensor_source<1, 16, "(32,16):(32,1)">>,
// CHECK: transpose %{{.*}} permutation = [1, 0] {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 16, "(32,16):(32,1)">, #nv_tensor_ir.tensor_source<1, 16, "(32,16):(32,1)">>,
// CHECK: mul %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 512, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<1, 512, "(32,16):(1,32)">>, #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 16, "(32,16):(32,1)">, #nv_tensor_ir.tensor_source<1, 16, "(32,16):(32,1)">>>,
// CHECK: slice %{{.*}} starts = [0, 16] limits = [32, 32] strides = [1, 1] {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 512, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<1, 512, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<0, 16, "(32,16):(32,1)">, #nv_tensor_ir.tensor_source<1, 16, "(32,16):(32,1)">>,
// CHECK: div %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,16):(32,1)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(32,1)">>, #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 512, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<1, 512, "(32,16):(1,32)">, #nv_tensor_ir.tensor_source<0, 16, "(32,16):(32,1)">, #nv_tensor_ir.tensor_source<1, 16, "(32,16):(32,1)">>>,

nv_tensor_ir.graph @diamond_slice_transpose(
  %in0: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"},
  %in1: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}
) -> (tensor<32x16xf32> {nv_tensor_ir.stride = "(1,32)"}) {
  %add = add %in0, %in1 : tensor<32x32xf32>
  %transpose = transpose %add permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
  %mul = mul %add, %transpose : tensor<32x32xf32>
  %slice0 = slice %mul starts = [0, 0] limits = [32, 16] strides = [1, 1] : tensor<32x32xf32> -> tensor<32x16xf32>
  %slice1 = slice %mul starts = [0, 16] limits = [32, 32] strides = [1, 1] : tensor<32x32xf32> -> tensor<32x16xf32>
  %div = div %slice0, %slice1 : tensor<32x16xf32>
  results %div : tensor<32x16xf32>
}

// -----

// Transpose diamond: neg(a) consumed with two different layouts (direct + transposed).
// Graph splitting must clone neg so we get one neg per layout.
// CHECK-LABEL: nv_tensor_ir.graph @transpose_diamond
// CHECK-DAG: neg %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,8):(1,8)">,
// CHECK-DAG: neg %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,8):(8,1)">,
// CHECK: transpose %{{.*}} permutation = [1, 0] {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,8):(8,1)">,
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,8):(1,8)">, #nv_tensor_ir.tensor_source<0, 0, "(8,8):(8,1)">>,
// CHECK: results %{{.*}}

nv_tensor_ir.graph @transpose_diamond(%a: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}) {
  %neg = neg %a : tensor<8x8xf32>
  %transpose = transpose %neg permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
  %add = add %neg, %transpose : tensor<8x8xf32>
  results %add : tensor<8x8xf32>
}

// -----

// Same-layout multi-consumer: neg(a) used twice with the same layout (mul %neg, %neg).
// No splitting: memoization returns the same materialized value for both uses.
// When both operands are the same value, result layout is a single tensor_source.
// CHECK-LABEL: nv_tensor_ir.graph @same_layout_multi_consumer
// CHECK: neg %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,4):(1,4)">,
// CHECK: mul %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(4,4):(1,4)">, #nv_tensor_ir.tensor_source<0, 0, "(4,4):(1,4)">>,
// CHECK: results %{{.*}}

nv_tensor_ir.graph @same_layout_multi_consumer(%a: tensor<4x4xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<4x4xf32> {nv_tensor_ir.stride = "(1,4)"}) {
  %neg = neg %a : tensor<4x4xf32>
  %mul = mul %neg, %neg : tensor<4x4xf32>
  results %mul : tensor<4x4xf32>
}

// -----

// Slice diamond: add feeds two slices with different offsets, reconverging at sub.
// Splitting clones the add (and its operands) per slice path due to different offsets.
// Output order: first add -> first slice -> second add -> second slice -> sub -> results.
// CHECK-LABEL: nv_tensor_ir.graph @slice_diamond
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>,
// CHECK: slice %{{.*}} starts = [0, 0] limits = [8, 4] strides = [1, 1]
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 32, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 32, "(8,4):(1,8)">>,
// CHECK: slice %{{.*}} starts = [0, 4] limits = [8, 8] strides = [1, 1]
// CHECK: sub %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>, #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 32, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 32, "(8,4):(1,8)">>>,
// CHECK: results %{{.*}}

nv_tensor_ir.graph @slice_diamond(
  %in0: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"},
  %in1: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}
) -> (tensor<8x4xf32> {nv_tensor_ir.stride = "(1,8)"}) {
  %add = add %in0, %in1 : tensor<8x8xf32>
  %slice0 = slice %add starts = [0, 0] limits = [8, 4] strides = [1, 1] : tensor<8x8xf32> -> tensor<8x4xf32>
  %slice1 = slice %add starts = [0, 4] limits = [8, 8] strides = [1, 1] : tensor<8x8xf32> -> tensor<8x4xf32>
  %sub = sub %slice0, %slice1 : tensor<8x4xf32>
  results %sub : tensor<8x4xf32>
}

// -----

// Longer program: transpose diamond then slice diamond, with reshape in the mix.
// add(a,b) feeds transpose (one layout) and neg (other layout); mul reconverges.
// mul then feeds two slices (different offsets); sub reconverges.
// CHECK-LABEL: nv_tensor_ir.graph @multi_diamond_reshape_transpose_slice
// First path (transpose branch): add -> transpose; second path: add -> neg -> reshape; mul reconverges.
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(16,1)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(16,1)">>,
// CHECK: transpose %{{.*}} permutation = [1, 0]
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>,
// CHECK: neg %{{.*}}
// CHECK: reshape %{{.*}}
// CHECK: mul %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(16,1)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(16,1)">>, #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>>,
// CHECK: slice %{{.*}} starts = [0, 0] limits = [8, 4] strides = [1, 1]
// Second path (slice diamond): another add/transpose/add/neg/reshape/mul then second slice.
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 4, "(8,4):(16,1)">, #nv_tensor_ir.tensor_source<1, 4, "(8,4):(16,1)">>,
// CHECK: transpose %{{.*}} permutation = [1, 0]
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 32, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 32, "(8,4):(1,8)">>,
// CHECK: neg %{{.*}}
// CHECK: reshape %{{.*}}
// CHECK: mul %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 4, "(8,4):(16,1)">, #nv_tensor_ir.tensor_source<1, 4, "(8,4):(16,1)">>, #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 32, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 32, "(8,4):(1,8)">>>,
// CHECK: slice %{{.*}} starts = [0, 4] limits = [8, 8] strides = [1, 1]
// CHECK: sub %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(16,1)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(16,1)">, #nv_tensor_ir.tensor_source<0, 0, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>, #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 4, "(8,4):(16,1)">, #nv_tensor_ir.tensor_source<1, 4, "(8,4):(16,1)">, #nv_tensor_ir.tensor_source<0, 32, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 32, "(8,4):(1,8)">>>,
// CHECK: results %{{.*}}

nv_tensor_ir.graph @multi_diamond_reshape_transpose_slice(
  %a: tensor<16x8xf32> {nv_tensor_ir.stride = "(1,16)"},
  %b: tensor<16x8xf32> {nv_tensor_ir.stride = "(1,16)"}
) -> (tensor<8x4xf32> {nv_tensor_ir.stride = "(1,8)"}) {
  %add = add %a, %b : tensor<16x8xf32>
  %transpose = transpose %add permutation = [1, 0] : tensor<16x8xf32> -> tensor<8x16xf32>
  %neg = neg %add : tensor<16x8xf32>
  %neg_transposed = transpose %neg permutation = [1, 0] : tensor<16x8xf32> -> tensor<8x16xf32>
  %neg_reshaped = reshape %neg_transposed : tensor<8x16xf32> -> tensor<16x8xf32>
  %reshape_neg = transpose %neg_reshaped permutation = [1, 0] : tensor<16x8xf32> -> tensor<8x16xf32>
  %mul = mul %transpose, %reshape_neg : tensor<8x16xf32>
  %slice0 = slice %mul starts = [0, 0] limits = [8, 4] strides = [1, 1] : tensor<8x16xf32> -> tensor<8x4xf32>
  %slice1 = slice %mul starts = [0, 4] limits = [8, 8] strides = [1, 1] : tensor<8x16xf32> -> tensor<8x4xf32>
  %sub = sub %slice0, %slice1 : tensor<8x4xf32>
  results %sub : tensor<8x4xf32>
}

// -----

// Long program: multiple reconvergent DAGs — transpose + reshape branches, then slice diamond.
// add(a,b) -> transpose (path A) and reshape (path B); mul reconverges. neg(mul) -> two slices -> add reconverges.
// CHECK-LABEL: nv_tensor_ir.graph @longer_multi_reconvergent
// First path: add, transpose, add, reshape, mul, neg, first slice.
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(8,1)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(8,1)">>,
// CHECK: transpose %{{.*}} permutation = [1, 0]
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>,
// CHECK: reshape %{{.*}}
// CHECK: mul %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(8,1)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(8,1)">>, #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>>,
// CHECK: neg %{{.*}}
// CHECK: slice %{{.*}} starts = [0, 0] limits = [8, 4] strides = [1, 1]
// Second path (slice diamond): add, transpose, add, reshape, mul, neg, second slice.
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 4, "(8,4):(8,1)">, #nv_tensor_ir.tensor_source<1, 4, "(8,4):(8,1)">>,
// CHECK: transpose %{{.*}} permutation = [1, 0]
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 32, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 32, "(8,4):(1,8)">>,
// CHECK: reshape %{{.*}}
// CHECK: mul %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 4, "(8,4):(8,1)">, #nv_tensor_ir.tensor_source<1, 4, "(8,4):(8,1)">>, #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 32, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 32, "(8,4):(1,8)">>>,
// CHECK: neg %{{.*}}
// CHECK: slice %{{.*}} starts = [0, 4] limits = [8, 8] strides = [1, 1]
// Reconverge at add(slice0, slice1).
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(8,1)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(8,1)">, #nv_tensor_ir.tensor_source<0, 0, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>, #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 4, "(8,4):(8,1)">, #nv_tensor_ir.tensor_source<1, 4, "(8,4):(8,1)">, #nv_tensor_ir.tensor_source<0, 32, "(8,4):(1,8)">, #nv_tensor_ir.tensor_source<1, 32, "(8,4):(1,8)">>>,
// CHECK: results %{{.*}}

nv_tensor_ir.graph @longer_multi_reconvergent(
  %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"},
  %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}
) -> (tensor<8x4xf32> {nv_tensor_ir.stride = "(1,8)"}) {
  %add = add %a, %b : tensor<8x8xf32>
  %t = transpose %add permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
  %r_reshape = reshape %t : tensor<8x8xf32> -> tensor<8x8xf32>
  %r = transpose %r_reshape permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
  %mul = mul %t, %r : tensor<8x8xf32>
  %neg = neg %mul : tensor<8x8xf32>
  %slice0 = slice %neg starts = [0, 0] limits = [8, 4] strides = [1, 1] : tensor<8x8xf32> -> tensor<8x4xf32>
  %slice1 = slice %neg starts = [0, 4] limits = [8, 8] strides = [1, 1] : tensor<8x8xf32> -> tensor<8x4xf32>
  %add2 = add %slice0, %slice1 : tensor<8x4xf32>
  results %add2 : tensor<8x4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @concat_diamond_input
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,1,32):(1,0,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,1,32):(1,0,32)">>,
// CHECK-SAME: iter_space_id = 1
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,1,32):(32,0,1)">, #nv_tensor_ir.tensor_source<0, 0, "(32,1,32):(32,0,1)">>,
// CHECK-SAME: iter_space_id = 2
// CHECK: transpose %{{.*}} permutation = [1, 0]
// CHECK: concatenate %{{.*}}, %{{.*}} dimension = 1 {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(32,1,32):(1,0,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,1,32):(32,0,1)">>,
// CHECK-SAME: iter_space_id = 0
// CHECK: results %{{.*}}

nv_tensor_ir.graph @concat_diamond_input(
  %in: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}
) -> (tensor<32x64xf32> {nv_tensor_ir.stride = "(1,32)"}) {
  %add = add %in, %in : tensor<32x32xf32>
  %trans = transpose %add permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
  %concat = concatenate %add, %trans dimension = 1
      : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x64xf32>
  results %concat : tensor<32x64xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @concat_diamond_output
// CHECK: concatenate %{{.*}}, %{{.*}} dimension = 1 {layout = #nv_tensor_ir.concat_source<dim = 2, #nv_tensor_ir.tensor_source<0, 0, "(2,16,1,16):(16,1,0,32)">, #nv_tensor_ir.tensor_source<0, 0, "(2,16,1,16):(16,1,0,32)">>,
// CHECK-SAME: iter_space_id = 0
// CHECK: concatenate %{{.*}}, %{{.*}} dimension = 1 {layout = #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(1,16,2,16):(0,32,16,1)">, #nv_tensor_ir.tensor_source<0, 0, "(1,16,2,16):(0,32,16,1)">>,
// CHECK-SAME: iter_space_id = 0
// CHECK: transpose %{{.*}} permutation = [1, 0]
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.concat_source<dim = 2, #nv_tensor_ir.tensor_source<0, 0, "(2,16,1,16):(16,1,0,32)">, #nv_tensor_ir.tensor_source<0, 0, "(2,16,1,16):(16,1,0,32)">>, #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(1,16,2,16):(0,32,16,1)">, #nv_tensor_ir.tensor_source<0, 0, "(1,16,2,16):(0,32,16,1)">>>,
// CHECK-SAME: iter_space_id = 0
// CHECK: results %{{.*}}

nv_tensor_ir.graph @concat_diamond_output(
  %in: tensor<32x16xf32> {nv_tensor_ir.stride = "(1,32)"}
) -> (tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}) {
  %concat = concatenate %in, %in dimension = 1
      : (tensor<32x16xf32>, tensor<32x16xf32>) -> tensor<32x32xf32>
  %trans = transpose %concat permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
  %add = add %concat, %trans : tensor<32x32xf32>
  results %add : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @concat_pruned_first_operand
// CHECK: concatenate %{{.*}}, %{{.*}} dimension = 1 {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<1, 4, "(10,1,5):(16,0,1)">, [1]>,
// CHECK-SAME: iter_space_id = 0
// CHECK: slice %{{.*}} starts = [0, 20] limits = [10, 25] strides = [1, 1]
// CHECK: results %{{.*}}

nv_tensor_ir.graph @concat_pruned_first_operand(
  %in1: tensor<32x16xf32>,
  %in2: tensor<32x16xf32>
) -> tensor<10x5xf32> {
  %concat = concatenate %in1, %in2 dimension = 1
      : (tensor<32x16xf32>, tensor<32x16xf32>) -> tensor<32x32xf32>
  %slice = slice %concat starts = [0, 20] limits = [10, 25] strides = [1, 1] : tensor<32x32xf32> -> tensor<10x5xf32>
  results %slice : tensor<10x5xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @concat_pruned_last_operand
// CHECK: concatenate %{{.*}}, %{{.*}} dimension = 1 {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 4, "(10,1,5):(16,0,1)">, [0]>,
// CHECK-SAME: iter_space_id = 0
// CHECK: slice %{{.*}} starts = [0, 4] limits = [10, 9] strides = [1, 1]
// CHECK: results %{{.*}}

nv_tensor_ir.graph @concat_pruned_last_operand(
  %in1: tensor<32x16xf32>,
  %in2: tensor<32x16xf32>
) -> tensor<10x5xf32> {
  %concat = concatenate %in1, %in2 dimension = 1
      : (tensor<32x16xf32>, tensor<32x16xf32>) -> tensor<32x32xf32>
  %slice = slice %concat starts = [0, 4] limits = [10, 9] strides = [1, 1] : tensor<32x32xf32> -> tensor<10x5xf32>
  results %slice : tensor<10x5xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @reduce_diamond_input
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">>,
// CHECK-SAME: iter_space_id = 1
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,32):(32,1)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(32,1)">>,
// CHECK-SAME: iter_space_id = 1
// CHECK: transpose %{{.*}} permutation = [1, 0]
// CHECK: mul %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(32,1)">>,
// CHECK-SAME: iter_space_id = 1
// CHECK: reduce(%{{.*}}) <dimensions = [1], reduction_mode = <add>> {layout = #nv_tensor_ir.reduction_source<"(32,(32)):(32,(1))", #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(32,1)">>>,
// CHECK-SAME: iter_space_id = 0
// CHECK: results %{{.*}}

nv_tensor_ir.graph @reduce_diamond_input(
  %in: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}
) -> (tensor<32x1xf32> {nv_tensor_ir.stride = "(1,0)"}) {
  %add = add %in, %in : tensor<32x32xf32>
  %trans = transpose %add permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
  %mul = mul %add, %trans : tensor<32x32xf32>
  %reduce = reduce(%mul)<dimensions = [1], reduction_mode = <add>> : tensor<32x32xf32> -> tensor<32x1xf32>
  results %reduce : tensor<32x1xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @reduce_diamond_output
// CHECK: reduce(%{{.*}}) <dimensions = [1], reduction_mode = <add>> {layout = #nv_tensor_ir.reduction_source<"(32,32,(32)):(32,0,(1))", #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">>,
// CHECK-SAME: iter_space_id = 0
// CHECK: broadcast %{{.*}}
// CHECK: reduce(%{{.*}}) <dimensions = [1], reduction_mode = <add>> {layout = #nv_tensor_ir.reduction_source<"(32,32,(32)):(0,32,(1))", #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">>,
// CHECK-SAME: iter_space_id = 0
// CHECK: broadcast %{{.*}}
// CHECK: transpose %{{.*}} permutation = [1, 0]
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.reduction_source<"(32,32,(32)):(32,0,(1))", #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">>, #nv_tensor_ir.reduction_source<"(32,32,(32)):(0,32,(1))", #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">>>,
// CHECK-SAME: iter_space_id = 0
// CHECK: results %{{.*}}

nv_tensor_ir.graph @reduce_diamond_output(
  %in: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}
) -> (tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}) {
  %reduce = reduce(%in)<dimensions = [1], reduction_mode = <add>> : tensor<32x32xf32> -> tensor<32x1xf32>
  %bcast = broadcast %reduce : tensor<32x1xf32> -> tensor<32x32xf32>
  %trans = transpose %bcast permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
  %add = add %bcast, %trans : tensor<32x32xf32>
  results %add : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @matmul_diamond_input
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">>,
// CHECK-SAME: iter_space_id = 1
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(32,32):(32,1)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(32,1)">>,
// CHECK-SAME: iter_space_id = 2
// CHECK: transpose %{{.*}} permutation = [1, 0]
// CHECK: matmul(%{{.*}}, %{{.*}}) {layout = #nv_tensor_ir.matmul_source<"(32,32,32):(1024,32,1)", 1, 32, 32, 32, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(32,1)">>,
// CHECK-SAME: iter_space_id = 0
// CHECK: results %{{.*}}

nv_tensor_ir.graph @matmul_diamond_input(
  %in: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}
) -> (tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}) {
  %add = add %in, %in : tensor<32x32xf32>
  %trans = transpose %add permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
  %matmul = matmul(%add, %trans) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  results %matmul : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @matmul_diamond_output
// CHECK: matmul(%{{.*}}, %{{.*}}) {layout = #nv_tensor_ir.matmul_source<"(32,32,32):(1024,32,1)", 1, 32, 32, 32, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">>,
// CHECK-SAME: iter_space_id = 0
// CHECK: matmul(%{{.*}}, %{{.*}}) {layout = #nv_tensor_ir.matmul_source<"(32,32,32):(32,1024,1)", 1, 32, 32, 32, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">>,
// CHECK-SAME: iter_space_id = 0
// CHECK: transpose %{{.*}} permutation = [1, 0]
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.matmul_source<"(32,32,32):(1024,32,1)", 1, 32, 32, 32, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">>, #nv_tensor_ir.matmul_source<"(32,32,32):(32,1024,1)", 1, 32, 32, 32, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">, #nv_tensor_ir.tensor_source<0, 0, "(32,32):(1,32)">>>,
// CHECK-SAME: iter_space_id = 0
// CHECK: results %{{.*}}

nv_tensor_ir.graph @matmul_diamond_output(
  %in: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}
) -> (tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}) {
  %matmul = matmul(%in, %in) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %trans = transpose %matmul permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
  %add = add %matmul, %trans : tensor<32x32xf32>
  results %add : tensor<32x32xf32>
}

// -----

// iota + reshape + broadcast + transpose: graph splitting back-propagates the
// iteration-space layout onto iota for codegen.
// CHECK-LABEL: nv_tensor_ir.graph @iota_layout_modifying_ops
// CHECK: iota dimension = 0 {layout = #nv_tensor_ir.tensor_source<-1, 0, "(8,4,16):(16,0,1)">,
// CHECK-SAME: iter_space_id = 0
// CHECK: reshape %{{.*}} {layout = #nv_tensor_ir.tensor_source<-1, 0, "(8,4,16):(16,0,1)">,
// CHECK-SAME: iter_space_id = 0
// CHECK: broadcast %{{.*}} {layout = #nv_tensor_ir.tensor_source<-1, 0, "(8,4,16):(16,0,1)">,
// CHECK-SAME: iter_space_id = 0
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<-1, 0, "(8,4,16):(16,0,1)">,
// CHECK-SAME: iter_space_id = 0
nv_tensor_ir.graph @iota_layout_modifying_ops() -> tensor<8x4x16xf32> {
  %iota_0 = iota dimension = 0 : tensor<128xf32>
  %reshaped = reshape %iota_0 : tensor<128xf32> -> tensor<8x1x16xf32>
  %0 = transpose %reshaped permutation = [2, 1, 0] : tensor<8x1x16xf32> -> tensor<16x1x8xf32>
  %broadcast = broadcast %0 : tensor<16x1x8xf32> -> tensor<16x4x8xf32>
  %1 = transpose %broadcast permutation = [2, 1, 0] : tensor<16x4x8xf32> -> tensor<8x4x16xf32>
  results %1 : tensor<8x4x16xf32>
}
