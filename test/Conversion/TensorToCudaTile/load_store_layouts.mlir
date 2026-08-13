// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// Load emission tests for graphs with diamond patterns, where a single input
// is read with different layouts.
// ============================================================================

// CHECK-LABEL: @loads_unary_elementwise
// CHECK: %[[TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[NEG1:.+]] = negf %[[TILE1]]
// CHECK: %[[NEG2:.+]] = negf %[[TILE2]]
// CHECK: addf %[[NEG1]], %[[NEG2]]
module {
  nv_tensor_ir.graph @loads_unary_elementwise(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %neg = neg %arg0 : tensor<32x32xf32>
    %trans = transpose %neg permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %neg, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_binary_elementwise
// CHECK: %[[ARG0_TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[ARG1_TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[ARG0_TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[ARG1_TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[MUL1:.+]] = mulf %[[ARG0_TILE1]], %[[ARG1_TILE1]]
// CHECK: %[[MUL2:.+]] = mulf %[[ARG0_TILE2]], %[[ARG1_TILE2]]
// CHECK: addf %[[MUL1]], %[[MUL2]]
module {
  nv_tensor_ir.graph @loads_binary_elementwise(
      %arg0: tensor<32x32xf32>,
      %arg1: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %mul = mul %arg0, %arg1 : tensor<32x32xf32>
    %trans = transpose %mul permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %mul, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_binary_elementwise_single_input
// CHECK: %[[TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[MUL1:.+]] = mulf %[[TILE1]], %[[TILE1]]
// CHECK: %[[MUL2:.+]] = mulf %[[TILE2]], %[[TILE2]]
// CHECK: addf %[[MUL1]], %[[MUL2]]
module {
  nv_tensor_ir.graph @loads_binary_elementwise_single_input(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %mul = mul %arg0, %arg0 : tensor<32x32xf32>
    %trans = transpose %mul permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %mul, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_binary_elementwise_lhs_input
// CHECK: %[[TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[ABS1:.+]] = absf %[[TILE1]]
// CHECK: %[[MUL1:.+]] = mulf %[[TILE1]], %[[ABS1]]
// CHECK: %[[ABS2:.+]] = absf %[[TILE2]]
// CHECK: %[[MUL2:.+]] = mulf %[[TILE2]], %[[ABS2]]
// CHECK: addf %[[MUL1]], %[[MUL2]]
module {
  nv_tensor_ir.graph @loads_binary_elementwise_lhs_input(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %abs = abs %arg0 : tensor<32x32xf32>
    %mul = mul %arg0, %abs : tensor<32x32xf32>
    %trans = transpose %mul permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %mul, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_binary_elementwise_rhs_input
// CHECK: %[[TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[ABS1:.+]] = absf %[[TILE1]]
// CHECK: %[[MUL1:.+]] = mulf %[[ABS1]], %[[TILE1]]
// CHECK: %[[ABS2:.+]] = absf %[[TILE2]]
// CHECK: %[[MUL2:.+]] = mulf %[[ABS2]], %[[TILE2]]
// CHECK: addf %[[MUL1]], %[[MUL2]]
module {
  nv_tensor_ir.graph @loads_binary_elementwise_rhs_input(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %abs = abs %arg0 : tensor<32x32xf32>
    %mul = mul %abs, %arg0 : tensor<32x32xf32>
    %trans = transpose %mul permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %mul, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_ternary_elementwise_select
// CHECK: %[[ARG0_TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xi1, strides=[32,1]>
// CHECK: %[[ARG1_TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[ARG2_TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[ARG0_TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xi1, strides=[1,32]>
// CHECK: %[[ARG1_TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[ARG2_TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[SELECT1:.+]] = select %[[ARG0_TILE1]], %[[ARG1_TILE1]], %[[ARG2_TILE1]]
// CHECK: %[[SELECT2:.+]] = select %[[ARG0_TILE2]], %[[ARG1_TILE2]], %[[ARG2_TILE2]]
// CHECK: addf %[[SELECT1]], %[[SELECT2]]
module {
  nv_tensor_ir.graph @loads_ternary_elementwise_select(
      %arg0: tensor<32x32xi1>,
      %arg1: tensor<32x32xf32>,
      %arg2: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %select = binary_select %arg0, %arg1, %arg2 : tensor<32x32xf32>
    %trans = transpose %select permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %select, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_transpose_kernel
// CHECK: %[[TILE:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: store_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
module {
  nv_tensor_ir.graph @loads_transpose_kernel(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %trans = transpose %arg0 permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    results %trans : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_multiple_convert
// CHECK: %[[TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf16, strides=[32,1]>
// CHECK: %[[TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf16, strides=[1,32]>
// CHECK: %[[CONVERT1:.+]] = ftof %[[TILE1]]
// CHECK: %[[CONVERT2:.+]] = ftof %[[TILE2]]
// CHECK: addf %[[CONVERT1]], %[[CONVERT2]]
module {
  nv_tensor_ir.graph @loads_multiple_convert(
      %arg0: tensor<32x32xf16>) ->
      (tensor<32x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %convert = convert %arg0 : tensor<32x32xf16> -> tensor<32x32xf32>
    %trans = transpose %convert permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %out = add %convert, %trans : tensor<32x32xf32>
    results %out : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: @loads_multiple_compare
// CHECK: %[[ARG0_TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[ARG1_TILE1:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[32,1]>
// CHECK: %[[ARG0_TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[ARG1_TILE2:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<32x32xf32, strides=[1,32]>
// CHECK: %[[COMPARE1:.+]] = cmpf less_than unordered %[[ARG0_TILE1]], %[[ARG1_TILE1]]
// CHECK: %[[COMPARE2:.+]] = cmpf less_than unordered %[[ARG0_TILE2]], %[[ARG1_TILE2]]
// CHECK: andi %[[COMPARE1]], %[[COMPARE2]]
module {
  nv_tensor_ir.graph @loads_multiple_compare(
      %arg0: tensor<32x32xf32>,
      %arg1: tensor<32x32xf32>) ->
      (tensor<32x32xi1>)
      attributes {tile_size = array<i32: 16, 16>} {
    %compare = cmp %arg0 ult %arg1 : tensor<32x32xf32>
    %trans = transpose %compare permutation = [1, 0] : tensor<32x32xi1> -> tensor<32x32xi1>
    %out = and %compare, %trans : tensor<32x32xi1>
    results %out : tensor<32x32xi1>
  }
}

// -----

// ============================================================================
// Store emission tests to verify the explicit strides are not lost when the
// graph iteration space doesn't match the tensor rank.
// ============================================================================

// CHECK-LABEL: @store_output_rank_increase
// CHECK: %[[TILE:.+]], %{{.+}} = load_view_tko {{.+}} tensor_view<4x8x4x8xf32, strides=[1,4,32,128]>
// CHECK: store_view_tko {{.+}} tensor_view<4x8x4x8xf32, strides=[8,1,512,64]>
module {
  nv_tensor_ir.graph @store_output_rank_increase(
      %arg0: tensor<32x32xf32>) ->
      (tensor<32x32xf32> {nv_tensor_ir.stride = "(1,64)"})
      attributes {tile_size = array<i32: 4, 8, 2, 1>} {
    %reshape1_input_reversed = transpose %arg0 permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %reshape1_reversed = reshape %reshape1_input_reversed : tensor<32x32xf32> -> tensor<8x4x8x4xf32>
    %reshape2_input_reversed = transpose %reshape1_reversed permutation = [3, 2, 1, 0] : tensor<8x4x8x4xf32> -> tensor<4x8x4x8xf32>
    %reshape2_reversed = reshape %reshape2_input_reversed : tensor<4x8x4x8xf32> -> tensor<32x32xf32>
    %reshape2 = transpose %reshape2_reversed permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    results %reshape2 : tensor<32x32xf32>
  }
}

// -----


// CHECK-LABEL: @store_output_rank_decrease
// CHECK: %[[TILE:.+]], %{{.+}} = load_view_tko {{.+}} partition_view<tile=(32), tensor_view<1024xf32, strides=[1]>>
// CHECK: store_view_tko {{.+}} : tile<32xf32>, partition_view<tile=(32), tensor_view<1024xf32, strides=[1]>>
module {
  nv_tensor_ir.graph @store_output_rank_decrease(
      %arg0: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"}) ->
      (tensor<32x32xf32> {nv_tensor_ir.stride = "(32,1)"})
      attributes {tile_size = array<i32: 32>} {
    %trans = transpose %arg0 permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    results %trans : tensor<32x32xf32>
  }
}
