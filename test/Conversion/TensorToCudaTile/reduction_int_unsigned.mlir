// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// Reductions with small contracting dimension (no loop)
// ============================================================================

// CHECK-LABEL: @test_reduce_add_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = addi %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_add_small(
    %arg0: tensor<64x16xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <add>>
    : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_avg_small
// CHECK: %[[C16:.*]] = constant <i32: 16>
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = addi %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: %[[RESULT:.*]] = divi %[[REDUCE]], %[[C16]] unsigned
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_avg_small(
    %arg0: tensor<64x16xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <avg>>
    : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_norm1_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = addi %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_norm1_small(
    %arg0: tensor<64x16xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <norm1>>
    : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_mul_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[1 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = muli %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_mul_small(
    %arg0: tensor<64x16xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <mul>>
    : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_mul_no_zeros_small
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[CMP:.*]] = cmpi equal %[[ARG0]], %[[ZERO]]
// CHECK: %[[SELECT:.*]] = select %[[CMP]], %[[ONE]], %[[ARG0]]
// CHECK: %[[REDUCE:.*]] = reduce %[[SELECT]] dim=1 identities=[1 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = muli %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_mul_no_zeros_small(
    %arg0: tensor<64x16xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <mul_no_zeros>>
    : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_max_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = maxi %[[RHS]], %[[LHS]] unsigned
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_max_small(
    %arg0: tensor<64x16xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <max>>
    : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_amax_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = maxi %[[RHS]], %[[LHS]] unsigned
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_amax_small(
    %arg0: tensor<64x16xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <amax>>
    : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_min_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[-1 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = mini %[[RHS]], %[[LHS]] unsigned
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_min_small(
    %arg0: tensor<64x16xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <min>>
    : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// ============================================================================
// Reductions with large contracting dimension (loop)
// ============================================================================

// CHECK-LABEL: @test_reduce_add_large
// CHECK: %[[ACCUM:.*]] = constant <i32: 0> : tile<32x128xi32>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = addi %[[IARG]], %[[ARG0]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = addi %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_add_large(
    %arg0: tensor<64x8192xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <add>>
    : tensor<64x8192xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_avg_large
// CHECK-DAG: %[[C8192:.*]] = constant <i32: 8192>
// CHECK-DAG: %[[ACCUM:.*]] = constant <i32: 0> : tile<32x128xi32>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = addi %[[IARG]], %[[ARG0]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = addi %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: %[[RESULT:.*]] = divi %[[REDUCE]], %[[C8192]] unsigned
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_avg_large(
    %arg0: tensor<64x8192xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <avg>>
    : tensor<64x8192xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_norm1_large
// CHECK: %[[ACCUM:.*]] = constant <i32: 0> : tile<32x128xi32>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = addi %[[IARG]], %[[ARG0]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = addi %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_norm1_large(
    %arg0: tensor<64x8192xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <norm1>>
    : tensor<64x8192xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_mul_large
// CHECK: %[[ACCUM:.*]] = constant <i32: 1> : tile<32x128xi32>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = muli %[[IARG]], %[[ARG0]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[1 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = muli %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_mul_large(
    %arg0: tensor<64x8192xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <mul>>
    : tensor<64x8192xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_mul_no_zeros_large
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ACCUM:.*]] = constant <i32: 1> : tile<32x128xi32>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[CMP:.*]] = cmpi equal %[[ARG0]], %[[ZERO]]
// CHECK:   %[[SELECT:.*]] = select %[[CMP]], %[[ACCUM]], %[[ARG0]]
// CHECK:   %[[INNER:.*]] = muli %[[IARG]], %[[SELECT]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[1 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = muli %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_mul_no_zeros_large(
    %arg0: tensor<64x8192xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <mul_no_zeros>>
    : tensor<64x8192xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_max_large
// CHECK: %[[ACCUM:.*]] = constant <i32: 0> : tile<32x128xi32>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = maxi %[[IARG]], %[[ARG0]] unsigned
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = maxi %[[RHS]], %[[LHS]] unsigned
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_max_large(
    %arg0: tensor<64x8192xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <max>>
    : tensor<64x8192xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_amax_large
// CHECK: %[[ACCUM:.*]] = constant <i32: 0> : tile<32x128xi32>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = maxi %[[IARG]], %[[ARG0]] unsigned
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = maxi %[[RHS]], %[[LHS]] unsigned
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_amax_large(
    %arg0: tensor<64x8192xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <amax>>
    : tensor<64x8192xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_min_large
// CHECK: %[[ACCUM:.*]] = constant <i32: -1>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = mini %[[IARG]], %[[ARG0]] unsigned
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[-1 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = mini %[[RHS]], %[[LHS]] unsigned
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_min_large(
    %arg0: tensor<64x8192xui32>
    ) -> (tensor<64x1xui32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <min>>
    : tensor<64x8192xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// ============================================================================
// Verify identity value for min reductions
// ============================================================================

// CHECK-LABEL: @test_reduce_min_i8
// CHECK: constant <i8: -1> : tile<32x128xi8>
// CHECK: identities=[-1 : i8]

nv_tensor_ir.graph @test_reduce_min_i8(
    %arg0: tensor<64x8192xui8>
    ) -> (tensor<64x1xui8>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <min>>
    : tensor<64x8192xui8> -> tensor<64x1xui8>
  results %out : tensor<64x1xui8>
}

// -----

// CHECK-LABEL: @test_reduce_min_i16
// CHECK: constant <i16: -1> : tile<32x128xi16>
// CHECK: identities=[-1 : i16]

nv_tensor_ir.graph @test_reduce_min_i16(
    %arg0: tensor<64x8192xui16>
    ) -> (tensor<64x1xui16>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <min>>
    : tensor<64x8192xui16> -> tensor<64x1xui16>
  results %out : tensor<64x1xui16>
}

// -----

// CHECK-LABEL: @test_reduce_min_i64
// CHECK: constant <i64: -1> : tile<32x128xi64>
// CHECK: identities=[-1]

nv_tensor_ir.graph @test_reduce_min_i64(
    %arg0: tensor<64x8192xui64>
    ) -> (tensor<64x1xui64>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <min>>
    : tensor<64x8192xui64> -> tensor<64x1xui64>
  results %out : tensor<64x1xui64>
}

// -----

// ============================================================================
// Verify large reduction with small types (mode=avg)
// ============================================================================

// CHECK-LABEL: @test_reduce_avg_i8
// CHECK: %[[ZERO:.*]] = constant <i8: 0> : tile<32xi8>
// CHECK: store_view_tko weak %[[ZERO]]

nv_tensor_ir.graph @test_reduce_avg_i8(
    %arg0: tensor<64x256xui8>
    ) -> (tensor<64x1xui8>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <avg>>
    : tensor<64x256xui8> -> tensor<64x1xui8>
  results %out : tensor<64x1xui8>
}

// -----

// CHECK-LABEL: @test_reduce_avg_i16
// CHECK: %[[ZERO:.*]] = constant <i16: 0> : tile<32xi16>
// CHECK: store_view_tko weak %[[ZERO]]

nv_tensor_ir.graph @test_reduce_avg_i16(
    %arg0: tensor<64x65536xui16>
    ) -> (tensor<64x1xui16>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <avg>>
    : tensor<64x65536xui16> -> tensor<64x1xui16>
  results %out : tensor<64x1xui16>
}
