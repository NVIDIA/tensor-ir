// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// Reductions with small contracting dimension (no loop)
// ============================================================================

// CHECK-LABEL: @test_reduce_add_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_add_small(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <add>>
    : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_avg_small
// CHECK: %[[C16:.*]] = constant <f32: 1.600000e+01>
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: %[[RESULT:.*]] = divf %[[REDUCE]], %[[C16]]
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_avg_small(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <avg>>
    : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_norm1_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[ABS:.*]] = absf %[[ARG0]]
// CHECK: %[[REDUCE:.*]] = reduce %[[ABS]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_norm1_small(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <norm1>>
    : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_norm2_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[SQR:.*]] = mulf %[[ARG0]], %[[ARG0]]
// CHECK: %[[REDUCE:.*]] = reduce %[[SQR]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: %[[RESULT:.*]] = sqrt %[[REDUCE]]
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_norm2_small(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <norm2>>
    : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_mul_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[1.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = mulf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_mul_small(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <mul>>
    : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_mul_no_zeros_small
// CHECK-DAG: %[[ZERO:.*]] = constant <f32: 0.000000e+00>
// CHECK-DAG: %[[ONE:.*]] = constant <f32: 1.000000e+00>
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[CMP:.*]] = cmpf equal ordered %[[ARG0]], %[[ZERO]]
// CHECK: %[[SELECT:.*]] = select %[[CMP]], %[[ONE]], %[[ARG0]]
// CHECK: %[[REDUCE:.*]] = reduce %[[SELECT]] dim=1 identities=[1.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = mulf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_mul_no_zeros_small(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <mul_no_zeros>>
    : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_max_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[0xFF800000 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = maxf %[[RHS]], %[[LHS]] propagate_nan
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_max_small(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <max>>
    : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_amax_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[ABS:.*]] = absf %[[ARG0]]
// CHECK: %[[REDUCE:.*]] = reduce %[[ABS]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = maxf %[[RHS]], %[[LHS]] propagate_nan
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_amax_small(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <amax>>
    : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_min_small
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: %[[REDUCE:.*]] = reduce %[[ARG0]] dim=1 identities=[0x7F800000 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = minf %[[RHS]], %[[LHS]] propagate_nan
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_min_small(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <min>>
    : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// ============================================================================
// Reductions with large contracting dimension (loop)
// ============================================================================

// CHECK-LABEL: @test_reduce_add_large
// CHECK: %[[ACCUM:.*]] = constant <f32: 0.000000e+00>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = addf %[[IARG]], %[[ARG0]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_add_large(
    %arg0: tensor<64x8192xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <add>>
    : tensor<64x8192xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_avg_large
// CHECK-DAG: %[[C8192:.*]] = constant <f32: 8.192000e+03>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = addf %[[IARG]], %[[ARG0]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: %[[RESULT:.*]] = divf %[[REDUCE]], %[[C8192]]
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_avg_large(
    %arg0: tensor<64x8192xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <avg>>
    : tensor<64x8192xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_norm1_large
// CHECK: %[[ACCUM:.*]] = constant <f32: 0.000000e+00>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[ABS:.*]] = absf %[[ARG0]]
// CHECK:   %[[INNER:.*]] = addf %[[IARG]], %[[ABS]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_norm1_large(
    %arg0: tensor<64x8192xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <norm1>>
    : tensor<64x8192xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_norm2_large
// CHECK: %[[ACCUM:.*]] = constant <f32: 0.000000e+00>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[SQR:.*]] = mulf %[[ARG0]], %[[ARG0]]
// CHECK:   %[[INNER:.*]] = addf %[[SQR]], %[[IARG]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: %[[RESULT:.*]] = sqrt %[[REDUCE]]
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_norm2_large(
    %arg0: tensor<64x8192xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <norm2>>
    : tensor<64x8192xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_mul_large
// CHECK: %[[ACCUM:.*]] = constant <f32: 1.000000e+00>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = mulf %[[IARG]], %[[ARG0]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[1.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = mulf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_mul_large(
    %arg0: tensor<64x8192xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <mul>>
    : tensor<64x8192xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_mul_no_zeros_large
// CHECK-DAG: %[[ZERO:.*]] = constant <f32: 0.000000e+00>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 1.000000e+00>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[CMP:.*]] = cmpf equal ordered %[[ARG0]], %[[ZERO]]
// CHECK:   %[[SELECT:.*]] = select %[[CMP]], %[[ACCUM]], %[[ARG0]]
// CHECK:   %[[INNER:.*]] = mulf %[[IARG]], %[[SELECT]]
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[1.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = mulf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_mul_no_zeros_large(
    %arg0: tensor<64x8192xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <mul_no_zeros>>
    : tensor<64x8192xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_max_large
// CHECK: %[[ACCUM:.*]] = constant <f32: 0xFF800000>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = maxf %[[IARG]], %[[ARG0]] propagate_nan
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0xFF800000 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = maxf %[[RHS]], %[[LHS]] propagate_nan
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_max_large(
    %arg0: tensor<64x8192xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <max>>
    : tensor<64x8192xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_amax_large
// CHECK: %[[ACCUM:.*]] = constant <f32: 0.000000e+00>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[ABS:.*]] = absf %[[ARG0]]
// CHECK:   %[[INNER:.*]] = maxf %[[IARG]], %[[ABS]] propagate_nan
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = maxf %[[RHS]], %[[LHS]] propagate_nan
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_amax_large(
    %arg0: tensor<64x8192xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <amax>>
    : tensor<64x8192xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_min_large
// CHECK: %[[ACCUM:.*]] = constant <f32: 0x7F800000>
// CHECK: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ACCUM]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK:   %[[INNER:.*]] = minf %[[IARG]], %[[ARG0]] propagate_nan
// CHECK:   continue %[[INNER]]
// CHECK: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0x7F800000 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = minf %[[RHS]], %[[LHS]] propagate_nan
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[REDUCE]]

nv_tensor_ir.graph @test_reduce_min_large(
    %arg0: tensor<64x8192xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <min>>
    : tensor<64x8192xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}
