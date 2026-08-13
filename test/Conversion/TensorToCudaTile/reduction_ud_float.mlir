// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// Arithmetic operations (floating point)
// ============================================================================

// CHECK-LABEL: @test_reduce_add
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_add(%arg0: tensor<64x16xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.addf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_sub
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = subf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_sub(%arg0: tensor<64x16xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.subf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_mul
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[1.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = mulf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_mul(%arg0: tensor<64x16xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [1.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.mulf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_div
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[1.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = divf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_div(%arg0: tensor<64x16xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [1.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.divf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_rem
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[1.234500e+04 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = remf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_rem(%arg0: tensor<64x16xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [12345.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.remf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_max
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0xFF800000 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = maxf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_max(%arg0: tensor<64x16xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0xFF800000 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.maximumf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_min
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0x7F800000 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = minf %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_min(%arg0: tensor<64x16xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0x7F800000 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.minimumf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_neg
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   %[[NEG:.*]] = negf %[[LHS]]
// CHECK:   %[[RES:.*]] = addf %[[RHS]], %[[NEG]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_neg(%arg0: tensor<64x16xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.negf %val : f32
    %1 = arith.addf %acc, %0 : f32
    nv_tensor_ir.yield %1 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// ============================================================================
// Comparison operations
// ============================================================================

// CHECK-LABEL: @test_reduce_compare_ordered
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   cmpf equal ordered %[[LHS]], %[[RHS]]
// CHECK:   cmpf not_equal ordered %[[LHS]], %[[RHS]]
// CHECK:   cmpf less_than ordered %[[LHS]], %[[RHS]]
// CHECK:   cmpf less_than_or_equal ordered %[[LHS]], %[[RHS]]
// CHECK:   cmpf greater_than ordered %[[LHS]], %[[RHS]]
// CHECK:   cmpf greater_than_or_equal ordered %[[LHS]], %[[RHS]]
// CHECK:   %[[RES:.*]] = select %{{.*}}, %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_compare_ordered(%arg0: tensor<64x16xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.cmpf oeq, %val, %acc : f32
    %1 = arith.cmpf one, %val, %acc : f32
    %2 = arith.cmpf olt, %val, %acc : f32
    %3 = arith.cmpf ole, %val, %acc : f32
    %4 = arith.cmpf ogt, %val, %acc : f32
    %5 = arith.cmpf oge, %val, %acc : f32
    %6 = arith.xori %0, %1 : i1
    %7 = arith.xori %2, %6 : i1
    %8 = arith.xori %3, %7 : i1
    %9 = arith.xori %4, %8 : i1
    %10 = arith.xori %5, %9 : i1
    %11 = arith.select %10, %acc, %val : f32
    nv_tensor_ir.yield %11 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_compare_unordered
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK:   cmpf equal unordered %[[LHS]], %[[RHS]]
// CHECK:   cmpf not_equal unordered %[[LHS]], %[[RHS]]
// CHECK:   cmpf less_than unordered %[[LHS]], %[[RHS]]
// CHECK:   cmpf less_than_or_equal unordered %[[LHS]], %[[RHS]]
// CHECK:   cmpf greater_than unordered %[[LHS]], %[[RHS]]
// CHECK:   cmpf greater_than_or_equal unordered %[[LHS]], %[[RHS]]
// CHECK:   %[[RES:.*]] = select %{{.*}}, %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_compare_unordered(%arg0: tensor<64x16xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.cmpf ueq, %val, %acc : f32
    %1 = arith.cmpf une, %val, %acc : f32
    %2 = arith.cmpf ult, %val, %acc : f32
    %3 = arith.cmpf ule, %val, %acc : f32
    %4 = arith.cmpf ugt, %val, %acc : f32
    %5 = arith.cmpf uge, %val, %acc : f32
    %6 = arith.xori %0, %1 : i1
    %7 = arith.xori %2, %6 : i1
    %8 = arith.xori %3, %7 : i1
    %9 = arith.xori %4, %8 : i1
    %10 = arith.xori %5, %9 : i1
    %11 = arith.select %10, %acc, %val : f32
    nv_tensor_ir.yield %11 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_with_const
// CHECK-DAG: %[[ZERO_TILE:.*]] = constant <f32: 0.000000e+00> : tile<32x128xf32>
// CHECK-DAG: constant <f32: 0.000000e+00> : tile<f32>
// CHECK-DAG: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ZERO_TILE]])
// CHECK-DAG:   %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x128xf32>
// CHECK-DAG:   %[[CMP0:.*]] = cmpf greater_than_or_equal ordered %[[ARG0]], %[[ZERO_TILE]]
// CHECK-DAG:   %[[RES0:.*]] = select %[[CMP0]], %{{.*}}, %[[IARG]]
// CHECK-DAG:   continue %[[RES0]]
// CHECK-DAG: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0.000000e+00 : f32]
// CHECK-DAG:   (%[[LHS:.*]]: tile<f32>, %[[RHS:.*]]: tile<f32>)
// CHECK-DAG:   %[[CMP:.*]] = cmpf greater_than_or_equal ordered %[[LHS]], {{.*}}
// CHECK-DAG:   %[[RES:.*]] = select %[[CMP]], %{{.*}}, %[[RHS]]
// CHECK-DAG:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_with_const(%arg0: tensor<64x1024xf32>)
    -> (tensor<64x1xf32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.constant 0.0 : f32
    %1 = arith.cmpf oge, %val, %0 : f32
    %2 = arith.addf %acc, %val : f32
    %3 = arith.select %1, %2, %acc : f32
    nv_tensor_ir.yield %3 : f32
  } : tensor<64x1024xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}
