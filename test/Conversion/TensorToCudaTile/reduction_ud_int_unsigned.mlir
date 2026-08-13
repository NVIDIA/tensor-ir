// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// Arithmetic operations (unsigned integer)
// ============================================================================

// CHECK-LABEL: @test_reduce_add
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = addi %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_add(%arg0: tensor<64x16xui32>)
    -> (tensor<64x1xui32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0 : ui32]> (%acc: i32, %val: i32) {
    %0 = arith.addi %acc, %val : i32
    nv_tensor_ir.yield %0 : i32
  } : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_sub
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = subi %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_sub(%arg0: tensor<64x16xui32>)
    -> (tensor<64x1xui32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0 : ui32]> (%acc: i32, %val: i32) {
    %0 = arith.subi %acc, %val : i32
    nv_tensor_ir.yield %0 : i32
  } : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_mul
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[1 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = muli %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_mul(%arg0: tensor<64x16xui32>)
    -> (tensor<64x1xui32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [1 : ui32]> (%acc: i32, %val: i32) {
    %0 = arith.muli %acc, %val : i32
    nv_tensor_ir.yield %0 : i32
  } : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_max
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = maxi %[[RHS]], %[[LHS]] unsigned
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_max(%arg0: tensor<64x16xui32>)
    -> (tensor<64x1xui32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0 : ui32]> (%acc: i32, %val: i32) {
    %0 = arith.maxui %acc, %val : i32
    nv_tensor_ir.yield %0 : i32
  } : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_min
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[-1 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = mini %[[RHS]], %[[LHS]] unsigned
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_min(%arg0: tensor<64x16xui32>)
    -> (tensor<64x1xui32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [4294967295 : ui32]> (%acc: i32, %val: i32) {
    %0 = arith.minui %acc, %val : i32
    nv_tensor_ir.yield %0 : i32
  } : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// ============================================================================
// Bitwise operations (unsigned integer)
// ============================================================================

// CHECK-LABEL: @test_reduce_and
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[-1 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = andi %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_and(%arg0: tensor<64x16xui32>)
    -> (tensor<64x1xui32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0xFFFFFFFF : ui32]> (%acc: i32, %val: i32) {
    %0 = arith.andi %acc, %val : i32
    nv_tensor_ir.yield %0 : i32
  } : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_or
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = ori %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_or(%arg0: tensor<64x16xui32>)
    -> (tensor<64x1xui32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0 : ui32]> (%acc: i32, %val: i32) {
    %0 = arith.ori %acc, %val : i32
    nv_tensor_ir.yield %0 : i32
  } : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_xor
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   %[[RES:.*]] = xori %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_xor(%arg0: tensor<64x16xui32>)
    -> (tensor<64x1xui32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0 : ui32]> (%acc: i32, %val: i32) {
    %0 = arith.xori %acc, %val : i32
    nv_tensor_ir.yield %0 : i32
  } : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// ============================================================================
// Comparison operations
// ============================================================================

// CHECK-LABEL: @test_reduce_compare
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1 identities=[0 : i32]
// CHECK:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK:   cmpi equal %[[LHS]], %[[RHS]]
// CHECK:   cmpi not_equal %[[LHS]], %[[RHS]]
// CHECK:   cmpi less_than %[[LHS]], %[[RHS]], unsigned
// CHECK:   cmpi less_than_or_equal %[[LHS]], %[[RHS]], unsigned
// CHECK:   cmpi greater_than %[[LHS]], %[[RHS]], unsigned
// CHECK:   cmpi greater_than_or_equal %[[LHS]], %[[RHS]], unsigned
// CHECK:   %[[RES:.*]] = select %{{.*}}, %[[RHS]], %[[LHS]]
// CHECK:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_compare(%arg0: tensor<64x16xui32>)
    -> (tensor<64x1xui32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0 : ui32]> (%acc: i32, %val: i32) {
    %0 = arith.cmpi eq, %val, %acc : i32
    %1 = arith.cmpi ne, %val, %acc : i32
    %2 = arith.cmpi ult, %val, %acc : i32
    %3 = arith.cmpi ule, %val, %acc : i32
    %4 = arith.cmpi ugt, %val, %acc : i32
    %5 = arith.cmpi uge, %val, %acc : i32
    %6 = arith.xori %0, %1 : i1
    %7 = arith.xori %2, %6 : i1
    %8 = arith.xori %3, %7 : i1
    %9 = arith.xori %4, %8 : i1
    %10 = arith.xori %5, %9 : i1
    %11 = arith.select %10, %acc, %val : i32
    nv_tensor_ir.yield %11 : i32
  } : tensor<64x16xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}

// -----

// CHECK-LABEL: @test_reduce_with_const
// CHECK-DAG: %[[ZERO_TILE:.*]] = constant <i32: 0> : tile<32x128xi32>
// CHECK-DAG: constant <i32: 0> : tile<i32>
// CHECK-DAG: %[[LOOP:.*]] = for {{.*}} iter_values(%[[IARG:.*]] = %[[ZERO_TILE]])
// CHECK-DAG:   %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x128xi32>
// CHECK-DAG:   %[[CMP0:.*]] = cmpi greater_than_or_equal %[[ARG0]], %[[ZERO_TILE]], unsigned
// CHECK-DAG:   %[[RES0:.*]] = select %[[CMP0]], %{{.*}}, %[[IARG]]
// CHECK-DAG:   continue %[[RES0]]
// CHECK-DAG: %[[REDUCE:.*]] = reduce %[[LOOP]] dim=1 identities=[0 : i32]
// CHECK-DAG:   (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>)
// CHECK-DAG:   %[[CMP:.*]] = cmpi greater_than_or_equal %[[LHS]], {{.*}}, unsigned
// CHECK-DAG:   %[[RES:.*]] = select %[[CMP]], %{{.*}}, %[[RHS]]
// CHECK-DAG:   yield %[[RES]]

nv_tensor_ir.graph @test_reduce_with_const(%arg0: tensor<64x1024xui32>)
    -> (tensor<64x1xui32>) attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0 : ui32]> (%acc: i32, %val: i32) {
    %0 = arith.constant 0 : i32
    %1 = arith.cmpi uge, %val, %0 : i32
    %2 = arith.addi %acc, %val : i32
    %3 = arith.select %1, %2, %acc : i32
    nv_tensor_ir.yield %3 : i32
  } : tensor<64x1024xui32> -> tensor<64x1xui32>
  results %out : tensor<64x1xui32>
}
