// RUN: tensor_ir-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(tir-tile-reductions{reduction-tile-size=4}))' | FileCheck %s
// RUN: tensor_ir-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(canonicalize,tir-tile-reductions{reduction-tile-size=4}))' | FileCheck %s

// A scalar constant captured by the combiner must become a tensor constant.
// CHECK-LABEL: func.func @captured_float
func.func @captured_float() -> tensor<1xf32> {
  %zero = arith.constant 0.0 : f32
  %input = nv_tensor_ir.constant dense<1.0> : tensor<8xf32>
  // CHECK: scf.for
  // CHECK: %[[ZERO:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<4xf32>
  // CHECK: %[[SUM:.*]] = arith.addf {{.*}} : tensor<4xf32>
  // CHECK: %[[MAX:.*]] = arith.maximumf %[[SUM]], %[[ZERO]] {{.*}} : tensor<4xf32>
  // CHECK: scf.yield %[[MAX]] : tensor<4xf32>
  %result = nv_tensor_ir.reduce_ud(%input) <dimensions = [0], identity = [0.0 : f32]>
      (%lhs: f32, %rhs: f32) {
    %sum = arith.addf %lhs, %rhs : f32
    %max = arith.maximumf %sum, %zero : f32
    nv_tensor_ir.yield %max : f32
  } : tensor<8xf32> -> tensor<1xf32>
  return %result : tensor<1xf32>
}

// -----

// Canonicalization can hoist a body-local integer constant before tiling.
// CHECK-LABEL: func.func @hoisted_integer
func.func @hoisted_integer() -> tensor<1xsi32> {
  %input = nv_tensor_ir.constant dense<1> : tensor<8xsi32>
  // CHECK: scf.for
  // CHECK: %[[ZERO:.*]] = arith.constant {{.*}}dense<0> : tensor<4xi32>
  // CHECK: %[[SUM:.*]] = arith.addi {{.*}} : tensor<4xi32>
  // CHECK: %[[MAX:.*]] = arith.maxsi %[[SUM]], %[[ZERO]] {{.*}} : tensor<4xi32>
  %result = nv_tensor_ir.reduce_ud(%input) <dimensions = [0], identity = [0 : si32]>
      (%lhs: i32, %rhs: i32) {
    %zero = arith.constant 0 : i32
    %sum = arith.addi %lhs, %rhs : i32
    %max = arith.maxsi %sum, %zero : i32
    nv_tensor_ir.yield %max : i32
  } : tensor<8xsi32> -> tensor<1xsi32>
  return %result : tensor<1xsi32>
}

// -----

// Captures used only by the terminator also need to be mapped.
// CHECK-LABEL: func.func @yield_capture
func.func @yield_capture() -> tensor<1xf32> {
  %zero = arith.constant 0.0 : f32
  %input = nv_tensor_ir.constant dense<0.0> : tensor<8xf32>
  // CHECK: scf.for
  // CHECK: %[[ZERO:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<4xf32>
  // CHECK: scf.yield %[[ZERO]] : tensor<4xf32>
  %result = nv_tensor_ir.reduce_ud(%input) <dimensions = [0], identity = [0.0 : f32]>
      (%lhs: f32, %rhs: f32) {
    nv_tensor_ir.yield %zero : f32
  } : tensor<8xf32> -> tensor<1xf32>
  return %result : tensor<1xf32>
}

// -----

// Nonconstant captures must be broadcast without cloning their producers.
// CHECK-LABEL: func.func @scalar_capture
// CHECK-SAME: %[[SCALAR:.*]]: f32
func.func @scalar_capture(%scalar: f32) -> tensor<1xf32> {
  %input = nv_tensor_ir.constant dense<1.0> : tensor<8xf32>
  // CHECK: scf.for
  // CHECK: %[[SPLAT:.*]] = nv_tensor_ir.splat %[[SCALAR]] : tensor<4xf32>
  // CHECK: arith.maximumf {{.*}}, %[[SPLAT]] {{.*}} : tensor<4xf32>
  %result = nv_tensor_ir.reduce_ud(%input) <dimensions = [0], identity = [0.0 : f32]>
      (%lhs: f32, %rhs: f32) {
    %sum = arith.addf %lhs, %rhs : f32
    %max = arith.maximumf %sum, %scalar : f32
    nv_tensor_ir.yield %max : f32
  } : tensor<8xf32> -> tensor<1xf32>
  return %result : tensor<1xf32>
}

// -----

// Scalar arith integers are signless; TensorIR splats need signedness bridges.
// CHECK-LABEL: func.func @integer_capture
// CHECK-SAME: %[[SCALAR:.*]]: i32
func.func @integer_capture(%scalar: i32) -> tensor<1xsi32> {
  %input = nv_tensor_ir.constant dense<1> : tensor<8xsi32>
  // CHECK: scf.for
  // CHECK: %[[SIGNED:.*]] = builtin.unrealized_conversion_cast %[[SCALAR]] : i32 to si32
  // CHECK: %[[SPLAT:.*]] = nv_tensor_ir.splat %[[SIGNED]] : tensor<4xsi32>
  // CHECK: %[[SIGNLESS:.*]] = builtin.unrealized_conversion_cast %[[SPLAT]] : tensor<4xsi32> to tensor<4xi32>
  // CHECK: arith.maxsi {{.*}}, %[[SIGNLESS]] {{.*}} : tensor<4xi32>
  %result = nv_tensor_ir.reduce_ud(%input) <dimensions = [0], identity = [0 : si32]>
      (%lhs: i32, %rhs: i32) {
    %sum = arith.addi %lhs, %rhs : i32
    %max = arith.maxsi %sum, %scalar : i32
    nv_tensor_ir.yield %max : i32
  } : tensor<8xsi32> -> tensor<1xsi32>
  return %result : tensor<1xsi32>
}
