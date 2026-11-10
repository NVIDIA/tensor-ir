// RUN: tensor_ir-opt %s --split-input-file --tir-erase-shape-assumptions | FileCheck %s --implicit-check-not=assume_shape
// RUN: tensor_ir-opt %s --split-input-file --pass-pipeline='builtin.module(nv_tensor_ir.graph(tir-erase-shape-assumptions),func.func(tir-erase-shape-assumptions))' | FileCheck %s --implicit-check-not=assume_shape

// CHECK-LABEL: nv_tensor_ir.graph @graph
//   CHECK-NOT: dim
//   CHECK-NOT: arith.
//       CHECK: %[[RESULT:.*]] = cos
//       CHECK: results %[[RESULT]]
nv_tensor_ir.graph @graph(%input: tensor<?xf32>, %shape: tensor<?xf32>) -> tensor<?xf32> {
  %extent = dim %shape dimension = 0 : tensor<?xf32>
  %two = arith.constant 2 : index
  %scaled = arith.muli %extent, %two : index
  assume_shape %input, [%scaled] : tensor<?xf32>
  %result = cos %input : tensor<?xf32>
  results %result : tensor<?xf32>
}

// -----

// CHECK-LABEL: func.func @function
//       CHECK: %[[SCALED:.*]] = arith.muli
//       CHECK: return {{.*}}, %[[SCALED]] : tensor<?xf32>, index
func.func @function(%input: tensor<?xf32>, %extent: index) -> (tensor<?xf32>, index) {
  %two = arith.constant 2 : index
  %scaled = arith.muli %extent, %two : index
  nv_tensor_ir.assume_shape %input, [%scaled] : tensor<?xf32>
  return %input, %scaled : tensor<?xf32>, index
}

// -----

// CHECK-LABEL: func.func @nested
//       CHECK: scf.if
//       CHECK: scf.yield
//       CHECK: } else {
//       CHECK: arith.addi
//       CHECK: scf.yield
//       CHECK: return
func.func @nested(%input: tensor<?xf32>, %extent: index, %condition: i1) -> index {
  %result = scf.if %condition -> (index) {
    nv_tensor_ir.assume_shape %input, [%extent] : tensor<?xf32>
    scf.yield %extent : index
  } else {
    %one = arith.constant 1 : index
    %next = arith.addi %extent, %one : index
    nv_tensor_ir.assume_shape %input, [%next] : tensor<?xf32>
    scf.yield %next : index
  }
  return %result : index
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @without_assumptions
//       CHECK: dim
//       CHECK: arith.constant 2 : index
//       CHECK: arith.muli
//       CHECK: %[[RESULT:.*]] = cos
//       CHECK: results %[[RESULT]]
nv_tensor_ir.graph @without_assumptions(%input: tensor<?xf32>) -> tensor<?xf32> {
  %extent = dim %input dimension = 0 : tensor<?xf32>
  %two = arith.constant 2 : index
  %scaled = arith.muli %extent, %two : index
  %result = cos %input : tensor<?xf32>
  results %result : tensor<?xf32>
}
