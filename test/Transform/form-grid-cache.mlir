// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' | FileCheck %s

// The same producer is needed in both branches and after the conditional.
// Branch-local loads, compute values, and scalar setup must not escape their
// regions. The verifier checks dominance; captures check reuse within a scope.
// CHECK-LABEL: func.func @shared_producer
//       CHECK: scf.forall
//       CHECK: %[[BRANCH:.*]] = scf.if
//       CHECK: %[[THEN_LOAD:.*]] = nv_tensor_ir.load
//       CHECK: %[[THEN_SCALAR:.*]] = arith.addf
//       CHECK: %[[THEN_SPLAT:.*]] = nv_tensor_ir.splat %[[THEN_SCALAR]]
//       CHECK: %[[THEN_VALUE:.*]] = nv_tensor_ir.add %[[THEN_LOAD]], %[[THEN_SPLAT]]
//       CHECK: scf.yield %[[THEN_VALUE]]
//       CHECK: } else {
//       CHECK: %[[ELSE_LOAD:.*]] = nv_tensor_ir.load
//       CHECK: %[[ELSE_SCALAR:.*]] = arith.addf
//       CHECK: %[[ELSE_SPLAT:.*]] = nv_tensor_ir.splat %[[ELSE_SCALAR]]
//       CHECK: %[[ELSE_VALUE:.*]] = nv_tensor_ir.add %[[ELSE_LOAD]], %[[ELSE_SPLAT]]
//       CHECK: scf.yield %[[ELSE_VALUE]]
//       CHECK: }
//       CHECK: %[[OUTER_LOAD:.*]] = nv_tensor_ir.load
//       CHECK: %[[OUTER_SCALAR:.*]] = arith.addf
//       CHECK: %[[OUTER_SPLAT:.*]] = nv_tensor_ir.splat %[[OUTER_SCALAR]]
//       CHECK: %[[OUTER_VALUE:.*]] = nv_tensor_ir.add %[[OUTER_LOAD]], %[[OUTER_SPLAT]]
//       CHECK: %[[RESULT:.*]] = nv_tensor_ir.add %[[BRANCH]], %[[OUTER_VALUE]]
//       CHECK: nv_tensor_ir.store %[[RESULT]]
#shared = #nv_tensor_ir.composite_source<
    #nv_tensor_ir.tensor_source<0, 0, "(16):(0)">,
    #nv_tensor_ir.tensor_source<-1, 0, "(16):(0)">>
#half = #nv_tensor_ir.composite_source<
    #nv_tensor_ir.tensor_source<0, 0, "(8):(0)">,
    #nv_tensor_ir.tensor_source<-1, 0, "(8):(0)">>
#concat = #nv_tensor_ir.concat_source<dim = 0, #half, #half>
#result = #nv_tensor_ir.composite_source<#concat, #shared>
func.func @shared_producer(%input: !ptr.ptr<#ptr.generic_space>,
    %output: !ptr.ptr<#ptr.generic_space>, %scalar: f32) attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #result,
  tile_size = array<i32: 1>
} {
  %zero = arith.constant 0 : index
  %input_tile = nv_tensor_ir.load %input[%zero]
      view offset: [0], sizes: [1], strides: [1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<1xf32>
  %twice = arith.addf %scalar, %scalar : f32
  %splat = nv_tensor_ir.splat %twice {
      layout = #nv_tensor_ir.tensor_source<-1, 0, "(16):(0)">}
      : tensor<1xf32>
  %shared = nv_tensor_ir.add %input_tile, %splat {layout = #shared}
      : tensor<1xf32>
  %half = nv_tensor_ir.broadcast %shared {layout = #half}
      : tensor<1xf32> -> tensor<8xf32>
  %concat = nv_tensor_ir.concatenate %half, %half dimension = 0
      {layout = #concat}
      : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  %whole = nv_tensor_ir.broadcast %shared {layout = #shared}
      : tensor<1xf32> -> tensor<16xf32>
  %result = nv_tensor_ir.add %concat, %whole {layout = #result}
      : tensor<16xf32>
  nv_tensor_ir.store %result, %output[%zero]
      view offset: [0], sizes: [16], strides: [1], alignment: 4
      : tensor<16xf32>, !ptr.ptr<#ptr.generic_space>
  return
}
