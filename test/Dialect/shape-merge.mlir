// RUN: tensor_ir-opt %s | FileCheck %s

// A static operand extent refines a corresponding dynamic extent.
// CHECK-LABEL: nv_tensor_ir.graph @concatenateMergedExtent
nv_tensor_ir.graph @concatenateMergedExtent(
  %a: tensor<?x2xf32>,
  %b: tensor<4x3xf32>
) -> (tensor<4x5xf32>) {
  %result = concatenate %a, %b dimension = 1
      : (tensor<?x2xf32>, tensor<4x3xf32>) -> tensor<4x5xf32>
  results %result : tensor<4x5xf32>
}

