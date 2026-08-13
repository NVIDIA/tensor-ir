// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics

nv_tensor_ir.graph @operandShapesMustMatch(
  %lhs: tensor<8x32xf32>,
  %rhs: tensor<8x16xsi32>
) -> (tensor<8x32xf32>) {
  // expected-error@below {{failed to verify that all of {lhs, rhs} have same shape}}
  %result = pow %lhs, %rhs
      : (tensor<8x32xf32>, tensor<8x16xsi32>) -> tensor<8x32xf32>
  results %result : tensor<8x32xf32>
}
