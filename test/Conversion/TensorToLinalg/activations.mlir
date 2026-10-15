// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)' \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg,convert-tensor-to-linalg)' \
// RUN:   | tensor_ir-opt --split-input-file --mlir-print-local-scope \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.

// CHECK-LABEL: func.func @relu_fwd(
nv_tensor_ir.graph @relu_fwd(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = relu_fwd %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 0.000000e+00
// CHECK: arith.maximumf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @relu_bwd(
nv_tensor_ir.graph @relu_bwd(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = relu_bwd %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 0.000000e+00
// CHECK: arith.constant 0.000000e+00
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @sigmoid_fwd(
nv_tensor_ir.graph @sigmoid_fwd(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = sigmoid_fwd %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @sigmoid_bwd(
nv_tensor_ir.graph @sigmoid_bwd(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = sigmoid_bwd %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @tanh_fwd(
nv_tensor_ir.graph @tanh_fwd(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = tanh_fwd %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.tanh
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @tanh_bwd(
nv_tensor_ir.graph @tanh_bwd(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = tanh_bwd %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 1.000000e+00
// CHECK: math.tanh
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @gelu_fwd(
nv_tensor_ir.graph @gelu_fwd(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = gelu_fwd %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 5.000000e-01
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.constant
// CHECK: arith.mulf
// CHECK: math.erf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @gelu_bwd(
nv_tensor_ir.graph @gelu_bwd(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = gelu_bwd %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.mulf
// CHECK: math.erf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @gelu_approx_tanh_fwd(
nv_tensor_ir.graph @gelu_approx_tanh_fwd(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = gelu_approx_tanh_fwd %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: math.tanh
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @gelu_approx_tanh_bwd(
nv_tensor_ir.graph @gelu_approx_tanh_bwd(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = gelu_approx_tanh_bwd %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: math.tanh
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @softplus_fwd(
nv_tensor_ir.graph @softplus_fwd(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = softplus_fwd<beta = 2.0 : f64> %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.constant 2.000000e+00
// CHECK: arith.mulf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: math.log
// CHECK: arith.constant 5.000000e-01
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @softplus_bwd(
nv_tensor_ir.graph @softplus_bwd(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = softplus_bwd<beta = 2.0 : f64> %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 2.000000e+00
// CHECK: arith.mulf
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @swish_fwd(
nv_tensor_ir.graph @swish_fwd(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = swish_fwd<beta = 2.0 : f64> %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 2.000000e+00
// CHECK: arith.mulf
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @swish_bwd(
nv_tensor_ir.graph @swish_bwd(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = swish_bwd<beta = 2.0 : f64> %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.constant 2.000000e+00
// CHECK: arith.mulf
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @elu_fwd(
nv_tensor_ir.graph @elu_fwd(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = elu_fwd<beta = 2.0 : f64> %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.constant 2.000000e+00
// CHECK: math.exp
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: arith.constant 0.000000e+00
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @elu_bwd(
nv_tensor_ir.graph @elu_bwd(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = elu_bwd<beta = 2.0 : f64> %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.constant 1.000000e+00
// CHECK: arith.constant 2.000000e+00
// CHECK: math.exp
// CHECK: arith.mulf
// CHECK: arith.constant 0.000000e+00
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: arith.mulf
// CHECK: linalg.yield
