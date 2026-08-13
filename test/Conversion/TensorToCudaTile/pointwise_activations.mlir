// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s
// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// Verify lowering of activation functions for a floating point type (F32).
// Other floating point types are verified in the E2E test.

// Test RELU(x) = max(0, x)
// CHECK-LABEL: @relu_f32
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK-DAG: %[[ZERO:.*]] = constant <f32: 0.000000e+00> : [[TILE]]
// CHECK-DAG: %[[RESULT:.*]] = maxf %[[ZERO]], %[[ARG0]] propagate_nan : [[TILE]]
// CHECK-DAG: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @relu_f32(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>) {
  %result = relu_fwd %arg0 : tensor<32x64xf32>
  results %result : tensor<32x64xf32>
}

// -----

// Test SIGMOID(x) = 1 / (1 + exp(-x))
// CHECK-LABEL: @sigmoid_f32
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK-DAG: %[[ONE:.*]] = constant <f32: 1.000000e+00> : [[TILE]]
// CHECK-DAG: %[[NEG:.*]] = negf %[[ARG0]] : [[TILE]]
// CHECK-DAG: %[[EXP:.*]] = exp %[[NEG]] : [[TILE]]
// CHECK-DAG: %[[ONEPLUS:.*]] = addf %[[ONE]], %[[EXP]] : [[TILE]]
// CHECK-DAG: %[[RESULT:.*]] = divf %[[ONE]], %[[ONEPLUS]] : [[TILE]]
// CHECK-DAG: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @sigmoid_f32(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>) {
  %result = sigmoid_fwd %arg0 : tensor<32x64xf32>
  results %result : tensor<32x64xf32>
}

// -----

// Test ERF(x) via Abramowitz-Stegun approximation (experimental=false).
// CHECK-LABEL: @erf_f32
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK-DAG: constant <f32: 0.327591091> : [[TILE]]
// CHECK-DAG: constant <f32: 0.254829586> : [[TILE]]
// CHECK-DAG: constant <f32: -0.284496725> : [[TILE]]
// CHECK-DAG: constant <f32: 1.42141378> : [[TILE]]
// CHECK-DAG: constant <f32: -1.45315206> : [[TILE]]
// CHECK-DAG: constant <f32: 1.06140542> : [[TILE]]
// CHECK-DAG: %[[ZERO:.*]] = constant <f32: 0.000000e+00> : [[TILE]]
// CHECK-DAG: %[[ONE:.*]] = constant <f32: 1.000000e+00> : [[TILE]]
// CHECK: %[[IS_NONNEG:.*]] = cmpf greater_than_or_equal ordered %[[ARG0]], %[[ZERO]]
// CHECK: %[[ABS:.*]] = select %[[IS_NONNEG]], %[[ARG0]], {{.*}}
// CHECK: divf %[[ONE]], {{.*}}
// CHECK: %[[EXP:.*]] = exp {{.*}}
// CHECK: %[[SCALED:.*]] = mulf {{.*}}, %[[EXP]]
// CHECK: %[[POS:.*]] = subf %[[ONE]], %[[SCALED]]
// CHECK: %[[RESULT:.*]] = select %[[IS_NONNEG]], %[[POS]], {{.*}}
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @erf_f32(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>) {
  %result = erf %arg0 : tensor<32x64xf32>
  results %result : tensor<32x64xf32>
}

// -----

// Test GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))) with approximated erf.
// CHECK-LABEL: @gelu_fwd_f32
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK-DAG: %[[SQRT2:.*]] = constant <f32: 1.414{{[0-9]+}}> : [[TILE]]
// CHECK-DAG: %[[ONE:.*]] = constant <f32: 1.000000e+00> : [[TILE]]
// CHECK-DAG: %[[HALF:.*]] = constant <f32: 5.000000e-01> : [[TILE]]
// CHECK-DAG: constant <f32: 0.327591091> : [[TILE]]
// CHECK-DAG: constant <f32: 0.254829586> : [[TILE]]
// CHECK-DAG: constant <f32: -0.284496725> : [[TILE]]
// CHECK-DAG: constant <f32: 1.42141378> : [[TILE]]
// CHECK-DAG: constant <f32: -1.45315206> : [[TILE]]
// CHECK-DAG: constant <f32: 1.06140542> : [[TILE]]
// CHECK-DAG: %[[DIV:.*]] = divf %[[ARG0]], %[[SQRT2]] : [[TILE]]
// CHECK: %[[IS_NONNEG:.*]] = cmpf greater_than_or_equal ordered %[[DIV]], {{.*}}
// CHECK: select %[[IS_NONNEG]], %[[DIV]], {{.*}}
// CHECK: %[[EXP:.*]] = exp {{.*}}
// CHECK: %[[SCALED:.*]] = mulf {{.*}}, %[[EXP]]
// CHECK: %[[POS:.*]] = subf {{.*}}, %[[SCALED]]
// CHECK: %[[ERF:.*]] = select %[[IS_NONNEG]], %[[POS]], {{.*}}
// CHECK: %[[ADD:.*]] = addf %[[ONE]], %[[ERF]] : [[TILE]]
// CHECK: %[[XHALF:.*]] = mulf %[[ARG0]], %[[HALF]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = mulf %[[XHALF]], %[[ADD]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @gelu_fwd_f32(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>) {
  %result = gelu_fwd %arg0 : tensor<32x64xf32>
  results %result : tensor<32x64xf32>
}

// -----

// Test GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// CHECK-LABEL: @gelu_approx_tanh_f32
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK-DAG: %[[CST1:.*]] = constant <f32: 0.797884583> : [[TILE]]
// CHECK-DAG: %[[CST2:.*]] = constant <f32: 4.471500e-02> : [[TILE]]
// CHECK-DAG: %[[ONE:.*]] = constant <f32: 1.000000e+00> : [[TILE]]
// CHECK-DAG: %[[HALF:.*]] = constant <f32: 5.000000e-01> : [[TILE]]
// CHECK-DAG: %[[XSQR:.*]] = mulf %[[ARG0]], %[[ARG0]] : [[TILE]]
// CHECK-DAG: %[[XCUBE:.*]] = mulf %[[XSQR]], %[[ARG0]] : [[TILE]]
// CHECK-DAG: %[[MUL1:.*]] = mulf %[[CST2]], %[[XCUBE]] : [[TILE]]
// CHECK-DAG: %[[ADD1:.*]] = addf {{.*}}%[[MUL1]]{{.*}} : [[TILE]]
// CHECK-DAG: %[[MUL2:.*]] = mulf %[[CST1]], %[[ADD1]] : [[TILE]]
// CHECK-DAG: %[[TANH:.*]] = tanh %[[MUL2]] : [[TILE]]
// CHECK-DAG: %[[ADD2:.*]] = addf %[[ONE]], %[[TANH]] : [[TILE]]
// CHECK-DAG: %[[XHALF:.*]] = mulf %[[ARG0]], %[[HALF]] : [[TILE]]
// CHECK-DAG: %[[RESULT:.*]] = mulf %[[XHALF]], %[[ADD2]] : [[TILE]]
// CHECK-DAG: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @gelu_approx_tanh_f32(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>) {
  %result = gelu_approx_tanh_fwd %arg0 : tensor<32x64xf32>
  results %result : tensor<32x64xf32>
}

// -----

// Test SOFTPLUS(x) = 1/β * log(1 + exp(β * x))
// CHECK-LABEL: @softplus_f32
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK-DAG: %[[ONE:.*]] = constant <f32: 1.000000e+00> : [[TILE]]
// CHECK-DAG: %[[TWO:.*]] = constant <f32: 2.000000e+00> : [[TILE]]
// CHECK-DAG: %[[HALF:.*]] = constant <f32: 5.000000e-01> : [[TILE]]
// CHECK-DAG: %[[MUL:.*]] = mulf %[[ARG0]], %[[TWO]] : [[TILE]]
// CHECK-DAG: %[[EXP:.*]] = exp %[[MUL]] : [[TILE]]
// CHECK-DAG: %[[ADD:.*]] = addf %[[ONE]], %[[EXP]] : [[TILE]]
// CHECK-DAG: %[[LOG:.*]] = log %[[ADD]] : [[TILE]]
// CHECK-DAG: %[[RESULT:.*]] = mulf %[[HALF]], %[[LOG]] : [[TILE]]
// CHECK-DAG: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @softplus_f32(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>) {
  %result = softplus_fwd <beta=2.0> %arg0 : tensor<32x64xf32>
  results %result : tensor<32x64xf32>
}

// -----

// Test SWISH(x) = x / (1 + exp(-β * x))
// CHECK-LABEL: @swish_f32
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK-DAG: %[[TWO:.*]] = constant <f32: 2.000000e+00> : [[TILE]]
// CHECK-DAG: %[[MUL:.*]] = mulf %[[ARG0]], %[[TWO]] : [[TILE]]
// CHECK-DAG: %[[ONE:.*]] = constant <f32: 1.000000e+00> : [[TILE]]
// CHECK-DAG: %[[NEG:.*]] = negf %[[MUL]] : [[TILE]]
// CHECK-DAG: %[[EXP:.*]] = exp %[[NEG]] : [[TILE]]
// CHECK-DAG: %[[ONEPLUS:.*]] = addf %[[ONE]], %[[EXP]] : [[TILE]]
// CHECK-DAG: %[[SIGMOID:.*]] = divf %[[ONE]], %[[ONEPLUS]] : [[TILE]]
// CHECK-DAG: %[[RESULT:.*]] = mulf %[[ARG0]], %[[SIGMOID]] : [[TILE]]
// CHECK-DAG: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @swish_f32(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>) {
  %result = swish_fwd <beta=2.0> %arg0 : tensor<32x64xf32>
  results %result : tensor<32x64xf32>
}

// -----

// Test ELU(x) = {x if x > 0; β * (exp(x) - 1) if x ≤ 0}
// CHECK-LABEL: @elu_f32
// CHECK-DAG: %[[TWO:.*]] = constant <f32: 2.000000e+00> : [[TILE:tile<([0-9]+x)+f32>]]
// CHECK-DAG: %[[ONE:.*]] = constant <f32: 1.000000e+00> : [[TILE]]
// CHECK-DAG: %[[ZERO:.*]] = constant <f32: 0.000000e+00> : [[TILE]]
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK-DAG: %[[EXP:.*]] = exp %[[ARG0]] : [[TILE]]
// CHECK-DAG: %[[EXPM1:.*]] = subf %[[EXP]], %[[ONE]] : [[TILE]]
// CHECK-DAG: %[[RHS:.*]] = mulf %[[EXPM1]], %[[TWO]] : [[TILE]]
// CHECK-DAG: %[[PRED:.*]] = cmpf greater_than ordered %[[ARG0]], %[[ZERO]] : [[TILE]] -> [[TILE_PRED:tile<([0-9]+x)+i1>]]
// CHECK-DAG: %[[RESULT:.*]] = select %[[PRED]], %[[ARG0]], %[[RHS]] : [[TILE_PRED]], [[TILE]]
// CHECK-DAG: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @elu_f32(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>) {
  %result = elu_fwd <beta=2.0> %arg0 : tensor<32x64xf32>
  results %result : tensor<32x64xf32>
}
