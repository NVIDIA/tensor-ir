// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" %s | FileCheck %s
// RUN: tensor_ir-opt -layout-propagation-pipeline %s | FileCheck %s

// Verify Abramowitz-Stegun 7.1.26 approximation:
//   erf(x) ~= sign(x) * (1 - poly(t) * exp(-abs(x)^2)),
// where t = 1 / (1 + p * abs(x)).
// CHECK-LABEL: @erf_f32
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]], token
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
nv_tensor_ir.graph @erf_f32(%arg0: tensor<128xf32>) -> (tensor<128xf32>) {
  %result = erf %arg0 : tensor<128xf32>
  results %result : tensor<128xf32>
}
