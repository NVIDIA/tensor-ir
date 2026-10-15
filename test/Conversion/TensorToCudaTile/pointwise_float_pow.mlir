// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map target-bytecode-version=13.3" -split-input-file %s | FileCheck %s --check-prefixes=CHECK,V13_3
// RUN: tensor_ir-opt -layout-propagation-pipeline="bytecode-version=13.3" -split-input-file %s | FileCheck %s --check-prefixes=CHECK,V13_3
// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map target-bytecode-version=13.4" -split-input-file %s | FileCheck %s --check-prefixes=CHECK,V13_4
// RUN: tensor_ir-opt -layout-propagation-pipeline="bytecode-version=13.4" -split-input-file %s | FileCheck %s --check-prefixes=CHECK,V13_4

// Signed integer exponents use fpowi starting with bytecode 13.4.

// CHECK-LABEL: @test_pow_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: fpowf %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @test_pow_op(%arg0: tensor<128xf32>,
                                %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = pow %arg0, %arg1 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_pow_exp_sint
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE_EXP:tile<[0-9]+xi32>]]
// V13_4-NOT: itof
// V13_4: fpowi %[[ARG0]], %[[ARG1]] : [[TILE]], [[TILE_EXP]]
// V13_3: %[[EXP:.*]] = itof %[[ARG1]] signed : [[TILE_EXP]] -> [[TILE]]
// V13_3: fpowf %[[ARG0]], %[[EXP]] : [[TILE]]
nv_tensor_ir.graph @test_pow_exp_sint(%arg0: tensor<128xf32>,
                                      %arg1: tensor<128xsi32>) -> tensor<128xf32> {
  %out = pow %arg0, %arg1 : (tensor<128xf32>, tensor<128xsi32>) -> tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_pow_exp_uint
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE_EXP:tile<[0-9]+xi32>]]
// CHECK: %[[EXP:.*]] = itof %[[ARG1]] unsigned : [[TILE_EXP]] -> [[TILE]]
// CHECK: fpowf %[[ARG0]], %[[EXP]] : [[TILE]]
nv_tensor_ir.graph @test_pow_exp_uint(%arg0: tensor<128xf32>,
                                      %arg1: tensor<128xui32>) -> tensor<128xf32> {
  %out = pow %arg0, %arg1 : (tensor<128xf32>, tensor<128xui32>) -> tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_pow_exp_i64
// CHECK: %[[EXP:.*]] = itof %{{.*}} signed : tile<{{[0-9]+}}xi64> -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: fpowf %{{.*}}, %[[EXP]] : [[TILE]]
nv_tensor_ir.graph @test_pow_exp_i64(%arg0: tensor<128xf32>,
                                     %arg1: tensor<128xsi64>) -> tensor<128xf32> {
  %out = pow %arg0, %arg1 : (tensor<128xf32>, tensor<128xsi64>) -> tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_pow_exp_si8
// V13_3: %[[EXP:.*]] = itof %{{.*}} signed : [[EXP_TILE:tile<[0-9]+xi8>]] -> [[TILE:tile<[0-9]+xf32>]]
// V13_3: fpowf %{{.*}}, %[[EXP]] : [[TILE]]
// V13_4-NOT: itof
// V13_4: fpowi %{{.*}}, %{{.*}} : tile<{{[0-9]+}}xf32>, tile<{{[0-9]+}}xi8>
nv_tensor_ir.graph @test_pow_exp_si8(%arg0: tensor<128xf32>,
                                      %arg1: tensor<128xsi8>) -> tensor<128xf32> {
  %out = pow %arg0, %arg1 : (tensor<128xf32>, tensor<128xsi8>) -> tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_pow_exp_si16
// V13_3: %[[EXP:.*]] = itof %{{.*}} signed : [[EXP_TILE:tile<[0-9]+xi16>]] -> [[TILE:tile<[0-9]+xf32>]]
// V13_3: fpowf %{{.*}}, %[[EXP]] : [[TILE]]
// V13_4-NOT: itof
// V13_4: fpowi %{{.*}}, %{{.*}} : tile<{{[0-9]+}}xf32>, tile<{{[0-9]+}}xi16>
nv_tensor_ir.graph @test_pow_exp_si16(%arg0: tensor<128xf32>,
                                      %arg1: tensor<128xsi16>) -> tensor<128xf32> {
  %out = pow %arg0, %arg1 : (tensor<128xf32>, tensor<128xsi16>) -> tensor<128xf32>
  results %out : tensor<128xf32>
}
