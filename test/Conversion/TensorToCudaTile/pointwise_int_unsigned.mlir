// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s
// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// Verify lowering of pointwise operations for an unsigned integer type (UI32).
// Other unsigned integer types are verified in the E2E test.
// Comparisons and type conversions are tested separately.

// CHECK-LABEL: @test_constant_op
// CHECK: constant <i32: 1> : tile<{{[0-9]+}}xi32>
nv_tensor_ir.graph @test_constant_op() -> tensor<128xui32> {
  %out = constant dense<1> : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_splat_op
// CHECK: constant <i32: 1> : tile<{{[0-9]+}}xi32>
nv_tensor_ir.graph @test_splat_op() -> tensor<128xui32> {
  %cst = constant 1 : ui32
  %out = splat %cst : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_abs_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @test_abs_op(%arg0: tensor<128xui32>) -> tensor<128xui32> {
  %out = abs %arg0 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_add_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: addi %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @test_add_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = add %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_div_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: divi %[[ARG0]], %[[ARG1]] unsigned : [[TILE]]
nv_tensor_ir.graph @test_div_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = div %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_mod_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[DIV:.*]] = divi %[[ARG0]], %[[ARG1]] unsigned : [[TILE]]
// CHECK: %[[MUL:.*]] = muli %[[DIV]], %[[ARG1]] : [[TILE]]
// CHECK: subi %[[ARG0]], %[[MUL]] : [[TILE]]
nv_tensor_ir.graph @test_mod_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = mod %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// `rem` is the truncated counterpart of `mod` and lowers to a single
// `cuda_tile.remi` carrying the signedness of the element type.
// CHECK-LABEL: @test_rem_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: remi %[[ARG0]], %[[ARG1]] unsigned : [[TILE]]
nv_tensor_ir.graph @test_rem_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = rem %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_mul_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: muli %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @test_mul_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = mul %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_sub_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: subi %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @test_sub_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = sub %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_min_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: mini %[[ARG0]], %[[ARG1]] unsigned : [[TILE]]
nv_tensor_ir.graph @test_min_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = min %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_max_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: maxi %[[ARG0]], %[[ARG1]] unsigned : [[TILE]]
nv_tensor_ir.graph @test_max_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = max %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_add_square_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[SQR:.*]] = muli %[[ARG1]], %[[ARG1]] : [[TILE]]
// CHECK: addi {{.*}}%[[SQR]]{{.*}} : [[TILE]]
nv_tensor_ir.graph @test_add_square_op(%arg0: tensor<128xui32>,
                                       %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = add_square %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_binary_select_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[PRED:tile<[0-9]+xi1>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG2:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: select %[[ARG0]], %[[ARG1]], %[[ARG2]] : [[PRED]], [[TILE]]
nv_tensor_ir.graph @test_binary_select_op(%arg0: tensor<128xi1>,
                                          %arg1: tensor<128xui32>,
                                          %arg2: tensor<128xui32>) -> tensor<128xui32> {
  %out = binary_select %arg0, %arg1, %arg2 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_logical_not_op
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi1>]]
// CHECK-DAG: %[[ONE:.*]] = constant <i1: true> : [[TILE]]
// CHECK-DAG: xori %[[ARG0]], %[[ONE]] : [[TILE]]
nv_tensor_ir.graph @test_logical_not_op(%arg0: tensor<128xi1>) -> tensor<128xi1> {
  %out = not %arg0 : tensor<128xi1>
  results %out : tensor<128xi1>
}

// -----

// CHECK-LABEL: @test_logical_and_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi1>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: andi %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @test_logical_and_op(%arg0: tensor<128xi1>,
                                        %arg1: tensor<128xi1>) -> tensor<128xi1> {
  %out = and %arg0, %arg1 : tensor<128xi1>
  results %out : tensor<128xi1>
}

// -----

// CHECK-LABEL: @test_logical_or_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi1>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: ori %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @test_logical_or_op(%arg0: tensor<128xi1>,
                                       %arg1: tensor<128xi1>) -> tensor<128xi1> {
  %out = or %arg0, %arg1 : tensor<128xi1>
  results %out : tensor<128xi1>
}
