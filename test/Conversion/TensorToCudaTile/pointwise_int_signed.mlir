// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s
// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// Verify lowering of pointwise operations for a signed integer type (SI32).
// Other signed integer types are verified in the E2E test.
// Comparisons and type conversions are tested separately.

// CHECK-LABEL: @test_constant_op
// CHECK: %[[RESULT:.*]] = constant <i32: -1> : tile<{{[0-9]+}}xi32>
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_constant_op() -> tensor<128xsi32> {
  %out = constant dense<-1> : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_splat_op
// CHECK: %[[RESULT:.*]] = constant <i32: -1> : tile<{{[0-9]+}}xi32>
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_splat_op() -> tensor<128xsi32> {
  %cst = constant -1 : si32
  %out = splat %cst : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_abs_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[RESULT:.*]] = absi %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_abs_op(%arg0: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = abs %arg0 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_add_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = addi %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_add_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = add %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_div_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = divi %[[ARG0]], %[[ARG1]] signed : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_div_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = div %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_mod_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[DIV:.*]] = divi %[[ARG0]], %[[ARG1]] signed rounding<negative_inf> : [[TILE]]
// CHECK: %[[MUL:.*]] = muli %[[DIV]], %[[ARG1]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = subi %[[ARG0]], %[[MUL]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mod_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = mod %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// `rem` is the truncated counterpart of `mod` and lowers to a single
// `cuda_tile.remi` carrying the signedness of the element type.
// CHECK-LABEL: @test_rem_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = remi %[[ARG0]], %[[ARG1]] signed : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_rem_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = rem %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_mul_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = muli %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mul_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = mul %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_sub_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = subi %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_sub_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = sub %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_min_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = mini %[[ARG0]], %[[ARG1]] signed : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_min_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = min %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_max_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = maxi %[[ARG0]], %[[ARG1]] signed : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_max_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = max %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_add_square_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[SQR:.*]] = muli %[[ARG1]], %[[ARG1]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = addi %[[ARG0]], %[[SQR]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_add_square_op(%arg0: tensor<128xsi32>,
                                       %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = add_square %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_binary_select_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[PRED:tile<[0-9]+xi1>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG2:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = select %[[ARG0]], %[[ARG1]], %[[ARG2]] : [[PRED]], [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_binary_select_op(%arg0: tensor<128xi1>,
                                          %arg1: tensor<128xsi32>,
                                          %arg2: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = binary_select %arg0, %arg1, %arg2 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}
