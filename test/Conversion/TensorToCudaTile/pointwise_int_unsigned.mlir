// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s --check-prefixes=CHECK,AFFINE
// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s --check-prefixes=CHECK,LAYOUT

// Verify lowering of pointwise operations for an unsigned integer type (UI32).
// Other unsigned integer types are verified in the E2E test.
// Comparisons and type conversions are tested separately.

// CHECK-LABEL: entry @test_constant_op(
// CHECK: %[[RESULT:.*]] = constant <i32: 1> : tile<{{[0-9]+}}xi32>
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_constant_op() -> tensor<128xui32> {
  %out = constant dense<1> : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// Compile-time splats become shaped constants in both pipelines.
// CHECK-LABEL: entry @test_splat_op(
// AFFINE: %[[RESULT:.*]] = constant <i32: 1> : tile<{{[0-9]+}}xi32>
// AFFINE: store_view_tko weak %[[RESULT]]
// LAYOUT: %[[RESULT:.*]] = constant <i32: 1> : tile<{{[0-9]+}}xi32>
// LAYOUT-NOT: reshape
// LAYOUT-NOT: broadcast
// LAYOUT: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_splat_op() -> tensor<128xui32> {
  %cst = constant 1 : ui32
  %out = splat %cst : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: entry @test_abs_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @test_abs_op(%arg0: tensor<128xui32>) -> tensor<128xui32> {
  %out = abs %arg0 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// Integer negation wraps for both signed and unsigned TensorIR types.
// CHECK-LABEL: @test_neg_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[RESULT:.*]] = negi %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_neg_op(%arg0: tensor<128xui32>) -> tensor<128xui32> {
  %out = negi %arg0 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: entry @test_add_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = addi %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_add_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = add %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: entry @test_div_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = divi %[[ARG0]], %[[ARG1]] unsigned : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_div_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = div %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// Unsigned `mod` is the plain remainder.
// CHECK-LABEL: entry @test_mod_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = remi %[[ARG0]], %[[ARG1]] unsigned : [[TILE]]
// CHECK-NOT: select
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mod_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = mod %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// `rem` lowers to a single `cuda_tile.remi` carrying the signedness.
// CHECK-LABEL: entry @test_rem_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = remi %[[ARG0]], %[[ARG1]] unsigned : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_rem_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = rem %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: entry @test_mul_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = muli %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mul_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = mul %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_mul_high_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = mulhii %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mul_high_op(%arg0: tensor<128xui32>,
                                     %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = mul_high %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: entry @test_sub_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = subi %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_sub_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = sub %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: entry @test_min_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = mini %[[ARG0]], %[[ARG1]] unsigned : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_min_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = min %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: entry @test_max_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = maxi %[[ARG0]], %[[ARG1]] unsigned : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_max_op(%arg0: tensor<128xui32>,
                                %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = max %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: entry @test_add_square_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[SQR:.*]] = muli %[[ARG1]], %[[ARG1]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = addi {{.*}}%[[SQR]]{{.*}} : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_add_square_op(%arg0: tensor<128xui32>,
                                       %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = add_square %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: @test_shift_right_unsigned_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[SAFE:.*]] = mini %[[ARG1]], %{{.*}} unsigned : [[TILE]]
// CHECK: %[[SHIFTED:.*]] = shri %[[ARG0]], %[[SAFE]] unsigned : [[TILE]]
// CHECK: %[[IN_RANGE:.*]] = cmpi less_than %[[ARG1]], %{{.*}}, unsigned : [[TILE]]
// CHECK: %[[RESULT:.*]] = select %[[IN_RANGE]], %[[SHIFTED]], %{{.*}}
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_shift_right_unsigned_op(
    %arg0: tensor<128xui32>, %arg1: tensor<128xui32>) -> tensor<128xui32> {
  %out = shift_right %arg0, %arg1 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: entry @test_binary_select_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[PRED:tile<[0-9]+xi1>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG2:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = select %[[ARG0]], %[[ARG1]], %[[ARG2]] : [[PRED]], [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_binary_select_op(%arg0: tensor<128xi1>,
                                          %arg1: tensor<128xui32>,
                                          %arg2: tensor<128xui32>) -> tensor<128xui32> {
  %out = binary_select %arg0, %arg1, %arg2 : tensor<128xui32>
  results %out : tensor<128xui32>
}

// -----

// CHECK-LABEL: entry @test_logical_not_op(
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi1>]]
// CHECK-DAG: %[[ONE:.*]] = constant <i1: true> : [[TILE]]
// CHECK: %[[RESULT:.*]] = xori %[[ARG0]], %[[ONE]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_logical_not_op(%arg0: tensor<128xi1>) -> tensor<128xi1> {
  %out = not %arg0 : tensor<128xi1>
  results %out : tensor<128xi1>
}

// -----

// CHECK-LABEL: entry @test_logical_and_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi1>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = andi %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_logical_and_op(%arg0: tensor<128xi1>,
                                        %arg1: tensor<128xi1>) -> tensor<128xi1> {
  %out = and %arg0, %arg1 : tensor<128xi1>
  results %out : tensor<128xi1>
}

// -----

// CHECK-LABEL: entry @test_logical_or_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi1>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = ori %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_logical_or_op(%arg0: tensor<128xi1>,
                                       %arg1: tensor<128xi1>) -> tensor<128xi1> {
  %out = or %arg0, %arg1 : tensor<128xi1>
  results %out : tensor<128xi1>
}

// -----

// CHECK-LABEL: @boolean_pointwise
// CHECK: xori %{{.*}}, %{{.*}} : tile<{{.*}}xi1>
// CHECK: mulhii %{{.*}}, %{{.*}} : tile<{{.*}}xi1>
nv_tensor_ir.graph @boolean_pointwise(%a: tensor<128xi1>, %b: tensor<128xi1>)
    -> (tensor<128xi1>, tensor<128xi1>, tensor<128xi1>, tensor<128xi1>) {
  %xor = xor %a, %b : tensor<128xi1>
  %high = mul_high %a, %b : tensor<128xi1>
  %left = shift_left %a, %b : tensor<128xi1>
  %right = shift_right %a, %b : tensor<128xi1>
  results %xor, %high, %left, %right : tensor<128xi1>, tensor<128xi1>, tensor<128xi1>, tensor<128xi1>
}

// -----

// The count remains i64 until it is bounded, including counts above 2^32.
// CHECK-LABEL: @shift_ui64
// CHECK: %[[SAFE:.*]] = mini %[[COUNT:.*]], %{{.*}} unsigned : [[TILE:tile<[0-9]+xi64>]]
// CHECK: shli %{{.*}}, %[[SAFE]] : [[TILE]]
// CHECK: cmpi less_than %[[COUNT]], %{{.*}}, unsigned : [[TILE]]
// CHECK: select
nv_tensor_ir.graph @shift_ui64(%amount: tensor<128xui64>, %value: tensor<128xui64>) -> tensor<128xui64> {
  %result = shift_left %value, %amount : tensor<128xui64>
  results %result : tensor<128xui64>
}

// -----

// CHECK-LABEL: entry @boolean_negation
// CHECK: negi %{{.*}} : tile<{{.*}}xi1>
nv_tensor_ir.graph @boolean_negation(%x: tensor<128xi1>) -> tensor<128xi1> {
  %r = negi %x : tensor<128xi1>
  results %r : tensor<128xi1>
}
