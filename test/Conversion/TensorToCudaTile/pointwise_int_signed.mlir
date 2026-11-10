// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map target-bytecode-version=13.2" -split-input-file %s | FileCheck %s --check-prefixes=CHECK,AFFINE
// RUN: tensor_ir-opt -layout-propagation-pipeline="bytecode-version=13.2" -split-input-file %s | FileCheck %s --check-prefixes=CHECK,LAYOUT

// Verify lowering of pointwise operations for a signed integer type (SI32).
// Other signed integer types are verified in the E2E test.
// Comparisons and type conversions are tested separately.

// CHECK-LABEL: entry @test_constant_op(
// CHECK: %[[RESULT:.*]] = constant <i32: -1> : tile<{{[0-9]+}}xi32>
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_constant_op() -> tensor<128xsi32> {
  %out = constant dense<-1> : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// Compile-time splats become shaped constants in both pipelines.
// CHECK-LABEL: entry @test_splat_op(
// AFFINE: %[[RESULT:.*]] = constant <i32: -1> : tile<{{[0-9]+}}xi32>
// AFFINE: store_view_tko weak %[[RESULT]]
// LAYOUT: %[[RESULT:.*]] = constant <i32: -1> : tile<{{[0-9]+}}xi32>
// LAYOUT-NOT: reshape
// LAYOUT-NOT: broadcast
// LAYOUT: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_splat_op() -> tensor<128xsi32> {
  %cst = constant -1 : si32
  %out = splat %cst : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: entry @test_abs_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[RESULT:.*]] = absi %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_abs_op(%arg0: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = abs %arg0 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_neg_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[RESULT:.*]] = negi %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_neg_op(%arg0: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = negi %arg0 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: entry @test_add_op(
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

// CHECK-LABEL: entry @test_div_op(
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

// Floored `mod` lowers to `remi` plus a sign-conditional select.
// CHECK-LABEL: entry @test_mod_op(
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK-DAG: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0> : [[TILE]]
// CHECK: %[[REM:.*]] = remi %[[ARG0]], %[[ARG1]] signed : [[TILE]]
// CHECK: %[[NONZERO:.*]] = cmpi not_equal %[[REM]], %[[ZERO]], signed : [[TILE]]
// CHECK: %[[SIGN_BITS:.*]] = xori %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: %[[SIGNS_DIFFER:.*]] = cmpi less_than %[[SIGN_BITS]], %[[ZERO]], signed : [[TILE]]
// CHECK: %[[FIX:.*]] = andi %[[NONZERO]], %[[SIGNS_DIFFER]]
// CHECK: %[[FIXED:.*]] = addi %[[REM]], %[[ARG1]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = select %[[FIX]], %[[FIXED]], %[[REM]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mod_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = mod %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// `rem` lowers to a single `cuda_tile.remi` carrying the signedness.
// CHECK-LABEL: entry @test_rem_op(
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

// CHECK-LABEL: entry @test_mul_op(
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

// CHECK-LABEL: entry @test_sub_op(
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

// CHECK-LABEL: entry @test_min_op(
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

// CHECK-LABEL: entry @test_max_op(
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

// CHECK-LABEL: entry @test_add_square_op(
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

// CHECK-LABEL: @test_and_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = andi %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_and_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = and %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_or_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = ori %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_or_op(%arg0: tensor<128xsi32>,
                               %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = or %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_xor_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = xori %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_xor_op(%arg0: tensor<128xsi32>,
                                %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = xor %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_shift_left_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[SAFE:.*]] = mini %[[ARG1]], %{{.*}} unsigned : [[TILE]]
// CHECK: %[[SHIFTED:.*]] = shli %[[ARG0]], %[[SAFE]] : [[TILE]]
// CHECK: %[[IN_RANGE:.*]] = cmpi less_than %[[ARG1]], %{{.*}}, unsigned : [[TILE]]
// CHECK: %[[RESULT:.*]] = select %[[IN_RANGE]], %[[SHIFTED]], %{{.*}}
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_shift_left_op(%arg0: tensor<128xsi32>,
                                       %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = shift_left %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: @test_shift_right_signed_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xi32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[SAFE:.*]] = mini %[[ARG1]], %{{.*}} unsigned : [[TILE]]
// CHECK: %[[RESULT:.*]] = shri %[[ARG0]], %[[SAFE]] signed : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_shift_right_signed_op(
    %arg0: tensor<128xsi32>, %arg1: tensor<128xsi32>) -> tensor<128xsi32> {
  %out = shift_right %arg0, %arg1 : tensor<128xsi32>
  results %out : tensor<128xsi32>
}

// -----

// CHECK-LABEL: entry @test_binary_select_op(
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

// -----

// CHECK-LABEL: @shift_overflow
// CHECK: shli %{{.*}}, %{{.*}} : tile<{{.*}}xi32>
// CHECK: shli %{{.*}}, %{{.*}} overflow<no_signed_wrap> : tile<{{.*}}xi32>
// CHECK: shli %{{.*}}, %{{.*}} overflow<no_unsigned_wrap> : tile<{{.*}}xi32>
// CHECK: shli %{{.*}}, %{{.*}} overflow<no_wrap> : tile<{{.*}}xi32>
nv_tensor_ir.graph @shift_overflow(%a: tensor<128xsi32>, %b: tensor<128xsi32>)
    -> (tensor<128xsi32>, tensor<128xsi32>, tensor<128xsi32>, tensor<128xsi32>) {
  %none = shift_left %a, %b : tensor<128xsi32>
  %nsw = shift_left %a, %b overflow <no_signed_wrap> : tensor<128xsi32>
  %nuw = shift_left %a, %b overflow <no_unsigned_wrap> : tensor<128xsi32>
  %nw = shift_left %a, %b overflow <no_wrap> : tensor<128xsi32>
  results %none, %nsw, %nuw, %nw : tensor<128xsi32>, tensor<128xsi32>, tensor<128xsi32>, tensor<128xsi32>
}

// -----

// CHECK-LABEL: entry @integer_negation
// CHECK: negi %{{.*}} : tile<{{.*}}xi32>
// CHECK: negi %{{.*}} overflow<no_signed_wrap> : tile<{{.*}}xi32>
// CHECK: negi %{{.*}} overflow<no_wrap> : tile<{{.*}}xi32>
nv_tensor_ir.graph @integer_negation(%x: tensor<128xsi32>) -> tensor<128xsi32> {
  %a = negi %x : tensor<128xsi32>
  %b = negi %a overflow <no_signed_wrap> : tensor<128xsi32>
  %c = negi %b overflow <no_wrap> : tensor<128xsi32>
  results %c : tensor<128xsi32>
}
