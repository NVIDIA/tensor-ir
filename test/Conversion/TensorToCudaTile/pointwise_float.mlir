// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s
// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// Verify lowering of pointwise operations for a floating point type (F32).
// Exp is additionally checked for f16/bf16 because those follow XLA's
// approximate-f32-then-round-back policy.
// Activations, comparisons and type conversions are tested separately.
// Float pow (fpowf) is covered in pointwise_float_pow.mlir.

// CHECK-LABEL: @test_constant_op
// CHECK: %[[RESULT:.*]] = constant <f32: 0.000000e+00> : tile<{{[0-9]+}}xf32>
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_constant_op() -> tensor<128xf32> {
  %out = constant dense<0.0> : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_splat_op
// CHECK: %[[RESULT:.*]] = constant <f32: 0.000000e+00> : tile<{{[0-9]+}}xf32>
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_splat_op() -> tensor<128xf32> {
  %cst = constant 0.0 : f32
  %out = splat %cst : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_ceil_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = ceil %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_ceil_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = ceil %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_cos_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = cos %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_cos_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = cos %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_exp_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = exp %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_exp_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = exp %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_exp_op_bf16
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[BF16_TILE:tile<[0-9]+xbf16>]]
// CHECK: %[[UP:.*]] = ftof %[[ARG0]] : [[BF16_TILE]] -> [[F32_TILE:tile<[0-9]+xf32>]]
// CHECK: %[[EXP:.*]] = exp %[[UP]] rounding<approx> : [[F32_TILE]]
// CHECK: %[[RESULT:.*]] = ftof %[[EXP]] : [[F32_TILE]] -> [[BF16_TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_exp_op_bf16(%arg0: tensor<128xbf16>) -> tensor<128xbf16> {
  %out = exp %arg0 : tensor<128xbf16>
  results %out : tensor<128xbf16>
}

// -----

// CHECK-LABEL: @test_exp_op_f16
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[F16_TILE:tile<[0-9]+xf16>]]
// CHECK: %[[UP:.*]] = ftof %[[ARG0]] : [[F16_TILE]] -> [[F32_TILE:tile<[0-9]+xf32>]]
// CHECK: %[[EXP:.*]] = exp %[[UP]] rounding<approx> : [[F32_TILE]]
// CHECK: %[[RESULT:.*]] = ftof %[[EXP]] : [[F32_TILE]] -> [[F16_TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_exp_op_f16(%arg0: tensor<128xf16>) -> tensor<128xf16> {
  %out = exp %arg0 : tensor<128xf16>
  results %out : tensor<128xf16>
}

// -----

// CHECK-LABEL: @test_floor_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = floor %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_floor_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = floor %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_log_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = log %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_log_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = log %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_neg_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = negf %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_neg_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = neg %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_rsqrt_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = rsqrt %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_rsqrt_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = rsqrt %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_sin_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = sin %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_sin_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = sin %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_tan_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = tan %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_tan_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = tan %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_tanh_fwd_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = tanh %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_tanh_fwd_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = tanh_fwd %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_sqrt_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = sqrt %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_sqrt_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = sqrt %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_reciprocal_op
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK-DAG: %[[ONE:.*]] = constant <f32: 1.000000e+00> : [[TILE]]
// CHECK: %[[RESULT:.*]] = divf %[[ONE]], %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_reciprocal_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = reciprocal %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_abs_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = absf %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_abs_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = abs %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_add_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = addf %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_add_op(%arg0: tensor<128xf32>,
                                %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = add %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_div_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = divf %[[ARG0]], %[[ARG1]] rounding<full> : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_div_op(%arg0: tensor<128xf32>,
                                %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = div %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_div_op_f32_to_bf16
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[F32_TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[F32_TILE]]
// CHECK: %[[DIV:.*]] = divf %[[ARG0]], %[[ARG1]] rounding<full> : [[F32_TILE]]
// CHECK: %[[RESULT:.*]] = ftof %[[DIV]] : [[F32_TILE]] -> [[BF16_TILE:tile<[0-9]+xbf16>]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_div_op_f32_to_bf16(%arg0: tensor<128xf32>,
                                            %arg1: tensor<128xf32>) -> tensor<128xbf16> {
  %div = div %arg0, %arg1 : tensor<128xf32>
  %out = convert %div : tensor<128xf32> -> tensor<128xbf16>
  results %out : tensor<128xbf16>
}

// -----

// CHECK-LABEL: @test_div_op_f32_to_low_precision_multiple_users
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[F32_TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[F32_TILE]]
// CHECK: %[[DIV:.*]] = divf %[[ARG0]], %[[ARG1]] rounding<full> : [[F32_TILE]]
// CHECK: %[[RESULT:.*]] = ftof %[[DIV]] : [[F32_TILE]] -> [[BF16_TILE:tile<[0-9]+xbf16>]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_div_op_f32_to_low_precision_multiple_users(
    %arg0: tensor<128xf32>,
    %arg1: tensor<128xf32>)
    -> tensor<128xbf16> {
  %div = div %arg0, %arg1 : tensor<128xf32>
  %out0 = convert %div : tensor<128xf32> -> tensor<128xbf16>
  %out1 = convert %div : tensor<128xf32> -> tensor<128xf16>
  results %out0 : tensor<128xbf16>
}

// -----

// CHECK-LABEL: @test_div_op_f16
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[F16_TILE:tile<[0-9]+xf16>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[F16_TILE]]
// CHECK: %[[UP0:.*]] = ftof %[[ARG0]] : [[F16_TILE]] -> [[F32_TILE:tile<[0-9]+xf32>]]
// CHECK: %[[UP1:.*]] = ftof %[[ARG1]] : [[F16_TILE]] -> [[F32_TILE]]
// CHECK: %[[DIV:.*]] = divf %[[UP0]], %[[UP1]] rounding<full> : [[F32_TILE]]
// CHECK: %[[RESULT:.*]] = ftof %[[DIV]] : [[F32_TILE]] -> [[F16_TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_div_op_f16(%arg0: tensor<128xf16>,
                                    %arg1: tensor<128xf16>) -> tensor<128xf16> {
  %out = div %arg0, %arg1 : tensor<128xf16>
  results %out : tensor<128xf16>
}

// -----

// CHECK-LABEL: @test_div_op_bf16
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[BF16_TILE:tile<[0-9]+xbf16>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[BF16_TILE]]
// CHECK: %[[UP0:.*]] = ftof %[[ARG0]] : [[BF16_TILE]] -> [[F32_TILE:tile<[0-9]+xf32>]]
// CHECK: %[[UP1:.*]] = ftof %[[ARG1]] : [[BF16_TILE]] -> [[F32_TILE]]
// CHECK: %[[DIV:.*]] = divf %[[UP0]], %[[UP1]] rounding<full> : [[F32_TILE]]
// CHECK: %[[RESULT:.*]] = ftof %[[DIV]] : [[F32_TILE]] -> [[BF16_TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_div_op_bf16(%arg0: tensor<128xbf16>,
                                     %arg1: tensor<128xbf16>) -> tensor<128xbf16> {
  %out = div %arg0, %arg1 : tensor<128xbf16>
  results %out : tensor<128xbf16>
}

// -----

// CHECK-LABEL: @test_mod_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[DIV:.*]] = divf %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: %[[FLOOR:.*]] = floor %[[DIV]] : [[TILE]]
// CHECK: %[[MUL:.*]] = mulf %[[FLOOR]], %[[ARG1]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = subf %[[ARG0]], %[[MUL]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mod_op(%arg0: tensor<128xf32>,
                                %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = mod %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// `rem` is the truncated counterpart of `mod` and lowers to a single
// `cuda_tile.remf` (no div/floor/mul/sub expansion).
// CHECK-LABEL: @test_rem_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = remf %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_rem_op(%arg0: tensor<128xf32>,
                                %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = rem %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_mul_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = mulf %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mul_op(%arg0: tensor<128xf32>,
                                %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = mul %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_sub_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = subf %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_sub_op(%arg0: tensor<128xf32>,
                                %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = sub %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_min_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = minf %[[ARG0]], %[[ARG1]] propagate_nan : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_min_op(%arg0: tensor<128xf32>,
                                %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = min %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_max_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = maxf %[[ARG0]], %[[ARG1]] propagate_nan : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_max_op(%arg0: tensor<128xf32>,
                                %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = max %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_add_square_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[SQR:.*]] = mulf %[[ARG1]], %[[ARG1]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = addf {{.*}}%[[SQR]]{{.*}} : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_add_square_op(%arg0: tensor<128xf32>,
                                       %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = add_square %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_binary_select_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[PRED:tile<[0-9]+xi1>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG2:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = select %[[ARG0]], %[[ARG1]], %[[ARG2]] : [[PRED]], [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_binary_select_op(%arg0: tensor<128xi1>,
                                          %arg1: tensor<128xf32>,
                                          %arg2: tensor<128xf32>) -> tensor<128xf32> {
  %out = binary_select %arg0, %arg1, %arg2 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_atan2_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = atan2 %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_atan2_op(%arg0: tensor<128xf32>,
                                  %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = atan2 %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}
