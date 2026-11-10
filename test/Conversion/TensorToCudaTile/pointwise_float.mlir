// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s --check-prefixes=CHECK,AFFINE
// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s --check-prefixes=CHECK,LAYOUT

// Verify lowering of pointwise operations for a floating point type (F32).
// Exp is additionally checked for f16/bf16, which use full f32 evaluation
// before rounding back to the input dtype.
// Activations, comparisons and type conversions are tested separately.
// Float pow (fpowf) is covered in pointwise_float_pow.mlir.

// CHECK-LABEL: @test_fma_op
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[ACC:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = fma %[[LHS]], %[[RHS]], %[[ACC]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_fma_op(
    %lhs: tensor<128xf32>, %rhs: tensor<128xf32>, %acc: tensor<128xf32>)
    -> tensor<128xf32> {
  %out = fma %lhs, %rhs, %acc : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_fma_attrs
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} rounding<positive_inf> flush_to_zero :
nv_tensor_ir.graph @test_fma_attrs(
    %lhs: tensor<128xf32>, %rhs: tensor<128xf32>, %acc: tensor<128xf32>)
    -> tensor<128xf32> {
  %out = fma %lhs, %rhs, %acc <rounding_mode = positive_inf, ftz> : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_fma_op_f16
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf16>]]
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[ACC:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = fma %[[LHS]], %[[RHS]], %[[ACC]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_fma_op_f16(
    %lhs: tensor<128xf16>, %rhs: tensor<128xf16>, %acc: tensor<128xf16>)
    -> tensor<128xf16> {
  %out = fma %lhs, %rhs, %acc : tensor<128xf16>
  results %out : tensor<128xf16>
}

// -----

// CHECK-LABEL: entry @test_constant_op(
// CHECK: %[[RESULT:.*]] = constant <f32: 0.000000e+00> : tile<{{[0-9]+}}xf32>
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_constant_op() -> tensor<128xf32> {
  %out = constant dense<0.0> : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// Compile-time splats become shaped constants in both pipelines.
// CHECK-LABEL: entry @test_splat_op(
// AFFINE: %[[RESULT:.*]] = constant <f32: 0.000000e+00> : tile<{{[0-9]+}}xf32>
// AFFINE: store_view_tko weak %[[RESULT]]
// LAYOUT: %[[RESULT:.*]] = constant <f32: 0.000000e+00> : tile<{{[0-9]+}}xf32>
// LAYOUT-NOT: reshape
// LAYOUT-NOT: broadcast
// LAYOUT: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_splat_op() -> tensor<128xf32> {
  %cst = constant 0.0 : f32
  %out = splat %cst : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_ceil_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = ceil %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_ceil_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = ceil %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_cos_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = cos %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_cos_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = cos %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_cosh_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = cosh %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_cosh_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = cosh %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_exp_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = exp %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_exp_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = exp %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_exp2_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = exp2 %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_exp2_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = exp2 %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_exp2_ftz_op
// CHECK: %[[RESULT:.*]] = exp2 %{{.*}} flush_to_zero :
nv_tensor_ir.graph @test_exp2_ftz_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = exp2 %arg0 <ftz> : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_exp_op_bf16(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[BF16_TILE:tile<[0-9]+xbf16>]]
// CHECK: %[[UP:.*]] = ftof %[[ARG0]] : [[BF16_TILE]] -> [[F32_TILE:tile<[0-9]+xf32>]]
// CHECK: %[[EXP:.*]] = exp %[[UP]] : [[F32_TILE]]
// CHECK: %[[RESULT:.*]] = ftof %[[EXP]] : [[F32_TILE]] -> [[BF16_TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_exp_op_bf16(%arg0: tensor<128xbf16>) -> tensor<128xbf16> {
  %out = exp %arg0 : tensor<128xbf16>
  results %out : tensor<128xbf16>
}

// -----

// CHECK-LABEL: entry @test_exp_op_f16(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[F16_TILE:tile<[0-9]+xf16>]]
// CHECK: %[[UP:.*]] = ftof %[[ARG0]] : [[F16_TILE]] -> [[F32_TILE:tile<[0-9]+xf32>]]
// CHECK: %[[EXP:.*]] = exp %[[UP]] : [[F32_TILE]]
// CHECK: %[[RESULT:.*]] = ftof %[[EXP]] : [[F32_TILE]] -> [[F16_TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_exp_op_f16(%arg0: tensor<128xf16>) -> tensor<128xf16> {
  %out = exp %arg0 : tensor<128xf16>
  results %out : tensor<128xf16>
}

// -----

// CHECK-LABEL: entry @test_floor_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = floor %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_floor_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = floor %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_log_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = log %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_log_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = log %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_log2_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = log2 %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_log2_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = log2 %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_neg_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = negf %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_neg_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = neg %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_rsqrt_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = rsqrt %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_rsqrt_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = rsqrt %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_sin_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = sin %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_sin_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = sin %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @test_sinh_op
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = sinh %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_sinh_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = sinh %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_tan_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = tan %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_tan_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = tan %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_tanh_fwd_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = tanh %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_tanh_fwd_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = tanh_fwd %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_sqrt_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = sqrt %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_sqrt_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = sqrt %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_reciprocal_op(
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK-DAG: %[[ONE:.*]] = constant <f32: 1.000000e+00> : [[TILE]]
// CHECK: %[[RESULT:.*]] = divf %[[ONE]], %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_reciprocal_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = reciprocal %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_abs_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[RESULT:.*]] = absf %[[ARG0]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_abs_op(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %out = abs %arg0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_add_op(
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

// CHECK-LABEL: entry @test_div_op(
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

// CHECK-LABEL: entry @test_div_op_f32_to_bf16(
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

// CHECK-LABEL: entry @test_div_op_f32_to_low_precision_multiple_users(
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

// CHECK-LABEL: entry @test_div_op_f16(
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

// CHECK-LABEL: entry @test_div_op_bf16(
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

// Floored `mod` lowers to `remf` plus a sign-conditional select.
// CHECK-LABEL: entry @test_mod_op(
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK-DAG: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK-DAG: %[[ZERO:.*]] = constant <f32: 0.000000e+00> : [[TILE]]
// CHECK: %[[REM:.*]] = remf %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK-DAG: %[[NONZERO:.*]] = cmpf not_equal ordered %[[REM]], %[[ZERO]] : [[TILE]]
// CHECK-DAG: %[[REM_NEG:.*]] = cmpf less_than ordered %[[REM]], %[[ZERO]] : [[TILE]]
// CHECK-DAG: %[[RHS_NEG:.*]] = cmpf less_than ordered %[[ARG1]], %[[ZERO]] : [[TILE]]
// CHECK: %[[SIGNS_DIFFER:.*]] = xori %[[REM_NEG]], %[[RHS_NEG]]
// CHECK: %[[FIX:.*]] = andi %[[NONZERO]], %[[SIGNS_DIFFER]]
// CHECK: %[[FIXED:.*]] = addf %[[REM]], %[[ARG1]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = select %[[FIX]], %[[FIXED]], %[[REM]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mod_op(%arg0: tensor<128xf32>,
                                %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = mod %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// f16 and bf16 `mod`: same lowering.
// CHECK-LABEL: entry @test_mod_op_f16(
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf16>]]
// CHECK-DAG: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[REM:.*]] = remf %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: %[[FIXED:.*]] = addf %[[REM]], %[[ARG1]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = select %{{.*}}, %[[FIXED]], %[[REM]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mod_op_f16(%arg0: tensor<128xf16>,
                                    %arg1: tensor<128xf16>) -> tensor<128xf16> {
  %out = mod %arg0, %arg1 : tensor<128xf16>
  results %out : tensor<128xf16>
}

// -----

// CHECK-LABEL: entry @test_mod_op_bf16(
// CHECK-DAG: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xbf16>]]
// CHECK-DAG: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[REM:.*]] = remf %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: %[[FIXED:.*]] = addf %[[REM]], %[[ARG1]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = select %{{.*}}, %[[FIXED]], %[[REM]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_mod_op_bf16(%arg0: tensor<128xbf16>,
                                     %arg1: tensor<128xbf16>) -> tensor<128xbf16> {
  %out = mod %arg0, %arg1 : tensor<128xbf16>
  results %out : tensor<128xbf16>
}

// -----

// `rem` lowers to a single `cuda_tile.remf`.
// CHECK-LABEL: entry @test_rem_op(
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

// CHECK-LABEL: entry @test_mul_op(
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

// CHECK-LABEL: entry @test_sub_op(
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

// CHECK-LABEL: entry @test_min_op(
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

// CHECK-LABEL: entry @test_max_op(
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

// CHECK-LABEL: entry @test_add_square_op(
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

// CHECK-LABEL: entry @test_binary_select_op(
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

// CHECK-LABEL: entry @test_atan2_op(
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<[0-9]+xf32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[RESULT:.*]] = atan2 %[[ARG0]], %[[ARG1]] : [[TILE]]
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_atan2_op(%arg0: tensor<128xf32>,
                                  %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %out = atan2 %arg0, %arg1 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @tir_log1p
// CHECK-DAG: %[[one:.*]] = constant <f32: 1.0{{.*}}> : tile<[[tile:.*]]xf32>
// CHECK: %[[u:.*]] = addf %[[x:.*]], %[[one]]  : tile<[[tile]]xf32>
// CHECK: %[[um1:.*]] = subf %[[u]], %[[one]] : tile<[[tile]]xf32>
// CHECK: %[[rhs:.*]] = divf %[[x]], %[[um1]]  : tile<[[tile]]xf32>
// CHECK: %[[lhs:.*]] = log %[[u]] : tile<[[tile]]xf32>
// CHECK: %[[approx:.*]] = mulf %[[lhs]], %[[rhs]]  : tile<[[tile]]xf32>
// CHECK: %[[cmp:.*]] = cmpf equal ordered %[[u]], %[[one]] : tile<[[tile]]xf32> -> tile<[[tile]]xi1>
// CHECK: %[[sel:.*]] = select %[[cmp]], %[[x]], %[[approx]] : tile<[[tile]]xi1>, tile<[[tile]]xf32>
// Overflow: 1 + x == inf must return inf, since x / (u - 1) would be NaN.
// CHECK: %[[is_inf:.*]] = cmpf equal ordered %[[u]], %{{.*}} : tile<[[tile]]xf32> -> tile<[[tile]]xi1>
// CHECK: select %[[is_inf]], %[[u]], %[[sel]] : tile<[[tile]]xi1>, tile<[[tile]]xf32>
nv_tensor_ir.graph @tir_log1p(
    %input: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %result = log1p %input : tensor<32x32xf32>
  results %result : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: @tir_expm1
// CHECK-DAG: %[[one:.*]] = constant <f32: 1.0{{.*}}> : tile<[[tile:.*]]xf32>
// CHECK-DAG: %[[neg_one:.*]] = constant <f32: -1.0{{.*}}> : tile<[[tile]]xf32>
// CHECK: %[[e:.*]] = exp %[[x:.*]] : tile<[[tile]]xf32>
// CHECK: %[[l:.*]] = log %[[e]] : tile<[[tile]]xf32>
// CHECK: %[[rhs:.*]] = divf %[[x]], %[[l]] : tile<[[tile]]xf32>
// CHECK: %[[em1:.*]] = subf %[[e]], %[[one]] : tile<[[tile]]xf32>
// CHECK: %[[approx:.*]] = mulf %[[em1]], %[[rhs]] : tile<[[tile]]xf32>
// CHECK: %[[is_neg_one:.*]] = cmpf equal ordered %[[em1]], %[[neg_one]] : tile<[[tile]]xf32> -> tile<[[tile]]xi1>
// CHECK: %[[sel_neg:.*]] = select %[[is_neg_one]], %[[neg_one]], %[[approx]] : tile<[[tile]]xi1>, tile<[[tile]]xf32>
// CHECK: %[[is_one:.*]] = cmpf equal ordered %[[e]], %[[one]] : tile<[[tile]]xf32> -> tile<[[tile]]xi1>
// CHECK: %[[sel_one:.*]] = select %[[is_one]], %[[x]], %[[sel_neg]] : tile<[[tile]]xi1>, tile<[[tile]]xf32>
// Overflow: exp(x) == inf must return inf, since x / log(inf) would be NaN.
// CHECK: %[[is_inf:.*]] = cmpf equal ordered %[[e]], %{{.*}} : tile<[[tile]]xf32> -> tile<[[tile]]xi1>
// CHECK: select %[[is_inf]], %[[e]], %[[sel_one]] : tile<[[tile]]xi1>, tile<[[tile]]xf32>
nv_tensor_ir.graph @tir_expm1(
    %input: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %result = expm1 %input : tensor<32x32xf32>
  results %result : tensor<32x32xf32>
}
