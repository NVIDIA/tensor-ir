// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)' \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg,convert-tensor-to-linalg)' \
// RUN:   | tensor_ir-opt --split-input-file --mlir-print-local-scope \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.

// CHECK-LABEL: func.func @add_float(
nv_tensor_ir.graph @add_float(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = add %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @sub_float(
nv_tensor_ir.graph @sub_float(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = sub %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.subf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @mul_float(
nv_tensor_ir.graph @mul_float(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = mul %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @div_float(
nv_tensor_ir.graph @div_float(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = div %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.divf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @rem_float(
nv_tensor_ir.graph @rem_float(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = rem %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.remf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @min_float(
nv_tensor_ir.graph @min_float(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = min %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.minimumf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @max_float(
nv_tensor_ir.graph @max_float(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = max %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.maximumf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @atan2_float(
nv_tensor_ir.graph @atan2_float(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = atan2 %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.atan2
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @abs_float(
nv_tensor_ir.graph @abs_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = abs %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.absf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @ceil_float(
nv_tensor_ir.graph @ceil_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = ceil %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.ceil
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cos_float(
nv_tensor_ir.graph @cos_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = cos %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.cos
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @exp_float(
nv_tensor_ir.graph @exp_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = exp %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.exp
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @floor_float(
nv_tensor_ir.graph @floor_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = floor %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.floor
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @log_float(
nv_tensor_ir.graph @log_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = log %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.log
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @neg_float(
nv_tensor_ir.graph @neg_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = neg %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.negf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @rsqrt_float(
nv_tensor_ir.graph @rsqrt_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = rsqrt %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.rsqrt
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @sin_float(
nv_tensor_ir.graph @sin_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = sin %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.sin
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @sqrt_float(
nv_tensor_ir.graph @sqrt_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = sqrt %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.sqrt
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @tan_float(
nv_tensor_ir.graph @tan_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = tan %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.tan
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @erf_float(
nv_tensor_ir.graph @erf_float(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = erf %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: math.erf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @floor_modulo(
nv_tensor_ir.graph @floor_modulo(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = mod %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: %[[Q:.*]] = arith.divf %[[A:.*]], %[[B:.*]] : f32
// CHECK: %[[F:.*]] = math.floor %[[Q]] : f32
// CHECK: %[[P:.*]] = arith.mulf %[[F]], %[[B]] : f32
// CHECK: %[[R:.*]] = arith.subf %[[A]], %[[P]] : f32
// CHECK: linalg.yield %[[R]]

// -----

// CHECK-LABEL: func.func @add_square(
nv_tensor_ir.graph @add_square(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = add_square %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: %[[S:.*]] = arith.mulf %[[B:.*]], %[[B]] : f32
// CHECK: %[[R:.*]] = arith.addf %{{.*}}, %[[S]] : f32
// CHECK: linalg.yield %[[R]]

// -----

// CHECK-LABEL: func.func @reciprocal(
nv_tensor_ir.graph @reciprocal(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = reciprocal %a : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[R:.*]] = arith.divf %[[ONE]], %{{.*}} : f32
// CHECK: linalg.yield %[[R]]

// -----

// CHECK-LABEL: func.func @compare_ueq(
nv_tensor_ir.graph @compare_ueq(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a ueq %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf ueq,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_une(
nv_tensor_ir.graph @compare_une(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a une %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf une,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_ugt(
nv_tensor_ir.graph @compare_ugt(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a ugt %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf ugt,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_uge(
nv_tensor_ir.graph @compare_uge(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a uge %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf uge,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_ult(
nv_tensor_ir.graph @compare_ult(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a ult %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf ult,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_ule(
nv_tensor_ir.graph @compare_ule(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a ule %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf ule,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_oeq(
nv_tensor_ir.graph @compare_oeq(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a oeq %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf oeq,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_one(
nv_tensor_ir.graph @compare_one(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a one %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf one,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_ogt(
nv_tensor_ir.graph @compare_ogt(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a ogt %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf ogt,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_oge(
nv_tensor_ir.graph @compare_oge(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a oge %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf oge,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_olt(
nv_tensor_ir.graph @compare_olt(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a olt %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf olt,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @compare_ole(
nv_tensor_ir.graph @compare_ole(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xi1> {
  %r = cmp %a ole %b : tensor<2xf32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf ole,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @logical_and(
nv_tensor_ir.graph @logical_and(%a: tensor<2xi1>, %b: tensor<2xi1>) -> tensor<2xi1> {
  %r = and %a, %b : tensor<2xi1>
  results %r : tensor<2xi1>
}
// CHECK: arith.andi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @logical_or(
nv_tensor_ir.graph @logical_or(%a: tensor<2xi1>, %b: tensor<2xi1>) -> tensor<2xi1> {
  %r = or %a, %b : tensor<2xi1>
  results %r : tensor<2xi1>
}
// CHECK: arith.ori
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @logical_not(
nv_tensor_ir.graph @logical_not(%a: tensor<2xi1>) -> tensor<2xi1> {
  %r = not %a : tensor<2xi1>
  results %r : tensor<2xi1>
}
// CHECK: arith.xori
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @select(
nv_tensor_ir.graph @select(%p: tensor<2xi1>, %a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %r = binary_select %p, %a, %b : tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.select
// CHECK: linalg.yield

// -----

nv_tensor_ir.graph @pointwise(
    %lhs: tensor<4x8xf32>, %rhs: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %sum = add %lhs, %rhs : tensor<4x8xf32>
  %result = exp %sum : tensor<4x8xf32>
  results %result : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @pointwise(
// CHECK-SAME:      %[[LHS:.+]]: tensor<4x8xf32>, %[[RHS:.+]]: tensor<4x8xf32>
// CHECK:         %[[ADD_EMPTY:.+]] = tensor.empty() : tensor<4x8xf32>
// CHECK:         %[[SUM:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[LHS]], %[[RHS]] : tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-SAME:        outs(%[[ADD_EMPTY]] : tensor<4x8xf32>)
// CHECK:           %[[SCALAR_SUM:.+]] = arith.addf
// CHECK:           linalg.yield %[[SCALAR_SUM]] : f32
// CHECK:         %[[EXP_EMPTY:.+]] = tensor.empty() : tensor<4x8xf32>
// CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[SUM]] : tensor<4x8xf32>)
// CHECK-SAME:        outs(%[[EXP_EMPTY]] : tensor<4x8xf32>)
// CHECK:           %[[SCALAR_EXP:.+]] = math.exp
// CHECK:           linalg.yield %[[SCALAR_EXP]] : f32
// CHECK:         return %[[RESULT]] : tensor<4x8xf32>
