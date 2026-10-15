// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)' \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir. --implicit-check-not=unrealized_conversion_cast
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg,convert-tensor-to-linalg)' \
// RUN:   | tensor_ir-opt --split-input-file --mlir-print-local-scope \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir. --implicit-check-not=unrealized_conversion_cast

// CHECK-LABEL: func.func @add_si32(
nv_tensor_ir.graph @add_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xsi32> {
  %r = add %a, %b : tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.addi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @sub_si32(
nv_tensor_ir.graph @sub_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xsi32> {
  %r = sub %a, %b : tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.subi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @mul_si32(
nv_tensor_ir.graph @mul_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xsi32> {
  %r = mul %a, %b : tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.muli
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @div_si32(
nv_tensor_ir.graph @div_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xsi32> {
  %r = div %a, %b : tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.divsi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @rem_si32(
nv_tensor_ir.graph @rem_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xsi32> {
  %r = rem %a, %b : tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.remsi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @min_si32(
nv_tensor_ir.graph @min_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xsi32> {
  %r = min %a, %b : tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.minsi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @max_si32(
nv_tensor_ir.graph @max_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xsi32> {
  %r = max %a, %b : tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.maxsi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_eq_si32(
nv_tensor_ir.graph @cmp_eq_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xi1> {
  %r = cmp %a eq %b : tensor<2xsi32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi eq,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_neq_si32(
nv_tensor_ir.graph @cmp_neq_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xi1> {
  %r = cmp %a neq %b : tensor<2xsi32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi ne,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_gt_si32(
nv_tensor_ir.graph @cmp_gt_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xi1> {
  %r = cmp %a gt %b : tensor<2xsi32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi sgt,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_ge_si32(
nv_tensor_ir.graph @cmp_ge_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xi1> {
  %r = cmp %a ge %b : tensor<2xsi32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi sge,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_lt_si32(
nv_tensor_ir.graph @cmp_lt_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xi1> {
  %r = cmp %a lt %b : tensor<2xsi32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi slt,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_le_si32(
nv_tensor_ir.graph @cmp_le_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xi1> {
  %r = cmp %a le %b : tensor<2xsi32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi sle,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @add_ui32(
nv_tensor_ir.graph @add_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xui32> {
  %r = add %a, %b : tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.addi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @sub_ui32(
nv_tensor_ir.graph @sub_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xui32> {
  %r = sub %a, %b : tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.subi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @mul_ui32(
nv_tensor_ir.graph @mul_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xui32> {
  %r = mul %a, %b : tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.muli
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @div_ui32(
nv_tensor_ir.graph @div_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xui32> {
  %r = div %a, %b : tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.divui
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @rem_ui32(
nv_tensor_ir.graph @rem_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xui32> {
  %r = rem %a, %b : tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.remui
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @min_ui32(
nv_tensor_ir.graph @min_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xui32> {
  %r = min %a, %b : tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.minui
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @max_ui32(
nv_tensor_ir.graph @max_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xui32> {
  %r = max %a, %b : tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.maxui
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_eq_ui32(
nv_tensor_ir.graph @cmp_eq_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xi1> {
  %r = cmp %a eq %b : tensor<2xui32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi eq,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_neq_ui32(
nv_tensor_ir.graph @cmp_neq_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xi1> {
  %r = cmp %a neq %b : tensor<2xui32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi ne,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_gt_ui32(
nv_tensor_ir.graph @cmp_gt_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xi1> {
  %r = cmp %a gt %b : tensor<2xui32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi ugt,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_ge_ui32(
nv_tensor_ir.graph @cmp_ge_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xi1> {
  %r = cmp %a ge %b : tensor<2xui32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi uge,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_lt_ui32(
nv_tensor_ir.graph @cmp_lt_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xi1> {
  %r = cmp %a lt %b : tensor<2xui32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi ult,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cmp_le_ui32(
nv_tensor_ir.graph @cmp_le_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xi1> {
  %r = cmp %a le %b : tensor<2xui32>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi ule,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_si8_si32(
nv_tensor_ir.graph @cast_si8_si32(%a: tensor<2xsi8>) -> tensor<2xsi32> {
  %r = convert %a : tensor<2xsi8> -> tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.extsi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_ui8_ui32(
nv_tensor_ir.graph @cast_ui8_ui32(%a: tensor<2xui8>) -> tensor<2xui32> {
  %r = convert %a : tensor<2xui8> -> tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.extui
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_si64_si8(
nv_tensor_ir.graph @cast_si64_si8(%a: tensor<2xsi64>) -> tensor<2xsi8> {
  %r = convert %a : tensor<2xsi64> -> tensor<2xsi8>
  results %r : tensor<2xsi8>
}
// CHECK: arith.trunci
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_ui64_ui8(
nv_tensor_ir.graph @cast_ui64_ui8(%a: tensor<2xui64>) -> tensor<2xui8> {
  %r = convert %a : tensor<2xui64> -> tensor<2xui8>
  results %r : tensor<2xui8>
}
// CHECK: arith.trunci
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_f32_si32(
nv_tensor_ir.graph @cast_f32_si32(%a: tensor<2xf32>) -> tensor<2xsi32> {
  %r = convert %a : tensor<2xf32> -> tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.fptosi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_f32_ui32(
nv_tensor_ir.graph @cast_f32_ui32(%a: tensor<2xf32>) -> tensor<2xui32> {
  %r = convert %a : tensor<2xf32> -> tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.fptoui
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_si32_f32(
nv_tensor_ir.graph @cast_si32_f32(%a: tensor<2xsi32>) -> tensor<2xf32> {
  %r = convert %a : tensor<2xsi32> -> tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.sitofp
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_ui32_f32(
nv_tensor_ir.graph @cast_ui32_f32(%a: tensor<2xui32>) -> tensor<2xf32> {
  %r = convert %a : tensor<2xui32> -> tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.uitofp
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_si32_i1(
nv_tensor_ir.graph @cast_si32_i1(%a: tensor<2xsi32>) -> tensor<2xi1> {
  %r = convert %a : tensor<2xsi32> -> tensor<2xi1>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi ne,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_ui32_i1(
nv_tensor_ir.graph @cast_ui32_i1(%a: tensor<2xui32>) -> tensor<2xi1> {
  %r = convert %a : tensor<2xui32> -> tensor<2xi1>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpi ne,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @mod_si32(
// CHECK-SAME: tensor<2xi32>
nv_tensor_ir.graph @mod_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xsi32> {
  %r = mod %a, %b : tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.floordivsi
// CHECK: arith.muli
// CHECK: arith.subi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @abs_si32(
// CHECK-SAME: tensor<2xi32>
nv_tensor_ir.graph @abs_si32(%a: tensor<2xsi32>) -> tensor<2xsi32> {
  %r = abs %a : tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: math.absi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @reduce_min_si32(
// CHECK-SAME: tensor<2x3xi32>
nv_tensor_ir.graph @reduce_min_si32(%a: tensor<2x3xsi32>) -> tensor<2x1xsi32> {
  %r = reduce(%a)<dimensions = [1], reduction_mode = <min>> : tensor<2x3xsi32> -> tensor<2x1xsi32>
  results %r : tensor<2x1xsi32>
}
// CHECK: arith.constant 2147483647 : i32
// CHECK: linalg.reduce
// CHECK: arith.minsi

// -----

// CHECK-LABEL: func.func @reduce_max_si32(
// CHECK-SAME: tensor<2x3xi32>
nv_tensor_ir.graph @reduce_max_si32(%a: tensor<2x3xsi32>) -> tensor<2x1xsi32> {
  %r = reduce(%a)<dimensions = [1], reduction_mode = <max>> : tensor<2x3xsi32> -> tensor<2x1xsi32>
  results %r : tensor<2x1xsi32>
}
// CHECK: arith.constant -2147483648 : i32
// CHECK: linalg.reduce
// CHECK: arith.maxsi

// -----

// CHECK-LABEL: func.func @reduce_avg_si32(
// CHECK-SAME: tensor<2x3xi32>
nv_tensor_ir.graph @reduce_avg_si32(%a: tensor<2x3xsi32>) -> tensor<2x1xsi32> {
  %r = reduce(%a)<dimensions = [1], reduction_mode = <avg>> : tensor<2x3xsi32> -> tensor<2x1xsi32>
  results %r : tensor<2x1xsi32>
}
// CHECK: arith.constant 0 : i32
// CHECK: linalg.reduce
// CHECK: arith.divsi

// -----

// CHECK-LABEL: func.func @integer_destinations_si32(
// CHECK-SAME: %[[N:.*]]: index, %[[V:.*]]: i32
nv_tensor_ir.graph @integer_destinations_si32(%n: index, %v: si32) -> (tensor<?xsi32>, tensor<?xsi32>) {
  %fill = splat %v dynamic_dims(%n) : tensor<?xsi32>
  %indices = iota dimension = 0 dynamic_dims(%n) : tensor<?xsi32>
  results %fill, %indices : tensor<?xsi32>, tensor<?xsi32>
}
// CHECK: tensor.empty(%[[N]]) : tensor<?xi32>
// CHECK: linalg.fill ins(%[[V]] : i32)
// CHECK: linalg.index 0
// CHECK: arith.index_cast %{{.*}} : index to i32

// -----

// CHECK-LABEL: func.func @mod_ui32(
// CHECK-SAME: tensor<2xi32>
nv_tensor_ir.graph @mod_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xui32> {
  %r = mod %a, %b : tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.divui
// CHECK: arith.muli
// CHECK: arith.subi
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @abs_ui32(
// CHECK-SAME: tensor<2xi32>
nv_tensor_ir.graph @abs_ui32(%a: tensor<2xui32>) -> tensor<2xui32> {
  %r = abs %a : tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK-NOT: math.absi
// CHECK: return

// -----

// CHECK-LABEL: func.func @reduce_min_ui32(
// CHECK-SAME: tensor<2x3xi32>
nv_tensor_ir.graph @reduce_min_ui32(%a: tensor<2x3xui32>) -> tensor<2x1xui32> {
  %r = reduce(%a)<dimensions = [1], reduction_mode = <min>> : tensor<2x3xui32> -> tensor<2x1xui32>
  results %r : tensor<2x1xui32>
}
// CHECK: arith.constant -1 : i32
// CHECK: linalg.reduce
// CHECK: arith.minui

// -----

// CHECK-LABEL: func.func @reduce_max_ui32(
// CHECK-SAME: tensor<2x3xi32>
nv_tensor_ir.graph @reduce_max_ui32(%a: tensor<2x3xui32>) -> tensor<2x1xui32> {
  %r = reduce(%a)<dimensions = [1], reduction_mode = <max>> : tensor<2x3xui32> -> tensor<2x1xui32>
  results %r : tensor<2x1xui32>
}
// CHECK: arith.constant 0 : i32
// CHECK: linalg.reduce
// CHECK: arith.maxui

// -----

// CHECK-LABEL: func.func @reduce_avg_ui32(
// CHECK-SAME: tensor<2x3xi32>
nv_tensor_ir.graph @reduce_avg_ui32(%a: tensor<2x3xui32>) -> tensor<2x1xui32> {
  %r = reduce(%a)<dimensions = [1], reduction_mode = <avg>> : tensor<2x3xui32> -> tensor<2x1xui32>
  results %r : tensor<2x1xui32>
}
// CHECK: arith.constant 0 : i32
// CHECK: linalg.reduce
// CHECK: arith.divui

// -----

// CHECK-LABEL: func.func @integer_destinations_ui32(
// CHECK-SAME: %[[N:.*]]: index, %[[V:.*]]: i32
nv_tensor_ir.graph @integer_destinations_ui32(%n: index, %v: ui32) -> (tensor<?xui32>, tensor<?xui32>) {
  %fill = splat %v dynamic_dims(%n) : tensor<?xui32>
  %indices = iota dimension = 0 dynamic_dims(%n) : tensor<?xui32>
  results %fill, %indices : tensor<?xui32>, tensor<?xui32>
}
// CHECK: tensor.empty(%[[N]]) : tensor<?xi32>
// CHECK: linalg.fill ins(%[[V]] : i32)
// CHECK: linalg.index 0
// CHECK: arith.index_cast %{{.*}} : index to i32

// -----

// CHECK-LABEL: func.func @integer_constants(
nv_tensor_ir.graph @integer_constants() -> (tensor<2xsi8>, tensor<2xui8>) {
  %s = constant dense<[-1, -128]> : tensor<2xsi8>
  %u = constant dense<[255, 128]> : tensor<2xui8>
  results %s, %u : tensor<2xsi8>, tensor<2xui8>
}
// CHECK-DAG: arith.constant dense<[-1, -128]> : tensor<2xi8>
// CHECK-DAG: arith.constant dense<[-1, -128]> : tensor<2xi8>
// CHECK: return

// -----

// CHECK-LABEL: func.func @unsigned_contraction(
// CHECK-SAME: tensor<2xi8>
nv_tensor_ir.graph @unsigned_contraction(%a: tensor<2xui8>, %b: tensor<2xui8>) -> tensor<ui32> {
  %r = contraction(%a, %b) maps (k) -> (k), (k), () : (tensor<2xui8>, tensor<2xui8>) -> tensor<ui32>
  results %r : tensor<ui32>
}
// CHECK: %[[A:.*]] = arith.extui %{{.*}} : i8 to i33
// CHECK: %[[B:.*]] = arith.extui %{{.*}} : i8 to i33
// CHECK: %[[P:.*]] = arith.muli %[[A]], %[[B]] : i33
// CHECK: %[[ACC:.*]] = arith.extsi %{{.*}} : i32 to i33
// CHECK: %[[S:.*]] = arith.addi %[[ACC]], %[[P]] : i33
// CHECK: arith.maxsi %[[S]], %{{.*}} : i33
// CHECK: %[[CLAMP:.*]] = arith.minsi %{{.*}}, %{{.*}} : i33
// CHECK: arith.trunci %[[CLAMP]] : i33 to i32

// -----

// Existing func boundaries also require conversion, including call results.
// CHECK-LABEL: func.func private @callee(tensor<2xi32>) -> tensor<2xi32>
func.func private @callee(tensor<2xui32>) -> tensor<2xui32>
// CHECK-LABEL: func.func @caller(
// CHECK-SAME: %[[A:.*]]: tensor<2xi32>
func.func @caller(%a: tensor<2xui32>) -> tensor<2xui32> {
  %r = func.call @callee(%a) : (tensor<2xui32>) -> tensor<2xui32>
  return %r : tensor<2xui32>
}
// CHECK: %[[R:.*]] = call @callee(%[[A]]) : (tensor<2xi32>) -> tensor<2xi32>
// CHECK: return %[[R]] : tensor<2xi32>

// -----

// CHECK-LABEL: func.func @boolean_widen(
nv_tensor_ir.graph @boolean_widen(%a: tensor<2xi1>) -> tensor<2xsi32> {
  %r = convert %a : tensor<2xi1> -> tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: arith.extui %{{.*}} : i1 to i32

// -----

// MMA products retain their precision, and the signed accumulator saturates.
// Unsigned inputs must be zero-extended before multiplication.
// CHECK-LABEL: func.func @unsigned_matmul(
nv_tensor_ir.graph @unsigned_matmul(%a: tensor<1x2xui8>, %b: tensor<2x1xui8>) -> tensor<1x1xui32> {
  %r = matmul(%a, %b) : (tensor<1x2xui8>, tensor<2x1xui8>) -> tensor<1x1xui32>
  results %r : tensor<1x1xui32>
}
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: %[[A:.*]] = arith.extui %{{.*}} : i8 to i33
// CHECK: %[[B:.*]] = arith.extui %{{.*}} : i8 to i33
// CHECK: %[[P:.*]] = arith.muli %[[A]], %[[B]] : i33
// CHECK: %[[ACC:.*]] = arith.extsi %{{.*}} : i32 to i33
// CHECK: %[[S:.*]] = arith.addi %[[ACC]], %[[P]] : i33
// CHECK: arith.maxsi %[[S]], %{{.*}} : i33
// CHECK: %[[CLAMP:.*]] = arith.minsi %{{.*}}, %{{.*}} : i33
// CHECK: arith.trunci %[[CLAMP]] : i33 to i32

// -----

// CHECK-LABEL: func.func @signed_batch_matmul(
nv_tensor_ir.graph @signed_batch_matmul(%a: tensor<2x1x2xsi8>, %b: tensor<2x2x1xsi8>) -> tensor<2x1x1xsi32> {
  %r = matmul(%a, %b) : (tensor<2x1x2xsi8>, tensor<2x2x1xsi8>) -> tensor<2x1x1xsi32>
  results %r : tensor<2x1x1xsi32>
}
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK: %[[A:.*]] = arith.extsi %{{.*}} : i8 to i33
// CHECK: %[[B:.*]] = arith.extsi %{{.*}} : i8 to i33
// CHECK: %[[P:.*]] = arith.muli %[[A]], %[[B]] : i33
// CHECK: %[[ACC:.*]] = arith.extsi %{{.*}} : i32 to i33
// CHECK: %[[S:.*]] = arith.addi %[[ACC]], %[[P]] : i33
// CHECK: arith.maxsi %[[S]], %{{.*}} : i33
// CHECK: %[[CLAMP:.*]] = arith.minsi %{{.*}}, %{{.*}} : i33
// CHECK: arith.trunci %[[CLAMP]] : i33 to i32

// -----

// CHECK-LABEL: func.func @integer_reduce_drop_dimensions(
nv_tensor_ir.graph @integer_reduce_drop_dimensions(%a: tensor<2xui32>) -> tensor<ui32> {
  %r = reduce(%a)<dimensions = [0], reduction_mode = <min>> : tensor<2xui32> -> tensor<ui32>
  results %r : tensor<ui32>
}
// CHECK: arith.constant -1 : i32
// CHECK: linalg.reduce
// CHECK: arith.minui
// CHECK-NOT: linalg.broadcast
// CHECK: return %{{.*}} : tensor<i32>

// -----

// CHECK-LABEL: func.func @pow_si32(
nv_tensor_ir.graph @pow_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xsi32> {
  %r = pow %a, %b : (tensor<2xsi32>, tensor<2xsi32>) -> tensor<2xsi32>
  results %r : tensor<2xsi32>
}
// CHECK: math.ipowi {{.*}} : i32

// -----

// CHECK-LABEL: func.func @pow_ui32(
nv_tensor_ir.graph @pow_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xui32> {
  %r = pow %a, %b : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
  results %r : tensor<2xui32>
}
// CHECK: arith.extui {{.*}} : i32 to i64
// CHECK: arith.extui {{.*}} : i32 to i64
// CHECK: math.ipowi {{.*}} : i64
// CHECK: arith.trunci {{.*}} : i64 to i32

// -----

// CHECK-LABEL: func.func @pow_ui8(
nv_tensor_ir.graph @pow_ui8(%a: tensor<2xui8>, %b: tensor<2xui8>) -> tensor<2xui8> {
  %r = pow %a, %b : (tensor<2xui8>, tensor<2xui8>) -> tensor<2xui8>
  results %r : tensor<2xui8>
}
// CHECK: arith.extui {{.*}} : i8 to i16
// CHECK: arith.extui {{.*}} : i8 to i16
// CHECK: math.ipowi {{.*}} : i16
// CHECK: arith.trunci {{.*}} : i16 to i8
