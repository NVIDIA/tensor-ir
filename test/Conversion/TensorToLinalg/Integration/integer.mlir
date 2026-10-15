// RUN: tensor_ir-opt %s --tir-linalg-pipeline \
// RUN:   | mlir-runner -e main -entry-point-result=void


// Exercise signedness at function calls and constants as well as arithmetic.
// Expected values are exact in f64, including the complete ui32 range.

func.func @div_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.div %a, %b : tensor<2xsi32>
  %f = nv_tensor_ir.convert %r : tensor<2xsi32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @rem_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.rem %a, %b : tensor<2xsi32>
  %f = nv_tensor_ir.convert %r : tensor<2xsi32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @mod_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.mod %a, %b : tensor<2xsi32>
  %f = nv_tensor_ir.convert %r : tensor<2xsi32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @min_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.min %a, %b : tensor<2xsi32>
  %f = nv_tensor_ir.convert %r : tensor<2xsi32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @max_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.max %a, %b : tensor<2xsi32>
  %f = nv_tensor_ir.convert %r : tensor<2xsi32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @abs_si32(%a: tensor<2xsi32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.abs %a : tensor<2xsi32>
  %f = nv_tensor_ir.convert %r : tensor<2xsi32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @cmp_si32(%a: tensor<2xsi32>, %b: tensor<2xsi32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.cmp %a gt %b : tensor<2xsi32>
  %f = nv_tensor_ir.convert %r : tensor<2xi1> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @div_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.div %a, %b : tensor<2xui32>
  %f = nv_tensor_ir.convert %r : tensor<2xui32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @rem_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.rem %a, %b : tensor<2xui32>
  %f = nv_tensor_ir.convert %r : tensor<2xui32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @mod_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.mod %a, %b : tensor<2xui32>
  %f = nv_tensor_ir.convert %r : tensor<2xui32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @min_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.min %a, %b : tensor<2xui32>
  %f = nv_tensor_ir.convert %r : tensor<2xui32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @max_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.max %a, %b : tensor<2xui32>
  %f = nv_tensor_ir.convert %r : tensor<2xui32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @abs_ui32(%a: tensor<2xui32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.abs %a : tensor<2xui32>
  %f = nv_tensor_ir.convert %r : tensor<2xui32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @cmp_ui32(%a: tensor<2xui32>, %b: tensor<2xui32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.cmp %a gt %b : tensor<2xui32>
  %f = nv_tensor_ir.convert %r : tensor<2xi1> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @cast_si8_si32(%a: tensor<2xsi8>) -> tensor<2xf64> {
  %r = nv_tensor_ir.convert %a : tensor<2xsi8> -> tensor<2xsi32>
  %f = nv_tensor_ir.convert %r : tensor<2xsi32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @cast_ui8_ui32(%a: tensor<2xui8>) -> tensor<2xf64> {
  %r = nv_tensor_ir.convert %a : tensor<2xui8> -> tensor<2xui32>
  %f = nv_tensor_ir.convert %r : tensor<2xui32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @cast_si32_si8(%a: tensor<2xsi32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.convert %a : tensor<2xsi32> -> tensor<2xsi8>
  %f = nv_tensor_ir.convert %r : tensor<2xsi8> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @cast_ui32_ui8(%a: tensor<2xui32>) -> tensor<2xf64> {
  %r = nv_tensor_ir.convert %a : tensor<2xui32> -> tensor<2xui8>
  %f = nv_tensor_ir.convert %r : tensor<2xui8> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @cast_i1_si32(%a: tensor<2xi1>) -> tensor<2xf64> {
  %r = nv_tensor_ir.convert %a : tensor<2xi1> -> tensor<2xsi32>
  %f = nv_tensor_ir.convert %r : tensor<2xsi32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @cast_f64_ui32(%a: tensor<2xf64>) -> tensor<2xf64> {
  %r = nv_tensor_ir.convert %a : tensor<2xf64> -> tensor<2xui32>
  %f = nv_tensor_ir.convert %r : tensor<2xui32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @cast_f64_si32(%a: tensor<2xf64>) -> tensor<2xf64> {
  %r = nv_tensor_ir.convert %a : tensor<2xf64> -> tensor<2xsi32>
  %f = nv_tensor_ir.convert %r : tensor<2xsi32> -> tensor<2xf64>
  return %f : tensor<2xf64>
}

func.func @reduce_min_si32(%a: tensor<2xsi32>) -> tensor<f64> {
  %r = nv_tensor_ir.reduce(%a)<dimensions = [0], reduction_mode = <min>> : tensor<2xsi32> -> tensor<si32>
  %f = nv_tensor_ir.convert %r : tensor<si32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @reduce_max_si32(%a: tensor<2xsi32>) -> tensor<f64> {
  %r = nv_tensor_ir.reduce(%a)<dimensions = [0], reduction_mode = <max>> : tensor<2xsi32> -> tensor<si32>
  %f = nv_tensor_ir.convert %r : tensor<si32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @reduce_avg_si32(%a: tensor<2xsi32>) -> tensor<f64> {
  %r = nv_tensor_ir.reduce(%a)<dimensions = [0], reduction_mode = <avg>> : tensor<2xsi32> -> tensor<si32>
  %f = nv_tensor_ir.convert %r : tensor<si32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @reduce_min_ui32(%a: tensor<2xui32>) -> tensor<f64> {
  %r = nv_tensor_ir.reduce(%a)<dimensions = [0], reduction_mode = <min>> : tensor<2xui32> -> tensor<ui32>
  %f = nv_tensor_ir.convert %r : tensor<ui32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @reduce_max_ui32(%a: tensor<2xui32>) -> tensor<f64> {
  %r = nv_tensor_ir.reduce(%a)<dimensions = [0], reduction_mode = <max>> : tensor<2xui32> -> tensor<ui32>
  %f = nv_tensor_ir.convert %r : tensor<ui32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @reduce_avg_ui32(%a: tensor<2xui32>) -> tensor<f64> {
  %r = nv_tensor_ir.reduce(%a)<dimensions = [0], reduction_mode = <avg>> : tensor<2xui32> -> tensor<ui32>
  %f = nv_tensor_ir.convert %r : tensor<ui32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @contraction(%a: tensor<2xui8>, %b: tensor<2xui8>) -> tensor<f64> {
  %r = nv_tensor_ir.contraction(%a, %b) maps (k) -> (k), (k), () : (tensor<2xui8>, tensor<2xui8>) -> tensor<ui32>
  %flat = nv_tensor_ir.reshape %r : tensor<ui32> -> tensor<ui32>
  %f = nv_tensor_ir.convert %flat : tensor<ui32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @matmul(%a: tensor<1x2xui8>, %b: tensor<2x1xui8>) -> tensor<f64> {
  %r = nv_tensor_ir.matmul(%a, %b) : (tensor<1x2xui8>, tensor<2x1xui8>) -> tensor<1x1xui32>
  %flat = nv_tensor_ir.reshape %r : tensor<1x1xui32> -> tensor<ui32>
  %f = nv_tensor_ir.convert %flat : tensor<ui32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @batch_matmul(%a: tensor<1x1x2xui8>, %b: tensor<1x2x1xui8>) -> tensor<f64> {
  %r = nv_tensor_ir.matmul(%a, %b) : (tensor<1x1x2xui8>, tensor<1x2x1xui8>) -> tensor<1x1x1xui32>
  %flat = nv_tensor_ir.reshape %r : tensor<1x1x1xui32> -> tensor<ui32>
  %f = nv_tensor_ir.convert %flat : tensor<ui32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @pow_si32(%a: tensor<si32>, %b: tensor<si32>) -> tensor<f64> {
  %r = nv_tensor_ir.pow %a, %b : (tensor<si32>, tensor<si32>) -> tensor<si32>
  %f = nv_tensor_ir.convert %r : tensor<si32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @pow_ui32(%a: tensor<ui32>, %b: tensor<ui32>) -> tensor<f64> {
  %r = nv_tensor_ir.pow %a, %b : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
  %f = nv_tensor_ir.convert %r : tensor<ui32> -> tensor<f64>
  return %f : tensor<f64>
}

func.func @main() {
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %a_si32 = nv_tensor_ir.constant dense<[-5, 5]> : tensor<2xsi32>
  %b_si32 = nv_tensor_ir.constant dense<[3, -3]> : tensor<2xsi32>
  %r0 = func.call @div_si32(%a_si32, %b_si32) : (tensor<2xsi32>, tensor<2xsi32>) -> tensor<2xf64>
  %v0_0 = tensor.extract %r0[%i0] : tensor<2xf64>
  %e0_0 = arith.constant -1.0 : f64
  %ok0_0 = arith.cmpf oeq, %v0_0, %e0_0 : f64
  cf.assert %ok0_0, "div_si32 element 0"
  %v0_1 = tensor.extract %r0[%i1] : tensor<2xf64>
  %e0_1 = arith.constant -1.0 : f64
  %ok0_1 = arith.cmpf oeq, %v0_1, %e0_1 : f64
  cf.assert %ok0_1, "div_si32 element 1"
  %r1 = func.call @rem_si32(%a_si32, %b_si32) : (tensor<2xsi32>, tensor<2xsi32>) -> tensor<2xf64>
  %v1_0 = tensor.extract %r1[%i0] : tensor<2xf64>
  %e1_0 = arith.constant -2.0 : f64
  %ok1_0 = arith.cmpf oeq, %v1_0, %e1_0 : f64
  cf.assert %ok1_0, "rem_si32 element 0"
  %v1_1 = tensor.extract %r1[%i1] : tensor<2xf64>
  %e1_1 = arith.constant 2.0 : f64
  %ok1_1 = arith.cmpf oeq, %v1_1, %e1_1 : f64
  cf.assert %ok1_1, "rem_si32 element 1"
  %r2 = func.call @mod_si32(%a_si32, %b_si32) : (tensor<2xsi32>, tensor<2xsi32>) -> tensor<2xf64>
  %v2_0 = tensor.extract %r2[%i0] : tensor<2xf64>
  %e2_0 = arith.constant 1.0 : f64
  %ok2_0 = arith.cmpf oeq, %v2_0, %e2_0 : f64
  cf.assert %ok2_0, "mod_si32 element 0"
  %v2_1 = tensor.extract %r2[%i1] : tensor<2xf64>
  %e2_1 = arith.constant -1.0 : f64
  %ok2_1 = arith.cmpf oeq, %v2_1, %e2_1 : f64
  cf.assert %ok2_1, "mod_si32 element 1"
  %r3 = func.call @min_si32(%a_si32, %b_si32) : (tensor<2xsi32>, tensor<2xsi32>) -> tensor<2xf64>
  %v3_0 = tensor.extract %r3[%i0] : tensor<2xf64>
  %e3_0 = arith.constant -5.0 : f64
  %ok3_0 = arith.cmpf oeq, %v3_0, %e3_0 : f64
  cf.assert %ok3_0, "min_si32 element 0"
  %v3_1 = tensor.extract %r3[%i1] : tensor<2xf64>
  %e3_1 = arith.constant -3.0 : f64
  %ok3_1 = arith.cmpf oeq, %v3_1, %e3_1 : f64
  cf.assert %ok3_1, "min_si32 element 1"
  %r4 = func.call @max_si32(%a_si32, %b_si32) : (tensor<2xsi32>, tensor<2xsi32>) -> tensor<2xf64>
  %v4_0 = tensor.extract %r4[%i0] : tensor<2xf64>
  %e4_0 = arith.constant 3.0 : f64
  %ok4_0 = arith.cmpf oeq, %v4_0, %e4_0 : f64
  cf.assert %ok4_0, "max_si32 element 0"
  %v4_1 = tensor.extract %r4[%i1] : tensor<2xf64>
  %e4_1 = arith.constant 5.0 : f64
  %ok4_1 = arith.cmpf oeq, %v4_1, %e4_1 : f64
  cf.assert %ok4_1, "max_si32 element 1"
  %r5 = func.call @abs_si32(%a_si32) : (tensor<2xsi32>) -> tensor<2xf64>
  %v5_0 = tensor.extract %r5[%i0] : tensor<2xf64>
  %e5_0 = arith.constant 5.0 : f64
  %ok5_0 = arith.cmpf oeq, %v5_0, %e5_0 : f64
  cf.assert %ok5_0, "abs_si32 element 0"
  %v5_1 = tensor.extract %r5[%i1] : tensor<2xf64>
  %e5_1 = arith.constant 5.0 : f64
  %ok5_1 = arith.cmpf oeq, %v5_1, %e5_1 : f64
  cf.assert %ok5_1, "abs_si32 element 1"
  %r6 = func.call @cmp_si32(%a_si32, %b_si32) : (tensor<2xsi32>, tensor<2xsi32>) -> tensor<2xf64>
  %v6_0 = tensor.extract %r6[%i0] : tensor<2xf64>
  %e6_0 = arith.constant 0.0 : f64
  %ok6_0 = arith.cmpf oeq, %v6_0, %e6_0 : f64
  cf.assert %ok6_0, "cmp_si32 element 0"
  %v6_1 = tensor.extract %r6[%i1] : tensor<2xf64>
  %e6_1 = arith.constant 1.0 : f64
  %ok6_1 = arith.cmpf oeq, %v6_1, %e6_1 : f64
  cf.assert %ok6_1, "cmp_si32 element 1"
  %a_ui32 = nv_tensor_ir.constant dense<[4294967295, 2147483648]> : tensor<2xui32>
  %b_ui32 = nv_tensor_ir.constant dense<[2, 3]> : tensor<2xui32>
  %r7 = func.call @div_ui32(%a_ui32, %b_ui32) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xf64>
  %v7_0 = tensor.extract %r7[%i0] : tensor<2xf64>
  %e7_0 = arith.constant 2147483647.0 : f64
  %ok7_0 = arith.cmpf oeq, %v7_0, %e7_0 : f64
  cf.assert %ok7_0, "div_ui32 element 0"
  %v7_1 = tensor.extract %r7[%i1] : tensor<2xf64>
  %e7_1 = arith.constant 715827882.0 : f64
  %ok7_1 = arith.cmpf oeq, %v7_1, %e7_1 : f64
  cf.assert %ok7_1, "div_ui32 element 1"
  %r8 = func.call @rem_ui32(%a_ui32, %b_ui32) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xf64>
  %v8_0 = tensor.extract %r8[%i0] : tensor<2xf64>
  %e8_0 = arith.constant 1.0 : f64
  %ok8_0 = arith.cmpf oeq, %v8_0, %e8_0 : f64
  cf.assert %ok8_0, "rem_ui32 element 0"
  %v8_1 = tensor.extract %r8[%i1] : tensor<2xf64>
  %e8_1 = arith.constant 2.0 : f64
  %ok8_1 = arith.cmpf oeq, %v8_1, %e8_1 : f64
  cf.assert %ok8_1, "rem_ui32 element 1"
  %r9 = func.call @mod_ui32(%a_ui32, %b_ui32) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xf64>
  %v9_0 = tensor.extract %r9[%i0] : tensor<2xf64>
  %e9_0 = arith.constant 1.0 : f64
  %ok9_0 = arith.cmpf oeq, %v9_0, %e9_0 : f64
  cf.assert %ok9_0, "mod_ui32 element 0"
  %v9_1 = tensor.extract %r9[%i1] : tensor<2xf64>
  %e9_1 = arith.constant 2.0 : f64
  %ok9_1 = arith.cmpf oeq, %v9_1, %e9_1 : f64
  cf.assert %ok9_1, "mod_ui32 element 1"
  %r10 = func.call @min_ui32(%a_ui32, %b_ui32) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xf64>
  %v10_0 = tensor.extract %r10[%i0] : tensor<2xf64>
  %e10_0 = arith.constant 2.0 : f64
  %ok10_0 = arith.cmpf oeq, %v10_0, %e10_0 : f64
  cf.assert %ok10_0, "min_ui32 element 0"
  %v10_1 = tensor.extract %r10[%i1] : tensor<2xf64>
  %e10_1 = arith.constant 3.0 : f64
  %ok10_1 = arith.cmpf oeq, %v10_1, %e10_1 : f64
  cf.assert %ok10_1, "min_ui32 element 1"
  %r11 = func.call @max_ui32(%a_ui32, %b_ui32) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xf64>
  %v11_0 = tensor.extract %r11[%i0] : tensor<2xf64>
  %e11_0 = arith.constant 4294967295.0 : f64
  %ok11_0 = arith.cmpf oeq, %v11_0, %e11_0 : f64
  cf.assert %ok11_0, "max_ui32 element 0"
  %v11_1 = tensor.extract %r11[%i1] : tensor<2xf64>
  %e11_1 = arith.constant 2147483648.0 : f64
  %ok11_1 = arith.cmpf oeq, %v11_1, %e11_1 : f64
  cf.assert %ok11_1, "max_ui32 element 1"
  %r12 = func.call @abs_ui32(%a_ui32) : (tensor<2xui32>) -> tensor<2xf64>
  %v12_0 = tensor.extract %r12[%i0] : tensor<2xf64>
  %e12_0 = arith.constant 4294967295.0 : f64
  %ok12_0 = arith.cmpf oeq, %v12_0, %e12_0 : f64
  cf.assert %ok12_0, "abs_ui32 element 0"
  %v12_1 = tensor.extract %r12[%i1] : tensor<2xf64>
  %e12_1 = arith.constant 2147483648.0 : f64
  %ok12_1 = arith.cmpf oeq, %v12_1, %e12_1 : f64
  cf.assert %ok12_1, "abs_ui32 element 1"
  %r13 = func.call @cmp_ui32(%a_ui32, %b_ui32) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xf64>
  %v13_0 = tensor.extract %r13[%i0] : tensor<2xf64>
  %e13_0 = arith.constant 1.0 : f64
  %ok13_0 = arith.cmpf oeq, %v13_0, %e13_0 : f64
  cf.assert %ok13_0, "cmp_ui32 element 0"
  %v13_1 = tensor.extract %r13[%i1] : tensor<2xf64>
  %e13_1 = arith.constant 1.0 : f64
  %ok13_1 = arith.cmpf oeq, %v13_1, %e13_1 : f64
  cf.assert %ok13_1, "cmp_ui32 element 1"
  %cast_si8_si32_input = nv_tensor_ir.constant dense<[-1, -128]> : tensor<2xsi8>
  %r14 = func.call @cast_si8_si32(%cast_si8_si32_input) : (tensor<2xsi8>) -> tensor<2xf64>
  %v14_0 = tensor.extract %r14[%i0] : tensor<2xf64>
  %e14_0 = arith.constant -1.0 : f64
  %ok14_0 = arith.cmpf oeq, %v14_0, %e14_0 : f64
  cf.assert %ok14_0, "cast_si8_si32 element 0"
  %v14_1 = tensor.extract %r14[%i1] : tensor<2xf64>
  %e14_1 = arith.constant -128.0 : f64
  %ok14_1 = arith.cmpf oeq, %v14_1, %e14_1 : f64
  cf.assert %ok14_1, "cast_si8_si32 element 1"
  %cast_ui8_ui32_input = nv_tensor_ir.constant dense<[255, 128]> : tensor<2xui8>
  %r15 = func.call @cast_ui8_ui32(%cast_ui8_ui32_input) : (tensor<2xui8>) -> tensor<2xf64>
  %v15_0 = tensor.extract %r15[%i0] : tensor<2xf64>
  %e15_0 = arith.constant 255.0 : f64
  %ok15_0 = arith.cmpf oeq, %v15_0, %e15_0 : f64
  cf.assert %ok15_0, "cast_ui8_ui32 element 0"
  %v15_1 = tensor.extract %r15[%i1] : tensor<2xf64>
  %e15_1 = arith.constant 128.0 : f64
  %ok15_1 = arith.cmpf oeq, %v15_1, %e15_1 : f64
  cf.assert %ok15_1, "cast_ui8_ui32 element 1"
  %cast_si32_si8_input = nv_tensor_ir.constant dense<[257, -129]> : tensor<2xsi32>
  %r16 = func.call @cast_si32_si8(%cast_si32_si8_input) : (tensor<2xsi32>) -> tensor<2xf64>
  %v16_0 = tensor.extract %r16[%i0] : tensor<2xf64>
  %e16_0 = arith.constant 1.0 : f64
  %ok16_0 = arith.cmpf oeq, %v16_0, %e16_0 : f64
  cf.assert %ok16_0, "cast_si32_si8 element 0"
  %v16_1 = tensor.extract %r16[%i1] : tensor<2xf64>
  %e16_1 = arith.constant 127.0 : f64
  %ok16_1 = arith.cmpf oeq, %v16_1, %e16_1 : f64
  cf.assert %ok16_1, "cast_si32_si8 element 1"
  %cast_ui32_ui8_input = nv_tensor_ir.constant dense<[257, 511]> : tensor<2xui32>
  %r17 = func.call @cast_ui32_ui8(%cast_ui32_ui8_input) : (tensor<2xui32>) -> tensor<2xf64>
  %v17_0 = tensor.extract %r17[%i0] : tensor<2xf64>
  %e17_0 = arith.constant 1.0 : f64
  %ok17_0 = arith.cmpf oeq, %v17_0, %e17_0 : f64
  cf.assert %ok17_0, "cast_ui32_ui8 element 0"
  %v17_1 = tensor.extract %r17[%i1] : tensor<2xf64>
  %e17_1 = arith.constant 255.0 : f64
  %ok17_1 = arith.cmpf oeq, %v17_1, %e17_1 : f64
  cf.assert %ok17_1, "cast_ui32_ui8 element 1"
  %cast_i1_si32_input = arith.constant dense<[true, false]> : tensor<2xi1>
  %r18 = func.call @cast_i1_si32(%cast_i1_si32_input) : (tensor<2xi1>) -> tensor<2xf64>
  %v18_0 = tensor.extract %r18[%i0] : tensor<2xf64>
  %e18_0 = arith.constant 1.0 : f64
  %ok18_0 = arith.cmpf oeq, %v18_0, %e18_0 : f64
  cf.assert %ok18_0, "cast_i1_si32 element 0"
  %v18_1 = tensor.extract %r18[%i1] : tensor<2xf64>
  %e18_1 = arith.constant 0.0 : f64
  %ok18_1 = arith.cmpf oeq, %v18_1, %e18_1 : f64
  cf.assert %ok18_1, "cast_i1_si32 element 1"
  %cast_f64_ui32_input = arith.constant dense<[4294967295.0, 2147483648.0]> : tensor<2xf64>
  %r19 = func.call @cast_f64_ui32(%cast_f64_ui32_input) : (tensor<2xf64>) -> tensor<2xf64>
  %v19_0 = tensor.extract %r19[%i0] : tensor<2xf64>
  %e19_0 = arith.constant 4294967295.0 : f64
  %ok19_0 = arith.cmpf oeq, %v19_0, %e19_0 : f64
  cf.assert %ok19_0, "cast_f64_ui32 element 0"
  %v19_1 = tensor.extract %r19[%i1] : tensor<2xf64>
  %e19_1 = arith.constant 2147483648.0 : f64
  %ok19_1 = arith.cmpf oeq, %v19_1, %e19_1 : f64
  cf.assert %ok19_1, "cast_f64_ui32 element 1"
  %cast_f64_si32_input = arith.constant dense<[-5.9, 5.9]> : tensor<2xf64>
  %r20 = func.call @cast_f64_si32(%cast_f64_si32_input) : (tensor<2xf64>) -> tensor<2xf64>
  %v20_0 = tensor.extract %r20[%i0] : tensor<2xf64>
  %e20_0 = arith.constant -5.0 : f64
  %ok20_0 = arith.cmpf oeq, %v20_0, %e20_0 : f64
  cf.assert %ok20_0, "cast_f64_si32 element 0"
  %v20_1 = tensor.extract %r20[%i1] : tensor<2xf64>
  %e20_1 = arith.constant 5.0 : f64
  %ok20_1 = arith.cmpf oeq, %v20_1, %e20_1 : f64
  cf.assert %ok20_1, "cast_f64_si32 element 1"
  %reduce_si32_input = nv_tensor_ir.constant dense<[-5, 2]> : tensor<2xsi32>
  %r21 = func.call @reduce_min_si32(%reduce_si32_input) : (tensor<2xsi32>) -> tensor<f64>
  %v21_0 = tensor.extract %r21[] : tensor<f64>
  %e21_0 = arith.constant -5.0 : f64
  %ok21_0 = arith.cmpf oeq, %v21_0, %e21_0 : f64
  cf.assert %ok21_0, "reduce_min_si32 element 0"
  %r22 = func.call @reduce_max_si32(%reduce_si32_input) : (tensor<2xsi32>) -> tensor<f64>
  %v22_0 = tensor.extract %r22[] : tensor<f64>
  %e22_0 = arith.constant 2.0 : f64
  %ok22_0 = arith.cmpf oeq, %v22_0, %e22_0 : f64
  cf.assert %ok22_0, "reduce_max_si32 element 0"
  %r23 = func.call @reduce_avg_si32(%reduce_si32_input) : (tensor<2xsi32>) -> tensor<f64>
  %v23_0 = tensor.extract %r23[] : tensor<f64>
  %e23_0 = arith.constant -1.0 : f64
  %ok23_0 = arith.cmpf oeq, %v23_0, %e23_0 : f64
  cf.assert %ok23_0, "reduce_avg_si32 element 0"
  %reduce_ui32_input = nv_tensor_ir.constant dense<[2147483648, 3]> : tensor<2xui32>
  %r24 = func.call @reduce_min_ui32(%reduce_ui32_input) : (tensor<2xui32>) -> tensor<f64>
  %v24_0 = tensor.extract %r24[] : tensor<f64>
  %e24_0 = arith.constant 3.0 : f64
  %ok24_0 = arith.cmpf oeq, %v24_0, %e24_0 : f64
  cf.assert %ok24_0, "reduce_min_ui32 element 0"
  %r25 = func.call @reduce_max_ui32(%reduce_ui32_input) : (tensor<2xui32>) -> tensor<f64>
  %v25_0 = tensor.extract %r25[] : tensor<f64>
  %e25_0 = arith.constant 2147483648.0 : f64
  %ok25_0 = arith.cmpf oeq, %v25_0, %e25_0 : f64
  cf.assert %ok25_0, "reduce_max_ui32 element 0"
  %r26 = func.call @reduce_avg_ui32(%reduce_ui32_input) : (tensor<2xui32>) -> tensor<f64>
  %v26_0 = tensor.extract %r26[] : tensor<f64>
  %e26_0 = arith.constant 1073741825.0 : f64
  %ok26_0 = arith.cmpf oeq, %v26_0, %e26_0 : f64
  cf.assert %ok26_0, "reduce_avg_ui32 element 0"
  %contraction_a = nv_tensor_ir.constant dense<[255, 2]> : tensor<2xui8>
  %contraction_b = nv_tensor_ir.constant dense<[2, 255]> : tensor<2xui8>
  %r27 = func.call @contraction(%contraction_a, %contraction_b) : (tensor<2xui8>, tensor<2xui8>) -> tensor<f64>
  %v27_0 = tensor.extract %r27[] : tensor<f64>
  %e27_0 = arith.constant 1020.0 : f64
  %ok27_0 = arith.cmpf oeq, %v27_0, %e27_0 : f64
  cf.assert %ok27_0, "contraction element 0"
  %matmul_a = nv_tensor_ir.constant dense<[[255, 2]]> : tensor<1x2xui8>
  %matmul_b = nv_tensor_ir.constant dense<[[2], [255]]> : tensor<2x1xui8>
  %r28 = func.call @matmul(%matmul_a, %matmul_b) : (tensor<1x2xui8>, tensor<2x1xui8>) -> tensor<f64>
  %v28_0 = tensor.extract %r28[] : tensor<f64>
  %e28_0 = arith.constant 1020.0 : f64
  %ok28_0 = arith.cmpf oeq, %v28_0, %e28_0 : f64
  cf.assert %ok28_0, "matmul element 0"
  %batch_matmul_a = nv_tensor_ir.constant dense<[[[255, 2]]]> : tensor<1x1x2xui8>
  %batch_matmul_b = nv_tensor_ir.constant dense<[[[2], [255]]]> : tensor<1x2x1xui8>
  %r29 = func.call @batch_matmul(%batch_matmul_a, %batch_matmul_b) : (tensor<1x1x2xui8>, tensor<1x2x1xui8>) -> tensor<f64>
  %v29_0 = tensor.extract %r29[] : tensor<f64>
  %e29_0 = arith.constant 1020.0 : f64
  %ok29_0 = arith.cmpf oeq, %v29_0, %e29_0 : f64
  cf.assert %ok29_0, "batch_matmul element 0"
  %pow_base_si32 = nv_tensor_ir.constant dense<-2> : tensor<si32>
  %pow_exp_si32 = nv_tensor_ir.constant dense<3> : tensor<si32>
  %pow_si32 = func.call @pow_si32(%pow_base_si32, %pow_exp_si32) : (tensor<si32>, tensor<si32>) -> tensor<f64>
  %pow_value_si32 = tensor.extract %pow_si32[] : tensor<f64>
  %pow_expected_si32 = arith.constant -8.0 : f64
  %pow_ok_si32 = arith.cmpf oeq, %pow_value_si32, %pow_expected_si32 : f64
  cf.assert %pow_ok_si32, "pow si32"
  %pow_base_ui32 = nv_tensor_ir.constant dense<4294967295> : tensor<ui32>
  %pow_exp_ui32 = nv_tensor_ir.constant dense<4294967295> : tensor<ui32>
  %pow_ui32 = func.call @pow_ui32(%pow_base_ui32, %pow_exp_ui32) : (tensor<ui32>, tensor<ui32>) -> tensor<f64>
  %pow_value_ui32 = tensor.extract %pow_ui32[] : tensor<f64>
  %pow_expected_ui32 = arith.constant 4294967295.0 : f64
  %pow_ok_ui32 = arith.cmpf oeq, %pow_value_ui32, %pow_expected_ui32 : f64
  cf.assert %pow_ok_ui32, "pow ui32"
  return
}
