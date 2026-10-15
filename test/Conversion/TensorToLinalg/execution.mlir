// RUN: tensor_ir-opt %s --tir-linalg-pipeline \
// RUN:   | mlir-runner -e main -entry-point-result=void

// Assertions compare converted computations against independently calculated
// values. Exact cases use zero tolerance; activation checks use 1e-6 absolute
// tolerance. Dynamic casts keep runtime batch sizes unknown to the pass.
func.func @close(%actual: f32, %expected: f32, %tolerance: f32) {
  %diff = arith.subf %actual, %expected : f32
  %abs = math.absf %diff : f32
  %ok = arith.cmpf ole, %abs, %tolerance : f32
  cf.assert %ok, "TensorToLinalg numerical mismatch"
  return
}

func.func @batch_dot(%a: tensor<?x2xf32>, %b: tensor<?x2xf32>) -> tensor<?xf32> {
  %r = nv_tensor_ir.contraction(%a, %b) maps (batch, k) -> (batch, k), (batch, k), (batch) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

func.func @mixed_precision(%a: tensor<2xf16>, %b: tensor<2xf16>) -> tensor<f32> {
  %r = nv_tensor_ir.contraction(%a, %b) maps (k) -> (k), (k), () : (tensor<2xf16>, tensor<2xf16>) -> tensor<f32>
  return %r : tensor<f32>
}

// f16 values are widened before multiplication in both matmul forms.
func.func @mixed_matmul(%a: tensor<1x2xf16>, %b: tensor<2x1xf16>) -> tensor<1x1xf32> {
  %r = nv_tensor_ir.matmul(%a, %b) : (tensor<1x2xf16>, tensor<2x1xf16>) -> tensor<1x1xf32>
  return %r : tensor<1x1xf32>
}

func.func @mixed_batch_matmul(%a: tensor<1x1x2xf16>, %b: tensor<1x2x1xf16>) -> tensor<1x1x1xf32> {
  %r = nv_tensor_ir.matmul(%a, %b) : (tensor<1x1x2xf16>, tensor<1x2x1xf16>) -> tensor<1x1x1xf32>
  return %r : tensor<1x1x1xf32>
}

func.func @empty_reduction(%a: tensor<2x0xf32>, %b: tensor<0x3xf32>, %acc: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = nv_tensor_ir.contraction(%a, %b) accum(%acc : tensor<2x3xf32>) maps (m, n, k) -> (m, k), (k, n), (m, n) : (tensor<2x0xf32>, tensor<0x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

func.func @mod_rem(%a: tensor<2xf32>, %b: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %m = nv_tensor_ir.mod %a, %b : tensor<2xf32>
  %r = nv_tensor_ir.rem %a, %b : tensor<2xf32>
  return %m, %r : tensor<2xf32>, tensor<2xf32>
}

func.func @activations(%a: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) {
  %sigmoid = nv_tensor_ir.sigmoid_fwd %a : tensor<1xf32>
  %softplus = nv_tensor_ir.softplus_fwd<beta = 2.0 : f64> %a : tensor<1xf32>
  %swish = nv_tensor_ir.swish_fwd<beta = 2.0 : f64> %a : tensor<1xf32>
  return %sigmoid, %softplus, %swish : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>
}

func.func @special(%a: tensor<1xf32>, %b: tensor<1xf32>) -> (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) {
  %o = nv_tensor_ir.cmp %a oeq %b : tensor<1xf32>
  %u = nv_tensor_ir.cmp %a ueq %b : tensor<1xf32>
  %bool = nv_tensor_ir.convert %a : tensor<1xf32> -> tensor<1xi1>
  %min = nv_tensor_ir.min %a, %b : tensor<1xf32>
  %max = nv_tensor_ir.max %a, %b : tensor<1xf32>
  return %o, %u, %bool, %min, %max : tensor<1xi1>, tensor<1xi1>, tensor<1xi1>, tensor<1xf32>, tensor<1xf32>
}

func.func @main() {
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32
  %tol = arith.constant 1.0e-6 : f32
  %unit = arith.constant dense<[[2.0, 3.0]]> : tensor<1x2xf32>
  %full_input = arith.constant dense<[[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]> : tensor<3x2xf32>
  %empty = arith.constant dense<> : tensor<0x2xf32>
  %u = tensor.cast %unit : tensor<1x2xf32> to tensor<?x2xf32>
  %f = tensor.cast %full_input : tensor<3x2xf32> to tensor<?x2xf32>
  %e = tensor.cast %empty : tensor<0x2xf32> to tensor<?x2xf32>
  %lhs_unit = func.call @batch_dot(%u, %f) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?xf32>
  %lhs_unit_size = tensor.dim %lhs_unit, %i0 : tensor<?xf32>
  %lhs_unit_n = arith.constant 3 : index
  %lhs_unit_size_ok = arith.cmpi eq, %lhs_unit_size, %lhs_unit_n : index
  cf.assert %lhs_unit_size_ok, "broadcast extent"
  %lhs_unit_0 = tensor.extract %lhs_unit[%i0] : tensor<?xf32>
  %lhs_unit_expected_0 = arith.constant 23.0 : f32
  func.call @close(%lhs_unit_0, %lhs_unit_expected_0, %zero) : (f32, f32, f32) -> ()
  %lhs_unit_1 = tensor.extract %lhs_unit[%i1] : tensor<?xf32>
  %lhs_unit_expected_1 = arith.constant 33.0 : f32
  func.call @close(%lhs_unit_1, %lhs_unit_expected_1, %zero) : (f32, f32, f32) -> ()
  %lhs_unit_2 = tensor.extract %lhs_unit[%i2] : tensor<?xf32>
  %lhs_unit_expected_2 = arith.constant 43.0 : f32
  func.call @close(%lhs_unit_2, %lhs_unit_expected_2, %zero) : (f32, f32, f32) -> ()
  %rhs_unit = func.call @batch_dot(%f, %u) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?xf32>
  %rhs_unit_size = tensor.dim %rhs_unit, %i0 : tensor<?xf32>
  %rhs_unit_n = arith.constant 3 : index
  %rhs_unit_size_ok = arith.cmpi eq, %rhs_unit_size, %rhs_unit_n : index
  cf.assert %rhs_unit_size_ok, "broadcast extent"
  %rhs_unit_0 = tensor.extract %rhs_unit[%i0] : tensor<?xf32>
  %rhs_unit_expected_0 = arith.constant 23.0 : f32
  func.call @close(%rhs_unit_0, %rhs_unit_expected_0, %zero) : (f32, f32, f32) -> ()
  %rhs_unit_1 = tensor.extract %rhs_unit[%i1] : tensor<?xf32>
  %rhs_unit_expected_1 = arith.constant 33.0 : f32
  func.call @close(%rhs_unit_1, %rhs_unit_expected_1, %zero) : (f32, f32, f32) -> ()
  %rhs_unit_2 = tensor.extract %rhs_unit[%i2] : tensor<?xf32>
  %rhs_unit_expected_2 = arith.constant 43.0 : f32
  func.call @close(%rhs_unit_2, %rhs_unit_expected_2, %zero) : (f32, f32, f32) -> ()
  %full = func.call @batch_dot(%f, %f) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?xf32>
  %full_size = tensor.dim %full, %i0 : tensor<?xf32>
  %full_n = arith.constant 3 : index
  %full_size_ok = arith.cmpi eq, %full_size, %full_n : index
  cf.assert %full_size_ok, "broadcast extent"
  %full_0 = tensor.extract %full[%i0] : tensor<?xf32>
  %full_expected_0 = arith.constant 41.0 : f32
  func.call @close(%full_0, %full_expected_0, %zero) : (f32, f32, f32) -> ()
  %full_1 = tensor.extract %full[%i1] : tensor<?xf32>
  %full_expected_1 = arith.constant 85.0 : f32
  func.call @close(%full_1, %full_expected_1, %zero) : (f32, f32, f32) -> ()
  %full_2 = tensor.extract %full[%i2] : tensor<?xf32>
  %full_expected_2 = arith.constant 145.0 : f32
  func.call @close(%full_2, %full_expected_2, %zero) : (f32, f32, f32) -> ()
  %both_unit = func.call @batch_dot(%u, %u) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?xf32>
  %both_unit_size = tensor.dim %both_unit, %i0 : tensor<?xf32>
  %both_unit_n = arith.constant 1 : index
  %both_unit_size_ok = arith.cmpi eq, %both_unit_size, %both_unit_n : index
  cf.assert %both_unit_size_ok, "broadcast extent"
  %both_unit_0 = tensor.extract %both_unit[%i0] : tensor<?xf32>
  %both_unit_expected_0 = arith.constant 13.0 : f32
  func.call @close(%both_unit_0, %both_unit_expected_0, %zero) : (f32, f32, f32) -> ()
  %lhs_zero = func.call @batch_dot(%e, %u) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?xf32>
  %lhs_zero_size = tensor.dim %lhs_zero, %i0 : tensor<?xf32>
  %lhs_zero_ok = arith.cmpi eq, %lhs_zero_size, %i0 : index
  cf.assert %lhs_zero_ok, "unit/zero broadcasting must be empty"
  %rhs_zero = func.call @batch_dot(%u, %e) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?xf32>
  %rhs_zero_size = tensor.dim %rhs_zero, %i0 : tensor<?xf32>
  %rhs_zero_ok = arith.cmpi eq, %rhs_zero_size, %i0 : index
  cf.assert %rhs_zero_ok, "unit/zero broadcasting must be empty"
  %both_zero = func.call @batch_dot(%e, %e) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?xf32>
  %both_zero_size = tensor.dim %both_zero, %i0 : tensor<?xf32>
  %both_zero_ok = arith.cmpi eq, %both_zero_size, %i0 : index
  cf.assert %both_zero_ok, "unit/zero broadcasting must be empty"
  %half = arith.constant dense<1.0009765625> : tensor<2xf16>
  %dot = func.call @mixed_precision(%half, %half) : (tensor<2xf16>, tensor<2xf16>) -> tensor<f32>
  %dot_value = tensor.extract %dot[] : tensor<f32>
  %dot_expected = arith.constant 2.0039081573486328125 : f32
  func.call @close(%dot_value, %dot_expected, %zero) : (f32, f32, f32) -> ()
  %half_a = arith.constant dense<1.0009765625> : tensor<1x2xf16>
  %half_b = arith.constant dense<1.0009765625> : tensor<2x1xf16>
  %matmul = func.call @mixed_matmul(%half_a, %half_b) : (tensor<1x2xf16>, tensor<2x1xf16>) -> tensor<1x1xf32>
  %matmul_value = tensor.extract %matmul[%i0, %i0] : tensor<1x1xf32>
  func.call @close(%matmul_value, %dot_expected, %zero) : (f32, f32, f32) -> ()
  %half_batch_a = arith.constant dense<1.0009765625> : tensor<1x1x2xf16>
  %half_batch_b = arith.constant dense<1.0009765625> : tensor<1x2x1xf16>
  %batch_matmul = func.call @mixed_batch_matmul(%half_batch_a, %half_batch_b) : (tensor<1x1x2xf16>, tensor<1x2x1xf16>) -> tensor<1x1x1xf32>
  %batch_matmul_value = tensor.extract %batch_matmul[%i0, %i0, %i0] : tensor<1x1x1xf32>
  func.call @close(%batch_matmul_value, %dot_expected, %zero) : (f32, f32, f32) -> ()
  %empty_a = arith.constant dense<> : tensor<2x0xf32>
  %empty_b = arith.constant dense<> : tensor<0x3xf32>
  %acc = arith.constant dense<7.0> : tensor<2x3xf32>
  %unchanged = func.call @empty_reduction(%empty_a, %empty_b, %acc) : (tensor<2x0xf32>, tensor<0x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %acc_value = tensor.extract %unchanged[%i1, %i2] : tensor<2x3xf32>
  %seven = arith.constant 7.0 : f32
  func.call @close(%acc_value, %seven, %zero) : (f32, f32, f32) -> ()
  %a = arith.constant dense<[-5.0, 5.0]> : tensor<2xf32>
  %b = arith.constant dense<[3.0, -3.0]> : tensor<2xf32>
  %m, %rem = func.call @mod_rem(%a, %b) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  %mod0 = tensor.extract %m[%i0] : tensor<2xf32>
  %mod_expect0 = arith.constant 1.0 : f32
  func.call @close(%mod0, %mod_expect0, %zero) : (f32, f32, f32) -> ()
  %mod1 = tensor.extract %m[%i1] : tensor<2xf32>
  %mod_expect1 = arith.constant -1.0 : f32
  func.call @close(%mod1, %mod_expect1, %zero) : (f32, f32, f32) -> ()
  %rem0 = tensor.extract %rem[%i0] : tensor<2xf32>
  %rem_expect0 = arith.constant -2.0 : f32
  func.call @close(%rem0, %rem_expect0, %zero) : (f32, f32, f32) -> ()
  %rem1 = tensor.extract %rem[%i1] : tensor<2xf32>
  %rem_expect1 = arith.constant 2.0 : f32
  func.call @close(%rem1, %rem_expect1, %zero) : (f32, f32, f32) -> ()
  %activation_input = arith.constant dense<-1.0> : tensor<1xf32>
  %sigmoid, %softplus, %swish = func.call @activations(%activation_input) : (tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
  %sigmoid_value = tensor.extract %sigmoid[%i0] : tensor<1xf32>
  %sigmoid_expected = arith.constant 0.2689414213699951 : f32
  func.call @close(%sigmoid_value, %sigmoid_expected, %tol) : (f32, f32, f32) -> ()
  %softplus_value = tensor.extract %softplus[%i0] : tensor<1xf32>
  %softplus_expected = arith.constant 0.06346400552148625 : f32
  func.call @close(%softplus_value, %softplus_expected, %tol) : (f32, f32, f32) -> ()
  %swish_value = tensor.extract %swish[%i0] : tensor<1xf32>
  %swish_expected = arith.constant -0.11920292202211755 : f32
  func.call @close(%swish_value, %swish_expected, %tol) : (f32, f32, f32) -> ()
  %nan = arith.constant dense<0x7FC00000> : tensor<1xf32>
  %one = arith.constant dense<1.0> : tensor<1xf32>
  %ordered, %unordered, %bool, %min, %max = func.call @special(%nan, %one) : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>, tensor<1xf32>, tensor<1xf32>)
  %o = tensor.extract %ordered[%i0] : tensor<1xi1>
  %u_cmp = tensor.extract %unordered[%i0] : tensor<1xi1>
  %bool_value = tensor.extract %bool[%i0] : tensor<1xi1>
  %true = arith.constant true
  %not_o = arith.xori %o, %true : i1
  cf.assert %not_o, "ordered comparison rejects NaN"
  cf.assert %u_cmp, "unordered comparison accepts NaN"
  cf.assert %bool_value, "NaN converts to true"
  %min_value = tensor.extract %min[%i0] : tensor<1xf32>
  %max_value = tensor.extract %max[%i0] : tensor<1xf32>
  %min_nan = arith.cmpf uno, %min_value, %min_value : f32
  %max_nan = arith.cmpf uno, %max_value, %max_value : f32
  cf.assert %min_nan, "minimum must propagate NaN"
  cf.assert %max_nan, "maximum must propagate NaN"
  return
}
