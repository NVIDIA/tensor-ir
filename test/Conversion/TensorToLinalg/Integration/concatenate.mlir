// RUN: tensor_ir-opt %s --tir-linalg-pipeline \
// RUN:   | mlir-runner -e main -entry-point-result=void --shared-libs=%mlir_c_runner_utils

func.func @rows(%a: tensor<?x2xf32>, %b: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %r = nv_tensor_ir.concatenate %a, %b dimension = 0 : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  return %r : tensor<?x2xf32>
}

func.func @columns(%a: tensor<2x?xf32>, %b: tensor<2x?xf32>) -> tensor<2x?xf32> {
  %r = nv_tensor_ir.concatenate %a, %b dimension = 1 : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
  return %r : tensor<2x?xf32>
}

// Compare every extent and element, including empty tensors.
func.func @check(%a: tensor<?x?xf32>, %expected: tensor<?x?xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %m = tensor.dim %a, %zero : tensor<?x?xf32>
  %n = tensor.dim %a, %one : tensor<?x?xf32>
  %em = tensor.dim %expected, %zero : tensor<?x?xf32>
  %en = tensor.dim %expected, %one : tensor<?x?xf32>
  %same_m = arith.cmpi eq, %m, %em : index
  %same_n = arith.cmpi eq, %n, %en : index
  cf.assert %same_m, "concatenation row count"
  cf.assert %same_n, "concatenation column count"
  scf.for %i = %zero to %m step %one {
    scf.for %j = %zero to %n step %one {
      %v = tensor.extract %a[%i, %j] : tensor<?x?xf32>
      %e = tensor.extract %expected[%i, %j] : tensor<?x?xf32>
      %same = arith.cmpf oeq, %v, %e : f32
      cf.assert %same, "concatenation element order"
    }
  }
  return
}

// Exercise static empty operands at both ends of a nonempty concatenation.
func.func @static_empty_operand(%empty: tensor<0x2xf32>, %a: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %r = nv_tensor_ir.concatenate %empty, %a, %empty dimension = 0 : (tensor<0x2xf32>, tensor<1x2xf32>, tensor<0x2xf32>) -> tensor<1x2xf32>
  return %r : tensor<1x2xf32>
}

func.func @empty_payload(%a: tensor<0x?xf32>, %b: tensor<0x?xf32>) -> tensor<0x?xf32> {
  %r = nv_tensor_ir.concatenate %a, %b dimension = 1 : (tensor<0x?xf32>, tensor<0x?xf32>) -> tensor<0x?xf32>
  return %r : tensor<0x?xf32>
}

func.func @main() {
  %a = arith.constant dense<[[1.0, 2.0]]> : tensor<1x2xf32>
  %b = arith.constant dense<[[3.0, 4.0], [5.0, 6.0]]> : tensor<2x2xf32>
  %empty = arith.constant dense<> : tensor<0x2xf32>
  %expected = arith.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  %da = tensor.cast %a : tensor<1x2xf32> to tensor<?x2xf32>
  %db = tensor.cast %b : tensor<2x2xf32> to tensor<?x2xf32>
  %de = tensor.cast %empty : tensor<0x2xf32> to tensor<?x2xf32>
  %a_ref = tensor.cast %a : tensor<1x2xf32> to tensor<?x?xf32>
  %empty_ref = tensor.cast %empty : tensor<0x2xf32> to tensor<?x?xf32>
  %expected_ref = tensor.cast %expected : tensor<3x2xf32> to tensor<?x?xf32>
  %r = func.call @rows(%da, %db) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  %r_view = tensor.cast %r : tensor<?x2xf32> to tensor<?x?xf32>
  func.call @check(%r_view, %expected_ref) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
  %empty_first = func.call @rows(%de, %da) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  %empty_first_view = tensor.cast %empty_first : tensor<?x2xf32> to tensor<?x?xf32>
  func.call @check(%empty_first_view, %a_ref) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
  %empty_last = func.call @rows(%da, %de) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  %empty_last_view = tensor.cast %empty_last : tensor<?x2xf32> to tensor<?x?xf32>
  func.call @check(%empty_last_view, %a_ref) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
  %all_empty = func.call @rows(%de, %de) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  %all_empty_view = tensor.cast %all_empty : tensor<?x2xf32> to tensor<?x?xf32>
  func.call @check(%all_empty_view, %empty_ref) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
  %column = arith.constant dense<[[7.0], [8.0]]> : tensor<2x1xf32>
  %column_expected = arith.constant dense<[[3.0, 4.0, 7.0], [5.0, 6.0, 8.0]]> : tensor<2x3xf32>
  %dc = tensor.cast %column : tensor<2x1xf32> to tensor<2x?xf32>
  %db_columns = tensor.cast %b : tensor<2x2xf32> to tensor<2x?xf32>
  %column_ref = tensor.cast %column_expected : tensor<2x3xf32> to tensor<?x?xf32>
  %columns = func.call @columns(%db_columns, %dc) : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
  %columns_view = tensor.cast %columns : tensor<2x?xf32> to tensor<?x?xf32>
  func.call @check(%columns_view, %column_ref) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
  %empty_columns = arith.constant dense<> : tensor<2x0xf32>
  %dec = tensor.cast %empty_columns : tensor<2x0xf32> to tensor<2x?xf32>
  %empty_column_ref = tensor.cast %empty_columns : tensor<2x0xf32> to tensor<?x?xf32>
  %ec = func.call @columns(%dec, %dec) : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
  %ec_view = tensor.cast %ec : tensor<2x?xf32> to tensor<?x?xf32>
  func.call @check(%ec_view, %empty_column_ref) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
  %static_empty_r = func.call @static_empty_operand(%empty, %a) : (tensor<0x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
  %static_empty_view = tensor.cast %static_empty_r : tensor<1x2xf32> to tensor<?x?xf32>
  func.call @check(%static_empty_view, %a_ref) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
  %empty_a = arith.constant dense<> : tensor<0x2xf32>
  %empty_b = arith.constant dense<> : tensor<0x3xf32>
  %empty_expected = arith.constant dense<> : tensor<0x5xf32>
  %empty_a_dyn = tensor.cast %empty_a : tensor<0x2xf32> to tensor<0x?xf32>
  %empty_b_dyn = tensor.cast %empty_b : tensor<0x3xf32> to tensor<0x?xf32>
  %payload = func.call @empty_payload(%empty_a_dyn, %empty_b_dyn) : (tensor<0x?xf32>, tensor<0x?xf32>) -> tensor<0x?xf32>
  %payload_view = tensor.cast %payload : tensor<0x?xf32> to tensor<?x?xf32>
  %payload_ref = tensor.cast %empty_expected : tensor<0x5xf32> to tensor<?x?xf32>
  func.call @check(%payload_view, %payload_ref) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
  // Copies preserve high unsigned bits across integer type conversion.
  %u0 = nv_tensor_ir.constant dense<4294967295> : tensor<1xui32>
  %u1 = nv_tensor_ir.constant dense<2147483648> : tensor<1xui32>
  %ur = nv_tensor_ir.concatenate %u0, %u1 dimension = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
  %uf = nv_tensor_ir.convert %ur : tensor<2xui32> -> tensor<2xf64>
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %v0 = tensor.extract %uf[%i0] : tensor<2xf64>
  %v1 = tensor.extract %uf[%i1] : tensor<2xf64>
  %e0 = arith.constant 4294967295.0 : f64
  %e1 = arith.constant 2147483648.0 : f64
  %ok0 = arith.cmpf oeq, %v0, %e0 : f64
  %ok1 = arith.cmpf oeq, %v1, %e1 : f64
  cf.assert %ok0, "concatenation preserves ui32 maximum"
  cf.assert %ok1, "concatenation preserves ui32 high bit"
  return
}
