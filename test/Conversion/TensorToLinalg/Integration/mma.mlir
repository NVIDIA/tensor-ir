// RUN: tensor_ir-opt %s --tir-linalg-pipeline \
// RUN:   | mlir-runner -e main -entry-point-result=void


// CUDA Tile integer MMA retains products and saturates a signed accumulator.
// One-product cases make these expectations independent of reduction order.

func.func @equal(%actual: f64, %expected: f64) {
  %ok = arith.cmpf oeq, %actual, %expected : f64
  cf.assert %ok, "MMA numerical semantics mismatch"
  return
}

func.func @contraction_si8(%a: tensor<1xsi8>, %b: tensor<1xsi8>, %acc: tensor<si32>) -> f64 {
  %r = nv_tensor_ir.contraction(%a, %b) accum(%acc : tensor<si32>) maps (k) -> (k), (k), () : (tensor<1xsi8>, tensor<1xsi8>) -> tensor<si32>
  %f = nv_tensor_ir.convert %r : tensor<si32> -> tensor<f64>
  %i0 = arith.constant 0 : index
  %v = tensor.extract %f[] : tensor<f64>
  return %v : f64
}

func.func @contraction_ui8(%a: tensor<1xui8>, %b: tensor<1xui8>, %acc: tensor<si32>) -> f64 {
  %r = nv_tensor_ir.contraction(%a, %b) accum(%acc : tensor<si32>) maps (k) -> (k), (k), () : (tensor<1xui8>, tensor<1xui8>) -> tensor<si32>
  %f = nv_tensor_ir.convert %r : tensor<si32> -> tensor<f64>
  %i0 = arith.constant 0 : index
  %v = tensor.extract %f[] : tensor<f64>
  return %v : f64
}

func.func @matmul_si8(%a: tensor<1x1xsi8>, %b: tensor<1x1xsi8>, %acc: tensor<1x1xsi32>) -> f64 {
  %r = nv_tensor_ir.matmul(%a, %b) accum(%acc : tensor<1x1xsi32>) : (tensor<1x1xsi8>, tensor<1x1xsi8>) -> tensor<1x1xsi32>
  %f = nv_tensor_ir.convert %r : tensor<1x1xsi32> -> tensor<1x1xf64>
  %i0 = arith.constant 0 : index
  %v = tensor.extract %f[%i0, %i0] : tensor<1x1xf64>
  return %v : f64
}

func.func @matmul_ui8(%a: tensor<1x1xui8>, %b: tensor<1x1xui8>, %acc: tensor<1x1xsi32>) -> f64 {
  %r = nv_tensor_ir.matmul(%a, %b) accum(%acc : tensor<1x1xsi32>) : (tensor<1x1xui8>, tensor<1x1xui8>) -> tensor<1x1xsi32>
  %f = nv_tensor_ir.convert %r : tensor<1x1xsi32> -> tensor<1x1xf64>
  %i0 = arith.constant 0 : index
  %v = tensor.extract %f[%i0, %i0] : tensor<1x1xf64>
  return %v : f64
}

func.func @batch_matmul_si8(%a: tensor<1x1x1xsi8>, %b: tensor<1x1x1xsi8>, %acc: tensor<1x1x1xsi32>) -> f64 {
  %r = nv_tensor_ir.matmul(%a, %b) accum(%acc : tensor<1x1x1xsi32>) : (tensor<1x1x1xsi8>, tensor<1x1x1xsi8>) -> tensor<1x1x1xsi32>
  %f = nv_tensor_ir.convert %r : tensor<1x1x1xsi32> -> tensor<1x1x1xf64>
  %i0 = arith.constant 0 : index
  %v = tensor.extract %f[%i0, %i0, %i0] : tensor<1x1x1xf64>
  return %v : f64
}

func.func @batch_matmul_ui8(%a: tensor<1x1x1xui8>, %b: tensor<1x1x1xui8>, %acc: tensor<1x1x1xsi32>) -> f64 {
  %r = nv_tensor_ir.matmul(%a, %b) accum(%acc : tensor<1x1x1xsi32>) : (tensor<1x1x1xui8>, tensor<1x1x1xui8>) -> tensor<1x1x1xsi32>
  %f = nv_tensor_ir.convert %r : tensor<1x1x1xsi32> -> tensor<1x1x1xf64>
  %i0 = arith.constant 0 : index
  %v = tensor.extract %f[%i0, %i0, %i0] : tensor<1x1x1xf64>
  return %v : f64
}

func.func @unsigned_accumulator(%a: tensor<1xui8>, %b: tensor<1xui8>, %acc: tensor<ui32>) -> f64 {
  %r = nv_tensor_ir.contraction(%a, %b) accum(%acc : tensor<ui32>) maps (k) -> (k), (k), () : (tensor<1xui8>, tensor<1xui8>) -> tensor<ui32>
  %f = nv_tensor_ir.convert %r : tensor<ui32> -> tensor<f64>
  %v = tensor.extract %f[] : tensor<f64>
  return %v : f64
}

func.func @main() {
  %contraction_si8_0_a = nv_tensor_ir.constant dense<127> : tensor<1xsi8>
  %contraction_si8_0_b = nv_tensor_ir.constant dense<127> : tensor<1xsi8>
  %contraction_si8_0_acc = nv_tensor_ir.constant dense<2147483640> : tensor<si32>
  %contraction_si8_0_r = func.call @contraction_si8(%contraction_si8_0_a, %contraction_si8_0_b, %contraction_si8_0_acc) : (tensor<1xsi8>, tensor<1xsi8>, tensor<si32>) -> f64
  %contraction_si8_0_e = arith.constant 2147483647.0 : f64
  func.call @equal(%contraction_si8_0_r, %contraction_si8_0_e) : (f64, f64) -> ()
  %contraction_si8_1_a = nv_tensor_ir.constant dense<-128> : tensor<1xsi8>
  %contraction_si8_1_b = nv_tensor_ir.constant dense<127> : tensor<1xsi8>
  %contraction_si8_1_acc = nv_tensor_ir.constant dense<-2147483640> : tensor<si32>
  %contraction_si8_1_r = func.call @contraction_si8(%contraction_si8_1_a, %contraction_si8_1_b, %contraction_si8_1_acc) : (tensor<1xsi8>, tensor<1xsi8>, tensor<si32>) -> f64
  %contraction_si8_1_e = arith.constant -2147483648.0 : f64
  func.call @equal(%contraction_si8_1_r, %contraction_si8_1_e) : (f64, f64) -> ()
  %contraction_si8_2_a = nv_tensor_ir.constant dense<-128> : tensor<1xsi8>
  %contraction_si8_2_b = nv_tensor_ir.constant dense<127> : tensor<1xsi8>
  %contraction_si8_2_acc = nv_tensor_ir.constant dense<0> : tensor<si32>
  %contraction_si8_2_r = func.call @contraction_si8(%contraction_si8_2_a, %contraction_si8_2_b, %contraction_si8_2_acc) : (tensor<1xsi8>, tensor<1xsi8>, tensor<si32>) -> f64
  %contraction_si8_2_e = arith.constant -16256.0 : f64
  func.call @equal(%contraction_si8_2_r, %contraction_si8_2_e) : (f64, f64) -> ()
  %contraction_si8_3_a = nv_tensor_ir.constant dense<127> : tensor<1xsi8>
  %contraction_si8_3_b = nv_tensor_ir.constant dense<127> : tensor<1xsi8>
  %contraction_si8_3_acc = nv_tensor_ir.constant dense<0> : tensor<si32>
  %contraction_si8_3_r = func.call @contraction_si8(%contraction_si8_3_a, %contraction_si8_3_b, %contraction_si8_3_acc) : (tensor<1xsi8>, tensor<1xsi8>, tensor<si32>) -> f64
  %contraction_si8_3_e = arith.constant 16129.0 : f64
  func.call @equal(%contraction_si8_3_r, %contraction_si8_3_e) : (f64, f64) -> ()
  %contraction_ui8_0_a = nv_tensor_ir.constant dense<255> : tensor<1xui8>
  %contraction_ui8_0_b = nv_tensor_ir.constant dense<255> : tensor<1xui8>
  %contraction_ui8_0_acc = nv_tensor_ir.constant dense<0> : tensor<si32>
  %contraction_ui8_0_r = func.call @contraction_ui8(%contraction_ui8_0_a, %contraction_ui8_0_b, %contraction_ui8_0_acc) : (tensor<1xui8>, tensor<1xui8>, tensor<si32>) -> f64
  %contraction_ui8_0_e = arith.constant 65025.0 : f64
  func.call @equal(%contraction_ui8_0_r, %contraction_ui8_0_e) : (f64, f64) -> ()
  %contraction_ui8_1_a = nv_tensor_ir.constant dense<255> : tensor<1xui8>
  %contraction_ui8_1_b = nv_tensor_ir.constant dense<255> : tensor<1xui8>
  %contraction_ui8_1_acc = nv_tensor_ir.constant dense<2147483640> : tensor<si32>
  %contraction_ui8_1_r = func.call @contraction_ui8(%contraction_ui8_1_a, %contraction_ui8_1_b, %contraction_ui8_1_acc) : (tensor<1xui8>, tensor<1xui8>, tensor<si32>) -> f64
  %contraction_ui8_1_e = arith.constant 2147483647.0 : f64
  func.call @equal(%contraction_ui8_1_r, %contraction_ui8_1_e) : (f64, f64) -> ()
  %contraction_ui8_2_a = nv_tensor_ir.constant dense<255> : tensor<1xui8>
  %contraction_ui8_2_b = nv_tensor_ir.constant dense<2> : tensor<1xui8>
  %contraction_ui8_2_acc = nv_tensor_ir.constant dense<-1> : tensor<si32>
  %contraction_ui8_2_r = func.call @contraction_ui8(%contraction_ui8_2_a, %contraction_ui8_2_b, %contraction_ui8_2_acc) : (tensor<1xui8>, tensor<1xui8>, tensor<si32>) -> f64
  %contraction_ui8_2_e = arith.constant 509.0 : f64
  func.call @equal(%contraction_ui8_2_r, %contraction_ui8_2_e) : (f64, f64) -> ()
  %matmul_si8_0_a = nv_tensor_ir.constant dense<127> : tensor<1x1xsi8>
  %matmul_si8_0_b = nv_tensor_ir.constant dense<127> : tensor<1x1xsi8>
  %matmul_si8_0_acc = nv_tensor_ir.constant dense<2147483640> : tensor<1x1xsi32>
  %matmul_si8_0_r = func.call @matmul_si8(%matmul_si8_0_a, %matmul_si8_0_b, %matmul_si8_0_acc) : (tensor<1x1xsi8>, tensor<1x1xsi8>, tensor<1x1xsi32>) -> f64
  %matmul_si8_0_e = arith.constant 2147483647.0 : f64
  func.call @equal(%matmul_si8_0_r, %matmul_si8_0_e) : (f64, f64) -> ()
  %matmul_si8_1_a = nv_tensor_ir.constant dense<-128> : tensor<1x1xsi8>
  %matmul_si8_1_b = nv_tensor_ir.constant dense<127> : tensor<1x1xsi8>
  %matmul_si8_1_acc = nv_tensor_ir.constant dense<-2147483640> : tensor<1x1xsi32>
  %matmul_si8_1_r = func.call @matmul_si8(%matmul_si8_1_a, %matmul_si8_1_b, %matmul_si8_1_acc) : (tensor<1x1xsi8>, tensor<1x1xsi8>, tensor<1x1xsi32>) -> f64
  %matmul_si8_1_e = arith.constant -2147483648.0 : f64
  func.call @equal(%matmul_si8_1_r, %matmul_si8_1_e) : (f64, f64) -> ()
  %matmul_si8_2_a = nv_tensor_ir.constant dense<-128> : tensor<1x1xsi8>
  %matmul_si8_2_b = nv_tensor_ir.constant dense<127> : tensor<1x1xsi8>
  %matmul_si8_2_acc = nv_tensor_ir.constant dense<0> : tensor<1x1xsi32>
  %matmul_si8_2_r = func.call @matmul_si8(%matmul_si8_2_a, %matmul_si8_2_b, %matmul_si8_2_acc) : (tensor<1x1xsi8>, tensor<1x1xsi8>, tensor<1x1xsi32>) -> f64
  %matmul_si8_2_e = arith.constant -16256.0 : f64
  func.call @equal(%matmul_si8_2_r, %matmul_si8_2_e) : (f64, f64) -> ()
  %matmul_si8_3_a = nv_tensor_ir.constant dense<127> : tensor<1x1xsi8>
  %matmul_si8_3_b = nv_tensor_ir.constant dense<127> : tensor<1x1xsi8>
  %matmul_si8_3_acc = nv_tensor_ir.constant dense<0> : tensor<1x1xsi32>
  %matmul_si8_3_r = func.call @matmul_si8(%matmul_si8_3_a, %matmul_si8_3_b, %matmul_si8_3_acc) : (tensor<1x1xsi8>, tensor<1x1xsi8>, tensor<1x1xsi32>) -> f64
  %matmul_si8_3_e = arith.constant 16129.0 : f64
  func.call @equal(%matmul_si8_3_r, %matmul_si8_3_e) : (f64, f64) -> ()
  %matmul_ui8_0_a = nv_tensor_ir.constant dense<255> : tensor<1x1xui8>
  %matmul_ui8_0_b = nv_tensor_ir.constant dense<255> : tensor<1x1xui8>
  %matmul_ui8_0_acc = nv_tensor_ir.constant dense<0> : tensor<1x1xsi32>
  %matmul_ui8_0_r = func.call @matmul_ui8(%matmul_ui8_0_a, %matmul_ui8_0_b, %matmul_ui8_0_acc) : (tensor<1x1xui8>, tensor<1x1xui8>, tensor<1x1xsi32>) -> f64
  %matmul_ui8_0_e = arith.constant 65025.0 : f64
  func.call @equal(%matmul_ui8_0_r, %matmul_ui8_0_e) : (f64, f64) -> ()
  %matmul_ui8_1_a = nv_tensor_ir.constant dense<255> : tensor<1x1xui8>
  %matmul_ui8_1_b = nv_tensor_ir.constant dense<255> : tensor<1x1xui8>
  %matmul_ui8_1_acc = nv_tensor_ir.constant dense<2147483640> : tensor<1x1xsi32>
  %matmul_ui8_1_r = func.call @matmul_ui8(%matmul_ui8_1_a, %matmul_ui8_1_b, %matmul_ui8_1_acc) : (tensor<1x1xui8>, tensor<1x1xui8>, tensor<1x1xsi32>) -> f64
  %matmul_ui8_1_e = arith.constant 2147483647.0 : f64
  func.call @equal(%matmul_ui8_1_r, %matmul_ui8_1_e) : (f64, f64) -> ()
  %matmul_ui8_2_a = nv_tensor_ir.constant dense<255> : tensor<1x1xui8>
  %matmul_ui8_2_b = nv_tensor_ir.constant dense<2> : tensor<1x1xui8>
  %matmul_ui8_2_acc = nv_tensor_ir.constant dense<-1> : tensor<1x1xsi32>
  %matmul_ui8_2_r = func.call @matmul_ui8(%matmul_ui8_2_a, %matmul_ui8_2_b, %matmul_ui8_2_acc) : (tensor<1x1xui8>, tensor<1x1xui8>, tensor<1x1xsi32>) -> f64
  %matmul_ui8_2_e = arith.constant 509.0 : f64
  func.call @equal(%matmul_ui8_2_r, %matmul_ui8_2_e) : (f64, f64) -> ()
  %batch_matmul_si8_0_a = nv_tensor_ir.constant dense<127> : tensor<1x1x1xsi8>
  %batch_matmul_si8_0_b = nv_tensor_ir.constant dense<127> : tensor<1x1x1xsi8>
  %batch_matmul_si8_0_acc = nv_tensor_ir.constant dense<2147483640> : tensor<1x1x1xsi32>
  %batch_matmul_si8_0_r = func.call @batch_matmul_si8(%batch_matmul_si8_0_a, %batch_matmul_si8_0_b, %batch_matmul_si8_0_acc) : (tensor<1x1x1xsi8>, tensor<1x1x1xsi8>, tensor<1x1x1xsi32>) -> f64
  %batch_matmul_si8_0_e = arith.constant 2147483647.0 : f64
  func.call @equal(%batch_matmul_si8_0_r, %batch_matmul_si8_0_e) : (f64, f64) -> ()
  %batch_matmul_si8_1_a = nv_tensor_ir.constant dense<-128> : tensor<1x1x1xsi8>
  %batch_matmul_si8_1_b = nv_tensor_ir.constant dense<127> : tensor<1x1x1xsi8>
  %batch_matmul_si8_1_acc = nv_tensor_ir.constant dense<-2147483640> : tensor<1x1x1xsi32>
  %batch_matmul_si8_1_r = func.call @batch_matmul_si8(%batch_matmul_si8_1_a, %batch_matmul_si8_1_b, %batch_matmul_si8_1_acc) : (tensor<1x1x1xsi8>, tensor<1x1x1xsi8>, tensor<1x1x1xsi32>) -> f64
  %batch_matmul_si8_1_e = arith.constant -2147483648.0 : f64
  func.call @equal(%batch_matmul_si8_1_r, %batch_matmul_si8_1_e) : (f64, f64) -> ()
  %batch_matmul_si8_2_a = nv_tensor_ir.constant dense<-128> : tensor<1x1x1xsi8>
  %batch_matmul_si8_2_b = nv_tensor_ir.constant dense<127> : tensor<1x1x1xsi8>
  %batch_matmul_si8_2_acc = nv_tensor_ir.constant dense<0> : tensor<1x1x1xsi32>
  %batch_matmul_si8_2_r = func.call @batch_matmul_si8(%batch_matmul_si8_2_a, %batch_matmul_si8_2_b, %batch_matmul_si8_2_acc) : (tensor<1x1x1xsi8>, tensor<1x1x1xsi8>, tensor<1x1x1xsi32>) -> f64
  %batch_matmul_si8_2_e = arith.constant -16256.0 : f64
  func.call @equal(%batch_matmul_si8_2_r, %batch_matmul_si8_2_e) : (f64, f64) -> ()
  %batch_matmul_si8_3_a = nv_tensor_ir.constant dense<127> : tensor<1x1x1xsi8>
  %batch_matmul_si8_3_b = nv_tensor_ir.constant dense<127> : tensor<1x1x1xsi8>
  %batch_matmul_si8_3_acc = nv_tensor_ir.constant dense<0> : tensor<1x1x1xsi32>
  %batch_matmul_si8_3_r = func.call @batch_matmul_si8(%batch_matmul_si8_3_a, %batch_matmul_si8_3_b, %batch_matmul_si8_3_acc) : (tensor<1x1x1xsi8>, tensor<1x1x1xsi8>, tensor<1x1x1xsi32>) -> f64
  %batch_matmul_si8_3_e = arith.constant 16129.0 : f64
  func.call @equal(%batch_matmul_si8_3_r, %batch_matmul_si8_3_e) : (f64, f64) -> ()
  %batch_matmul_ui8_0_a = nv_tensor_ir.constant dense<255> : tensor<1x1x1xui8>
  %batch_matmul_ui8_0_b = nv_tensor_ir.constant dense<255> : tensor<1x1x1xui8>
  %batch_matmul_ui8_0_acc = nv_tensor_ir.constant dense<0> : tensor<1x1x1xsi32>
  %batch_matmul_ui8_0_r = func.call @batch_matmul_ui8(%batch_matmul_ui8_0_a, %batch_matmul_ui8_0_b, %batch_matmul_ui8_0_acc) : (tensor<1x1x1xui8>, tensor<1x1x1xui8>, tensor<1x1x1xsi32>) -> f64
  %batch_matmul_ui8_0_e = arith.constant 65025.0 : f64
  func.call @equal(%batch_matmul_ui8_0_r, %batch_matmul_ui8_0_e) : (f64, f64) -> ()
  %batch_matmul_ui8_1_a = nv_tensor_ir.constant dense<255> : tensor<1x1x1xui8>
  %batch_matmul_ui8_1_b = nv_tensor_ir.constant dense<255> : tensor<1x1x1xui8>
  %batch_matmul_ui8_1_acc = nv_tensor_ir.constant dense<2147483640> : tensor<1x1x1xsi32>
  %batch_matmul_ui8_1_r = func.call @batch_matmul_ui8(%batch_matmul_ui8_1_a, %batch_matmul_ui8_1_b, %batch_matmul_ui8_1_acc) : (tensor<1x1x1xui8>, tensor<1x1x1xui8>, tensor<1x1x1xsi32>) -> f64
  %batch_matmul_ui8_1_e = arith.constant 2147483647.0 : f64
  func.call @equal(%batch_matmul_ui8_1_r, %batch_matmul_ui8_1_e) : (f64, f64) -> ()
  %batch_matmul_ui8_2_a = nv_tensor_ir.constant dense<255> : tensor<1x1x1xui8>
  %batch_matmul_ui8_2_b = nv_tensor_ir.constant dense<2> : tensor<1x1x1xui8>
  %batch_matmul_ui8_2_acc = nv_tensor_ir.constant dense<-1> : tensor<1x1x1xsi32>
  %batch_matmul_ui8_2_r = func.call @batch_matmul_ui8(%batch_matmul_ui8_2_a, %batch_matmul_ui8_2_b, %batch_matmul_ui8_2_acc) : (tensor<1x1x1xui8>, tensor<1x1x1xui8>, tensor<1x1x1xsi32>) -> f64
  %batch_matmul_ui8_2_e = arith.constant 509.0 : f64
  func.call @equal(%batch_matmul_ui8_2_r, %batch_matmul_ui8_2_e) : (f64, f64) -> ()
  %one = nv_tensor_ir.constant dense<1> : tensor<1xui8>
  %minus_one_bits = nv_tensor_ir.constant dense<4294967295> : tensor<ui32>
  %unsigned_r = func.call @unsigned_accumulator(%one, %one, %minus_one_bits) : (tensor<1xui8>, tensor<1xui8>, tensor<ui32>) -> f64
  %zero = arith.constant 0.0 : f64
  func.call @equal(%unsigned_r, %zero) : (f64, f64) -> ()
  // Each MMA input has its own signedness.
  %mixed_a = nv_tensor_ir.constant dense<255> : tensor<1x1xui8>
  %mixed_b = nv_tensor_ir.constant dense<-128> : tensor<1x1xsi8>
  %mixed_r = nv_tensor_ir.matmul(%mixed_a, %mixed_b) : (tensor<1x1xui8>, tensor<1x1xsi8>) -> tensor<1x1xsi32>
  %mixed_f = nv_tensor_ir.convert %mixed_r : tensor<1x1xsi32> -> tensor<1x1xf64>
  %i0 = arith.constant 0 : index
  %mixed_v = tensor.extract %mixed_f[%i0, %i0] : tensor<1x1xf64>
  %mixed_expected = arith.constant -32640.0 : f64
  func.call @equal(%mixed_v, %mixed_expected) : (f64, f64) -> ()
  return
}
