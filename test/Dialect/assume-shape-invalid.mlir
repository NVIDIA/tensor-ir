// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics

func.func @wrong_extent_count(%a: tensor<?x?xf32>, %n: index) {
  // expected-error @below {{expected 2 extent(s), but got 1}}
  nv_tensor_ir.assume_shape %a, [%n] : tensor<?x?xf32>
  return
}

// -----

func.func @negative_static_extent(%a: tensor<?xf32>) {
  // expected-error @below {{expected nonnegative extent #0, but got -1}}
  nv_tensor_ir.assume_shape %a, [-1] : tensor<?xf32>
  return
}

// -----

func.func @negative_constant_extent(%a: tensor<?xf32>) {
  %negative = arith.constant -2 : index
  // expected-error @below {{expected nonnegative extent #0, but got -2}}
  nv_tensor_ir.assume_shape %a, [%negative] : tensor<?xf32>
  return
}

// -----

func.func @incompatible_static_extent(%a: tensor<4xf32>) {
  // expected-error @below {{extent #0 is 8 but the tensor type requires 4}}
  nv_tensor_ir.assume_shape %a, [8] : tensor<4xf32>
  return
}

// -----

func.func @incompatible_constant_extent(%a: tensor<4xf32>) {
  %eight = arith.constant 8 : index
  // expected-error @below {{extent #0 is 8 but the tensor type requires 4}}
  nv_tensor_ir.assume_shape %a, [%eight] : tensor<4xf32>
  return
}

// -----

func.func @missing_dynamic_extent(%a: tensor<?xf32>) {
  // expected-error @below {{expected 1 dynamic sizes values}}
  "nv_tensor_ir.assume_shape"(%a) <{static_sizes = array<i64: -9223372036854775808>}> : (tensor<?xf32>) -> ()
  return
}
