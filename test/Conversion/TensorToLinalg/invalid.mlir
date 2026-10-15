// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)'

nv_tensor_ir.graph @missing_splat_size(%v: f32) -> tensor<?xf32> {
  // expected-error@below {{failed to legalize operation 'nv_tensor_ir.splat'}}
  %r = splat %v : tensor<?xf32>
  results %r : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @missing_iota_size() -> tensor<?xf32> {
  // expected-error@below {{failed to legalize operation 'nv_tensor_ir.iota'}}
  %r = iota dimension = 0 : tensor<?xf32>
  results %r : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @missing_reshape_size(%v: tensor<2x?xf32>) -> tensor<?x2xf32> {
  // expected-error@below {{failed to legalize operation 'nv_tensor_ir.reshape'}}
  %r = reshape %v : tensor<2x?xf32> -> tensor<?x2xf32>
  results %r : tensor<?x2xf32>
}

// -----

nv_tensor_ir.graph @missing_broadcast_size(%v: tensor<1xf32>) -> tensor<?xf32> {
  // expected-error@below {{failed to legalize operation 'nv_tensor_ir.broadcast'}}
  %r = broadcast %v : tensor<1xf32> -> tensor<?xf32>
  results %r : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @mixed_contraction_types(%a: tensor<2x3xf16>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error@below {{failed to legalize operation 'nv_tensor_ir.contraction'}}
  %r = contraction(%a, %b) maps (m, n, k) -> (m, k), (k, n), (m, n) : (tensor<2x3xf16>, tensor<3x4xf32>) -> tensor<2x4xf32>
  results %r : tensor<2x4xf32>
}

// -----

// Input verification rejects the cycle before the conversion can run.
nv_tensor_ir.graph @cycle(%a: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error@below {{operand #1 does not dominate this use}}
  %x = add %a, %y : tensor<2xf32>
  // expected-note@below {{operand defined here}}
  %y = mul %x, %a : tensor<2xf32>
  results %x : tensor<2xf32>
}

// -----

nv_tensor_ir.graph @unsupported_unsigned_pow_width(%a: tensor<2xui64>, %b: tensor<2xui64>) -> tensor<2xui64> {
  // expected-error@below {{failed to legalize operation 'nv_tensor_ir.pow'}}
  %r = pow %a, %b : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xui64>
  results %r : tensor<2xui64>
}

// -----

nv_tensor_ir.graph @integer_norm2(%a: tensor<4xsi32>) -> tensor<1xsi32> {
  // expected-error@below {{failed to legalize operation 'nv_tensor_ir.reduce'}}
  %r = reduce(%a)<dimensions = [0], reduction_mode = <norm2>> : tensor<4xsi32> -> tensor<1xsi32>
  results %r : tensor<1xsi32>
}
