// RUN: tensor_ir-opt %s --verify-diagnostics --split-input-file

// -----

nv_tensor_ir.graph @dynamic(%initial: tensor<2x8xf32>, %values: tensor<?x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<?x2x8xf32>) {
  // expected-error@below {{TensorIR numeric tensor with static, non-empty dimensions}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64: 0>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    nv_tensor_ir.yield %carry, %value : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<?x2x8xf32>) -> (tensor<2x8xf32>, tensor<?x2x8xf32>)
  results %final, %history : tensor<2x8xf32>, tensor<?x2x8xf32>
}

// -----

nv_tensor_ir.graph @empty(%initial: tensor<2x8xf32>, %values: tensor<0x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<0x2x8xf32>) {
  // expected-error@below {{TensorIR numeric tensor with static, non-empty dimensions}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64: 0>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    nv_tensor_ir.yield %carry, %value : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<0x2x8xf32>) -> (tensor<2x8xf32>, tensor<0x2x8xf32>)
  results %final, %history : tensor<2x8xf32>, tensor<0x2x8xf32>
}

// -----

nv_tensor_ir.graph @negative_axis(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32>) {
  // expected-error@below {{whose value is non-negative}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: -1>, output_dimensions = array<i64: 0>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    nv_tensor_ir.yield %carry, %value : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> (tensor<2x8xf32>, tensor<5x2x8xf32>)
  results %final, %history : tensor<2x8xf32>, tensor<5x2x8xf32>
}

// -----

nv_tensor_ir.graph @axis_count(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32>) {
  // expected-error@below {{one scan dimension per scanned input}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64>, output_dimensions = array<i64: 0>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    nv_tensor_ir.yield %carry, %value : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> (tensor<2x8xf32>, tensor<5x2x8xf32>)
  results %final, %history : tensor<2x8xf32>, tensor<5x2x8xf32>
}

// -----

nv_tensor_ir.graph @output_count(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32>) {
  // expected-error@below {{one output dimension per scan_slice output}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    nv_tensor_ir.yield %carry, %value : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> (tensor<2x8xf32>, tensor<5x2x8xf32>)
  results %final, %history : tensor<2x8xf32>, tensor<5x2x8xf32>
}

// -----

nv_tensor_ir.graph @axis_range(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@below {{scan dimension is out of range}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 3>, output_dimensions = array<i64: 0>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    nv_tensor_ir.yield %carry, %value : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> (tensor<2x8xf32>, tensor<5x2x8xf32>)
  results %final, %history : tensor<2x8xf32>, tensor<5x2x8xf32>
}

// -----

nv_tensor_ir.graph @output_axis_range(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@below {{scan_slice output dimension is out of range}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64: 3>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    nv_tensor_ir.yield %carry, %value : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> (tensor<2x8xf32>, tensor<5x2x8xf32>)
  results %final, %history : tensor<2x8xf32>, tensor<5x2x8xf32>
}

// -----

nv_tensor_ir.graph @body_argument(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@below {{body scanned argument must have the sliced input type}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64: 0>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<8x2xf32>):
    nv_tensor_ir.yield %carry, %carry : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> (tensor<2x8xf32>, tensor<5x2x8xf32>)
  results %final, %history : tensor<2x8xf32>, tensor<5x2x8xf32>
}

// -----

nv_tensor_ir.graph @wrong_carry_result(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf16>, tensor<5x2x8xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@below {{inferred type(s)}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64: 0>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    nv_tensor_ir.yield %carry, %value : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> (tensor<2x8xf16>, tensor<5x2x8xf32>)
  results %final, %history : tensor<2x8xf16>, tensor<5x2x8xf32>
}

// -----

nv_tensor_ir.graph @wrong_history_extent(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<6x2x8xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@below {{inferred type(s)}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64: 0>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    nv_tensor_ir.yield %carry, %value : tensor<2x8xf32>, tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> (tensor<2x8xf32>, tensor<6x2x8xf32>)
  results %final, %history : tensor<2x8xf32>, tensor<6x2x8xf32>
}

// -----

nv_tensor_ir.graph @yield_count(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@below {{body must yield the next carry}}
  %final, %history = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64: 0>
  }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    nv_tensor_ir.yield %carry : tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> (tensor<2x8xf32>, tensor<5x2x8xf32>)
  results %final, %history : tensor<2x8xf32>, tensor<5x2x8xf32>
}

// -----

nv_tensor_ir.graph @unequal_extents(%initial: tensor<8xf32>,
    %a: tensor<5x8xf32>, %b: tensor<6x8xf32>) -> tensor<8xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@below {{requires equal positive scan extents}}
  %final = "nv_tensor_ir.scan_slice"(%initial, %a, %b) <{
    scan_dimensions = array<i64: 0, 0>, output_dimensions = array<i64>
  }> ({
  ^bb0(%carry: tensor<8xf32>, %x: tensor<8xf32>, %y: tensor<8xf32>):
    nv_tensor_ir.yield %carry : tensor<8xf32>
  }) : (tensor<8xf32>, tensor<5x8xf32>, tensor<6x8xf32>) -> tensor<8xf32>
  results %final : tensor<8xf32>
}

// -----

nv_tensor_ir.graph @yield_carry_type(%initial: tensor<8xf32>,
    %values: tensor<5x8xbf16>) -> tensor<8xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@below {{body must yield a carry matching the input carry}}
  %final = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64>
  }> ({
  ^bb0(%carry: tensor<8xf32>, %x: tensor<8xbf16>):
    nv_tensor_ir.yield %x : tensor<8xbf16>
  }) : (tensor<8xf32>, tensor<5x8xbf16>) -> tensor<8xf32>
  results %final : tensor<8xf32>
}

// -----

nv_tensor_ir.graph @body_argument_count(%initial: tensor<8xf32>,
    %values: tensor<5x8xf32>) -> tensor<8xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@below {{body arguments must match the carry and sliced inputs}}
  %final = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64>
  }> ({
  ^bb0(%carry: tensor<8xf32>):
    nv_tensor_ir.yield %carry : tensor<8xf32>
  }) : (tensor<8xf32>, tensor<5x8xf32>) -> tensor<8xf32>
  results %final : tensor<8xf32>
}

// -----

nv_tensor_ir.graph @body_carry_type(%initial: tensor<8xf32>,
    %values: tensor<5x8xf32>) -> tensor<8xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@below {{body arguments must match the carry and sliced inputs}}
  %final = "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64>
  }> ({
  ^bb0(%carry: tensor<8xbf16>, %x: tensor<8xf32>):
    nv_tensor_ir.yield %x : tensor<8xf32>
  }) : (tensor<8xf32>, tensor<5x8xf32>) -> tensor<8xf32>
  results %final : tensor<8xf32>
}

// -----

nv_tensor_ir.graph @no_scanned_input(%initial: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error@below {{one carry and at least one scanned input}}
  %final = "nv_tensor_ir.scan_slice"(%initial) <{
    scan_dimensions = array<i64>, output_dimensions = array<i64>
  }> ({
  ^bb0(%carry: tensor<8xf32>):
    nv_tensor_ir.yield %carry : tensor<8xf32>
  }) : (tensor<8xf32>) -> tensor<8xf32>
  results %final : tensor<8xf32>
}

// -----

nv_tensor_ir.graph @no_results(%initial: tensor<8xf32>, %values: tensor<5x8xf32>) -> tensor<8xf32> {
  // expected-error@below {{at least one result}}
  "nv_tensor_ir.scan_slice"(%initial, %values) <{
    scan_dimensions = array<i64: 0>, output_dimensions = array<i64>
  }> ({
  ^bb0(%carry: tensor<8xf32>, %value: tensor<8xf32>):
    nv_tensor_ir.yield %carry : tensor<8xf32>
  }) : (tensor<8xf32>, tensor<5x8xf32>) -> ()
  results %initial : tensor<8xf32>
}
