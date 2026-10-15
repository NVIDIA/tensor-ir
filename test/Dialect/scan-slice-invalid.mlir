// RUN: tensor_ir-opt %s --verify-diagnostics --split-input-file

// A scan_slice body argument is not a graph argument. Diagnose nesting explicitly
// before checking the inner scan_slice's input provenance.
nv_tensor_ir.graph @scan_slice_rejects_nested_scan_slice(
    %initial: tensor<8x8xf32>, %values: tensor<5x8x8xf32>)
    -> tensor<8x8xf32> {
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<8x8xf32>, tensor<5x8x8xf32>) {
    ^bb0(%carry: tensor<8x8xf32>, %value: tensor<8x8xf32>):
      // expected-error@below {{must be directly inside a graph or a bufferized program; nested scans are not supported}}
      %inner = scan_slice(%carry, %value)
          scan_dimensions = [0] output_dimensions = []
          : (tensor<8x8xf32>, tensor<8x8xf32>) {
        ^bb0(%inner_carry: tensor<8x8xf32>, %slice: tensor<8xf32>):
          yield %inner_carry : tensor<8x8xf32>
      }
      yield %inner : tensor<8x8xf32>
  }
  results %final : tensor<8x8xf32>
}

// -----

// An ordinary function is not a graph or a bufferized program.
func.func @scan_slice_rejects_non_graph_parent(%initial: tensor<8xf32>,
                                       %values: tensor<5x8xf32>) -> tensor<8xf32> {
  // expected-error@below {{must be directly inside a graph or a bufferized program}}
  %final = nv_tensor_ir.scan_slice(%initial, %values)
      scan_dimensions = [0] output_dimensions = []
      : (tensor<8xf32>, tensor<5x8xf32>) {
    ^bb0(%carry: tensor<8xf32>, %value: tensor<8xf32>):
      yield %carry : tensor<8xf32>
  }
  return %final : tensor<8xf32>
}

// -----

nv_tensor_ir.graph @scan_slice_rejects_non_argument_input(
    %values: tensor<5x2x8xf32>, %initial: tensor<2x8xf32>
  ) -> tensor<2x8xf32> {
  %derived = nv_tensor_ir.add %values, %values : tensor<5x2x8xf32>
  // expected-error@below{{scanned inputs must be graph arguments}}
  %final = "nv_tensor_ir.scan_slice"(%initial, %derived) <{
      output_dimensions = array<i64>,
      reverse = false,
      scan_dimensions = array<i64: 0>
    }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    %next = nv_tensor_ir.add %carry, %value : tensor<2x8xf32>
    nv_tensor_ir.yield %next : tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> tensor<2x8xf32>
  results %final : tensor<2x8xf32>
}

// -----

nv_tensor_ir.graph @scan_slice_rejects_capture(
    %values: tensor<5x2x8xf32>, %initial: tensor<2x8xf32>
  ) -> tensor<2x8xf32> {
  // expected-note@below {{required by region isolation constraints}}
  %final = "nv_tensor_ir.scan_slice"(%initial, %values) <{
      output_dimensions = array<i64>,
      reverse = false,
      scan_dimensions = array<i64: 0>
    }> ({
  ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
    // expected-error@below {{using value defined outside the region}}
    %next = nv_tensor_ir.add %carry, %initial : tensor<2x8xf32>
    nv_tensor_ir.yield %next : tensor<2x8xf32>
  }) : (tensor<2x8xf32>, tensor<5x2x8xf32>) -> tensor<2x8xf32>
  results %final : tensor<2x8xf32>
}
