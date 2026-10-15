// RUN: tensor_ir-opt %s \
// RUN:   --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides),test-layout-normalization)' \
// RUN:   --split-input-file --verify-diagnostics -o /dev/null

// All graph results are streaming: no result supplies the iteration carrier.
// expected-error@below {{scan_slice graph requires a non-streaming result carrying its layout}}
nv_tensor_ir.graph @history_only(%initial: tensor<8xf32>,
    %values: tensor<5x8xf32>) -> tensor<5x8xf32> {
  %final, %history = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = [0]
      : (tensor<8xf32>, tensor<5x8xf32>) {
    ^bb0(%carry: tensor<8xf32>, %value: tensor<8xf32>):
      %next = add %carry, %value : tensor<8xf32>
      yield %next, %next : tensor<8xf32>, tensor<8xf32>
  }
  results %history : tensor<5x8xf32>
}

// -----

// Two individually valid scans are unsupported in one layout-propagated graph.
// expected-error@below {{layout propagation supports at most one top-level scan_slice}}
nv_tensor_ir.graph @two_scans(%initial: tensor<8xf32>,
    %values: tensor<5x8xf32>) -> (tensor<8xf32>, tensor<5x8xf32>) {
  %final, %history = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = [0]
      : (tensor<8xf32>, tensor<5x8xf32>) {
    ^bb0(%carry: tensor<8xf32>, %value: tensor<8xf32>):
      %next = add %carry, %value : tensor<8xf32>
      yield %next, %next : tensor<8xf32>, tensor<8xf32>
  }
    %other = scan_slice(%final, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<8xf32>, tensor<5x8xf32>) {
    ^bb0(%carry: tensor<8xf32>, %value: tensor<8xf32>):
      yield %carry : tensor<8xf32>
  }
  results %other, %history : tensor<8xf32>, tensor<5x8xf32>
}

// -----

// An unrelated ordinary result cannot supply the scan_slice layout for its history.
// expected-error@below {{cannot recover normalized scan_slice layout from graph results}}
nv_tensor_ir.graph @unrelated_result(%initial: tensor<8xf32>,
    %values: tensor<5x8xf32>) -> (tensor<8xf32>, tensor<5x8xf32>) {
  %final, %history = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = [0]
      : (tensor<8xf32>, tensor<5x8xf32>) {
    ^bb0(%carry: tensor<8xf32>, %value: tensor<8xf32>):
      %next = add %carry, %value : tensor<8xf32>
      yield %next, %next : tensor<8xf32>, tensor<8xf32>
  }
  results %initial, %history : tensor<8xf32>, tensor<5x8xf32>
}
