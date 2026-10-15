// RUN: tensor_ir-opt %s \
// RUN:   --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid))' \
// RUN:   --split-input-file --verify-diagnostics

nv_tensor_ir.graph @unsupported_body_op(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32>)
    attributes {tile_size = array<i32: 1, 8>} {
  %final, %history = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = [0]
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %next = add %carry, %value : tensor<2x8xf32>
      // expected-error@below {{is not supported inside scan_slice}}
      %step = pow %next, %value : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
      yield %next, %step : tensor<2x8xf32>, tensor<2x8xf32>
  }
  results %final, %history : tensor<2x8xf32>, tensor<5x2x8xf32>
}

// -----

nv_tensor_ir.graph @non_add_reduction(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x1xf32>)
    attributes {tile_size = array<i32: 1, 8>} {
  %final, %history = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = [0]
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %next = add %carry, %value : tensor<2x8xf32>
      // expected-error@below {{scan_slice supports only a fully tiled innermost add reduction}}
      %step = reduce(%next) <dimensions = [1], reduction_mode = <max>> : tensor<2x8xf32> -> tensor<2x1xf32>
      yield %next, %step : tensor<2x8xf32>, tensor<2x1xf32>
  }
  results %final, %history : tensor<2x8xf32>, tensor<5x2x1xf32>
}

// -----

nv_tensor_ir.graph @outer_reduction(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x1x8xf32>)
    attributes {tile_size = array<i32: 2, 8>} {
  %final, %history = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = [0]
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %next = add %carry, %value : tensor<2x8xf32>
      // expected-error@below {{scan_slice supports only a fully tiled innermost add reduction}}
      %step = reduce(%next) <dimensions = [0], reduction_mode = <add>> : tensor<2x8xf32> -> tensor<1x8xf32>
      yield %next, %step : tensor<2x8xf32>, tensor<1x8xf32>
  }
  results %final, %history : tensor<2x8xf32>, tensor<5x1x8xf32>
}

// -----

// Broadcast the reduction back to the carry shape so TileSelection does not
// reject a projected result first; FormGrid must enforce the full reduction tile.
nv_tensor_ir.graph @partial_reduction_tile(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32>)
    attributes {tile_size = array<i32: 1, 4>} {
  %final, %history = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = [0]
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %next = add %carry, %value : tensor<2x8xf32>
      // expected-error@below {{scan_slice supports only a fully tiled innermost add reduction}}
      %reduced = reduce(%next) <dimensions = [1], reduction_mode = <add>> : tensor<2x8xf32> -> tensor<2x1xf32>
      %step = broadcast %reduced : tensor<2x1xf32> -> tensor<2x8xf32>
      yield %next, %step : tensor<2x8xf32>, tensor<2x8xf32>
  }
  results %final, %history : tensor<2x8xf32>, tensor<5x2x8xf32>
}

// -----

nv_tensor_ir.graph @strided_history(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x2x8xf32> {nv_tensor_ir.stride = "(64,32,2)"})
    attributes {tile_size = array<i32: 1, 8>} {
  // expected-error@below {{scan_slice side outputs require contiguous strides}}
  %final, %history = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = [0]
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %next = add %carry, %value : tensor<2x8xf32>
      %step = neg %next : tensor<2x8xf32>
      yield %next, %step : tensor<2x8xf32>, tensor<2x8xf32>
  }
  results %final, %history : tensor<2x8xf32>, tensor<5x2x8xf32>
}

// -----

// Multiple reduction axes remain unsupported even when both are fully tiled.
nv_tensor_ir.graph @multiple_reduction_axes(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> (tensor<2x8xf32>, tensor<5x1x1xf32>)
    attributes {tile_size = array<i32: 2, 8>} {
  %final, %history = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = [0]
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %next = add %carry, %value : tensor<2x8xf32>
      // expected-error@below {{scan_slice supports only a fully tiled innermost add reduction}}
      %step = reduce(%next) <dimensions = [0, 1], reduction_mode = <add>> : tensor<2x8xf32> -> tensor<1x1xf32>
      yield %next, %step : tensor<2x8xf32>, tensor<1x1xf32>
  }
  results %final, %history : tensor<2x8xf32>, tensor<5x1x1xf32>
}

// -----

// Equal element counts do not make a reshape compatible with the carry tile.
nv_tensor_ir.graph @nonunit_reshape(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> tensor<2x8xf32> attributes {tile_size = array<i32: 1, 8>} {
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      // expected-error@below {{scan_slice body reshape only supports inserting or removing unit dimensions}}
      %pairs = reshape %value : tensor<2x8xf32> -> tensor<8x2xf32>
      %sums = reduce(%pairs) <dimensions = [1], reduction_mode = <add>>
          : tensor<8x2xf32> -> tensor<8x1xf32>
      %row = reshape %sums : tensor<8x1xf32> -> tensor<1x8xf32>
      %wide = broadcast %row : tensor<1x8xf32> -> tensor<2x8xf32>
      %next = add %carry, %wide : tensor<2x8xf32>
      yield %next : tensor<2x8xf32>
  }
  results %final : tensor<2x8xf32>
}

// -----

// The reduction must cover the full logical dimension after unit-axis insertion.
nv_tensor_ir.graph @partial_unit_reshape_reduce(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> tensor<2x8xf32> attributes {tile_size = array<i32: 1, 4>} {
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %expanded = reshape %carry : tensor<2x8xf32> -> tensor<2x1x8xf32>
      // expected-error@below {{scan_slice supports only a fully tiled innermost add reduction}}
      %sum = reduce(%expanded) <dimensions = [2], reduction_mode = <add>>
          : tensor<2x1x8xf32> -> tensor<2x1x1xf32>
      %row = reshape %sum : tensor<2x1x1xf32> -> tensor<2x1xf32>
      %next = broadcast %row : tensor<2x1xf32> -> tensor<2x8xf32>
      yield %next : tensor<2x8xf32>
  }
  results %final : tensor<2x8xf32>
}

// -----

// Equal element counts do not establish a scan_slice input/carry axis mapping.
nv_tensor_ir.graph @nonunit_input_slice(%initial: tensor<2x8xf32>, %values: tensor<5x4x4xf32>)
    -> tensor<2x8xf32> attributes {tile_size = array<i32: 1, 8>} {
  // expected-error@below {{scan_slice input non-unit dimensions must match a prefix of the non-unit carry dimensions}}
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<2x8xf32>, tensor<5x4x4xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<4x4xf32>):
      yield %carry : tensor<2x8xf32>
  }
  results %final : tensor<2x8xf32>
}

// -----

// This intermediate broadcast needs a separate iteration-space mapping.
nv_tensor_ir.graph @noncarry_broadcast(%initial: tensor<2x8xf32>, %values: tensor<5x2x8xf32>)
    -> tensor<2x8xf32> attributes {tile_size = array<i32: 1, 8>} {
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %sum = reduce(%value) <dimensions = [1], reduction_mode = <add>>
          : tensor<2x8xf32> -> tensor<2x1xf32>
      %expanded = reshape %sum : tensor<2x1xf32> -> tensor<2x1x1xf32>
      // expected-error@below {{scan_slice body broadcast dimensions must be one or match the corresponding carry dimension}}
      %wide = broadcast %expanded : tensor<2x1x1xf32> -> tensor<2x1x4xf32>
      %total = reduce(%wide) <dimensions = [2], reduction_mode = <add>>
          : tensor<2x1x4xf32> -> tensor<2x1x1xf32>
      %row = reshape %total : tensor<2x1x1xf32> -> tensor<2x1xf32>
      %next = broadcast %row : tensor<2x1xf32> -> tensor<2x8xf32>
      yield %next : tensor<2x8xf32>
  }
  results %final : tensor<2x8xf32>
}

// -----

// Equal axis extents and tile sizes must not hide a row-to-column remapping.
nv_tensor_ir.graph @unit_reshape_axis_remap(%initial: tensor<8x8xf32>, %values: tensor<5x8xf32>)
    -> tensor<8x8xf32> attributes {tile_size = array<i32: 1, 1>} {
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<8x8xf32>, tensor<5x8xf32>) {
    ^bb0(%carry: tensor<8x8xf32>, %value: tensor<8xf32>):
      %row = reshape %value : tensor<8xf32> -> tensor<1x8xf32>
      // expected-error@below {{scan_slice body broadcast input axis must be -1 or match its dimension position}}
      %wide = broadcast %row : tensor<1x8xf32> -> tensor<8x8xf32>
      %next = add %carry, %wide : tensor<8x8xf32>
      yield %next : tensor<8x8xf32>
  }
  results %final : tensor<8x8xf32>
}

// -----

// Even a unit-only shifted profile is not a positional broadcast. Previously
// matching the whole output profile up to unit dimensions accepted this case.
nv_tensor_ir.graph @shifted_broadcast_profile(%initial: tensor<2x8xf32>, %values: tensor<5x2xf32>)
    -> tensor<2x8xf32> attributes {tile_size = array<i32: 1, 8>} {
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<2x8xf32>, tensor<5x2xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2xf32>):
      %shifted = reshape %value : tensor<2xf32> -> tensor<1x2x1xf32>
      // expected-error@below {{scan_slice body broadcast input axis must be -1 or match its dimension position}}
      %wide = broadcast %shifted : tensor<1x2x1xf32> -> tensor<1x2x8xf32>
      %flat = reshape %wide : tensor<1x2x8xf32> -> tensor<2x8xf32>
      %next = add %carry, %flat : tensor<2x8xf32>
      yield %next : tensor<2x8xf32>
  }
  results %final : tensor<2x8xf32>
}
