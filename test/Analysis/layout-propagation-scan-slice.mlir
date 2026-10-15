// RUN: tensor_ir-opt %s \
// RUN:   --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides),test-layout-propagation)' \
// RUN:   --split-input-file -o /dev/null \
// RUN:   | FileCheck %s --check-prefix=ANNOTATE
// RUN: tensor_ir-opt %s \
// RUN:   --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides),test-layout-normalization)' \
// RUN:   --split-input-file -o /dev/null \
// RUN:   | FileCheck %s

// Streaming outputs do not select the carrier; their projected views retain ABI order.
// NEXT checks show that scan_slice body operations are not analyzed.
// ANNOTATE-LABEL: layout propagation for @streaming_metadata
//   ANNOTATE-NEXT: nv_tensor_ir.neg: #nv_tensor_ir.tensor_source<0, 0, "(2,8):(8,1)">
//   ANNOTATE-NEXT: nv_tensor_ir.scan_slice: [[SOURCE:#nv_tensor_ir.scan_slice_source<carry =]]
//    ANNOTATE-SAME: #nv_tensor_ir.tensor_source<0, 0, "(2,8):(8,1)">, output_dims = [1, 0], inputs = [
//    ANNOTATE-SAME: #nv_tensor_ir.scan_slice_input<source =
//    ANNOTATE-SAME: #nv_tensor_ir.tensor_source<1, 0, "(2,8):(16,2)">, dim = 0, extent = 5, stride = 64>,
//    ANNOTATE-SAME: #nv_tensor_ir.scan_slice_input<source =
//    ANNOTATE-SAME: #nv_tensor_ir.tensor_source<2, 0, "(2):(5)">, dim = 1, extent = 5, stride = 1>]
//   ANNOTATE-NEXT: nv_tensor_ir.exp: #nv_tensor_ir.scan_slice_source<carry =

//    CHECK-LABEL: layout normalization for @streaming_metadata
//     CHECK-NEXT: iteration_space: [[SOURCE:#nv_tensor_ir.scan_slice_source<carry = #nv_tensor_ir.tensor_source<0, 0, "\(2,8\):\(8,1\)">, output_dims = \[1, 0\], inputs = \[.*\]>]]{{$}}
//     CHECK-NEXT: result_layouts: {{\[}}[[SOURCE]], [[SOURCE]], [[SOURCE]]{{\]}}
//     CHECK-NEXT: result_views: [#nv_tensor_ir.tensor_source<0, 0, "(2,8):(40,1)">,
//     CHECK-SAME: #nv_tensor_ir.tensor_source<1, 0, "(2,8):(1,0)">,
//     CHECK-SAME: #nv_tensor_ir.tensor_source<5, 0, "(2,8):(8,1)">]

nv_tensor_ir.graph @streaming_metadata(
    %initial: tensor<2x8xf32>,
    %values: tensor<5x2x8xf32> {nv_tensor_ir.stride = "(64,16,2)"},
    %scales: tensor<2x5xf32>)
    -> (tensor<2x5x8xbf16>, tensor<5x2x1xf32>, tensor<2x8xf32>)
    attributes {tile_size = array<i32: 1, 8>} {
  %initial_carry = neg %initial : tensor<2x8xf32>
  %final, %history, %totals = scan_slice(%initial_carry, %values, %scales)
      scan_dimensions = [0, 1] output_dimensions = [1, 0] {reverse = true}
      : (tensor<2x8xf32>, tensor<5x2x8xf32>, tensor<2x5xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>, %scale: tensor<2xf32>):
      %column = reshape %scale : tensor<2xf32> -> tensor<2x1xf32>
      %wide = broadcast %column : tensor<2x1xf32> -> tensor<2x8xf32>
      %scaled = mul %value, %wide : tensor<2x8xf32>
      %next = add %carry, %scaled : tensor<2x8xf32>
      %low = convert %next : tensor<2x8xf32> -> tensor<2x8xbf16>
      %total = reduce(%next) <dimensions = [1], reduction_mode = <add>>
          : tensor<2x8xf32> -> tensor<2x1xf32>
      yield %next, %low, %total : tensor<2x8xf32>, tensor<2x8xbf16>, tensor<2x1xf32>
  }
  %out = exp %final : tensor<2x8xf32>
  results %history, %totals, %out : tensor<2x5x8xbf16>, tensor<5x2x1xf32>, tensor<2x8xf32>
}

// -----

// A composite initial carry, no history, and a shared final carry consumer.
// ANNOTATE-LABEL: layout propagation for @composite_carry
//   ANNOTATE-NEXT: nv_tensor_ir.add: #nv_tensor_ir.composite_source<
//   ANNOTATE-NEXT: nv_tensor_ir.scan_slice: #nv_tensor_ir.scan_slice_source<carry = #nv_tensor_ir.composite_source<
//    ANNOTATE-SAME: output_dims = [], inputs = [#nv_tensor_ir.scan_slice_input<source =
//    ANNOTATE-SAME: #nv_tensor_ir.tensor_source<2, 0, "(2,8):(8,1)">, dim = 0, extent = 5, stride = 16>]
//   ANNOTATE-NEXT: nv_tensor_ir.add: #nv_tensor_ir.scan_slice_source<carry =

//    CHECK-LABEL: layout normalization for @composite_carry
//     CHECK-NEXT: iteration_space: #nv_tensor_ir.scan_slice_source<carry = [[CARRY:#nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "\(2,8\):\(8,1\)">, #nv_tensor_ir.tensor_source<1, 0, "\(2,8\):\(8,1\)">>]],
//     CHECK-SAME: output_dims = [], inputs = [#nv_tensor_ir.scan_slice_input<source =
//     CHECK-SAME: #nv_tensor_ir.tensor_source<2, 0, "(2,8):(8,1)">, dim = 0, extent = 5, stride = 16>]>
//     CHECK-NEXT: result_layouts: [#nv_tensor_ir.scan_slice_source<carry = [[CARRY]]
//     CHECK-NEXT: result_views: [#nv_tensor_ir.tensor_source<3, 0, "(2,8):(8,1)">]

nv_tensor_ir.graph @composite_carry(%a: tensor<2x8xf32>,
    %b: tensor<2x8xf32>, %values: tensor<5x2x8xf32>) -> tensor<2x8xf32>
    attributes {tile_size = array<i32: 1, 8>} {
  %initial = add %a, %b : tensor<2x8xf32>
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<2x8xf32>, tensor<5x2x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x8xf32>):
      %next = add %carry, %value : tensor<2x8xf32>
      yield %next : tensor<2x8xf32>
  }
  %out = add %final, %final : tensor<2x8xf32>
  results %out : tensor<2x8xf32>
}

// -----

// Neither annotation nor normalization changes the native input slice profile.
// ANNOTATE-LABEL: layout propagation for @native_slice_profile
//      ANNOTATE: nv_tensor_ir.scan_slice:
// ANNOTATE-SAME: inputs = [#nv_tensor_ir.scan_slice_input<source = #nv_tensor_ir.tensor_source<1, 0, "(2,1,8):(8,8,1)">,
// ANNOTATE-SAME: dim = 0, extent = 5, stride = 16>]
//   CHECK-LABEL: layout normalization for @native_slice_profile
//         CHECK: iteration_space:
//    CHECK-SAME: inputs = [#nv_tensor_ir.scan_slice_input<source = #nv_tensor_ir.tensor_source<1, 0, "(2,1,8):(8,8,1)">,
//    CHECK-SAME: dim = 0, extent = 5, stride = 16>]
nv_tensor_ir.graph @native_slice_profile(%initial: tensor<2x8xf32>, %values: tensor<5x2x1x8xf32>)
    -> tensor<2x8xf32> {
  %final = scan_slice(%initial, %values) scan_dimensions = [0] output_dimensions = []
      : (tensor<2x8xf32>, tensor<5x2x1x8xf32>) {
    ^bb0(%carry: tensor<2x8xf32>, %value: tensor<2x1x8xf32>):
      %flat = reshape %value : tensor<2x1x8xf32> -> tensor<2x8xf32>
      %next = add %carry, %flat : tensor<2x8xf32>
      yield %next : tensor<2x8xf32>
  }
  results %final : tensor<2x8xf32>
}
