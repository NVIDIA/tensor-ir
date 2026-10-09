// RUN: tensor_ir-opt %s -split-input-file -verify-diagnostics

nv_tensor_ir.graph @map_count(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{attribute 'indexing_maps' failed to satisfy constraint}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @map_type(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{attribute 'indexing_maps' failed to satisfy constraint}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [42 : i64,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @symbols(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{indexing map 0 must be a symbol-free projected permutation}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2)[s0] -> (d0, s0)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @duplicate(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{indexing map 0 must be a symbol-free projected permutation}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d0)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @expression(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{indexing map 0 must be a symbol-free projected permutation}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0 + d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @constant(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{indexing map 0 must be a symbol-free projected permutation}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @domain_rank(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{expects indexing maps to have the same domain rank}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @tensor_rank(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{indexing map 1 result count must match its tensor rank}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @result_rank(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7x3xf32> {
  // expected-error@below {{indexing map 2 result count must match its tensor rank}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7x3xf32>
  results %result : tensor<2x7x3xf32>
}

// -----

nv_tensor_ir.graph @unused_domain(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{domain dimension 3 must be a batch, product, or contraction dimension}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d2, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @one_sided_reduction(
    %a: tensor<2x5xf32>, %b: tensor<7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{domain dimension 2 must be a batch, product, or contraction dimension}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @result_only(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7x3xf32> {
  // expected-error@below {{domain dimension 3 must be a batch, product, or contraction dimension}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d2, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7x3xf32>
  results %result : tensor<2x7x3xf32>
}

// -----

nv_tensor_ir.graph @product_order(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<7x2xf32> {
  // expected-error@below {{expects C dimensions in batch, lhs product, rhs product order}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d1, d0)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<7x2xf32>
  results %result : tensor<7x2xf32>
}

// -----

nv_tensor_ir.graph @batch_order(
    %a: tensor<2x3x5xf32>, %b: tensor<2x5x7xf32>) -> tensor<3x2x7xf32> {
  // expected-error@below {{expects C dimensions in batch, lhs product, rhs product order}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d1, d0, d2)>]
  }> : (tensor<2x3x5xf32>, tensor<2x5x7xf32>) -> tensor<3x2x7xf32>
  results %result : tensor<3x2x7xf32>
}

// -----

nv_tensor_ir.graph @contracting_extent(
    %a: tensor<2x5x7xf32>, %b: tensor<5x8x3xf32>) -> tensor<2x3xf32> {
  // expected-error@below {{expects contraction domain dimension 3 of A and B to match}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>]
  }> : (tensor<2x5x7xf32>, tensor<5x8x3xf32>) -> tensor<2x3xf32>
  results %result : tensor<2x3xf32>
}

// -----

nv_tensor_ir.graph @lhs_extent(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<3x7xf32> {
  // expected-error@below {{expects C product dimension 0 to be inherited exactly from A}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<3x7xf32>
  results %result : tensor<3x7xf32>
}

// -----

nv_tensor_ir.graph @rhs_extent(
    %a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x8xf32> {
  // expected-error@below {{expects C product dimension 1 to be inherited exactly from B}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x8xf32>
  results %result : tensor<2x8xf32>
}

// -----

nv_tensor_ir.graph @batch_extent(
    %a: tensor<2x3x5xf32>, %b: tensor<4x5x7xf32>) -> tensor<2x3x7xf32> {
  // expected-error@below {{expects C batch dimension 0}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
  }> : (tensor<2x3x5xf32>, tensor<4x5x7xf32>) -> tensor<2x3x7xf32>
  results %result : tensor<2x3x7xf32>
}

// -----

nv_tensor_ir.graph @batch_result(
    %a: tensor<2x3x5xf32>, %b: tensor<1x5x7xf32>) -> tensor<1x3x7xf32> {
  // expected-error@below {{expects C batch dimension 0}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
  }> : (tensor<2x3x5xf32>, tensor<1x5x7xf32>) -> tensor<1x3x7xf32>
  results %result : tensor<1x3x7xf32>
}

// -----

nv_tensor_ir.graph @dynamic_product(
    %a: tensor<?x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{expects C product dimension 0}}
  %result = "nv_tensor_ir.contraction"(%a, %b) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<?x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @acc_shape(
    %a: tensor<2x5xf16>, %b: tensor<5x7xf16>,
    %acc: tensor<3x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{expects accumulator type to match result type}}
  %result = "nv_tensor_ir.contraction"(%a, %b, %acc) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf16>, tensor<5x7xf16>, tensor<3x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @acc_element(
    %a: tensor<2x5xf16>, %b: tensor<5x7xf16>,
    %acc: tensor<2x7xf16>) -> tensor<2x7xf32> {
  // expected-error@below {{expects accumulator type to match result type}}
  %result = "nv_tensor_ir.contraction"(%a, %b, %acc) <{
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>]
  }> : (tensor<2x5xf16>, tensor<5x7xf16>, tensor<2x7xf16>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}

// -----

nv_tensor_ir.graph @duplicate_domain_name(%a: tensor<2x5xf32>, %b: tensor<5x7xf32>) -> tensor<2x7xf32> {
  // expected-error@below {{duplicate indexing-map dimension m}}
  %result = contraction(%a, %b) maps (m, n, m) -> (m, m), (m, n), (m, n)
      : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
  results %result : tensor<2x7xf32>
}
