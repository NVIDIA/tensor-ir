// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics --test-shape-relations -o /dev/null

nv_tensor_ir.graph @flatten_volumes(
    %a0: tensor<?x2xf32>, %b0: tensor<?x4xf32>,
    %a1: tensor<?x2xf32>, %b1: tensor<?x3xf32>,
    %a2: tensor<?x0xf32>, %b2: tensor<?x0xf32>) -> tensor<?xf32> {
  %a0_flat = reshape %a0 : tensor<?x2xf32> -> tensor<?xf32>
  %b0_flat = reshape %b0 : tensor<?x4xf32> -> tensor<?xf32>
  %out0 = add %a0_flat, %b0_flat : tensor<?xf32>
  %a0_size = nv_tensor_ir.dim %a0 dimension = 0 : tensor<?x2xf32>
  %b0_size = nv_tensor_ir.dim %b0 dimension = 0 : tensor<?x4xf32>
  %a0_scale = arith.constant 2 : index
  %b0_scale = arith.constant 4 : index
  %a0_volume = arith.muli %a0_size, %a0_scale : index
  %b0_volume = arith.muli %b0_size, %b0_scale : index
  // expected-remark @below {{volume_equal_2_4: true}}
  %equal0 = arith.cmpi eq, %a0_volume, %b0_volume {test.shape_query = "volume_equal_2_4"} : index
  %a1_flat = reshape %a1 : tensor<?x2xf32> -> tensor<?xf32>
  %b1_flat = reshape %b1 : tensor<?x3xf32> -> tensor<?xf32>
  %out1 = add %a1_flat, %b1_flat : tensor<?xf32>
  %a1_size = nv_tensor_ir.dim %a1 dimension = 0 : tensor<?x2xf32>
  %b1_size = nv_tensor_ir.dim %b1 dimension = 0 : tensor<?x3xf32>
  %a1_scale = arith.constant 2 : index
  %b1_scale = arith.constant 3 : index
  %a1_volume = arith.muli %a1_size, %a1_scale : index
  %b1_volume = arith.muli %b1_size, %b1_scale : index
  // expected-remark @below {{volume_equal_2_3: true}}
  %equal1 = arith.cmpi eq, %a1_volume, %b1_volume {test.shape_query = "volume_equal_2_3"} : index
  %a2_flat = reshape %a2 : tensor<?x0xf32> -> tensor<?xf32>
  %b2_flat = reshape %b2 : tensor<?x0xf32> -> tensor<?xf32>
  %out2 = add %a2_flat, %b2_flat : tensor<?xf32>
  %a2_size = nv_tensor_ir.dim %a2 dimension = 0 : tensor<?x0xf32>
  %b2_size = nv_tensor_ir.dim %b2 dimension = 0 : tensor<?x0xf32>
  %a2_scale = arith.constant 0 : index
  %b2_scale = arith.constant 0 : index
  %a2_volume = arith.muli %a2_size, %a2_scale : index
  %b2_volume = arith.muli %b2_size, %b2_scale : index
  // expected-remark @below {{volume_equal_0_0: true}}
  %equal2 = arith.cmpi eq, %a2_volume, %b2_volume {test.shape_query = "volume_equal_0_0"} : index
  // expected-remark @below {{dynamic_axes_equal_0_0: unknown}}
  %axes = arith.cmpi eq, %a2_size, %b2_size {test.shape_query = "dynamic_axes_equal_0_0"} : index
  results %out0 : tensor<?xf32>
}

// -----

// expected-remark @below {{argument 0: [3, 4]}}
// expected-remark @below {{argument 1: [12]}}
// expected-remark @below {{result 0: [12]}}
nv_tensor_ir.graph @scaled_constant(%a: tensor<?x4xf32>, %b: tensor<12xf32>) -> tensor<?xf32> attributes {test.print_signature} {
  %a_flat = reshape %a : tensor<?x4xf32> -> tensor<?xf32>
  %b_flat = reshape %b : tensor<12xf32> -> tensor<?xf32>
  %out = add %a_flat, %b_flat : tensor<?xf32>
  // expected-remark @below {{dimension: 3}}
  %a_size = nv_tensor_ir.dim %a dimension = 0 {test.shape_query = "dimension"} : tensor<?x4xf32>
  results %out : tensor<?xf32>
}

// -----

// expected-remark @below {{argument 0: [s0, 2]}}
// expected-remark @below {{argument 1: [2*s2 + 3]}}
// expected-remark @below {{argument 2: [s2]}}
// expected-remark @below {{result 0: [2*s2 + 3]}}
nv_tensor_ir.graph @division_offset_3(%a: tensor<?x2xf32>, %b: tensor<?xf32>, %root: tensor<?xf32>) -> tensor<?xf32> attributes {test.print_signature} {
  %n = nv_tensor_ir.dim %root dimension = 0 : tensor<?xf32>
  %two = arith.constant 2 : index
  %offset = arith.constant 3 : index
  %twice = arith.muli %n, %two : index
  %extent = arith.addi %twice, %offset : index
  nv_tensor_ir.assume_shape %b, [%extent] : tensor<?xf32>
  %a_flat = reshape %a : tensor<?x2xf32> -> tensor<?xf32>
  %out = add %a_flat, %b : tensor<?xf32>
  %a_size = nv_tensor_ir.dim %a dimension = 0 : tensor<?x2xf32>
  %a_volume = arith.muli %a_size, %two : index
  %b_size = nv_tensor_ir.dim %b dimension = 0 : tensor<?xf32>
  // expected-remark @below {{volume_equal: true}}
  %equal = arith.cmpi eq, %a_volume, %b_size {test.shape_query = "volume_equal"} : index
  // expected-remark @below {{fractional_binding: unknown}}
  %binding = arith.cmpi eq, %a_size, %n {test.shape_query = "fractional_binding"} : index
  results %out : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @division_offset_4(%a: tensor<?x2xf32>, %b: tensor<?xf32>, %root: tensor<?xf32>) -> tensor<?xf32> {
  %n = nv_tensor_ir.dim %root dimension = 0 : tensor<?xf32>
  %two = arith.constant 2 : index
  %offset = arith.constant 4 : index
  %twice = arith.muli %n, %two : index
  %extent = arith.addi %twice, %offset : index
  nv_tensor_ir.assume_shape %b, [%extent] : tensor<?xf32>
  %a_flat = reshape %a : tensor<?x2xf32> -> tensor<?xf32>
  %out = add %a_flat, %b : tensor<?xf32>
  %a_size = nv_tensor_ir.dim %a dimension = 0 : tensor<?x2xf32>
  %a_volume = arith.muli %a_size, %two : index
  %b_size = nv_tensor_ir.dim %b dimension = 0 : tensor<?xf32>
  // expected-remark @below {{volume_equal: true}}
  %equal = arith.cmpi eq, %a_volume, %b_size {test.shape_query = "volume_equal"} : index
  results %out : tensor<?xf32>
}

// -----

// expected-remark @below {{argument 0: [0, 2]}}
// expected-remark @below {{argument 1: [0, 2]}}
// expected-remark @below {{result 0: [0]}}
nv_tensor_ir.graph @acyclic_scaled_binding(%a: tensor<?x2xf32>, %b: tensor<?x2xf32>) -> tensor<?xf32> attributes {test.print_signature} {
  %n = nv_tensor_ir.dim %a dimension = 0 : tensor<?x2xf32>
  %two = arith.constant 2 : index
  %twice = arith.muli %n, %two : index
  nv_tensor_ir.assume_shape %b, [%twice, 2] : tensor<?x2xf32>
  %a_flat = reshape %a : tensor<?x2xf32> -> tensor<?xf32>
  %b_flat = reshape %b : tensor<?x2xf32> -> tensor<?xf32>
  %out = add %a_flat, %b_flat : tensor<?xf32>
  %b_size = nv_tensor_ir.dim %b dimension = 0 : tensor<?x2xf32>
  // expected-remark @below {{dynamic_axes_equal: true}}
  %equal = arith.cmpi eq, %n, %b_size {test.shape_query = "dynamic_axes_equal"} : index
  results %out : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @runtime_factors(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?xf32> {
  %shared = nv_tensor_ir.dim %a dimension = 1 : tensor<?x?xf32>
  %b0 = nv_tensor_ir.dim %b dimension = 0 : tensor<?x?xf32>
  nv_tensor_ir.assume_shape %b, [%b0, %shared] : tensor<?x?xf32>
  %a_flat = reshape %a : tensor<?x?xf32> -> tensor<?xf32>
  %b_flat = reshape %b : tensor<?x?xf32> -> tensor<?xf32>
  %out = add %a_flat, %b_flat : tensor<?xf32>
  %a0 = nv_tensor_ir.dim %a dimension = 0 : tensor<?x?xf32>
  %b1 = nv_tensor_ir.dim %b dimension = 1 : tensor<?x?xf32>
  // expected-remark @below {{dynamic_axes_equal: unknown}}
  %axes = arith.cmpi eq, %a0, %b0 {test.shape_query = "dynamic_axes_equal"} : index
  // expected-remark @below {{shared_axis_equal: true}}
  %shared_equal = arith.cmpi eq, %shared, %b1 {test.shape_query = "shared_axis_equal"} : index
  %a_volume = arith.muli %a0, %shared : index
  %b_volume = arith.muli %b0, %b1 : index
  // expected-remark @below {{volume_equal: true}}
  %volume = arith.cmpi eq, %a_volume, %b_volume {test.shape_query = "volume_equal"} : index
  results %out : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @contraction_dynamic_batch(%a: tensor<?x?x4xf32>, %b: tensor<?x4x5xf32>, %acc: tensor<?x?x5xf32>) -> tensor<?x?x5xf32> {
  %out = contraction(%a, %b) accum(%acc : tensor<?x?x5xf32>)
      maps (batch, m, n, k) -> (batch, m, k), (batch, k, n), (batch, m, n)
      : (tensor<?x?x4xf32>, tensor<?x4x5xf32>) -> tensor<?x?x5xf32>
  %a_batch = nv_tensor_ir.dim %a dimension = 0 : tensor<?x?x4xf32>
  %b_batch = nv_tensor_ir.dim %b dimension = 0 : tensor<?x4x5xf32>
  %out_batch = nv_tensor_ir.dim %out dimension = 0 : tensor<?x?x5xf32>
  // expected-remark @below {{operand_batches_equal: true}}
  %inputs = arith.cmpi eq, %a_batch, %b_batch {test.shape_query = "operand_batches_equal"} : index
  // expected-remark @below {{result_batch_equal: true}}
  %output = arith.cmpi eq, %a_batch, %out_batch {test.shape_query = "result_batch_equal"} : index
  %acc_batch = nv_tensor_ir.dim %acc dimension = 0 : tensor<?x?x5xf32>
  // expected-remark @below {{accumulator_batch_equal: true}}
  %acc_eq = arith.cmpi eq, %acc_batch, %out_batch {test.shape_query = "accumulator_batch_equal"} : index
  results %out : tensor<?x?x5xf32>
}

// -----

nv_tensor_ir.graph @contraction_unit_batch(%a: tensor<1x?x4xf32>, %b: tensor<?x4x5xf32>) -> tensor<1x?x5xf32> {
  %out = contraction(%a, %b)
      maps (batch, m, n, k) -> (batch, m, k), (batch, k, n), (batch, m, n)
      : (tensor<1x?x4xf32>, tensor<?x4x5xf32>) -> tensor<1x?x5xf32>
  // expected-remark @below {{dynamic_batch: 1}}
  %b_batch = nv_tensor_ir.dim %b dimension = 0 {test.shape_query = "dynamic_batch"} : tensor<?x4x5xf32>
  results %out : tensor<1x?x5xf32>
}

// -----

nv_tensor_ir.graph @contraction_empty_batch(%a: tensor<?x?x4xf32>, %b: tensor<0x4x5xf32>) -> tensor<0x?x5xf32> {
  %out = contraction(%a, %b)
      maps (batch, m, n, k) -> (batch, m, k), (batch, k, n), (batch, m, n)
      : (tensor<?x?x4xf32>, tensor<0x4x5xf32>) -> tensor<0x?x5xf32>
  // expected-remark @below {{dynamic_batch: 0}}
  %a_batch = nv_tensor_ir.dim %a dimension = 0 {test.shape_query = "dynamic_batch"} : tensor<?x?x4xf32>
  results %out : tensor<0x?x5xf32>
}

// -----

nv_tensor_ir.graph @contracted_domain_order(%a: tensor<4x?x?xf32>, %b: tensor<?x?x5xf32>, %roots: tensor<?x?x?xf32>) -> tensor<4x5xf32> {
  %n0 = nv_tensor_ir.dim %roots dimension = 0 : tensor<?x?x?xf32>
  %n1 = nv_tensor_ir.dim %roots dimension = 1 : tensor<?x?x?xf32>
  %n2 = nv_tensor_ir.dim %roots dimension = 2 : tensor<?x?x?xf32>
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %twice0 = arith.muli %n0, %two : index
  %twice2 = arith.muli %n2, %two : index
  %thrice1 = arith.muli %n1, %three : index
  nv_tensor_ir.assume_shape %a, [4, %twice0, %n1] : tensor<4x?x?xf32>
  nv_tensor_ir.assume_shape %b, [%twice2, %thrice1, 5] : tensor<?x?x5xf32>
  %out = contraction(%a, %b)
      maps (m, n, k0, k1) -> (m, k0, k1), (k1, k0, n), (m, n)
      : (tensor<4x?x?xf32>, tensor<?x?x5xf32>) -> tensor<4x5xf32>
  %a1 = nv_tensor_ir.dim %a dimension = 1 : tensor<4x?x?xf32>
  %a2 = nv_tensor_ir.dim %a dimension = 2 : tensor<4x?x?xf32>
  %b0 = nv_tensor_ir.dim %b dimension = 0 : tensor<?x?x5xf32>
  %b1 = nv_tensor_ir.dim %b dimension = 1 : tensor<?x?x5xf32>
  // expected-remark @below {{first_contracted_axis_equal: true}}
  %first = arith.cmpi eq, %a1, %b1 {test.shape_query = "first_contracted_axis_equal"} : index
  // expected-remark @below {{second_contracted_axis_equal: true}}
  %second = arith.cmpi eq, %a2, %b0 {test.shape_query = "second_contracted_axis_equal"} : index
  results %out : tensor<4x5xf32>
}

// -----

nv_tensor_ir.graph @query_api(%a: tensor<?x4xf32>, %b: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %a0 = nv_tensor_ir.dim %a dimension = 0 : tensor<?x4xf32>
  %a1 = nv_tensor_ir.dim %a dimension = 1 : tensor<?x4xf32>
  %b0 = nv_tensor_ir.dim %b dimension = 0 : tensor<?x4xf32>
  // expected-remark @below {{constant: 4}}
  %four = "arith.constant"() {value = 4 : index, test.shape_query = "constant"} : () -> index
  %five = arith.constant 5 : index
  // expected-remark @below {{unequal_constants: false}}
  %different = arith.cmpi eq, %four, %five {test.shape_query = "unequal_constants"} : index
  // expected-remark @below {{same_dimension: true}}
  %same = arith.cmpi eq, %a1, %four {test.shape_query = "same_dimension"} : index
  // expected-remark @below {{distinct_dimensions: unknown}}
  %unknown = arith.cmpi eq, %a0, %b0 {test.shape_query = "distinct_dimensions"} : index
  // expected-remark @below {{index_arithmetic: s0 + 4}}
  %sum = arith.addi %a0, %four {test.shape_query = "index_arithmetic"} : index
  // expected-remark @below {{exact_affine_division: 2*s0 + s1}}
  %exact = affine.apply affine_map<(d0, d1) -> ((4 * d0 + 2 * d1) floordiv 2)>(%a0, %b0) {test.shape_query = "exact_affine_division"}
  // expected-remark @below {{constant_affine_division: s0 + 1}}
  %constant_div = affine.apply affine_map<(d0)[s0] -> (d0 + s0 floordiv 3)>(%a0)[%four] {test.shape_query = "constant_affine_division"}
  %negative = arith.constant -4 : index
  // expected-remark @below {{negative_floor: s0 - 2}}
  %negative_floor = affine.apply affine_map<(d0)[s0] -> (d0 + s0 floordiv 3)>(%a0)[%negative] {test.shape_query = "negative_floor"}
  // expected-remark @below {{negative_ceil: s0 - 1}}
  %negative_ceil = affine.apply affine_map<(d0)[s0] -> (d0 + s0 ceildiv 3)>(%a0)[%negative] {test.shape_query = "negative_ceil"}
  // expected-remark @below {{negative_mod: s0 + 2}}
  %negative_mod = affine.apply affine_map<(d0)[s0] -> (d0 + s0 mod 3)>(%a0)[%negative] {test.shape_query = "negative_mod"}
  // expected-remark @below {{unsupported_affine_division: s8, symbols = {s8 = internal}}}
  %inexact = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%a0) {test.shape_query = "unsupported_affine_division", test.symbols}
  // expected-remark @below {{unsupported_index: s9, symbols = {s9 = internal}}}
  %unsupported = arith.remui %a0, %five {test.shape_query = "unsupported_index", test.symbols} : index
  results %a : tensor<?x4xf32>
}

// -----

nv_tensor_ir.graph @late_boundary_assumption(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?x4xf32> {
  %reshaped = reshape %a : tensor<?xf32> -> tensor<?x4xf32>
  %zero = arith.constant 0 : index
  %n = tensor.dim %b, %zero : tensor<?xf32>
  %four = arith.constant 4 : index
  %scaled = arith.muli %n, %four : index
  nv_tensor_ir.assume_shape %a, [%scaled] : tensor<?xf32>
  %rows = nv_tensor_ir.dim %reshaped dimension = 0 : tensor<?x4xf32>
  // expected-remark @below {{reshaped_axis_equal: true}}
  %equal = arith.cmpi eq, %rows, %n {test.shape_query = "reshaped_axis_equal"} : index
  results %reshaped : tensor<?x4xf32>
}

// -----

nv_tensor_ir.graph @scalar_assumption(%a: tensor<?xf32>, %n: index) -> tensor<?xf32> {
  nv_tensor_ir.assume_shape %a, [%n] : tensor<?xf32>
  %extent = nv_tensor_ir.dim %a dimension = 0 : tensor<?xf32>
  // expected-remark @below {{scalar_extent_equal: true}}
  %equal = arith.cmpi eq, %n, %extent {test.shape_query = "scalar_extent_equal"} : index
  %zero = arith.constant 0 : index
  // expected-remark @below {{scalar_query: s0}}
  %query = arith.addi %n, %zero {test.shape_query = "scalar_query"} : index
  results %a : tensor<?xf32>
}

// -----

// An out-of-range dimension query is an opaque value, which the promise on
// %b relates to %b's extent.
nv_tensor_ir.graph @out_of_range_tensor_dim(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
  %index = arith.constant 2 : index
  // expected-remark @below {{invalid_dimension: s1}}
  %invalid = tensor.dim {test.shape_query = "invalid_dimension"} %a, %index : tensor<?xf32>
  nv_tensor_ir.assume_shape %b, [%invalid] : tensor<?xf32>
  %zero = arith.constant 0 : index
  // expected-remark @below {{wrapped_invalid_dimension: s1}}
  %wrapped = arith.addi %invalid, %zero {test.shape_query = "wrapped_invalid_dimension"} : index
  nv_tensor_ir.assume_shape %b, [%wrapped] : tensor<?xf32>
  // expected-remark @below {{wrapped_invalid_equal: unknown}}
  %equal = arith.cmpi eq, %wrapped, %zero {test.shape_query = "wrapped_invalid_equal"} : index
  results %b : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @late_intermediate_assumption(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?x4xf32> {
  %reshaped = reshape %a : tensor<?xf32> -> tensor<?x4xf32>
  %intermediate = cos %a : tensor<?xf32>
  %n = nv_tensor_ir.dim %b dimension = 0 : tensor<?xf32>
  %four = arith.constant 4 : index
  %extent = arith.muli %n, %four : index
  nv_tensor_ir.assume_shape %intermediate, [%extent] : tensor<?xf32>
  %rows = nv_tensor_ir.dim %reshaped dimension = 0 : tensor<?x4xf32>
  // expected-remark @below {{reshaped_axis_equal: true}}
  %equal = arith.cmpi eq, %rows, %n {test.shape_query = "reshaped_axis_equal"} : index
  results %reshaped : tensor<?x4xf32>
}

// -----

// expected-remark @below {{argument 0: [s0]}}
// expected-remark @below {{result 0: [s0]}}
// expected-remark @below {{result 1: [4]}}
nv_tensor_ir.graph @value_bounds_dynamic_extents(%a: tensor<?xf32>, %condition: i1) -> (tensor<?xf32>, tensor<?xf32>) attributes {test.print_signature} {
  %n = nv_tensor_ir.dim %a dimension = 0 : tensor<?xf32>
  %selected = arith.select %condition, %n, %n : index
  %dynamic = iota dimension = 0 dynamic_dims(%selected) : tensor<?xf32>
  %four = arith.constant 4 : index
  %selected_constant = arith.select %condition, %four, %four : index
  %constant = iota dimension = 0 dynamic_dims(%selected_constant) : tensor<?xf32>
  %dynamic_extent = dim %dynamic dimension = 0 : tensor<?xf32>
  // expected-remark @below {{dynamic_extent_equal: true}}
  %equal = arith.cmpi eq, %dynamic_extent, %n {test.shape_query = "dynamic_extent_equal"} : index
  // expected-remark @below {{constant_extent: 4}}
  %constant_extent = dim %constant dimension = 0 {test.shape_query = "constant_extent"} : tensor<?xf32>
  results %dynamic, %constant : tensor<?xf32>, tensor<?xf32>
}

// -----

nv_tensor_ir.graph @query_order(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
  %b0 = nv_tensor_ir.dim %b dimension = 0 : tensor<?xf32>
  %two = arith.constant 2 : index
  %twice = arith.muli %b0, %two : index
  nv_tensor_ir.assume_shape %a, [%twice] : tensor<?xf32>
  // expected-remark @below {{first_extent: 2*s1}}
  %a0 = nv_tensor_ir.dim %a dimension = 0 {test.shape_query = "first_extent"} : tensor<?xf32>
  // expected-remark @below {{second_extent: s1}}
  %b_query = nv_tensor_ir.dim %b dimension = 0 {test.shape_query = "second_extent"} : tensor<?xf32>
  // expected-remark @below {{equivalent_extents: true}}
  %equal = arith.cmpi eq, %a0, %twice {test.shape_query = "equivalent_extents"} : index
  results %a : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @deferred_reshape_query(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?x4xf32> {
  %reshaped = reshape %a : tensor<?xf32> -> tensor<?x4xf32>
  %intermediate = cos %a : tensor<?xf32>
  %n = nv_tensor_ir.dim %b dimension = 0 : tensor<?xf32>
  %four = arith.constant 4 : index
  %extent = arith.muli %n, %four : index
  nv_tensor_ir.assume_shape %intermediate, [%extent] : tensor<?xf32>
  %rows = nv_tensor_ir.dim %reshaped dimension = 0 : tensor<?x4xf32>
  // expected-remark @below {{reshaped_axis_equal: true}}
  %row_equal = arith.cmpi eq, %rows, %n {test.shape_query = "reshaped_axis_equal"} : index
  %total = nv_tensor_ir.dim %a dimension = 0 : tensor<?xf32>
  // expected-remark @below {{total_extent_equal: true}}
  %total_equal = arith.cmpi eq, %total, %extent {test.shape_query = "total_extent_equal"} : index
  results %reshaped : tensor<?x4xf32>
}

// -----

func.func @sparse_symbol_maps(%a: tensor<?x?xf32>, %unused: tensor<?xf32>, %n: index) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %a, %c1 : tensor<?x?xf32>
  // expected-remark @below {{sparse: s1 + s3, symbols = {s1 = arg0.dim1, s3 = arg2}}}
  %sum = arith.addi %dim, %n {test.shape_query = "sparse", test.symbols} : index
  // expected-remark @below {{constant_symbols: 9, symbols = {}}}
  %c9 = arith.constant {test.shape_query = "constant_symbols", test.symbols} 9 : index
  // expected-remark @below {{other_component: s2, symbols = {s2 = arg1.dim0}}}
  %other = nv_tensor_ir.dim %unused dimension = 0 {test.shape_query = "other_component", test.symbols} : tensor<?xf32>
  // expected-remark @below {{repeated_sparse: s1 + s3, symbols = {s1 = arg0.dim1, s3 = arg2}}}
  %repeat = arith.addi %dim, %n {test.shape_query = "repeated_sparse", test.symbols} : index
  return %a : tensor<?x?xf32>
}

// -----

// expected-remark @below {{argument 0: [s0]}}
// expected-remark @below {{argument 1: [s1, 4]}}
// expected-remark @below {{argument 2: s2}}
// expected-remark @below {{result 0: [s1, 4]}}
// expected-remark @below {{result 1: [s1, 4]}}
// expected-remark @below {{result 2: s2}}
func.func @memref_arguments(%unused: memref<?xf32>, %buffer: memref<?x4xf32>, %n: index, %tag: i32) -> (memref<?x4xf32>, tensor<?x4xf32>, index, i32) attributes {test.print_signature} {
  %tensor = bufferization.to_tensor %buffer restrict : memref<?x4xf32> to tensor<?x4xf32>
  %c0 = arith.constant 0 : index
  // expected-remark @below {{tensor_extent: s1, symbols = {s1 = arg1.dim0}}}
  %td = tensor.dim {test.shape_query = "tensor_extent", test.symbols} %tensor, %c0 : tensor<?x4xf32>
  // expected-remark @below {{buffer_extent: s1, symbols = {s1 = arg1.dim0}}}
  %md = memref.dim {test.shape_query = "buffer_extent", test.symbols} %buffer, %c0 : memref<?x4xf32>
  // expected-remark @below {{same_extent: true}}
  %equal = arith.cmpi eq, %td, %md {test.shape_query = "same_extent"} : index
  return %buffer, %tensor, %n, %tag : memref<?x4xf32>, tensor<?x4xf32>, index, i32
}

// -----

func.func private @external_signature(memref<?xf32>, index) -> tensor<?xf32>

// -----

// expected-remark @below {{argument 0: []}}
// expected-remark @below {{result 0: []}}
func.func @rank_zero(%a: tensor<f32>) -> tensor<f32> attributes {test.print_signature} {
  return %a : tensor<f32>
}

// -----

// Both branches preserve the input extent in %same. The branch-dependent value
// in %different has no single polynomial equality with that extent.
// expected-remark @below {{argument 0: [s0]}}
// expected-remark @below {{result 0: [s0]}}
func.func @scf_if_extents(%a: tensor<?xf32>, %condition: i1) -> tensor<?xf32> attributes {test.print_signature} {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %a, %c0 : tensor<?xf32>
  %c1 = arith.constant 1 : index
  // expected-remark @below {{if_extent: s0, symbols = {s0 = arg0.dim0}}}
  %same = scf.if %condition -> index {
    scf.yield %n : index
  } else {
    %copy = arith.addi %n, %c0 : index
    scf.yield %copy : index
  } {test.shape_query = "if_extent", test.symbols}
  // expected-remark @below {{if_extent_equal: true}}
  %equal = arith.cmpi eq, %same, %n {test.shape_query = "if_extent_equal"} : index
  // expected-remark @below {{if_varying_extent: s3, symbols = {s3 = internal}}}
  %different = scf.if %condition -> index {
    scf.yield %n : index
  } else {
    %next = arith.addi %n, %c1 : index
    scf.yield %next : index
  } {test.shape_query = "if_varying_extent", test.symbols}
  // expected-remark @below {{if_varying_equal: unknown}}
  %unknown = arith.cmpi eq, %different, %n {test.shape_query = "if_varying_equal"} : index
  %out = nv_tensor_ir.iota dimension = 0 dynamic_dims(%same) : tensor<?xf32>
  %extent = nv_tensor_ir.dim %out dimension = 0 : tensor<?xf32>
  // expected-remark @below {{if_result_shape_equal: true}}
  %shape_equal = arith.cmpi eq, %extent, %n {test.shape_query = "if_result_shape_equal"} : index
  return %out : tensor<?xf32>
}

// -----

// Loop-carried scalars and tensor dimensions can remain equal to their initial
// values even with a dynamic trip count. An incremented extent is not invariant.
func.func @for_invariant_relations(%input: tensor<?x4xf32>, %n: index, %upper: index) -> tensor<?x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %initial = tensor.dim %input, %c0 : tensor<?x4xf32>
  // expected-remark @below {{invariant_scalar: s1, symbols = {s1 = arg1}}}
  %scalar = scf.for %iv = %c0 to %upper step %c1 iter_args(%iter = %n) -> index {
    scf.yield %iter : index
  } {test.shape_query = "invariant_scalar", test.symbols}
  // expected-remark @below {{invariant_scalar_equal: true}}
  %same_scalar = arith.cmpi eq, %scalar, %n {test.shape_query = "invariant_scalar_equal"} : index
  %tensor = scf.for %iv = %c0 to %upper step %c1 iter_args(%iter = %input) -> tensor<?x4xf32> {
    scf.yield %iter : tensor<?x4xf32>
  }
  // expected-remark @below {{invariant_extent: s0, symbols = {s0 = arg0.dim0}}}
  %dim = tensor.dim {test.shape_query = "invariant_extent", test.symbols} %tensor, %c0 : tensor<?x4xf32>
  // expected-remark @below {{invariant_shape_equal: true}}
  %same_shape = arith.cmpi eq, %dim, %initial {test.shape_query = "invariant_shape_equal"} : index
  // expected-remark @below {{changing_scalar: s9, symbols = {s9 = internal}}}
  %changing = scf.for %iv = %c0 to %upper step %c1 iter_args(%iter = %n) -> index {
    %next = arith.addi %iter, %c1 : index
    scf.yield %next : index
  } {test.shape_query = "changing_scalar", test.symbols}
  // expected-remark @below {{changing_scalar_equal: unknown}}
  %different_scalar = arith.cmpi eq, %changing, %n {test.shape_query = "changing_scalar_equal"} : index
  return %tensor : tensor<?x4xf32>
}

// -----

nv_tensor_ir.graph @mixed(%a: tensor<?xf32>) -> tensor<?xf32> {
  %c7 = arith.constant 7 : index
  %c3 = arith.constant 3 : index
  %a0 = nv_tensor_ir.dim %a dimension = 0 : tensor<?xf32>
  // expected-remark @below {{mixed_index: s0 + 3}}
  %r = affine.apply affine_map<(d0)[s0] -> (d0 + s0 floordiv 2)>(%a0)[%c7] {test.shape_query = "mixed_index"}
  %t = iota dimension = 0 dynamic_dims(%r) : tensor<?xf32>
  // expected-remark @below {{mixed_extent: s0 + 3}}
  %td = nv_tensor_ir.dim %t dimension = 0 {test.shape_query = "mixed_extent"} : tensor<?xf32>
  %expect = arith.addi %a0, %c3 : index
  // expected-remark @below {{mixed_extent_eq: true}}
  %eq = arith.cmpi eq, %td, %expect {test.shape_query = "mixed_extent_eq"} : index
  results %a : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @divisible_after_solve(%x: tensor<?xf32>, %condition: i1) -> tensor<?xf32> {
  %c = concatenate %x, %x dimension = 0 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %sz = nv_tensor_ir.dim %c dimension = 0 : tensor<?xf32>
  // expected-remark @below {{bounded_size: 2*s0}}
  %bounded = arith.select %condition, %sz, %sz {test.shape_query = "bounded_size"} : index
  // expected-remark @below {{half: s0}}
  %h = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%bounded) {test.shape_query = "half"}
  %t = iota dimension = 0 dynamic_dims(%h) : tensor<?xf32>
  %td = nv_tensor_ir.dim %t dimension = 0 : tensor<?xf32>
  %xd = nv_tensor_ir.dim %x dimension = 0 : tensor<?xf32>
  // expected-remark @below {{half_extent_eq: true}}
  %eq = arith.cmpi eq, %td, %xd {test.shape_query = "half_extent_eq"} : index
  results %c : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @transpose_extents(%a: tensor<?x4xf32>) -> tensor<4x?xf32> {
  %t = transpose %a permutation = [1, 0] : tensor<?x4xf32> -> tensor<4x?xf32>
  // expected-remark @below {{transposed: s0}}
  %t1 = nv_tensor_ir.dim %t dimension = 1 {test.shape_query = "transposed"} : tensor<4x?xf32>
  results %t : tensor<4x?xf32>
}

// -----

nv_tensor_ir.graph @broadcast_extents(%a: tensor<?x1xf32>, %n: index) -> tensor<?x?xf32> {
  %a0 = nv_tensor_ir.dim %a dimension = 0 : tensor<?x1xf32>
  %b = broadcast %a dynamic_dims(%a0, %n) : tensor<?x1xf32> -> tensor<?x?xf32>
  // expected-remark @below {{kept: s0}}
  %b0 = nv_tensor_ir.dim %b dimension = 0 {test.shape_query = "kept"} : tensor<?x?xf32>
  // expected-remark @below {{broadcast: s1}}
  %b1 = nv_tensor_ir.dim %b dimension = 1 {test.shape_query = "broadcast"} : tensor<?x?xf32>
  results %b : tensor<?x?xf32>
}

// -----

nv_tensor_ir.graph @concatenate_extents(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c = concatenate %a, %b dimension = 0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark @below {{concatenated: s0 + s2}}
  %c0 = nv_tensor_ir.dim %c dimension = 0 {test.shape_query = "concatenated"} : tensor<?x?xf32>
  %a1 = nv_tensor_ir.dim %a dimension = 1 : tensor<?x?xf32>
  %b1 = nv_tensor_ir.dim %b dimension = 1 : tensor<?x?xf32>
  // expected-remark @below {{other_equal: true}}
  %equal = arith.cmpi eq, %a1, %b1 {test.shape_query = "other_equal"} : index
  results %c : tensor<?x?xf32>
}

// -----

nv_tensor_ir.graph @slice_extents(%a: tensor<?xf32>) -> tensor<10xf32> {
  %s = slice %a starts = [0] limits = [10] strides = [1] : tensor<?xf32> -> tensor<10xf32>
  // expected-remark @below {{input_extent: s0}}
  %a0 = nv_tensor_ir.dim %a dimension = 0 {test.shape_query = "input_extent"} : tensor<?xf32>
  results %s : tensor<10xf32>
}

// -----

func.func @extract_slice_extents(%t: tensor<?x?xf32>, %n: index) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %s = tensor.extract_slice %t[0, 0] [1, %n] [1, 1] : tensor<?x?xf32> to tensor<?xf32>
  // expected-remark @below {{retained: s2}}
  %d = tensor.dim {test.shape_query = "retained"} %s, %c0 : tensor<?xf32>
  return %s : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @generator_extents(%x: f32, %n: index) -> tensor<?xf32> {
  %s = splat %x dynamic_dims(%n) : tensor<?xf32>
  %i = iota dimension = 0 dynamic_dims(%n) : tensor<?xf32>
  // expected-remark @below {{splat: s0}}
  %s0 = nv_tensor_ir.dim %s dimension = 0 {test.shape_query = "splat"} : tensor<?xf32>
  %i0 = nv_tensor_ir.dim %i dimension = 0 : tensor<?xf32>
  // expected-remark @below {{iota_equal: true}}
  %equal = arith.cmpi eq, %s0, %i0 {test.shape_query = "iota_equal"} : index
  results %s : tensor<?xf32>
}

// -----

func.func @materialize_extents(%src: tensor<?xf32>, %dest: tensor<?xf32>) -> tensor<?xf32> {
  %r = bufferization.materialize_in_destination %src in %dest : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %c0 = arith.constant 0 : index
  %s0 = tensor.dim %src, %c0 : tensor<?xf32>
  %d0 = tensor.dim %dest, %c0 : tensor<?xf32>
  %r0 = tensor.dim %r, %c0 : tensor<?xf32>
  // expected-remark @below {{source_destination_equal: true}}
  %sd = arith.cmpi eq, %s0, %d0 {test.shape_query = "source_destination_equal"} : index
  // expected-remark @below {{result_destination_equal: true}}
  %rd = arith.cmpi eq, %r0, %d0 {test.shape_query = "result_destination_equal"} : index
  return %r : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @matmul_extents(%a: tensor<?x4xf32>, %b: tensor<4x?xf32>) -> tensor<?x?xf32> {
  %m = matmul(%a, %b) : (tensor<?x4xf32>, tensor<4x?xf32>) -> tensor<?x?xf32>
  // expected-remark @below {{rows: s0}}
  %m0 = nv_tensor_ir.dim %m dimension = 0 {test.shape_query = "rows"} : tensor<?x?xf32>
  // expected-remark @below {{columns: s1}}
  %m1 = nv_tensor_ir.dim %m dimension = 1 {test.shape_query = "columns"} : tensor<?x?xf32>
  results %m : tensor<?x?xf32>
}

// -----

nv_tensor_ir.graph @reduce_extents(%a: tensor<?x?xf32>) -> (tensor<?x1xf32>, tensor<?xf32>) {
  %keep = nv_tensor_ir.reduce(%a)<dimensions = [1], reduction_mode = <add>> : tensor<?x?xf32> -> tensor<?x1xf32>
  %drop = nv_tensor_ir.reduce(%a)<dimensions = [1], reduction_mode = <add>> : tensor<?x?xf32> -> tensor<?xf32>
  // expected-remark @below {{kept_unit: s0}}
  %k0 = nv_tensor_ir.dim %keep dimension = 0 {test.shape_query = "kept_unit"} : tensor<?x1xf32>
  // expected-remark @below {{dropped: s0}}
  %d0 = nv_tensor_ir.dim %drop dimension = 0 {test.shape_query = "dropped"} : tensor<?xf32>
  results %keep, %drop : tensor<?x1xf32>, tensor<?xf32>
}

// -----

nv_tensor_ir.graph @reduce_ud_extents(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> (tensor<?x1xf32>, tensor<?x1xf32>) {
  %r:2 = reduce_ud(%a, %b)<dimensions = [1], identity = [0.0 : f32, 0.0 : f32]>
      (%lhs_a: f32, %lhs_b: f32, %rhs_a: f32, %rhs_b: f32) {
    nv_tensor_ir.yield %lhs_a, %lhs_b : f32, f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x1xf32>, tensor<?x1xf32>
  %a0 = nv_tensor_ir.dim %a dimension = 0 : tensor<?x?xf32>
  %b0 = nv_tensor_ir.dim %b dimension = 0 : tensor<?x?xf32>
  // expected-remark @below {{inputs_equal: true}}
  %equal = arith.cmpi eq, %a0, %b0 {test.shape_query = "inputs_equal"} : index
  // expected-remark @below {{result_rows: s0}}
  %r0 = nv_tensor_ir.dim %r#1 dimension = 0 {test.shape_query = "result_rows"} : tensor<?x1xf32>
  results %r#0, %r#1 : tensor<?x1xf32>, tensor<?x1xf32>
}

// -----

func.func @execute_region_extents(%a: tensor<?xf32>, %n: index) -> (tensor<?xf32>, index) {
  %t, %i = scf.execute_region -> (tensor<?xf32>, index) {
    scf.yield %a, %n : tensor<?xf32>, index
  }
  %c0 = arith.constant 0 : index
  %t0 = tensor.dim %t, %c0 : tensor<?xf32>
  %a0 = tensor.dim %a, %c0 : tensor<?xf32>
  // expected-remark @below {{region_tensor_equal: true}}
  %tensor_equal = arith.cmpi eq, %t0, %a0 {test.shape_query = "region_tensor_equal"} : index
  // expected-remark @below {{region_index: s1}}
  %index = arith.addi %i, %c0 {test.shape_query = "region_index"} : index
  return %t, %i : tensor<?xf32>, index
}

// -----

func.func @value_bound_shapes(%a: tensor<?x4xf32>, %condition: i1) -> (memref<?x4xf32>, tensor<?x4xf32>) {
  %buffer = bufferization.to_buffer %a : tensor<?x4xf32> to memref<?x4xf32>
  %c0 = arith.constant 0 : index
  // expected-remark @below {{buffer_rows: s0, symbols = {s0 = arg0.dim0}}}
  %buffer_rows = memref.dim {test.shape_query = "buffer_rows", test.symbols} %buffer, %c0 : memref<?x4xf32>
  %selected = scf.if %condition -> tensor<?x4xf32> {
    scf.yield %a : tensor<?x4xf32>
  } else {
    scf.yield %a : tensor<?x4xf32>
  }
  // expected-remark @below {{if_rows: s0, symbols = {s0 = arg0.dim0}}}
  %selected_rows = tensor.dim {test.shape_query = "if_rows", test.symbols} %selected, %c0 : tensor<?x4xf32>
  return %buffer, %selected : memref<?x4xf32>, tensor<?x4xf32>
}

// -----

// Equalities hold between values the models cannot relate to arguments.
nv_tensor_ir.graph @opaque_equalities(%a: tensor<?xf32>) -> tensor<?xf32> {
  %a0 = nv_tensor_ir.dim %a dimension = 0 : tensor<?xf32>
  %c5 = arith.constant 5 : index
  %u = arith.remui %a0, %c5 : index
  %v = arith.muli %u, %u : index
  %w = arith.muli %u, %u : index
  // expected-remark @below {{self_equal: true}}
  %self = arith.cmpi eq, %v, %v {test.shape_query = "self_equal"} : index
  // expected-remark @below {{squares_equal: true}}
  %equal = arith.cmpi eq, %v, %w {test.shape_query = "squares_equal"} : index
  // expected-remark @below {{squares_difference: 0}}
  %difference = arith.subi %v, %w {test.shape_query = "squares_difference"} : index
  results %a : tensor<?xf32>
}

// -----

// Dimension queries share the extent of the tensor they query, whichever
// query comes first.
func.func private @make_tensor() -> tensor<?xf32>
func.func @opaque_tensor_extent() -> tensor<?xf32> {
  %t = func.call @make_tensor() : () -> tensor<?xf32>
  %c0 = arith.constant 0 : index
  // expected-remark @below {{tensor_dim: s0, symbols = {s0 = internal.dim0}}}
  %d = tensor.dim {test.shape_query = "tensor_dim", test.symbols} %t, %c0 : tensor<?xf32>
  // expected-remark @below {{graph_dim: s0, symbols = {s0 = internal.dim0}}}
  %e = nv_tensor_ir.dim %t dimension = 0 {test.shape_query = "graph_dim", test.symbols} : tensor<?xf32>
  return %t : tensor<?xf32>
}

// -----

// A difference that is a rational multiple of an unsolved residual is zero.
nv_tensor_ir.graph @rational_residual(%a: tensor<?x?x2xf32>) -> tensor<?x4xf32> {
  %r = reshape %a : tensor<?x?x2xf32> -> tensor<?x4xf32>
  %a0 = nv_tensor_ir.dim %a dimension = 0 : tensor<?x?x2xf32>
  %a1 = nv_tensor_ir.dim %a dimension = 1 : tensor<?x?x2xf32>
  %r0 = nv_tensor_ir.dim %r dimension = 0 : tensor<?x4xf32>
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %p = arith.muli %a0, %a1 : index
  %lhs = arith.muli %p, %c3 : index
  %rhs = arith.muli %r0, %c6 : index
  // expected-remark @below {{rational_multiple: true}}
  %eq = arith.cmpi eq, %lhs, %rhs {test.shape_query = "rational_multiple"} : index
  // expected-remark @below {{negated_multiple: true}}
  %neg = arith.cmpi eq, %rhs, %lhs {test.shape_query = "negated_multiple"} : index
  results %r : tensor<?x4xf32>
}

// -----

// A divisor is the gcd of the coefficients of an extent; an upper bound holds
// over non-negative symbols.
nv_tensor_ir.graph @divisors_and_bounds(%a: tensor<?x4xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
  %r = reshape %a : tensor<?x4xf32> -> tensor<?xf32>
  %out = add %r, %b : tensor<?xf32>
  // expected-remark @below {{scaled: 4*s0, divisor = 4, upper_bound = none}}
  %scaled = nv_tensor_ir.dim %b dimension = 0 {test.shape_query = "scaled", test.bounds} : tensor<?xf32>
  %rows = nv_tensor_ir.dim %a dimension = 0 : tensor<?x4xf32>
  %c6 = arith.constant 6 : index
  // expected-remark @below {{bounded: -s0 + 6, divisor = 1, upper_bound = 6}}
  %bounded = arith.subi %c6, %rows {test.shape_query = "bounded", test.bounds} : index
  %c12 = arith.constant 12 : index
  // expected-remark @below {{constant: 18, divisor = 18, upper_bound = 18}}
  %constant = arith.addi %c12, %c6 {test.shape_query = "constant", test.bounds} : index
  results %out : tensor<?xf32>
}

// -----


// Integer arithmetic on other types relates only the shapes of its operands.
// expected-remark @below {{argument 0: [s0]}}
// expected-remark @below {{argument 1: [s0]}}
// expected-remark @below {{result 0: [s0]}}
func.func @non_index_arithmetic(%a: tensor<?xi32>, %b: tensor<?xi32>, %x: i32, %y: i32) -> (tensor<?xi32>, i32) attributes {test.print_signature} {
  %sum = arith.addi %x, %y : i32
  %product = arith.muli %a, %b : tensor<?xi32>
  return %product, %sum : tensor<?xi32>, i32
}
