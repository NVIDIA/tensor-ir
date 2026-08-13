// RUN: tensor_ir-opt -verify-diagnostics -split-input-file %s

// Tests for layout propagation attribute verifiers.

//===----------------------------------------------------------------------===//
// TensorSourceAttr invalid tests
//===----------------------------------------------------------------------===//

// Invalid layout string format.
func.func @invalid_layout_string(%in0: tensor<16xf32>)
    // expected-error @+1 {{Invalid layout: not_a_layout}}
    attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "not_a_layout">} {
  return
}

// -----

// Layout with depth > 2 is unsupported.
func.func @unsupported_layout_depth(%in0: tensor<16xf32>)
    // expected-error @+1 {{Unsupported layout}}
    attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(((2,2),2),2):(((1,2),4),8)">} {
  return
}

// -----
// Number of dynamic values should match the number of dynamic sizes.
func.func @incorrect_dynamic_offsets(%in0: tensor<?x?xf32>)
    // expected-error @+1 {{Layout has 3 dynamic size(s) but 2 dynamic offset(s) provided}}
    attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(1,?)", [0, 1]>} {
  return
}

// -----

// Dynamic offsets must be a permutation of iota.
func.func @incorrect_dynamic_offsets(%in0: tensor<?x?xf32>)
    // expected-error @+1 {{Dynamic offsets must be in the range [0..2) and unique}}
    attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,64):(1,?)", [1, 2]>} {
  return
}

// -----

//===----------------------------------------------------------------------===//
// CompositeSourceAttr invalid tests
//===----------------------------------------------------------------------===//

// Sources with different shapes.
func.func @composite_different_shapes(%in0: tensor<16xf32>, %in1: tensor<8xf32>)
    // expected-error @+1 {{Underlying sources must have the same shape}}
    attributes {layout = #nv_tensor_ir.composite_source<
      #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">,
      #nv_tensor_ir.tensor_source<1, 0, "(8):(1)">
    >} {
  return
}

// -----

//===----------------------------------------------------------------------===//
// ConcatSourceAttr invalid tests
//===----------------------------------------------------------------------===//

func.func @concat_dynamic_dimension(%in0: tensor<?xf32>, %in1: tensor<?xf32>)
    // expected-error @+1 {{Dynamic shapes for concatenation are not supported}}
    attributes {layout = #nv_tensor_ir.concat_source<dim = 0,
      #nv_tensor_ir.tensor_source<0, 0, "(?):(1)", [0]>,
      #nv_tensor_ir.tensor_source<1, 0, "(?):(1)", [0]>
    >} {
  return
}

// -----

func.func @concat_invalid_dimension(%in0: tensor<16xf32>, %in1: tensor<8xf32>)
    // expected-error @+1 {{Concatenation dimension is invalid}}
    attributes {layout = #nv_tensor_ir.concat_source<dim = 1,
      #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">,
      #nv_tensor_ir.tensor_source<1, 0, "(8):(1)">
    >} {
  return
}

// -----

func.func @concat_different_shapes(%in0: tensor<16x2xf32>, %in1: tensor<8x2xf32>)
    // expected-error @+1 {{Concatenated sources must have the same shape, except for the concatenation dimension}}
    attributes {layout = #nv_tensor_ir.concat_source<dim = 1,
      #nv_tensor_ir.tensor_source<0, 0, "(16,2):(1,16)">,
      #nv_tensor_ir.tensor_source<1, 0, "(8,2):(1,8)">
    >} {
  return
}

// -----

//===----------------------------------------------------------------------===//
// ReductionSourceAttr invalid tests
//===----------------------------------------------------------------------===//

// Invalid layout string format.
func.func @reduction_invalid_view(%in0: tensor<16x32xf32>)
    // expected-error @+1 {{Invalid view: not_a_layout}}
    attributes {layout = #nv_tensor_ir.reduction_source<"not_a_layout",
      #nv_tensor_ir.tensor_source<0, 0, "(16,32):(1,16)">>} {
  return
}

// -----

// Dynamic dimensions are not supported.
func.func @reduction_dynamic_dimension(%in0: tensor<?x32xf32>)
    // expected-error @+1 {{Reduction view must have static shape and stride}}
    attributes {layout = #nv_tensor_ir.reduction_source<"(?,(32)):(32,(1))",
      #nv_tensor_ir.tensor_source<0, 0, "(?,32):(32,1)", [0]>>} {
  return
}

// -----

// Invalid view layout (non-contiguous).
func.func @reduction_invalid_view(%in0: tensor<16x32xf32>)
    // expected-error @+1 {{Invalid view for the source: (16,(32)):(1,(32))}}
    attributes {layout = #nv_tensor_ir.reduction_source<"(16,(32)):(1,(32))",
      #nv_tensor_ir.tensor_source<0, 0, "(16,32):(1,16)">>} {
  return
}

// -----

// Invalid view layout (different shape).
func.func @reduction_invalid_view(%in0: tensor<16x32xf32>)
    // expected-error @+1 {{Invalid view for the source: (16,(16)):(1,(16))}}
    attributes {layout = #nv_tensor_ir.reduction_source<"(16,(16)):(1,(16))",
      #nv_tensor_ir.tensor_source<0, 0, "(16,32):(1,16)">>} {
  return
}

// -----

//===----------------------------------------------------------------------===//
// MatmulSourceAttr invalid tests
//===----------------------------------------------------------------------===//

// Invalid layout string format.
func.func @matmul_invalid_view(%in0: tensor<2x8xf32>, %in1: tensor<8x4xf32>)
    // expected-error @+1 {{Invalid view: not_a_layout}}
    attributes {layout = #nv_tensor_ir.matmul_source<"not_a_layout", 1, 2, 4, 8,
      #nv_tensor_ir.tensor_source<0, 0, "(2,8):(1,2)">,
      #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>} {
  return
}

// -----

// Dynamic LHS dimensions are not supported.
func.func @matmul_dynamic_lhs(%in0: tensor<?x8xf32>, %in1: tensor<8x4xf32>)
    // expected-error @+1 {{Matmul view must have static shape and stride}}
    attributes {layout = #nv_tensor_ir.matmul_source<"(?,4,8):(0,1,4)", 1, 0, 4, 8,
      #nv_tensor_ir.tensor_source<0, 0, "(?,8):(8,1)", [0]>,
      #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>} {
  return
}

// -----

// Dynamic RHS dimensions are not supported.
func.func @matmul_dynamic_rhs(%in0: tensor<4x8xf32>, %in1: tensor<8x?xf32>)
    // expected-error @+1 {{Matmul view must have static shape and stride}}
    attributes {layout = #nv_tensor_ir.matmul_source<"(4,?,8):(0,1,4)", 1, 4, 0, 8,
      #nv_tensor_ir.tensor_source<0, 0, "(4,8):(8,1)">,
      #nv_tensor_ir.tensor_source<1, 0, "(8,?):(1,8)", [0]>>} {
  return
}

// -----

// Invalid LHS underlying source.
func.func @matmul_invalid_lhs(%in0: tensor<4x8xf32>, %in1: tensor<8x4xf32>)
    // expected-error @+1 {{Incorrect underlying LHS layout}}
    attributes {layout = #nv_tensor_ir.matmul_source<"(2,4,8):(1,2,8)", 1, 2, 4, 8,
      #nv_tensor_ir.tensor_source<0, 0, "(4,8):(1,4)">,
      #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>} {
  return
}

// -----

// Invalid RHS underlying source.
func.func @matmul_invalid_rhs(%in0: tensor<2x8xf32>, %in1: tensor<8x2xf32>)
    // expected-error @+1 {{Incorrect underlying RHS layout}}
    attributes {layout = #nv_tensor_ir.matmul_source<"(2,4,8):(1,2,8)", 1, 2, 4, 8,
      #nv_tensor_ir.tensor_source<0, 0, "(2,8):(1,2)">,
      #nv_tensor_ir.tensor_source<1, 0, "(8,2):(1,8)">>} {
  return
}

// -----

// Invalid view layout.
func.func @reduction_invalid_out(%in0: tensor<2x8xf32>, %in1: tensor<8x4xf32>)
    // expected-error @+1 {{Incorrect matmul view size}}
    attributes {layout = #nv_tensor_ir.matmul_source<"(2,4):(1,2)", 1, 2, 4, 8,
      #nv_tensor_ir.tensor_source<0, 0, "(2,8):(1,2)">,
      #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>} {
  return
}
