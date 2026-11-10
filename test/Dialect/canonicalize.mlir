// RUN: tensor_ir-opt %s -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: nv_tensor_ir.graph @merge_broadcasts
// CHECK-SAME:  %[[INPUT:.*]]: tensor<1x1xf32>
// CHECK:       %[[RESULT:.*]] = broadcast %[[INPUT]]
// CHECK-SAME:    : tensor<1x1xf32> -> tensor<4x8xf32>
// CHECK-NOT:   broadcast
// CHECK:       results %[[RESULT]] : tensor<4x8xf32>
nv_tensor_ir.graph @merge_broadcasts(
    %input: tensor<1x1xf32>) -> tensor<4x8xf32> {
  %0 = broadcast %input : tensor<1x1xf32> -> tensor<4x1xf32>
  %1 = broadcast %0 : tensor<4x1xf32> -> tensor<4x8xf32>
  results %1 : tensor<4x8xf32>
}

// -----

// Dynamic shapes do not prevent broadcast merging.
// CHECK-LABEL: nv_tensor_ir.graph @merge_dynamic_broadcasts
// CHECK-SAME:  %[[INPUT:.*]]: tensor<?x1x1xf32>
// CHECK-SAME:  %[[BATCH:.*]]: index
// CHECK:       %[[RESULT:.*]] = broadcast %[[INPUT]]
// CHECK-SAME:    dynamic_dims(%[[BATCH]])
// CHECK-SAME:    : tensor<?x1x1xf32> -> tensor<?x4x8xf32>
// CHECK-NOT:   broadcast
// CHECK:       results %[[RESULT]] : tensor<?x4x8xf32>
nv_tensor_ir.graph @merge_dynamic_broadcasts(
    %input: tensor<?x1x1xf32>, %batch: index) -> tensor<?x4x8xf32> {
  %0 = broadcast %input dynamic_dims(%batch) : tensor<?x1x1xf32> -> tensor<?x4x1xf32>
  %1 = broadcast %0
      dynamic_dims(%batch)
      : tensor<?x4x1xf32> -> tensor<?x4x8xf32>
  results %1 : tensor<?x4x8xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @merge_lossless_converts
// CHECK-SAME:  %[[INPUT:.*]]: tensor<8xf16>
// CHECK-DAG:   %[[F64:.*]] = convert %[[INPUT]]
// CHECK-SAME:    : tensor<8xf16> -> tensor<8xf64>
// CHECK-DAG:   %[[F8:.*]] = convert %[[INPUT]]
// CHECK-SAME:    : tensor<8xf16> -> tensor<8xf8E4M3FN>
// CHECK-NOT:   tensor<8xf32>
// CHECK:       results %[[F64]], %[[INPUT]], %[[F8]]
nv_tensor_ir.graph @merge_lossless_converts(
    %input: tensor<8xf16>)
    -> (tensor<8xf64>, tensor<8xf16>, tensor<8xf8E4M3FN>) {
  %wide = convert %input : tensor<8xf16> -> tensor<8xf32>
  %to_f64 = convert %wide : tensor<8xf32> -> tensor<8xf64>
  %to_f16 = convert %wide : tensor<8xf32> -> tensor<8xf16>
  %to_f8 = convert %wide : tensor<8xf32> -> tensor<8xf8E4M3FN>
  results %to_f64, %to_f16, %to_f8
      : tensor<8xf64>, tensor<8xf16>, tensor<8xf8E4M3FN>
}

// -----

// A lossy first conversion prevents convert merging.
// CHECK-LABEL: nv_tensor_ir.graph @keep_lossy_converts
// CHECK-SAME:  %[[INPUT:.*]]: tensor<8xf32>
// CHECK:       %[[INTEGER:.*]] = convert %[[INPUT]]
// CHECK-SAME:    : tensor<8xf32> -> tensor<8xsi32>
// CHECK:       %[[RESULT:.*]] = convert %[[INTEGER]]
// CHECK-SAME:    : tensor<8xsi32> -> tensor<8xf32>
// CHECK:       results %[[RESULT]] : tensor<8xf32>
nv_tensor_ir.graph @keep_lossy_converts(
    %input: tensor<8xf32>) -> tensor<8xf32> {
  %0 = convert %input : tensor<8xf32> -> tensor<8xsi32>
  %1 = convert %0 : tensor<8xsi32> -> tensor<8xf32>
  results %1 : tensor<8xf32>
}

// -----

// Dynamic shapes do not prevent lossless convert merging.
// CHECK-LABEL: nv_tensor_ir.graph @merge_dynamic_converts
// CHECK-SAME:  %[[INPUT:.*]]: tensor<?xf16>
// CHECK:       %[[RESULT:.*]] = convert %[[INPUT]]
// CHECK-SAME:    : tensor<?xf16> -> tensor<?xf64>
// CHECK-NOT:   convert
// CHECK:       results %[[RESULT]] : tensor<?xf64>
nv_tensor_ir.graph @merge_dynamic_converts(
    %input: tensor<?xf16>) -> tensor<?xf64> {
  %0 = convert %input : tensor<?xf16> -> tensor<?xf32>
  %1 = convert %0 : tensor<?xf32> -> tensor<?xf64>
  results %1 : tensor<?xf64>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @constant_splat
// CHECK:       %[[RESULT:.*]] = constant dense<2.500000e+00> : tensor<2x3xf32>
// CHECK-NOT:   splat
// CHECK:       results %[[RESULT]] : tensor<2x3xf32>
nv_tensor_ir.graph @constant_splat() -> tensor<2x3xf32> {
  %value = constant 2.5 : f32
  %result = splat %value : tensor<2x3xf32>
  results %result : tensor<2x3xf32>
}

// -----

// Dynamic splat results are not converted to dense constants.
// CHECK-LABEL: nv_tensor_ir.graph @keep_dynamic_splat
// CHECK-SAME:  %[[N:.*]]: index
// CHECK:       %[[VALUE:.*]] = constant 2.500000e+00 : f32
// CHECK:       %[[RESULT:.*]] = splat %[[VALUE]]
// CHECK-SAME:    dynamic_dims(%[[N]])
// CHECK-SAME:    : tensor<?xf32>
// CHECK:       results %[[RESULT]] : tensor<?xf32>
nv_tensor_ir.graph @keep_dynamic_splat(%n: index) -> tensor<?xf32> {
  %value = constant 2.5 : f32
  %result = splat %value dynamic_dims(%n) : tensor<?xf32>
  results %result : tensor<?xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @merge_transposes
// CHECK-SAME:  %[[INPUT:.*]]: tensor<2x3x4xf32>
// CHECK:       %[[RESULT:.*]] = transpose %[[INPUT]]
// CHECK-SAME:    permutation = [2, 1, 0]
// CHECK-SAME:    : tensor<2x3x4xf32> -> tensor<4x3x2xf32>
// CHECK-NOT:   transpose
// CHECK:       results %[[RESULT]] : tensor<4x3x2xf32>
nv_tensor_ir.graph @merge_transposes(
    %input: tensor<2x3x4xf32>) -> tensor<4x3x2xf32> {
  %0 = transpose %input permutation = [1, 2, 0]
      : tensor<2x3x4xf32> -> tensor<3x4x2xf32>
  %1 = transpose %0 permutation = [1, 0, 2]
      : tensor<3x4x2xf32> -> tensor<4x3x2xf32>
  results %1 : tensor<4x3x2xf32>
}

// -----

// Dynamic shapes do not prevent transpose merging.
// CHECK-LABEL: nv_tensor_ir.graph @merge_dynamic_transposes
// CHECK-SAME:  %[[INPUT:.*]]: tensor<?x3x4xf32>
// CHECK:       %[[RESULT:.*]] = transpose %[[INPUT]]
// CHECK-SAME:    permutation = [2, 1, 0]
// CHECK-SAME:    : tensor<?x3x4xf32> -> tensor<4x3x?xf32>
// CHECK-NOT:   transpose
// CHECK:       results %[[RESULT]] : tensor<4x3x?xf32>
nv_tensor_ir.graph @merge_dynamic_transposes(
    %input: tensor<?x3x4xf32>) -> tensor<4x3x?xf32> {
  %0 = transpose %input permutation = [1, 2, 0]
      : tensor<?x3x4xf32> -> tensor<3x4x?xf32>
  %1 = transpose %0 permutation = [1, 0, 2]
      : tensor<3x4x?xf32> -> tensor<4x3x?xf32>
  results %1 : tensor<4x3x?xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @merge_slices
// CHECK-SAME:  %[[INPUT:.*]]: tensor<20xf32>
// CHECK:       %[[RESULT:.*]] = slice %[[INPUT]]
// CHECK-SAME:    starts = [4] limits = [11] strides = [6]
// CHECK-SAME:    : tensor<20xf32> -> tensor<2xf32>
// CHECK-NOT:   slice
// CHECK:       results %[[RESULT]] : tensor<2xf32>
nv_tensor_ir.graph @merge_slices(
    %input: tensor<20xf32>) -> tensor<2xf32> {
  %0 = slice %input starts = [2] limits = [18] strides = [2]
      : tensor<20xf32> -> tensor<8xf32>
  %1 = slice %0 starts = [1] limits = [7] strides = [3]
      : tensor<8xf32> -> tensor<2xf32>
  results %1 : tensor<2xf32>
}

// -----

// A dynamic source shape does not prevent slice merging.
// CHECK-LABEL: nv_tensor_ir.graph @merge_dynamic_source_slices
// CHECK-SAME:  %[[INPUT:.*]]: tensor<?xf32>
// CHECK:       %[[RESULT:.*]] = slice %[[INPUT]]
// CHECK-SAME:    starts = [4] limits = [11] strides = [6]
// CHECK-SAME:    : tensor<?xf32> -> tensor<2xf32>
// CHECK-NOT:   slice
// CHECK:       results %[[RESULT]] : tensor<2xf32>
nv_tensor_ir.graph @merge_dynamic_source_slices(
    %input: tensor<?xf32>) -> tensor<2xf32> {
  %0 = slice %input starts = [2] limits = [18] strides = [2]
      : tensor<?xf32> -> tensor<8xf32>
  %1 = slice %0 starts = [1] limits = [7] strides = [3]
      : tensor<8xf32> -> tensor<2xf32>
  results %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @merge_reshapes
// CHECK-SAME:  %[[INPUT:.*]]: tensor<2x3xf32>
// CHECK:       %[[RESULT:.*]] = reshape %[[INPUT]]
// CHECK-SAME:    : tensor<2x3xf32> -> tensor<6xf32>
// CHECK-NOT:   reshape
// CHECK:       results %[[RESULT]] : tensor<6xf32>
nv_tensor_ir.graph @merge_reshapes(
    %input: tensor<2x3xf32>) -> tensor<6xf32> {
  %0 = reshape %input : tensor<2x3xf32> -> tensor<3x2xf32>
  %1 = reshape %0 : tensor<3x2xf32> -> tensor<6xf32>
  results %1 : tensor<6xf32>
}

// -----

// Dynamic shapes prevent reshape merging.
// CHECK-LABEL: nv_tensor_ir.graph @keep_dynamic_reshapes
// CHECK-SAME:  %[[INPUT:.*]]: tensor<?x3xf32>, %[[COLS:.*]]: index, %[[ELEMS:.*]]: index
// CHECK:       %[[FIRST:.*]] = reshape %[[INPUT]]
// CHECK-SAME:    dynamic_dims(%[[COLS]])
// CHECK:       %[[SECOND:.*]] = reshape %[[FIRST]]
// CHECK-SAME:    dynamic_dims(%[[ELEMS]])
// CHECK:       results %[[SECOND]] : tensor<?xf32>
nv_tensor_ir.graph @keep_dynamic_reshapes(
    %input: tensor<?x3xf32>, %cols: index, %elems: index) -> tensor<?xf32> {
  %0 = reshape %input dynamic_dims(%cols) : tensor<?x3xf32> -> tensor<3x?xf32>
  %1 = reshape %0 dynamic_dims(%elems) : tensor<3x?xf32> -> tensor<?xf32>
  results %1 : tensor<?xf32>
}

// -----

// Preserve dynamic reshape sizes while folding other identity views.
// CHECK-LABEL: nv_tensor_ir.graph @fold_dynamic_identity_views
// CHECK-SAME:  %[[INPUT:.*]]: tensor<?x3xf32>
// CHECK:       %[[RESHAPED:.*]] = reshape %[[INPUT]] dynamic_dims
// CHECK-NOT:   concatenate
// CHECK-NOT:   transpose
// CHECK:       results %[[RESHAPED]], %[[INPUT]], %[[INPUT]]
nv_tensor_ir.graph @fold_dynamic_identity_views(
    %input: tensor<?x3xf32>, %rows: index)
    -> (tensor<?x3xf32>, tensor<?x3xf32>, tensor<?x3xf32>) {
  %reshape = reshape %input dynamic_dims(%rows) : tensor<?x3xf32> -> tensor<?x3xf32>
  %concatenate = concatenate %input dimension = 0
      : (tensor<?x3xf32>) -> tensor<?x3xf32>
  %transpose = transpose %input permutation = [0, 1]
      : tensor<?x3xf32> -> tensor<?x3xf32>
  results %reshape, %concatenate, %transpose
      : tensor<?x3xf32>, tensor<?x3xf32>, tensor<?x3xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @identity_folders
// CHECK-SAME:  %[[INPUT:.*]]: tensor<2x3xf32>
// CHECK-NOT:   reshape
// CHECK-NOT:   concatenate
// CHECK-NOT:   slice
// CHECK-NOT:   transpose
// CHECK-NOT:   convert
// CHECK:       results %[[INPUT]], %[[INPUT]], %[[INPUT]], %[[INPUT]], %[[INPUT]]
nv_tensor_ir.graph @identity_folders(
    %input: tensor<2x3xf32>)
    -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>,
        tensor<2x3xf32>, tensor<2x3xf32>) {
  %reshape = reshape %input : tensor<2x3xf32> -> tensor<2x3xf32>
  %concatenate = concatenate %input dimension = 0
      : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %slice = slice %input starts = [0, 0] limits = [2, 3] strides = [1, 1]
      : tensor<2x3xf32> -> tensor<2x3xf32>
  %transpose = transpose %input permutation = [0, 1]
      : tensor<2x3xf32> -> tensor<2x3xf32>
  %convert = convert %input : tensor<2x3xf32> -> tensor<2x3xf32>
  results %reshape, %concatenate, %slice, %transpose, %convert
      : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>,
        tensor<2x3xf32>, tensor<2x3xf32>
}

// -----

// Identity converts with dynamic shapes are folded.
// CHECK-LABEL: nv_tensor_ir.graph @fold_dynamic_identity_convert
// CHECK-SAME:  %[[INPUT:.*]]: tensor<?xf32>
// CHECK-NOT:   convert
// CHECK:       results %[[INPUT]] : tensor<?xf32>
nv_tensor_ir.graph @fold_dynamic_identity_convert(
    %input: tensor<?xf32>) -> tensor<?xf32> {
  %0 = convert %input : tensor<?xf32> -> tensor<?xf32>
  results %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @fold_unit_reduction
// CHECK-SAME:  %[[INPUT:.*]]: tensor<?x1x3xf32>
// CHECK-NOT:   reduce
// CHECK:       results %[[INPUT]] : tensor<?x1x3xf32>
nv_tensor_ir.graph @fold_unit_reduction(
    %input: tensor<?x1x3xf32>) -> tensor<?x1x3xf32> {
  %0 = reduce(%input)<dimensions = [1], reduction_mode = <add>>
      : tensor<?x1x3xf32> -> tensor<?x1x3xf32>
  results %0 : tensor<?x1x3xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @keep_nonunit_reduction
// CHECK-SAME:  %[[INPUT:.*]]: tensor<2x4x3xf32>
// CHECK:       %[[RESULT:.*]] = reduce(%[[INPUT]])
// CHECK:       results %[[RESULT]] : tensor<2x1x3xf32>
nv_tensor_ir.graph @keep_nonunit_reduction(
    %input: tensor<2x4x3xf32>) -> tensor<2x1x3xf32> {
  %0 = reduce(%input)<dimensions = [1], reduction_mode = <add>>
      : tensor<2x4x3xf32> -> tensor<2x1x3xf32>
  results %0 : tensor<2x1x3xf32>
}

// -----

// Unit reductions without a prologue or epilogue fold to their input.
// CHECK-LABEL: nv_tensor_ir.graph @fold_unit_reduction_modes
// CHECK-SAME:  %[[FLOATS:.*]]: tensor<4x1xf32>, %[[INTEGERS:.*]]: tensor<4x1xsi32>
// CHECK-NOT:   reduce
// CHECK:       results %[[FLOATS]], %[[FLOATS]], %[[FLOATS]], %[[INTEGERS]], %[[INTEGERS]], %[[INTEGERS]]
nv_tensor_ir.graph @fold_unit_reduction_modes(
    %floats: tensor<4x1xf32>, %integers: tensor<4x1xsi32>)
    -> (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>,
        tensor<4x1xsi32>, tensor<4x1xsi32>, tensor<4x1xsi32>) {
  %0 = reduce(%floats) <dimensions = [1], reduction_mode = <mul>>
      : tensor<4x1xf32> -> tensor<4x1xf32>
  %1 = reduce(%floats) <dimensions = [1], reduction_mode = <min>>
      : tensor<4x1xf32> -> tensor<4x1xf32>
  %2 = reduce(%floats) <dimensions = [1], reduction_mode = <max>>
      : tensor<4x1xf32> -> tensor<4x1xf32>
  %3 = reduce(%integers) <dimensions = [1], reduction_mode = <add>>
      : tensor<4x1xsi32> -> tensor<4x1xsi32>
  %4 = reduce(%integers) <dimensions = [1], reduction_mode = <min>>
      : tensor<4x1xsi32> -> tensor<4x1xsi32>
  %5 = reduce(%integers) <dimensions = [1], reduction_mode = <max>>
      : tensor<4x1xsi32> -> tensor<4x1xsi32>
  results %0, %1, %2, %3, %4, %5 : tensor<4x1xf32>, tensor<4x1xf32>,
      tensor<4x1xf32>, tensor<4x1xsi32>, tensor<4x1xsi32>, tensor<4x1xsi32>
}

// -----

// A prologue or epilogue changes even a single element, so these stay.
// CHECK-LABEL: nv_tensor_ir.graph @keep_unit_reduction_modes
// CHECK:       reduction_mode = <amax>
// CHECK:       reduction_mode = <norm1>
// CHECK:       reduction_mode = <norm2>
// CHECK:       reduction_mode = <mul_no_zeros>
// CHECK:       reduction_mode = <avg>
// CHECK:       reduction_mode = <amax>
// CHECK:       reduction_mode = <mul_no_zeros>
nv_tensor_ir.graph @keep_unit_reduction_modes(
    %floats: tensor<4x1xf32>, %integers: tensor<4x1xsi32>)
    -> (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>,
        tensor<4x1xf32>, tensor<4x1xsi32>, tensor<4x1xsi32>) {
  %0 = reduce(%floats) <dimensions = [1], reduction_mode = <amax>>
      : tensor<4x1xf32> -> tensor<4x1xf32>
  %1 = reduce(%floats) <dimensions = [1], reduction_mode = <norm1>>
      : tensor<4x1xf32> -> tensor<4x1xf32>
  %2 = reduce(%floats) <dimensions = [1], reduction_mode = <norm2>>
      : tensor<4x1xf32> -> tensor<4x1xf32>
  %3 = reduce(%floats) <dimensions = [1], reduction_mode = <mul_no_zeros>>
      : tensor<4x1xf32> -> tensor<4x1xf32>
  %4 = reduce(%floats) <dimensions = [1], reduction_mode = <avg>>
      : tensor<4x1xf32> -> tensor<4x1xf32>
  %5 = reduce(%integers) <dimensions = [1], reduction_mode = <amax>>
      : tensor<4x1xsi32> -> tensor<4x1xsi32>
  %6 = reduce(%integers) <dimensions = [1], reduction_mode = <mul_no_zeros>>
      : tensor<4x1xsi32> -> tensor<4x1xsi32>
  results %0, %1, %2, %3, %4, %5, %6 : tensor<4x1xf32>, tensor<4x1xf32>,
      tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xsi32>,
      tensor<4x1xsi32>
}

// -----

// Constant operands of commutative ops are canonicalized to the RHS.
// CHECK-LABEL: nv_tensor_ir.graph @commutative_constants
// CHECK-SAME:  %[[FLOAT:.*]]: tensor<4xf32>, %[[BOOL:.*]]: tensor<4xi1>
// CHECK-DAG:   %[[CF:.*]] = constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-DAG:   %[[CB:.*]] = constant dense<true> : tensor<4xi1>
// CHECK-DAG:   %[[ADD:.*]] = add %[[FLOAT]], %[[CF]] : tensor<4xf32>
// CHECK-DAG:   %[[MUL:.*]] = mul %[[FLOAT]], %[[CF]] : tensor<4xf32>
// CHECK-DAG:   %[[MAX:.*]] = max %[[FLOAT]], %[[CF]] : tensor<4xf32>
// CHECK-DAG:   %[[MIN:.*]] = min %[[FLOAT]], %[[CF]] : tensor<4xf32>
// CHECK-DAG:   %[[AND:.*]] = and %[[BOOL]], %[[CB]] : tensor<4xi1>
// CHECK-DAG:   %[[OR:.*]] = or %[[BOOL]], %[[CB]] : tensor<4xi1>
// CHECK:       results %[[ADD]], %[[MUL]], %[[MAX]], %[[MIN]], %[[AND]], %[[OR]]
nv_tensor_ir.graph @commutative_constants(
    %float: tensor<4xf32>, %bool: tensor<4xi1>)
    -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>,
        tensor<4xi1>, tensor<4xi1>) {
  %cf = constant dense<1.0> : tensor<4xf32>
  %cb = constant dense<true> : tensor<4xi1>
  %add = add %cf, %float : tensor<4xf32>
  %mul = mul %cf, %float : tensor<4xf32>
  %max = max %cf, %float : tensor<4xf32>
  %min = min %cf, %float : tensor<4xf32>
  %and = and %cb, %bool : tensor<4xi1>
  %or = or %cb, %bool : tensor<4xi1>
  results %add, %mul, %max, %min, %and, %or
      : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>,
        tensor<4xi1>, tensor<4xi1>
}

// -----

// Nested concatenates along the same dimension are flattened; a nested
// concatenate along another dimension is kept.
// CHECK-LABEL: nv_tensor_ir.graph @flatten_concatenates
// CHECK-SAME:  %[[A:.*]]: tensor<2x4xf32>, %[[B:.*]]: tensor<2x4xf32>, %[[C:.*]]: tensor<2x4xf32>
// CHECK:       %[[FLAT:.*]] = concatenate %[[A]], %[[B]], %[[C]], %[[A]] dimension = 1
// CHECK-SAME:    -> tensor<2x16xf32>
// CHECK:       %[[ROWS:.*]] = concatenate %[[A]], %[[B]] dimension = 0
// CHECK:       %[[MIXED:.*]] = concatenate %[[ROWS]], %[[ROWS]] dimension = 1
// CHECK:       results %[[FLAT]], %[[MIXED]]
nv_tensor_ir.graph @flatten_concatenates(
    %a: tensor<2x4xf32>, %b: tensor<2x4xf32>, %c: tensor<2x4xf32>)
    -> (tensor<2x16xf32>, tensor<4x8xf32>) {
  %inner = concatenate %b, %c dimension = 1
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  %flat = concatenate %a, %inner, %a dimension = 1
      : (tensor<2x4xf32>, tensor<2x8xf32>, tensor<2x4xf32>) -> tensor<2x16xf32>
  %rows = concatenate %a, %b dimension = 0
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  %mixed = concatenate %rows, %rows dimension = 1
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x8xf32>
  results %flat, %mixed : tensor<2x16xf32>, tensor<4x8xf32>
}

// -----

// Inputs of static zero extent are dropped unless they determine a static
// result extent.
// CHECK-LABEL: nv_tensor_ir.graph @drop_zero_extent_concatenate_inputs
// CHECK-SAME:  %[[A:.*]]: tensor<2x4xf32>, %[[B:.*]]: tensor<2x3xf32>, %[[Z:.*]]: tensor<2x0xf32>, %[[D:.*]]: tensor<?x4xf32>
// CHECK:       %[[DROPPED:.*]] = concatenate %[[A]], %[[B]] dimension = 1
// CHECK-SAME:    : (tensor<2x4xf32>, tensor<2x3xf32>) -> tensor<2x7xf32>
// CHECK:       %[[KEPT:.*]] = concatenate %[[D]], %[[Z]] dimension = 1
// CHECK-SAME:    : (tensor<?x4xf32>, tensor<2x0xf32>) -> tensor<2x4xf32>
// CHECK:       results %[[DROPPED]], %[[A]], %[[KEPT]]
nv_tensor_ir.graph @drop_zero_extent_concatenate_inputs(
    %a: tensor<2x4xf32>, %b: tensor<2x3xf32>, %z: tensor<2x0xf32>,
    %d: tensor<?x4xf32>)
    -> (tensor<2x7xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  %dropped = concatenate %z, %a, %z, %b dimension = 1
      : (tensor<2x0xf32>, tensor<2x4xf32>, tensor<2x0xf32>, tensor<2x3xf32>)
      -> tensor<2x7xf32>
  %single = concatenate %a, %z dimension = 1
      : (tensor<2x4xf32>, tensor<2x0xf32>) -> tensor<2x4xf32>
  %kept = concatenate %d, %z dimension = 1
      : (tensor<?x4xf32>, tensor<2x0xf32>) -> tensor<2x4xf32>
  results %dropped, %single, %kept
      : tensor<2x7xf32>, tensor<2x4xf32>, tensor<2x4xf32>
}

// -----

// A slice of a concatenate reads one input or a narrower concatenate of input
// slices, with unit or non-unit stride along the concatenation dimension.
// CHECK-LABEL: nv_tensor_ir.graph @slice_concatenate
// CHECK-SAME:  %[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<4x8xf32>, %[[C:.*]]: tensor<4x8xf32>
// CHECK:       %[[ONE:.*]] = slice %[[B]] starts = [1, 2] limits = [3, 6] strides = [1, 1]
// CHECK-SAME:    : tensor<4x8xf32> -> tensor<2x4xf32>
// CHECK:       %[[A0:.*]] = slice %[[A]] starts = [0, 4] limits = [4, 8] strides = [1, 1]
// CHECK:       %[[B0:.*]] = slice %[[B]] starts = [0, 0] limits = [4, 4] strides = [1, 1]
// CHECK:       %[[TWO:.*]] = concatenate %[[A0]], %[[B0]] dimension = 1
// CHECK-SAME:    -> tensor<4x8xf32>
// CHECK:       %[[B1:.*]] = slice %[[B]] starts = [0, 1] limits = [4, 2] strides = [1, 7]
// CHECK-SAME:    : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK:       %[[C1:.*]] = slice %[[C]] starts = [0, 0] limits = [4, 8] strides = [1, 7]
// CHECK-SAME:    : tensor<4x8xf32> -> tensor<4x2xf32>
// CHECK:       %[[STRIDED:.*]] = concatenate %[[B1]], %[[C1]] dimension = 1
// CHECK-SAME:    -> tensor<4x3xf32>
// CHECK:       results %[[ONE]], %[[TWO]], %[[STRIDED]]
nv_tensor_ir.graph @slice_concatenate(
    %a: tensor<4x8xf32>, %b: tensor<4x8xf32>, %c: tensor<4x8xf32>)
    -> (tensor<2x4xf32>, tensor<4x8xf32>, tensor<4x3xf32>) {
  %cat = concatenate %a, %b, %c dimension = 1
      : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x24xf32>
  %one = slice %cat starts = [1, 10] limits = [3, 14] strides = [1, 1]
      : tensor<4x24xf32> -> tensor<2x4xf32>
  %two = slice %cat starts = [0, 4] limits = [4, 12] strides = [1, 1]
      : tensor<4x24xf32> -> tensor<4x8xf32>
  %strided = slice %cat starts = [0, 9] limits = [4, 24] strides = [1, 7]
      : tensor<4x24xf32> -> tensor<4x3xf32>
  results %one, %two, %strided
      : tensor<2x4xf32>, tensor<4x8xf32>, tensor<4x3xf32>
}

// -----

// A slice that reads every input, or that ends past an input of dynamic
// extent other than the last, is kept. The last input holds every index after
// its offset even when its extent is dynamic.
// CHECK-LABEL: nv_tensor_ir.graph @slice_concatenate_no_fold
// CHECK-SAME:  %[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<4x8xf32>, %[[D:.*]]: tensor<4x?xf32>
// CHECK:       %[[CAT:.*]] = concatenate %[[A]], %[[B]] dimension = 1
// CHECK:       %[[EVERY:.*]] = slice %[[CAT]] starts = [0, 1] limits = [2, 15]
// CHECK:       %[[DYN:.*]] = concatenate %[[D]], %[[A]] dimension = 1
// CHECK:       %[[BLOCKED:.*]] = slice %[[DYN]] starts = [0, 0] limits = [4, 2]
// CHECK:       %[[TAIL:.*]] = slice %[[D]] starts = [0, 2] limits = [4, 4] strides = [1, 1]
// CHECK-SAME:    : tensor<4x?xf32> -> tensor<4x2xf32>
// CHECK:       results %[[EVERY]], %[[BLOCKED]], %[[TAIL]]
nv_tensor_ir.graph @slice_concatenate_no_fold(
    %a: tensor<4x8xf32>, %b: tensor<4x8xf32>, %d: tensor<4x?xf32>)
    -> (tensor<2x14xf32>, tensor<4x2xf32>, tensor<4x2xf32>) {
  %cat = concatenate %a, %b dimension = 1
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x16xf32>
  %every = slice %cat starts = [0, 1] limits = [2, 15] strides = [1, 1]
      : tensor<4x16xf32> -> tensor<2x14xf32>
  %dyn = concatenate %d, %a dimension = 1
      : (tensor<4x?xf32>, tensor<4x8xf32>) -> tensor<4x?xf32>
  %blocked = slice %dyn starts = [0, 0] limits = [4, 2] strides = [1, 1]
      : tensor<4x?xf32> -> tensor<4x2xf32>
  %tail_cat = concatenate %a, %d dimension = 1
      : (tensor<4x8xf32>, tensor<4x?xf32>) -> tensor<4x?xf32>
  %tail = slice %tail_cat starts = [0, 10] limits = [4, 12] strides = [1, 1]
      : tensor<4x?xf32> -> tensor<4x2xf32>
  results %every, %blocked, %tail
      : tensor<2x14xf32>, tensor<4x2xf32>, tensor<4x2xf32>
}

// -----

// Consecutive unit-stride slices of one value that are adjacent along the
// concatenation dimension merge into one slice.
// CHECK-LABEL: nv_tensor_ir.graph @merge_adjacent_concatenated_slices
// CHECK-SAME:  %[[X:.*]]: tensor<8x16xf32>, %[[Y:.*]]: tensor<4x2xf32>
// CHECK:       %[[WHOLE:.*]] = slice %[[X]] starts = [2, 0] limits = [6, 10] strides = [1, 1]
// CHECK-SAME:    : tensor<8x16xf32> -> tensor<4x10xf32>
// CHECK:       %[[LEFT:.*]] = slice %[[X]] starts = [2, 0] limits = [6, 7] strides = [1, 1]
// CHECK:       %[[PART:.*]] = concatenate %[[LEFT]], %[[Y]] dimension = 1
// CHECK-SAME:    : (tensor<4x7xf32>, tensor<4x2xf32>) -> tensor<4x9xf32>
// CHECK:       results %[[WHOLE]], %[[PART]]
nv_tensor_ir.graph @merge_adjacent_concatenated_slices(
    %x: tensor<8x16xf32>, %y: tensor<4x2xf32>)
    -> (tensor<4x10xf32>, tensor<4x9xf32>) {
  %s0 = slice %x starts = [2, 0] limits = [6, 3] strides = [1, 1]
      : tensor<8x16xf32> -> tensor<4x3xf32>
  %s1 = slice %x starts = [2, 3] limits = [6, 7] strides = [1, 1]
      : tensor<8x16xf32> -> tensor<4x4xf32>
  %s2 = slice %x starts = [2, 7] limits = [6, 10] strides = [1, 1]
      : tensor<8x16xf32> -> tensor<4x3xf32>
  %whole = concatenate %s0, %s1, %s2 dimension = 1
      : (tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x10xf32>
  %part = concatenate %s0, %s1, %y dimension = 1
      : (tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x9xf32>
  results %whole, %part : tensor<4x10xf32>, tensor<4x9xf32>
}

// -----

// Slices are kept when they are adjacent only along another dimension, not
// adjacent, strided along the concatenation dimension, or of different values.
// CHECK-LABEL: nv_tensor_ir.graph @keep_concatenated_slices
// CHECK:       %[[R0:.*]] = slice
// CHECK:       %[[R1:.*]] = slice
// CHECK:       %[[OTHER:.*]] = concatenate %[[R0]], %[[R1]] dimension = 1
// CHECK:       %[[GAP:.*]] = slice
// CHECK:       %[[NOT_ADJACENT:.*]] = concatenate %[[R0]], %[[GAP]] dimension = 1
// CHECK:       %[[T0:.*]] = slice
// CHECK:       %[[T1:.*]] = slice
// CHECK:       %[[STRIDED:.*]] = concatenate %[[T0]], %[[T1]] dimension = 1
// CHECK:       %[[W0:.*]] = slice
// CHECK:       %[[DIFFERENT:.*]] = concatenate %[[R0]], %[[W0]] dimension = 1
// CHECK:       results %[[OTHER]], %[[NOT_ADJACENT]], %[[STRIDED]], %[[DIFFERENT]]
nv_tensor_ir.graph @keep_concatenated_slices(
    %x: tensor<8x16xf32>, %w: tensor<8x16xf32>)
    -> (tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x4xf32>, tensor<2x8xf32>) {
  %r0 = slice %x starts = [0, 0] limits = [2, 4] strides = [1, 1]
      : tensor<8x16xf32> -> tensor<2x4xf32>
  %r1 = slice %x starts = [2, 0] limits = [4, 4] strides = [1, 1]
      : tensor<8x16xf32> -> tensor<2x4xf32>
  %other_dim = concatenate %r0, %r1 dimension = 1
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  %gap = slice %x starts = [0, 5] limits = [2, 9] strides = [1, 1]
      : tensor<8x16xf32> -> tensor<2x4xf32>
  %not_adjacent = concatenate %r0, %gap dimension = 1
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  %t0 = slice %x starts = [0, 0] limits = [2, 4] strides = [1, 2]
      : tensor<8x16xf32> -> tensor<2x2xf32>
  %t1 = slice %x starts = [0, 4] limits = [2, 8] strides = [1, 2]
      : tensor<8x16xf32> -> tensor<2x2xf32>
  %strided = concatenate %t0, %t1 dimension = 1
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %w0 = slice %w starts = [0, 4] limits = [2, 8] strides = [1, 1]
      : tensor<8x16xf32> -> tensor<2x4xf32>
  %different = concatenate %r0, %w0 dimension = 1
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  results %other_dim, %not_adjacent, %strided, %different
      : tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x4xf32>, tensor<2x8xf32>
}

// -----

// Small shared concatenates are flattened.
// CHECK-LABEL: nv_tensor_ir.graph @shared_concatenates
// CHECK-SAME:  %[[X:.*]]: tensor<2xf32>
// CHECK:       %[[OUTER:.*]] = concatenate %[[X]], %[[X]], %[[X]], %[[X]] dimension = 0
// CHECK:       results %[[OUTER]] : tensor<8xf32>
nv_tensor_ir.graph @shared_concatenates(%x: tensor<2xf32>) -> tensor<8xf32> {
  %inner = concatenate %x, %x dimension = 0
      : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
  %outer = concatenate %inner, %inner dimension = 0
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
  results %outer : tensor<8xf32>
}

// -----

// Flattening stops at 64 operands even through repeated shared inputs.
// CHECK-LABEL: nv_tensor_ir.graph @concatenate_expansion_budget
// CHECK-SAME:  %[[X:.*]]: tensor<1xf32>
// CHECK:       %[[INNER:.*]] = concatenate %[[X]]{{(, %[a-zA-Z0-9_]+){63} }}dimension = 0
// CHECK:       %[[OUTER:.*]] = concatenate %[[INNER]], %[[INNER]] dimension = 0
// CHECK:       results %[[OUTER]] : tensor<128xf32>
nv_tensor_ir.graph @concatenate_expansion_budget(%x: tensor<1xf32>) -> tensor<128xf32> {
  %c0 = concatenate %x, %x dimension = 0
      : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
  %c1 = concatenate %c0, %c0 dimension = 0
      : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
  %c2 = concatenate %c1, %c1 dimension = 0
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
  %c3 = concatenate %c2, %c2 dimension = 0
      : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  %c4 = concatenate %c3, %c3 dimension = 0
      : (tensor<16xf32>, tensor<16xf32>) -> tensor<32xf32>
  %c5 = concatenate %c4, %c4 dimension = 0
      : (tensor<32xf32>, tensor<32xf32>) -> tensor<64xf32>
  %c6 = concatenate %c5, %c5 dimension = 0
      : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
  results %c6 : tensor<128xf32>
}
