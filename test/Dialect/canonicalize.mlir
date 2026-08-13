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

// Identity view-like operations fold with dynamic shapes.
// CHECK-LABEL: nv_tensor_ir.graph @fold_dynamic_identity_views
// CHECK-SAME:  %[[INPUT:.*]]: tensor<?x3xf32>
// CHECK-NOT:   reshape
// CHECK-NOT:   concatenate
// CHECK-NOT:   transpose
// CHECK:       results %[[INPUT]], %[[INPUT]], %[[INPUT]]
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

// Layout-annotated operations must not be rebuilt by canonicalization.
// CHECK-LABEL: nv_tensor_ir.graph @preserve_layout_annotations
// CHECK-SAME:  %[[B_INPUT:.*]]: tensor<1x1xf32>
// CHECK-SAME:  %[[C_INPUT:.*]]: tensor<4xf16>
// CHECK-SAME:  %[[T_INPUT:.*]]: tensor<2x3xf32>
// CHECK-SAME:  %[[S_INPUT:.*]]: tensor<20xf32>
// CHECK:       %[[SCALAR:.*]] = constant 2.500000e+00 : f32
// CHECK:       %[[B0:.*]] = broadcast %[[B_INPUT]] {{.*}}layout
// CHECK:       %[[B1:.*]] = broadcast %[[B0]]
// CHECK:       %[[C0:.*]] = convert %[[C_INPUT]]
// CHECK:       %[[C1:.*]] = convert %[[C0]] {{.*}}layout
// CHECK:       %[[SPLAT:.*]] = splat %[[SCALAR]] {{.*}}layout
// CHECK:       %[[T0:.*]] = transpose %[[T_INPUT]] {{.*}}layout
// CHECK:       %[[T1:.*]] = transpose %[[T0]]
// CHECK:       %[[S0:.*]] = slice %[[S_INPUT]]
// CHECK:       %[[S1:.*]] = slice %[[S0]] {{.*}}layout
// CHECK:       results %[[B1]], %[[C1]], %[[SPLAT]], %[[T1]], %[[S1]]
nv_tensor_ir.graph @preserve_layout_annotations(
    %broadcast_input: tensor<1x1xf32>,
    %convert_input: tensor<4xf16>,
    %transpose_input: tensor<2x3xf32>,
    %slice_input: tensor<20xf32>)
    -> (tensor<4x8xf32>, tensor<4xf64>, tensor<4xf32>,
        tensor<2x3xf32>, tensor<2xf32>) {
  %b0 = broadcast %broadcast_input
      {layout = #nv_tensor_ir.tensor_source<-1, 0, "(4,1):(1,0)">}
      : tensor<1x1xf32> -> tensor<4x1xf32>
  %b1 = broadcast %b0 : tensor<4x1xf32> -> tensor<4x8xf32>

  %c0 = convert %convert_input : tensor<4xf16> -> tensor<4xf32>
  %c1 = convert %c0
      {layout = #nv_tensor_ir.tensor_source<-1, 0, "(4):(1)">}
      : tensor<4xf32> -> tensor<4xf64>

  %scalar = constant 2.5 : f32
  %splat = splat %scalar
      {layout = #nv_tensor_ir.tensor_source<-1, 0, "(4):(0)">}
      : tensor<4xf32>

  %t0 = transpose %transpose_input permutation = [1, 0]
      {layout = #nv_tensor_ir.tensor_source<-1, 0, "(3,2):(2,1)">}
      : tensor<2x3xf32> -> tensor<3x2xf32>
  %t1 = transpose %t0 permutation = [1, 0]
      : tensor<3x2xf32> -> tensor<2x3xf32>

  %s0 = slice %slice_input starts = [2] limits = [18] strides = [2]
      : tensor<20xf32> -> tensor<8xf32>
  %s1 = slice %s0 starts = [1] limits = [7] strides = [3]
      {layout = #nv_tensor_ir.tensor_source<-1, 0, "(2):(1)">}
      : tensor<8xf32> -> tensor<2xf32>

  results %b1, %c1, %splat, %t1, %s1
      : tensor<4x8xf32>, tensor<4xf64>, tensor<4xf32>,
        tensor<2x3xf32>, tensor<2xf32>
}
