// RUN: split-file %s %t
// RUN: tensor_ir-opt %t/constraints.mlir --split-input-file --verify-diagnostics --test-shape-relations -o /dev/null
// RUN: tensor_ir-opt %t/materialize.mlir --split-input-file --verify-diagnostics --test-shape-relations='materialize' -o /dev/null

//--- constraints.mlir

// Each contradiction is reported once, by verification; queries that touch it
// answer that they do not know.

nv_tensor_ir.graph @contradictory_assumptions(%a: tensor<?xf32>) -> tensor<?xf32> {
  nv_tensor_ir.assume_shape %a, [4] : tensor<?xf32>
  // expected-error @below {{dimension 0 contradicts the inferred extent}}
  nv_tensor_ir.assume_shape %a, [8] : tensor<?xf32>
  results %a : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @contradictory_reshape_volume(%a: tensor<?xf32>) -> tensor<?x4xf32> {
  // expected-error @below {{contradictory shape equality}}
  %reshaped = reshape %a : tensor<?xf32> -> tensor<?x4xf32>
  nv_tensor_ir.assume_shape %reshaped, [2, 4] : tensor<?x4xf32>
  %intermediate = cos %a : tensor<?xf32>
  nv_tensor_ir.assume_shape %intermediate, [12] : tensor<?xf32>
  results %reshaped : tensor<?x4xf32>
}

// -----

nv_tensor_ir.graph @queried_contradiction(%a: tensor<8xf32>, %b: tensor<?xf32>) -> tensor<8xf32> {
  nv_tensor_ir.assume_shape %b, [4] : tensor<?xf32>
  // expected-error @below {{dimension 0 contradicts the inferred extent}}
  nv_tensor_ir.assume_shape %b, [8] : tensor<?xf32>
  // expected-remark @below {{bad_extent: unknown}}
  %extent = nv_tensor_ir.dim %b dimension = 0 {test.shape_query = "bad_extent"} : tensor<?xf32>
  // expected-remark @below {{bad_equal: unknown}}
  %equal = arith.cmpi eq, %extent, %extent {test.shape_query = "bad_equal"} : index
  results %a : tensor<8xf32>
}

// -----

// A contradiction leaves the queries of unrelated symbols answered.
nv_tensor_ir.graph @disconnected_contradiction(%a: tensor<8xf32>, %b: tensor<?xf32>) -> tensor<8xf32> {
  nv_tensor_ir.assume_shape %b, [4] : tensor<?xf32>
  // expected-error @below {{dimension 0 contradicts the inferred extent}}
  nv_tensor_ir.assume_shape %b, [8] : tensor<?xf32>
  // expected-remark @below {{good_extent: 8}}
  %extent = nv_tensor_ir.dim %a dimension = 0 {test.shape_query = "good_extent"} : tensor<8xf32>
  results %a : tensor<8xf32>
}

// -----

// A region with several blocks relates nothing and reports nothing.
nv_tensor_ir.graph @multiblock_region(%b: tensor<?xf32>) -> tensor<?xf32> {
  %region = scf.execute_region -> tensor<?xf32> {
    cf.br ^next
  ^next:
    scf.yield %b : tensor<?xf32>
  }
  // expected-remark @below {{region_input_extent: s0}}
  %extent = nv_tensor_ir.dim %b dimension = 0 {test.shape_query = "region_input_extent"} : tensor<?xf32>
  results %region : tensor<?xf32>
}

// -----

nv_tensor_ir.graph @disconnected_invalid_dimension(%a: tensor<8xf32>, %b: tensor<?xf32>) -> tensor<8xf32> {
  %index = arith.constant 3 : index
  %invalid = tensor.dim %b, %index : tensor<?xf32>
  %wrapped = arith.addi %invalid, %index : index
  nv_tensor_ir.assume_shape %b, [%wrapped] : tensor<?xf32>
  // expected-remark @below {{good_extent: 8}}
  %extent = nv_tensor_ir.dim %a dimension = 0 {test.shape_query = "good_extent"} : tensor<8xf32>
  results %a : tensor<8xf32>
}

// -----

// A contradiction reached through a value-bounds relation fails every query
// connected to it.
func.func @failed_class_bindings(%a: tensor<?xf32>, %t: tensor<?xf32>,
    %cond: i1, %ub: index) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-remark @below {{before: unknown}}
  %first = nv_tensor_ir.dim %a dimension = 0 {test.shape_query = "before"} : tensor<?xf32>
  %r = scf.for %iv = %c0 to %ub step %c1 iter_args(%it = %a) -> tensor<?xf32> {
    scf.yield %it : tensor<?xf32>
  }
  %d = tensor.dim %r, %c0 : tensor<?xf32>
  // expected-remark @below {{linked: unknown}}
  %q = arith.select %cond, %d, %d {test.shape_query = "linked"} : index
  nv_tensor_ir.assume_shape %t, [%q] : tensor<?xf32>
  nv_tensor_ir.assume_shape %t, [4] : tensor<?xf32>
  // expected-error @below {{dimension 0 contradicts the inferred extent}}
  nv_tensor_ir.assume_shape %t, [8] : tensor<?xf32>
  // expected-remark @below {{after: unknown}}
  %second = nv_tensor_ir.dim %a dimension = 0 {test.shape_query = "after"} : tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// A slice relates no input extent to its static output, so a contradiction
// among the output's consumers leaves the input extent known.
nv_tensor_ir.graph @slice_isolation(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<10xf32> {
  %s = slice %a starts = [0] limits = [10] strides = [1] : tensor<?xf32> -> tensor<10xf32>
  %d = nv_tensor_ir.dim %s dimension = 0 : tensor<10xf32>
  nv_tensor_ir.assume_shape %b, [%d] : tensor<?xf32>
  %c4 = arith.constant 4 : index
  // expected-error @below {{dimension 0 contradicts the inferred extent}}
  nv_tensor_ir.assume_shape %b, [%c4] : tensor<?xf32>
  // expected-remark @below {{input_extent: s0}}
  %q = nv_tensor_ir.dim %a dimension = 0 {test.shape_query = "input_extent"} : tensor<?xf32>
  results %s : tensor<10xf32>
}

//--- materialize.mlir

func.func @overflow(%a: tensor<?x?xf32>, %b : index) -> (tensor<?xf32>, index) {
  %big = arith.constant 4611686018427387904 : index
  %c4 = arith.constant 4 : index
  nv_tensor_ir.assume_shape %a, [%big, %c4] : tensor<?x?xf32>
  %a_flat = nv_tensor_ir.reshape %a dynamic_dims(%b) : tensor<?x?xf32> -> tensor<?xf32>
  %dim = nv_tensor_ir.dim %a_flat dimension = 0 : tensor<?xf32>
  // expected-error @below {{cannot materialize extent 18446744073709551616}}
  %expr = affine.apply affine_map<(d0) -> (d0)> (%dim) {test.materialize}
  return %a_flat, %expr : tensor<?xf32>, index
}
