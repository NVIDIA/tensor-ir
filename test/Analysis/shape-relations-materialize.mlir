// RUN: tensor_ir-opt %s --split-input-file --test-shape-relations='materialize' | FileCheck %s

// CHECK-LABEL: func.func @materialize_expr_folds(
//  CHECK-SAME:  %[[arg0:.+]]: tensor<?x2xf32>, %[[arg1:.+]]: tensor<?x3xf32>
func.func @materialize_expr_folds(%a: tensor<?x2xf32>, %b: tensor<?x3xf32>, %c: index) -> (tensor<?xf32>, index) {
  %a_flat = nv_tensor_ir.reshape %a dynamic_dims(%c) : tensor<?x2xf32> -> tensor<?xf32>
  %b_flat = nv_tensor_ir.reshape %b dynamic_dims(%c) : tensor<?x3xf32> -> tensor<?xf32>
  %out = nv_tensor_ir.add %a_flat, %b_flat : tensor<?xf32>
  %dim = nv_tensor_ir.dim %out dimension = 0 : tensor<?xf32>
  %expr = affine.apply affine_map<(d0) -> (d0*4)> (%dim) {test.materialize}
  // CHECK-DAG: %[[dim:.+]] = nv_tensor_ir.dim %[[arg1]] dimension = 0 : tensor<?x3xf32>
  // CHECK-DAG: %[[c12:.+]] = arith.constant 12 : index
  // CHECK-DAG: %[[expr:.+]] = arith.muli %[[dim]], %[[c12]] : index
  // CHECK: return %{{.+}}, %[[expr]] :
  return %out, %expr : tensor<?xf32>, index
}

// -----

// CHECK-LABEL: func.func @concat_size(
//  CHECK-SAME:  %[[arg0:.+]]: tensor<?x2xf32>, %[[arg1:.+]]: tensor<?x3xf32>)
func.func @concat_size(%a: tensor<?x2xf32>, %b: tensor<?x3xf32>) -> (tensor<?xf32>, index) {
  %a_flat = nv_tensor_ir.reshape %a : tensor<?x2xf32> -> tensor<?xf32>
  %b_flat = nv_tensor_ir.reshape %b : tensor<?x3xf32> -> tensor<?xf32>
  %out = nv_tensor_ir.concatenate %a_flat, %b_flat dimension = 0 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %dim = nv_tensor_ir.dim %out dimension = 0 : tensor<?xf32>
  %expr = affine.apply affine_map<(d0) -> (d0)> (%dim) {test.materialize}
  // CHECK-DAG: %[[dim0_0:.+]] = nv_tensor_ir.dim %[[arg0]] dimension = 0 : tensor<?x2xf32>
  // CHECK-DAG: %[[dim1_0:.+]] = nv_tensor_ir.dim %[[arg1]] dimension = 0 : tensor<?x3xf32>
  // CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
  // CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[m0:.+]] = arith.muli %[[dim0_0]], %[[c2]] : index
  // CHECK-DAG: %[[m1:.+]] = arith.muli %[[dim1_0]], %[[c3]] : index
  // CHECK-DAG: %[[expr:.+]] = arith.addi %[[m0]], %[[m1]] : index
  // CHECK: return %{{.+}}, %[[expr]] :
  return %out, %expr : tensor<?xf32>, index
}

// -----

// CHECK-LABEL: func.func @materialize_buffer_extents(
// CHECK-SAME: %[[BUFFER:[^:]+]]: memref<?x4xf32>
func.func @materialize_buffer_extents(%buffer: memref<?x4xf32>) -> (index, index) {
  %tensor = bufferization.to_tensor %buffer restrict : memref<?x4xf32> to tensor<?x4xf32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %tensor, %c0 : tensor<?x4xf32>
  %first = affine.apply affine_map<(d0) -> (d0 * 4)>(%dim) {test.materialize}
  %second = affine.apply affine_map<(d0) -> (d0 * 2)>(%dim) {test.materialize}
  // CHECK: %[[ZERO:.+]] = arith.constant 0 : index
  // CHECK: %[[DIM:.+]] = memref.dim %[[BUFFER]], %[[ZERO]] : memref<?x4xf32>
  // CHECK: %[[FOUR:.+]] = arith.constant 4 : index
  // CHECK: %[[FIRST:.+]] = arith.muli %[[DIM]], %[[FOUR]] : index
  // CHECK-NOT: memref.dim
  // CHECK: %[[TWO:.+]] = arith.constant 2 : index
  // CHECK: %[[SECOND:.+]] = arith.muli %[[DIM]], %[[TWO]] : index
  // CHECK: return %[[FIRST]], %[[SECOND]] : index, index
  return %first, %second : index, index
}
