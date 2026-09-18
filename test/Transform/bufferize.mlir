// RUN: tensor_ir-opt %s --tir-bufferize | FileCheck %s

// Bufferization changes the program boundary and introduces direct pointer
// accesses, but it does not choose tiles or create a grid.
// CHECK-LABEL: func.func @add_dispatch(
// CHECK-SAME: %[[LHS:[^:]+]]: memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space>
// CHECK-SAME: %[[RHS:[^:]+]]: memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space>
// CHECK-SAME: %[[OUT:[^:]+]]: memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space>
// CHECK-SAME: attributes {nv_tensor_ir.bufferized_program
// CHECK: %{{.*}}, %{{.*}}, %[[LHS_SIZES:.*]]:2, %[[LHS_STRIDES:.*]]:2 = memref.extract_strided_metadata %[[LHS]]
// CHECK: %[[LHS_PTR:.*]] = ptr.to_ptr %[[LHS]]
// CHECK: %[[LHS_TILE:.*]] = nv_tensor_ir.load %[[LHS_PTR]][%{{.*}}, %{{.*}}] view offset: [0], sizes: [%[[LHS_SIZES]]#0, 32], strides: [%[[LHS_STRIDES]]#0, 1], alignment: 16
// CHECK: %{{.*}}, %{{.*}}, %[[RHS_SIZES:.*]]:2, %[[RHS_STRIDES:.*]]:2 = memref.extract_strided_metadata %[[RHS]]
// CHECK: %[[RHS_PTR:.*]] = ptr.to_ptr %[[RHS]]
// CHECK: %[[RHS_TILE:.*]] = nv_tensor_ir.load %[[RHS_PTR]][%{{.*}}, %{{.*}}] view offset: [0], sizes: [%[[RHS_SIZES]]#0, 32], strides: [%[[RHS_STRIDES]]#0, 1], alignment: 16
// CHECK: %[[SUM:.*]] = nv_tensor_ir.add %[[LHS_TILE]], %[[RHS_TILE]]
// CHECK: %{{.*}}, %{{.*}}, %[[OUT_SIZES:.*]]:2, %[[OUT_STRIDES:.*]]:2 = memref.extract_strided_metadata %[[OUT]]
// CHECK: %[[OUT_PTR:.*]] = ptr.to_ptr %[[OUT]]
// CHECK: nv_tensor_ir.store %[[SUM]], %[[OUT_PTR]][%{{.*}}, %{{.*}}] view offset: [0], sizes: [%[[OUT_SIZES]]#0, 32], strides: [%[[OUT_STRIDES]]#0, 1], alignment: 16
// CHECK-NOT: scf.forall
nv_tensor_ir.graph @add(
    %lhs: tensor<?x32xf32> {nv_tensor_ir.alignment = 16 : i64,
                            nv_tensor_ir.stride = "(?,1)"},
    %rhs: tensor<?x32xf32> {nv_tensor_ir.alignment = 16 : i64,
                            nv_tensor_ir.stride = "(?,1)"})
    -> (tensor<?x32xf32> {nv_tensor_ir.alignment = 16 : i64,
                          nv_tensor_ir.stride = "(?,1)"}) {
  %sum = add %lhs, %rhs : tensor<?x32xf32>
  results %sum : tensor<?x32xf32>
}
