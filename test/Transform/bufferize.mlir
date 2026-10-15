// RUN: tensor_ir-opt %s --tir-bufferize --split-input-file | FileCheck %s --implicit-check-not='nv_tensor_ir.load' --implicit-check-not='nv_tensor_ir.store' --implicit-check-not='ptr.to_ptr'

// CHECK-LABEL: func.func @add_dispatch(
// CHECK-SAME: %[[LHS:[^:]+]]: memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space> {nv_tensor_ir.alignment = 16 : i64}
// CHECK-SAME: %[[RHS:[^:]+]]: memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space> {nv_tensor_ir.alignment = 16 : i64}
// CHECK-SAME: %[[OUT:[^:]+]]: memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space> {nv_tensor_ir.alignment = 16 : i64}
// CHECK-SAME: attributes {nv_tensor_ir.bufferized_program
// CHECK: %[[LHS_TENSOR:.*]] = bufferization.to_tensor %[[LHS]] restrict : memref<?x32xf32, strided<[?, 1]>, #ptr.generic_space> to tensor<?x32xf32>
// CHECK: %[[RHS_TENSOR:.*]] = bufferization.to_tensor %[[RHS]] restrict
// CHECK: %[[SUM:.*]] = nv_tensor_ir.add %[[LHS_TENSOR]], %[[RHS_TENSOR]]
// CHECK: bufferization.materialize_in_destination %[[SUM]] in restrict writable %[[OUT]]
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

// -----

// CHECK-LABEL: func.func @identity_layout_dispatch(
// CHECK-SAME: %[[INPUT:[^:]+]]: memref<8x4xf16, #ptr.generic_space> {nv_tensor_ir.alignment = 2 : i64}
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<8x4xf16, #ptr.generic_space> {nv_tensor_ir.alignment = 2 : i64}
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[INPUT]] restrict
// CHECK: %[[RESULT:.*]] = nv_tensor_ir.cos %[[TENSOR]]
// CHECK: bufferization.materialize_in_destination %[[RESULT]] in restrict writable %[[OUTPUT]]
nv_tensor_ir.graph @identity_layout(%input: tensor<8x4xf16>)
    -> tensor<8x4xf16> {
  %result = cos %input : tensor<8x4xf16>
  results %result : tensor<8x4xf16>
}


// -----

// Tests that allow_tma = false is recorded on the boundary memref argument,
// where FormGrid reads it, and that the default (true) is not materialized.

// CHECK-LABEL: func.func @allow_tma_dispatch(
// CHECK-SAME: %[[INPUT:[^:]+]]: memref<8x4xf16, #ptr.generic_space> {nv_tensor_ir.alignment = 2 : i64, nv_tensor_ir.allow_tma = false}
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<8x4xf16, #ptr.generic_space> {nv_tensor_ir.alignment = 2 : i64})
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[INPUT]] restrict
// CHECK: %[[RESULT:.*]] = nv_tensor_ir.cos %[[TENSOR]]
// CHECK: bufferization.materialize_in_destination %[[RESULT]] in restrict writable %[[OUTPUT]]
nv_tensor_ir.graph @allow_tma(
    %input: tensor<8x4xf16> {nv_tensor_ir.allow_tma = false})
    -> tensor<8x4xf16> {
  %result = cos %input : tensor<8x4xf16>
  results %result : tensor<8x4xf16>
}
