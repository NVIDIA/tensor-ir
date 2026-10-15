// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)' \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg,convert-tensor-to-linalg)' \
// RUN:   | tensor_ir-opt --split-input-file --mlir-print-local-scope \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.

nv_tensor_ir.graph @collapse(%input: tensor<2x3x4xf32>) -> tensor<6x4xf32> {
  %result = reshape %input : tensor<2x3x4xf32> -> tensor<6x4xf32>
  results %result : tensor<6x4xf32>
}

// CHECK-LABEL: func.func @collapse(
// CHECK-SAME:      %[[INPUT:.+]]: tensor<2x3x4xf32>
// CHECK:         %[[RESULT:.+]] = tensor.collapse_shape %[[INPUT]]
// CHECK-SAME:      : tensor<2x3x4xf32> into tensor<6x4xf32>
// CHECK:         return %[[RESULT]] : tensor<6x4xf32>

// -----

nv_tensor_ir.graph @expand(%input: tensor<6x4xf32>) -> tensor<2x3x4xf32> {
  %result = reshape %input : tensor<6x4xf32> -> tensor<2x3x4xf32>
  results %result : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @expand(
// CHECK-SAME:      %[[INPUT:.+]]: tensor<6x4xf32>
// CHECK:         %[[RESULT:.+]] = tensor.expand_shape %[[INPUT]]
// CHECK-SAME:      output_shape [2, 3, 4] : tensor<6x4xf32> into tensor<2x3x4xf32>
// CHECK:         return %[[RESULT]] : tensor<2x3x4xf32>

// -----

nv_tensor_ir.graph @general_reshape(
    %input: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %result = reshape %input : tensor<2x3xf32> -> tensor<3x2xf32>
  results %result : tensor<3x2xf32>
}

// CHECK-LABEL: func.func @general_reshape(
// CHECK-SAME:      %[[INPUT:.+]]: tensor<2x3xf32>
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK:         %[[SHAPE:.+]] = tensor.from_elements %[[C3]], %[[C2]]
// CHECK:         %[[RESULT:.+]] = tensor.reshape %[[INPUT]](%[[SHAPE]])
// CHECK:         return %[[RESULT]] : tensor<3x2xf32>

// -----

// CHECK-LABEL: func.func @dynamic_expand(
// CHECK-SAME: %[[N:.*]]: index
nv_tensor_ir.graph @dynamic_expand(%n: index, %a: tensor<?xf32>) -> tensor<?x4xf32> {
  %r = reshape %a dynamic_dims(%n) : tensor<?xf32> -> tensor<?x4xf32>
  results %r : tensor<?x4xf32>
}
// CHECK: tensor.expand_shape
// CHECK-SAME: output_shape [%[[N]], 4]

// -----

// CHECK-LABEL: func.func @dynamic_general_reshape(
// CHECK-SAME: %[[N:.*]]: index
nv_tensor_ir.graph @dynamic_general_reshape(%n: index, %a: tensor<2x?xf32>) -> tensor<?x2xf32> {
  %r = reshape %a dynamic_dims(%n) : tensor<2x?xf32> -> tensor<?x2xf32>
  results %r : tensor<?x2xf32>
}
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[N]], %[[C2]]
// CHECK: tensor.reshape %{{.*}}(%[[SHAPE]])

// -----

nv_tensor_ir.graph @broadcast_slice_transpose(
    %input: tensor<1x4xf32>, %size: index) -> tensor<4x1xf32> {
  %broadcast = broadcast %input dynamic_dims(%size)
      : tensor<1x4xf32> -> tensor<?x4xf32>
  %slice = slice %broadcast starts = [0, 0] limits = [4, 4]
      strides = [1, 1] : tensor<?x4xf32> -> tensor<4x4xf32>
  %transpose = transpose %slice permutation = [1, 0]
      : tensor<4x4xf32> -> tensor<4x4xf32>
  %result = slice %transpose starts = [0, 0] limits = [4, 1]
      strides = [1, 1] : tensor<4x4xf32> -> tensor<4x1xf32>
  results %result : tensor<4x1xf32>
}

// CHECK-LABEL: func.func @broadcast_slice_transpose(
// CHECK-SAME:      %[[INPUT:.+]]: tensor<1x4xf32>, %[[SIZE:.+]]: index
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INPUT]]
// CHECK:         %[[BCAST_EMPTY:.+]] = tensor.empty(%[[SIZE]]) : tensor<?x4xf32>
// CHECK:         %[[BROADCAST:.+]] = linalg.broadcast ins(%[[COLLAPSED]]
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[BROADCAST]]
// CHECK:         %[[TRANSPOSE_EMPTY:.+]] = tensor.empty() : tensor<4x4xf32>
// CHECK:         %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[SLICE]]
// CHECK-SAME:        outs(%[[TRANSPOSE_EMPTY]] : tensor<4x4xf32>) permutation = [1, 0]
// CHECK:         %[[RESULT:.+]] = tensor.extract_slice %[[TRANSPOSE]]
// CHECK:         return %[[RESULT]] : tensor<4x1xf32>
