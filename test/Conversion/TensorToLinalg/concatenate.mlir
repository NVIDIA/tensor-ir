// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)' \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir. --implicit-check-not=tensor.insert_slice --implicit-check-not=unrealized_conversion_cast
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg,convert-tensor-to-linalg)' \
// RUN:   | tensor_ir-opt --split-input-file --mlir-print-local-scope \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir. --implicit-check-not=tensor.insert_slice --implicit-check-not=unrealized_conversion_cast

// CHECK-LABEL: func.func @rows(
// CHECK-SAME: %[[A:.*]]: tensor<2x4xf32>, %[[B:.*]]: tensor<3x4xf32>
nv_tensor_ir.graph @rows(%a: tensor<2x4xf32>, %b: tensor<3x4xf32>) -> tensor<5x4xf32> {
  %r = concatenate %a, %b dimension = 0 : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<5x4xf32>
  results %r : tensor<5x4xf32>
}
// CHECK: %[[R:.*]] = tensor.concat dim(0) %[[A]], %[[B]]
// CHECK-SAME: -> tensor<5x4xf32>
// CHECK: return %[[R]]

// -----

// CHECK-LABEL: func.func @columns(
// CHECK-SAME: %[[A:.*]]: tensor<2x2xf32>, %[[B:.*]]: tensor<2x3xf32>, %[[C:.*]]: tensor<2x1xf32>
nv_tensor_ir.graph @columns(%a: tensor<2x2xf32>, %b: tensor<2x3xf32>, %c: tensor<2x1xf32>) -> tensor<2x6xf32> {
  %r = concatenate %a, %b, %c dimension = 1 : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x6xf32>
  results %r : tensor<2x6xf32>
}
// CHECK: %[[R:.*]] = tensor.concat dim(1) %[[A]], %[[B]], %[[C]]
// CHECK-SAME: -> tensor<2x6xf32>
// CHECK: return %[[R]]

// -----

// CHECK-LABEL: func.func @unsigned_bits(
// CHECK-SAME: %[[A:.*]]: tensor<2xi32>, %[[B:.*]]: tensor<3xi32>
nv_tensor_ir.graph @unsigned_bits(%a: tensor<2xui32>, %b: tensor<3xui32>) -> tensor<5xui32> {
  %r = concatenate %a, %b dimension = 0 : (tensor<2xui32>, tensor<3xui32>) -> tensor<5xui32>
  results %r : tensor<5xui32>
}
// CHECK: %[[R:.*]] = tensor.concat dim(0) %[[A]], %[[B]]
// CHECK-SAME: -> tensor<5xi32>
// CHECK: return %[[R]] : tensor<5xi32>

// -----

// CHECK-LABEL: func.func @dynamic_middle_axis(
nv_tensor_ir.graph @dynamic_middle_axis(%a: tensor<?x?x4xf32>, %b: tensor<?x2x4xf32>, %c: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  %r = concatenate %a, %b, %c dimension = 1 : (tensor<?x?x4xf32>, tensor<?x2x4xf32>, tensor<?x?x4xf32>) -> tensor<?x?x4xf32>
  results %r : tensor<?x?x4xf32>
}
// CHECK: %[[R:.*]] = tensor.concat dim(1) %arg0, %arg1, %arg2
// CHECK-SAME: -> tensor<?x?x4xf32>
// CHECK: return %[[R]] : tensor<?x?x4xf32>

// -----

// A static non-concatenation dimension refines a dynamic input dimension.
// CHECK-LABEL: func.func @refined_shape(
nv_tensor_ir.graph @refined_shape(%a: tensor<?x2xf32>, %b: tensor<3x?xf32>) -> tensor<3x?xf32> {
  %r = concatenate %a, %b dimension = 1 : (tensor<?x2xf32>, tensor<3x?xf32>) -> tensor<3x?xf32>
  results %r : tensor<3x?xf32>
}
// CHECK: %[[R:.*]] = tensor.concat dim(1) %arg0, %arg1
// CHECK-SAME: -> tensor<3x?xf32>
// CHECK: return %[[R]] : tensor<3x?xf32>

// -----

// CHECK-LABEL: func.func @single_input(
// CHECK-SAME: %[[A:.*]]: tensor<?x4xi16>
nv_tensor_ir.graph @single_input(%a: tensor<?x4xsi16>) -> tensor<?x4xsi16> {
  %r = concatenate %a dimension = 0 : (tensor<?x4xsi16>) -> tensor<?x4xsi16>
  results %r : tensor<?x4xsi16>
}
// CHECK-NOT: tensor.insert_slice
// CHECK: return %[[A]] : tensor<?x4xi16>

// -----

// CHECK-LABEL: func.func @empty_inputs(
nv_tensor_ir.graph @empty_inputs(%a: tensor<0x4xi1>, %b: tensor<0x4xi1>) -> tensor<0x4xi1> {
  %r = concatenate %a, %b dimension = 0 : (tensor<0x4xi1>, tensor<0x4xi1>) -> tensor<0x4xi1>
  results %r : tensor<0x4xi1>
}
// CHECK: %[[E:.*]] = tensor.empty() : tensor<0x4xi1>
// CHECK: return %[[E]] : tensor<0x4xi1>

// -----

// Statically empty operands do not become invalid boundary subviews.
// CHECK-LABEL: func.func @skip_empty_operands(
// CHECK-SAME: %{{.*}}: tensor<0x2xf32>, %[[A:.*]]: tensor<1x2xf32>, %[[B:.*]]: tensor<2x2xf32>
nv_tensor_ir.graph @skip_empty_operands(%empty: tensor<0x2xf32>, %a: tensor<1x2xf32>, %b: tensor<2x2xf32>) -> tensor<3x2xf32> {
  %r = concatenate %empty, %a, %empty, %b, %empty dimension = 0 : (tensor<0x2xf32>, tensor<1x2xf32>, tensor<0x2xf32>, tensor<2x2xf32>, tensor<0x2xf32>) -> tensor<3x2xf32>
  results %r : tensor<3x2xf32>
}
// CHECK: %[[R:.*]] = tensor.concat dim(0) %[[A]], %[[B]]
// CHECK-SAME: -> tensor<3x2xf32>
// CHECK: return %[[R]]

// -----

// An empty payload still needs the sum of dynamic concatenation extents.
// CHECK-LABEL: func.func @empty_dynamic_shape(
nv_tensor_ir.graph @empty_dynamic_shape(%a: tensor<0x?x2xf32>, %b: tensor<0x3x2xf32>) -> tensor<0x?x2xf32> {
  %r = concatenate %a, %b dimension = 1 : (tensor<0x?x2xf32>, tensor<0x3x2xf32>) -> tensor<0x?x2xf32>
  results %r : tensor<0x?x2xf32>
}
// CHECK: tensor.dim %arg0
// CHECK: %[[SUM:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[E:.*]] = tensor.empty(%[[SUM]]) : tensor<0x?x2xf32>
// CHECK: return %[[E]] : tensor<0x?x2xf32>
