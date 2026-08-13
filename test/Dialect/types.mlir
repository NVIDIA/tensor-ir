// RUN: tensor_ir-opt %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: tensor_ir-opt %s -split-input-file | tensor_ir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: tensor_ir-opt -mlir-print-op-generic %s -split-input-file | tensor_ir-opt | FileCheck %s

// CHECK:  nv_tensor_ir.tensor_type = tensor<1x2x3x4xf32>
module attributes {nv_tensor_ir.tensor_type = tensor<1x2x3x4xf32>} {}

// CHECK:  nv_tensor_ir.dynamic_tensor_type = tensor<?x?x128x128xsi8>
module attributes {nv_tensor_ir.dynamic_tensor_type = tensor<?x?x128x128xsi8>} {}

// CHECK:  nv_tensor_ir.dynamic_tensor_type = tensor<?x?x128x128xui16>
module attributes {nv_tensor_ir.dynamic_tensor_type = tensor<?x?x128x128xui16>} {}

// CHECK:  nv_tensor_ir.tensor_type = tensor<4x3x16x16xf32>
module attributes {nv_tensor_ir.tensor_type = tensor<4x3x16x16xf32>} {}

// CHECK:  nv_tensor_ir.tensor_type = tensor<4x3x?x?xf32>
module attributes {nv_tensor_ir.tensor_type = tensor<4x3x?x?xf32>} {}