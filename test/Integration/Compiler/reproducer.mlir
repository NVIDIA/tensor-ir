// RUN: tensor_ir-compiler %s --reproducer-dir=%t --verify
// RUN: cat %t-*/analyze-tensorir-graph.mlir | FileCheck %s
// RUN: cat %t-*/lower-tensorir-to-cudatile.mlir | FileCheck %s
// RUN: cat %t-*/select-tile.mlir | FileCheck %s


module {
  // CHECK-LABEL: func
  nv_tensor_ir.graph @func(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %mul = mul %a, %b : tensor<8x8xf32>
    // CHECK: mlir_reproducer
    // CHECK: pipeline
    results %mul : tensor<8x8xf32>
  }
}
