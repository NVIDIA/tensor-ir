// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-tile-reductions{reduction-tile-size=1}))' | FileCheck %s

// The degenerate epilogue refines the tile shape, but replacing the original
// reduction still requires its declared dynamic result type.
// CHECK-LABEL: func.func @dynamic_result(
// CHECK: %[[LOOP:.*]] = scf.for
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[LOOP]] : tensor<4x1xf32> to tensor<?x?xf32>
// CHECK: return %[[CAST]] : tensor<?x?xf32>
func.func @dynamic_result() -> tensor<?x?xf32> {
  %input = nv_tensor_ir.constant dense<1.0> : tensor<4x8xf32>
  %result = nv_tensor_ir.reduce(%input)
      <dimensions = [1], reduction_mode = <add>>
      : tensor<4x8xf32> -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
