// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-tile-reductions{reduction-tile-size=1}))' | FileCheck %s

// The degenerate epilogue already has the reduction's exact result type.
// CHECK-LABEL: func.func @exact_result(
// CHECK: %[[LOOP:.*]] = scf.for
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: return %[[LOOP]] : tensor<4x1xf32>
func.func @exact_result() -> tensor<4x1xf32> {
  %input = nv_tensor_ir.constant dense<1.0> : tensor<4x8xf32>
  %result = nv_tensor_ir.reduce(%input)
      <dimensions = [1], reduction_mode = <add>>
      : tensor<4x8xf32> -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}
