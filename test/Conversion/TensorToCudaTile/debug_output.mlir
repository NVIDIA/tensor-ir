// REQUIRES: asserts
// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   -debug-only=convert-tensor-to-cuda-tile %s 2>&1 | FileCheck %s

// ============================================================================
// Debug output coverage test.
// Exercises getDynamicSizes(), getDynamicStrides(), isInput(), and the debug
// lambda in locateIterationSpaceDimSizes.
//
// Uses a multi-input graph with 3D tensors to:
//   - Print descriptor info for both input and output tensors
//   - Exercise isInput() for Input/Output classification
//   - Call getDynamicSizes()/getDynamicStrides() (empty for static shapes)
//   - Map all 3 iteration space dimensions via locateIterationSpaceDimSizes
// ============================================================================

// CHECK:       CUDA Tile input/output tensor descriptors:
// CHECK:       Tensor kind: Input
// CHECK:       Dynamic sizes:
// CHECK:       Iteration space map:
// CHECK:       Dynamic strides:
// CHECK:       Tensor kind: Input
// CHECK:       Dynamic sizes:
// CHECK:       Dynamic strides:
// CHECK:       Tensor kind: Output
// CHECK:       Dynamic sizes:
// CHECK:       Dynamic strides:
// CHECK:       Location found for iteration space dimension 0
// CHECK:       Location found for iteration space dimension 1
// CHECK:       Location found for iteration space dimension 2
// CHECK:       Iteration space dimension locations identified:

module {
  nv_tensor_ir.graph @debug_coverage(
    %a: tensor<16x32x64xf32> {nv_tensor_ir.stride = "(2048,64,1)"},
    %b: tensor<16x32x64xf32> {nv_tensor_ir.stride = "(2048,64,1)"}
  ) -> (tensor<16x32x64xf32> {nv_tensor_ir.stride = "(2048,64,1)"}) {
    %result = add %a, %b : tensor<16x32x64xf32>
    results %result : tensor<16x32x64xf32>
  }
}
