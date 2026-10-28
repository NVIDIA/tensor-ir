// RUN: split-file %s %t
// RUN: tensor_ir-opt %t/empty-auto.mlir -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   -verify-diagnostics
// RUN: tensor_ir-opt %t/dynamic-persistence.mlir -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map \
// RUN:     persistence=static sm-count=1 tile-size=2,2" -verify-diagnostics
// RUN: tensor_ir-opt %t/uniform-index.mlir -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map tile-size=2" \
// RUN:   -verify-diagnostics
// RUN: tensor_ir-opt %t/descriptorless.mlir -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   -verify-diagnostics

//--- empty-auto.mlir

// An empty extent can lower with explicit tiles, but automatic tile analysis
// cannot currently select tiles for it. This is not a rank-zero graph.
// expected-error @+2 {{cannot analyze tile size: the graph's tensor shape or stride is not statically known}}
// expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @empty_auto(%a: tensor<0x8xf32>) -> tensor<0x8xf32> {
  %r = abs %a : tensor<0x8xf32>
  results %r : tensor<0x8xf32>
}

//--- uniform-index.mlir

// A dynamic splat is valid TensorIR, but affine-map signature conversion does
// not support a uniform index argument. Diagnose rather than partially lower.
// expected-error @+2 {{Failed to convert graph signature}}
// expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @uniform_index(%value: f32, %n: index) -> tensor<?xf32> {
  %r = splat %value dynamic_dims(%n) : tensor<?xf32>
  results %r : tensor<?xf32>
}

//--- descriptorless.mlir

// An unused tensor computation has an iteration space but no input or output
// descriptor from which lowering can obtain its extents.
// expected-error @+2 {{cannot locate iteration-space dimensions without tensor descriptors}}
// expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @descriptorless() {
  %unused = nv_tensor_ir.constant dense<1.0> : tensor<8x8xf32>
  results
}

//--- dynamic-persistence.mlir

// Dynamic descriptors are supported by ordinary lowering, but static
// persistence requires a compile-time grid size even with explicit tiles.
// expected-error @+2 {{cannot compute grid size: the graph's tensor shape is not statically known}}
// expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @dynamic_persistence(%a: tensor<8x?xf32>)
    -> tensor<8x?xf32> {
  %r = abs %a : tensor<8x?xf32>
  results %r : tensor<8x?xf32>
}
