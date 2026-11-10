// RUN: split-file %s %t
// RUN: tensor_ir-opt %t/undiscovered.mlir -split-input-file \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   -verify-diagnostics
// RUN: tensor_ir-opt %t/discovered.mlir -split-input-file \
// RUN:   -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   -verify-diagnostics
// RUN: tensor_ir-opt %t/uniform-index.mlir -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map tile-size=2" \
// RUN:   -verify-diagnostics
// RUN: tensor_ir-opt %t/dynamic-persistence.mlir -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map \
// RUN:   persistence=static sm-count=1 tile-size=2,2" -verify-diagnostics

//--- undiscovered.mlir

// Graphs that carry their own iteration-space attributes.

// An operation map establishes the iteration-space rank, but cannot replace
// the input descriptor map that the affine lowering requires.
// expected-error @+2 {{Iteration space map not found for affine map codegen}}
// expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @missing_map(%a: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %r = abs %a {
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
    : tensor<8x8xf32>
  results %r : tensor<8x8xf32>
}

// -----

// Having other descriptor attributes does not supply an iteration-space map.
// expected-error @+2 {{Iteration space map not found for affine map codegen}}
// expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @missing_map_with_strides(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8, 1)"})
    -> tensor<8x8xf32> {
  %r = abs %a {
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
    : tensor<8x8xf32>
  results %r : tensor<8x8xf32>
}

// -----

// Deliberately omit iteration space 1 from every input and output descriptor.
// The lowering must diagnose the missing dimension source instead of relying
// on assertions before dereferencing a null tensor view.

// expected-error@+2 {{failed to locate tensor dimension for iteration space 1, dimension 0}}
// expected-error@+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @missing_iteration_space_descriptor(
    %input: tensor<8x16xf16> {
      nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef]>,
      nv_tensor_ir.iter_space_ids = array<i32: 0>,
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>},
    %scale: tensor<1x16xf16> {
      nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef]>,
      nv_tensor_ir.iter_space_ids = array<i32: 0>,
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
) -> tensor<8x16xf16> {
  %input_f32 = convert %input {
    nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>,
    nv_tensor_ir.iter_space_id = 0 : i32,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<8x16xf16> -> tensor<8x16xf32>
  %scale_f32 = convert %scale {
    nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, def]>,
    nv_tensor_ir.iter_space_id = 0 : i32,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<1x16xf16> -> tensor<1x16xf32>
  %squared = mul %input_f32, %input_f32 {
    nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>,
    nv_tensor_ir.iter_space_id = 0 : i32,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<8x16xf32>
  %sum = reduce(%squared) <dimensions = [1], reduction_mode = <add>> {
    nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>,
    nv_tensor_ir.iter_space_id = 0 : i32,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<8x16xf32> -> tensor<8x1xf32>
  %inv = rsqrt %sum {
    nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>,
    nv_tensor_ir.iter_space_id = 0 : i32,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<8x1xf32>
  %inv_b = broadcast %inv {
    nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>,
    nv_tensor_ir.iter_space_id = 1 : i32,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<8x1xf32> -> tensor<8x16xf32>
  %scale_b = broadcast %scale_f32 {
    nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>,
    nv_tensor_ir.iter_space_id = 0 : i32,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<1x16xf32> -> tensor<8x16xf32>
  %normalized = mul %input_f32, %inv_b {
    nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>,
    nv_tensor_ir.iter_space_id = 1 : i32,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<8x16xf32>
  %result_f32 = mul %normalized, %scale_b {
    nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>,
    nv_tensor_ir.iter_space_id = 0 : i32,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<8x16xf32>
  %result = convert %result_f32 {
    nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>,
    nv_tensor_ir.iter_space_id = 0 : i32,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<8x16xf32> -> tensor<8x16xf16>
  results %result : tensor<8x16xf16>
}

//--- discovered.mlir

// Graphs annotated by discover-iteration-space-info.

nv_tensor_ir.graph @integer_reduction(%a: tensor<8x8xsi32>)
    -> tensor<8x1xsi32> {
  // expected-error @+2 {{Unsupported element type for reduction operation}}
  // expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.reduce'}}
  %r = reduce(%a) <dimensions = [1], reduction_mode = <add>>
    : tensor<8x8xsi32> -> tensor<8x1xsi32>
  results %r : tensor<8x1xsi32>
}

// -----

nv_tensor_ir.graph @multi_reduction(%a: tensor<8x8xf32>)
    -> tensor<1x1xf32> {
  // expected-error @+2 {{Multiple dimensions are not supported for reduction}}
  // expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.reduce'}}
  %r = reduce(%a) <dimensions = [0, 1], reduction_mode = <add>>
    : tensor<8x8xf32> -> tensor<1x1xf32>
  results %r : tensor<1x1xf32>
}

// -----

nv_tensor_ir.graph @customize_reduction(%a: tensor<8xf32>)
    -> tensor<1xf32> {
  // expected-error @+2 {{Customize reduction mode is not supported}}
  // expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.reduce'}}
  %r = reduce(%a) <dimensions = [0], reduction_mode = <customize>>
    : tensor<8xf32> -> tensor<1xf32>
  results %r : tensor<1xf32>
}

// -----

// An empty extent can lower with explicit tiles, but automatic tile analysis
// cannot currently select tiles for it. This is not a rank-zero graph.
// expected-error @+2 {{cannot analyze tile size: the graph's tensor shape or stride is not statically known}}
// expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @empty_auto(%a: tensor<0x8xf32>) -> tensor<0x8xf32> {
  %r = abs %a : tensor<0x8xf32>
  results %r : tensor<0x8xf32>
}

// -----

// An unused tensor computation has an iteration space but no input or output
// descriptor from which lowering can obtain its extents.
// expected-error @+2 {{cannot locate iteration-space dimensions without tensor descriptors}}
// expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.graph'}}
nv_tensor_ir.graph @descriptorless() {
  %unused = nv_tensor_ir.constant dense<1.0> : tensor<8x8xf32>
  results
}

// -----

nv_tensor_ir.graph @matmul_accumulator_affine_unsupported(
    %a: tensor<32x128xf32>,
    %b: tensor<128x16xf32>,
    %acc: tensor<32x16xf32>
    ) -> tensor<32x16xf32> {
  // expected-error@below {{custom accumulator is not supported by TensorToCudaTile}}
  %out = matmul(%a, %b) accum(%acc : tensor<32x16xf32>)
    : (tensor<32x128xf32>, tensor<128x16xf32>) -> tensor<32x16xf32>
  results %out : tensor<32x16xf32>
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
