// RUN: tensor_ir-opt -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" %s --verify-diagnostics

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
