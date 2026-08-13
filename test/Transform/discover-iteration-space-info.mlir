// RUN: tensor_ir-opt -discover-iteration-space-info -mlir-print-local-scope -split-input-file %s | FileCheck %s --check-prefix=DISCOVER
// RUN: tensor_ir-opt -discover-iteration-space-info -remove-iteration-space-maps -split-input-file %s | FileCheck %s --check-prefix=REMOVE

// DISCOVER-LABEL: nv_tensor_ir.graph @empty_graph
//   DISCOVER-NOT: nv_tensor_ir.iter_space_map
nv_tensor_ir.graph @empty_graph() -> () {
  results
}

// REMOVE-LABEL: nv_tensor_ir.graph @empty_graph
// REMOVE-NOT: nv_tensor_ir.iter_space_map

// -----

// DISCOVER-LABEL: nv_tensor_ir.graph @inputs_only
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>
nv_tensor_ir.graph @inputs_only(%a: tensor<10x20xf32>,
                                %b: tensor<10x20xf32>) -> () {
  results
}

// REMOVE-LABEL: nv_tensor_ir.graph @inputs_only
// REMOVE-NOT: nv_tensor_ir.iter_space_map

// -----

// DISCOVER-LABEL: nv_tensor_ir.graph @inputs_and_outputs_only
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>
nv_tensor_ir.graph @inputs_and_outputs_only(%a: tensor<10x20xf32>) -> tensor<10x20xf32> {
  results %a : tensor<10x20xf32>
}

// REMOVE-LABEL: nv_tensor_ir.graph @inputs_and_outputs_only
// REMOVE-NOT: nv_tensor_ir.iter_space_map

// -----

// DISCOVER-LABEL: nv_tensor_ir.graph @simple_elementwise
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>
nv_tensor_ir.graph @simple_elementwise(%a: tensor<10x20xf32>,
                                       %b: tensor<10x20xf32>) -> tensor<10x20xf32> {
  // DISCOVER: add
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>
  %0 = add %a, %b : tensor<10x20xf32>

  // DISCOVER: mul
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>
  %1 = mul %0, %a : tensor<10x20xf32>

  results %1 : tensor<10x20xf32>
}

// REMOVE-LABEL: nv_tensor_ir.graph @simple_elementwise
// REMOVE-NOT: nv_tensor_ir.iter_space_map

// -----

// DISCOVER-LABEL: nv_tensor_ir.graph @simple_matmul
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d2)>
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d2, d1)>
nv_tensor_ir.graph @simple_matmul(%a: tensor<10x15xf32>,
                                  %b: tensor<15x20xf32>)
                                   -> tensor<10x20xf32> {
  // DISCOVER: matmul
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1)>
  %0 = matmul(%a, %b) : (tensor<10x15xf32>,
                         tensor<15x20xf32>)
                      -> tensor<10x20xf32>
  results %0 : tensor<10x20xf32>
}

// REMOVE-LABEL: nv_tensor_ir.graph @simple_matmul
// REMOVE-NOT: nv_tensor_ir.iter_space_map

// -----

// DISCOVER-LABEL: nv_tensor_ir.graph @matmul_epilogue
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d2)>
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d2, d1)>
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1)>
nv_tensor_ir.graph @matmul_epilogue(%a: tensor<10x15xf32>,
                           %b: tensor<15x20xf32>,
                           %c: tensor<10x20xf32>)
                            -> tensor<10x20xf32> {
  // DISCOVER: matmul
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1)>
  %0 = matmul(%a, %b) : (tensor<10x15xf32>,
                         tensor<15x20xf32>)
                      -> tensor<10x20xf32>

  // DISCOVER: add
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1)>
  %1 = add %0, %c : tensor<10x20xf32>
  results %1 : tensor<10x20xf32>
}
// REMOVE-LABEL: nv_tensor_ir.graph @matmul_epilogue
// REMOVE-NOT: nv_tensor_ir.iter_space_map

// -----

// DISCOVER-LABEL: nv_tensor_ir.graph @matmul_prologue_epilogue
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d2)>
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d2, d1)>
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d2)>
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1)>
nv_tensor_ir.graph @matmul_prologue_epilogue(%a: tensor<10x15xf32>,
                                             %b: tensor<15x20xf32>,
                                             %c: tensor<10x15xf32>,
                                             %d: tensor<10x20xf32>)
                                              -> tensor<10x20xf32> {
  // DISCOVER: add
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, undef, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d2)>
  %ac = add %a, %c : tensor<10x15xf32>

  // DISCOVER: mul
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, def, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d2, d1)>
  %scaled_b = mul %b, %b : tensor<15x20xf32>

  // DISCOVER: matmul
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1)>
  %matmul = matmul(%ac, %scaled_b) : (tensor<10x15xf32>,
                                      tensor<15x20xf32>)
                                   -> tensor<10x20xf32>

  // DISCOVER: add
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1)>
  %biased = add %matmul, %d : tensor<10x20xf32>
  results %biased : tensor<10x20xf32>
}

// REMOVE-LABEL: nv_tensor_ir.graph @matmul_prologue_epilogue
// REMOVE-NOT: nv_tensor_ir.iter_space_map

// -----

// DISCOVER-LABEL: nv_tensor_ir.graph @batched_matmul
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
nv_tensor_ir.graph @batched_matmul(
  %a: tensor<4x10x15xf32>,
  %b: tensor<4x15x20xf32>
) -> tensor<4x10x20xf32> {
  // DISCOVER: matmul
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  %0 = matmul(%a, %b) : (tensor<4x10x15xf32>,
                         tensor<4x15x20xf32>)
                      -> tensor<4x10x20xf32>
  results %0 : tensor<4x10x20xf32>
}

// REMOVE-LABEL: nv_tensor_ir.graph @batched_matmul
// REMOVE-NOT: nv_tensor_ir.iter_space_map

// -----

// DISCOVER-LABEL: nv_tensor_ir.graph @high_rank_matmul
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[undef, undef, undef, undef, undef]>
// DISCOVER-SAME: nv_tensor_ir.iter_space_ids = array<i32: 0>
// DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
nv_tensor_ir.graph @high_rank_matmul(
  %a: tensor<2x3x10x15xf32>,
  %b: tensor<2x3x15x20xf32>
) -> tensor<2x3x10x20xf32> {
  // DISCOVER: matmul
  // DISCOVER-SAME: nv_tensor_ir.iter_space_dim_domains = #nv_tensor_ir<iter_space_dim_domains[def, def, def, def, def]>
  // DISCOVER-SAME: nv_tensor_ir.iter_space_id = 0 : i32
  // DISCOVER-SAME: nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
  %0 = matmul(%a, %b) : (tensor<2x3x10x15xf32>,
                         tensor<2x3x15x20xf32>)
                      -> tensor<2x3x10x20xf32>
  results %0 : tensor<2x3x10x20xf32>
}

// REMOVE-LABEL: nv_tensor_ir.graph @high_rank_matmul
// REMOVE-NOT: nv_tensor_ir.iter_space_map
