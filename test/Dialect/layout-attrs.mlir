// RUN: tensor_ir-opt -mlir-print-local-scope -split-input-file %s | FileCheck %s
// RUN: tensor_ir-opt -mlir-print-local-scope -split-input-file %s \
// RUN:   | tensor_ir-opt -mlir-print-local-scope -split-input-file | FileCheck %s

// Tests to verify layout propagation attributes' constraints, parsing and printing.

//===----------------------------------------------------------------------===//
// IterSpaceDimDomainsAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @iter_space_dim_domains
// CHECK-SAME: attributes {domains = #nv_tensor_ir<iter_space_dim_domains[def, undef, def]>}
func.func @iter_space_dim_domains() attributes {domains = #nv_tensor_ir<iter_space_dim_domains[def, undef, def]>} {
  return
}

//===----------------------------------------------------------------------===//
// TensorSourceAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tensor_source_1d
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">}
func.func @tensor_source_1d(%in0: tensor<16xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">} {
  return
}

// CHECK-LABEL: func.func @tensor_source_3d
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<1, 0, "(2,4,8):(1,2,8)">}
func.func @tensor_source_3d(%in0: tensor<2x4x8xf32>, %in1: tensor<2x4x8xf32>) attributes {layout = #nv_tensor_ir.tensor_source<1, 0, "(2,4,8):(1,2,8)">} {
  return
}

// CHECK-LABEL: func.func @tensor_source_with_offset
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 128, "(16):(1)">}
func.func @tensor_source_with_offset(%in0: tensor<256xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 128, "(16):(1)">} {
  return
}

// CHECK-LABEL: func.func @tensor_source_negative_tensor_id
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<-1, 0, "(4,4):(0,0)">}
func.func @tensor_source_negative_tensor_id() attributes {layout = #nv_tensor_ir.tensor_source<-1, 0, "(4,4):(0,0)">} {
  return
}

// CHECK-LABEL: func.func @tensor_source_hierarchical
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "((2,4),(8,2)):((1,2),(8,64))">}
func.func @tensor_source_hierarchical(%in0: tensor<8x16xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "((2,4),(8,2)):((1,2),(8,64))">} {
  return
}

//===----------------------------------------------------------------------===//
// CompositeSourceAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @composite_source_two
// CHECK-SAME: attributes {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(16):(1)">>}
func.func @composite_source_two(%in0: tensor<16xf32>, %in1: tensor<16xf32>) attributes {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(16):(1)">>} {
  return
}

// CHECK-LABEL: func.func @composite_source_three
// CHECK-SAME: attributes {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(8):(1)">, #nv_tensor_ir.tensor_source<2, 0, "(8):(1)">>}
func.func @composite_source_three(%in0: tensor<8xf32>, %in1: tensor<8xf32>, %in2: tensor<8xf32>) attributes {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(8):(1)">, #nv_tensor_ir.tensor_source<2, 0, "(8):(1)">>} {
  return
}

// CHECK-LABEL: func.func @composite_source_with_offsets
// CHECK-SAME: attributes {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<0, 16, "(16):(1)">>}
func.func @composite_source_with_offsets(%in0: tensor<32xf32>) attributes {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<0, 16, "(16):(1)">>} {
  return
}

// CHECK-LABEL: func.func @composite_source_different_strides
// CHECK-SAME: attributes {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(4,8):(1,4)">, #nv_tensor_ir.tensor_source<1, 0, "(4,8):(8,1)">>}
func.func @composite_source_different_strides(%in0: tensor<4x8xf32>, %in1: tensor<4x8xf32>) attributes {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(4,8):(1,4)">, #nv_tensor_ir.tensor_source<1, 0, "(4,8):(8,1)">>} {
  return
}

//===----------------------------------------------------------------------===//
// ConcatSourceAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @concat_source_two
// CHECK-SAME: attributes {layout = #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(16):(1)">>}
func.func @concat_source_two(%in0: tensor<16xf32>, %in1: tensor<16xf32>) attributes {layout = #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(16):(1)">>} {
  return
}

// CHECK-LABEL: func.func @concat_source_three
// CHECK-SAME: attributes {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(16,2):(1,16)">, #nv_tensor_ir.tensor_source<1, 0, "(16,4):(1,16)">, #nv_tensor_ir.tensor_source<2, 0, "(16,6):(1,16)">>}
func.func @concat_source_three(%in0: tensor<16x2xf32>, %in1: tensor<16x4xf32>, %in2: tensor<16x8xf32>) attributes {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(16,2):(1,16)">, #nv_tensor_ir.tensor_source<1, 0, "(16,4):(1,16)">, #nv_tensor_ir.tensor_source<2, 0, "(16,6):(1,16)">>} {
  return
}

// CHECK-LABEL: func.func @concat_source_unit_dims
// CHECK-SAME: attributes {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(16,1):(1,0)">, #nv_tensor_ir.tensor_source<1, 0, "(16,1):(1,0)">>}
func.func @concat_source_unit_dims(%in0: tensor<16x1xf32>, %in1: tensor<16x1xf32>) attributes {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(16,1):(1,0)">, #nv_tensor_ir.tensor_source<1, 0, "(16,1):(1,0)">>} {
  return
}

//===----------------------------------------------------------------------===//
// ReductionSourceAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @reduction_source_one
// CHECK-SAME: attributes {layout = #nv_tensor_ir.reduction_source<"(16,(32)):(1,(16))", #nv_tensor_ir.tensor_source<0, 0, "(16,32):(1,16)">>}
func.func @reduction_source_one(%in0: tensor<16x32xf32>) attributes {layout = #nv_tensor_ir.reduction_source<"(16,(32)):(1,(16))", #nv_tensor_ir.tensor_source<0, 0, "(16,32):(1,16)">>} {
  return
}

// CHECK-LABEL: func.func @reduction_source_two
// CHECK-SAME: attributes {layout = #nv_tensor_ir.reduction_source<"(16,(8,4)):(1,(16,128))", #nv_tensor_ir.tensor_source<0, 0, "(16,8,4):(1,16,128)">>}
func.func @reduction_source_two(%in0: tensor<16x8x4xf32>) attributes {layout = #nv_tensor_ir.reduction_source<"(16,(8,4)):(1,(16,128))", #nv_tensor_ir.tensor_source<0, 0, "(16,8,4):(1,16,128)">>} {
  return
}

//===----------------------------------------------------------------------===//
// MatmulSourceAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @matmul_source_no_batch
// CHECK-SAME: attributes {layout = #nv_tensor_ir.matmul_source<"(16,16,32):(1,16,256)", 1, 16, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(16,32):(1,16)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(1,32)">>}
func.func @matmul_source_no_batch(%in0: tensor<16x32xf32>, %in1: tensor<32x16xf32>) attributes {layout = #nv_tensor_ir.matmul_source<"(16,16,32):(1,16,256)", 1, 16, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(16,32):(1,16)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(1,32)">>} {
  return
}

// CHECK-LABEL: func.func @matmul_source_batch_one
// CHECK-SAME: attributes {layout = #nv_tensor_ir.matmul_source<"(4,16,16,32):(1,4,64,1024)", 4, 16, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(4,16,32):(1,4,64)">, #nv_tensor_ir.tensor_source<1, 0, "(4,32,16):(1,4,128)">>}
func.func @matmul_source_batch_one(%in0: tensor<4x16x32xf32>, %in1: tensor<4x32x16xf32>) attributes {layout = #nv_tensor_ir.matmul_source<"(4,16,16,32):(1,4,64,1024)", 4, 16, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(4,16,32):(1,4,64)">, #nv_tensor_ir.tensor_source<1, 0, "(4,32,16):(1,4,128)">>} {
  return
}

// CHECK-LABEL: func.func @matmul_source_batch_two
// CHECK-SAME: attributes {layout = #nv_tensor_ir.matmul_source<"(2,4,16,16,32):(1,2,8,128,2048)", 8, 16, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(2,4,16,32):(1,2,8,128)">, #nv_tensor_ir.tensor_source<1, 0, "(2,4,32,16):(1,2,8,256)">>}
func.func @matmul_source_batch_two(%in0: tensor<2x4x16x32xf32>, %in1: tensor<2x4x32x16xf32>) attributes {layout = #nv_tensor_ir.matmul_source<"(2,4,16,16,32):(1,2,8,128,2048)", 8, 16, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(2,4,16,32):(1,2,8,128)">, #nv_tensor_ir.tensor_source<1, 0, "(2,4,32,16):(1,2,8,256)">>} {
  return
}

//===----------------------------------------------------------------------===//
// Dynamic shapes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tensor_dynamic_shape_1d
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?):(1)", [0]>}
func.func @tensor_dynamic_shape_1d(%in0: tensor<?xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?):(1)", [0]>} {
  return
}

// CHECK-LABEL: func.func @tensor_dynamic_broadcast
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,32):(0,1)">}
func.func @tensor_dynamic_broadcast(%in0: tensor<1x32xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,32):(0,1)">} {
  return
}

// CHECK-LABEL: func.func @tensor_dynamic_stride_1d
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(32):(?)", [0]>}
func.func @tensor_dynamic_stride_1d(%in0: tensor<32xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(32):(?)", [0]>} {
  return
}

// CHECK-LABEL: func.func @tensor_dynamic_shape_and_stride_1d
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?):(?)", [0, 1]>}
func.func @tensor_dynamic_shape_and_stride_1d(%in0: tensor<?xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?):(?)", [0, 1]>} {
  return
}

// CHECK-LABEL: func.func @tensor_dynamic_shape_2d
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(1,1024)", [0, 1]>}
func.func @tensor_dynamic_shape_2d(%in0: tensor<?x?xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(1,1024)", [0, 1]>} {
  return
}

// CHECK-LABEL: func.func @tensor_dynamic_stride_2d
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(32,64):(?,?)", [0, 1]>}
func.func @tensor_dynamic_stride_2d(%in0: tensor<32x64xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(32,64):(?,?)", [0, 1]>} {
  return
}

// CHECK-LABEL: func.func @tensor_dynamic_shape_and_stride_2d
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(?,?)", [0, 1, 2, 3]>}
func.func @tensor_dynamic_shape_and_stride_2d(%in0: tensor<?x?xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(?,?)", [0, 1, 2, 3]>} {
  return
}

// CHECK-LABEL: func.func @tensor_dynamic_mixed
// CHECK-SAME: attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,32):(1,?)", [0, 1]>}
func.func @tensor_dynamic_mixed(%in0: tensor<?x32xf32>) attributes {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,32):(1,?)", [0, 1]>} {
  return
}

// -----

#layout_attr_add = #nv_tensor_ir.tensor_source<0, 0, "(32,32,32):(1024,32,1)">
#layout_attr_mul = #nv_tensor_ir.tensor_source<0, 0, "(32,32,32):(1024,32,1)">

nv_tensor_ir.graph @print_parse_two_layouts(%arg0: tensor<32x32x32xf32>) -> (tensor<32x32x32xf32>) {
    // CHECK: %{{.*}} = add {{.*}} {nv_tensor_ir.layout = #nv_tensor_ir.tensor_source<0, 0, "(32,32,32):(1024,32,1)">}
    %add = add %arg0, %arg0 {nv_tensor_ir.layout = #layout_attr_add} : tensor<32x32x32xf32>

    // CHECK: %{{.*}} = mul {{.*}} {nv_tensor_ir.layout = #nv_tensor_ir.tensor_source<0, 0, "(32,32,32):(1024,32,1)">}
    %mul = mul %add, %add {nv_tensor_ir.layout = #nv_tensor_ir.tensor_source<0, 0, "(32,32,32):(1024,32,1)">} : tensor<32x32x32xf32>
    results %add : tensor<32x32x32xf32>
}

// -----

#layout_attr_1 = #nv_tensor_ir.tensor_source<0, 0, "(32,32,32):(1024,32,1)">
#layout_attr_2 = #nv_tensor_ir.tensor_source<0, 0, "(32,32,32):(1024,1,32)">
#composite_layout = #nv_tensor_ir.composite_source<#layout_attr_1, #layout_attr_2>

nv_tensor_ir.graph @composite_layout_attr(%arg0: tensor<32x32x32xf32>) -> (tensor<32x32x32xf32>) {
    %transpose = transpose %arg0 permutation = [2, 0, 1] : tensor<32x32x32xf32> -> tensor<32x32x32xf32>
    // CHECK: {{.*}} = add {{.*}} {nv_tensor_ir.layout = #nv_tensor_ir.composite_source<
    // CHECK-SAME:                    #nv_tensor_ir.tensor_source<0, 0, "(32,32,32):(1024,32,1)">,
    // CHECK-SAME:                    #nv_tensor_ir.tensor_source<0, 0, "(32,32,32):(1024,1,32)">>}
    // CHECK: results %{{.*}} : tensor<32x32x32xf32>
    %add = add %transpose, %arg0 {nv_tensor_ir.layout = #composite_layout} : tensor<32x32x32xf32>
    results %add : tensor<32x32x32xf32>
}
