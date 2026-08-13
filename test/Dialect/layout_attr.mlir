// RUN: tensor_ir-opt %s -split-input-file | tensor_ir-opt | FileCheck %s

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
