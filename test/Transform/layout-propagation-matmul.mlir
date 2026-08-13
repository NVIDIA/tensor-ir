// RUN: tensor_ir-opt -layout-propagation-annotation -split-input-file %s | FileCheck %s

// CHECK-LABEL: @matmul_no_batch
// CHECK: matmul({{.*}} {layout = #nv_tensor_ir.matmul_source<"(8,16,32):(512,32,1)", 1, 8, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(8,32):(1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(32,16):(1,32)">>}
nv_tensor_ir.graph @matmul_no_batch(
        %in0: tensor<8x32xf32> {nv_tensor_ir.stride = "(1,8)"},
        %in1: tensor<32x16xf32> {nv_tensor_ir.stride = "(1,32)"}) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
    results %matmul : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: @matmul_batch_one
// CHECK: matmul({{.*}} {layout = #nv_tensor_ir.matmul_source<"(4,8,16,32):(4096,512,32,1)", 4, 8, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(4,8,32):(1,4,32)">, #nv_tensor_ir.tensor_source<1, 0, "(4,32,16):(1,4,128)">>}
nv_tensor_ir.graph @matmul_batch_one(
        %in0: tensor<4x8x32xf32> {nv_tensor_ir.stride = "(1,4,32)"},
        %in1: tensor<4x32x16xf32> {nv_tensor_ir.stride = "(1,4,128)"}) -> (tensor<4x8x16xf32> {nv_tensor_ir.stride = "(1,4,32)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
    results %matmul : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @matmul_batch_two
// CHECK: matmul({{.*}} {layout = #nv_tensor_ir.matmul_source<"(2,4,8,16,32):(16384,4096,512,32,1)", 8, 8, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(2,4,8,32):(1,2,8,64)">, #nv_tensor_ir.tensor_source<1, 0, "(2,4,32,16):(1,2,8,256)">>}
nv_tensor_ir.graph @matmul_batch_two(
        %in0: tensor<2x4x8x32xf32> {nv_tensor_ir.stride = "(1,2,8,64)"},
        %in1: tensor<2x4x32x16xf32> {nv_tensor_ir.stride = "(1,2,8,256)"}) -> (tensor<2x4x8x16xf32> {nv_tensor_ir.stride = "(1,2,8,64)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<2x4x8x32xf32>, tensor<2x4x32x16xf32>) -> tensor<2x4x8x16xf32>
    results %matmul : tensor<2x4x8x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_reshape
// CHECK: reshape {{.*}} {layout = #nv_tensor_ir.matmul_source<"((16,8,4),32):((32,512,4096),1)", 4, 8, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(4,8,32):(1,4,32)">, #nv_tensor_ir.tensor_source<1, 0, "(4,32,16):(1,4,128)">>}
nv_tensor_ir.graph @matmul_reshape(
        %in0: tensor<4x8x32xf32> {nv_tensor_ir.stride = "(1,4,32)"},
        %in1: tensor<4x32x16xf32> {nv_tensor_ir.stride = "(1,4,128)"}) -> (tensor<512xf32>) {
    %matmul = matmul(%in0, %in1) : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
    %transposed = transpose %matmul permutation = [2, 1, 0] : tensor<4x8x16xf32> -> tensor<16x8x4xf32>
    %reshape = reshape %transposed : tensor<16x8x4xf32> -> tensor<512xf32>
    results %reshape : tensor<512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_broadcast
// CHECK: broadcast {{.*}} {layout = #nv_tensor_ir.matmul_source<"(4,8,16,32):(0,512,32,1)", 1, 8, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(1,8,32):(0,1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(1,32,16):(0,1,32)">>}
nv_tensor_ir.graph @matmul_broadcast(
        %in0: tensor<1x8x32xf32> {nv_tensor_ir.stride = "(0,1,8)"},
        %in1: tensor<1x32x16xf32> {nv_tensor_ir.stride = "(0,1,32)"}) -> (tensor<4x8x16xf32> {nv_tensor_ir.stride = "(1,4,32)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<1x8x32xf32>, tensor<1x32x16xf32>) -> tensor<1x8x16xf32>
    %broadcast = broadcast %matmul : tensor<1x8x16xf32> -> tensor<4x8x16xf32>
    results %broadcast : tensor<4x8x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_transpose
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.matmul_source<"(16,8,4,32):(32,512,4096,1)", 4, 8, 16, 32, #nv_tensor_ir.tensor_source<0, 0, "(4,8,32):(1,4,32)">, #nv_tensor_ir.tensor_source<1, 0, "(4,32,16):(1,4,128)">>}
nv_tensor_ir.graph @matmul_transpose(
        %in0: tensor<4x8x32xf32> {nv_tensor_ir.stride = "(1,4,32)"},
        %in1: tensor<4x32x16xf32> {nv_tensor_ir.stride = "(1,4,128)"}) -> (tensor<16x8x4xf32> {nv_tensor_ir.stride = "(1,16,128)"}) {
    %matmul = matmul(%in0, %in1) : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
    %transpose = transpose %matmul permutation = [2, 1, 0] : tensor<4x8x16xf32> -> tensor<16x8x4xf32>
    results %transpose : tensor<16x8x4xf32>
}
