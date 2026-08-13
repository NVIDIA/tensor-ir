// RUN: tensor_ir-opt -layout-propagation-annotation -split-input-file %s | FileCheck %s

// CHECK-LABEL: @reduction_single_dim
// CHECK: reduce({{.*}} {layout = #nv_tensor_ir.reduction_source<"(128,1,(64)):(64,0,(1))", #nv_tensor_ir.tensor_source<0, 0, "(128,64):(1,128)">>}
nv_tensor_ir.graph @reduction_single_dim(
        %in0: tensor<128x64xf32> {nv_tensor_ir.stride = "(1,128)"}) -> (tensor<128x1xf32> {nv_tensor_ir.stride = "(1,0)"}) {
    %reduce = reduce(%in0)<dimensions = [1], reduction_mode = <add>> : tensor<128x64xf32> -> tensor<128x1xf32>
    results %reduce : tensor<128x1xf32>
}

// -----

// CHECK-LABEL: @reduction_multiple_dims
// CHECK: reduce({{.*}} {layout = #nv_tensor_ir.reduction_source<"(128,1,1,(16,4)):(64,0,0,(4,1))", #nv_tensor_ir.tensor_source<0, 0, "(128,16,4):(1,128,2048)">>}
nv_tensor_ir.graph @reduction_multiple_dims(
        %in0: tensor<128x16x4xf32> {nv_tensor_ir.stride = "(1,128,2048)"}) -> (tensor<128x1x1xf32> {nv_tensor_ir.stride = "(1,0,0)"}) {
    %reduce = reduce(%in0)<dimensions = [1, 2], reduction_mode = <add>> : tensor<128x16x4xf32> -> tensor<128x1x1xf32>
    results %reduce : tensor<128x1x1xf32>
}

// -----

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduction_reshape
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.reduction_source<"(2,(2,8),16,(4,16)):(16384,(1,512),2,(4096,32))", #nv_tensor_ir.tensor_source<0, 0, "(2,4,8,16,32):(1,2,8,64,1024)">>}
nv_tensor_ir.graph @reduction_reshape(
        %in0: tensor<2x4x8x16x32xf32> {nv_tensor_ir.stride = "(1,2,8,64,1024)"}) -> (tensor<2x16x16xf32> {nv_tensor_ir.stride = "(1,2,32)"}) {
    %reduce = reduce(%in0)<dimensions = [1, 3], reduction_mode = <add>> : tensor<2x4x8x16x32xf32> -> tensor<2x1x8x1x32xf32>
    %transposed = transpose %reduce permutation = [4, 3, 2, 1, 0] : tensor<2x1x8x1x32xf32> -> tensor<32x1x8x1x2xf32>
    %reshaped = reshape %transposed : tensor<32x1x8x1x2xf32> -> tensor<16x16x2xf32>
    %reshape = transpose %reshaped permutation = [2, 1, 0] : tensor<16x16x2xf32> -> tensor<2x16x16xf32>
    results %reshape : tensor<2x16x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduction_broadcast
// CHECK: broadcast {{.*}} {layout = #nv_tensor_ir.reduction_source<"(2,4,8,256,32,(4,16)):(16384,0,512,0,1,(4096,32))", #nv_tensor_ir.tensor_source<0, 0, "(2,4,8,16,32):(1,2,8,64,1024)">>}
nv_tensor_ir.graph @reduction_broadcast(
        %in0: tensor<2x4x8x16x32xf32> {nv_tensor_ir.stride = "(1,2,8,64,1024)"}) -> (tensor<2x4x8x256x32xf32> {nv_tensor_ir.stride = "(1,2,8,64,16384)"}) {
    %reduce = reduce(%in0)<dimensions = [1, 3], reduction_mode = <add>> : tensor<2x4x8x16x32xf32> -> tensor<2x1x8x1x32xf32>
    %broadcast = broadcast %reduce : tensor<2x1x8x1x32xf32> -> tensor<2x4x8x256x32xf32>
    results %broadcast : tensor<2x4x8x256x32xf32>
}

// -----

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduction_transpose
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.reduction_source<"(32,1,8,1,2,(4,16)):(1,0,512,0,16384,(4096,32))", #nv_tensor_ir.tensor_source<0, 0, "(2,4,8,16,32):(1,2,8,64,1024)">>}
nv_tensor_ir.graph @reduction_transpose(
        %in0: tensor<2x4x8x16x32xf32> {nv_tensor_ir.stride = "(1,2,8,64,1024)"}) -> (tensor<32x1x8x1x2xf32> {nv_tensor_ir.stride = "(1,0,32,0,256)"}) {
    %reduce = reduce(%in0)<dimensions = [1, 3], reduction_mode = <add>> : tensor<2x4x8x16x32xf32> -> tensor<2x1x8x1x32xf32>
    %transpose = transpose %reduce permutation = [4, 3, 2, 1, 0] : tensor<2x1x8x1x32xf32> -> tensor<32x1x8x1x2xf32>
    results %transpose : tensor<32x1x8x1x2xf32>
}
