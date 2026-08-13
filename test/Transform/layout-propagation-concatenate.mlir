// RUN: tensor_ir-opt -layout-propagation-annotation -split-input-file %s | FileCheck %s

// CHECK-LABEL: @concatenate_simple
// CHECK: concatenate {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(8):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(16):(1)">>}
nv_tensor_ir.graph @concatenate_simple(
        %in0: tensor<8xf32>,
        %in1: tensor<16xf32>) -> (tensor<24xf32>) {
    %concat = concatenate %in0, %in1 dimension = 0
        : (tensor<8xf32>, tensor<16xf32>) -> tensor<24xf32>
    results %concat : tensor<24xf32>
}

// -----

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reshape_left_concat_dim_same
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(2,32):(1,2)">, #nv_tensor_ir.tensor_source<1, 0, "(4,32):(1,4)">>}
nv_tensor_ir.graph @reshape_left_concat_dim_same(
        %in0: tensor<2x4x8xf32> {nv_tensor_ir.stride = "(1,2,8)"},
        %in1: tensor<4x4x8xf32> {nv_tensor_ir.stride = "(1,4,16)"}) -> (tensor<6x32xf32> {nv_tensor_ir.stride = "(1,6)"}) {
    %concat = concatenate %in0, %in1 dimension = 0
        : (tensor<2x4x8xf32>, tensor<4x4x8xf32>) -> tensor<6x4x8xf32>
    %transposed = transpose %concat permutation = [2, 1, 0]
        : tensor<6x4x8xf32> -> tensor<8x4x6xf32>
    %reshaped = reshape %transposed : tensor<8x4x6xf32> -> tensor<32x6xf32>
    %out = transpose %reshaped permutation = [1, 0]
        : tensor<32x6xf32> -> tensor<6x32xf32>
    results %out : tensor<6x32xf32>
}

// -----

// CHECK-LABEL: @reshape_left_concat_dim_split
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(2,1,32):(1,0,2)">, #nv_tensor_ir.tensor_source<1, 0, "(2,2,32):(1,2,4)">>}
nv_tensor_ir.graph @reshape_left_concat_dim_split(
        %in0: tensor<2x4x8xf32> {nv_tensor_ir.stride = "(1,2,8)"},
        %in1: tensor<4x4x8xf32> {nv_tensor_ir.stride = "(1,4,16)"}) -> (tensor<2x3x32xf32> {nv_tensor_ir.stride = "(1,2,6)"}) {
    %concat = concatenate %in0, %in1 dimension = 0
        : (tensor<2x4x8xf32>, tensor<4x4x8xf32>) -> tensor<6x4x8xf32>
    %transposed = transpose %concat permutation = [2, 1, 0]
        : tensor<6x4x8xf32> -> tensor<8x4x6xf32>
    %reshaped = reshape %transposed : tensor<8x4x6xf32> -> tensor<32x3x2xf32>
    %out = transpose %reshaped permutation = [2, 1, 0]
        : tensor<32x3x2xf32> -> tensor<2x3x32xf32>
    results %out : tensor<2x3x32xf32>
}

// -----

// CHECK-LABEL: @reshape_middle_concat_dim_same
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 2, #nv_tensor_ir.tensor_source<0, 0, "(2,2,2,8):(1,2,4,8)">, #nv_tensor_ir.tensor_source<1, 0, "(2,2,4,8):(1,2,4,16)">>}
nv_tensor_ir.graph @reshape_middle_concat_dim_same(
        %in0: tensor<4x2x8xf32> {nv_tensor_ir.stride = "(1,4,8)"},
        %in1: tensor<4x4x8xf32> {nv_tensor_ir.stride = "(1,4,16)"}) -> (tensor<2x2x6x8xf32> {nv_tensor_ir.stride = "(1,2,4,24)"}) {
    %concat = concatenate %in0, %in1 dimension = 1
        : (tensor<4x2x8xf32>, tensor<4x4x8xf32>) -> tensor<4x6x8xf32>
    %transposed = transpose %concat permutation = [2, 1, 0]
        : tensor<4x6x8xf32> -> tensor<8x6x4xf32>
    %reshaped = reshape %transposed : tensor<8x6x4xf32> -> tensor<8x6x2x2xf32>
    %out = transpose %reshaped permutation = [3, 2, 1, 0]
        : tensor<8x6x2x2xf32> -> tensor<2x2x6x8xf32>
    results %out : tensor<2x2x6x8xf32>
}

// -----

// CHECK-LABEL: @reshape_middle_concat_dim_split
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(8,1,8):(1,0,8)">, #nv_tensor_ir.tensor_source<1, 0, "(8,2,8):(1,8,16)">>}
nv_tensor_ir.graph @reshape_middle_concat_dim_split(
        %in0: tensor<4x2x8xf32> {nv_tensor_ir.stride = "(1,4,8)"},
        %in1: tensor<4x4x8xf32> {nv_tensor_ir.stride = "(1,4,16)"}) -> (tensor<8x3x8xf32> {nv_tensor_ir.stride = "(1,8,24)"}) {
    %concat = concatenate %in0, %in1 dimension = 1
        : (tensor<4x2x8xf32>, tensor<4x4x8xf32>) -> tensor<4x6x8xf32>
    %transposed = transpose %concat permutation = [2, 1, 0]
        : tensor<4x6x8xf32> -> tensor<8x6x4xf32>
    %reshaped = reshape %transposed : tensor<8x6x4xf32> -> tensor<8x3x8xf32>
    %out = transpose %reshaped permutation = [2, 1, 0]
        : tensor<8x3x8xf32> -> tensor<8x3x8xf32>
    results %out : tensor<8x3x8xf32>
}

// -----

// CHECK-LABEL: @reshape_middle_concat_dim_join
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(8,8):(1,8)">, #nv_tensor_ir.tensor_source<1, 0, "(16,8):(1,16)">>}
nv_tensor_ir.graph @reshape_middle_concat_dim_join(
        %in0: tensor<4x2x8xf32> {nv_tensor_ir.stride = "(1,4,8)"},
        %in1: tensor<4x4x8xf32> {nv_tensor_ir.stride = "(1,4,16)"}) -> (tensor<24x8xf32> {nv_tensor_ir.stride = "(1,24)"}) {
    %concat = concatenate %in0, %in1 dimension = 1
        : (tensor<4x2x8xf32>, tensor<4x4x8xf32>) -> tensor<4x6x8xf32>
    %transposed = transpose %concat permutation = [2, 1, 0]
        : tensor<4x6x8xf32> -> tensor<8x6x4xf32>
    %reshaped = reshape %transposed : tensor<8x6x4xf32> -> tensor<8x24xf32>
    %out = transpose %reshaped permutation = [1, 0]
        : tensor<8x24xf32> -> tensor<24x8xf32>
    results %out : tensor<24x8xf32>
}

// -----

// CHECK-LABEL: @reshape_right_concat_dim_same
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 3, #nv_tensor_ir.tensor_source<0, 0, "(4,2,4,2):(1,4,8,32)">, #nv_tensor_ir.tensor_source<1, 0, "(4,2,4,4):(1,4,8,32)">>}
nv_tensor_ir.graph @reshape_right_concat_dim_same(
        %in0: tensor<4x8x2xf32> {nv_tensor_ir.stride = "(1,4,32)"},
        %in1: tensor<4x8x4xf32> {nv_tensor_ir.stride = "(1,4,32)"}) -> (tensor<4x2x4x6xf32> {nv_tensor_ir.stride = "(1,4,8,32)"}) {
    %concat = concatenate %in0, %in1 dimension = 2
        : (tensor<4x8x2xf32>, tensor<4x8x4xf32>) -> tensor<4x8x6xf32>
    %transposed = transpose %concat permutation = [2, 1, 0]
        : tensor<4x8x6xf32> -> tensor<6x8x4xf32>
    %reshaped = reshape %transposed : tensor<6x8x4xf32> -> tensor<6x4x2x4xf32>
    %out = transpose %reshaped permutation = [3, 2, 1, 0]
        : tensor<6x4x2x4xf32> -> tensor<4x2x4x6xf32>
    results %out : tensor<4x2x4x6xf32>
}

// -----

// CHECK-LABEL: @reshape_right_concat_dim_split
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 2, #nv_tensor_ir.tensor_source<0, 0, "(8,8,1):(1,8,0)">, #nv_tensor_ir.tensor_source<1, 0, "(8,8,2):(1,8,64)">>}
nv_tensor_ir.graph @reshape_right_concat_dim_split(
        %in0: tensor<4x8x2xf32> {nv_tensor_ir.stride = "(1,4,32)"},
        %in1: tensor<4x8x4xf32> {nv_tensor_ir.stride = "(1,4,32)"}) -> (tensor<8x8x3xf32> {nv_tensor_ir.stride = "(1,8,64)"}) {
    %concat = concatenate %in0, %in1 dimension = 2
        : (tensor<4x8x2xf32>, tensor<4x8x4xf32>) -> tensor<4x8x6xf32>
    %transposed = transpose %concat permutation = [2, 1, 0]
        : tensor<4x8x6xf32> -> tensor<6x8x4xf32>
    %reshaped = reshape %transposed : tensor<6x8x4xf32> -> tensor<3x8x8xf32>
    %out = transpose %reshaped permutation = [2, 1, 0]
        : tensor<3x8x8xf32> -> tensor<8x8x3xf32>
    results %out : tensor<8x8x3xf32>
}

// -----

// CHECK-LABEL: @reshape_right_concat_dim_join
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">, #nv_tensor_ir.tensor_source<1, 0, "(4,32):(1,4)">>}
nv_tensor_ir.graph @reshape_right_concat_dim_join(
        %in0: tensor<4x8x2xf32> {nv_tensor_ir.stride = "(1,4,32)"},
        %in1: tensor<4x8x4xf32> {nv_tensor_ir.stride = "(1,4,32)"}) -> (tensor<4x48xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %concat = concatenate %in0, %in1 dimension = 2
        : (tensor<4x8x2xf32>, tensor<4x8x4xf32>) -> tensor<4x8x6xf32>
    %transposed = transpose %concat permutation = [2, 1, 0]
        : tensor<4x8x6xf32> -> tensor<6x8x4xf32>
    %reshaped = reshape %transposed : tensor<6x8x4xf32> -> tensor<48x4xf32>
    %out = transpose %reshaped permutation = [1, 0]
        : tensor<48x4xf32> -> tensor<4x48xf32>
    results %out : tensor<4x48xf32>
}

// -----

// CHECK-LABEL: @reshape_left_concat_dim
// CHECK: reshape {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(1,2,32):(0,32,1)">, #nv_tensor_ir.tensor_source<1, 0, "(2,2,32):(64,32,1)">>}
nv_tensor_ir.graph @reshape_left_concat_dim(
        %in0: tensor<2x4x8xf32> {nv_tensor_ir.stride = "(32,8,1)"},
        %in1: tensor<4x4x8xf32> {nv_tensor_ir.stride = "(32,8,1)"}) -> tensor<3x2x32xf32> {
    %concat = concatenate %in0, %in1 dimension = 0
        : (tensor<2x4x8xf32>, tensor<4x4x8xf32>) -> tensor<6x4x8xf32>
    %out = reshape %concat : tensor<6x4x8xf32> -> tensor<3x2x32xf32>
    results %out : tensor<3x2x32xf32>
}

// -----

// CHECK-LABEL: @reshape_middle_concat_dim
// CHECK: reshape {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(4,1,16):(16,0,1)">, #nv_tensor_ir.tensor_source<1, 0, "(4,2,16):(32,16,1)">>}
nv_tensor_ir.graph @reshape_middle_concat_dim(
        %in0: tensor<4x2x8xf32> {nv_tensor_ir.stride = "(16,8,1)"},
        %in1: tensor<4x4x8xf32> {nv_tensor_ir.stride = "(32,8,1)"}) -> tensor<4x3x16xf32> {
    %concat = concatenate %in0, %in1 dimension = 1
        : (tensor<4x2x8xf32>, tensor<4x4x8xf32>) -> tensor<4x6x8xf32>
    %out = reshape %concat : tensor<4x6x8xf32> -> tensor<4x3x16xf32>
    results %out : tensor<4x3x16xf32>
}

// -----

// CHECK-LABEL: @reshape_right_concat_dim
// CHECK: reshape {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(32,1,2):(2,0,1)">, #nv_tensor_ir.tensor_source<1, 0, "(32,2,2):(4,2,1)">>}
nv_tensor_ir.graph @reshape_right_concat_dim(
        %in0: tensor<4x8x2xf32> {nv_tensor_ir.stride = "(16,2,1)"},
        %in1: tensor<4x8x4xf32> {nv_tensor_ir.stride = "(32,4,1)"}) -> tensor<32x3x2xf32> {
    %concat = concatenate %in0, %in1 dimension = 2
        : (tensor<4x8x2xf32>, tensor<4x8x4xf32>) -> tensor<4x8x6xf32>
    %out = reshape %concat : tensor<4x8x6xf32> -> tensor<32x3x2xf32>
    results %out : tensor<32x3x2xf32>
}

// -----

// CHECK-LABEL: @reshape_skip_unit_dims
// CHECK: reshape {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(1,1,10):(0,0,1)">, #nv_tensor_ir.tensor_source<1, 0, "(1,2,10):(0,10,1)">>}
nv_tensor_ir.graph @reshape_skip_unit_dims(
        %in0: tensor<10xf32>,
        %in1: tensor<20xf32>) -> tensor<1x3x10xf32> {
    %concat = concatenate %in0, %in1 dimension = 0
        : (tensor<10xf32>, tensor<20xf32>) -> tensor<30xf32>
    %out = reshape %concat : tensor<30xf32> -> tensor<1x3x10xf32>
    results %out : tensor<1x3x10xf32>
}

// -----

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @broadcast_concat
// CHECK: broadcast {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(4,1,8):(0,0,0)">, #nv_tensor_ir.tensor_source<1, 0, "(4,2,8):(0,1,0)">>}
nv_tensor_ir.graph @broadcast_concat(
        %in0: tensor<1x1x1xf32> {nv_tensor_ir.stride = "(0,0,0)"},
        %in1: tensor<1x2x1xf32> {nv_tensor_ir.stride = "(0,1,0)"}) -> (tensor<4x3x8xf32> {nv_tensor_ir.stride = "(1,4,12)"}) {
    %concat = concatenate %in0, %in1 dimension = 1
        : (tensor<1x1x1xf32>, tensor<1x2x1xf32>) -> tensor<1x3x1xf32>
    %out = broadcast %concat : tensor<1x3x1xf32> -> tensor<4x3x8xf32>
    results %out : tensor<4x3x8xf32>
}

// -----

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @transpose_concat
// CHECK: transpose {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 0, #nv_tensor_ir.tensor_source<0, 0, "(2,8):(8,1)">, #nv_tensor_ir.tensor_source<1, 0, "(4,8):(8,1)">>}
nv_tensor_ir.graph @transpose_concat(
        %in0: tensor<8x2xf32> {nv_tensor_ir.stride = "(1,8)"},
        %in1: tensor<8x4xf32> {nv_tensor_ir.stride = "(1,8)"}) -> (tensor<6x8xf32> {nv_tensor_ir.stride = "(1,6)"}) {
    %concat = concatenate %in0, %in1 dimension = 1
        : (tensor<8x2xf32>, tensor<8x4xf32>) -> tensor<8x6xf32>
    %out = transpose %concat permutation = [1, 0] : tensor<8x6xf32> -> tensor<6x8xf32>
    results %out : tensor<6x8xf32>
}

// -----

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @slice_concat_not_unit_step
// CHECK: slice {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(4,1):(4,0)">, #nv_tensor_ir.tensor_source<1, 0, "(4,2):(8,2)">, #nv_tensor_ir.tensor_source<2, 0, "(4,4):(16,2)">>}
nv_tensor_ir.graph @slice_concat_not_unit_step(
        %in0: tensor<8x2xf32>,
        %in1: tensor<8x4xf32>,
        %in2: tensor<8x8xf32>) -> tensor<4x7xf32> {
    %concat = concatenate %in0, %in1, %in2 dimension = 1
        : (tensor<8x2xf32>, tensor<8x4xf32>, tensor<8x8xf32>) -> tensor<8x14xf32>
    %out = slice %concat starts = [0, 0] limits = [8, 14] strides = [2, 2] : tensor<8x14xf32> -> tensor<4x7xf32>
    results %out : tensor<4x7xf32>
}

// -----

// CHECK-LABEL: @slice_concat_discard_first_source
// CHECK: slice {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(4,1)">, #nv_tensor_ir.tensor_source<2, 0, "(8,6):(8,1)">, [1, 2]>}
nv_tensor_ir.graph @slice_concat_discard_first_source(
        %in0: tensor<8x2xf32>,
        %in1: tensor<8x4xf32>,
        %in2: tensor<8x8xf32>) -> tensor<8x10xf32> {
    %concat = concatenate %in0, %in1, %in2 dimension = 1
        : (tensor<8x2xf32>, tensor<8x4xf32>, tensor<8x8xf32>) -> tensor<8x14xf32>
    %out = slice %concat starts = [0, 2] limits = [8, 12] strides = [1, 1] : tensor<8x14xf32> -> tensor<8x10xf32>
    results %out : tensor<8x10xf32>
}

// -----

// CHECK-LABEL: @slice_concat_discard_last_source
// CHECK: slice {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 1, "(8,1):(2,0)">, #nv_tensor_ir.tensor_source<1, 0, "(8,3):(4,1)">, [0, 1]>}
nv_tensor_ir.graph @slice_concat_discard_last_source(
        %in0: tensor<8x2xf32>,
        %in1: tensor<8x4xf32>,
        %in2: tensor<8x8xf32>) -> tensor<8x4xf32> {
    %concat = concatenate %in0, %in1, %in2 dimension = 1
        : (tensor<8x2xf32>, tensor<8x4xf32>, tensor<8x8xf32>) -> tensor<8x14xf32>
    %out = slice %concat starts = [0, 1] limits = [8, 5] strides = [1, 1] : tensor<8x14xf32> -> tensor<8x4xf32>
    results %out : tensor<8x4xf32>
}

// -----

// CHECK-LABEL: @slice_concat_discard_first_and_last_sources
// CHECK: slice {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<1, 1, "(8,2):(4,1)">, [1]>}
nv_tensor_ir.graph @slice_concat_discard_first_and_last_sources(
        %in0: tensor<8x2xf32>,
        %in1: tensor<8x4xf32>,
        %in2: tensor<8x8xf32>) -> tensor<8x2xf32> {
    %concat = concatenate %in0, %in1, %in2 dimension = 1
        : (tensor<8x2xf32>, tensor<8x4xf32>, tensor<8x8xf32>) -> tensor<8x14xf32>
    %out = slice %concat starts = [0, 3] limits = [8, 5] strides = [1, 1] : tensor<8x14xf32> -> tensor<8x2xf32>
    results %out : tensor<8x2xf32>
}

// -----

// CHECK-LABEL: @slice_concat_offset_between_sources
// CHECK: slice {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 0, "(8,1):(2,0)">, #nv_tensor_ir.tensor_source<1, 3, "(8,1):(4,0)">, #nv_tensor_ir.tensor_source<2, 4, "(8,1):(8,0)">>}
nv_tensor_ir.graph @slice_concat_offset_between_sources(
        %in0: tensor<8x2xf32>,
        %in1: tensor<8x4xf32>,
        %in2: tensor<8x8xf32>) -> tensor<8x3xf32> {
    %concat = concatenate %in0, %in1, %in2 dimension = 1
        : (tensor<8x2xf32>, tensor<8x4xf32>, tensor<8x8xf32>) -> tensor<8x14xf32>
    %out = slice %concat starts = [0, 0] limits = [8, 14] strides = [1, 5] : tensor<8x14xf32> -> tensor<8x3xf32>
    results %out : tensor<8x3xf32>
}

// -----

// CHECK-LABEL: @slice_concat_discard_last_with_offset
// CHECK: slice {{.*}} {layout = #nv_tensor_ir.concat_source<dim = 1, #nv_tensor_ir.tensor_source<0, 1, "(8,1):(2,0)">, #nv_tensor_ir.tensor_source<1, 1, "(8,2):(4,2)">, [0, 1]>}
nv_tensor_ir.graph @slice_concat_discard_last_with_offset(
        %in0: tensor<8x2xf32>,
        %in1: tensor<8x4xf32>,
        %in2: tensor<8x8xf32>) -> tensor<8x3xf32> {
    %concat = concatenate %in0, %in1, %in2 dimension = 1
        : (tensor<8x2xf32>, tensor<8x4xf32>, tensor<8x8xf32>) -> tensor<8x14xf32>
    %out = slice %concat starts = [0, 1] limits = [8, 7] strides = [1, 2] : tensor<8x14xf32> -> tensor<8x3xf32>
    results %out : tensor<8x3xf32>
}
