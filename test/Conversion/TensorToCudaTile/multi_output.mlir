// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// RMSNorm forward training uses one full result and one projected result. The
// reduction value remains broadcast for y, then the store path derives the
// compact [T_R,1] tile from that lowered SSA value.
// CHECK-LABEL: entry @rmsnorm_fwd_train
// CHECK: %[[REDUCE:.+]] = reduce {{.*}} : tile<32x16xf32> -> tile<32xf32>
// CHECK: %[[RESHAPE:.+]] = reshape %[[REDUCE]] : tile<32xf32> -> tile<32x1xf32>
// CHECK: %[[BCAST:.+]] = broadcast %[[RESHAPE]] : tile<32x1xf32> -> tile<32x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<32x16xf32>
// CHECK: %[[COMPACT:.+]] = extract %[[BCAST]]{{.*}} : tile<32x16xf32> -> tile<32x1xf32>
// CHECK: store_view_tko weak %[[COMPACT]]{{.*}} : tile<32x1xf32>
nv_tensor_ir.graph @rmsnorm_fwd_train(
    %x: tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"},
    %scale: tensor<16xf32> {nv_tensor_ir.stride = "(1)"}
    ) -> (tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"},
          tensor<64x1xf32> {nv_tensor_ir.stride = "(1,1)"})
    attributes {tile_size = array<i32: 32, 16>} {
  %sq = mul %x, %x : tensor<64x16xf32>
  %sum = reduce(%sq)<dimensions = [1], reduction_mode = <add>>
      : tensor<64x16xf32> -> tensor<64x1xf32>
  %wide = broadcast %sum : tensor<64x1xf32> -> tensor<64x16xf32>
  %scale_row = reshape %scale : tensor<16xf32> -> tensor<1x16xf32>
  %scale_wide = broadcast %scale_row
      : tensor<1x16xf32> -> tensor<64x16xf32>
  %scaled = mul %x, %scale_wide : tensor<64x16xf32>
  %y = mul %scaled, %wide : tensor<64x16xf32>
  results %y, %sum : tensor<64x16xf32>, tensor<64x1xf32>
}

// -----

// ABI order must not select the carrier. Reversing the two results retains the
// same computation tile and only reverses the two store destinations.
// CHECK-LABEL: entry @rmsnorm_fwd_train_reversed
// CHECK: %[[REDUCE:.+]] = reduce {{.*}} : tile<32x16xf32> -> tile<32xf32>
// CHECK: %[[RESHAPE:.+]] = reshape %[[REDUCE]] : tile<32xf32> -> tile<32x1xf32>
// CHECK: %[[BCAST:.+]] = broadcast %[[RESHAPE]] : tile<32x1xf32> -> tile<32x16xf32>
// CHECK: %[[COMPACT:.+]] = extract %[[BCAST]]{{.*}} : tile<32x16xf32> -> tile<32x1xf32>
// CHECK: store_view_tko weak %[[COMPACT]]{{.*}} : tile<32x1xf32>
// CHECK: store_view_tko weak {{.*}} : tile<32x16xf32>
nv_tensor_ir.graph @rmsnorm_fwd_train_reversed(
    %x: tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"},
    %scale: tensor<16xf32> {nv_tensor_ir.stride = "(1)"}
    ) -> (tensor<64x1xf32> {nv_tensor_ir.stride = "(1,1)"},
          tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"})
    attributes {tile_size = array<i32: 32, 16>} {
  %sq = mul %x, %x : tensor<64x16xf32>
  %sum = reduce(%sq)<dimensions = [1], reduction_mode = <add>>
      : tensor<64x16xf32> -> tensor<64x1xf32>
  %wide = broadcast %sum : tensor<64x1xf32> -> tensor<64x16xf32>
  %scale_row = reshape %scale : tensor<16xf32> -> tensor<1x16xf32>
  %scale_wide = broadcast %scale_row
      : tensor<1x16xf32> -> tensor<64x16xf32>
  %scaled = mul %x, %scale_wide : tensor<64x16xf32>
  %y = mul %scaled, %wide : tensor<64x16xf32>
  results %sum, %y : tensor<64x1xf32>, tensor<64x16xf32>
}

// -----

// A compact reduction result can be transposed at the graph ABI. Carrier
// lifting follows the SSA path backward, undoes the transpose, and restores
// the projected D dimension with zero stride. CUDA Tile still treats the
// explicit TransposeOp as a no-op.
// CHECK-LABEL: entry @rmsnorm_fwd_train_transposed_statistic(
// CHECK-SAME: %{{.+}}: tile<ptr<f32>>, %{{.+}}: tile<ptr<f32>>, %{{.+}}: tile<ptr<f32>>, %[[TR_PTR:.+]]: tile<ptr<f32>>)
// CHECK: %[[REDUCE:.+]] = reduce {{.*}} : tile<32x16xf32> -> tile<32xf32>
// CHECK: %[[RESHAPE:.+]] = reshape %[[REDUCE]] : tile<32xf32> -> tile<32x1xf32>
// CHECK: %[[BCAST:.+]] = broadcast %[[RESHAPE]] : tile<32x1xf32> -> tile<32x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<32x16xf32>
// CHECK: %[[COMPACT:.+]] = extract %[[BCAST]]{{.*}} : tile<32x16xf32> -> tile<32x1xf32>
// CHECK: %[[TR_VIEW:.+]] = make_tensor_view %[[TR_PTR]], shape = [64, 1], strides = [1, 1]
// CHECK: %[[TR_PART:.+]] = make_partition_view %[[TR_VIEW]] : partition_view<tile=(32x1), tensor_view<64x1xf32, strides=[1,1]>>
// CHECK: store_view_tko weak %[[COMPACT]], %[[TR_PART]]{{.*}} : tile<32x1xf32>
// CHECK-NOT: transpose
nv_tensor_ir.graph @rmsnorm_fwd_train_transposed_statistic(
    %x: tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"},
    %scale: tensor<16xf32> {nv_tensor_ir.stride = "(1)"}
    ) -> (tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"},
          tensor<1x64xf32> {nv_tensor_ir.stride = "(1,1)"})
    attributes {tile_size = array<i32: 32, 16>} {
  %sq = mul %x, %x : tensor<64x16xf32>
  %sum = reduce(%sq)<dimensions = [1], reduction_mode = <add>>
      : tensor<64x16xf32> -> tensor<64x1xf32>
  %wide = broadcast %sum : tensor<64x1xf32> -> tensor<64x16xf32>
  %scale_row = reshape %scale : tensor<16xf32> -> tensor<1x16xf32>
  %scale_wide = broadcast %scale_row
      : tensor<1x16xf32> -> tensor<64x16xf32>
  %scaled = mul %x, %scale_wide : tensor<64x16xf32>
  %y = mul %scaled, %wide : tensor<64x16xf32>
  %tr = transpose %sum permutation = [1, 0]
      : tensor<64x1xf32> -> tensor<1x64xf32>
  results %y, %tr : tensor<64x16xf32>, tensor<1x64xf32>
}

// -----

// Carrier selection and path transport are independent of ABI result order.
// CHECK-LABEL: entry @transposed_statistic_reversed(
// CHECK-SAME: %{{.+}}: tile<ptr<f32>>, %[[REV_TR_PTR:.+]]: tile<ptr<f32>>, %[[REV_Y_PTR:.+]]: tile<ptr<f32>>)
// CHECK: %[[REV_BCAST:.+]] = broadcast {{.*}} : tile<32x1xf32> -> tile<32x16xf32>
// CHECK: %[[REV_COMPACT:.+]] = extract %[[REV_BCAST]]{{.*}} : tile<32x16xf32> -> tile<32x1xf32>
// CHECK: %[[REV_TR_VIEW:.+]] = make_tensor_view %[[REV_TR_PTR]], shape = [64, 1], strides = [1, 1]
// CHECK: store_view_tko weak %[[REV_COMPACT]]{{.*}} : tile<32x1xf32>
// CHECK: %[[REV_Y_VIEW:.+]] = make_tensor_view %[[REV_Y_PTR]], shape = [64, 16], strides = [16, 1]
// CHECK: store_view_tko weak {{.*}} : tile<32x16xf32>
// CHECK-NOT: transpose
nv_tensor_ir.graph @transposed_statistic_reversed(
    %x: tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<1x64xf32> {nv_tensor_ir.stride = "(1,1)"},
          tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"})
    attributes {tile_size = array<i32: 32, 16>} {
  %sum = reduce(%x)<dimensions = [1], reduction_mode = <add>>
      : tensor<64x16xf32> -> tensor<64x1xf32>
  %wide = broadcast %sum : tensor<64x1xf32> -> tensor<64x16xf32>
  %y = mul %x, %wide : tensor<64x16xf32>
  %tr = transpose %sum permutation = [1, 0]
      : tensor<64x1xf32> -> tensor<1x64xf32>
  results %tr, %y : tensor<1x64xf32>, tensor<64x16xf32>
}

// -----

// Equal extents must not let shape-only broadcast attach the compact statistic
// to the wrong carrier axis. The explicit transpose still maps carrier row R
// to the contiguous physical output dimension.
// CHECK-LABEL: entry @transposed_statistic_equal_extents(
// CHECK-SAME: %{{.+}}: tile<ptr<f32>>, %{{.+}}: tile<ptr<f32>>, %[[EQ_TR_PTR:.+]]: tile<ptr<f32>>)
// CHECK: %[[EQ_BCAST:.+]] = broadcast {{.*}} : tile<8x1xf32> -> tile<8x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<8x16xf32>
// CHECK: %[[EQ_COMPACT:.+]] = extract %[[EQ_BCAST]]{{.*}} : tile<8x16xf32> -> tile<8x1xf32>
// CHECK: %[[EQ_VIEW:.+]] = make_tensor_view %[[EQ_TR_PTR]], shape = [16, 1], strides = [1, 1]
// CHECK: store_view_tko weak %[[EQ_COMPACT]]{{.*}} : tile<8x1xf32>
// CHECK-NOT: transpose
nv_tensor_ir.graph @transposed_statistic_equal_extents(
    %x: tensor<16x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<16x16xf32> {nv_tensor_ir.stride = "(16,1)"},
          tensor<1x16xf32> {nv_tensor_ir.stride = "(1,1)"})
    attributes {tile_size = array<i32: 8, 16>} {
  %sum = reduce(%x)<dimensions = [1], reduction_mode = <add>>
      : tensor<16x16xf32> -> tensor<16x1xf32>
  %wide = broadcast %sum : tensor<16x1xf32> -> tensor<16x16xf32>
  %y = mul %x, %wide : tensor<16x16xf32>
  %tr = transpose %sum permutation = [1, 0]
      : tensor<16x1xf32> -> tensor<1x16xf32>
  results %y, %tr : tensor<16x16xf32>, tensor<1x16xf32>
}

// -----

// A compact terminal reshape has the same equal-extent ambiguity as a
// transpose. Carrier lifting must follow the SSA path back to [R,1] before
// restoring the projected D dimension.
// CHECK-LABEL: entry @reshaped_statistic_equal_extents(
// CHECK-SAME: %{{.+}}: tile<ptr<f32>>, %{{.+}}: tile<ptr<f32>>, %[[EQ_RESHAPE_PTR:.+]]: tile<ptr<f32>>)
// CHECK: %[[EQ_RESHAPE_BCAST:.+]] = broadcast {{.*}} : tile<8x1xf32> -> tile<8x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<8x16xf32>
// CHECK: %[[EQ_RESHAPE_COMPACT:.+]] = extract %[[EQ_RESHAPE_BCAST]]{{.*}} : tile<8x16xf32> -> tile<8x1xf32>
// CHECK: %[[EQ_RESHAPE_VIEW:.+]] = make_tensor_view %[[EQ_RESHAPE_PTR]], shape = [16, 1], strides = [1, 1]
// CHECK: store_view_tko weak %[[EQ_RESHAPE_COMPACT]]{{.*}} : tile<8x1xf32>
nv_tensor_ir.graph @reshaped_statistic_equal_extents(
    %x: tensor<16x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<16x16xf32> {nv_tensor_ir.stride = "(16,1)"},
          tensor<1x16xf32> {nv_tensor_ir.stride = "(1,1)"})
    attributes {tile_size = array<i32: 8, 16>} {
  %sum = reduce(%x)<dimensions = [1], reduction_mode = <add>>
      : tensor<16x16xf32> -> tensor<16x1xf32>
  %wide = broadcast %sum : tensor<16x1xf32> -> tensor<16x16xf32>
  %y = mul %x, %wide : tensor<16x16xf32>
  %reshaped = reshape %sum : tensor<16x1xf32> -> tensor<1x16xf32>
  results %y, %reshaped : tensor<16x16xf32>, tensor<1x16xf32>
}

// -----

// A non-self-inverse permutation verifies that carrier lifting applies the
// actual inverse permutation, rather than relying on the 2-D swap case.
// CHECK-LABEL: entry @transposed_statistic_3d(
// CHECK-SAME: %{{.+}}: tile<ptr<f32>>, %{{.+}}: tile<ptr<f32>>, %[[TR3_PTR:.+]]: tile<ptr<f32>>)
// CHECK: %[[TR3_BCAST:.+]] = broadcast {{.*}} -> tile<4x8x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<4x8x16xf32>
// CHECK: %[[TR3_COMPACT:.+]] = extract %[[TR3_BCAST]]{{.*}} : tile<4x8x16xf32> -> tile<4x1x16xf32>
// CHECK: %[[TR3_VIEW:.+]] = make_tensor_view %[[TR3_PTR]], shape = [4, 1, 16], strides = [1, 64, 4]
// CHECK: store_view_tko weak %[[TR3_COMPACT]]{{.*}} : tile<4x1x16xf32>
// CHECK-NOT: transpose
nv_tensor_ir.graph @transposed_statistic_3d(
    %x: tensor<4x8x16xf32> {nv_tensor_ir.stride = "(128,16,1)"}
    ) -> (tensor<4x8x16xf32> {nv_tensor_ir.stride = "(128,16,1)"},
          tensor<16x4x1xf32> {nv_tensor_ir.stride = "(4,1,1)"})
    attributes {tile_size = array<i32: 4, 8, 16>} {
  %y = abs %x : tensor<4x8x16xf32>
  %sum = reduce(%x)<dimensions = [1], reduction_mode = <add>>
      : tensor<4x8x16xf32> -> tensor<4x1x16xf32>
  %tr = transpose %sum permutation = [2, 0, 1]
      : tensor<4x1x16xf32> -> tensor<16x4x1xf32>
  results %y, %tr : tensor<4x8x16xf32>, tensor<16x4x1xf32>
}

// -----

// The root tile covers all of D for unique inv_rms ownership, while the
// reduction child space loops over smaller T_D=128 tiles inside that CTA.
// CHECK-LABEL: entry @rmsnorm_fwd_looped_reduction
// CHECK: %[[LOOP:.+]] = for {{.*}} -> (tile<1x128xf32>)
// CHECK: %[[REDUCE:.+]] = reduce %[[LOOP]] dim=1 {{.*}} : tile<1x128xf32> -> tile<1xf32>
// CHECK: %[[BCAST:.+]] = broadcast {{.*}} -> tile<1x512xf32>
// CHECK: store_view_tko weak {{.*}} : tile<1x512xf32>
// CHECK: %[[COMPACT:.+]] = extract %[[BCAST]]{{.*}} : tile<1x512xf32> -> tile<1x1xf32>
// CHECK: store_view_tko weak %[[COMPACT]]{{.*}} : tile<1x1xf32>
nv_tensor_ir.graph @rmsnorm_fwd_looped_reduction(
    %x: tensor<8x512xf32> {nv_tensor_ir.stride = "(512,1)"}
    ) -> (tensor<8x512xf32> {nv_tensor_ir.stride = "(512,1)"},
          tensor<8x1xf32> {nv_tensor_ir.stride = "(1,1)"})
    attributes {tile_size = array<i32: 1, 512>} {
  %sq = mul %x, %x : tensor<8x512xf32>
  %sum = reduce(%sq)<dimensions = [1], reduction_mode = <add>>
      : tensor<8x512xf32> -> tensor<8x1xf32>
  %wide = broadcast %sum : tensor<8x1xf32> -> tensor<8x512xf32>
  %y = mul %x, %wide : tensor<8x512xf32>
  results %y, %sum : tensor<8x512xf32>, tensor<8x1xf32>
}

// -----

// RMSNorm backward kernel 1 uses explicit P and G graph dimensions. It writes
// dx and the partial dScale workspace from one carrier.
// CHECK-LABEL: entry @rmsnorm_bwd_kernel1
// CHECK: %[[BCAST:.+]] = broadcast {{.*}} -> tile<1x8x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<1x8x16xf32>
// CHECK: %[[PARTIAL:.+]] = extract %[[BCAST]]{{.*}} : tile<1x8x16xf32> -> tile<1x1x16xf32>
// CHECK: store_view_tko weak %[[PARTIAL]]{{.*}} : tile<1x1x16xf32>
nv_tensor_ir.graph @rmsnorm_bwd_kernel1(
    %x: tensor<4x8x16xf32> {nv_tensor_ir.stride = "(128,16,1)"}
    ) -> (tensor<4x8x16xf32> {nv_tensor_ir.stride = "(128,16,1)"},
          tensor<4x1x16xf32> {nv_tensor_ir.stride = "(16,16,1)"})
    attributes {tile_size = array<i32: 1, 8, 16>} {
  %dx = neg %x : tensor<4x8x16xf32>
  %partial = reduce(%x)<dimensions = [1], reduction_mode = <add>>
      : tensor<4x8x16xf32> -> tensor<4x1x16xf32>
  results %dx, %partial : tensor<4x8x16xf32>, tensor<4x1x16xf32>
}

// -----

// RMSNorm backward kernel 2 is a separate TensorIR graph that consumes the
// explicit workspace produced by kernel 1.
// CHECK-LABEL: entry @rmsnorm_bwd_kernel2
// CHECK: %[[REDUCE:.+]] = reduce {{.*}} : tile<16x4xf32> -> tile<16xf32>
// CHECK: store_view_tko weak %[[REDUCE]]{{.*}} : tile<16xf32>
nv_tensor_ir.graph @rmsnorm_bwd_kernel2(
    %partial: tensor<4x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<1x16xf32> {nv_tensor_ir.stride = "(16,1)"})
    attributes {tile_size = array<i32: 16>} {
  %dscale = reduce(%partial)<dimensions = [0], reduction_mode = <add>>
      : tensor<4x16xf32> -> tensor<1x16xf32>
  results %dscale : tensor<1x16xf32>
}

// -----

// LayerNorm forward training returns y, mean, and invVar. The paired ReduceUD
// is materialized once and both of its lowered SSA results are stored.
// CHECK-LABEL: entry @layernorm_fwd_train
// CHECK: %[[REDUCE:.+]]:2 = reduce {{.*}} -> tile<32xf32>, tile<32xf32>
// CHECK: store_view_tko weak {{.*}} : tile<32x16xf32>
// CHECK: extract {{.*}} : tile<32x16xf32> -> tile<32x1xf32>
// CHECK: store_view_tko weak {{.*}} : tile<32x1xf32>
// CHECK: extract {{.*}} : tile<32x16xf32> -> tile<32x1xf32>
// CHECK: store_view_tko weak {{.*}} : tile<32x1xf32>
nv_tensor_ir.graph @layernorm_fwd_train(
    %x: tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"},
    %xx: tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"},
          tensor<64x1xf32> {nv_tensor_ir.stride = "(1,1)"},
          tensor<64x1xf32> {nv_tensor_ir.stride = "(1,1)"})
    attributes {tile_size = array<i32: 32, 16>} {
  %sum, %sumsq = reduce_ud(%x, %xx)
      <dimensions = [1], identity = [0.0 : f32, 0.0 : f32]>
      (%acc0: f32, %acc1: f32, %val0: f32, %val1: f32) {
    %next0 = arith.addf %acc0, %val0 : f32
    %next1 = arith.addf %acc1, %val1 : f32
    nv_tensor_ir.yield %next0, %next1 : f32, f32
  } : tensor<64x16xf32>, tensor<64x16xf32>
      -> tensor<64x1xf32>, tensor<64x1xf32>
  %mean = broadcast %sum : tensor<64x1xf32> -> tensor<64x16xf32>
  %inv = broadcast %sumsq : tensor<64x1xf32> -> tensor<64x16xf32>
  %centered = sub %x, %mean : tensor<64x16xf32>
  %y = mul %centered, %inv : tensor<64x16xf32>
  results %y, %sum, %sumsq
      : tensor<64x16xf32>, tensor<64x1xf32>, tensor<64x1xf32>
}

// -----

// LayerNorm backward kernel 1 returns dx and two projected workspaces.
// CHECK-LABEL: entry @layernorm_bwd_kernel1
// CHECK: %[[REDUCE:.+]]:2 = reduce {{.*}} -> tile<1x16xf32>, tile<1x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<1x8x16xf32>
// CHECK: extract {{.*}} : tile<1x8x16xf32> -> tile<1x1x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<1x1x16xf32>
// CHECK: extract {{.*}} : tile<1x8x16xf32> -> tile<1x1x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<1x1x16xf32>
nv_tensor_ir.graph @layernorm_bwd_kernel1(
    %x: tensor<4x8x16xf32> {nv_tensor_ir.stride = "(128,16,1)"},
    %dy: tensor<4x8x16xf32> {nv_tensor_ir.stride = "(128,16,1)"}
    ) -> (tensor<4x8x16xf32> {nv_tensor_ir.stride = "(128,16,1)"},
          tensor<4x1x16xf32> {nv_tensor_ir.stride = "(16,16,1)"},
          tensor<4x1x16xf32> {nv_tensor_ir.stride = "(16,16,1)"})
    attributes {tile_size = array<i32: 1, 8, 16>} {
  %dx = sub %x, %dy : tensor<4x8x16xf32>
  %partial_scale, %partial_bias = reduce_ud(%x, %dy)
      <dimensions = [1], identity = [0.0 : f32, 0.0 : f32]>
      (%acc0: f32, %acc1: f32, %val0: f32, %val1: f32) {
    %next0 = arith.addf %acc0, %val0 : f32
    %next1 = arith.addf %acc1, %val1 : f32
    nv_tensor_ir.yield %next0, %next1 : f32, f32
  } : tensor<4x8x16xf32>, tensor<4x8x16xf32>
      -> tensor<4x1x16xf32>, tensor<4x1x16xf32>
  results %dx, %partial_scale, %partial_bias
      : tensor<4x8x16xf32>, tensor<4x1x16xf32>, tensor<4x1x16xf32>
}

// -----

// LayerNorm backward kernel 2 consumes both workspaces in a paired ReduceUD.
// CHECK-LABEL: entry @layernorm_bwd_kernel2
// CHECK: %[[REDUCE:.+]]:2 = reduce {{.*}} -> tile<16xf32>, tile<16xf32>
// CHECK: extract {{.*}} : tile<4x16xf32> -> tile<1x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<1x16xf32>
// CHECK: extract {{.*}} : tile<4x16xf32> -> tile<1x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<1x16xf32>
nv_tensor_ir.graph @layernorm_bwd_kernel2(
    %partial_scale: tensor<4x16xf32> {nv_tensor_ir.stride = "(16,1)"},
    %partial_bias: tensor<4x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<1x16xf32> {nv_tensor_ir.stride = "(16,1)"},
          tensor<1x16xf32> {nv_tensor_ir.stride = "(16,1)"})
    attributes {tile_size = array<i32: 4, 16>} {
  %dscale, %dbias = reduce_ud(%partial_scale, %partial_bias)
      <dimensions = [0], identity = [0.0 : f32, 0.0 : f32]>
      (%acc0: f32, %acc1: f32, %val0: f32, %val1: f32) {
    %next0 = arith.addf %acc0, %val0 : f32
    %next1 = arith.addf %acc1, %val1 : f32
    nv_tensor_ir.yield %next0, %next1 : f32, f32
  } : tensor<4x16xf32>, tensor<4x16xf32>
      -> tensor<1x16xf32>, tensor<1x16xf32>
  results %dscale, %dbias : tensor<1x16xf32>, tensor<1x16xf32>
}

// -----

// Path-first transport keeps the input in the common SSA anchor domain. The
// transpose is represented only by output 1's physical view, even when all
// extents are equal.
// CHECK-LABEL: entry @square_original_and_transpose(
// CHECK-SAME: %[[SQ_IN:.+]]: tile<ptr<f32>>, %[[SQ_Y:.+]]: tile<ptr<f32>>, %[[SQ_T:.+]]: tile<ptr<f32>>)
// CHECK: %[[SQ_IN_VIEW:.+]] = make_tensor_view %[[SQ_IN]], shape = [8, 8], strides = [8, 1]
// CHECK: %[[SQ_IN_PART:.+]] = make_partition_view %[[SQ_IN_VIEW]]
// CHECK: %[[SQ_TILE:.+]], %{{.+}} = load_view_tko weak %[[SQ_IN_PART]]
// CHECK: %[[SQ_ABS:.+]] = absf %[[SQ_TILE]]
// CHECK: %[[SQ_Y_VIEW:.+]] = make_tensor_view %[[SQ_Y]], shape = [8, 8], strides = [8, 1]
// CHECK: %[[SQ_Y_PART:.+]] = make_partition_view %[[SQ_Y_VIEW]]
// CHECK: store_view_tko weak %[[SQ_ABS]], %[[SQ_Y_PART]]
// CHECK: %[[SQ_T_VIEW:.+]] = make_tensor_view %[[SQ_T]], shape = [8, 8], strides = [1, 8]
// CHECK: %[[SQ_T_PART:.+]] = make_partition_view %[[SQ_T_VIEW]]
// CHECK: store_view_tko weak %[[SQ_TILE]], %[[SQ_T_PART]]
// CHECK-NOT: transpose
nv_tensor_ir.graph @square_original_and_transpose(
    %x: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
    ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
          tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"})
    attributes {tile_size = array<i32: 8, 8>} {
  %y = abs %x : tensor<8x8xf32>
  %t = transpose %x permutation = [1, 0]
      : tensor<8x8xf32> -> tensor<8x8xf32>
  results %y, %t : tensor<8x8xf32>, tensor<8x8xf32>
}

// -----

// A non-square transpose also remains in the [8,16] anchor domain. Both
// results reuse one input tile; output 1 uses the inverse-transposed store view.
// CHECK-LABEL: entry @original_and_transpose(
// CHECK-SAME: %[[NS_IN:.+]]: tile<ptr<f32>>, %[[NS_Y:.+]]: tile<ptr<f32>>, %[[NS_T:.+]]: tile<ptr<f32>>)
// CHECK: %[[NS_IN_VIEW:.+]] = make_tensor_view %[[NS_IN]], shape = [8, 16], strides = [16, 1]
// CHECK: %[[NS_IN_PART:.+]] = make_partition_view %[[NS_IN_VIEW]]
// CHECK: %[[NS_TILE:.+]], %{{.+}} = load_view_tko weak %[[NS_IN_PART]]
// CHECK: %[[NS_ABS:.+]] = absf %[[NS_TILE]]
// CHECK: %[[NS_Y_VIEW:.+]] = make_tensor_view %[[NS_Y]], shape = [8, 16], strides = [16, 1]
// CHECK: %[[NS_Y_PART:.+]] = make_partition_view %[[NS_Y_VIEW]]
// CHECK: store_view_tko weak %[[NS_ABS]], %[[NS_Y_PART]]
// CHECK: %[[NS_T_VIEW:.+]] = make_tensor_view %[[NS_T]], shape = [8, 16], strides = [1, 8]
// CHECK: %[[NS_T_PART:.+]] = make_partition_view %[[NS_T_VIEW]]
// CHECK: store_view_tko weak %[[NS_TILE]], %[[NS_T_PART]]
// CHECK-NOT: transpose
nv_tensor_ir.graph @original_and_transpose(
    %x: tensor<8x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(16,1)"},
          tensor<16x8xf32> {nv_tensor_ir.stride = "(8,1)"})
    attributes {tile_size = array<i32: 8, 16>} {
  %y = abs %x : tensor<8x16xf32>
  %t = transpose %x permutation = [1, 0]
      : tensor<8x16xf32> -> tensor<16x8xf32>
  results %y, %t : tensor<8x16xf32>, tensor<16x8xf32>
}

// -----

// Non-default physical output strides do not change the anchor domain. Tile
// analysis constrains its dimensions using the two carrier-relative views.
// CHECK-LABEL: module attributes {tensor_ir.resolved_iteration_space_shape = array<i64: 8, 16>, tensor_ir.resolved_tile_size = array<i32: 1, 16>}
// CHECK: entry @original_and_transpose_non_default_output_strides(
// CHECK-SAME: %{{.+}}: tile<ptr<f32>>, %[[STRIDED_Y:.+]]: tile<ptr<f32>>, %[[STRIDED_T:.+]]: tile<ptr<f32>>)
// CHECK: %[[STRIDED_Y_VIEW:.+]] = make_tensor_view %[[STRIDED_Y]], shape = [8, 16], strides = [1, 8]
// CHECK: %[[STRIDED_Y_PART:.+]] = make_partition_view %[[STRIDED_Y_VIEW]]
// CHECK: store_view_tko weak {{.*}}, %[[STRIDED_Y_PART]]{{.*}} : tile<1x16xf32>
// CHECK: %[[STRIDED_T_VIEW:.+]] = make_tensor_view %[[STRIDED_T]], shape = [8, 16], strides = [16, 1]
// CHECK: %[[STRIDED_T_PART:.+]] = make_partition_view %[[STRIDED_T_VIEW]]
// CHECK: store_view_tko weak {{.*}}, %[[STRIDED_T_PART]]{{.*}} : tile<1x16xf32>
// CHECK-NOT: transpose
nv_tensor_ir.graph @original_and_transpose_non_default_output_strides(
    %x: tensor<8x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(1,8)"},
          tensor<16x8xf32> {nv_tensor_ir.stride = "(1,16)"}) {
  %y = abs %x : tensor<8x16xf32>
  %t = transpose %x permutation = [1, 0]
      : tensor<8x16xf32> -> tensor<16x8xf32>
  results %y, %t : tensor<8x16xf32>, tensor<16x8xf32>
}

// -----

// Independent projected branches can reduce different carrier dimensions.
// CHECK-LABEL: entry @crossed_reductions
// CHECK: extract {{.*}} : tile<1x8x16xf32> -> tile<1x1x16xf32>
// CHECK: store_view_tko weak {{.*}} : tile<1x1x16xf32>
// CHECK: extract {{.*}} : tile<1x8x16xf32> -> tile<1x8x1xf32>
// CHECK: store_view_tko weak {{.*}} : tile<1x8x1xf32>
nv_tensor_ir.graph @crossed_reductions(
    %x: tensor<64x8x16xf32> {nv_tensor_ir.stride = "(128,16,1)"}
    ) -> (tensor<64x1x16xf32> {nv_tensor_ir.stride = "(16,16,1)"},
          tensor<64x8x1xf32> {nv_tensor_ir.stride = "(8,1,1)"})
    attributes {tile_size = array<i32: 1, 8, 16>} {
  %c = reduce(%x)<dimensions = [1], reduction_mode = <add>>
      : tensor<64x8x16xf32> -> tensor<64x1x16xf32>
  %d = reduce(%x)<dimensions = [2], reduction_mode = <add>>
      : tensor<64x8x16xf32> -> tensor<64x8x1xf32>
  results %c, %d : tensor<64x1x16xf32>, tensor<64x8x1xf32>
}
