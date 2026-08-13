// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s

// Test codegen for graphs with multiple iteration spaces.

// -----

// CHECK-LABEL: entry @reduce_broadcast_transition
// CHECK-SAME: %[[INPUT_PTR:.*]]: tile<ptr<f32>>, %[[BIAS_PTR:.*]]: tile<ptr<f32>>, %[[OUTPUT_PTR:.*]]: tile<ptr<f32>>

/// Create tensor views for input.
// CHECK-DAG: %[[INPUT_VIEW:.*]] = make_tensor_view %[[INPUT_PTR]], shape = [16, 32, 64]
// CHECK-DAG: %[[BIAS_VIEW:.*]] = make_tensor_view %[[BIAS_PTR]], shape = [16, 48, 64]
// CHECK-DAG: %[[OUTPUT_VIEW:.*]] = make_tensor_view %[[OUTPUT_PTR]], shape = [16, 48, 64]

/// Get block IDs.
// CHECK: %[[BLOCK_X:.*]], %{{.*}}, %{{.*}} = get_tile_block_id

/// Two iteration spaces: input tensor has different shape than bias/output.
/// Each iteration space gets its own get_index_space_shape.
// CHECK: %[[INPUT_PVIEW:.*]] = make_partition_view %[[INPUT_VIEW]]
// CHECK: %[[INPUT_SHAPE:.*]]:3 = get_index_space_shape %[[INPUT_PVIEW]]
// CHECK: %[[BIAS_PVIEW:.*]] = make_partition_view %[[BIAS_VIEW]]
// CHECK: %[[BIAS_SHAPE:.*]]:3 = get_index_space_shape %[[BIAS_PVIEW]]

/// Load input using input iteration space coordinates.
// CHECK: make_partition_view %[[INPUT_VIEW]]
// CHECK: remi %[[BLOCK_X]], %[[INPUT_SHAPE]]#2
// CHECK: divi %{{.*}}, %[[INPUT_SHAPE]]#2
// CHECK: remi %{{.*}}, %[[INPUT_SHAPE]]#1
// CHECK: divi %{{.*}}, %[[INPUT_SHAPE]]#1
// CHECK: remi %{{.*}}, %[[INPUT_SHAPE]]#0
// CHECK: %[[INPUT_TILE:.*]], %{{.*}} = load_view_tko

/// Load bias using bias/output iteration space coordinates.
// CHECK: make_partition_view %[[BIAS_VIEW]]
// CHECK: remi %[[BLOCK_X]], %[[BIAS_SHAPE]]#2
// CHECK: divi %{{.*}}, %[[BIAS_SHAPE]]#2
// CHECK: remi %{{.*}}, %[[BIAS_SHAPE]]#1
// CHECK: divi %{{.*}}, %[[BIAS_SHAPE]]#1
// CHECK: remi %{{.*}}, %[[BIAS_SHAPE]]#0
// CHECK: %[[BIAS_TILE:.*]], %{{.*}} = load_view_tko

/// Compute: reduce -> reshape -> broadcast -> add.
// CHECK: %[[REDUCED:.*]] = reduce %[[INPUT_TILE]]
// CHECK: %[[RESHAPED:.*]] = reshape %[[REDUCED]]
// CHECK: %[[BROADCASTED:.*]] = broadcast %[[RESHAPED]]
// CHECK: %[[RESULT:.*]] = addf %[[BROADCASTED]], %[[BIAS_TILE]]

/// Store output using bias/output iteration space coordinates.
// CHECK: make_partition_view %[[OUTPUT_VIEW]]
// CHECK: remi %[[BLOCK_X]], %[[BIAS_SHAPE]]#2
// CHECK: divi %{{.*}}, %[[BIAS_SHAPE]]#2
// CHECK: remi %{{.*}}, %[[BIAS_SHAPE]]#1
// CHECK: divi %{{.*}}, %[[BIAS_SHAPE]]#1
// CHECK: remi %{{.*}}, %[[BIAS_SHAPE]]#0
// CHECK: store_view_tko {{.*}} %[[RESULT]]
// CHECK: return
module {
  nv_tensor_ir.graph @reduce_broadcast_transition(
      %input: tensor<16x32x64xf32>,
      %bias: tensor<16x48x64xf32>
  ) -> tensor<16x48x64xf32> {
    %reduced = reduce(%input) <dimensions = [1], reduction_mode = <add>>
        : tensor<16x32x64xf32> -> tensor<16x1x64xf32>

    %broadcast = broadcast %reduced : tensor<16x1x64xf32> -> tensor<16x48x64xf32>

    %result = add %broadcast, %bias : tensor<16x48x64xf32>
    results %result : tensor<16x48x64xf32>
  }
}

// -----

// Broadcast -> mul -> reduce -> broadcast -> add. Two iteration spaces with
// compute in both: iter_space 0 has broadcast+mul+reduce, iter_space 1 has
// broadcast+add.

// CHECK-LABEL: entry @broadcast_reduce_broadcast
// CHECK-SAME: %[[A_PTR:.*]]: tile<ptr<f32>>, %[[C_PTR:.*]]: tile<ptr<f32>>, %[[B_PTR:.*]]: tile<ptr<f32>>, %[[OUT_PTR:.*]]: tile<ptr<f32>>

/// Iter space 0 (16x1x64) and iter space 1 (16x48x64) get separate shapes.
// CHECK-DAG: %[[A_VIEW:.*]] = make_tensor_view %[[A_PTR]], shape = [16, 1, 64]
// CHECK-DAG: %[[C_VIEW:.*]] = make_tensor_view %[[C_PTR]], shape = [16, 32, 64]
// CHECK-DAG: %[[B_VIEW:.*]] = make_tensor_view %[[B_PTR]], shape = [16, 48, 64]
// CHECK: %[[BLOCK_X:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[SHAPE_0:.*]]:3 = get_index_space_shape
// CHECK: %[[SHAPE_1:.*]]:3 = get_index_space_shape

/// Loads use their respective iteration space shapes.
// CHECK: %[[A_TILE:.*]], %{{.*}} = load_view_tko
// CHECK: %[[C_TILE:.*]], %{{.*}} = load_view_tko
// CHECK: %[[B_TILE:.*]], %{{.*}} = load_view_tko

/// Compute chain: broadcast -> mul -> reduce -> reshape -> broadcast -> add.
// CHECK: %[[BCAST1:.*]] = broadcast %[[A_TILE]]
// CHECK: %[[MUL:.*]] = mulf %[[BCAST1]], %[[C_TILE]]
// CHECK: %[[REDUCED:.*]] = reduce %[[MUL]]
// CHECK: %[[RESHAPED:.*]] = reshape %[[REDUCED]]
// CHECK: %[[BCAST2:.*]] = broadcast %[[RESHAPED]]
// CHECK: %[[RESULT:.*]] = addf %[[BCAST2]], %[[B_TILE]]

/// Store uses iter space 1 coordinates.
// CHECK: store_view_tko {{.*}} %[[RESULT]]
// CHECK: return
module {
  nv_tensor_ir.graph @broadcast_reduce_broadcast(
      %inputA: tensor<16x1x64xf32>,
      %inputC: tensor<16x32x64xf32>,
      %inputB: tensor<16x48x64xf32>
  ) -> (tensor<16x48x64xf32>) {
    %broadcast1 = broadcast %inputA : tensor<16x1x64xf32> -> tensor<16x32x64xf32>
    %mul_result = mul %broadcast1, %inputC : tensor<16x32x64xf32>
    %reduced = reduce(%mul_result) <dimensions = [1], reduction_mode = <add>>
        : tensor<16x32x64xf32> -> tensor<16x1x64xf32>
    %broadcast2 = broadcast %reduced : tensor<16x1x64xf32> -> tensor<16x48x64xf32>
    %result = add %broadcast2, %inputB : tensor<16x48x64xf32>
    results %result : tensor<16x48x64xf32>
  }
}

// -----

// Two successive reduce -> broadcast transitions creating 3 iteration spaces:
//   iter_space 0: input (16x32x64) -> reduce
//   iter_space 1: broadcast (16x48x64) -> add -> reduce
//   iter_space 2: broadcast (16x64x64) -> add -> output

// CHECK-LABEL: entry @two_transitions
// CHECK-SAME: %[[INPUT_PTR:.*]]: tile<ptr<f32>>, %[[BIAS48_PTR:.*]]: tile<ptr<f32>>, %[[BIAS64_PTR:.*]]: tile<ptr<f32>>, %[[OUT_PTR:.*]]: tile<ptr<f32>>

/// Three iteration spaces get three separate get_index_space_shape ops.
// CHECK: %[[BLOCK_X:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[SHAPE_0:.*]]:3 = get_index_space_shape
// CHECK: %[[SHAPE_1:.*]]:3 = get_index_space_shape
// CHECK: %[[SHAPE_2:.*]]:3 = get_index_space_shape

/// Load from all three iteration spaces.
// CHECK: %[[INPUT_TILE:.*]], %{{.*}} = load_view_tko
// CHECK: %[[BIAS48_TILE:.*]], %{{.*}} = load_view_tko
// CHECK: %[[BIAS64_TILE:.*]], %{{.*}} = load_view_tko

/// First transition: reduce -> reshape -> broadcast -> add.
// CHECK: %[[RED1:.*]] = reduce %[[INPUT_TILE]]
// CHECK: %[[RESHAPE1:.*]] = reshape %[[RED1]]
// CHECK: %[[BCAST1:.*]] = broadcast %[[RESHAPE1]]
// CHECK: %[[ADD1:.*]] = addf %[[BCAST1]], %[[BIAS48_TILE]]

/// Second transition: reduce -> reshape -> broadcast -> add.
// CHECK: %[[RED2:.*]] = reduce %[[ADD1]]
// CHECK: %[[RESHAPE2:.*]] = reshape %[[RED2]]
// CHECK: %[[BCAST2:.*]] = broadcast %[[RESHAPE2]]
// CHECK: %[[ADD2:.*]] = addf %[[BCAST2]], %[[BIAS64_TILE]]

/// Store uses iter space 2 coordinates.
// CHECK: store_view_tko {{.*}} %[[ADD2]]
// CHECK: return
module {
  nv_tensor_ir.graph @two_transitions(
      %input: tensor<16x32x64xf32>,
      %bias48: tensor<16x48x64xf32>,
      %bias64: tensor<16x64x64xf32>
  ) -> tensor<16x64x64xf32> {
    %reduced1 = reduce(%input) <dimensions = [1], reduction_mode = <add>>
        : tensor<16x32x64xf32> -> tensor<16x1x64xf32>
    %bcast1 = broadcast %reduced1 : tensor<16x1x64xf32> -> tensor<16x48x64xf32>
    %add1 = add %bcast1, %bias48 : tensor<16x48x64xf32>
    %reduced2 = reduce(%add1) <dimensions = [1], reduction_mode = <add>>
        : tensor<16x48x64xf32> -> tensor<16x1x64xf32>
    %bcast2 = broadcast %reduced2 : tensor<16x1x64xf32> -> tensor<16x64x64xf32>
    %result = add %bcast2, %bias64 : tensor<16x64x64xf32>
    results %result : tensor<16x64x64xf32>
  }
}

// -----

// Reduce -> broadcast back to the same size (32 -> 1 -> 32). This still creates
// a transition because the dimension domain was defined before being contracted.

// CHECK-LABEL: entry @same_size_transition
// CHECK-SAME: %[[INPUT_PTR:.*]]: tile<ptr<f32>>, %[[BIAS_PTR:.*]]: tile<ptr<f32>>, %[[OUT_PTR:.*]]: tile<ptr<f32>>

/// Two iteration spaces with same tensor shapes but different index spaces.
// CHECK: %[[BLOCK_X:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[SHAPE_0:.*]]:3 = get_index_space_shape
// CHECK: %[[SHAPE_1:.*]]:3 = get_index_space_shape

// CHECK: %[[INPUT_TILE:.*]], %{{.*}} = load_view_tko
// CHECK: %[[BIAS_TILE:.*]], %{{.*}} = load_view_tko

/// Compute: reduce -> reshape -> broadcast -> add.
// CHECK: %[[REDUCED:.*]] = reduce %[[INPUT_TILE]]
// CHECK: %[[RESHAPED:.*]] = reshape %[[REDUCED]]
// CHECK: %[[BROADCASTED:.*]] = broadcast %[[RESHAPED]]
// CHECK: %[[RESULT:.*]] = addf %[[BROADCASTED]], %[[BIAS_TILE]]

// CHECK: store_view_tko {{.*}} %[[RESULT]]
// CHECK: return
module {
  nv_tensor_ir.graph @same_size_transition(
      %input: tensor<16x32x64xf32>,
      %bias: tensor<16x32x64xf32>
  ) -> tensor<16x32x64xf32> {
    %reduced = reduce(%input) <dimensions = [1], reduction_mode = <add>>
        : tensor<16x32x64xf32> -> tensor<16x1x64xf32>
    %broadcast = broadcast %reduced : tensor<16x1x64xf32> -> tensor<16x32x64xf32>
    %result = add %broadcast, %bias : tensor<16x32x64xf32>
    results %result : tensor<16x32x64xf32>
  }
}
