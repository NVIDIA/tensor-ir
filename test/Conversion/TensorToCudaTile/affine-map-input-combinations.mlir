// RUN: split-file %s %t
// RUN: tensor_ir-opt %t/discovered.mlir -split-input-file \
// RUN:   -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map tile-size=2,2" \
// RUN:   | tensor_ir-opt -split-input-file | FileCheck %s --check-prefix=DISC
// RUN: tensor_ir-opt %t/explicit.mlir -split-input-file \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   | tensor_ir-opt -split-input-file | FileCheck %s --check-prefix=EXPLICIT

//--- discovered.mlir

// A uniform scalar is a value, not a tensor descriptor. Its splat must use
// the scalar argument directly and feed the tensor computation.
// DISC-LABEL: entry @scalar_splat
// DISC-SAME: (%[[S:[^:]+]]: tile<f32>, %[[IN:[^:]+]]: tile<ptr<f32>>,
// DISC-SAME: %[[OUT:[^:]+]]: tile<ptr<f32>>)
// DISC: %[[VIEW:.+]] = make_tensor_view %[[IN]],
// DISC: %[[OUT_VIEW:.+]] = make_tensor_view %[[OUT]],
// DISC: %[[PART:.+]] = make_partition_view %[[VIEW]]
// DISC: %[[COUNTS:.+]]:2 = get_index_space_shape %[[PART]]
// DISC: %[[LOAD_PART:.+]] = make_partition_view %[[VIEW]]
// DISC: %[[LOADED:.+]], %{{.+}} = load_view_tko weak %[[LOAD_PART]]
// DISC: %[[RESHAPED:.+]] = reshape %[[S]] : tile<f32> -> tile<1x1xf32>
// DISC: %[[SPLAT:.+]] = broadcast %[[RESHAPED]]
// DISC-SAME: : tile<1x1xf32> -> tile<2x2xf32>
// DISC: %[[SUM:.+]] = addf %[[LOADED]], %[[SPLAT]]
// DISC: %[[OUT_PART:.+]] = make_partition_view %[[OUT_VIEW]]
// DISC: store_view_tko weak %[[SUM]], %[[OUT_PART]]
nv_tensor_ir.graph @scalar_splat(%s: f32, %a: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %b = splat %s : tensor<8x8xf32>
  %r = add %a, %b : tensor<8x8xf32>
  results %r : tensor<8x8xf32>
}

//--- explicit.mlir

// The dynamic input defines the iteration-space extents. An unused static
// peer must not replace that definition or introduce static-size clamping.
// EXPLICIT-LABEL: entry @static_peer
// EXPLICIT-SAME: (%[[IN:[^:]+]]: tile<ptr<f32>>,
// EXPLICIT-SAME: %[[M:[^:]+]]: tile<i32>, %[[N:[^:]+]]: tile<i32>,
// EXPLICIT-SAME: %[[PEER:[^:]+]]: tile<ptr<f32>>, %[[OUT:[^:]+]]: tile<ptr<f32>>,
// EXPLICIT: %[[MB:.+]] = assume bounded<0, ?>, %[[M]] : tile<i32>
// EXPLICIT: %[[NB:.+]] = assume bounded<0, ?>, %[[N]] : tile<i32>
// EXPLICIT: %[[STRIDE:.+]] = assume bounded<0, ?>, %[[NB]] : tile<i32>
// EXPLICIT: %[[VIEW:.+]] = make_tensor_view %[[IN]],
// EXPLICIT-SAME: shape = [%[[MB]], %[[NB]]], strides = [%[[STRIDE]], 1]
// EXPLICIT: %[[PEER_VIEW:.+]] = make_tensor_view %[[PEER]], shape = [8, 8]
// EXPLICIT: %[[OUT_VIEW:.+]] = make_tensor_view %[[OUT]],
// EXPLICIT: %[[BLOCK:.+]], %{{.+}}, %{{.+}} = get_tile_block_id
// EXPLICIT: %[[GRID_PART:.+]] = make_partition_view %[[VIEW]]
// EXPLICIT: %[[GRID:.+]]:2 = get_index_space_shape %[[GRID_PART]]
// EXPLICIT: %[[IN_PART:.+]] = make_partition_view %[[VIEW]]
// EXPLICIT: %[[LOADED:.+]], %{{.+}} = load_view_tko weak %[[IN_PART]]
// EXPLICIT: %[[PEER_PART:.+]] = make_partition_view %[[PEER_VIEW]]
// EXPLICIT-NOT: get_index_space_shape
// EXPLICIT: %[[COL:.+]] = remi %[[BLOCK]], %[[GRID]]#1 unsigned
// EXPLICIT: %[[ROW_ID:.+]] = divi %[[BLOCK]], %[[GRID]]#1 unsigned
// EXPLICIT: %[[ROW:.+]] = remi %[[ROW_ID]], %[[GRID]]#0 unsigned
// EXPLICIT-NOT: remi
// EXPLICIT: load_view_tko weak %[[PEER_PART]][%[[ROW]], %[[COL]]]
// EXPLICIT: %[[ZERO:.+]] = constant <f32: 0.000000e+00> : tile<2x2xf32>
// EXPLICIT: %[[RELU:.+]] = maxf %[[ZERO]], %[[LOADED]] propagate_nan
// EXPLICIT: %[[OUT_PART:.+]] = make_partition_view %[[OUT_VIEW]]
// EXPLICIT: store_view_tko weak %[[RELU]], %[[OUT_PART]]
nv_tensor_ir.graph @static_peer(
    %a: tensor<?x?xf32> {
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>},
    %b: tensor<8x8xf32> {
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>})
    -> tensor<?x?xf32> {
  %r = relu_fwd %a {
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
    : tensor<?x?xf32>
  results %r : tensor<?x?xf32>
}

// -----

// Unused arguments are legal. With only rank-three operations but a rank-two
// unused input, mixed-rank tile selection must include the output rank and
// skip the non-tensor uniform argument.
// EXPLICIT-LABEL: entry @output_rank
// EXPLICIT-SAME: (%[[IN:[^:]+]]: tile<ptr<f32>>, %{{[^:]+}}: tile<f32>,
// EXPLICIT-SAME: %[[OUT:[^:]+]]: tile<ptr<f32>>)
// EXPLICIT: %[[IN_VIEW:.+]] = make_tensor_view %[[IN]], shape = [8, 8]
// EXPLICIT: %[[OUT_VIEW:.+]] = make_tensor_view %[[OUT]], shape = [8, 8, 4]
// EXPLICIT: %[[VALUE:.+]] = constant <f32: 1.000000e+00> : tile<2x2x2xf32>
// EXPLICIT: %[[OUT_PART:.+]] = make_partition_view %[[OUT_VIEW]]
// EXPLICIT-SAME: partition_view<tile=(2x2x2),
// EXPLICIT: store_view_tko weak %[[VALUE]], %[[OUT_PART]]
nv_tensor_ir.graph @output_rank(
    %a: tensor<8x8xf32> {
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1)>},
    %s: f32) -> tensor<8x8x4xf32> {
  %r = "nv_tensor_ir.constant"() {
    value = dense<1.0> : tensor<8x8x4xf32>,
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  } : () -> tensor<8x8x4xf32>
  results %r : tensor<8x8x4xf32>
}

// -----

// A size-one peer needs clamping when the defining input extent is dynamic.
// The peer load must use a coordinate reduced by its own index-space extent,
// then broadcast the loaded tile into the dynamic input's tile shape.
// EXPLICIT-LABEL: entry @dynamic_broadcast_peer
// EXPLICIT-SAME: (%[[IN:[^:]+]]: tile<ptr<f32>>, %{{[^:]+}}: tile<i32>,
// EXPLICIT-SAME: %[[PEER:[^:]+]]: tile<ptr<f32>>, %[[OUT:[^:]+]]: tile<ptr<f32>>,
// EXPLICIT: %[[VIEW:.+]] = make_tensor_view %[[IN]],
// EXPLICIT: %[[PEER_VIEW:.+]] = make_tensor_view %[[PEER]], shape = [1, 8]
// EXPLICIT: %[[OUT_VIEW:.+]] = make_tensor_view %[[OUT]],
// EXPLICIT: %[[GRID_PART:.+]] = make_partition_view %[[VIEW]]
// EXPLICIT: %[[GRID:.+]]:2 = get_index_space_shape %[[GRID_PART]]
// EXPLICIT: %[[IN_PART:.+]] = make_partition_view %[[VIEW]]
// EXPLICIT: %[[LOADED:.+]], %{{.+}} = load_view_tko weak %[[IN_PART]]
// EXPLICIT: %[[PEER_PART:.+]] = make_partition_view %[[PEER_VIEW]]
// EXPLICIT: %[[ROW:.+]] = remi %{{.+}}, %[[GRID]]#0 unsigned
// EXPLICIT: %[[PEER_COUNTS:.+]]:2 = get_index_space_shape %[[PEER_PART]]
// EXPLICIT: %[[CLAMPED:.+]] = remi %[[ROW]], %[[PEER_COUNTS]]#0 unsigned
// EXPLICIT: %[[PEER_TILE:.+]], %{{.+}} = load_view_tko weak %[[PEER_PART]][%[[CLAMPED]],
// EXPLICIT-SAME: -> tile<1x2xf32>, token
// EXPLICIT: %[[BROADCAST:.+]] = broadcast %[[PEER_TILE]]
// EXPLICIT-SAME: : tile<1x2xf32> -> tile<2x2xf32>
// EXPLICIT: %[[SUM:.+]] = addf %[[LOADED]], %[[BROADCAST]]
// EXPLICIT: %[[OUT_PART:.+]] = make_partition_view %[[OUT_VIEW]]
// EXPLICIT: store_view_tko weak %[[SUM]], %[[OUT_PART]]
nv_tensor_ir.graph @dynamic_broadcast_peer(
    %a: tensor<?x8xf32> {
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>},
    %b: tensor<1x8xf32> {
      nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>})
    -> tensor<?x8xf32> {
  %broadcast = broadcast %b {
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
    : tensor<1x8xf32> -> tensor<?x8xf32>
  %r = add %a, %broadcast {
    nv_tensor_ir.iter_space_map = affine_map<(d0, d1) -> (d0, d1)>}
    : tensor<?x8xf32>
  results %r : tensor<?x8xf32>
}
