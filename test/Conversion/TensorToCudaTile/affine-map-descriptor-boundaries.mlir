// RUN: split-file %s %t
// RUN: tensor_ir-opt %t/descriptors.mlir -split-input-file \
// RUN:   -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   | tensor_ir-opt -split-input-file | FileCheck %s --check-prefix=DESC
// RUN: tensor_ir-opt %t/tile-rank.mlir -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map tile-size=4" \
// RUN:   | tensor_ir-opt | FileCheck %s --check-prefix=SHORT
// RUN: tensor_ir-opt %t/tile-rank.mlir -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map \
// RUN:     tile-size=4,8,16" | tensor_ir-opt | FileCheck %s --check-prefix=LONG
// RUN: tensor_ir-opt %t/persistent-small.mlir -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map \
// RUN:     persistence=static sm-count=148 tile-size=2,2" \
// RUN:   | tensor_ir-opt | FileCheck %s --check-prefix=SMALL \
// RUN:     --implicit-check-not='for %'

//--- descriptors.mlir

// A scalar before a dynamic tensor must consume its own argument slot;
// the output pointer must follow the complete input descriptor.
// DESC-LABEL: entry @scalar_input
// DESC-SAME: (%{{[^:]+}}: tile<f32>, %[[IN:[^:]+]]: tile<ptr<f32>>,
// DESC-SAME: %[[M:[^:]+]]: tile<i32>, %[[OUT:[^:]+]]: tile<ptr<f32>>,
// DESC-SAME: %[[OUT_M:[^:]+]]: tile<i32>)
// DESC: %[[M_BOUND:.+]] = assume bounded<0, ?>, %[[M]] : tile<i32>
// DESC: %[[VIEW:.+]] = make_tensor_view %[[IN]], shape = [%[[M_BOUND]], 8]
// DESC: %[[OM_BOUND:.+]] = assume bounded<0, ?>, %[[OUT_M]] : tile<i32>
// DESC: %[[OUT_VIEW:.+]] = make_tensor_view %[[OUT]],
// DESC-SAME: shape = [%[[OM_BOUND]], 8]
// DESC: %[[TILE:.+]], %{{.+}} = load_view_tko {{.*}} -> tile<2x2xf32>, token
// DESC: %[[ABS:.+]] = absf %[[TILE]] : tile<2x2xf32>
// DESC: %[[PART:.+]] = make_partition_view %[[OUT_VIEW]]
// DESC: store_view_tko weak %[[ABS]], %[[PART]]
nv_tensor_ir.graph @scalar_input(%scale: f32, %a: tensor<?x8xf32>)
    -> tensor<?x8xf32> {
  %r = abs %a : tensor<?x8xf32>
  results %r : tensor<?x8xf32>
}

// -----

// With no input tensors, tile selection must use the output tensor rank.
// DESC-LABEL: entry @constant_output
// DESC-SAME: (%[[OUT:[^:]+]]: tile<ptr<f32>>)
// DESC: %[[VIEW:.+]] = make_tensor_view %[[OUT]], shape = [8, 8]
// DESC: %[[VALUE:.+]] = constant <f32: 2.000000e+00> : tile<2x2xf32>
// DESC: %[[PART:.+]] = make_partition_view %[[VIEW]]
// DESC: store_view_tko weak %[[VALUE]], %[[PART]]
nv_tensor_ir.graph @constant_output() -> tensor<8x8xf32> {
  %r = nv_tensor_ir.constant dense<2.0> : tensor<8x8xf32>
  results %r : tensor<8x8xf32>
}

// -----

// Capping inferred divisibility must not truncate the actual row-major stride:
// 1073741824 * 4 = 4294967296, which needs more than 32 bits.
// DESC-LABEL: entry @divisibility_saturation
// DESC-SAME: (%[[IN:[^:]+]]: tile<ptr<f32>>,
// DESC: %[[VIEW:.+]] = make_tensor_view %[[IN]],
// DESC-SAME: strides = [4294967296, 4, 1]
// DESC: %[[PART:.+]] = make_partition_view %[[VIEW]]
// DESC: %[[TILE:.+]], %{{.+}} = load_view_tko weak %[[PART]]
// DESC: %[[ABS:.+]] = absf %[[TILE]] : tile<2x2x2xf32>
// DESC: store_view_tko weak %[[ABS]],
nv_tensor_ir.graph @divisibility_saturation(
    %a: tensor<?x1073741824x4xf32>) -> tensor<?x1073741824x4xf32> {
  %r = abs %a : tensor<?x1073741824x4xf32>
  results %r : tensor<?x1073741824x4xf32>
}

// -----

// DESC-LABEL: entry @explicit_stride_divisibility
// DESC-SAME:  (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[IN_M:.+]]: tile<i32>, %[[IN_N:.+]]: tile<i32>, %[[IN_STRIDE0:.+]]: tile<i32>, %{{.+}}: tile<ptr<f32>>
// DESC:       %[[IN_M_BOUND:.+]] = assume bounded<0, ?>, %[[IN_M]] : tile<i32>
// DESC:       %[[IN_N_BOUND:.+]] = assume bounded<0, ?>, %[[IN_N]] : tile<i32>
// DESC:       %[[IN_STRIDE0_DIV:.+]] = assume div_by<8>, %[[IN_STRIDE0]] : tile<i32>
// DESC:       %[[IN_STRIDE0_BOUND:.+]] = assume bounded<0, ?>, %[[IN_STRIDE0_DIV]] : tile<i32>
// DESC:       %[[IN_STRIDE0_FINAL:.+]] = assume div_by<8>, %[[IN_STRIDE0_BOUND]] : tile<i32>
// DESC:       make_tensor_view %[[IN_PTR]], shape = [%[[IN_M_BOUND]], %[[IN_N_BOUND]]], strides = [%[[IN_STRIDE0_FINAL]], 1]
nv_tensor_ir.graph @explicit_stride_divisibility(
    %in: tensor<?x?xf32> {
      nv_tensor_ir.stride = "(? { div = 8 }, 1)"
    }
) -> tensor<?x?xf32> {
    %relu_fwd = relu_fwd %in : tensor<?x?xf32>
    results %relu_fwd : tensor<?x?xf32>
}

// -----

// DESC-LABEL: entry @implicit_row_major_static_stride_divisibility
// DESC-SAME:  (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[IN_D0:.+]]: tile<i32>, %[[IN_D2:.+]]: tile<i32>, %{{.+}}: tile<ptr<f32>>
// DESC:       %[[IN_D0_BOUND:.+]] = assume bounded<0, ?>, %[[IN_D0]] : tile<i32>
// DESC:       %[[IN_D2_BOUND:.+]] = assume bounded<0, ?>, %[[IN_D2]] : tile<i32>
// DESC:       %[[C16:.+]] = constant <i32: 16> : tile<i32>
// DESC:       %[[STRIDE0:.+]] = muli %[[IN_D2_BOUND]], %[[C16]] : tile<i32>
// DESC:       %[[STRIDE0_DIV:.+]] = assume div_by<16>, %[[STRIDE0]] : tile<i32>
// DESC:       %[[STRIDE0_BOUND:.+]] = assume bounded<0, ?>, %[[STRIDE0_DIV]] : tile<i32>
// DESC:       %[[STRIDE0_FINAL:.+]] = assume div_by<16>, %[[STRIDE0_BOUND]] : tile<i32>
// DESC:       %[[STRIDE1_BOUND:.+]] = assume bounded<0, ?>, %[[IN_D2_BOUND]] : tile<i32>
// DESC:       make_tensor_view %[[IN_PTR]], shape = [%[[IN_D0_BOUND]], 16, %[[IN_D2_BOUND]]], strides = [%[[STRIDE0_FINAL]], %[[STRIDE1_BOUND]], 1]
nv_tensor_ir.graph @implicit_row_major_static_stride_divisibility(
    %in: tensor<?x16x?xf32>
) -> tensor<?x16x?xf32> {
    %relu_fwd = relu_fwd %in : tensor<?x16x?xf32>
    results %relu_fwd : tensor<?x16x?xf32>
}

//--- tile-rank.mlir

// A short tile list fills missing dimensions with 2; a long list is truncated.
// SHORT: tensor_ir.resolved_tile_size = array<i32: 4, 2>
// LONG: tensor_ir.resolved_tile_size = array<i32: 4, 8>
// SHORT-LABEL: entry @tile_rank
// SHORT: %[[TILE:.+]], %{{.+}} = load_view_tko {{.*}} -> tile<4x2xf32>, token
// SHORT: %[[ABS:.+]] = absf %[[TILE]] : tile<4x2xf32>
// SHORT: store_view_tko weak %[[ABS]],
// LONG-LABEL: entry @tile_rank
// LONG: %[[TILE:.+]], %{{.+}} = load_view_tko {{.*}} -> tile<4x8xf32>, token
// LONG: %[[ABS:.+]] = absf %[[TILE]] : tile<4x8xf32>
// LONG: store_view_tko weak %[[ABS]],
nv_tensor_ir.graph @tile_rank(%a: tensor<8x16xf32>)
    -> tensor<8x16xf32> {
  %r = abs %a : tensor<8x16xf32>
  results %r : tensor<8x16xf32>
}

//--- persistent-small.mlir

// One tile fits within the requested residency: no persistence loop is needed.
// SMALL-LABEL: entry @persistent_small
// SMALL: %[[TILE:.+]], %{{.+}} = load_view_tko {{.*}} -> tile<2x2xf32>, token
// SMALL: %[[ABS:.+]] = absf %[[TILE]] : tile<2x2xf32>
// SMALL: store_view_tko weak %[[ABS]],
// SMALL: return
nv_tensor_ir.graph @persistent_small(%a: tensor<2x2xf32>)
    -> tensor<2x2xf32> {
  %r = abs %a : tensor<2x2xf32>
  results %r : tensor<2x2xf32>
}
