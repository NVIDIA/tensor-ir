// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s

// CHECK-LABEL: entry @explicit_stride_divisibility
// CHECK-SAME:  (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[IN_M:.+]]: tile<i32>, %[[IN_N:.+]]: tile<i32>, %[[IN_STRIDE0:.+]]: tile<i32>, %{{.+}}: tile<ptr<f32>>
// CHECK:       %[[IN_M_BOUND:.+]] = assume bounded<0, ?>, %[[IN_M]] : tile<i32>
// CHECK:       %[[IN_N_BOUND:.+]] = assume bounded<0, ?>, %[[IN_N]] : tile<i32>
// CHECK:       %[[IN_STRIDE0_DIV:.+]] = assume div_by<8>, %[[IN_STRIDE0]] : tile<i32>
// CHECK:       %[[IN_STRIDE0_BOUND:.+]] = assume bounded<0, ?>, %[[IN_STRIDE0_DIV]] : tile<i32>
// CHECK:       %[[IN_STRIDE0_FINAL:.+]] = assume div_by<8>, %[[IN_STRIDE0_BOUND]] : tile<i32>
// CHECK:       make_tensor_view %[[IN_PTR]], shape = [%[[IN_M_BOUND]], %[[IN_N_BOUND]]], strides = [%[[IN_STRIDE0_FINAL]], 1]
nv_tensor_ir.graph @explicit_stride_divisibility(
    %in: tensor<?x?xf32> {
      nv_tensor_ir.stride = "(? { div = 8 }, 1)"
    }
) -> tensor<?x?xf32> {
    %relu_fwd = relu_fwd %in : tensor<?x?xf32>
    results %relu_fwd : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: entry @implicit_row_major_static_stride_divisibility
// CHECK-SAME:  (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[IN_D0:.+]]: tile<i32>, %[[IN_D2:.+]]: tile<i32>, %{{.+}}: tile<ptr<f32>>
// CHECK:       %[[IN_D0_BOUND:.+]] = assume bounded<0, ?>, %[[IN_D0]] : tile<i32>
// CHECK:       %[[IN_D2_BOUND:.+]] = assume bounded<0, ?>, %[[IN_D2]] : tile<i32>
// CHECK:       %[[C16:.+]] = constant <i32: 16> : tile<i32>
// CHECK:       %[[STRIDE0:.+]] = muli %[[IN_D2_BOUND]], %[[C16]] : tile<i32>
// CHECK:       %[[STRIDE0_DIV:.+]] = assume div_by<16>, %[[STRIDE0]] : tile<i32>
// CHECK:       %[[STRIDE0_BOUND:.+]] = assume bounded<0, ?>, %[[STRIDE0_DIV]] : tile<i32>
// CHECK:       %[[STRIDE0_FINAL:.+]] = assume div_by<16>, %[[STRIDE0_BOUND]] : tile<i32>
// CHECK:       %[[STRIDE1_BOUND:.+]] = assume bounded<0, ?>, %[[IN_D2_BOUND]] : tile<i32>
// CHECK:       make_tensor_view %[[IN_PTR]], shape = [%[[IN_D0_BOUND]], 16, %[[IN_D2_BOUND]]], strides = [%[[STRIDE0_FINAL]], %[[STRIDE1_BOUND]], 1]
nv_tensor_ir.graph @implicit_row_major_static_stride_divisibility(
    %in: tensor<?x16x?xf32>
) -> tensor<?x16x?xf32> {
    %relu_fwd = relu_fwd %in : tensor<?x16x?xf32>
    results %relu_fwd : tensor<?x16x?xf32>
}
