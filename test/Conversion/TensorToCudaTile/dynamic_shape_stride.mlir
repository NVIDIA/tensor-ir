// RUN: tensor_ir-opt -materialize-default-strides -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s

// ============================================================================
// Tests for mixed dynamic shapes and strides.
// Verifies that dynamic sizes and strides are counted independently.
// ============================================================================

// Case 1: shape (?, 128) with stride (1, ?)
// - 1 dynamic size (dim 0)
// - 1 dynamic stride (stride 1)
// Expected: ptr + 1 size arg + 1 stride arg = 3 args per tensor

// CHECK-LABEL: entry @dynamic_shape_partial_dynamic_stride_colmajor
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       CHECK: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, 128], strides = [1, %{{.+}}]
nv_tensor_ir.graph @dynamic_shape_partial_dynamic_stride_colmajor(
    %in: tensor<?x128xf32> {nv_tensor_ir.stride = "(1,?)"}
) -> (tensor<?x128xf32> {nv_tensor_ir.stride = "(1,?)"}) {
    %relu_fwd = relu_fwd %in : tensor<?x128xf32>
    results %relu_fwd : tensor<?x128xf32>
}

// -----

// Case 2: shape (?, ?) with stride (1, ?)
// - 2 dynamic sizes (dim 0 and dim 1)
// - 1 dynamic stride (stride 1)
// Expected: ptr + 2 size args + 1 stride arg = 4 args per tensor

// CHECK-LABEL: entry @fully_dynamic_shape_partial_dynamic_stride
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       CHECK: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, %{{.+}}], strides = [1, %{{.+}}]
nv_tensor_ir.graph @fully_dynamic_shape_partial_dynamic_stride(
    %in: tensor<?x?xf32> {nv_tensor_ir.stride = "(1,?)"}
) -> (tensor<?x?xf32> {nv_tensor_ir.stride = "(1,?)"}) {
    %relu_fwd = relu_fwd %in : tensor<?x?xf32>
    results %relu_fwd : tensor<?x?xf32>
}

// -----

// Case 3: shape (?, ?) with stride (?, ?)
// - 2 dynamic sizes (dim 0 and dim 1)
// - 2 dynamic strides (stride 0 and stride 1)
// Expected: ptr + 2 size args + 2 stride args = 5 args per tensor

// CHECK-LABEL: entry @fully_dynamic_shape_fully_dynamic_stride
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       CHECK: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, %{{.+}}], strides = [%{{.+}}, %{{.+}}]
nv_tensor_ir.graph @fully_dynamic_shape_fully_dynamic_stride(
    %in: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,?)"}
) -> (tensor<?x?xf32> {nv_tensor_ir.stride = "(?,?)"}) {
    %relu_fwd = relu_fwd %in : tensor<?x?xf32>
    results %relu_fwd : tensor<?x?xf32>
}

// -----

// Case 4: shape (?, 128) with static strides (1, 128) - both strides are static
// Note: For col-major layout, stride[0]=1 and stride[1]=dim[0]. Here stride[1]=128
// is explicitly provided as static, even though dim[0] is dynamic.
// - 1 dynamic size (dim 0)
// - 0 dynamic strides (both are static)
// Expected: ptr + 1 size arg + 0 stride args = 2 args per tensor

// CHECK-LABEL: entry @dynamic_shape_static_stride
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>)
//       CHECK: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, 128], strides = [1, 128]
nv_tensor_ir.graph @dynamic_shape_static_stride(
    %in: tensor<?x128xf32> {nv_tensor_ir.stride = "(1,128)"}
) -> (tensor<?x128xf32> {nv_tensor_ir.stride = "(1,128)"}) {
    %relu_fwd = relu_fwd %in : tensor<?x128xf32>
    results %relu_fwd : tensor<?x128xf32>
}

// -----

// Case 5: Fully static shape and stride - baseline test
// - 0 dynamic sizes
// - 0 dynamic strides
// Expected: ptr only = 1 arg per tensor

// CHECK-LABEL: entry @static_shape_static_stride
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK: make_tensor_view %[[IN_PTR]], shape = [64, 128], strides = [1, 64]
nv_tensor_ir.graph @static_shape_static_stride(
    %in: tensor<64x128xf32> {nv_tensor_ir.stride = "(1,64)"}
) -> (tensor<64x128xf32> {nv_tensor_ir.stride = "(1,64)"}) {
    %relu_fwd = relu_fwd %in : tensor<64x128xf32>
    results %relu_fwd : tensor<64x128xf32>
}

// -----

// Case 6: No explicit stride attribute - uses default row-major layout
// For shape (?, 128), default row-major strides are (128, 1)
// Strides are COMPUTED from sizes, not passed as args.
// - 1 dynamic size (dim 0)
// - 0 stride args (computed from sizes)
// Expected: ptr + 1 size arg = 2 args per tensor

// CHECK-LABEL: entry @no_stride_attribute_dynamic
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[IN_SIZE0:.+]]: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>)
//       CHECK: %[[BOUNDED_IN_SIZE0:.+]] = assume bounded<0, ?>, %[[IN_SIZE0]] : tile<i32>
//       CHECK: make_tensor_view %[[IN_PTR]], shape = [%[[BOUNDED_IN_SIZE0]], 128], strides = [128, 1]
nv_tensor_ir.graph @no_stride_attribute_dynamic(
    %in: tensor<?x128xf32>
) -> tensor<?x128xf32> {
    %relu_fwd = relu_fwd %in : tensor<?x128xf32>
    results %relu_fwd : tensor<?x128xf32>
}

// -----

// Case 7: No explicit stride attribute - fully static shape
// For shape (64, 128), default row-major strides are (128, 1) - all static
// - 0 dynamic sizes
// - 0 dynamic strides
// Expected: ptr only = 1 arg per tensor

// CHECK-LABEL: entry @no_stride_attribute_static
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK: make_tensor_view %[[IN_PTR]], shape = [64, 128], strides = [128, 1]
nv_tensor_ir.graph @no_stride_attribute_static(
    %in: tensor<64x128xf32>
) -> tensor<64x128xf32> {
    %relu_fwd = relu_fwd %in : tensor<64x128xf32>
    results %relu_fwd : tensor<64x128xf32>
}

// -----

// Case 8: 3D tensor with partial dynamic shape, no stride attribute
// For shape (?, ?, 128), default row-major strides are (dim[1]*128, 128, 1)
// Strides are COMPUTED from sizes, not passed as args.
// - 2 dynamic sizes
// - 0 stride args (computed from sizes)
// Expected: ptr + 2 size args = 3 args per tensor

// CHECK-LABEL: entry @no_stride_3d_partial_dynamic
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       CHECK: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, %{{.+}}, 128], strides = [%{{.+}}, 128, 1]
nv_tensor_ir.graph @no_stride_3d_partial_dynamic(
    %in: tensor<?x?x128xf32>
) -> tensor<?x?x128xf32> {
    %relu_fwd = relu_fwd %in : tensor<?x?x128xf32>
    results %relu_fwd : tensor<?x?x128xf32>
}

// -----

// Case 9: 3D tensor with fully dynamic shape, no stride attribute
// For shape (?, ?, ?), default row-major strides are (dim[1]*dim[2], dim[2], 1)
// Strides are COMPUTED from sizes, not passed as args.
// - 3 dynamic sizes
// - 0 stride args (computed from sizes)
// Expected: ptr + 3 size args = 4 args per tensor

// CHECK-LABEL: entry @no_stride_3d_fully_dynamic
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       CHECK: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, %{{.+}}, %{{.+}}], strides = [%{{.+}}, %{{.+}}, 1]
nv_tensor_ir.graph @no_stride_3d_fully_dynamic(
    %in: tensor<?x?x?xf32>
) -> tensor<?x?x?xf32> {
    %relu_fwd = relu_fwd %in : tensor<?x?x?xf32>
    results %relu_fwd : tensor<?x?x?xf32>
}

// -----

// Case 10: 3D tensor with only first dim dynamic, no stride attribute
// For shape (?, 64, 128), default row-major strides are (8192, 128, 1)
// Strides are COMPUTED from sizes, not passed as args.
// - 1 dynamic size
// - 0 stride args (computed from sizes)
// Expected: ptr + 1 size arg = 2 args per tensor

// CHECK-LABEL: entry @no_stride_3d_first_dim_dynamic
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>)
//       CHECK: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, 64, 128], strides = [8192, 128, 1]
nv_tensor_ir.graph @no_stride_3d_first_dim_dynamic(
    %in: tensor<?x64x128xf32>
) -> tensor<?x64x128xf32> {
    %relu_fwd = relu_fwd %in : tensor<?x64x128xf32>
    results %relu_fwd : tensor<?x64x128xf32>
}
