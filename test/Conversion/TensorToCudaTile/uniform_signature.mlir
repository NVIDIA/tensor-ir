// RUN: tensor_ir-opt --pass-pipeline="builtin.module(nv_tensor_ir.graph(materialize-default-strides, discover-iteration-space-info), convert-tensor-to-cuda-tile{codegen-strategy=affine_map uniform-signature=true})" -split-input-file %s | FileCheck --check-prefix=UNIFORM %s
// RUN: tensor_ir-opt --pass-pipeline="builtin.module(nv_tensor_ir.graph(materialize-default-strides, discover-iteration-space-info), convert-tensor-to-cuda-tile{codegen-strategy=affine_map uniform-signature=false})" -split-input-file %s | FileCheck --check-prefix=NON-UNIFORM %s

// ============================================================================
// Tests for --uniform-signature flag.
//
// With uniform-signature=true, ALL tensor sizes and strides are emitted as
// kernel arguments, even for statically-known dimensions.
// With uniform-signature=false (default), only dynamic dims produce args.
// ============================================================================

// Case 1: Fully static shape and stride.
// uniform=true:  ptr + 2 sizes + 2 strides = 5 args per tensor (10 total)
// uniform=false: ptr only = 1 arg per tensor (2 total)

// UNIFORM-LABEL: entry @static_shape_static_stride
//  UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       UNIFORM: make_tensor_view %[[IN_PTR]], shape = [64, 128], strides = [1, 64]

// NON-UNIFORM-LABEL: entry @static_shape_static_stride
//  NON-UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       NON-UNIFORM: make_tensor_view %[[IN_PTR]], shape = [64, 128], strides = [1, 64]
nv_tensor_ir.graph @static_shape_static_stride(
    %in: tensor<64x128xf32> {nv_tensor_ir.stride = "(1,64)"}
) -> (tensor<64x128xf32> {nv_tensor_ir.stride = "(1,64)"}) {
    %relu_fwd = relu_fwd %in : tensor<64x128xf32>
    results %relu_fwd : tensor<64x128xf32>
}

// -----

// Case 2: Fully static shape, no explicit stride.
// uniform=true:  ptr + 2 sizes = 3 args per tensor (6 total, no strides
//                since hasExplicitStride=false)
// uniform=false: ptr only = 1 arg per tensor (2 total)

// UNIFORM-LABEL: entry @static_shape_no_stride
//  UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       UNIFORM: make_tensor_view %[[IN_PTR]], shape = [64, 128], strides = [128, 1]

// NON-UNIFORM-LABEL: entry @static_shape_no_stride
//  NON-UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       NON-UNIFORM: make_tensor_view %[[IN_PTR]], shape = [64, 128], strides = [128, 1]
nv_tensor_ir.graph @static_shape_no_stride(
    %in: tensor<64x128xf32>
) -> tensor<64x128xf32> {
    %relu_fwd = relu_fwd %in : tensor<64x128xf32>
    results %relu_fwd : tensor<64x128xf32>
}

// -----

// Case 3: Partial dynamic shape (?, 128) with partial dynamic stride (1, ?).
// uniform=true:  ptr + 2 sizes + 2 strides = 5 args per tensor (10 total)
// uniform=false: ptr + 1 size + 1 stride = 3 args per tensor (6 total)

// UNIFORM-LABEL: entry @partial_dynamic_shape_stride
//  UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       UNIFORM: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, 128], strides = [1, %{{.+}}]

// NON-UNIFORM-LABEL: entry @partial_dynamic_shape_stride
//  NON-UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       NON-UNIFORM: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, 128], strides = [1, %{{.+}}]
nv_tensor_ir.graph @partial_dynamic_shape_stride(
    %in: tensor<?x128xf32> {nv_tensor_ir.stride = "(1,?)"}
) -> (tensor<?x128xf32> {nv_tensor_ir.stride = "(1,?)"}) {
    %relu_fwd = relu_fwd %in : tensor<?x128xf32>
    results %relu_fwd : tensor<?x128xf32>
}

// -----

// Case 4: Fully dynamic shape and stride.
// uniform=true:  ptr + 2 sizes + 2 strides = 5 args per tensor (10 total)
// uniform=false: ptr + 2 sizes + 2 strides = 5 args per tensor (10 total)
// Both modes produce the same result since everything is already dynamic.

// UNIFORM-LABEL: entry @fully_dynamic
//  UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       UNIFORM: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, %{{.+}}], strides = [%{{.+}}, %{{.+}}]

// NON-UNIFORM-LABEL: entry @fully_dynamic
//  NON-UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       NON-UNIFORM: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, %{{.+}}], strides = [%{{.+}}, %{{.+}}]
nv_tensor_ir.graph @fully_dynamic(
    %in: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,?)"}
) -> (tensor<?x?xf32> {nv_tensor_ir.stride = "(?,?)"}) {
    %relu_fwd = relu_fwd %in : tensor<?x?xf32>
    results %relu_fwd : tensor<?x?xf32>
}

// -----

// Case 5: 3D fully static, no stride.
// uniform=true:  ptr + 3 sizes = 4 args per tensor (8 total)
// uniform=false: ptr only = 1 arg per tensor (2 total)

// UNIFORM-LABEL: entry @static_3d_no_stride
//  UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       UNIFORM: make_tensor_view %[[IN_PTR]], shape = [4, 8, 16], strides = [128, 16, 1]

// NON-UNIFORM-LABEL: entry @static_3d_no_stride
//  NON-UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       NON-UNIFORM: make_tensor_view %[[IN_PTR]], shape = [4, 8, 16], strides = [128, 16, 1]
nv_tensor_ir.graph @static_3d_no_stride(
    %in: tensor<4x8x16xf32>
) -> tensor<4x8x16xf32> {
    %relu_fwd = relu_fwd %in : tensor<4x8x16xf32>
    results %relu_fwd : tensor<4x8x16xf32>
}

// -----

// Case 6: Partial dynamic shape (?, 128), no stride attribute.
// Strides are computed from sizes (no stride args in either mode).
// uniform=true:  ptr + 2 sizes = 3 args per tensor (6 total)
// uniform=false: ptr + 1 size  = 2 args per tensor (4 total)

// UNIFORM-LABEL: entry @partial_dynamic_no_stride
//  UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %{{.+}}: tile<i32>)
//       UNIFORM: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, 128], strides = [128, 1]

// NON-UNIFORM-LABEL: entry @partial_dynamic_no_stride
//  NON-UNIFORM-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>, %[[OUT_PTR:.+]]: tile<ptr<f32>>, %{{.+}}: tile<i32>)
//       NON-UNIFORM: make_tensor_view %[[IN_PTR]], shape = [%{{.+}}, 128], strides = [128, 1]
nv_tensor_ir.graph @partial_dynamic_no_stride(
    %in: tensor<?x128xf32>
) -> tensor<?x128xf32> {
    %relu_fwd = relu_fwd %in : tensor<?x128xf32>
    results %relu_fwd : tensor<?x128xf32>
}
