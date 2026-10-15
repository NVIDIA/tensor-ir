// RUN: tensor_ir-opt %s -materialize-default-strides -test-layout-normalization -o /dev/null | FileCheck %s

// Each diamond doubles the paths back to the input. Normalization must visit
// shared traversal states once, even when selecting among common carriers.
// CHECK-LABEL: layout normalization for @shared_diamonds
// CHECK: iteration_space: #nv_tensor_ir.tensor_source<0, 0, "(1024):(1)">
// CHECK-NEXT: result_layouts: [#nv_tensor_ir.tensor_source<0, 0, "(1024):(1)">, #nv_tensor_ir.tensor_source<0, 0, "(1024):(1)">]
nv_tensor_ir.graph @shared_diamonds(%x: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
  %a0 = exp %x : tensor<1024xf32>
  %b0 = cos %x : tensor<1024xf32>
  %v0 = add %a0, %b0 : tensor<1024xf32>
  %a1 = exp %v0 : tensor<1024xf32>
  %b1 = cos %v0 : tensor<1024xf32>
  %v1 = add %a1, %b1 : tensor<1024xf32>
  %a2 = exp %v1 : tensor<1024xf32>
  %b2 = cos %v1 : tensor<1024xf32>
  %v2 = add %a2, %b2 : tensor<1024xf32>
  %a3 = exp %v2 : tensor<1024xf32>
  %b3 = cos %v2 : tensor<1024xf32>
  %v3 = add %a3, %b3 : tensor<1024xf32>
  %a4 = exp %v3 : tensor<1024xf32>
  %b4 = cos %v3 : tensor<1024xf32>
  %v4 = add %a4, %b4 : tensor<1024xf32>
  %a5 = exp %v4 : tensor<1024xf32>
  %b5 = cos %v4 : tensor<1024xf32>
  %v5 = add %a5, %b5 : tensor<1024xf32>
  %a6 = exp %v5 : tensor<1024xf32>
  %b6 = cos %v5 : tensor<1024xf32>
  %v6 = add %a6, %b6 : tensor<1024xf32>
  %a7 = exp %v6 : tensor<1024xf32>
  %b7 = cos %v6 : tensor<1024xf32>
  %v7 = add %a7, %b7 : tensor<1024xf32>
  %a8 = exp %v7 : tensor<1024xf32>
  %b8 = cos %v7 : tensor<1024xf32>
  %v8 = add %a8, %b8 : tensor<1024xf32>
  %a9 = exp %v8 : tensor<1024xf32>
  %b9 = cos %v8 : tensor<1024xf32>
  %v9 = add %a9, %b9 : tensor<1024xf32>
  %a10 = exp %v9 : tensor<1024xf32>
  %b10 = cos %v9 : tensor<1024xf32>
  %v10 = add %a10, %b10 : tensor<1024xf32>
  %a11 = exp %v10 : tensor<1024xf32>
  %b11 = cos %v10 : tensor<1024xf32>
  %v11 = add %a11, %b11 : tensor<1024xf32>
  %a12 = exp %v11 : tensor<1024xf32>
  %b12 = cos %v11 : tensor<1024xf32>
  %v12 = add %a12, %b12 : tensor<1024xf32>
  %a13 = exp %v12 : tensor<1024xf32>
  %b13 = cos %v12 : tensor<1024xf32>
  %v13 = add %a13, %b13 : tensor<1024xf32>
  %a14 = exp %v13 : tensor<1024xf32>
  %b14 = cos %v13 : tensor<1024xf32>
  %v14 = add %a14, %b14 : tensor<1024xf32>
  %a15 = exp %v14 : tensor<1024xf32>
  %b15 = cos %v14 : tensor<1024xf32>
  %v15 = add %a15, %b15 : tensor<1024xf32>
  %a16 = exp %v15 : tensor<1024xf32>
  %b16 = cos %v15 : tensor<1024xf32>
  %v16 = add %a16, %b16 : tensor<1024xf32>
  %a17 = exp %v16 : tensor<1024xf32>
  %b17 = cos %v16 : tensor<1024xf32>
  %v17 = add %a17, %b17 : tensor<1024xf32>
  %a18 = exp %v17 : tensor<1024xf32>
  %b18 = cos %v17 : tensor<1024xf32>
  %v18 = add %a18, %b18 : tensor<1024xf32>
  %a19 = exp %v18 : tensor<1024xf32>
  %b19 = cos %v18 : tensor<1024xf32>
  %v19 = add %a19, %b19 : tensor<1024xf32>
  %a20 = exp %v19 : tensor<1024xf32>
  %b20 = cos %v19 : tensor<1024xf32>
  %v20 = add %a20, %b20 : tensor<1024xf32>
  %a21 = exp %v20 : tensor<1024xf32>
  %b21 = cos %v20 : tensor<1024xf32>
  %v21 = add %a21, %b21 : tensor<1024xf32>
  %a22 = exp %v21 : tensor<1024xf32>
  %b22 = cos %v21 : tensor<1024xf32>
  %v22 = add %a22, %b22 : tensor<1024xf32>
  %a23 = exp %v22 : tensor<1024xf32>
  %b23 = cos %v22 : tensor<1024xf32>
  %v23 = add %a23, %b23 : tensor<1024xf32>
  %a24 = exp %v23 : tensor<1024xf32>
  %b24 = cos %v23 : tensor<1024xf32>
  %v24 = add %a24, %b24 : tensor<1024xf32>
  %a25 = exp %v24 : tensor<1024xf32>
  %b25 = cos %v24 : tensor<1024xf32>
  %v25 = add %a25, %b25 : tensor<1024xf32>
  %a26 = exp %v25 : tensor<1024xf32>
  %b26 = cos %v25 : tensor<1024xf32>
  %v26 = add %a26, %b26 : tensor<1024xf32>
  %a27 = exp %v26 : tensor<1024xf32>
  %b27 = cos %v26 : tensor<1024xf32>
  %v27 = add %a27, %b27 : tensor<1024xf32>
  %a28 = exp %v27 : tensor<1024xf32>
  %b28 = cos %v27 : tensor<1024xf32>
  %v28 = add %a28, %b28 : tensor<1024xf32>
  %a29 = exp %v28 : tensor<1024xf32>
  %b29 = cos %v28 : tensor<1024xf32>
  %v29 = add %a29, %b29 : tensor<1024xf32>
  %a30 = exp %v29 : tensor<1024xf32>
  %b30 = cos %v29 : tensor<1024xf32>
  %v30 = add %a30, %b30 : tensor<1024xf32>
  %a31 = exp %v30 : tensor<1024xf32>
  %b31 = cos %v30 : tensor<1024xf32>
  %v31 = add %a31, %b31 : tensor<1024xf32>
  %out = exp %v31 : tensor<1024xf32>
  results %v31, %out : tensor<1024xf32>, tensor<1024xf32>
}
