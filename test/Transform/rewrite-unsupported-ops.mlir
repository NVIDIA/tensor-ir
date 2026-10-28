// RUN: tensor_ir-opt -materialize-default-strides --tir-rewrite-unsupported-ops -split-input-file %s | FileCheck %s

// CHECK-LABEL: @matmul_bf16_bf16_bf16
nv_tensor_ir.graph @matmul_bf16_bf16_bf16(
  %a: tensor<128x64xbf16>,
  %b: tensor<64x128xbf16>) -> (
    tensor<128x128xbf16>) {
  %c = matmul(%a, %b) : (tensor<128x64xbf16>, tensor<64x128xbf16>) -> tensor<128x128xbf16>
  // CHECK: %[[res:.*]] = matmul(%{{.*}}, %{{.*}}) : (tensor<128x64xbf16>, tensor<64x128xbf16>) -> tensor<128x128xf32>
  // CHECK: %{{.*}} = convert %[[res]] : tensor<128x128xf32> -> tensor<128x128xbf16>
  results %c : tensor<128x128xbf16>
}

// -----

// CHECK-LABEL: @matmul_bf16_bf16_f16
nv_tensor_ir.graph @matmul_bf16_bf16_f16(
  %a: tensor<128x64xbf16>,
  %b: tensor<64x128xbf16>) -> (
    tensor<128x128xf16>) {
  %c = matmul(%a, %b) : (tensor<128x64xbf16>, tensor<64x128xbf16>) -> tensor<128x128xf16>
  // CHECK: %[[res:.*]] = matmul(%{{.*}}, %{{.*}}) : (tensor<128x64xbf16>, tensor<64x128xbf16>) -> tensor<128x128xf32>
  // CHECK: %{{.*}} = convert %[[res]] : tensor<128x128xf32> -> tensor<128x128xf16>
  results %c : tensor<128x128xf16>
}

// -----

// CHECK-LABEL: @matmul_f32_f32_f16
nv_tensor_ir.graph @matmul_f32_f32_f16(
  %a: tensor<128x64xf32>,
  %b: tensor<64x128xf32>) -> (
    tensor<128x128xf16>) {
  %c = matmul(%a, %b) : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf16>
  // CHECK: %[[res:.*]] = matmul(%{{.*}}, %{{.*}}) : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
  // CHECK: %{{.*}} = convert %[[res]] : tensor<128x128xf32> -> tensor<128x128xf16>
  results %c : tensor<128x128xf16>
}

// -----

// CHECK-LABEL: @matmul_bf16_bf16_bf16_acc
nv_tensor_ir.graph @matmul_bf16_bf16_bf16_acc(
  %a: tensor<128x64xbf16>,
  %b: tensor<64x128xbf16>,
  %acc: tensor<128x128xbf16>) -> (
    tensor<128x128xbf16>) {
  // CHECK: %[[acc:.*]] = convert %{{.*}} : tensor<128x128xbf16> -> tensor<128x128xf32>
  %c = matmul(%a, %b) accum(%acc : tensor<128x128xbf16>) : (tensor<128x64xbf16>, tensor<64x128xbf16>) -> tensor<128x128xbf16>
  // CHECK: %[[res:.*]] = matmul(%{{.*}}, %{{.*}}) accum(%[[acc]] : tensor<128x128xf32>) : (tensor<128x64xbf16>, tensor<64x128xbf16>) -> tensor<128x128xf32>
  // CHECK: %{{.*}} = convert %[[res]] : tensor<128x128xf32> -> tensor<128x128xbf16>
  results %c : tensor<128x128xbf16>
}
