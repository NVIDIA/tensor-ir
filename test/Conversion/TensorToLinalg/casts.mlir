// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)' \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg,convert-tensor-to-linalg)' \
// RUN:   | tensor_ir-opt --split-input-file --mlir-print-local-scope \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.

// CHECK-LABEL: func.func @cast_f16_f32(
nv_tensor_ir.graph @cast_f16_f32(%a: tensor<2xf16>) -> tensor<2xf32> {
  %r = convert %a : tensor<2xf16> -> tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.extf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_f64_f32(
nv_tensor_ir.graph @cast_f64_f32(%a: tensor<2xf64>) -> tensor<2xf32> {
  %r = convert %a : tensor<2xf64> -> tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.truncf
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_f32_i1(
nv_tensor_ir.graph @cast_f32_i1(%a: tensor<2xf32>) -> tensor<2xi1> {
  %r = convert %a : tensor<2xf32> -> tensor<2xi1>
  results %r : tensor<2xi1>
}
// CHECK: arith.cmpf une,
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_i1_f32(
nv_tensor_ir.graph @cast_i1_f32(%a: tensor<2xi1>) -> tensor<2xf32> {
  %r = convert %a : tensor<2xi1> -> tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK: arith.uitofp
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_f32_f32(
nv_tensor_ir.graph @cast_f32_f32(%a: tensor<2xf32>) -> tensor<2xf32> {
  %r = convert %a : tensor<2xf32> -> tensor<2xf32>
  results %r : tensor<2xf32>
}
// CHECK-NOT: arith.
// CHECK-NOT: linalg.generic
// CHECK: return %arg0

// -----

// CHECK-LABEL: func.func @cast_f16_bf16(
nv_tensor_ir.graph @cast_f16_bf16(%a: tensor<2xf16>) -> tensor<2xbf16> {
  %r = convert %a : tensor<2xf16> -> tensor<2xbf16>
  results %r : tensor<2xbf16>
}
// CHECK: arith.convertf %{{.*}} : f16 to bf16
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_bf16_f16(
nv_tensor_ir.graph @cast_bf16_f16(%a: tensor<2xbf16>) -> tensor<2xf16> {
  %r = convert %a : tensor<2xbf16> -> tensor<2xf16>
  results %r : tensor<2xf16>
}
// CHECK: arith.convertf %{{.*}} : bf16 to f16
// CHECK: linalg.yield

// -----

// CHECK-LABEL: func.func @cast_f8E5M2_f8E4M3FN(
nv_tensor_ir.graph @cast_f8E5M2_f8E4M3FN(%a: tensor<2xf8E5M2>) -> tensor<2xf8E4M3FN> {
  %r = convert %a : tensor<2xf8E5M2> -> tensor<2xf8E4M3FN>
  results %r : tensor<2xf8E4M3FN>
}
// CHECK: arith.convertf %{{.*}} : f8E5M2 to f8E4M3FN
// CHECK: linalg.yield
