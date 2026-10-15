// RUN: tensor_ir-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg)' \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(convert-tensor-to-linalg,convert-tensor-to-linalg)' \
// RUN:   | tensor_ir-opt --split-input-file --mlir-print-local-scope \
// RUN:   | FileCheck %s --implicit-check-not=nv_tensor_ir.

// CHECK-LABEL: func.func @reduce_add(
nv_tensor_ir.graph @reduce_add(%a: tensor<2x3x4xf32>) -> tensor<1x3x1xf32> {
  %r = reduce(%a)<dimensions = [0, 2], reduction_mode = <add>> : tensor<2x3x4xf32> -> tensor<1x3x1xf32>
  results %r : tensor<1x3x1xf32>
}
// CHECK: %[[ID:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: linalg.fill ins(%[[ID]] : f32)
// CHECK: linalg.reduce
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0, 2]

// -----

// CHECK-LABEL: func.func @reduce_mul(
nv_tensor_ir.graph @reduce_mul(%a: tensor<2x3x4xf32>) -> tensor<1x3x1xf32> {
  %r = reduce(%a)<dimensions = [0, 2], reduction_mode = <mul>> : tensor<2x3x4xf32> -> tensor<1x3x1xf32>
  results %r : tensor<1x3x1xf32>
}
// CHECK: %[[ID:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: linalg.fill ins(%[[ID]] : f32)
// CHECK: linalg.reduce
// CHECK-SAME: arith.mulf
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0, 2]

// -----

// CHECK-LABEL: func.func @reduce_min(
nv_tensor_ir.graph @reduce_min(%a: tensor<2x3x4xf32>) -> tensor<1x3x1xf32> {
  %r = reduce(%a)<dimensions = [0, 2], reduction_mode = <min>> : tensor<2x3x4xf32> -> tensor<1x3x1xf32>
  results %r : tensor<1x3x1xf32>
}
// CHECK: %[[ID:.*]] = arith.constant 0x7F800000 : f32
// CHECK: linalg.fill ins(%[[ID]] : f32)
// CHECK: linalg.reduce
// CHECK: arith.minimumf
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0, 2]

// -----

// CHECK-LABEL: func.func @reduce_max(
nv_tensor_ir.graph @reduce_max(%a: tensor<2x3x4xf32>) -> tensor<1x3x1xf32> {
  %r = reduce(%a)<dimensions = [0, 2], reduction_mode = <max>> : tensor<2x3x4xf32> -> tensor<1x3x1xf32>
  results %r : tensor<1x3x1xf32>
}
// CHECK: %[[ID:.*]] = arith.constant 0xFF800000 : f32
// CHECK: linalg.fill ins(%[[ID]] : f32)
// CHECK: linalg.reduce
// CHECK: arith.maximumf
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0, 2]

// -----

// CHECK-LABEL: func.func @reduce_amax(
nv_tensor_ir.graph @reduce_amax(%a: tensor<2x3x4xf32>) -> tensor<1x3x1xf32> {
  %r = reduce(%a)<dimensions = [0, 2], reduction_mode = <amax>> : tensor<2x3x4xf32> -> tensor<1x3x1xf32>
  results %r : tensor<1x3x1xf32>
}
// CHECK: %[[ID:.*]] = arith.constant 0xFF800000 : f32
// CHECK: linalg.fill ins(%[[ID]] : f32)
// CHECK: linalg.reduce
// CHECK: math.absf
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0, 2]

// -----

// CHECK-LABEL: func.func @reduce_avg(
nv_tensor_ir.graph @reduce_avg(%a: tensor<2x3x4xf32>) -> tensor<1x3x1xf32> {
  %r = reduce(%a)<dimensions = [0, 2], reduction_mode = <avg>> : tensor<2x3x4xf32> -> tensor<1x3x1xf32>
  results %r : tensor<1x3x1xf32>
}
// CHECK: %[[ID:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: linalg.fill ins(%[[ID]] : f32)
// CHECK: linalg.reduce
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0, 2]
// CHECK: arith.muli
// CHECK: arith.index_cast
// CHECK: arith.sitofp
// CHECK: linalg.generic
// CHECK: arith.divf

// -----

// CHECK-LABEL: func.func @reduce_norm1(
nv_tensor_ir.graph @reduce_norm1(%a: tensor<2x3x4xf32>) -> tensor<1x3x1xf32> {
  %r = reduce(%a)<dimensions = [0, 2], reduction_mode = <norm1>> : tensor<2x3x4xf32> -> tensor<1x3x1xf32>
  results %r : tensor<1x3x1xf32>
}
// CHECK: %[[ID:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: linalg.fill ins(%[[ID]] : f32)
// CHECK: linalg.reduce
// CHECK: math.absf
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0, 2]

// -----

// CHECK-LABEL: func.func @reduce_norm2(
nv_tensor_ir.graph @reduce_norm2(%a: tensor<2x3x4xf32>) -> tensor<1x3x1xf32> {
  %r = reduce(%a)<dimensions = [0, 2], reduction_mode = <norm2>> : tensor<2x3x4xf32> -> tensor<1x3x1xf32>
  results %r : tensor<1x3x1xf32>
}
// CHECK: %[[ID:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: linalg.fill ins(%[[ID]] : f32)
// CHECK: linalg.reduce
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0, 2]
// CHECK: linalg.generic
// CHECK: math.sqrt

// -----

// CHECK-LABEL: func.func @reduce_mul_no_zeros(
nv_tensor_ir.graph @reduce_mul_no_zeros(%a: tensor<2x3x4xf32>) -> tensor<1x3x1xf32> {
  %r = reduce(%a)<dimensions = [0, 2], reduction_mode = <mul_no_zeros>> : tensor<2x3x4xf32> -> tensor<1x3x1xf32>
  results %r : tensor<1x3x1xf32>
}
// CHECK: %[[ID:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: linalg.fill ins(%[[ID]] : f32)
// CHECK: linalg.reduce
// CHECK: arith.cmpf oeq
// CHECK: arith.mulf
// CHECK: arith.select
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0, 2]

// -----

// CHECK-LABEL: func.func @reduce_all(
nv_tensor_ir.graph @reduce_all(%a: tensor<2x3xf32>) -> tensor<1x1xf32> {
  %r = reduce(%a)<dimensions = [0, 1], reduction_mode = <add>> : tensor<2x3xf32> -> tensor<1x1xf32>
  results %r : tensor<1x1xf32>
}
// CHECK: tensor.empty() : tensor<f32>
// CHECK: linalg.reduce
// CHECK-SAME: dimensions = [0, 1]
// CHECK: linalg.broadcast
// CHECK-SAME: ins(%{{.*}} : tensor<f32>)

// -----

// CHECK-LABEL: func.func @reduce_ud_noncommutative(
nv_tensor_ir.graph @reduce_ud_noncommutative(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>, %c: tensor<2x3xf32>, %d: tensor<2x3xf32>) -> (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) {
  %r:4 = reduce_ud(%a, %b, %c, %d)<dimensions = [1], identity = [1.0 : f32, 0.0 : f32, 0.0 : f32, 1.0 : f32]>
      (%a00: f32, %a01: f32, %a10: f32, %a11: f32, %b00: f32, %b01: f32, %b10: f32, %b11: f32) {
    %p00 = arith.mulf %a00, %b00 : f32
    %q00 = arith.mulf %a01, %b10 : f32
    %r00 = arith.addf %p00, %q00 : f32
    %p01 = arith.mulf %a00, %b01 : f32
    %q01 = arith.mulf %a01, %b11 : f32
    %r01 = arith.addf %p01, %q01 : f32
    %p10 = arith.mulf %a10, %b00 : f32
    %q10 = arith.mulf %a11, %b10 : f32
    %r10 = arith.addf %p10, %q10 : f32
    %p11 = arith.mulf %a10, %b01 : f32
    %q11 = arith.mulf %a11, %b11 : f32
    %r11 = arith.addf %p11, %q11 : f32
    nv_tensor_ir.yield %r00, %r01, %r10, %r11 : f32, f32, f32, f32
  } : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32> -> tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>
  results %r#0, %r#1, %r#2, %r#3 : tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>
}
// CHECK: %[[R:.*]]:4 = linalg.reduce
// CHECK-SAME: dimensions = [1]
// CHECK: (%[[B00:.*]]: f32, %[[B01:.*]]: f32, %[[B10:.*]]: f32, %[[B11:.*]]: f32, %[[A00:.*]]: f32, %[[A01:.*]]: f32, %[[A10:.*]]: f32, %[[A11:.*]]: f32) {
// CHECK: %[[P00:.*]] = arith.mulf %[[A00]], %[[B00]] : f32
// CHECK: %[[Q00:.*]] = arith.mulf %[[A01]], %[[B10]] : f32
// CHECK: %[[R00:.*]] = arith.addf %[[P00]], %[[Q00]] : f32
// CHECK: %[[P01:.*]] = arith.mulf %[[A00]], %[[B01]] : f32
// CHECK: linalg.yield %[[R00]],
// CHECK: linalg.broadcast ins(%[[R]]#0
// CHECK: linalg.broadcast ins(%[[R]]#1
// CHECK: linalg.broadcast ins(%[[R]]#2
// CHECK: linalg.broadcast ins(%[[R]]#3

// -----

// CHECK-LABEL: func.func @reduce_unsorted(
nv_tensor_ir.graph @reduce_unsorted(%a: tensor<2x3x4xf32>) -> tensor<1x3x1xf32> {
  %r = reduce(%a)<dimensions = [2, 0], reduction_mode = <add>> : tensor<2x3x4xf32> -> tensor<1x3x1xf32>
  results %r : tensor<1x3x1xf32>
}
// CHECK: linalg.reduce
// CHECK-SAME: dimensions = [0, 2]
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0, 2]

// -----

// CHECK-LABEL: func.func @reduce_ud_unsorted(
nv_tensor_ir.graph @reduce_ud_unsorted(%a: tensor<2x3x4xf32>) -> tensor<3xf32> {
  %r = reduce_ud(%a)<dimensions = [2, 0], identity = [0.0 : f32]>
      (%lhs: f32, %rhs: f32) {
    %sum = arith.addf %lhs, %rhs : f32
    nv_tensor_ir.yield %sum : f32
  } : tensor<2x3x4xf32> -> tensor<3xf32>
  results %r : tensor<3xf32>
}
// CHECK: linalg.reduce
// CHECK-SAME: dimensions = [0, 2]

// -----

// CHECK-LABEL: func.func @reduce_min_finite(
nv_tensor_ir.graph @reduce_min_finite(%a: tensor<4xf8E4M3FN>) -> tensor<1xf8E4M3FN> {
  %r = reduce(%a)<dimensions = [0], reduction_mode = <min>> : tensor<4xf8E4M3FN> -> tensor<1xf8E4M3FN>
  results %r : tensor<1xf8E4M3FN>
}
// CHECK: %[[ID:.*]] = arith.constant 4.480000e+02 : f8E4M3FN
// CHECK: linalg.fill ins(%[[ID]] : f8E4M3FN)

// -----

// CHECK-LABEL: func.func @reduce_max_finite(
nv_tensor_ir.graph @reduce_max_finite(%a: tensor<4xf8E4M3FN>) -> tensor<1xf8E4M3FN> {
  %r = reduce(%a)<dimensions = [0], reduction_mode = <max>> : tensor<4xf8E4M3FN> -> tensor<1xf8E4M3FN>
  results %r : tensor<1xf8E4M3FN>
}
// CHECK: %[[ID:.*]] = arith.constant -4.480000e+02 : f8E4M3FN
// CHECK: linalg.fill ins(%[[ID]] : f8E4M3FN)

// -----

// CHECK-LABEL: func.func @reduce_amax_finite(
nv_tensor_ir.graph @reduce_amax_finite(%a: tensor<4xf8E4M3FN>) -> tensor<1xf8E4M3FN> {
  %r = reduce(%a)<dimensions = [0], reduction_mode = <amax>> : tensor<4xf8E4M3FN> -> tensor<1xf8E4M3FN>
  results %r : tensor<1xf8E4M3FN>
}
// CHECK: %[[ID:.*]] = arith.constant -4.480000e+02 : f8E4M3FN
// CHECK: linalg.fill ins(%[[ID]] : f8E4M3FN)

// -----

// CHECK-LABEL: func.func @reduce_ud_mixed_dropped(
nv_tensor_ir.graph @reduce_ud_mixed_dropped(%a: tensor<?x8xf32>, %b: tensor<4x?xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %r:2 = reduce_ud(%a, %b)<dimensions = [1], identity = [0.0 : f32, 0.0 : f32]>
      (%x: f32, %y: f32, %u: f32, %v: f32) {
    %s = arith.addf %x, %u : f32
    %t = arith.addf %y, %v : f32
    nv_tensor_ir.yield %s, %t : f32, f32
  } : tensor<?x8xf32>, tensor<4x?xf32> -> tensor<4xf32>, tensor<4xf32>
  results %r#0, %r#1 : tensor<4xf32>, tensor<4xf32>
}
// CHECK: tensor.cast %{{.*}} : tensor<?x8xf32> to tensor<4x8xf32>
// CHECK: linalg.reduce
// CHECK-SAME: tensor<4x8xf32>, tensor<4x8xf32>
// CHECK-NOT: linalg.broadcast
// CHECK: return {{.*}} : tensor<4xf32>, tensor<4xf32>

// -----

// CHECK-LABEL: func.func @reduce_ud_mixed_retained(
nv_tensor_ir.graph @reduce_ud_mixed_retained(%a: tensor<4x?xf32>, %b: tensor<?x8xf32>) -> (tensor<4x1xf32>, tensor<4x1xf32>) {
  %r:2 = reduce_ud(%a, %b)<dimensions = [1], identity = [0.0 : f32, 0.0 : f32]>
      (%x: f32, %y: f32, %u: f32, %v: f32) {
    %s = arith.addf %x, %u : f32
    %t = arith.addf %y, %v : f32
    nv_tensor_ir.yield %s, %t : f32, f32
  } : tensor<4x?xf32>, tensor<?x8xf32> -> tensor<4x1xf32>, tensor<4x1xf32>
  results %r#0, %r#1 : tensor<4x1xf32>, tensor<4x1xf32>
}
// CHECK: tensor.cast %{{.*}} : tensor<4x?xf32> to tensor<4x8xf32>
// CHECK: linalg.reduce
// CHECK-SAME: tensor<4x8xf32>, tensor<4x8xf32>
// CHECK: linalg.broadcast
// CHECK: return {{.*}} : tensor<4x1xf32>, tensor<4x1xf32>

// -----

// CHECK-LABEL: func.func @reduce_ud_mixed_dynamic(
nv_tensor_ir.graph @reduce_ud_mixed_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x8xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %r:2 = reduce_ud(%a, %b)<dimensions = [1], identity = [0.0 : f32, 0.0 : f32]>
      (%x: f32, %y: f32, %u: f32, %v: f32) {
    %s = arith.addf %x, %u : f32
    %t = arith.addf %y, %v : f32
    nv_tensor_ir.yield %s, %t : f32, f32
  } : tensor<?x?xf32>, tensor<?x8xf32> -> tensor<?xf32>, tensor<?xf32>
  results %r#0, %r#1 : tensor<?xf32>, tensor<?xf32>
}
// CHECK: tensor.cast %{{.*}} : tensor<?x?xf32> to tensor<?x8xf32>
// CHECK: linalg.reduce
// CHECK-SAME: tensor<?x8xf32>, tensor<?x8xf32>
// CHECK-NOT: linalg.broadcast
// CHECK: return {{.*}} : tensor<?xf32>, tensor<?xf32>
