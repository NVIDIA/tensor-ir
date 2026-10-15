// RUN: tensor_ir-opt %s --discover-iteration-space-info --convert-tensor-to-cuda-tile="codegen-strategy=affine_map" --split-input-file | FileCheck %s
// RUN: tensor_ir-opt %s --layout-propagation-pipeline --split-input-file | FileCheck %s

// The same input with different accuracy policies must remain distinct after
// graph transformations and CSE. Both codegen paths use the shared emitter.
// CHECK-LABEL: entry @explicit_modes(
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko
// CHECK-DAG: %[[FULL:.*]] = exp %[[INPUT]] : tile<{{[0-9]+}}xf32>
// CHECK-DAG: %[[APPROX:.*]] = exp %[[INPUT]] rounding<approx> : tile<{{[0-9]+}}xf32>
// CHECK-DAG: store_view_tko weak %[[FULL]],
// CHECK-DAG: store_view_tko weak %[[APPROX]],
nv_tensor_ir.graph @explicit_modes(%x: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  %full = exp %x <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<full>}> : tensor<128xf32>
  %approx = exp %x <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<approx>}> : tensor<128xf32>
  results %full, %approx : tensor<128xf32>, tensor<128xf32>
}

// -----

// Omitted modes use full even for low-precision inputs, preserving the dtype.
// CHECK-LABEL: entry @default_bf16(
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko
// CHECK: %[[WIDE:.*]] = ftof %[[INPUT]] : tile<{{[0-9]+}}xbf16> -> tile<{{[0-9]+}}xf32>
// CHECK: %[[EXP:.*]] = exp %[[WIDE]] : tile<{{[0-9]+}}xf32>
// CHECK: %[[NARROW:.*]] = ftof %[[EXP]] : tile<{{[0-9]+}}xf32> -> tile<{{[0-9]+}}xbf16>
// CHECK: store_view_tko weak %[[NARROW]],
nv_tensor_ir.graph @default_bf16(%x: tensor<128xbf16>) -> tensor<128xbf16> {
  %full = exp %x : tensor<128xbf16>
  results %full : tensor<128xbf16>
}

// -----

// CHECK-LABEL: entry @approx_f16(
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko
// CHECK: %[[WIDE:.*]] = ftof %[[INPUT]] : tile<{{[0-9]+}}xf16> -> tile<{{[0-9]+}}xf32>
// CHECK: %[[EXP:.*]] = exp %[[WIDE]] rounding<approx> : tile<{{[0-9]+}}xf32>
// CHECK: %[[NARROW:.*]] = ftof %[[EXP]] : tile<{{[0-9]+}}xf32> -> tile<{{[0-9]+}}xf16>
// CHECK: store_view_tko weak %[[NARROW]],
nv_tensor_ir.graph @approx_f16(%x: tensor<128xf16>) -> tensor<128xf16> {
  %approx = exp %x <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<approx>}> : tensor<128xf16>
  results %approx : tensor<128xf16>
}

// -----

// CHECK-LABEL: entry @full_f64(
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko
// CHECK-NOT: ftof
// CHECK: %[[EXP:.*]] = exp %[[INPUT]] : tile<{{[0-9]+}}xf64>
// CHECK-NOT: ftof
// CHECK: store_view_tko weak %[[EXP]],
nv_tensor_ir.graph @full_f64(%x: tensor<128xf64>) -> tensor<128xf64> {
  %full = exp %x <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<full>}> : tensor<128xf64>
  results %full : tensor<128xf64>
}
