// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s
// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// Verify type conversion lowering for all supported combinations of input and
// output types (exhaustive).

// ============================================================================
// Floating point to floating point conversions
// ============================================================================

// CHECK-LABEL: @convert_f64_to_f32
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_f64_to_f32(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_f64_to_f16
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_f64_to_f16(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_f64_to_bf16
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_f64_to_bf16(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_f32_to_f16
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_f32_to_f16(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_f32_to_bf16
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_f32_to_bf16(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_f32_to_f64
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_f32_to_f64(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_f16_to_f32
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_f16_to_f32(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_f16_to_f64
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_f16_to_f64(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_f16_to_bf16
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_f16_to_bf16(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_bf16_to_f32
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_bf16_to_f32(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_bf16_to_f64
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_bf16_to_f64(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_bf16_to_f16
// CHECK: ftof %{{.*}} : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_bf16_to_f16(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// ============================================================================
// Signed integer to signed integer conversions
// ============================================================================

// CHECK-LABEL: @convert_si64_to_si32
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_si64_to_si32(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_si64_to_si16
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_si64_to_si16(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_si64_to_si8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_si64_to_si8(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----


// CHECK-LABEL: @convert_si32_to_si64
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_si32_to_si64(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_si32_to_si16
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_si32_to_si16(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_si32_to_si8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_si32_to_si8(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// CHECK-LABEL: @convert_si16_to_si32
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_si16_to_si32(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_si16_to_si8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_si16_to_si8(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// CHECK-LABEL: @convert_si16_to_si64
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_si16_to_si64(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_si8_to_si64
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_si8_to_si64(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_si8_to_si32
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_si8_to_si32(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_si8_to_si16
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_si8_to_si16(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// ============================================================================
// Unsigned integer to unsigned integer conversions
// ============================================================================

// CHECK-LABEL: @convert_ui64_to_ui32
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_ui64_to_ui32(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_ui64_to_ui16
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_ui64_to_ui16(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_ui64_to_ui8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_ui64_to_ui8(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// CHECK-LABEL: @convert_ui32_to_ui64
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_ui32_to_ui64(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_ui32_to_ui16
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_ui32_to_ui16(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_ui32_to_ui8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_ui32_to_ui8(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// CHECK-LABEL: @convert_ui16_to_ui32
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_ui16_to_ui32(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_ui16_to_ui8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_ui16_to_ui8(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// CHECK-LABEL: @convert_ui16_to_ui64
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_ui16_to_ui64(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_ui8_to_ui32
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_ui8_to_ui32(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_ui8_to_ui16
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_ui8_to_ui16(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_ui8_to_ui64
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_ui8_to_ui64(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// ============================================================================
// Signed integer to floating point conversions
// ============================================================================

// CHECK-LABEL: @convert_si64_to_f32
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_si64_to_f32(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_si64_to_f16
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_si64_to_f16(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_si64_to_bf16
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_si64_to_bf16(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_si64_to_f64
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_si64_to_f64(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_si32_to_f32
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_si32_to_f32(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_si32_to_f16
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_si32_to_f16(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_si32_to_bf16
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_si32_to_bf16(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_si32_to_f64
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_si32_to_f64(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_si16_to_f32
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_si16_to_f32(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_si16_to_f16
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_si16_to_f16(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_si16_to_bf16
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_si16_to_bf16(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_si16_to_f64
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_si16_to_f64(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_si8_to_f32
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_si8_to_f32(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_si8_to_f16
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_si8_to_f16(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_si8_to_bf16
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_si8_to_bf16(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_si8_to_f64
// CHECK: itof %{{.*}} signed : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_si8_to_f64(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// ============================================================================
// Unsigned integer to floating point conversions
// ============================================================================

// CHECK-LABEL: @convert_ui64_to_f64
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_ui64_to_f64(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_ui64_to_f32
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_ui64_to_f32(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_ui64_to_f16
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_ui64_to_f16(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_ui64_to_bf16
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_ui64_to_bf16(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_ui32_to_f64
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_ui32_to_f64(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_ui32_to_f32
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_ui32_to_f32(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_ui32_to_f16
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_ui32_to_f16(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_ui32_to_bf16
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_ui32_to_bf16(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_ui16_to_f64
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_ui16_to_f64(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_ui16_to_f32
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_ui16_to_f32(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_ui16_to_f16
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_ui16_to_f16(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_ui16_to_bf16
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_ui16_to_bf16(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_ui8_to_f64
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_ui8_to_f64(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_ui8_to_f32
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_ui8_to_f32(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_ui8_to_f16
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_ui8_to_f16(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_ui8_to_bf16
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_ui8_to_bf16(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// ============================================================================
// Floating point to signed integer conversions
// ============================================================================

// CHECK-LABEL: @convert_f64_to_si64
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_f64_to_si64(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_f64_to_si32
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_f64_to_si32(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_f64_to_si16
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_f64_to_si16(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_f64_to_si8
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_f64_to_si8(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// CHECK-LABEL: @convert_f32_to_si64
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_f32_to_si64(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_f32_to_si32
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_f32_to_si32(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_f32_to_si16
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_f32_to_si16(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_f32_to_si8
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_f32_to_si8(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// CHECK-LABEL: @convert_f16_to_si64
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_f16_to_si64(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_f16_to_si32
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_f16_to_si32(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_f16_to_si16
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_f16_to_si16(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_f16_to_si8
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_f16_to_si8(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// CHECK-LABEL: @convert_bf16_to_si64
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_bf16_to_si64(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_bf16_to_si32
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_bf16_to_si32(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_bf16_to_si16
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_bf16_to_si16(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_bf16_to_si8
// CHECK: ftoi %{{.*}} signed : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_bf16_to_si8(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// ============================================================================
// Floating point to unsigned integer conversions
// ============================================================================

// CHECK-LABEL: @convert_f64_to_ui64
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_f64_to_ui64(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_f64_to_ui32
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_f64_to_ui32(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_f64_to_ui16
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_f64_to_ui16(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_f64_to_ui8
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf64> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_f64_to_ui8(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// CHECK-LABEL: @convert_f32_to_ui64
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_f32_to_ui64(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_f32_to_ui32
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_f32_to_ui32(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_f32_to_ui16
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_f32_to_ui16(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_f32_to_ui8
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf32> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_f32_to_ui8(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// CHECK-LABEL: @convert_f16_to_ui64
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_f16_to_ui64(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_f16_to_ui32
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_f16_to_ui32(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_f16_to_ui16
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_f16_to_ui16(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_f16_to_ui8
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xf16> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_f16_to_ui8(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// CHECK-LABEL: @convert_bf16_to_ui64
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_bf16_to_ui64(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_bf16_to_ui32
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_bf16_to_ui32(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_bf16_to_ui16
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_bf16_to_ui16(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_bf16_to_ui8
// CHECK: ftoi %{{.*}} unsigned : tile<[[DIMS:.*]]xbf16> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_bf16_to_ui8(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// ============================================================================
// Signed to unsigned integer casts
// ============================================================================

// CHECK-LABEL: @convert_si64_to_ui64
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @convert_si64_to_ui64(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_si64_to_ui32
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_si64_to_ui32(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_si64_to_ui16
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_si64_to_ui16(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_si64_to_ui8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_si64_to_ui8(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// CHECK-LABEL: @convert_si32_to_ui64
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_si32_to_ui64(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_si32_to_ui32
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @convert_si32_to_ui32(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_si32_to_ui16
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_si32_to_ui16(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_si32_to_ui8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_si32_to_ui8(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// CHECK-LABEL: @convert_si16_to_ui64
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_si16_to_ui64(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_si16_to_ui32
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_si16_to_ui32(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_si16_to_ui16
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @convert_si16_to_ui16(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_si16_to_ui8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_si16_to_ui8(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// CHECK-LABEL: @convert_si8_to_ui64
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_si8_to_ui64(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_si8_to_ui32
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_si8_to_ui32(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_si8_to_ui16
// CHECK: exti %{{.*}} signed : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_si8_to_ui16(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_si8_to_ui8
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @convert_si8_to_ui8(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// ============================================================================
// Unsigned to signed integer casts
// ============================================================================

// CHECK-LABEL: @convert_ui64_to_si64
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @convert_ui64_to_si64(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_ui64_to_si32
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_ui64_to_si32(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_ui64_to_si16
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_ui64_to_si16(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_ui64_to_si8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi64> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_ui64_to_si8(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// CHECK-LABEL: @convert_ui32_to_si64
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_ui32_to_si64(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_ui32_to_si32
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @convert_ui32_to_si32(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_ui32_to_si16
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_ui32_to_si16(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_ui32_to_si8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi32> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_ui32_to_si8(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// CHECK-LABEL: @convert_ui16_to_si64
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_ui16_to_si64(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_ui16_to_si32
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_ui16_to_si32(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_ui16_to_si16
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @convert_ui16_to_si16(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_ui16_to_si8
// CHECK: trunci %{{.*}} : tile<[[DIMS:.*]]xi16> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_ui16_to_si8(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// CHECK-LABEL: @convert_ui8_to_si64
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_ui8_to_si64(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_ui8_to_si32
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_ui8_to_si32(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_ui8_to_si16
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi8> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_ui8_to_si16(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_ui8_to_si8
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @convert_ui8_to_si8(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// ============================================================================
// Boolean conversions
// ============================================================================

// CHECK-LABEL: @convert_i1_to_f64
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xf64>
nv_tensor_ir.graph @convert_i1_to_f64(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xf64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xf64>
    results %convert : tensor<4x8x16xf64>
}

// -----

// CHECK-LABEL: @convert_i1_to_f32
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xf32>
nv_tensor_ir.graph @convert_i1_to_f32(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xf32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xf32>
    results %convert : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: @convert_i1_to_f16
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xf16>
nv_tensor_ir.graph @convert_i1_to_f16(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xf16>
    results %convert : tensor<4x8x16xf16>
}

// -----

// CHECK-LABEL: @convert_i1_to_bf16
// CHECK: itof %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xbf16>
nv_tensor_ir.graph @convert_i1_to_bf16(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xbf16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xbf16>
    results %convert : tensor<4x8x16xbf16>
}

// -----

// CHECK-LABEL: @convert_i1_to_si64
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_i1_to_si64(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xsi64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xsi64>
    results %convert : tensor<4x8x16xsi64>
}

// -----

// CHECK-LABEL: @convert_i1_to_si32
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_i1_to_si32(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xsi32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xsi32>
    results %convert : tensor<4x8x16xsi32>
}

// -----

// CHECK-LABEL: @convert_i1_to_si16
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_i1_to_si16(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xsi16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xsi16>
    results %convert : tensor<4x8x16xsi16>
}

// -----

// CHECK-LABEL: @convert_i1_to_si8
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_i1_to_si8(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xsi8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xsi8>
    results %convert : tensor<4x8x16xsi8>
}

// -----

// CHECK-LABEL: @convert_i1_to_ui64
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xi64>
nv_tensor_ir.graph @convert_i1_to_ui64(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xui64> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xui64>
    results %convert : tensor<4x8x16xui64>
}

// -----

// CHECK-LABEL: @convert_i1_to_ui32
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xi32>
nv_tensor_ir.graph @convert_i1_to_ui32(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xui32> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xui32>
    results %convert : tensor<4x8x16xui32>
}

// -----

// CHECK-LABEL: @convert_i1_to_ui16
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xi16>
nv_tensor_ir.graph @convert_i1_to_ui16(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xui16> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xui16>
    results %convert : tensor<4x8x16xui16>
}

// -----

// CHECK-LABEL: @convert_i1_to_ui8
// CHECK: exti %{{.*}} unsigned : tile<[[DIMS:.*]]xi1> -> tile<[[DIMS]]xi8>
nv_tensor_ir.graph @convert_i1_to_ui8(%arg0: tensor<4x8x16xi1>) -> tensor<4x8x16xui8> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xi1>) -> tensor<4x8x16xui8>
    results %convert : tensor<4x8x16xui8>
}

// -----

// ============================================================================
// Conversions to boolean (i1): non-zero becomes 1, zero becomes 0.
// ============================================================================

// CHECK-LABEL: @convert_f16_to_i1
// CHECK: %[[ZERO:.*]] = constant <f16: 0.000000e+00> : tile<[[DIMS:.*]]xf16>
// CHECK: cmpf not_equal ordered %{{.*}}, %[[ZERO]] : tile<[[DIMS]]xf16> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_f16_to_i1(%arg0: tensor<4x8x16xf16>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf16>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_bf16_to_i1
// CHECK: %[[ZERO:.*]] = constant <bf16: 0.000000e+00> : tile<[[DIMS:.*]]xbf16>
// CHECK: cmpf not_equal ordered %{{.*}}, %[[ZERO]] : tile<[[DIMS]]xbf16> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_bf16_to_i1(%arg0: tensor<4x8x16xbf16>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xbf16>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_f32_to_i1
// CHECK: %[[ZERO:.*]] = constant <f32: 0.000000e+00> : tile<[[DIMS:.*]]xf32>
// CHECK: cmpf not_equal ordered %{{.*}}, %[[ZERO]] : tile<[[DIMS]]xf32> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_f32_to_i1(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf32>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_f64_to_i1
// CHECK: %[[ZERO:.*]] = constant <f64: 0.000000e+00> : tile<[[DIMS:.*]]xf64>
// CHECK: cmpf not_equal ordered %{{.*}}, %[[ZERO]] : tile<[[DIMS]]xf64> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_f64_to_i1(%arg0: tensor<4x8x16xf64>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xf64>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_si8_to_i1
// CHECK: %[[ZERO:.*]] = constant <i8: 0> : tile<[[DIMS:.*]]xi8>
// CHECK: cmpi not_equal %{{.*}}, %[[ZERO]], signed : tile<[[DIMS]]xi8> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_si8_to_i1(%arg0: tensor<4x8x16xsi8>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi8>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_si16_to_i1
// CHECK: %[[ZERO:.*]] = constant <i16: 0> : tile<[[DIMS:.*]]xi16>
// CHECK: cmpi not_equal %{{.*}}, %[[ZERO]], signed : tile<[[DIMS]]xi16> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_si16_to_i1(%arg0: tensor<4x8x16xsi16>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi16>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_si32_to_i1
// CHECK: %[[ZERO:.*]] = constant <i32: 0> : tile<[[DIMS:.*]]xi32>
// CHECK: cmpi not_equal %{{.*}}, %[[ZERO]], signed : tile<[[DIMS]]xi32> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_si32_to_i1(%arg0: tensor<4x8x16xsi32>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi32>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_si64_to_i1
// CHECK: %[[ZERO:.*]] = constant <i64: 0> : tile<[[DIMS:.*]]xi64>
// CHECK: cmpi not_equal %{{.*}}, %[[ZERO]], signed : tile<[[DIMS]]xi64> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_si64_to_i1(%arg0: tensor<4x8x16xsi64>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xsi64>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_ui8_to_i1
// CHECK: %[[ZERO:.*]] = constant <i8: 0> : tile<[[DIMS:.*]]xi8>
// CHECK: cmpi not_equal %{{.*}}, %[[ZERO]], unsigned : tile<[[DIMS]]xi8> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_ui8_to_i1(%arg0: tensor<4x8x16xui8>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui8>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_ui16_to_i1
// CHECK: %[[ZERO:.*]] = constant <i16: 0> : tile<[[DIMS:.*]]xi16>
// CHECK: cmpi not_equal %{{.*}}, %[[ZERO]], unsigned : tile<[[DIMS]]xi16> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_ui16_to_i1(%arg0: tensor<4x8x16xui16>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui16>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_ui32_to_i1
// CHECK: %[[ZERO:.*]] = constant <i32: 0> : tile<[[DIMS:.*]]xi32>
// CHECK: cmpi not_equal %{{.*}}, %[[ZERO]], unsigned : tile<[[DIMS]]xi32> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_ui32_to_i1(%arg0: tensor<4x8x16xui32>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui32>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}

// -----

// CHECK-LABEL: @convert_ui64_to_i1
// CHECK: %[[ZERO:.*]] = constant <i64: 0> : tile<[[DIMS:.*]]xi64>
// CHECK: cmpi not_equal %{{.*}}, %[[ZERO]], unsigned : tile<[[DIMS]]xi64> -> tile<[[DIMS]]xi1>
nv_tensor_ir.graph @convert_ui64_to_i1(%arg0: tensor<4x8x16xui64>) -> tensor<4x8x16xi1> {
    %convert = "nv_tensor_ir.convert"(%arg0) : (tensor<4x8x16xui64>) -> tensor<4x8x16xi1>
    results %convert : tensor<4x8x16xi1>
}
