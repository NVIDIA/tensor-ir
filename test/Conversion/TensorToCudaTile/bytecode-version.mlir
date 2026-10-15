// RUN: tensor_ir-opt %s -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map target-bytecode-version=13.1" -split-input-file -verify-diagnostics
// RUN: tensor_ir-opt %s -layout-propagation-pipeline="bytecode-version=13.1" -split-input-file -verify-diagnostics
// RUN: tensor_ir-opt %s -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map target-bytecode-version=13.3" -split-input-file | FileCheck %s --check-prefix=V13_3
// RUN: tensor_ir-opt %s -layout-propagation-pipeline="bytecode-version=13.3" -split-input-file | FileCheck %s --check-prefix=V13_3

// V13_3-LABEL: entry @atan2_version
// V13_3: atan2
nv_tensor_ir.graph @atan2_version(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+2 {{atan2 requires CUDA Tile bytecode version 13.2 or newer, but target-bytecode-version is 13.1}}
  // expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.atan2'}}
  %result = atan2 %a, %b : tensor<128xf32>
  results %result : tensor<128xf32>
}

// -----

// Attribute requirements matter even when the operation itself dates to 13.1.
// V13_3-LABEL: entry @approx_exp_version
// V13_3: exp {{.*}} rounding<approx>
// expected-error @+1 {{requires CUDA Tile bytecode version 13.3 or newer, but target-bytecode-version is 13.1}}
module {
  nv_tensor_ir.graph @approx_exp_version(%a: tensor<128xf32>) -> tensor<128xf32> {
    %result = exp %a <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<approx>}> : tensor<128xf32>
    results %result : tensor<128xf32>
  }
}
// -----

// Nested element types also raise the required version.
// V13_3-LABEL: entry @type_version
// V13_3: ftof
// expected-error @+1 {{requires CUDA Tile bytecode version 13.3 or newer, but target-bytecode-version is 13.1}}
module {
  nv_tensor_ir.graph @type_version(%a: tensor<128xf4E2M1FN>) -> tensor<128xf32> {
    %result = convert %a : tensor<128xf4E2M1FN> -> tensor<128xf32>
    results %result : tensor<128xf32>
  }
}
