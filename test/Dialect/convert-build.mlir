// RUN: tensor_ir-opt %s --test-convert-op-build | FileCheck %s

module {
  nv_tensor_ir.graph @convert_build(
    %input: tensor<4x4xf32>
  ) -> tensor<4x4xf16> {
    %0 = convert %input : tensor<4x4xf32> -> tensor<4x4xf16>
    results %0 : tensor<4x4xf16>
  }
}

// CHECK: module attributes {test.convert_builder}

