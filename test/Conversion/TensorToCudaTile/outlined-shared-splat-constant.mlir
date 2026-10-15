// RUN: tensor_ir-opt %s --convert-tensor-to-cuda-tile | FileCheck %s

// Both splats retrieve the same converted scalar constant. Its SSA use list
// does not account for uses through the dialect conversion's value mapping.
// Erasing it while converting either splat can cause a double erasure.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    func.func @shared_splat_constant() attributes {gpu.kernel} {
      %c = nv_tensor_ir.constant 0 : si32
      %a = nv_tensor_ir.splat %c : tensor<2xsi32>
      %b = nv_tensor_ir.splat %c : tensor<4xsi32>
      return
    }
  }
}

// CHECK-LABEL: entry @shared_splat_constant()
// CHECK: constant <i32: 0> : tile<2xi32>
// CHECK-NEXT: {{.*}}constant <i32: 0> : tile<4xi32>
// CHECK-NEXT: return
