// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation" | FileCheck %s

module @outlined_dispatch_module attributes {gpu.container_module} {
  func.func @outlined_dispatch(%output: memref<32xf32>) {
    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@outlined
        blocks in (%one, %one, %one) threads in (%one, %one, %one)
        args(%output : memref<32xf32>)
    return
  }
  gpu.module @kernels {
    func.func @outlined(%output: memref<32xf32>) attributes {gpu.kernel} {
      return
    }
  }
}

// CHECK-NOT: gpu.container_module
// CHECK-NOT: func.func
// CHECK-NOT: gpu.launch_func
// CHECK-NOT: gpu.module
// CHECK-LABEL: cuda_tile.module @cuda_tile_outlined_dispatch_module {
// CHECK-NOT: func.func
// CHECK-NOT: gpu.launch_func
// CHECK-NOT: gpu.module
// CHECK: entry @outlined(%{{.*}}: tile<ptr<f32>>)
// CHECK-NEXT: return
