// RUN: tensor_ir-opt -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation num-ctas=2 occupancy=3 num-warps=8" %s | FileCheck %s

gpu.module @kernels {
  func.func @entry_optimization_hints() attributes {gpu.kernel} {
    return
  }
}

// CHECK-LABEL: entry @entry_optimization_hints(
//  CHECK-SAME: optimization_hints=<default = {num_cta_in_cga = 2, num_worker_warps_per_cta = 8, occupancy = 3}>
