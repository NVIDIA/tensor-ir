// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation num-ctas=2" --verify-diagnostics

module {
  gpu.module @kernels {
  func.func @persistent_kernel() attributes {gpu.kernel} {
    %block = gpu.block_id x
    %grid = gpu.grid_dim x
    %total = arith.constant 4 : index
    // expected-error @+1 {{static persistent kernels are incompatible with CGA clusters (num-ctas > 1)}}
    scf.for %tile = %block to %total step %grid {
    }
    return
  }
  }
}
