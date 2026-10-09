// RUN: tensor_ir-compiler %s --dump-artifact=%t.tileir
// RUN: FileCheck %s --input-file=%t.tileir.meta
// CHECK: format=cuda_tile_bytecode
// CHECK: runtime_kernel_name=
// CHECK: kernel_name=
// CHECK: runtime_api=cuLibraryLoadData,cuLibraryGetKernel,cuLaunchKernelEx{{\r?$}}
module {
  nv_tensor_ir.graph @metadata(%a: tensor<8xf32>) -> tensor<8xf32> {
    %result = add %a, %a : tensor<8xf32>
    results %result : tensor<8xf32>
  }
}
