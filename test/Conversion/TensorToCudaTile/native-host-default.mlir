// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(layout-propagation-pipeline)' | FileCheck %s

// CHECK: nv_tensor_ir.host_entrypoints = ["tensor_ir_host_entry_copy_dispatch_v1"]
// CHECK-SAME: nv_tensor_ir.host_object = dense_resource<
// CHECK-SAME: nv_tensor_ir.host_triple =
module {
  nv_tensor_ir.graph @copy(
      %arg: tensor<8xf32> {nv_tensor_ir.stride = "(1)"}) ->
      (tensor<8xf32> {nv_tensor_ir.stride = "(1)"})
      attributes {tile_size = array<i32: 4>} {
    results %arg : tensor<8xf32>
  }
}
