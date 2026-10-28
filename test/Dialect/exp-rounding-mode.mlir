// RUN: tensor_ir-opt %s --cse | FileCheck %s
// RUN: tensor_ir-opt %s --emit-bytecode | tensor_ir-opt --cse | FileCheck %s

// Policies are semantic: CSE merges equal policies, not full with approximate.
// CHECK-LABEL: nv_tensor_ir.graph @different_policies(
// CHECK-SAME: %[[INPUT:.*]]: tensor<128xf32>
// CHECK: %[[FULL:.*]] = exp %[[INPUT]] : tensor<128xf32>
// CHECK: %[[APPROX:.*]] = exp %[[INPUT]] {{<\{?rounding_mode = (#nv_tensor_ir.exp_rounding_mode)?<?approx>?\}?>}}
// CHECK-NOT: = exp
// CHECK: results %[[FULL]], %[[APPROX]], %[[FULL]]
nv_tensor_ir.graph @different_policies(%x: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
  %full = exp %x <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<full>}> : tensor<128xf32>
  %approx = exp %x <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<approx>}> : tensor<128xf32>
  %duplicate = exp %x : tensor<128xf32>
  results %full, %approx, %duplicate : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
}

// Properties and discardable attributes print in separate dictionaries.
// CHECK-LABEL: nv_tensor_ir.graph @separate_dictionaries(
// CHECK: exp %{{.*}} {{<\{?rounding_mode = (#nv_tensor_ir.exp_rounding_mode)?<?approx>?\}?>}} {test.tag = "kept"} : tensor<128xf32>
nv_tensor_ir.graph @separate_dictionaries(%x: tensor<128xf32>) -> tensor<128xf32> {
  %y = exp %x <{rounding_mode = #nv_tensor_ir.exp_rounding_mode<approx>}> {test.tag = "kept"} : tensor<128xf32>
  results %y : tensor<128xf32>
}
