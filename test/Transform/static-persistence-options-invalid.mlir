// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-static-persistence))' --verify-diagnostics
// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-static-persistence{sm-count=-1}))' --verify-diagnostics
// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-static-persistence{sm-count=2 occupancy=0}))' --verify-diagnostics
// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-static-persistence{sm-count=2 occupancy=-1}))' --verify-diagnostics

// expected-error @below {{static persistence requires sm-count >= 1 and occupancy >= 1}}
func.func @invalid_options() {
  return
}
