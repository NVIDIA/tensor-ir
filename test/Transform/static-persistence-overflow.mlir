// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-static-persistence{sm-count=2147483647 occupancy=2}))' --verify-diagnostics
// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-static-persistence{sm-count=9223372036854775807 occupancy=2}))' --verify-diagnostics

// expected-error @below {{static persistent grid size overflows 32-bit index range}}
func.func @grid_size_overflow() {
  return
}
