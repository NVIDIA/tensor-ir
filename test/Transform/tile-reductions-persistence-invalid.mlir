// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-tile-reductions{persistence=static sm-count=2147483647 occupancy=2}))' --verify-diagnostics

// expected-error @below {{static persistent grid size overflows 32-bit index range}}
func.func @grid_size_overflow() {
  return
}
