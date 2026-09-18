// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-tile-reductions{reduction-tile-size=7}))' --verify-diagnostics

// expected-error @below {{reduction-tile-size must be a positive power of two, got 7}}
func.func @invalid_tile_size() {
  return
}
