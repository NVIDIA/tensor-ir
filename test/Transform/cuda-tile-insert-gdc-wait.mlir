// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(cuda_tile.module(cuda_tile.entry(tir-cuda-tile-insert-gdc-wait)))' \
// RUN:   | FileCheck %s

// The wait is placed before the first global-memory access, and its token
// reaches every access it dominates: directly when the access has no token and
// through a join when it already consumes one.

// CHECK-LABEL: cuda_tile.module @token_threading
cuda_tile.module @token_threading {
  // CHECK: entry @kernel(%[[PTR:.+]]: tile<ptr<f32>>)
  entry @kernel(%ptr: tile<ptr<f32>>) {
    %mask = constant <i1: 1> : tile<i1>
    %padding = constant <f32: 0.0> : tile<f32>
    // CHECK:     %[[WAIT:.+]] = gdc_wait_tko -> token
    // CHECK:     %[[VAL:.+]], %[[LTOK:.+]] = load_ptr_tko weak %[[PTR]], {{.+}} token=%[[WAIT]]
    %val, %ltok = load_ptr_tko weak %ptr, %mask, %padding
        : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token
    // CHECK:     %[[JOIN:.+]] = join_tokens %[[WAIT]], %[[LTOK]]
    // CHECK:     store_ptr_tko weak %[[PTR]], %[[VAL]] token=%[[JOIN]]
    %stok = store_ptr_tko weak %ptr, %val token=%ltok
        : tile<ptr<f32>>, tile<f32> -> token
    return
  }
}

// -----

// Work that does not touch global memory stays above the wait so it overlaps
// with the tail of the producer grid.

// CHECK-LABEL: cuda_tile.module @wait_after_pure_compute
cuda_tile.module @wait_after_pure_compute {
  // CHECK: entry @kernel(%[[PTR:.+]]: tile<ptr<f32>>, %[[X:.+]]: tile<f32>)
  entry @kernel(%ptr: tile<ptr<f32>>, %x: tile<f32>) {
    // CHECK:     %[[SQ:.+]] = mulf %[[X]], %[[X]]
    // CHECK:     %[[WAIT:.+]] = gdc_wait_tko -> token
    // CHECK:     store_ptr_tko weak %[[PTR]], %[[SQ]] token=%[[WAIT]]
    %sq = mulf %x, %x rounding<nearest_even> : tile<f32>
    %stok = store_ptr_tko weak %ptr, %sq : tile<ptr<f32>>, tile<f32> -> token
    return
  }
}

// -----

// An access nested in a region is covered by a single wait placed before the
// top-level op that contains it.

// CHECK-LABEL: cuda_tile.module @nested_access
cuda_tile.module @nested_access {
  // CHECK: entry @kernel(%[[PTR:.+]]: tile<ptr<f32>>, %[[COND:.+]]: tile<i1>, %[[X:.+]]: tile<f32>)
  entry @kernel(%ptr: tile<ptr<f32>>, %cond: tile<i1>, %x: tile<f32>) {
    // CHECK:     %[[WAIT:.+]] = gdc_wait_tko -> token
    // CHECK:     if %[[COND]] {
    // CHECK:       store_ptr_tko weak %[[PTR]], %[[X]] token=%[[WAIT]]
    // CHECK:     }
    // CHECK-NOT: gdc_wait_tko
    if %cond {
      %stok = store_ptr_tko weak %ptr, %x : tile<ptr<f32>>, tile<f32> -> token
    }
    return
  }
}

// -----

// A loop-carried access is also covered, and one that already consumes a token
// joins it with the wait token.

// CHECK-LABEL: cuda_tile.module @loop_access
cuda_tile.module @loop_access {
  // CHECK: entry @kernel(%[[PTR:.+]]: tile<ptr<f32>>, %[[X:.+]]: tile<f32>)
  entry @kernel(%ptr: tile<ptr<f32>>, %x: tile<f32>) {
    %lb = constant <i32: 0> : tile<i32>
    %ub = constant <i32: 4> : tile<i32>
    %step = constant <i32: 1> : tile<i32>
    %init = make_token : token
    // CHECK:     %[[INIT:.+]] = make_token : token
    // CHECK:     %[[WAIT:.+]] = gdc_wait_tko -> token
    // CHECK:     for %{{.+}} in {{.+}} iter_values(%[[ITOK:.+]] = %[[INIT]]) -> (token) {
    // CHECK:       %[[JOIN:.+]] = join_tokens %[[WAIT]], %[[ITOK]]
    // CHECK:       %[[NEXT:.+]] = store_ptr_tko weak %[[PTR]], %[[X]] token=%[[JOIN]]
    // CHECK:       continue %[[NEXT]]
    %res = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%tok = %init) -> (token) {
      %next = store_ptr_tko weak %ptr, %x token=%tok
          : tile<ptr<f32>>, tile<f32> -> token
      continue %next : token
    }
    return
  }
}

// -----

// A kernel that already waits explicitly is left alone.

// CHECK-LABEL: cuda_tile.module @explicit_wait
cuda_tile.module @explicit_wait {
  // CHECK: entry @kernel(%[[PTR:.+]]: tile<ptr<f32>>, %[[X:.+]]: tile<f32>)
  entry @kernel(%ptr: tile<ptr<f32>>, %x: tile<f32>) {
    // CHECK:     %[[WAIT:.+]] = gdc_wait_tko -> token
    // CHECK-NOT: gdc_wait_tko
    // CHECK:     store_ptr_tko weak %[[PTR]], %[[X]] token=%[[WAIT]] :
    %wait = gdc_wait_tko -> token
    %stok = store_ptr_tko weak %ptr, %x token=%wait
        : tile<ptr<f32>>, tile<f32> -> token
    return
  }
}

// -----

// Releasing dependents early does not order this kernel's own accesses after
// the preceding grid, so the wait is still inserted.

// CHECK-LABEL: cuda_tile.module @launch_dependents_only
cuda_tile.module @launch_dependents_only {
  // CHECK: entry @kernel(%[[PTR:.+]]: tile<ptr<f32>>, %[[X:.+]]: tile<f32>)
  entry @kernel(%ptr: tile<ptr<f32>>, %x: tile<f32>) {
    // CHECK:     gdc_launch_dependents_tko
    // CHECK:     %[[WAIT:.+]] = gdc_wait_tko -> token
    // CHECK:     store_ptr_tko weak %[[PTR]], %[[X]] token=%[[WAIT]] :
    %dep = gdc_launch_dependents_tko -> token
    %stok = store_ptr_tko weak %ptr, %x : tile<ptr<f32>>, tile<f32> -> token
    return
  }
}

// -----

// Without any global-memory access there is nothing to protect.

// CHECK-LABEL: cuda_tile.module @no_global_memory
cuda_tile.module @no_global_memory {
  // CHECK: entry @kernel
  entry @kernel(%x: tile<f32>) {
    // CHECK-NOT: gdc_wait_tko
    %sq = mulf %x, %x rounding<nearest_even> : tile<f32>
    return
  }
}
