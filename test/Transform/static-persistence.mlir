// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(tir-static-persistence{sm-count=2 occupancy=3}))' | FileCheck %s
// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(tir-tile-reductions))' | FileCheck %s --check-prefix=REDUCTIONS

// Occupancy scales the physical grid. All body operations move into the loop,
// including uses of the tile ID nested in other regions.
// REDUCTIONS-LABEL: func.func @static_grid(
// REDUCTIONS: scf.forall (%[[TILE:.*]]) in (20) {
// REDUCTIONS-NEXT: %{{.*}} = arith.constant true
// REDUCTIONS: memref.store %[[TILE]], %{{.*}}[%[[TILE]]]
// CHECK-LABEL: func.func @static_grid(
// CHECK-SAME: %[[OUT:[^:]+]]: memref<20xindex>
func.func @static_grid(%out: memref<20xindex>) {
  // CHECK: %[[TOTAL:.*]] = arith.constant 20 : index
  // CHECK: %[[GRID:.*]] = arith.constant 6 : index
  // CHECK: scf.forall (%[[BLOCK:.*]]) in (6) {
  // CHECK-NEXT: scf.for %[[TILE:.*]] = %[[BLOCK]] to %[[TOTAL]] step %[[GRID]] {
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: scf.if %[[TRUE]] {
  // CHECK: memref.store %[[TILE]], %[[OUT]][%[[TILE]]]
  // CHECK: }
  // CHECK-NEXT: }
  // CHECK-NEXT: } {mapping = [#gpu.block<x>]}
  scf.forall (%tile) in (20) {
    %true = arith.constant true
    scf.if %true {
      memref.store %tile, %out[%tile] : memref<20xindex>
    }
  } {mapping = [#gpu.block<x>]}
  return
}

// -----

// Dynamic bounds retain the original total and clamp the grid to at least one.
// CHECK-LABEL: func.func @dynamic_grid(
// CHECK-SAME: %[[TOTAL:[^:]+]]: index, %[[OUT:[^:]+]]: memref<?xindex>
func.func @dynamic_grid(%total: index, %out: memref<?xindex>) {
  // CHECK: %[[LIMIT:.*]] = arith.constant 6 : index
  // CHECK: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[MIN:.*]] = arith.minui %[[TOTAL]], %[[LIMIT]] : index
  // CHECK: %[[GRID:.*]] = arith.maxui %[[MIN]], %[[ONE]] : index
  // CHECK: scf.forall (%[[BLOCK:.*]]) in (%[[GRID]]) {
  // CHECK-NEXT: scf.for %[[TILE:.*]] = %[[BLOCK]] to %[[TOTAL]] step %[[GRID]] {
  // CHECK-NEXT: memref.store %[[TILE]], %[[OUT]][%[[TILE]]]
  // CHECK-NEXT: }
  // CHECK-NEXT: } {mapping = [#gpu.block<x>]}
  scf.forall (%tile) in (%total) {
    memref.store %tile, %out[%tile] : memref<?xindex>
  } {mapping = [#gpu.block<x>]}
  return
}

// -----

// Below the cap, at the cap, and empty grids remain unchanged.
// CHECK-LABEL: func.func @fits_grid(
func.func @fits_grid(%out: memref<6xindex>) {
  // CHECK: scf.forall (%[[TILE:.*]]) in (3) {
  // CHECK-NEXT: memref.store %[[TILE]], %{{.*}}[%[[TILE]]]
  // CHECK-NEXT: }
  scf.forall (%tile) in (3) {
    memref.store %tile, %out[%tile] : memref<6xindex>
  }
  // CHECK: scf.forall (%[[TILE:.*]]) in (6) {
  // CHECK-NEXT: memref.store %[[TILE]], %{{.*}}[%[[TILE]]]
  // CHECK-NEXT: }
  scf.forall (%tile) in (6) {
    memref.store %tile, %out[%tile] : memref<6xindex>
  }
  // CHECK: scf.forall (%[[TILE:.*]]) in (0) {
  // CHECK-NEXT: memref.store %[[TILE]], %{{.*}}[%[[TILE]]]
  // CHECK-NEXT: }
  scf.forall (%tile) in (0) {
    memref.store %tile, %out[%tile] : memref<6xindex>
  }
  return
}
