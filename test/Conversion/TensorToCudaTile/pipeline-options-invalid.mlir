// RUN: not tensor_ir-opt %s -layout-propagation-pipeline="tile-size=1x8,64" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TILE_SIZE
// RUN: not tensor_ir-opt %s "-layout-propagation-pipeline=tile-size=4x1024 reduction-tile-size=3" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REDUCTION_TILE_SIZE
// RUN: not tensor_ir-opt %s -layout-propagation-pipeline='occupancy=0' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=OCCUPANCY
// RUN: not tensor_ir-opt %s -layout-propagation-pipeline='persistence-mode=static sm-count=0' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PERSISTENCE

// Pipeline options are validated before the input is processed.

// TILE_SIZE: invalid value for 'tile-size'
// REDUCTION_TILE_SIZE: reduction-tile-size: must be a positive power of two (got 3)
// OCCUPANCY: occupancy: must be > 0 (got 0)
// PERSISTENCE: Static persistent kernels require smCount > 0, but got 0
module {}
