// RUN: tensor_ir-opt "-layout-propagation-pipeline=tile-size=4 tile-size=1024 reduction-tile-size=3" %s --verify-diagnostics

// expected-error@unknown {{reduction_tile_size must be a positive power of two, got 3}}
module {}
