// RUN: tensor_ir-opt "-layout-propagation-pipeline=tile-size=4 tile-size=1024 reduction-tile-size=3" %s --verify-diagnostics

// The public pipeline nests reduction tiling under func.func, so a function is
// required to exercise option validation through the full pipeline.
module {
  // expected-error @below {{reduction-tile-size must be a positive power of two, got 3}}
  func.func @invalid_reduction_tile_size() {
    return
  }
}
