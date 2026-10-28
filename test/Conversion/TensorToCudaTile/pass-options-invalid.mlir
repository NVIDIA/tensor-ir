// RUN: split-file %s %t
// RUN: tensor_ir-opt %t/num-ctas-16.mlir \
// RUN:   -convert-tensor-to-cuda-tile="num-ctas=16" -verify-diagnostics
// RUN: tensor_ir-opt %t/num-ctas-range.mlir \
// RUN:   -convert-tensor-to-cuda-tile="num-ctas=0" -verify-diagnostics
// RUN: tensor_ir-opt %t/num-ctas-upper-range.mlir \
// RUN:   -convert-tensor-to-cuda-tile="num-ctas=9" -verify-diagnostics
// RUN: tensor_ir-opt %t/occupancy.mlir \
// RUN:   -convert-tensor-to-cuda-tile="occupancy=0" -verify-diagnostics
// RUN: tensor_ir-opt %t/num-warps.mlir \
// RUN:   -convert-tensor-to-cuda-tile="num-warps=0" -verify-diagnostics
// RUN: tensor_ir-opt %t/static-persistence-cga.mlir \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map \
// RUN:     persistence=static num-ctas=2 sm-count=80" -verify-diagnostics
// RUN: tensor_ir-opt %t/static-persistence-sm-count.mlir \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map \
// RUN:     persistence=static sm-count=0" -verify-diagnostics
// RUN: tensor_ir-opt %t/already-converted.mlir \
// RUN:   -convert-tensor-to-cuda-tile -verify-diagnostics
// RUN: tensor_ir-opt %t/outlined-affine-input.mlir \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   -verify-diagnostics
// RUN: tensor_ir-opt %t/valid-boundaries.mlir \
// RUN:   -convert-tensor-to-cuda-tile="num-ctas=8 occupancy=1 num-warps=1" \
// RUN:   -verify-diagnostics
// RUN: tensor_ir-opt %t/valid-static-persistence.mlir \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map \
// RUN:     persistence=static num-ctas=1 sm-count=1" -verify-diagnostics

// Verify each pass-level guard through its public option or input surface.
// split-file keeps every failure isolated so -verify-diagnostics proves the
// expected guard fired without accepting unrelated diagnostics.

//--- num-ctas-16.mlir

// expected-error @unknown {{num_ctas = 16 is currently not supported}}
module {}

//--- num-ctas-range.mlir

// expected-error @unknown {{num_ctas must be in range [1, 8]}}
module {}

//--- num-ctas-upper-range.mlir

// expected-error @unknown {{num_ctas must be in range [1, 8]}}
module {}

//--- occupancy.mlir

// expected-error @unknown {{occupancy must be >= 1}}
module {}

//--- num-warps.mlir

// expected-error @unknown {{num_warps must be >= 1}}
module {}

//--- static-persistence-cga.mlir

// expected-error @unknown {{Static persistent kernels are incompatible}}
module {}

//--- static-persistence-sm-count.mlir

// expected-error @unknown {{Static persistent kernels require sm_count >= 1}}
module {}

//--- already-converted.mlir

// expected-error @+1 {{module has already been converted to CUDA Tile}}
module {
  cuda_tile.module @already_converted {
  }
}

//--- outlined-affine-input.mlir

// expected-error @+1 {{affine-map conversion does not accept outlined}}
module {
  gpu.module @kernels {
    func.func @outlined() attributes {gpu.kernel} {
      return
    }
  }
}

//--- valid-boundaries.mlir

// No expected diagnostics: maximum supported cluster/residency values and the
// minimum supported worker count must be accepted.
module {}

//--- valid-static-persistence.mlir

// No expected diagnostics: static persistence is valid with one CTA per
// cluster and a positive SM count.
module {}
