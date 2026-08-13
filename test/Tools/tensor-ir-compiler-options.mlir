// RUN: tensor_ir-compiler %s --verbose 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEFAULT \
// RUN:       --implicit-check-not="Reference graph" \
// RUN:       --implicit-check-not="Launching kernel"
// RUN: not tensor_ir-compiler %s --dynamic-dims=16,abc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-DIMS
// RUN: not tensor_ir-compiler %s --dynamic-strides=8,xyz 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-STRIDES
// RUN: not tensor_ir-compiler %s --tile-size=8xy 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-TILE
// RUN: not tensor_ir-compiler %s --iterations=0 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-ITERATIONS

// DEFAULT: Launch: no
// DEFAULT: Compiled successfully
// BAD-DIMS: Invalid dynamic-dims
// BAD-STRIDES: Invalid dynamic-strides
// BAD-TILE: Invalid tile-size
// BAD-ITERATIONS: --iterations must be greater than 0

module {
  nv_tensor_ir.graph @compiler_options(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %result = add %a, %b : tensor<8x8xf32>
    results %result : tensor<8x8xf32>
  }
}
