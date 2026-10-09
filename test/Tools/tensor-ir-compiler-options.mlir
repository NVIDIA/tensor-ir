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
// RUN: tensor_ir-compiler %S/../Integration/Compiler/matmul_8x8x8.mlir \
// RUN:   --tile-size=2,2 --verbose 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CSV-TILE
// RUN: tensor_ir-compiler %s --repeat=7 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REPEAT
// RUN: not tensor_ir-compiler %s --tile-size=8x4,2 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-MIXED-TILE
// RUN: not tensor_ir-compiler %s --iterations=2 --repeat=2 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-REPEAT
// RUN: not tensor_ir-compiler %s --repeat=0 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-REPEAT-VALUE
// RUN: env TIR_OVERRIDE_OPTIONS=#verbose \
// RUN:   tensor_ir-compiler --verbose %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=OVERRIDE --implicit-check-not="Target SM:"

// DEFAULT: Launch: no
// DEFAULT: Compiled successfully
// BAD-DIMS: --dynamic-dims: cannot parse
// BAD-STRIDES: --dynamic-strides: cannot parse
// BAD-TILE: --tile-size: cannot parse
// BAD-ITERATIONS: --iterations must be greater than 0
// CSV-TILE: Invocation: {{.*}}tile-size=2x2
// CSV-TILE: Compiled successfully
// REPEAT: Compiled successfully
// BAD-MIXED-TILE: use either 'x' or ',' as the separator
// BAD-REPEAT: --iterations and --repeat cannot be used together
// BAD-REPEAT-VALUE: --iterations must be greater than 0
// OVERRIDE: Compiled successfully

module {
  nv_tensor_ir.graph @compiler_options(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %result = add %a, %b : tensor<8x8xf32>
    results %result : tensor<8x8xf32>
  }
}
