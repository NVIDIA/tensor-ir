// RUN: tensor_ir-compiler %s --launch --verbose 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LAUNCH \
// RUN:       --implicit-check-not="Verifying results"
// RUN: tensor_ir-compiler %s --verify --verbose 2>&1 \
// RUN:   | FileCheck %s --check-prefix=VERIFY
// RUN: tensor_ir-compiler %s --verify --verbose --tolerance=1e-2

// LAUNCH: Launch: yes
// LAUNCH: Verify: no
// LAUNCH: Running on device...
// VERIFY: Launch: yes
// VERIFY: Running on host...
// VERIFY: Running on device...
// VERIFY: Verification passed!

module {
  nv_tensor_ir.graph @compiler_runtime_options(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %result = add %a, %b : tensor<8x8xf32>
    results %result : tensor<8x8xf32>
  }
}
