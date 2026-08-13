// RUN: tensor_ir-compiler %s --verbose --tile-size=64 \
// RUN:   --dump-tileir-bc=%t.bc
// RUN: test -s %t.bc
// RUN: tensor_ir-compiler %s --verbose --tile-size=64 \
// RUN:   --load-tileir-bc=%t.bc
// RUN: env TENSOR_IR_LOAD_TILEIR_BC=%t.bc \
// RUN:   tensor_ir-compiler %s --verbose --tile-size=64
// RUN: not tensor_ir-compiler %s --load-tileir-bc=%t.missing.bc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MISSING
// RUN: tensor_ir-compiler %s --bytecode-version=current \
// RUN:   --dump-tileir-bc=%t.current.bc
// RUN: test -s %t.current.bc

// MISSING: Cannot load Tile IR bytecode

module {
  nv_tensor_ir.graph @compiler_bytecode(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %result = add %a, %b : tensor<8x8xf32>
    results %result : tensor<8x8xf32>
  }
}
