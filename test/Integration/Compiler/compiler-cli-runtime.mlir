// RUN: tensor_ir-compiler %s --launch --verbose 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LAUNCH \
// RUN:       --implicit-check-not="Verifying results"
// RUN: tensor_ir-compiler %s --verify --verbose 2>&1 \
// RUN:   | FileCheck %s --check-prefix=VERIFY
// RUN: tensor_ir-compiler %s --verify --verbose --tolerance=1e-2
// RUN: %PYTHON -c "import struct; open(r'%t.input.0', 'wb').write(struct.pack('<64f', *([1.0] * 64))); open(r'%t.input.1', 'wb').write(struct.pack('<64f', *([2.0] * 64)))"
// RUN: tensor_ir-compiler %s --input-prefix=%t.input \
// RUN:   --output-file=%t.output --verbose 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RAW
// RUN: %PYTHON -c "import struct; data=struct.unpack('<64f', open(r'%t.output', 'rb').read()); assert data == (3.0,) * 64"
// RUN: %PYTHON -c "open(r'%t.short.0', 'wb').write(b'x')"
// RUN: not tensor_ir-compiler %s --input-prefix=%t.short 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-SIZE

// LAUNCH: Launch: yes
// LAUNCH: Verify: no
// LAUNCH: Running on device...
// VERIFY: Launch: yes
// VERIFY: Running on host...
// VERIFY: Running on device...
// VERIFY: Verification passed!
// RAW: Launch: yes
// RAW: Reference graph built successfully
// RAW-NOT: Running on host
// RAW: Running on device
// BAD-SIZE: Input file size does not match tensor allocation

module {
  nv_tensor_ir.graph @compiler_runtime_options(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %result = add %a, %b : tensor<8x8xf32>
    results %result : tensor<8x8xf32>
  }
}
