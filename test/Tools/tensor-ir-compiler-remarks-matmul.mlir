// REQUIRES: tensor-ir-tileiras
// RUN: tensor_ir-compiler %s --tileiras-path="%tensor_ir_tileiras" --tile-size=64x64 --remarks=tensorcore --remarks-output-file=%t.yaml
// RUN: FileCheck %s --input-file=%t.yaml
// CHECK: --- !Passed
// CHECK: Pass:{{ *}}Tensor-core
// CHECK: Name:{{ *}}RemarkTensorCoreMMA
// CHECK: DebugLoc:{{ *}}{ File: {{'?}}gemm.py{{'?}}, Line: 8, Column: 3 }
// CHECK: Function:{{ *}}matmul
// CHECK: Instruction:{{ *}}Tensor-core
// CHECK: Shape:
// CHECK: NumCTAs:

module {
  nv_tensor_ir.graph @matmul(
    %a: tensor<128x64xf16> {nv_tensor_ir.stride = "(64,1)"},
    %b: tensor<64x128xf16> {nv_tensor_ir.stride = "(128,1)"}
  ) -> (tensor<128x128xf32> {nv_tensor_ir.stride = "(128,1)"}) {
    %c = "nv_tensor_ir.matmul"(%a, %b) : (tensor<128x64xf16>, tensor<64x128xf16>) -> tensor<128x128xf32> loc("gemm.py":8:3)
    results %c : tensor<128x128xf32>
  } loc("gemm.py":5:1)
}
