// REQUIRES: tensor-ir-tileiras
// RUN: tensor_ir-compiler %s --tileiras-path="%tensor_ir_tileiras" --tile-size=64x64 --remarks=memory --remarks-output-file=%t.yaml
// RUN: FileCheck %s --check-prefix=YAML --input-file=%t.yaml
// RUN: tensor_ir-compiler %s --tileiras-path="%tensor_ir_tileiras" --tile-size=64x64 --remarks=all --remarks-passed=inherit --remarks-analysis=none --remarks-missed=none --remarks-failed=none --remarks-output-file=- | FileCheck %s --check-prefix=PASSED

// YAML-DAG: --- !Passed
// YAML-DAG: --- !Failure
// YAML-DAG: Pass:{{ *}}Memory
// YAML-DAG: Name:{{ *}}RemarkMemoryLoadInstructionSelected
// YAML-DAG: Name:{{ *}}RemarkMemoryTmaLoadFailed
// YAML-DAG: DebugLoc:{{ *}}{ File: {{'?}}source.py{{'?}}, Line: 10, Column: 1 }
// YAML-DAG: Function:{{ *}}source_add
// YAML-DAG: Instruction:
// PASSED-NOT: --- !Missed
// PASSED-NOT: --- !Failure
// PASSED-NOT: --- !Analysis
// PASSED: --- !Passed
// PASSED: Name:{{ *}}RemarkMemoryLoadInstructionSelected
// PASSED-NOT: --- !Missed
// PASSED-NOT: --- !Failure
// PASSED-NOT: --- !Analysis

// An eight-byte-aligned matrix input forces TMA selection to report its
// alignment failure while ordinary global-memory loads remain available.
module {
  nv_tensor_ir.graph @source_add(
    %a: tensor<128x64xf16> {nv_tensor_ir.stride = "(64,1)", nv_tensor_ir.alignment = 8 : i64},
    %b: tensor<64x128xf16> {nv_tensor_ir.stride = "(128,1)"}
  ) -> (tensor<128x128xf32> {nv_tensor_ir.stride = "(128,1)"}) {
    %c = "nv_tensor_ir.matmul"(%a, %b) : (tensor<128x64xf16>, tensor<64x128xf16>) -> tensor<128x128xf32> loc("source.py":8:3)
    results %c : tensor<128x128xf32>
  } loc("source.py":10:1)
}
