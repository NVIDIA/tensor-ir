// REQUIRES: tensor-ir-tileiras
// RUN: tensor_ir-compiler %s --tileiras-path="%tensor_ir_tileiras" --tile-size=8x8 --remarks=memory --remarks-passed=none --remarks-failed=none --remarks-missed=tensorcore --remarks-output-file=%t.filtered.yaml
// RUN: test ! -s %t.filtered.yaml

module {
  nv_tensor_ir.graph @source_add(
    %a: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"},
    %b: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"}
  ) -> (tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"}) {
    %result = add %a, %b : tensor<?x?xf32> loc("source.py":12:7)
    results %result : tensor<?x?xf32> loc("source.py":13:3)
  } loc("source.py":10:1)
} loc("source.py":1:1)
