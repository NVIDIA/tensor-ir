// RUN: not tensor_ir-tblgen %s --gen-component-options \
// RUN:   -I %S/../../../include %mlir_include_dirs 2>&1 \
// RUN:   | FileCheck %s

include "tensor_ir/Options/OptionsBase.td"
def FirstOptions : ComponentOptions<"First"> { let invocationMember = "first"; }
// CHECK: error: expected exactly one CompilerInvocationOptions definition
