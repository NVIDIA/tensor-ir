// RUN: not tensor_ir-tblgen %s --gen-component-options \
// RUN:   -I %S/../../../include %mlir_include_dirs 2>&1 \
// RUN:   | FileCheck %s

include "tensor_ir/Options/OptionsBase.td"

// CHECK-DAG: error: component 'First' occurs more than once in CompilerInvocationOptions
def FirstOptions : ComponentOptions<"First"> { let invocationMember = "first"; }
// CHECK-DAG: error: invocation member 'first' is used by both components 'First' and 'Other'
def OtherOptions : ComponentOptions<"Other"> { let invocationMember = "first"; }
// CHECK-DAG: error: component 'Missing' has an invocationMember but is not listed in CompilerInvocationOptions
def MissingOptions : ComponentOptions<"Missing"> {
  let invocationMember = "missing";
}
// CHECK-DAG: error: component 'Tool' is listed in CompilerInvocationOptions but has no invocationMember
def ToolOptions : ComponentOptions<"Tool">;
def CompilerInvocation
    : CompilerInvocationOptions<
          [FirstOptions, OtherOptions, FirstOptions, ToolOptions]>;
