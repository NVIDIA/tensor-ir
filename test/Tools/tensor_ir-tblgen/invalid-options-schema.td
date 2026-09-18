// RUN: not tensor_ir-tblgen %s --gen-component-options \
// RUN:   -I %S/../../../include %mlir_include_dirs 2>&1 \
// RUN:   | FileCheck %s

include "tensor_ir/Options/OptionsBase.td"

def FirstOptions : ComponentOptions<"First"> {
  let invocationMember = "first";
  let options = [
    Option<"same", "shared", "unsigned", "1", "First shared option">,
    // CHECK-DAG: error: option field 'same' occurs more than once in component 'First'
    Option<"same", "duplicate-field", "unsigned", "2", "Duplicate field">,
    // CHECK-DAG: error: delimiter ':' of list option 'values' must be ',' or numeric-list delimiter 'x'
    ListOption<"values", "values", "int32_t", "Values", ":">,
    // CHECK-DAG: error: list option 'names' requires a numeric element type; elements are 'std::string'
    ListOption<"names", "names", "std::string", "Names">,
    // CHECK-DAG: error: option 'label' uses generic std::string C API conversions; use StringOption instead
    Option<"label", "label", "std::string", "\"\"", "Label">
  ];
}
// CHECK-DAG: error: component name 'First' is used by both 'DuplicateFirstOptions' and 'FirstOptions'
def DuplicateFirstOptions : ComponentOptions<"First">;
def SecondOptions : ComponentOptions<"Second"> {
  let invocationMember = "second";
  let options = [
    // CHECK-DAG: error: CLI flag '--shared' declared by both component 'First' and 'Second'
    Option<"same", "shared", "unsigned", "2", "Second shared option">
  ];
}

def CompilerInvocation
    : CompilerInvocationOptions<[FirstOptions, SecondOptions]>;
