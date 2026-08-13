// RUN: tensor_ir-compiler 2>&1 | FileCheck %s
// RUN: tensor_ir-compiler --help 2>&1 | FileCheck %s

// CHECK-LABEL: OVERVIEW: TensorIR Compiler and Test Tool
// CHECK-DAG: --iterations
// CHECK-DAG: --launch
// CHECK-DAG: --seed
// CHECK-DAG: --tolerance
// CHECK-DAG: --uniform-signature
// CHECK-DAG: --verify
// CHECK-DAG: TENSOR_IR_DUMP_IR
// CHECK-DAG: TENSOR_IR_DUMP_TILEIR_BC
// CHECK-DAG: TENSOR_IR_LOAD_TILEIR_BC
// CHECK-DAG: TENSOR_IR_PRINT_IR_AFTER_ALL
// CHECK-DAG: TENSOR_IR_PRINT_IR_TREE_DIR
// CHECK-DAG: TENSOR_IR_TIMING
// CHECK-DAG: 1/0, true/false, on/off, or yes/no
// CHECK-DAG: 0/false/off/no disables
// CHECK-DAG: default: false
// CHECK-NOT: --enable-gvn-hoist
