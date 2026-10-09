// RUN: tensor_ir-compiler 2>&1 | FileCheck %s
// RUN: tensor_ir-compiler --help 2>&1 | FileCheck %s

// CHECK-LABEL: OVERVIEW: TensorIR Compiler and Test Tool
// CHECK-DAG: --cta-count
// CHECK-DAG: --input-prefix
// CHECK-DAG: --iterations
// CHECK-DAG: --launch
// CHECK-DAG: --occupancy
// CHECK-DAG: --output-file
// CHECK-DAG: --persistence-mode
// CHECK-DAG: --reproducer-dir
// CHECK-DAG: --repeat
// CHECK-DAG: --seed
// CHECK-DAG: --sm-count
// CHECK-DAG: --tolerance
// CHECK-DAG: --verify
// CHECK-DAG: --warp-count
// CHECK-NOT: TENSOR_IR_
// CHECK-NOT: --enable-gvn-hoist
