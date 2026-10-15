// REQUIRES: cuda-tile-translate
// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(layout-propagation-pipeline{tile-size=64})' \
// RUN:   --mlir-print-debuginfo | FileCheck %s
// RUN: tensor_ir-compiler %s --tile-size=64 --codegen-strategy=layout-propagation \
// RUN:   --dump-tileir-bc=%t.layout.bc
// RUN: cuda-tile-translate --cudatilebc-to-mlir --mlir-print-debuginfo %t.layout.bc \
// RUN:   | FileCheck %s
// RUN: tensor_ir-compiler %s --tile-size=64 --codegen-strategy=affine-map \
// RUN:   --dump-tileir-bc=%t.affine.bc
// RUN: cuda-tile-translate --cudatilebc-to-mlir --mlir-print-debuginfo %t.affine.bc \
// RUN:   | FileCheck %s

// Check both the lowered module and serialized bytecode: bare file locations
// become unknown in TileIR bytecode unless the pipeline synthesizes scopes.
module {
  nv_tensor_ir.graph @source_add(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %result = add %a, %b : tensor<8x8xf32> loc("source.py":12:7)
    results %result : tensor<8x8xf32> loc("source.py":13:3)
  } loc("source.py":10:1)
} loc("source.py":1:1)

// CHECK: cuda_tile.module
// CHECK: entry @{{[^ (]+}}
// CHECK: addf {{.*}} loc(#[[ADD_LOC:[a-zA-Z0-9_]+]])
// CHECK-DAG: #[[FILE:[a-zA-Z0-9_]+]] = #cuda_tile.di_file<"source.py" in "">
// CHECK-DAG: #[[SOURCE_LOC:[a-zA-Z0-9_]+]] = loc("source.py":12:7)
// CHECK: #[[CU:[a-zA-Z0-9_]+]] = #cuda_tile.di_compile_unit<file = #[[FILE]]>
// CHECK: #[[SP:[a-zA-Z0-9_]+]] = #cuda_tile.di_subprogram<file = #[[FILE]], line = 10, {{.*}}compileUnit = #[[CU]], scopeLine = 10>
// CHECK: #[[ADD_LOC]] = #cuda_tile.di_loc<#[[SOURCE_LOC]] in #[[SP]]>
