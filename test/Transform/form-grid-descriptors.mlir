// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --mlir-print-debuginfo --split-input-file | FileCheck %s

// Sizes and strides are permuted together; physical and logical offsets add.
// The output store retains its source location.
// CHECK-LABEL: func.func @descriptor(
//  CHECK-SAME: %[[INPUT:[^:]+]]: !ptr.ptr
//  CHECK-SAME: %[[OUTPUT:[^:]+]]: !ptr.ptr
//  CHECK-SAME: %[[N:[^:]+]]: index
//  CHECK-SAME: %[[M:[^:]+]]: index
//  CHECK-SAME: %[[STRIDE:[^:]+]]: index
//       CHECK: arith.addi %[[M]],
//       CHECK: arith.addi %[[N]],
//       CHECK: scf.forall
//       CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[INPUT]]
//  CHECK-SAME: view offset: [8], sizes: [%[[M]], %[[N]]], strides: [1, %[[STRIDE]]], alignment: 4
//       CHECK: nv_tensor_ir.store %[[TILE]], %[[OUTPUT]]
//  CHECK-SAME: view offset: [9], sizes: [%[[M]], %[[N]]], strides: [1, %[[STRIDE]]], alignment: 8
//  CHECK-SAME: loc([[OUTPUT_LOC:#[a-zA-Z0-9]+]])
//       CHECK: [[OUTPUT_LOC]] = loc("output")
func.func @descriptor(%input: !ptr.ptr<#ptr.generic_space>,
    %output: !ptr.ptr<#ptr.generic_space>, %n: index, %m: index,
    %stride: index) attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(1,?)", [1, 0, 2]>,
  result_layouts = [#nv_tensor_ir.tensor_source<0, 5, "(?,?):(1,?)", [1, 0, 2]>],
  result_views = [#nv_tensor_ir.tensor_source<1, 2, "(?,?):(1,?)", [1, 0, 2]>],
  tile_size = array<i32: 4, 8>
} {
  %zero = arith.constant 0 : index
  %load = nv_tensor_ir.load %input[%zero, %zero]
      view offset: [3], sizes: [%n, %m], strides: [%stride, 1], alignment: 16
      : !ptr.ptr<#ptr.generic_space> to tensor<?x?xf32>
  nv_tensor_ir.store %load, %output[%zero, %zero]
      view offset: [7], sizes: [%n, %m], strides: [%stride, 1], alignment: 16
      : tensor<?x?xf32>, !ptr.ptr<#ptr.generic_space> loc("output")
  return
}

// -----

// Dynamic physical offsets remain SSA values when logical offsets are added.
// CHECK-LABEL: func.func @dynamic_offsets(
//  CHECK-SAME: %[[INPUT:[^:]+]]: !ptr.ptr
//  CHECK-SAME: %[[OUTPUT:[^:]+]]: !ptr.ptr
//  CHECK-SAME: %[[IN_OFFSET:[^:]+]]: index
//  CHECK-SAME: %[[OUT_OFFSET:[^:]+]]: index
//       CHECK: scf.forall (%[[I:.*]]) in (4)
//       CHECK: %[[FIVE:.*]] = arith.constant 5 : index
//       CHECK: %[[LOAD_OFFSET:.*]] = arith.addi %[[IN_OFFSET]], %[[FIVE]]
//       CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[INPUT]][%[[I]]]
//  CHECK-SAME: view offset: [%[[LOAD_OFFSET]]], sizes: [16], strides: [1], alignment: 4
//       CHECK: %[[TWO:.*]] = arith.constant 2 : index
//       CHECK: %[[STORE_OFFSET:.*]] = arith.addi %[[OUT_OFFSET]], %[[TWO]]
//       CHECK: nv_tensor_ir.store %[[TILE]], %[[OUTPUT]][%[[I]]]
//  CHECK-SAME: view offset: [%[[STORE_OFFSET]]], sizes: [16], strides: [1], alignment: 8
func.func @dynamic_offsets(%input: !ptr.ptr<#ptr.generic_space>,
    %output: !ptr.ptr<#ptr.generic_space>, %input_offset: index,
    %output_offset: index) attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">,
  result_layouts = [#nv_tensor_ir.tensor_source<0, 5, "(16):(1)">],
  result_views = [#nv_tensor_ir.tensor_source<1, 2, "(16):(1)">],
  tile_size = array<i32: 4>
} {
  %load = nv_tensor_ir.load %input[0]
      view offset: [%input_offset], sizes: [16], strides: [1], alignment: 16
      : !ptr.ptr<#ptr.generic_space> to tensor<16xf32>
  nv_tensor_ir.store %load, %output[0]
      view offset: [%output_offset], sizes: [16], strides: [1], alignment: 16
      : tensor<16xf32>, !ptr.ptr<#ptr.generic_space>
  return
}

// -----

// A zero logical offset preserves each dynamic physical offset directly.
// CHECK-LABEL: func.func @unchanged_dynamic_offsets(
//  CHECK-SAME: %[[INPUT:[^:]+]]: !ptr.ptr
//  CHECK-SAME: %[[OUTPUT:[^:]+]]: !ptr.ptr
//  CHECK-SAME: %[[OFFSET:[^:]+]]: index
//       CHECK: scf.forall
//   CHECK-NOT: arith.addi
//       CHECK: %[[TILE:.*]] = nv_tensor_ir.load %[[INPUT]]
//  CHECK-SAME: view offset: [%[[OFFSET]]], sizes: [16], strides: [1], alignment: 16
//   CHECK-NOT: arith.addi
//       CHECK: nv_tensor_ir.store %[[TILE]], %[[OUTPUT]]
//  CHECK-SAME: view offset: [%[[OFFSET]]], sizes: [16], strides: [1], alignment: 16
func.func @unchanged_dynamic_offsets(%input: !ptr.ptr<#ptr.generic_space>,
    %output: !ptr.ptr<#ptr.generic_space>, %offset: index) attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">,
  result_layouts = [#nv_tensor_ir.tensor_source<0, 0, "(16):(1)">],
  result_views = [#nv_tensor_ir.tensor_source<1, 0, "(16):(1)">],
  tile_size = array<i32: 4>
} {
  %load = nv_tensor_ir.load %input[0]
      view offset: [%offset], sizes: [16], strides: [1], alignment: 16
      : !ptr.ptr<#ptr.generic_space> to tensor<16xf32>
  nv_tensor_ir.store %load, %output[0]
      view offset: [%offset], sizes: [16], strides: [1], alignment: 16
      : tensor<16xf32>, !ptr.ptr<#ptr.generic_space>
  return
}
