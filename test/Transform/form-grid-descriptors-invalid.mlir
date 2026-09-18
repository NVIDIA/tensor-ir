// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --split-input-file --verify-diagnostics

func.func @bad_load(%input: !ptr.ptr<#ptr.generic_space>,
    %output: !ptr.ptr<#ptr.generic_space>, %n: index, %m: index,
    %stride: index) attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(1,?)", [1, 0, 2]>,
  result_layouts = [#nv_tensor_ir.tensor_source<0, 5, "(?,?):(1,?)", [1, 0, 2]>],
  result_views = [#nv_tensor_ir.tensor_source<1, 2, "(?,?):(1,?)", [1, 0, 2]>],
  tile_size = array<i32: 4, 8>
} {
  %zero = arith.constant 0 : index
  // expected-error @below {{layout dynamic-value mapping index 2 is out of range for tensor #0 with 2 dynamic descriptor value(s)}}
  %load = nv_tensor_ir.load %input[%zero, %zero]
      view offset: [3], sizes: [%n, %m], strides: [32, 1], alignment: 16
      : !ptr.ptr<#ptr.generic_space> to tensor<?x?xf32>
  nv_tensor_ir.store %load, %output[%zero, %zero]
      view offset: [7], sizes: [%n, %m], strides: [%stride, 1], alignment: 16
      : tensor<?x?xf32>, !ptr.ptr<#ptr.generic_space>
  return
}

// -----

func.func @bad_store(%input: !ptr.ptr<#ptr.generic_space>,
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
  // expected-error @below {{layout dynamic-value mapping index 2 is out of range for tensor #1 with 2 dynamic descriptor value(s)}}
  nv_tensor_ir.store %load, %output[%zero, %zero]
      view offset: [7], sizes: [%n, %m], strides: [32, 1], alignment: 16
      : tensor<?x?xf32>, !ptr.ptr<#ptr.generic_space>
  return
}

// -----

func.func @offset_overflow(%input: !ptr.ptr<#ptr.generic_space>,
    %output: !ptr.ptr<#ptr.generic_space>) attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">,
  result_layouts = [#nv_tensor_ir.tensor_source<0, 1, "(16):(1)">],
  result_views = [#nv_tensor_ir.tensor_source<1, 0, "(16):(1)">],
  tile_size = array<i32: 4>
} {
  // expected-error @below {{logical buffer offset overflows index range}}
  %load = nv_tensor_ir.load %input[0]
      view offset: [9223372036854775807], sizes: [16], strides: [1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<16xf32>
  nv_tensor_ir.store %load, %output[0]
      view offset: [0], sizes: [16], strides: [1], alignment: 4
      : tensor<16xf32>, !ptr.ptr<#ptr.generic_space>
  return
}
