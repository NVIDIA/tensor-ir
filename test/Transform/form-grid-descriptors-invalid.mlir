// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(func.func(tir-form-grid))' --split-input-file --verify-diagnostics

func.func @bad_load(
    %input: memref<?x?xf32, strided<[32, 1], offset: 3>, #ptr.generic_space>,
    %output: memref<?x?xf32, strided<[?, 1], offset: 7>, #ptr.generic_space>)
    attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(1,?)", [1, 0, 2]>,
  result_layouts = [#nv_tensor_ir.tensor_source<0, 5, "(?,?):(1,?)", [1, 0, 2]>],
  result_views = [#nv_tensor_ir.tensor_source<1, 2, "(?,?):(1,?)", [1, 0, 2]>],
  tile_size = array<i32: 4, 8>
} {
  // expected-error @below {{layout dynamic-value mapping index 2 is out of range for tensor #0 with 2 dynamic descriptor value(s)}}
  %load = bufferization.to_tensor %input restrict
      : memref<?x?xf32, strided<[32, 1], offset: 3>, #ptr.generic_space>
        to tensor<?x?xf32>
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<?x?xf32>,
         memref<?x?xf32, strided<[?, 1], offset: 7>, #ptr.generic_space>) -> ()
  return
}

// -----

func.func @bad_store(
    %input: memref<?x?xf32, strided<[?, 1], offset: 3>, #ptr.generic_space>,
    %output: memref<?x?xf32, strided<[32, 1], offset: 7>, #ptr.generic_space>)
    attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(1,?)", [1, 0, 2]>,
  result_layouts = [#nv_tensor_ir.tensor_source<0, 5, "(?,?):(1,?)", [1, 0, 2]>],
  result_views = [#nv_tensor_ir.tensor_source<1, 2, "(?,?):(1,?)", [1, 0, 2]>],
  tile_size = array<i32: 4, 8>
} {
  %load = bufferization.to_tensor %input restrict
      : memref<?x?xf32, strided<[?, 1], offset: 3>, #ptr.generic_space>
        to tensor<?x?xf32>
  // expected-error @below {{layout dynamic-value mapping index 2 is out of range for tensor #1 with 2 dynamic descriptor value(s)}}
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<?x?xf32>,
         memref<?x?xf32, strided<[32, 1], offset: 7>, #ptr.generic_space>) -> ()
  return
}

// -----

func.func @offset_overflow(
    %input: memref<16xf32, strided<[1], offset: 9223372036854775807>,
                   #ptr.generic_space>,
    %output: memref<16xf32, #ptr.generic_space>) attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">,
  result_layouts = [#nv_tensor_ir.tensor_source<0, 1, "(16):(1)">],
  result_views = [#nv_tensor_ir.tensor_source<1, 0, "(16):(1)">],
  tile_size = array<i32: 4>
} {
  // expected-error @below {{logical buffer offset overflows index range}}
  %load = bufferization.to_tensor %input restrict
      : memref<16xf32, strided<[1], offset: 9223372036854775807>,
               #ptr.generic_space> to tensor<16xf32>
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<16xf32>, memref<16xf32, #ptr.generic_space>) -> ()
  return
}

// -----

func.func @non_argument_boundary(%output: memref<16xf32, #ptr.generic_space>)
    attributes {
  nv_tensor_ir.bufferized_program,
  iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(16):(1)">,
  tile_size = array<i32: 4>
} {
  %buffer = memref.alloca() : memref<16xf32, #ptr.generic_space>
  // expected-error @below {{bufferized TensorIR boundary must be a memref function argument}}
  %load = bufferization.to_tensor %buffer restrict
      : memref<16xf32, #ptr.generic_space> to tensor<16xf32>
  bufferization.materialize_in_destination %load in restrict writable %output
      : (tensor<16xf32>, memref<16xf32, #ptr.generic_space>) -> ()
  return
}
