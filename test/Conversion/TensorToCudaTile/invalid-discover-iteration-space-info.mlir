// RUN: tensor_ir-opt -discover-iteration-space-info -verify-diagnostics %s
// RUN: tensor_ir-opt --discover-iteration-space-info --convert-tensor-to-cuda-tile="codegen-strategy=affine_map" --verify-diagnostics %s

// This file is to test that there are no crashes when the discover-iteration-space-info pass
// encounters a pattern that it does not support.

nv_tensor_ir.graph @graph_mQs5T7LOX0xWPzD4(
    %in: tensor<66x64x29xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(1856,1,64)"},
    %in_0: tensor<66x64x29xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(1856,1,64)"},
    %in_1: tensor<66x64x29xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(1856,1,64)"})
    -> (tensor<28x2x4xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(8,4,1)"}) {
  // expected-error@below {{concatenate is not supported by DiscoverIterationSpaceInfoPass}}
  %0 = concatenate %in, %in_0, %in_1 dimension = 2 : (tensor<66x64x29xf32>, tensor<66x64x29xf32>, tensor<66x64x29xf32>) -> tensor<66x64x87xf32>
  %1 = slice %0 starts = [60, 23, 30] limits = [66, 39, 78] strides = [1, 1, 1] : tensor<66x64x87xf32> -> tensor<6x16x48xf32>
  %2 = transpose %1 permutation = [0, 2, 1] : tensor<6x16x48xf32> -> tensor<6x48x16xf32>
  %3 = slice %2 starts = [1, 33, 12] limits = [5, 47, 16] strides = [1, 1, 1] : tensor<6x48x16xf32> -> tensor<4x14x4xf32>
  %4 = reshape %3 : tensor<4x14x4xf32> -> tensor<28x2x4xf32>
  %add = add %4, %4 : tensor<28x2x4xf32>
  results %add : tensor<28x2x4xf32>
}

// -----

nv_tensor_ir.graph @graph_5qAnzNy2GIWrf9ck(
    %in: tensor<64x23x6x35xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(4830,210,35,1)"},
    %in_0: tensor<805x128x3xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(384,3,1)"},
    %in_1: tensor<805x128x3xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(384,3,1)"},
    %in_2: tensor<805x384x3xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(1152,1,384)"})
    -> (tensor<133x140x2xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(280,2,1)"}) {
  %0 = reshape %in : tensor<64x23x6x35xf32> -> tensor<128x805x3xf32>
  %1 = transpose %0 permutation = [1, 0, 2] : tensor<128x805x3xf32> -> tensor<805x128x3xf32>
  // expected-error@below {{concatenate is not supported by DiscoverIterationSpaceInfoPass}}
  %2 = concatenate %1, %in_0, %in_1 dimension = 1 : (tensor<805x128x3xf32>, tensor<805x128x3xf32>, tensor<805x128x3xf32>) -> tensor<805x384x3xf32>
  %3 = concatenate %2, %in_2 dimension = 2 : (tensor<805x384x3xf32>, tensor<805x384x3xf32>) -> tensor<805x384x6xf32>
  %4 = slice %3 starts = [527, 144, 4] limits = [660, 284, 6] strides = [1, 1, 1] : tensor<805x384x6xf32> -> tensor<133x140x2xf32>
  %add = add %4, %4 : tensor<133x140x2xf32>
  results %add : tensor<133x140x2xf32>
}

// -----

nv_tensor_ir.graph @graph_HWT62KeAFqvzm9Qs(
    %in: tensor<73x5xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(5,1)"},
    %in_0: tensor<73x5xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(1,73)"},
    %in_1: tensor<73x5xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(1,73)"})
    -> (tensor<7x2xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(2,1)"}) {
  // expected-error@below {{concatenate is not supported by DiscoverIterationSpaceInfoPass}}
  %0 = concatenate %in, %in_0, %in_1 dimension = 1 : (tensor<73x5xf32>, tensor<73x5xf32>, tensor<73x5xf32>) -> tensor<73x15xf32>
  %1 = reshape %0 : tensor<73x15xf32> -> tensor<73x15xf32>
  %2 = transpose %1 permutation = [1, 0] : tensor<73x15xf32> -> tensor<15x73xf32>
  %3 = slice %2 starts = [12, 51] limits = [14, 58] strides = [1, 1] : tensor<15x73xf32> -> tensor<2x7xf32>
  %4 = transpose %3 permutation = [1, 0] : tensor<2x7xf32> -> tensor<7x2xf32>
  %add = add %4, %4 : tensor<7x2xf32>
  results %add : tensor<7x2xf32>
}

// -----

nv_tensor_ir.graph @graph_8KLUTIlHSbzdOYfE(
    %in: tensor<64x8xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(1,64)"},
    %in_0: tensor<2x6xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(6,1)"})
    -> (tensor<2x12xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(1,2)"}) {
  %0 = slice %in starts = [25, 5] limits = [31, 7] strides = [1, 1] : tensor<64x8xf32> -> tensor<6x2xf32>
  %1 = transpose %0 permutation = [1, 0] : tensor<6x2xf32> -> tensor<2x6xf32>
  // expected-error@below {{concatenate is not supported by DiscoverIterationSpaceInfoPass}}
  %2 = concatenate %1, %in_0 dimension = 1 : (tensor<2x6xf32>, tensor<2x6xf32>) -> tensor<2x12xf32>
  %add = add %2, %2 : tensor<2x12xf32>
  results %add : tensor<2x12xf32>
}
