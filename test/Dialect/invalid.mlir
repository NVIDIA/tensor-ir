// RUN: tensor_ir-opt %s --verify-diagnostics --split-input-file


nv_tensor_ir.graph @unsupportedTensorEncoding() {
  // expected-error@below{{must be TensorIR-compatible ranked tensor}}
  %result = iota dimension = 0 : tensor<4xf32, "unsupported">
  results
}

// -----

// expected-error@below{{expects argument #0 to be a TensorIR ranked tensor, but got 'tensor<10xi32>'}}
nv_tensor_ir.graph @someGraph(%arg0: tensor<10xi32>) {
  results
}

// -----
// expected-error@below{{expects argument #0 to be a TensorIR ranked tensor, but got 'tensor<10xsi9>'}}
nv_tensor_ir.graph @someGraph(%arg0: tensor<10xsi9>) {
  results
}

// -----
// expected-error@below{{expects argument #0 to be a TensorIR ranked tensor, but got 'tensor<10xui12>'}}
nv_tensor_ir.graph @someGraph(%arg0: tensor<10xui12>) {
  results
}

// -----
// expected-error@below{{expected non-function type}}
nv_tensor_ir.graph @someGraph(%arg0: tensor<(2,2)x4xf32>) {
  results
}

// -----

nv_tensor_ir.graph @reductionTestGraphIncompatibleInOut(
  %input: tensor<16x?x64xf32>
) -> tensor<16x64xf32> {
  // expected-error@below{{'nv_tensor_ir.reduce' op failed to verify that all of {input, output} have same rank}}
  %reduce = nv_tensor_ir.reduce(%input)<
      dimensions = [1],
      reduction_mode = <add> >
       : tensor<16x?x64xf32> -> tensor<16x64xf32>
  results %reduce : tensor<16x64xf32>
}

// -----
nv_tensor_ir.graph @reduceUDIdentityCount(
  %input: tensor<16x32xf32>
) -> tensor<16x1xf32> {
  // expected-error@below{{expects number of identities to match number of results, but got 0 identities and 1 results}}
  %reduce = reduce_ud(%input)<dimensions = [1], identity = []>
      (%acc: f32, %value: f32) {
    %sum = arith.addf %acc, %value : f32
    nv_tensor_ir.yield %sum : f32
  } : tensor<16x32xf32> -> tensor<16x1xf32>
  results %reduce : tensor<16x1xf32>
}

// -----
nv_tensor_ir.graph @reduceUDBlockArgumentCount(
  %input: tensor<16x32xf32>
) -> tensor<16x1xf32> {
  // expected-error@below{{expects body region to have exactly 2 arguments (prev_result_i, curr_operand_i), but got 1 arguments}}
  %reduce = reduce_ud(%input)<dimensions = [1], identity = [0.000000e+00 : f32]>
      (%acc: f32) {
    nv_tensor_ir.yield %acc : f32
  } : tensor<16x32xf32> -> tensor<16x1xf32>
  results %reduce : tensor<16x1xf32>
}

// -----
nv_tensor_ir.graph @reduceUDYieldType(
  %input: tensor<16x32xf32>
) -> tensor<16x1xf32> {
  // expected-error@below{{expects yield operand 0 type to match identity type, but got yield type: 'i32' and identity type: 'f32'}}
  %reduce = reduce_ud(%input)<dimensions = [1], identity = [0.000000e+00 : f32]>
      (%acc: f32, %value: f32) {
    %zero = arith.constant 0 : i32
    nv_tensor_ir.yield %zero : i32
  } : tensor<16x32xf32> -> tensor<16x1xf32>
  results %reduce : tensor<16x1xf32>
}

// -----
nv_tensor_ir.graph @binarySelectTestGraph(
  // expected-note@below{{prior use here}}
  %select: tensor<10xf32>,
  %lhs: tensor<10xf32>,
  %rhs: tensor<10xf32>
) -> tensor<10xf32> {

  // expected-error@below{{use of value '%select' expects different type than prior uses: 'tensor<10xi1>' vs 'tensor<10xf32>'}}
  %cmp1 = binary_select %select, %lhs, %rhs : tensor<10xf32>

  results %cmp1: tensor<10xf32>
}

// -----
nv_tensor_ir.graph @binaryOpTestGraph(
  // expected-note@below{{prior use here}}
  %lhs: tensor<10xf32>,
  %rhs: tensor<10xf32>
) -> tensor<10xf16> {
  // expected-error @below {{use of value '%lhs' expects different type than prior uses: 'tensor<10xf16>' vs 'tensor<10xf32>'}}
  %res = add %lhs, %rhs : tensor<10xf16>
  results %res: tensor<10xf16>
}

// -----
nv_tensor_ir.graph @broadcastTestGraph1(
  // expected-note@below{{prior use here}}
  %lhs: tensor<1x?xf32> {nv_tensor_ir.stride = "(?,1)"},
  %rhs: tensor<3x?x?xf32> {nv_tensor_ir.stride = "(?,?,1)"}
) -> (tensor<?x?x?xf32> {nv_tensor_ir.stride = "(?,?,1)"}) {
  // expected-error @below {{use of value '%lhs' expects different type than prior uses: 'tensor<?x?x?xf32>' vs 'tensor<1x?xf32>'}}
  %res = add %lhs, %rhs : tensor<?x?x?xf32>
  results %res: tensor<?x?x?xf32>
}

// -----
nv_tensor_ir.graph @broadcastTestGraph2(
  %lhs: tensor<2x?x?xf32> {nv_tensor_ir.stride = "(?,?,1)"},
  // expected-note@below{{prior use here}}
  %rhs: tensor<3x?x?xf32> {nv_tensor_ir.stride = "(?,?,1)"}
) -> (tensor<?x?x?xf32> {nv_tensor_ir.stride = "(?,?,1)"}) {
  // expected-error @below {{use of value '%rhs' expects different type than prior uses: 'tensor<2x?x?xf32>' vs 'tensor<3x?x?xf32>'}}
  %res = add %lhs, %rhs : tensor<2x?x?xf32>
  results %res: tensor<?x?x?xf32>
}

// -----
// expected-error @below {{'nv_tensor_ir.graph' op expects the # of ResultTypes should match the # of return value type, but got 1 and 0}}
nv_tensor_ir.graph @graphOpVerifyGraph(
  %lhs: tensor<?x?x?xf32> {nv_tensor_ir.stride = "(?,?,1)"},
  %rhs: tensor<?x?x?xf32> {nv_tensor_ir.stride = "(?,?,1)"}
) -> (tensor<?x?x?xf16> {nv_tensor_ir.stride = "(?,?,1)"}) {
  %res = add %lhs, %rhs : tensor<?x?x?xf32>
  results
}

// -----
// expected-error @below {{'nv_tensor_ir.graph' op expects ResultTypes[0] should match return value type, but got 'tensor<?x?x?xf16>' and 'tensor<?x?x?xf32>'}}
nv_tensor_ir.graph @graphOpVerifyGraph(
  %lhs: tensor<?x?x?xf32> {nv_tensor_ir.stride = "(?,?,1)"},
  %rhs: tensor<?x?x?xf32> {nv_tensor_ir.stride = "(?,?,1)"}
) -> (tensor<?x?x?xf16> {nv_tensor_ir.stride = "(?,?,1)"}) {
  %res = add %lhs, %rhs : tensor<?x?x?xf32>
  results %res: tensor<?x?x?xf32>
}

// -----
nv_tensor_ir.graph @graphOpVerifyGraph(
// expected-error @below {{duplicate key 'nv_tensor_ir.alignment' in dictionary attribute}}
  %lhs: tensor<?x?x?xf32> {nv_tensor_ir.alignment = 1, nv_tensor_ir.alignment = 4},
  %rhs: tensor<?x?x?xf32>
) -> tensor<?x?x?xf32> {
  %res = add %lhs, %rhs : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  results %res: tensor<?x?x?xf32>
}

// -----
// expected-error@below{{expects iter_space_map attribute to be applied to tensor type, but got 'f32'}}
nv_tensor_ir.graph @graphOpVerifyIterSpaceMapOnNonTensor(%arg0: f32 {nv_tensor_ir.iter_space_map = affine_map<(d0) -> (d0)>}) {
  results
}

// -----
// expected-error @below {{'nv_tensor_ir.graph' op expects iter_space_map is `AffineMapAttr`, but got "not_an_affine_map"}}
nv_tensor_ir.graph @graphOpVerifyIterSpaceMapNotAffineMap(
  %arg0: tensor<1x4xf32> {nv_tensor_ir.iter_space_map = "not_an_affine_map"}
) {
  results
}

// -----
// expected-error @below {{'nv_tensor_ir.graph' op does not support TensorIR signature attribute `nv_tensor_ir.strides`}}
nv_tensor_ir.graph @graphOpVerifyGraph(
  %lhs: tensor<?x?x?xf32> {nv_tensor_ir.strides = "(?,?,x)"},
  %rhs: tensor<?x?x?xf32>
) -> tensor<?x?x?xf32> {
  %res = add %lhs, %rhs : tensor<?x?x?xf32>
  results %res: tensor<?x?x?xf32>
}

// -----
// expected-error @below {{'nv_tensor_ir.graph' op expects alignment is `IntegerAttr`, but got "1"}}
nv_tensor_ir.graph @graphOpVerifyGraphAlign(
  %lhs: tensor<?x?x?xf32> {nv_tensor_ir.alignment = "1"},
  %rhs: tensor<?x?x?xf32>
) -> tensor<?x?x?xf32> {
  %res = add %lhs, %rhs : tensor<?x?x?xf32>
  results %res: tensor<?x?x?xf32>
}

// -----
// expected-error@below{{expects alignment >= 1, but got -1}}
nv_tensor_ir.graph @graphOpVerifyGraphAlign(%arg0: tensor<1x4xf32> {nv_tensor_ir.alignment = -1}) {
  results
}

// -----
// expected-error@below{{expects alignment attribute to be applied to tensor type, but got 'f32'}}
nv_tensor_ir.graph @graphOpVerifyGraphAlign(%arg0: f32 {nv_tensor_ir.alignment = -1}) {
  results
}

// -----
// expected-error@below{{expects stride attribute to be applied to tensor type, but got 'f32'}}
nv_tensor_ir.graph @graphOpVerifyGraphStride(%arg0: f32 {nv_tensor_ir.stride = "(0,1,1)"}) {
  results
}

// -----
// expected-error@below{{expects stride is `StringAttr`, but got 1 : i64}}
nv_tensor_ir.graph @graphOpVerifyGraphStrideType(
  %arg0: tensor<1x4xf32> {nv_tensor_ir.stride = 1 : i64}
) {
  results
}

// -----
// expected-error@below{{expects stride is valid for corresponding shape, but got "(0,1,1)"}}
nv_tensor_ir.graph @graphOpVerifyGraphStride(%arg0: tensor<1x4xf32> {nv_tensor_ir.alignment = 16, nv_tensor_ir.stride = "(0,1,1)"}) {
  results
}

// -----
// expected-error @below {{expects stride to be tcutegen::Stride, but got "(?,?,??)"}}
nv_tensor_ir.graph @graphOpVerifyGraphStride(
  %lhs: tensor<?x?x?xf32> {nv_tensor_ir.stride = "(?,?,??)"},
  %rhs: tensor<?x?x?xf32>
) -> tensor<?x?x?xf32> {
  %res = add %lhs, %rhs : tensor<?x?x?xf32>
  results %res: tensor<?x?x?xf32>
}

// -----
// expected-error @below {{expects stride to be tcutegen::Stride, but got "(?,?,x)"}}
nv_tensor_ir.graph @graphOpVerifyGraphStride(
  %lhs: tensor<?x?x?xf32> {nv_tensor_ir.stride = "(?,?,x)"},
  %rhs: tensor<?x?x?xf32>
) -> tensor<?x?x?xf32> {
  %res = add %lhs, %rhs : tensor<?x?x?xf32>
  results %res: tensor<?x?x?xf32>
}

// -----
// expected-error @below {{stride=0 is not allowed in graph input stride attribute}}
nv_tensor_ir.graph @graphOpVerifyStrideZeroRejected(
  %lhs: tensor<?x?x?xf32> {nv_tensor_ir.stride = "(0,?,1)"},
  %rhs: tensor<?x?x?xf32>
) -> tensor<?x?x?xf32> {
  %res = add %lhs, %rhs : tensor<?x?x?xf32>
  results %res: tensor<?x?x?xf32>
}

// -----
// Broadcast verifier: no-op broadcast (no dim changes from 1 to non-1).
nv_tensor_ir.graph @broadcastNoOp(
  %in: tensor<?x?x?xf16>
) -> (tensor<?x?x?xf16>){
  // expected-error @below {{no-op broadcast is not allowed}}
  %bc = broadcast %in : tensor<?x?x?xf16> -> tensor<?x?x?xf16>
  results %bc: tensor<?x?x?xf16>
}

// -----
nv_tensor_ir.graph @ConvertOp(
  %input: tensor<16x1x64xf32>
) -> (tensor<16x2x64xf64>){
  // expected-error @below {{'nv_tensor_ir.convert' op requires the same shape for all operands and results}}
  %1 = convert %input : tensor<16x1x64xf32> -> tensor<16x2x64xf64>
  results %1 : tensor<16x2x64xf64>
}

// -----
nv_tensor_ir.graph @ConvertOp(
  %input: tensor<16x1x64xf32>
) -> (tensor<16x64xf64>){
  // expected-error @below {{'nv_tensor_ir.convert' op requires the same shape for all operands and results}}
  %1 = convert %input : tensor<16x1x64xf32> -> tensor<16x64xf64>
  results %1 : tensor<16x64xf64>
}

// -----
nv_tensor_ir.graph @ConvertOp(
  %input: tensor<?x?x?xf32>
) -> (tensor<?x?xf64>){
  // expected-error @below {{'nv_tensor_ir.convert' op requires the same shape for all operands and results}}
  %1 = convert %input : tensor<?x?x?xf32> -> tensor<?x?xf64>
  results %1 : tensor<?x?xf64>
}

// -----
nv_tensor_ir.graph @cmpTestGraph(
  %lhs: tensor<10xf32>,
  %rhs: tensor<10xf32>
) -> tensor<10xi1> {
  // expected-error @below {{'nv_tensor_ir.cmp' op float tensor operands require a float comparator (oeq, one, ogt, etc.)}}
  %res = cmp %lhs "neq" %rhs : tensor<10xf32>
  results %res: tensor<10xi1>
}

// -----
nv_tensor_ir.graph @cmpTestGraph(
  %lhs: tensor<10xui32>,
  %rhs: tensor<10xui32>
) -> tensor<10xi1> {
  // expected-error @below {{'nv_tensor_ir.cmp' op integer tensor operands require an integer comparator (eq, neq, gt, ge, lt, le)}}
  %res = cmp %lhs "one" %rhs : tensor<10xui32>
  results %res: tensor<10xi1>
}


// -----
nv_tensor_ir.graph @powOp(
  %lhs: tensor<?x?x?xsi32>,
  %rhs: tensor<?x?x?xf32>
) -> tensor<?x?x?xf32> {
  // expected-error @below {{'nv_tensor_ir.pow' op failed to verify that all of {lhs, output} have same type}}
  %res = pow %lhs, %rhs : (tensor<?x?x?xsi32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  results %res: tensor<?x?x?xf32>
}

// -----
nv_tensor_ir.graph @powOp(
  %lhs: tensor<?x?x?xsi32>,
  %rhs: tensor<?x?x?xf32>
) -> tensor<?x?x?xsi32> {
  // expected-error @below {{'nv_tensor_ir.pow' op requires either both operands to have the same element type or base operand to have floating-point element type and exponent operand to have integer element type, but got base: 'si32', exponent: 'f32'}}
  %res = pow %lhs, %rhs : (tensor<?x?x?xsi32>, tensor<?x?x?xf32>) -> tensor<?x?x?xsi32>
  results %res: tensor<?x?x?xsi32>
}

// -----
nv_tensor_ir.graph @broadcastOpVerifySameDataTypes(
  %input: tensor<16x1x32xf32>
) -> (tensor<16x5x64xf64>){
  // expected-error @below {{requires input and output element types to match}}
  %1 = broadcast %input : tensor<16x1x32xf32> -> tensor<16x5x64xf64>
  results %1 : tensor<16x5x64xf64>
}

// -----
nv_tensor_ir.graph @broadcastOpVerifyBroadcastableShapes(
  %input: tensor<16x?x1xf32>
) -> (tensor<16x1x64xf32>){
  // expected-error @below {{broadcast dim 1 must change from a literal 1 in the input to ?/>1 in the output}}
  %1 = broadcast %input : tensor<16x?x1xf32> -> tensor<16x1x64xf32>
  results %1 : tensor<16x1x64xf32>
}

// -----
nv_tensor_ir.graph @broadcastOpVerifyRank(
  %input: tensor<1x64xf32>
) -> tensor<16x1x64xf32> {
  // expected-error @below {{requires input and output ranks to match}}
  %1 = broadcast %input : tensor<1x64xf32> -> tensor<16x1x64xf32>
  results %1 : tensor<16x1x64xf32>
}

// -----
nv_tensor_ir.graph @floatOpRejectsInteger(
  %input: tensor<4xsi32>
) -> tensor<4xsi32> {
  // expected-error @below {{requires a floating point type}}
  %1 = neg %input : tensor<4xsi32>
  results %1 : tensor<4xsi32>
}

// -----
nv_tensor_ir.graph @reshapeOpVerifyElementType(
  %input: tensor<2x3xf32>
) -> (tensor<6x1xf64>) {
  // expected-error @below {{failed to verify that all of {input, output} have same element type}}
  %reshaped = reshape %input : tensor<2x3xf32> -> tensor<6x1xf64>
  results %reshaped : tensor<6x1xf64>
}

// -----
nv_tensor_ir.graph @reshapeOpVerifyElementCount(
  %input: tensor<2x3xf32>
) -> (tensor<7x1xf32>) {
  // expected-error @below {{'nv_tensor_ir.reshape' op requires input and output to have the same number of elements, but got 6 elements in input and 7 elements in output}}
  %reshaped = reshape %input : tensor<2x3xf32> -> tensor<7x1xf32>
  results %reshaped : tensor<7x1xf32>
}

// -----
nv_tensor_ir.graph @reshapeOpVerifyElementCount3D(
  %input: tensor<4x3x2xf32>
) -> (tensor<1x22xf32>) {
  // expected-error @below {{'nv_tensor_ir.reshape' op requires input and output to have the same number of elements, but got 24 elements in input and 22 elements in output}}
  %reshaped = reshape %input : tensor<4x3x2xf32> -> tensor<1x22xf32>
  results %reshaped : tensor<1x22xf32>
}
// -----
nv_tensor_ir.graph @constantOp(
  %in: tensor<?x?x?xf32>
) -> tensor<?x?x?xf32> {
  // expected-error @below {{elements literal type must have static shape}}
  %c1 = nv_tensor_ir.constant dense<1.0> : tensor<?x?x?xf32>
  %c2 = nv_tensor_ir.add %c1, %in : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  results %c2: tensor<?x?x?xf32>
}


// -----
nv_tensor_ir.graph @transposeOpVerifyElementType(
  %input: tensor<2x3xf32>
) -> (tensor<3x2xf64>) {
  // expected-error @below {{'nv_tensor_ir.transpose' op requires the same element type for all operands and results}}
  %transposed = transpose %input permutation = [1, 0] : tensor<2x3xf32> -> tensor<3x2xf64>
  results %transposed : tensor<3x2xf64>
}

// -----
nv_tensor_ir.graph @transposeOpVerifyRankMismatch(
  %input: tensor<2x3xf32>
) -> (tensor<3x2x1xf32>) {
  // expected-error @below {{'nv_tensor_ir.transpose' op failed to verify that all of {input, output} have same rank}}
  %transposed = transpose %input permutation = [1, 0] : tensor<2x3xf32> -> tensor<3x2x1xf32>
  results %transposed : tensor<3x2x1xf32>
}

// -----
nv_tensor_ir.graph @transposeOpVerifyPermutationSize(
  %input: tensor<2x3x4xf32>
) -> (tensor<4x2x3xf32>) {
  // expected-error @below {{'nv_tensor_ir.transpose' op requires permutation array size to match tensor rank, but got permutation size 2 and tensor rank 3}}
  %transposed = transpose %input permutation = [2, 0] : tensor<2x3x4xf32> -> tensor<4x2x3xf32>
  results %transposed : tensor<4x2x3xf32>
}

// -----
nv_tensor_ir.graph @transposeOpVerifyInvalidIndex(
  %input: tensor<2x3xf32>
) -> (tensor<3x2xf32>) {
  // expected-error @below {{'nv_tensor_ir.transpose' op permutation contains invalid dimension index 2, expected value in range [0, 1]}}
  %transposed = transpose %input permutation = [1, 2] : tensor<2x3xf32> -> tensor<3x2xf32>
  results %transposed : tensor<3x2xf32>
}

// -----
nv_tensor_ir.graph @transposeOpVerifyNegativeIndex(
  %input: tensor<2x3xf32>
) -> (tensor<3x2xf32>) {
  // expected-error @below {{'nv_tensor_ir.transpose' op permutation contains invalid dimension index -1, expected value in range [0, 1]}}
  %transposed = transpose %input permutation = [1, -1] : tensor<2x3xf32> -> tensor<3x2xf32>
  results %transposed : tensor<3x2xf32>
}

// -----
nv_tensor_ir.graph @transposeOpVerifyDuplicateIndex(
  %input: tensor<2x3x4xf32>
) -> (tensor<3x2x3xf32>) {
  // expected-error @below {{'nv_tensor_ir.transpose' op permutation contains duplicate dimension index 1}}
  %transposed = transpose %input permutation = [1, 0, 1] : tensor<2x3x4xf32> -> tensor<3x2x3xf32>
  results %transposed : tensor<3x2x3xf32>
}

// -----
nv_tensor_ir.graph @transposeOpVerifyShapeMismatch(
  %input: tensor<2x3xf32>
) -> (tensor<2x3xf32>) {
  // expected-error @below {{'nv_tensor_ir.transpose' op output dimension 0 (size 2) does not match permuted input dimension 1 (size 3)}}
  %transposed = transpose %input permutation = [1, 0] : tensor<2x3xf32> -> tensor<2x3xf32>
  results %transposed : tensor<2x3xf32>
}

// -----
// Invalid slice: mismatched computed size vs output shape
nv_tensor_ir.graph @sliceInvalidSize(
  %input: tensor<4x6xf32>
) -> (tensor<2x4xf32>) {
  // expected-error @below {{'nv_tensor_ir.slice' op output dimension 1 (size 4) does not match computed slice size 3}}
  %s = slice %input starts = [1, 2] limits = [3, 5] strides = [1, 1]
       : tensor<4x6xf32> -> tensor<2x4xf32>
  results %s : tensor<2x4xf32>
}

// -----
// Invalid slice: out of bounds
nv_tensor_ir.graph @sliceOutOfBounds(
  %input: tensor<4x6xf32>
) -> (tensor<2x3xf32>) {
  // expected-error @below {{'nv_tensor_ir.slice' op slice out of bounds at dimension 0: start=5, limit=6, stride=1, input_dim=4}}
  %s = slice %input starts = [5, 2] limits = [6, 5] strides = [1, 1]
       : tensor<4x6xf32> -> tensor<2x3xf32>
  results %s : tensor<2x3xf32>
}

// -----
// Invalid slice: negative offset and non-positive stride
nv_tensor_ir.graph @sliceInvalidAttrs(
  %input: tensor<4x6xf32>
) -> (tensor<1x1xf32>) {
  // expected-error @below {{attribute 'starts' failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %s = slice %input starts = [-1, 0] limits = [1, 1] strides = [1, 0]
       : tensor<4x6xf32> -> tensor<1x1xf32>
  results %s : tensor<1x1xf32>
}

// -----
// Invalid slice: negative limit
nv_tensor_ir.graph @sliceNegativeLimit(
  %input: tensor<4x6xf32>
) -> (tensor<1x1xf32>) {
  // expected-error @below {{attribute 'limits' failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %s = slice %input starts = [0, 0] limits = [-1, 1] strides = [1, 1]
       : tensor<4x6xf32> -> tensor<1x1xf32>
  results %s : tensor<1x1xf32>
}

// -----
// Invalid slice: non-positive stride
nv_tensor_ir.graph @sliceNonPositiveStride(
  %input: tensor<4x6xf32>
) -> (tensor<1x1xf32>) {
  // expected-error @below {{attribute 'strides' failed to satisfy constraint: i64 dense array attribute whose value is positive}}
  %s = slice %input starts = [0, 0] limits = [1, 1] strides = [1, 0]
       : tensor<4x6xf32> -> tensor<1x1xf32>
  results %s : tensor<1x1xf32>
}

// -----
// Invalid slice: input and output ranks differ
nv_tensor_ir.graph @sliceRankMismatch(
  %input: tensor<4x6xf32>
) -> (tensor<2x3x1xf32>) {
  // expected-error @below {{'nv_tensor_ir.slice' op failed to verify that all of {input, output} have same rank}}
  %s = slice %input starts = [1, 2] limits = [3, 5] strides = [1, 1]
       : tensor<4x6xf32> -> tensor<2x3x1xf32>
  results %s : tensor<2x3x1xf32>
}

// -----
// Invalid concatenate: no inputs - parser error
nv_tensor_ir.graph @concatenateNoInputs() -> (tensor<2x3xf32>) {
  // expected-error @below {{expected 1 or more operands, but found 0}}
  %concat = concatenate dimension = 0 : () -> tensor<2x3xf32>
  results %concat : tensor<2x3xf32>
}

// -----
// Invalid concatenate: dimension out of bounds
nv_tensor_ir.graph @concatenateInvalidDim(
  %input1: tensor<2x3xf32>,
  %input2: tensor<1x3xf32>
) -> (tensor<3x3xf32>) {
  // expected-error @below {{'nv_tensor_ir.concatenate' op concatenation dimension 2 is out of bounds for tensor rank 2}}
  %concat = concatenate %input1, %input2 dimension = 2
            : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  results %concat : tensor<3x3xf32>
}

// -----
// Invalid concatenate: negative dimension
nv_tensor_ir.graph @concatenateNegativeDim(
  %input1: tensor<2x3xf32>,
  %input2: tensor<1x3xf32>
) -> (tensor<3x3xf32>) {
  // expected-error @below {{'nv_tensor_ir.concatenate' op concatenation dimension -1 is out of bounds for tensor rank 2}}
  %concat = concatenate %input1, %input2 dimension = -1
            : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  results %concat : tensor<3x3xf32>
}

// -----
// Invalid concatenate: mismatched element types
nv_tensor_ir.graph @concatenateMismatchedElementType(
  %input1: tensor<2x3xf32>,
  %input2: tensor<1x3xf64>
) -> (tensor<3x3xf32>) {
  // expected-error @below {{requires the same element type for all operands and results}}
  %concat = concatenate %input1, %input2 dimension = 0
            : (tensor<2x3xf32>, tensor<1x3xf64>) -> tensor<3x3xf32>
  results %concat : tensor<3x3xf32>
}

// -----
// Invalid concatenate: mismatched rank
nv_tensor_ir.graph @concatenateMismatchedRank(
  %input1: tensor<2x3xf32>,
  %input2: tensor<1x3x4xf32>
) -> (tensor<3x3xf32>) {
  // expected-error @below {{'nv_tensor_ir.concatenate' op input 1 has rank 3 but expected 2}}
  %concat = concatenate %input1, %input2 dimension = 0
            : (tensor<2x3xf32>, tensor<1x3x4xf32>) -> tensor<3x3xf32>
  results %concat : tensor<3x3xf32>
}

// -----
// Invalid concatenate: result rank differs from the input rank
nv_tensor_ir.graph @concatenateResultRankMismatch(
  %input1: tensor<2x3xf32>,
  %input2: tensor<1x3xf32>
) -> (tensor<3x3x1xf32>) {
  // expected-error @below {{'nv_tensor_ir.concatenate' op output has rank 3 but expected 2}}
  %concat = concatenate %input1, %input2 dimension = 0
            : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<3x3x1xf32>
  results %concat : tensor<3x3x1xf32>
}

// -----
// Invalid concatenate: mismatched shape in non-concat dimension
nv_tensor_ir.graph @concatenateMismatchedShape(
  %input1: tensor<2x3xf32>,
  %input2: tensor<1x2xf32>
) -> (tensor<3x3xf32>) {
  // expected-error @below {{expects non-concatenation dimensions of all inputs to match}}
  %concat = concatenate %input1, %input2 dimension = 0
            : (tensor<2x3xf32>, tensor<1x2xf32>) -> tensor<3x3xf32>
  results %concat : tensor<3x3xf32>
}

// -----
// Invalid concatenate: wrong output size in concat dimension
nv_tensor_ir.graph @concatenateWrongOutputSize(
  %input1: tensor<2x3xf32>,
  %input2: tensor<1x3xf32>
) -> (tensor<4x3xf32>) {
  // expected-error @below {{expects output concatenation dimension to equal the sum of input extents}}
  %concat = concatenate %input1, %input2 dimension = 0
            : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<4x3xf32>
  results %concat : tensor<4x3xf32>
}

// -----
// Invalid concatenate: wrong output size in non-concat dimension
nv_tensor_ir.graph @concatenateWrongOutputNonConcatDim(
  %input1: tensor<2x3xf32>,
  %input2: tensor<1x3xf32>
) -> (tensor<3x4xf32>) {
  // expected-error @below {{expects output non-concatenation dimensions to equal the merged input shape}}
  %concat = concatenate %input1, %input2 dimension = 0
            : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<3x4xf32>
  results %concat : tensor<3x4xf32>
}

// -----

// expected-error @below {{Invalid layout: (4..1):(1..3)}}
#invalid_layout = #nv_tensor_ir.tensor_source<0, 0, "(4..1):(1..3)">

nv_tensor_ir.graph @TensorSourceAttrWithInvalidLayout(
      %arg0: tensor<4x4xf32> {nv_tensor_ir.layout = #nv_tensor_ir.tensor_source<0, 0, "(4,1)">},
      %arg1: tensor<4x4xf32> {nv_tensor_ir.layout = #nv_tensor_ir.tensor_source<0, 0, "(4,1)">})
      -> (tensor<4x4xf32> {nv_tensor_ir.layout = #nv_tensor_ir.tensor_source<0, 0, "(4,1)">}) {
  %add = add %arg0, %arg1 {nv_tensor_ir.layout = #invalid_layout} : tensor<4x4xf32>
  results %add : tensor<4x4xf32>
}

// -----

nv_tensor_ir.graph @dimOutOfBounds(%input: tensor<?x4xf32>)
    -> (tensor<?x4xf32>) {
  // expected-error @below {{dimension 2 is out of bounds for tensor rank 2}}
  %size = dim %input dimension = 2 : tensor<?x4xf32>
  results %input : tensor<?x4xf32>
}

// -----

// Explicit metadata, when present, has one operand per dynamic result extent.
nv_tensor_ir.graph @splatWrongDynamicSizeCount(
  %value: f32, %d0: index
) -> (tensor<?x?xf32>) {
  // expected-error @below {{requires either no dynamic_dims operands or one for every dynamic result dimension, but got 1 operand(s) for 2 dynamic dimension(s)}}
  %result = splat %value dynamic_dims(%d0) : tensor<?x?xf32>
  results %result : tensor<?x?xf32>
}

// -----

nv_tensor_ir.graph @reshapeWrongDynamicSizeCount(
  %input: tensor<?xf32>, %d0: index
) -> (tensor<?x?xf32>) {
  // expected-error @below {{requires either no dynamic_dims operands or one for every dynamic result dimension, but got 1 operand(s) for 2 dynamic dimension(s)}}
  %result = reshape %input dynamic_dims(%d0)
      : tensor<?xf32> -> tensor<?x?xf32>
  results %result : tensor<?x?xf32>
}

// -----

// Invalid iota: negative dimension
nv_tensor_ir.graph @iotaNegativeDim() -> (tensor<2x3xf32>) {
  // expected-error @below {{'nv_tensor_ir.iota' op dimension -1 is out of bounds for tensor rank 2}}
  %iota = iota dimension = -1 : tensor<2x3xf32>
  results %iota : tensor<2x3xf32>
}

// -----

// Invalid iota: dimension out of bounds
nv_tensor_ir.graph @iotaDimOutOfBounds() -> (tensor<2x3xf32>) {
  // expected-error @below {{'nv_tensor_ir.iota' op dimension 3 is out of bounds for tensor rank 2}}
  %iota = iota dimension = 3 : tensor<2x3xf32>
  results %iota : tensor<2x3xf32>
}

// -----

// Invalid iota: static extent exceeds iota element type range (signed)
nv_tensor_ir.graph @iotaExtentTooLargeSigned() -> (tensor<40000xsi16>) {
  // expected-error @below {{extent 40000 along iota dimension 0 exceeds the maximum value representable in the iota element type}}
  %iota = iota dimension = 0 : tensor<40000xsi16>
  results %iota : tensor<40000xsi16>
}

// -----

// Invalid iota: static extent exceeds iota element type range (unsigned)
nv_tensor_ir.graph @iotaExtentTooLargeUnsigned() -> (tensor<70000xui16>) {
  // expected-error @below {{extent 70000 along iota dimension 0 exceeds the maximum value representable in the iota element type}}
  %iota = iota dimension = 0 : tensor<70000xui16>
  results %iota : tensor<70000xui16>
}
