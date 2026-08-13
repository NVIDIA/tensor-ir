// RUN: tensor_ir-opt %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: tensor_ir-opt %s -split-input-file | tensor_ir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: tensor_ir-opt -mlir-print-op-generic %s -split-input-file | tensor_ir-opt | FileCheck %s

// This the minimal example: an empty graph.
// CHECK: nv_tensor_ir.graph @someGraph() {
// CHECK-NEX: results
nv_tensor_ir.graph @someGraph() {
  results
}

// Pointwise binary operations
// CHECK-LABEL: nv_tensor_ir.graph @pointwiseBinaryFloatTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<10xf32>, %[[arg1:.*]]: tensor<10xf32>, %[[arg2:.*]]: tensor<10xf32>
nv_tensor_ir.graph @pointwiseBinaryFloatTestGraph(
  %arg0: tensor<10xf32>,
  %arg1: tensor<10xf32>,
  %arg2: tensor<10xf32>
) -> (tensor<10xf32>){
  // CHECK: %[[add:.*]] = add %[[arg0]], %[[arg1]] : tensor<10xf32>
  %add = add %arg0, %arg1 : tensor<10xf32>
  // CHECK: %[[add_square:.*]] = add_square %[[arg0]], %[[add]] : tensor<10xf32>
  %add_square = add_square %arg0, %add : tensor<10xf32>
  // CHECK: %[[div:.*]] = div %[[arg0]], %[[add_square]] : tensor<10xf32>
  %div = div %arg0, %add_square : tensor<10xf32>
  // CHECK: %[[max:.*]] = max %[[arg0]], %[[div]] : tensor<10xf32>
  %max = max %arg0, %div : tensor<10xf32>
  // CHECK: %[[min:.*]] = min %[[arg0]], %[[max]] : tensor<10xf32>
  %min = min %arg0, %max : tensor<10xf32>
  // CHECK: %[[mod:.*]] = mod %[[arg0]], %[[min]] : tensor<10xf32>
  %mod = mod %arg0, %min : tensor<10xf32>
  // CHECK: %[[mul:.*]] = mul %[[arg0]], %[[mod]] : tensor<10xf32>
  %mul = mul %arg0, %mod : tensor<10xf32>
  // CHECK: %[[pow:.*]] = pow %[[arg0]], %[[mul]] : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %pow = pow %arg0, %mul : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  // CHECK: %[[sub:.*]] = sub %[[arg0]], %[[pow]] : tensor<10xf32>
  %sub = sub %arg0, %pow : tensor<10xf32>
  // CHECK: %[[gelu:.*]] = gelu_bwd %[[arg0]], %[[sub]] : tensor<10xf32>
  %gelu = gelu_bwd %arg0, %sub : tensor<10xf32>
  // CHECK: %[[elu:.*]] = elu_fwd<beta = {{.*}}> %[[gelu]] : tensor<10xf32>
  %elu = elu_fwd<beta=1.0> %gelu : tensor<10xf32>
  // CHECK: %[[softplus:.*]] = softplus_fwd<beta = {{.*}}> %[[elu]] : tensor<10xf32>
  %softplus = softplus_fwd<beta=1.0> %elu :  tensor<10xf32>
  // CHECK: %[[swish:.*]] = swish_fwd<beta = {{.*}}> %[[softplus]] :  tensor<10xf32>
  %swish = swish_fwd<beta=1.0> %softplus :  tensor<10xf32>
  // CHECK: %[[sigmoid:.*]] = sigmoid_bwd %[[arg0]], %[[swish]] : tensor<10xf32>
  %sigmoid = sigmoid_bwd %arg0, %swish : tensor<10xf32>
  // CHECK: %[[tanh:.*]] = tanh_bwd %[[arg0]], %[[sigmoid]] : tensor<10xf32>
  %tanh = tanh_bwd %arg0, %sigmoid : tensor<10xf32>
  // CHECK: %[[geluapproxtanh:.*]] = gelu_approx_tanh_bwd %[[arg0]], %[[tanh]] : tensor<10xf32>
  %geluapproxtanh = gelu_approx_tanh_bwd %arg0, %tanh : tensor<10xf32>
  // CHECK: %[[relu:.*]] = relu_bwd %[[arg0]], %[[geluapproxtanh]] : tensor<10xf32>
  %relu = relu_bwd %arg0, %geluapproxtanh : tensor<10xf32>

  // CHECK: results %[[relu]] : tensor<10xf32>
  results %relu : tensor<10xf32>
}

// Pointwise binary operations
// CHECK-LABEL: nv_tensor_ir.graph @pointwiseBinaryUITestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<10xui32>, %[[arg1:.*]]: tensor<10xui32>
nv_tensor_ir.graph @pointwiseBinaryUITestGraph(
  %arg0: tensor<10xui32>,
  %arg1: tensor<10xui32>
) -> (tensor<10xui32>){
  // CHECK: %[[add:.*]] = add %[[arg0]], %[[arg1]] : tensor<10xui32>
  %add = add %arg0, %arg1 : tensor<10xui32>
  // CHECK: %[[add_square:.*]] = add_square %[[arg0]], %[[add]] : tensor<10xui32>
  %add_square = add_square %arg0, %add : tensor<10xui32>
  // CHECK: %[[div:.*]] = div %[[arg0]], %[[add_square]] : tensor<10xui32>
  %div = div %arg0, %add_square : tensor<10xui32>
  // CHECK: %[[max:.*]] = max %[[arg0]], %[[div]] : tensor<10xui32>
  %max = max %arg0, %div : tensor<10xui32>
  // CHECK: %[[min:.*]] = min %[[arg0]], %[[max]] : tensor<10xui32>
  %min = min %arg0, %max : tensor<10xui32>
  // CHECK: %[[mod:.*]] = mod %[[arg0]], %[[min]] : tensor<10xui32>
  %mod = mod %arg0, %min : tensor<10xui32>
  // CHECK: %[[mul:.*]] = mul %[[arg0]], %[[mod]] : tensor<10xui32>
  %mul = mul %arg0, %mod : tensor<10xui32>
  // CHECK: %[[pow:.*]] = pow %[[arg0]], %[[mul]] : (tensor<10xui32>, tensor<10xui32>) -> tensor<10xui32>
  %pow = pow %arg0, %mul : (tensor<10xui32>, tensor<10xui32>) -> tensor<10xui32>
  // CHECK: %[[sub:.*]] = sub %[[arg0]], %[[pow]] : tensor<10xui32>
  %sub = sub %arg0, %pow : tensor<10xui32>

  // CHECK: results %[[sub]] : tensor<10xui32>
  results %sub : tensor<10xui32>
}

// Pointwise binary operations
// CHECK-LABEL: nv_tensor_ir.graph @pointwiseBinarySITestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<10xsi32>, %[[arg1:.*]]: tensor<10xsi32>
nv_tensor_ir.graph @pointwiseBinarySITestGraph(
  %arg0: tensor<10xsi32>,
  %arg1: tensor<10xsi32>
) -> (tensor<10xsi32>){
  // CHECK: %[[add:.*]] = add %[[arg0]], %[[arg1]] : tensor<10xsi32>
  %add = add %arg0, %arg1 : tensor<10xsi32>
  // CHECK: %[[add_square:.*]] = add_square %[[arg0]], %[[add]] : tensor<10xsi32>
  %add_square = add_square %arg0, %add : tensor<10xsi32>
  // CHECK: %[[div:.*]] = div %[[arg0]], %[[add_square]] : tensor<10xsi32>
  %div = div %arg0, %add_square : tensor<10xsi32>
  // CHECK: %[[max:.*]] = max %[[arg0]], %[[div]] : tensor<10xsi32>
  %max = max %arg0, %div : tensor<10xsi32>
  // CHECK: %[[min:.*]] = min %[[arg0]], %[[max]] : tensor<10xsi32>
  %min = min %arg0, %max : tensor<10xsi32>
  // CHECK: %[[mod:.*]] = mod %[[arg0]], %[[min]] : tensor<10xsi32>
  %mod = mod %arg0, %min : tensor<10xsi32>
  // CHECK: %[[mul:.*]] = mul %[[arg0]], %[[mod]] : tensor<10xsi32>
  %mul = mul %arg0, %mod : tensor<10xsi32>
  // CHECK: %[[pow:.*]] = pow %[[arg0]], %[[mul]] : (tensor<10xsi32>, tensor<10xsi32>) -> tensor<10xsi32>
  %pow = pow %arg0, %mul : (tensor<10xsi32>, tensor<10xsi32>) -> tensor<10xsi32>
  // CHECK: %[[sub:.*]] = sub %[[arg0]], %[[pow]] : tensor<10xsi32>
  %sub = sub %arg0, %pow : tensor<10xsi32>

  // CHECK: results %[[sub]] : tensor<10xsi32>
  results %sub : tensor<10xsi32>
}


// CHECK-LABEL: nv_tensor_ir.graph @powTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<10xf32>, %[[arg1:.*]]: tensor<10xsi32>
nv_tensor_ir.graph @powTestGraph(
  %arg0: tensor<10xf32>,
  %arg1: tensor<10xsi32>
) -> (tensor<10xf32>){
  // CHECK: %[[pow:.*]] = pow %[[arg0]], %[[arg1]] : (tensor<10xf32>, tensor<10xsi32>) -> tensor<10xf32>
  %pow = pow %arg0, %arg1 : (tensor<10xf32>, tensor<10xsi32>) -> tensor<10xf32>
  // CHECK: results %[[pow]] : tensor<10xf32>
  results %pow : tensor<10xf32>
}


// Pointwise unary operations
// CHECK-LABEL: nv_tensor_ir.graph @pointwiseUnaryTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<10xf32>,
// CHECK-SAME: %[[arg1:.*]]: tensor<10xf32>
nv_tensor_ir.graph @pointwiseUnaryTestGraph
(
  %arg0: tensor<10xf32>,
  %arg1: tensor<10xf32>
) -> (tensor<10xf32>){
//CHECK:  %[[abs:.*]] = abs %[[arg0]] : tensor<10xf32>
  %abs = abs %arg0 : tensor<10xf32>
//CHECK:  %[[ceil:.*]] = ceil %[[abs]] : tensor<10xf32>
  %ceil = ceil %abs : tensor<10xf32>
//CHECK:  %[[cos:.*]] = cos %[[ceil]] : tensor<10xf32>
  %cos = cos %ceil : tensor<10xf32>
//CHECK:  %[[exp:.*]] = exp %[[cos]] : tensor<10xf32>
  %exp = exp %cos : tensor<10xf32>
//CHECK:  %[[floor:.*]] = floor %[[exp]] : tensor<10xf32>
  %floor = floor %exp : tensor<10xf32>
//CHECK:  %[[log:.*]] = log %[[floor]] : tensor<10xf32>
  %log = log %floor : tensor<10xf32>
//CHECK:  %[[neg:.*]] = neg %[[log]] : tensor<10xf32>
  %neg = neg %log : tensor<10xf32>
//CHECK:  %[[rsqrt:.*]] = rsqrt %[[neg]] : tensor<10xf32>
  %rsqrt = rsqrt %neg : tensor<10xf32>
//CHECK:  %[[sin:.*]] = sin %[[rsqrt]] : tensor<10xf32>
  %sin = sin %rsqrt : tensor<10xf32>
//CHECK:  %[[sqrt:.*]] = sqrt %[[sin]] : tensor<10xf32>
  %sqrt = sqrt %sin : tensor<10xf32>
//CHECK:  %[[tan:.*]] = tan %[[sqrt]] : tensor<10xf32>
  %tan = tan %sqrt : tensor<10xf32>
//CHECK:  %[[erf:.*]] = erf %[[tan]] : tensor<10xf32>
  %erf = erf %tan : tensor<10xf32>
//CHECK:  %[[reciprocal:.*]] = reciprocal %[[erf]] : tensor<10xf32>
  %reciprocal = reciprocal %erf : tensor<10xf32>
//CHECK:  %[[relufwd:.*]] = relu_fwd %[[reciprocal]] : tensor<10xf32>
  %relu_fwd = relu_fwd %reciprocal : tensor<10xf32>
//CHECK:  %[[tanhfwd:.*]] = tanh_fwd %[[reciprocal]] : tensor<10xf32>
  %tanh_fwd = tanh_fwd %reciprocal : tensor<10xf32>
//CHECK:  %[[sigmoidfwd:.*]] = sigmoid_fwd %[[tanhfwd]] : tensor<10xf32>
  %sigmoid_fwd = sigmoid_fwd %tanh_fwd : tensor<10xf32>
//CHECK:  %[[gelufwd:.*]] = gelu_fwd %[[sigmoidfwd]] : tensor<10xf32>
  %gelu_fwd = gelu_fwd %sigmoid_fwd : tensor<10xf32>
//CHECK:  %[[geluapproxtanhfwd:.*]] = gelu_approx_tanh_fwd %[[gelufwd]] : tensor<10xf32>
  %geluapproxtanh_fwd = gelu_approx_tanh_fwd %gelu_fwd : tensor<10xf32>
  // CHECK: results %[[geluapproxtanhfwd]] : tensor<10xf32>
  results %geluapproxtanh_fwd : tensor<10xf32>
}



// Ternary Operations
// CHECK-LABEL: nv_tensor_ir.graph @pointwiseTernaryTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<10xf32>, %[[arg1:.*]]: tensor<10xf32>
nv_tensor_ir.graph @pointwiseTernaryTestGraph(
  %arg0: tensor<10xf32>,
  %arg1: tensor<10xf32>
) -> (tensor<10xf32>){

  // CHECK: %[[SoftplusBwd:.*]] = softplus_bwd<beta = {{.*}}> %[[arg0]], %[[arg1]] :  tensor<10xf32>
  %0 = softplus_bwd<beta=1.0: f32> %arg0, %arg1 : tensor<10xf32>

  // CHECK: %[[SwishBwd:.*]] = swish_bwd<beta = {{.*}}> %[[SoftplusBwd]], %[[arg1]] :  tensor<10xf32>
  %1 = swish_bwd<beta=1.0: bf16> %0, %arg1 :  tensor<10xf32>

  // CHECK: %[[EluBwd:.*]] = elu_bwd<{{.*}}> %[[SwishBwd]], %[[arg1]] : tensor<10xf32>
  %2 = elu_bwd<beta=1.0: f16> %1, %arg1 : tensor<10xf32>

  results %2 : tensor<10xf32>
}


// Floating point Comparison Operations
// CHECK-LABEL: nv_tensor_ir.graph @pointwiseCmpfTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<10xf32>,
// CHECK-SAME: %[[arg1:.*]]: tensor<10xf32>
nv_tensor_ir.graph @pointwiseCmpfTestGraph(
  %arg0: tensor<10xf32>,
  %arg1: tensor<10xf32>
) -> (tensor<10xi1>, tensor<10xi1>, tensor<10xi1>, tensor<10xi1>, tensor<10xi1>, tensor<10xi1>,
      tensor<10xi1>, tensor<10xi1>, tensor<10xi1>, tensor<10xi1>, tensor<10xi1>, tensor<10xi1>){

  // CHECK: %[[cmp1:.*]] = cmp %[[arg0]] oeq %[[arg1]] : tensor<10xf32>
  %cmp1 = cmp %arg0 oeq %arg1 : tensor<10xf32>

  // CHECK: %[[cmp2:.*]] = cmp %[[arg0]] one %[[arg1]] : tensor<10xf32>
  %cmp2 = cmp %arg0 one %arg1 : tensor<10xf32>

  // CHECK: %[[cmp3:.*]] = cmp %[[arg0]] ogt %[[arg1]] : tensor<10xf32>
  %cmp3 = cmp %arg0 ogt %arg1 : tensor<10xf32>

  // CHECK: %[[cmp4:.*]] = cmp %[[arg0]] oge %[[arg1]] : tensor<10xf32>
  %cmp4 = cmp %arg0 oge %arg1 : tensor<10xf32>

  // CHECK: %[[cmp5:.*]] = cmp %[[arg0]] olt %[[arg1]] : tensor<10xf32>
  %cmp5 = cmp %arg0 olt %arg1 : tensor<10xf32>

  // CHECK: %[[cmp6:.*]] = cmp %[[arg0]] ole %[[arg1]] : tensor<10xf32>
  %cmp6 = cmp %arg0 ole %arg1 : tensor<10xf32>

  // CHECK: %[[cmp7:.*]] = cmp %[[arg0]] ueq %[[arg1]] : tensor<10xf32>
  %cmp7 = cmp %arg0 ueq %arg1 : tensor<10xf32>

  // CHECK: %[[cmp8:.*]] = cmp %[[arg0]] une %[[arg1]] : tensor<10xf32>
  %cmp8 = cmp %arg0 une %arg1 : tensor<10xf32>

  // CHECK: %[[cmp9:.*]] = cmp %[[arg0]] ugt %[[arg1]] : tensor<10xf32>
  %cmp9 = cmp %arg0 ugt %arg1 : tensor<10xf32>

  // CHECK: %[[cmp10:.*]] = cmp %[[arg0]] uge %[[arg1]] : tensor<10xf32>
  %cmp10 = cmp %arg0 uge %arg1 : tensor<10xf32>

  // CHECK: %[[cmp11:.*]] = cmp %[[arg0]] ult %[[arg1]] : tensor<10xf32>
  %cmp11 = cmp %arg0 ult %arg1 : tensor<10xf32>

  // CHECK: %[[cmp12:.*]] = cmp %[[arg0]] ule %[[arg1]] : tensor<10xf32>
  %cmp12 = cmp %arg0 ule %arg1 : tensor<10xf32>

  results %cmp1, %cmp2, %cmp3, %cmp4, %cmp5, %cmp6, %cmp7, %cmp8, %cmp9, %cmp10, %cmp11, %cmp12 :
      tensor<10xi1>, tensor<10xi1>, tensor<10xi1>,
      tensor<10xi1>, tensor<10xi1>, tensor<10xi1>,
      tensor<10xi1>, tensor<10xi1>, tensor<10xi1>,
      tensor<10xi1>, tensor<10xi1>, tensor<10xi1>
}

// Integer Comparison Operations
// CHECK-LABEL: nv_tensor_ir.graph @pointwiseCmpiTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<10xsi32>,
// CHECK-SAME: %[[arg1:.*]]: tensor<10xsi32>
nv_tensor_ir.graph @pointwiseCmpiTestGraph(
  %arg0: tensor<10xsi32>,
  %arg1: tensor<10xsi32>
) -> (tensor<10xi1>, tensor<10xi1>, tensor<10xi1>, tensor<10xi1>, tensor<10xi1>, tensor<10xi1>){

  // CHECK: %[[cmp1:.*]] = cmp %[[arg0]] eq %[[arg1]] : tensor<10xsi32>
  %cmp1 = cmp %arg0 eq %arg1 : tensor<10xsi32>

  // CHECK: %[[cmp2:.*]] = cmp %[[arg0]] neq %[[arg1]] : tensor<10xsi32>
  %cmp2 = cmp %arg0 neq %arg1 : tensor<10xsi32>

  // CHECK: %[[cmp3:.*]] = cmp %[[arg0]] gt %[[arg1]] : tensor<10xsi32>
  %cmp3 = cmp %arg0 gt %arg1 : tensor<10xsi32>

  // CHECK: %[[cmp4:.*]] = cmp %[[arg0]] ge %[[arg1]] : tensor<10xsi32>
  %cmp4 = cmp %arg0 ge %arg1 : tensor<10xsi32>

  // CHECK: %[[cmp5:.*]] = cmp %[[arg0]] lt %[[arg1]] : tensor<10xsi32>
  %cmp5 = cmp %arg0 lt %arg1 : tensor<10xsi32>

  // CHECK: %[[cmp6:.*]] = cmp %[[arg0]] le %[[arg1]] : tensor<10xsi32>
  %cmp6 = cmp %arg0 le %arg1 : tensor<10xsi32>

  results %cmp1, %cmp2, %cmp3, %cmp4, %cmp5, %cmp6 :
      tensor<10xi1>, tensor<10xi1>, tensor<10xi1>,
      tensor<10xi1>, tensor<10xi1>, tensor<10xi1>
}

// Boolean Operations
// CHECK-LABEL: nv_tensor_ir.graph @pointwiseBooleanTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<10xsi8>,
// CHECK-SAME: %[[arg1:.*]]: tensor<10xsi8>
nv_tensor_ir.graph @pointwiseBooleanTestGraph(
  %arg0: tensor<10xsi8>,
  %arg1: tensor<10xsi8>
) -> (tensor<10xi1>, tensor<10xi1>){

  // CHECK: %[[and:.*]] = and %[[arg0]], %[[arg1]] : tensor<10xsi8>
  %and = and %arg0, %arg1 : tensor<10xsi8>

  // CHECK: %[[or:.*]] = or %[[arg0]], %[[arg1]] : tensor<10xsi8>
  %or = or %arg0, %arg1 : tensor<10xsi8>

  // CHECK: %[[and:.*]] = not %[[or]] : tensor<10xi1>
  %not = not %or : tensor<10xi1>

  results %and, %not : tensor<10xi1>, tensor<10xi1>
}

// Matmul Operation
// CHECK-LABEL: nv_tensor_ir.graph @MatmulTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<8x16x32xf32> {nv_tensor_ir.stride = "(512,32,1)"},
// CHECK-SAME: %[[arg1:.*]]: tensor<8x32x48xf32> {nv_tensor_ir.stride = "(1536,48,1)"}
nv_tensor_ir.graph @MatmulTestGraph(
  %A: tensor<8x16x32xf32> {nv_tensor_ir.stride = "(512,32,1)"},
  %B: tensor<8x32x48xf32> {nv_tensor_ir.stride = "(1536,48,1)"}
) -> (tensor<8x16x48xf32> {nv_tensor_ir.stride = "(768,48,1)"}) {
  // CHECK: %[[matmul:.*]] = matmul(%[[arg0]], %[[arg1]])
  // CHECK: : (tensor<8x16x32xf32>, tensor<8x32x48xf32>) -> tensor<8x16x48xf32>
  %C = matmul(%A, %B)
    : (tensor<8x16x32xf32>, tensor<8x32x48xf32>) -> tensor<8x16x48xf32>
  results %C: tensor<8x16x48xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @MatmulAccumulatorTestGraph
// CHECK-SAME: %[[A:.*]]: tensor<16x32xf32>,
// CHECK-SAME: %[[B:.*]]: tensor<32x8xf32>,
// CHECK-SAME: %[[ACC:.*]]: tensor<16x8xf32>
nv_tensor_ir.graph @MatmulAccumulatorTestGraph(
  %A: tensor<16x32xf32>,
  %B: tensor<32x8xf32>,
  %Acc: tensor<16x8xf32>
) -> tensor<16x8xf32> {
  // CHECK: %[[RESULT:.*]] = matmul(%[[A]], %[[B]]) accum(%[[ACC]] : tensor<16x8xf32>)
  // CHECK-SAME: : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
  // CHECK: results %[[RESULT]] : tensor<16x8xf32>
  %C = matmul(%A, %B) accum(%Acc : tensor<16x8xf32>)
    : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
  results %C : tensor<16x8xf32>
}

// -----

// A unit batch dimension broadcasts to a dynamic peer without making the
// result batch dimension static.
// CHECK-LABEL: nv_tensor_ir.graph @MatmulDynamicUnitBatchBroadcastTestGraph
nv_tensor_ir.graph @MatmulDynamicUnitBatchBroadcastTestGraph(
  %A: tensor<?x16x32xf32>,
  %B: tensor<1x32x48xf32>
) -> tensor<?x16x48xf32> {
  // CHECK: matmul(%{{.*}}, %{{.*}}) : (tensor<?x16x32xf32>, tensor<1x32x48xf32>) -> tensor<?x16x48xf32>
  %C = matmul(%A, %B)
    : (tensor<?x16x32xf32>, tensor<1x32x48xf32>)
      -> tensor<?x16x48xf32>
  results %C : tensor<?x16x48xf32>
}

// Reduction Operation
// CHECK-LABEL: nv_tensor_ir.graph @reductionAddTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<16x32x64xf32>
nv_tensor_ir.graph @reductionAddTestGraph(
  %input: tensor<16x32x64xf32>
) -> (tensor<16x1x64xf32>){
  // CHECK: %[[reduce:.*]] = reduce(%[[arg0]])
  // CHECK: dimensions = [1]
  // CHECK: reduction_mode = <add>
  // CHECK: : tensor<16x32x64xf32> -> tensor<16x1x64xf32>
  %reduce = reduce(%input)<
      dimensions = [1],
      reduction_mode = <add> >
       : tensor<16x32x64xf32> -> tensor<16x1x64xf32>
  results %reduce : tensor<16x1x64xf32>
}

// CHECK-LABEL: nv_tensor_ir.graph @reductionAddTestGraphDynInput
// CHECK-SAME: %[[arg0:.*]]: tensor<16x?x64xf32>
nv_tensor_ir.graph @reductionAddTestGraphDynInput(
  %input: tensor<16x?x64xf32>
) -> (tensor<16x1x64xf32>){
  // CHECK: %[[reduce:.*]] = reduce(%[[arg0]])
  // CHECK: dimensions = [1]
  // CHECK: reduction_mode = <add>
  // CHECK: : tensor<16x?x64xf32> -> tensor<16x1x64xf32>
  %reduce = reduce(%input)<
      dimensions = [1],
      reduction_mode = <add> >
       : tensor<16x?x64xf32> -> tensor<16x1x64xf32>
  results %reduce : tensor<16x1x64xf32>
}

// CHECK-LABEL: nv_tensor_ir.graph @convertTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<16x1x64xf32>
nv_tensor_ir.graph @convertTestGraph(
  %input: tensor<16x1x64xf32>
) -> (tensor<16x1x64xf64>){
  // CHECK: %[[convert:.*]] = convert %[[arg0]] : tensor<16x1x64xf32> -> tensor<16x1x64xf64>
  %1 = convert %input
       : tensor<16x1x64xf32> -> tensor<16x1x64xf64>
  results %1 : tensor<16x1x64xf64>
}

// CHECK-LABEL: nv_tensor_ir.graph @splatTestGraph
// CHECK-SAME: %[[arg0:.*]]: f16
nv_tensor_ir.graph @splatTestGraph(
  %input: f16
) -> (tensor<16x1x64xf16>){
  // CHECK: %[[splat:.*]] = splat %[[arg0]] : tensor<16x1x64xf16>
  %1 = splat %input : tensor<16x1x64xf16>
  results %1 : tensor<16x1x64xf16>
}

// Binary Select Operation
// CHECK-LABEL: nv_tensor_ir.graph @binarySelectTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<10xi1>,
// CHECK-SAME: %[[arg1:.*]]: tensor<10xf32>,
// CHECK-SAME: %[[arg2:.*]]: tensor<10xf32>
nv_tensor_ir.graph @binarySelectTestGraph(
  %select: tensor<10xi1>,
  %lhs: tensor<10xf32>,
  %rhs: tensor<10xf32>
) -> (tensor<10xf32>){

  // CHECK: %[[cmp1:.*]] = binary_select %[[arg0]], %[[arg1]], %[[arg2]] : tensor<10xf32>
  %cmp1 = binary_select %select, %lhs, %rhs : tensor<10xf32>

  results %cmp1: tensor<10xf32>
}

// CHECK-LABEL: nv_tensor_ir.graph @broadcastTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<16x1x64xf32>
nv_tensor_ir.graph @broadcastTestGraph(
  %input: tensor<16x1x64xf32>
) -> (tensor<16x32x64xf32>){
  // CHECK: %[[broadcast:.*]] = broadcast %[[arg0]] : tensor<16x1x64xf32> -> tensor<16x32x64xf32>
  %1 = broadcast %input : tensor<16x1x64xf32> -> tensor<16x32x64xf32>
  results %1 : tensor<16x32x64xf32>
}

// -----

// si32 constant operations
// CHECK-LABEL: nv_tensor_ir.graph @si32_constantOp
// CHECK: %[[c0:.*]] = constant 1 : si32
// CHECK: %[[c1:.*]] = splat %[[c0]] : tensor<4xsi32>
// CHECK: %[[c2:.*]] = constant dense<1> : tensor<2x2xsi32>
// CHECK: %[[c3:.*]] = constant dense<[1, 2, 3, 4]> : tensor<4xsi32>
// CHECK: %[[c4:.*]] =
// CHECK-SAME{LITERAL} constant dense<[[1, 2], [3, 4]]> : tensor<2x2xsi32>
// CHECK: results %[[c1]], %[[c2]], %[[c3]], %[[c4]] :
// CHECK: tensor<4xsi32>, tensor<2x2xsi32>, tensor<4xsi32>, tensor<2x2xsi32>
nv_tensor_ir.graph @si32_constantOp() -> (tensor<4xsi32>, tensor<2x2xsi32>, tensor<4xsi32>, tensor<2x2xsi32>) {
  %c_si32 = constant 1 : si32
  %c_si32_splat = splat %c_si32 : tensor<4xsi32>
  %c_si32_dense_splat = constant dense<1> : tensor<2x2xsi32>
  %c_si32_dense_full = constant dense<[1,2,3,4]> : tensor<4xsi32>
  %c_si32_dense_full_2d = constant dense<[[1,2],[3,4]]> : tensor<2x2xsi32>
  results %c_si32_splat, %c_si32_dense_splat, %c_si32_dense_full, %c_si32_dense_full_2d :
          tensor<4xsi32>, tensor<2x2xsi32>, tensor<4xsi32>, tensor<2x2xsi32>
}

// -----


// f32 constant operations
// CHECK-LABEL: nv_tensor_ir.graph @f32_constantOp
// CHECK: %[[c0:.*]] = constant 4.000000e+00 : f32
// CHECK: %[[c1:.*]] = splat %[[c0]] : tensor<4xf32>
// CHECK: %[[c2:.*]] = constant dense<1.000000e+00> : tensor<2x2xf32>
// CHECK: %[[c3:.*]] = constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
// CHECK: %[[c4:.*]] =
// CHECK-SAME{LITERAL} constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
// CHECK: results %[[c1]], %[[c2]], %[[c3]], %[[c4]] :
// CHECK: tensor<4xf32>, tensor<2x2xf32>, tensor<4xf32>, tensor<2x2xf32>
nv_tensor_ir.graph @f32_constantOp() -> (tensor<4xf32>, tensor<2x2xf32>, tensor<4xf32>, tensor<2x2xf32>) {
  %c_f32 = constant 4.0 : f32
  %c_f32_splat = splat %c_f32 : tensor<4xf32>
  %c_f32_dense_splat = constant dense<1.0> : tensor<2x2xf32>
  %c_f32_dense_full = constant dense<[1.0,2.0,3.0,4.0]> : tensor<4xf32>
  %c_f32_dense_full_2d = constant dense<[[1.0,2.0],[3.0,4.0]]> : tensor<2x2xf32>
  results %c_f32_splat, %c_f32_dense_splat, %c_f32_dense_full, %c_f32_dense_full_2d :
          tensor<4xf32>, tensor<2x2xf32>, tensor<4xf32>, tensor<2x2xf32>
}

// -----

// ui32 constant operations
// CHECK-LABEL: nv_tensor_ir.graph @ui32_constantOp
// CHECK: %[[c0:.*]] = constant 1 : ui32
// CHECK: %[[c1:.*]] = splat %[[c0]] : tensor<4xui32>
// CHECK: %[[c2:.*]] = constant dense<1> : tensor<2x2xui32>
// CHECK: %[[c3:.*]] = constant dense<[1, 2, 3, 4]> : tensor<4xui32>
// CHECK: %[[c4:.*]] =
// CHECK-SAME{LITERAL} constant dense<[[1, 2], [3, 4]]> : tensor<2x2xui32>
// CHECK: results %[[c1]], %[[c2]], %[[c3]], %[[c4]] :
// CHECK: tensor<4xui32>, tensor<2x2xui32>, tensor<4xui32>, tensor<2x2xui32>
nv_tensor_ir.graph @ui32_constantOp() -> (tensor<4xui32>, tensor<2x2xui32>, tensor<4xui32>, tensor<2x2xui32>) {
  %c_ui32 = constant 1 : ui32
  %c_ui32_splat = splat %c_ui32 : tensor<4xui32>
  %c_ui32_dense_splat = constant dense<1> : tensor<2x2xui32>
  %c_ui32_dense_full = constant dense<[1,2,3,4]> : tensor<4xui32>
  %c_ui32_dense_full_2d = constant dense<[[1,2],[3,4]]> : tensor<2x2xui32>
  results %c_ui32_splat, %c_ui32_dense_splat, %c_ui32_dense_full, %c_ui32_dense_full_2d :
          tensor<4xui32>, tensor<2x2xui32>, tensor<4xui32>, tensor<2x2xui32>
}

// -----

// ReductionUD Operations
// CHECK-LABEL: nv_tensor_ir.graph @reductionUDWelfordTestGraph
// CHECK-SAME: %[[in:.*]]: tensor<8x32x64xf32>, %[[in_0:.*]]: tensor<8x32x64xf32>, %[[in_1:.*]]: tensor<8x32x64xf32>
nv_tensor_ir.graph @reductionUDWelfordTestGraph(
  %input: tensor<8x32x64xf32>,
  %m2: tensor<8x32x64xf32>,
  %weight: tensor<8x32x64xf32>
) -> (tensor<8x1x64xf32>, tensor<8x1x64xf32>, tensor<8x1x64xf32>) {
  // CHECK: %[[reduce:.*]]:3 = reduce_ud(%[[in]], %[[in_0]], %[[in_1]])
  // CHECK: dimensions = [1]
  // CHECK: identity = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK: (%[[arg0:.*]]: f32, %[[arg1:.*]]: f32, %[[arg2:.*]]: f32, %[[arg3:.*]]: f32, %[[arg4:.*]]: f32, %[[arg5:.*]]: f32)
  // CHECK: {
  // CHECK:   %[[sub:.*]] = arith.subf %[[arg3]], %[[arg0]] : f32
  // CHECK:   %[[add1:.*]] = arith.addf %[[arg2]], %[[arg5]] : f32
  // CHECK:   %[[div:.*]] = arith.divf %[[arg5]], %[[add1]] : f32
  // CHECK:   %[[mul1:.*]] = arith.mulf %[[sub]], %[[div]] : f32
  // CHECK:   %[[add2:.*]] = arith.addf %[[arg0]], %[[mul1]] : f32
  // CHECK:   %[[add3:.*]] = arith.addf %[[arg1]], %[[arg4]] : f32
  // CHECK:   %[[mul2:.*]] = arith.mulf %[[sub]], %[[sub]] : f32
  // CHECK:   %[[mul3:.*]] = arith.mulf %[[mul2]], %[[arg2]] : f32
  // CHECK:   %[[mul4:.*]] = arith.mulf %[[mul3]], %[[div]] : f32
  // CHECK:   %[[add4:.*]] = arith.addf %[[add3]], %[[mul4]] : f32
  // CHECK:   nv_tensor_ir.yield %[[mul1]], %[[mul4]], %[[sub]] : f32, f32, f32
  // CHECK: }
  // CHECK: : tensor<8x32x64xf32>, tensor<8x32x64xf32>, tensor<8x32x64xf32> -> tensor<8x1x64xf32>, tensor<8x1x64xf32>, tensor<8x1x64xf32>
  %reduce:3 = reduce_ud(%input, %m2, %weight)<
    dimensions = [1],
    identity = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]>
   (%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32) {
    %0 = arith.subf %arg3, %arg0 : f32
    %1 = arith.addf %arg2, %arg5 : f32
    %2 = arith.divf %arg5, %1 : f32
    %3 = arith.mulf %0, %2 : f32
    %4 = arith.addf %arg0, %3 : f32
    %5 = arith.addf %arg1, %arg4 : f32
    %6 = arith.mulf %0, %0 : f32
    %7 = arith.mulf %6, %arg2 : f32
    %8 = arith.mulf %7, %2 : f32
    %9 = arith.addf %5, %8 : f32
    nv_tensor_ir.yield %3, %8, %0 : f32, f32, f32
  } : tensor<8x32x64xf32>, tensor<8x32x64xf32>, tensor<8x32x64xf32> -> tensor<8x1x64xf32>, tensor<8x1x64xf32>, tensor<8x1x64xf32>
  // CHECK: results %[[reduce]]#0, %[[reduce]]#1, %[[reduce]]#2 : tensor<8x1x64xf32>, tensor<8x1x64xf32>, tensor<8x1x64xf32>
  results %reduce#0, %reduce#1, %reduce#2 : tensor<8x1x64xf32>, tensor<8x1x64xf32>, tensor<8x1x64xf32>
}

// -----
// CHECK-LABEL: nv_tensor_ir.graph @reductionUDWelfordTestGraphDynInput
// CHECK-SAME: %[[in:.*]]: tensor<8x?x64xf32>, %[[in_0:.*]]: tensor<8x?x64xf32>, %[[in_1:.*]]: tensor<8x?x64xf32>
nv_tensor_ir.graph @reductionUDWelfordTestGraphDynInput(
  %input: tensor<8x?x64xf32>,
  %m2: tensor<8x?x64xf32>,
  %weight: tensor<8x?x64xf32>
) -> (tensor<8x1x64xf32>, tensor<8x1x64xf32>, tensor<8x1x64xf32>) {
  // CHECK: %[[reduce:.*]]:3 = reduce_ud(%[[in]], %[[in_0]], %[[in_1]])
  // CHECK: dimensions = [1]
  // CHECK: identity = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK: (%[[arg0:.*]]: f32, %[[arg1:.*]]: f32, %[[arg2:.*]]: f32, %[[arg3:.*]]: f32, %[[arg4:.*]]: f32, %[[arg5:.*]]: f32)
  // CHECK: {
  // CHECK:   %[[sub:.*]] = arith.subf %[[arg3]], %[[arg0]] : f32
  // CHECK:   %[[add1:.*]] = arith.addf %[[arg2]], %[[arg5]] : f32
  // CHECK:   %[[div:.*]] = arith.divf %[[arg5]], %[[add1]] : f32
  // CHECK:   %[[mul1:.*]] = arith.mulf %[[sub]], %[[div]] : f32
  // CHECK:   %[[add2:.*]] = arith.addf %[[arg0]], %[[mul1]] : f32
  // CHECK:   %[[add3:.*]] = arith.addf %[[arg1]], %[[arg4]] : f32
  // CHECK:   %[[mul2:.*]] = arith.mulf %[[sub]], %[[sub]] : f32
  // CHECK:   %[[mul3:.*]] = arith.mulf %[[mul2]], %[[arg2]] : f32
  // CHECK:   %[[mul4:.*]] = arith.mulf %[[mul3]], %[[div]] : f32
  // CHECK:   %[[add4:.*]] = arith.addf %[[add3]], %[[mul4]] : f32
  // CHECK:   nv_tensor_ir.yield %[[mul1]], %[[mul4]], %[[sub]] : f32, f32, f32
  // CHECK: }
  // CHECK: : tensor<8x?x64xf32>, tensor<8x?x64xf32>, tensor<8x?x64xf32> -> tensor<8x1x64xf32>, tensor<8x1x64xf32>, tensor<8x1x64xf32>
  %reduce:3 = reduce_ud(%input, %m2, %weight)<
    dimensions = [1],
    identity = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]>
   (%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32) {
    %0 = arith.subf %arg3, %arg0 : f32
    %1 = arith.addf %arg2, %arg5 : f32
    %2 = arith.divf %arg5, %1 : f32
    %3 = arith.mulf %0, %2 : f32
    %4 = arith.addf %arg0, %3 : f32
    %5 = arith.addf %arg1, %arg4 : f32
    %6 = arith.mulf %0, %0 : f32
    %7 = arith.mulf %6, %arg2 : f32
    %8 = arith.mulf %7, %2 : f32
    %9 = arith.addf %5, %8 : f32
    nv_tensor_ir.yield %3, %8, %0 : f32, f32, f32
  } : tensor<8x?x64xf32>, tensor<8x?x64xf32>, tensor<8x?x64xf32> -> tensor<8x1x64xf32>, tensor<8x1x64xf32>, tensor<8x1x64xf32>
  // CHECK: results %[[reduce]]#0, %[[reduce]]#1, %[[reduce]]#2 : tensor<8x1x64xf32>, tensor<8x1x64xf32>, tensor<8x1x64xf32>
  results %reduce#0, %reduce#1, %reduce#2 : tensor<8x1x64xf32>, tensor<8x1x64xf32>, tensor<8x1x64xf32>
}
// -----

// Reshape operations
// CHECK-LABEL: nv_tensor_ir.graph @reshapeTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<2x3xf32>
nv_tensor_ir.graph @reshapeTestGraph(
  %input: tensor<2x3xf32>
) -> (tensor<6x1xf32>) {
  // CHECK: %[[reshape:.*]] = reshape %[[arg0]] : tensor<2x3xf32> -> tensor<6x1xf32>
  %reshaped = reshape %input : tensor<2x3xf32> -> tensor<6x1xf32>
  results %reshaped : tensor<6x1xf32>
}

// -----

// Reshape operations with dynamic dimensions
// CHECK-LABEL: nv_tensor_ir.graph @reshapeDynamicTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<?x?xf32>, %[[size:.*]]: index
nv_tensor_ir.graph @reshapeDynamicTestGraph(
  %input: tensor<?x?xf32>, %size: index
) -> (tensor<?x1xf32>) {
  // CHECK: %[[reshape:.*]] = reshape %[[arg0]] dynamic_dims(%[[size]]) : tensor<?x?xf32> -> tensor<?x1xf32>
  %reshaped = reshape %input dynamic_dims(%size) : tensor<?x?xf32> -> tensor<?x1xf32>
  results %reshaped : tensor<?x1xf32>
}

// -----

// Reshape operations with 3D tensors
// CHECK-LABEL: nv_tensor_ir.graph @reshape3DTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<4x3x2xf32>
nv_tensor_ir.graph @reshape3DTestGraph(
  %input: tensor<4x3x2xf32>
) -> (tensor<8x3xf32>) {
  // CHECK: %[[reshape:.*]] = reshape %[[arg0]] : tensor<4x3x2xf32> -> tensor<8x3xf32>
  %reshaped = reshape %input : tensor<4x3x2xf32> -> tensor<8x3xf32>
  results %reshaped : tensor<8x3xf32>
}

// -----

// Reshape operations with integer element type
// CHECK-LABEL: nv_tensor_ir.graph @reshapeIntTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<2x4xsi32>
nv_tensor_ir.graph @reshapeIntTestGraph(
  %input: tensor<2x4xsi32>
) -> (tensor<8x1xsi32>) {
  // CHECK: %[[reshape:.*]] = reshape %[[arg0]] : tensor<2x4xsi32> -> tensor<8x1xsi32>
  %reshaped = reshape %input : tensor<2x4xsi32> -> tensor<8x1xsi32>
  results %reshaped : tensor<8x1xsi32>
}

// -----

// Reshape operations with mixed static and dynamic dimensions
// CHECK-LABEL: nv_tensor_ir.graph @reshapeMixedTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<?x4x?xf32>, %[[rows:.*]]: index, %[[cols:.*]]: index
nv_tensor_ir.graph @reshapeMixedTestGraph(
  %input: tensor<?x4x?xf32>, %rows: index, %cols: index
) -> (tensor<?x?xf32>) {
  // CHECK: %[[reshape:.*]] = reshape %[[arg0]] dynamic_dims(%[[rows]], %[[cols]]) : tensor<?x4x?xf32> -> tensor<?x?xf32>
  %reshaped = reshape %input dynamic_dims(%rows, %cols)
      : tensor<?x4x?xf32> -> tensor<?x?xf32>
  results %reshaped : tensor<?x?xf32>
}

// -----

// Transpose operations
// CHECK-LABEL: nv_tensor_ir.graph @transposeTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<2x3xf32>
nv_tensor_ir.graph @transposeTestGraph(
  %input: tensor<2x3xf32>
) -> (tensor<3x2xf32>) {
  // CHECK: %[[transpose:.*]] = transpose %[[arg0]] permutation = [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
  %transposed = transpose %input permutation = [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
  results %transposed : tensor<3x2xf32>
}

// -----

// Transpose operations with 3D tensors
// CHECK-LABEL: nv_tensor_ir.graph @transpose3DTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<2x3x4xf32>
nv_tensor_ir.graph @transpose3DTestGraph(
  %input: tensor<2x3x4xf32>
) -> (tensor<4x2x3xf32>) {
  // CHECK: %[[transpose:.*]] = transpose %[[arg0]] permutation = [2, 0, 1] : tensor<2x3x4xf32> -> tensor<4x2x3xf32>
  %transposed = transpose %input permutation = [2, 0, 1] : tensor<2x3x4xf32> -> tensor<4x2x3xf32>
  results %transposed : tensor<4x2x3xf32>
}

// -----

// Transpose operations with identity permutation
// CHECK-LABEL: nv_tensor_ir.graph @transposeIdentityTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<2x3x4xf32>
nv_tensor_ir.graph @transposeIdentityTestGraph(
  %input: tensor<2x3x4xf32>
) -> (tensor<2x3x4xf32>) {
  // CHECK: %[[transpose:.*]] = transpose %[[arg0]] permutation = [0, 1, 2] : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
  %transposed = transpose %input permutation = [0, 1, 2] : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
  results %transposed : tensor<2x3x4xf32>
}

// -----

// Transpose operations with dynamic dimensions
// CHECK-LABEL: nv_tensor_ir.graph @transposeDynamicTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<?x?xf32>
nv_tensor_ir.graph @transposeDynamicTestGraph(
  %input: tensor<?x?xf32>
) -> (tensor<?x?xf32>) {
  // CHECK: %[[transpose:.*]] = transpose %[[arg0]] permutation = [1, 0] : tensor<?x?xf32> -> tensor<?x?xf32>
  %transposed = transpose %input permutation = [1, 0] : tensor<?x?xf32> -> tensor<?x?xf32>
  results %transposed : tensor<?x?xf32>
}

// -----

// Transpose operations with integer element type
// CHECK-LABEL: nv_tensor_ir.graph @transposeIntTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<3x2xsi32>
nv_tensor_ir.graph @transposeIntTestGraph(
  %input: tensor<3x2xsi32>
) -> (tensor<2x3xsi32>) {
  // CHECK: %[[transpose:.*]] = transpose %[[arg0]] permutation = [1, 0] : tensor<3x2xsi32> -> tensor<2x3xsi32>
  %transposed = transpose %input permutation = [1, 0] : tensor<3x2xsi32> -> tensor<2x3xsi32>
  results %transposed : tensor<2x3xsi32>
}

// -----

// Transpose operations with mixed static and dynamic dimensions
// CHECK-LABEL: nv_tensor_ir.graph @transposeMixedTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<?x4x3xf32>
nv_tensor_ir.graph @transposeMixedTestGraph(
  %input: tensor<?x4x3xf32>
) -> (tensor<3x?x4xf32>) {
  // CHECK: %[[transpose:.*]] = transpose %[[arg0]] permutation = [2, 0, 1] : tensor<?x4x3xf32> -> tensor<3x?x4xf32>
  %transposed = transpose %input permutation = [2, 0, 1] : tensor<?x4x3xf32> -> tensor<3x?x4xf32>
  results %transposed : tensor<3x?x4xf32>
}

// -----

// Test cost attribute with valid values
// CHECK-LABEL: nv_tensor_ir.graph @costAttributeTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<32x32xf32> {nv_tensor_ir.cost = 1 : i32}
// CHECK-SAME: %[[arg1:.*]]: tensor<32x32xf32> {nv_tensor_ir.cost = 5 : i32}
// CHECK-SAME: -> (tensor<32x32xf32> {nv_tensor_ir.cost = 10 : i32}, tensor<32x32xf32> {nv_tensor_ir.cost = -1 : i32})
nv_tensor_ir.graph @costAttributeTestGraph(
  %arg0: tensor<32x32xf32> {nv_tensor_ir.cost = 1 : i32},
  %arg1: tensor<32x32xf32> {nv_tensor_ir.cost = 5 : i32}
) -> (tensor<32x32xf32> {nv_tensor_ir.cost = 10 : i32}, tensor<32x32xf32> {nv_tensor_ir.cost = -1 : i32}) {
  // CHECK: %[[add:.*]] = add %[[arg0]], %[[arg1]] : tensor<32x32xf32>
  %add = add %arg0, %arg1 : tensor<32x32xf32>
  // CHECK: %[[mul:.*]] = mul %[[arg0]], %[[arg1]] : tensor<32x32xf32>
  %mul = mul %arg0, %arg1 : tensor<32x32xf32>
  // CHECK: results %[[add]], %[[mul]]
  results %add, %mul : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----

// Slice operation (basic)
// CHECK-LABEL: nv_tensor_ir.graph @sliceBasicTest
// CHECK-SAME: %[[arg0:.*]]: tensor<4x6xf32>
nv_tensor_ir.graph @sliceBasicTest(
  %input: tensor<4x6xf32>
) -> (tensor<2x3xf32>) {
  // CHECK: %[[slice:.*]] = slice %[[arg0]] starts = [1, 2] limits = [3, 5] strides = [1, 1] : tensor<4x6xf32> -> tensor<2x3xf32>
  %s = slice %input starts = [1, 2] limits = [3, 5] strides = [1, 1]
       : tensor<4x6xf32> -> tensor<2x3xf32>
  results %s : tensor<2x3xf32>
}

// -----

// Slice operation with strides
// CHECK-LABEL: nv_tensor_ir.graph @sliceStrideTest
// CHECK-SAME: %[[arg0:.*]]: tensor<8x8xf32>
nv_tensor_ir.graph @sliceStrideTest(
  %input: tensor<8x8xf32>
) -> (tensor<4x2xf32>) {
  // CHECK: %[[slice:.*]] = slice %[[arg0]] starts = [0, 1] limits = [8, 7] strides = [2, 3] : tensor<8x8xf32> -> tensor<4x2xf32>
  %s = slice %input starts = [0, 1] limits = [8, 7] strides = [2, 3]
       : tensor<8x8xf32> -> tensor<4x2xf32>
  results %s : tensor<4x2xf32>
}

// -----

// Concatenate operations
// CHECK-LABEL: nv_tensor_ir.graph @concatenateTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<2x3xf32>, %[[arg1:.*]]: tensor<1x3xf32>
nv_tensor_ir.graph @concatenateTestGraph(
  %input1: tensor<2x3xf32>,
  %input2: tensor<1x3xf32>
) -> (tensor<3x3xf32>) {
  // CHECK: %[[concat:.*]] = concatenate %[[arg0]], %[[arg1]] dimension = 0 : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  %concat = concatenate %input1, %input2 dimension = 0
            : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  results %concat : tensor<3x3xf32>
}

// -----

// Concatenate along different dimension
// CHECK-LABEL: nv_tensor_ir.graph @concatenateDim1TestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<2x2xf32>, %[[arg1:.*]]: tensor<2x3xf32>, %[[arg2:.*]]: tensor<2x1xf32>
nv_tensor_ir.graph @concatenateDim1TestGraph(
  %input1: tensor<2x2xf32>,
  %input2: tensor<2x3xf32>,
  %input3: tensor<2x1xf32>
) -> (tensor<2x6xf32>) {
  // CHECK: %[[concat:.*]] = concatenate %[[arg0]], %[[arg1]], %[[arg2]] dimension = 1 : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x6xf32>
  %concat = concatenate %input1, %input2, %input3 dimension = 1
            : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x6xf32>
  results %concat : tensor<2x6xf32>
}

// -----

// Concatenate with dynamic shapes
// CHECK-LABEL: nv_tensor_ir.graph @concatenateDynamicTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<?x3xf32>, %[[arg1:.*]]: tensor<?x3xf32>
nv_tensor_ir.graph @concatenateDynamicTestGraph(
  %input1: tensor<?x3xf32>,
  %input2: tensor<?x3xf32>
) -> (tensor<?x3xf32>) {
  // CHECK: %[[concat:.*]] = concatenate %[[arg0]], %[[arg1]] dimension = 0 : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
  %concat = concatenate %input1, %input2 dimension = 0
            : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
  results %concat : tensor<?x3xf32>
}

// -----

// Concatenate with different element types (integers)
// CHECK-LABEL: nv_tensor_ir.graph @concatenateIntTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<2x4xsi32>, %[[arg1:.*]]: tensor<3x4xsi32>
nv_tensor_ir.graph @concatenateIntTestGraph(
  %input1: tensor<2x4xsi32>,
  %input2: tensor<3x4xsi32>
) -> (tensor<5x4xsi32>) {
  // CHECK: %[[concat:.*]] = concatenate %[[arg0]], %[[arg1]] dimension = 0 : (tensor<2x4xsi32>, tensor<3x4xsi32>) -> tensor<5x4xsi32>
  %concat = concatenate %input1, %input2 dimension = 0
            : (tensor<2x4xsi32>, tensor<3x4xsi32>) -> tensor<5x4xsi32>
  results %concat : tensor<5x4xsi32>
}

// -----

// Concatenate 3D tensors
// CHECK-LABEL: nv_tensor_ir.graph @concatenate3DTestGraph
// CHECK-SAME: %[[arg0:.*]]: tensor<2x3x4xf32>, %[[arg1:.*]]: tensor<2x1x4xf32>
nv_tensor_ir.graph @concatenate3DTestGraph(
  %input1: tensor<2x3x4xf32>,
  %input2: tensor<2x1x4xf32>
) -> (tensor<2x4x4xf32>) {
  // CHECK: %[[concat:.*]] = concatenate %[[arg0]], %[[arg1]] dimension = 1 : (tensor<2x3x4xf32>, tensor<2x1x4xf32>) -> tensor<2x4x4xf32>
  %concat = concatenate %input1, %input2 dimension = 1
            : (tensor<2x3x4xf32>, tensor<2x1x4xf32>) -> tensor<2x4x4xf32>
  results %concat : tensor<2x4x4xf32>
}

// -----

// Iota operation

// CHECK-LABEL: nv_tensor_ir.graph @iotaTestGraph1
// CHECK: %[[iota:.*]] = iota dimension = 0 : tensor<2x3xf32>
// CHECK: results %[[iota]] : tensor<2x3xf32>
nv_tensor_ir.graph @iotaTestGraph1() -> (tensor<2x3xf32>) {
  %iota = iota dimension = 0 : tensor<2x3xf32>
  results %iota : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @iotaTestGraph2
// CHECK-SAME: %[[arg0:.*]]: tensor<2x3xf32>
// CHECK: %[[iota:.*]] = iota dimension = 1 : tensor<2x3xf32>
// CHECK: %[[add:.*]] = add %[[arg0]], %[[iota]] : tensor<2x3xf32>
// CHECK: results %[[add]] : tensor<2x3xf32>
nv_tensor_ir.graph @iotaTestGraph2(%arg0 : tensor<2x3xf32>) -> (tensor<2x3xf32>) {
  %iota = iota dimension = 1 : tensor<2x3xf32>
  %add = add %arg0, %iota : tensor<2x3xf32>
  results %add : tensor<2x3xf32>
}

// -----

// Dynamic source and broadcast shapes use scalar SSA extents.
// CHECK-LABEL: nv_tensor_ir.graph @dynamicShapeOperands
// CHECK: %[[rows:.*]] = dim %{{.*}} dimension = 0 : tensor<?x1xf32>
// CHECK: %[[splat:.*]] = splat %{{.*}} dynamic_dims(%{{.*}}, %{{.*}}) : tensor<?x?xf32>
// CHECK: %[[broadcast:.*]] = broadcast %{{.*}} dynamic_dims(%{{.*}}, %{{.*}}) : tensor<?x1xf32> -> tensor<?x?xf32>
// CHECK: %[[iota:.*]] = iota dimension = 1 dynamic_dims(%{{.*}}, %{{.*}}) : tensor<?x?xf32>
nv_tensor_ir.graph @dynamicShapeOperands(
  %value: f32, %input: tensor<?x1xf32>, %cols: index
) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %rows = dim %input dimension = 0 : tensor<?x1xf32>
  %filled = splat %value dynamic_dims(%rows, %cols)
      : tensor<?x?xf32>
  %broadcasted = broadcast %input dynamic_dims(%rows, %cols) : tensor<?x1xf32> -> tensor<?x?xf32>
  %indices = iota dimension = 1 dynamic_dims(%rows, %cols) : tensor<?x?xf32>
  results %filled, %broadcasted, %indices
      : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @iotaTestSSANameOpAsmInterface
// CHECK: %iota_0{{.*}} = iota dimension = 0 : tensor<2x3xf32>
// CHECK: %iota_1{{.*}} = iota dimension = 1 : tensor<2x3xf32>
nv_tensor_ir.graph @iotaTestSSANameOpAsmInterface() -> (tensor<2x3xf32>) {
  %0 = iota dimension = 0 : tensor<2x3xf32>
  %1 = iota dimension = 1 : tensor<2x3xf32>
  %add = add %0, %1 : tensor<2x3xf32>
  results %add : tensor<2x3xf32>
}
