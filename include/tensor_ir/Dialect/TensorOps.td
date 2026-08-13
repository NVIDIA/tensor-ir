// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_OPS
#define TENSOR_OPS

include "tensor_ir/Dialect/TensorDialect.td"
include "tensor_ir/Dialect/TensorTypes.td"
include "tensor_ir/Dialect/TensorAttrs.td"
include "tensor_ir/Dialect/TensorInterfaces.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/GPU/IR/GPUBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// Op Base
//===----------------------------------------------------------------------===//

class TensorIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<TensorIR_Dialect, mnemonic, traits> {
}

def LaunchIndx : AnyTypeOf<[Index, I32, I64]>;

def TensorIR_TypedAttrArray
    : TypedArrayAttrBase<TypedAttrInterface, "array of typed attributes">;

//===----------------------------------------------------------------------===//
// Graph Operations
//===----------------------------------------------------------------------===//

def TensorIR_GraphOp : TensorIR_Op<"graph",
    [SingleBlock, SingleBlockImplicitTerminator<"ResultsOp">,
     FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface]> {
  let summary = "Define a flat graph of TensorIR operations";
  let description = [{
    This operation is modeling a unordered list of operations forming a
    graph. The body is a single block in a "graph region", that is the
    ordering of the nodes does not matter.

    Note: TensorIR is a flat graph model by design. There is no loop/for
    construct at this level; iteration is expressed at lower dialect levels
    (CudaTile) during lowering.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<ArrayAttr>:$arg_attrs,
    OptionalAttr<ArrayAttr>:$res_attrs
  );
  let regions = (region SizedRegion<1>:$graphBody);
  let extraClassDeclaration = [{
    TensorIRDialect *getDialect() {
      return cast<TensorIRDialect>(static_cast<OpState *>(this)->getOperation()->getDialect());
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//

    // This will filter the `nv_tensor_ir.` prefix in front of operations inside the
    // graph body.
    static StringRef getDefaultDialect() {
      return TensorIRDialect::getDialectNamespace();
    }

    /// Get a special name to use when printing block arguments.
    void getAsmBlockArgumentNames(Region &region, OpAsmSetValueNameFn set_name_fn);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    // Returns the region on the current operation that is callable. This may
    // return null in the case of an external callable object, e.g. an external
    // function.
    Region *getCallableRegion() {
      return isExternal() ? nullptr : &this->getRegion();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the result values (the operands of ResultsOp) of this function.
    OperandRange getResults() { return getBody()->getTerminator()->getOperands(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def TensorIR_ResultsOp : TensorIR_Op<"results",
  [Pure, HasParent<"GraphOp">, ReturnLike, Terminator]> {
  let summary = "Define the result values for a TensorIR Graph.";
  let description = [{
    The `results` operation capture the tensor results of the graph.
    The operation takes variable number of operands and produces no results.
  }];
  let arguments = (ins Variadic<TensorIR_TensorType>:$results);
  let assemblyFormat = "$results attr-dict-with-keyword ( `:` type($results)^)? ";
}

//===----------------------------------------------------------------------===//
// Pointwise Operations
//===----------------------------------------------------------------------===//

// Class for pointwise operations.
class TensorIR_PointWiseOpBase<string mnemonic, list<Trait> traits = []>
    : TensorIR_Op<mnemonic,
                  !listconcat(
                      [
                        IterationSpaceInfoInterface,
                        Pure, OpAsmOpInterface,
                        PointwiseOpInterface,
                        LayoutPropInterface
                      ],
                      traits)> {
  let results = (outs TensorIR_TensorType:$output);
  string commonExtraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), getOperation()->getName().stripDialect());
    }
   }];
  let extraClassDeclaration = commonExtraClassDeclaration;
}

class TensorIR_PointWiseOpUnary<string mnemonic, list<Trait> traits = []> :
    TensorIR_PointWiseOpBase<mnemonic,
        !listconcat([SameOperandsAndResultType, ElementwiseMappable], traits)> {
  let summary = "Perform " # mnemonic # " operation pointwise on a tensor";
  let description = [{
    This operation takes one tensor and return a new tensor of the same
    shape.

    TODO: Document each of the operation.
    - the signness aspect of division
    - Invalid value behavior for pow or mod
    - Overflow behavior.
    - NaN propagation, infinity handling?
  }];
  let arguments = (ins
    Arg<TensorIR_TensorType, "Input tensor of the operation">:$input
  );
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

class TensorIR_PointWiseOpBinary<string mnemonic, list<Trait> traits = []> :
    TensorIR_PointWiseOpBase<mnemonic,
        !listconcat([SameOperandsAndResultType, ElementwiseMappable], traits)> {
  let summary = "Perform " # mnemonic # " operation pointwise on two tensors";
  let description = [{
    This operation takes two tensors of the same shape and return a new
    tensor of the same shape as the inputs.

    TODO: Document each of the operation.
    - the signness aspect of division
    - Invalid value behavior for pow or mod
    - Overflow behavior.
    - NaN propagation, infinity handling?
  }];
  let arguments = (ins
    Arg<TensorIR_TensorType, "Left-hand side tensor of the operation">:$lhs,
    Arg<TensorIR_TensorType, "Right-hand side tensor of the operation">:$rhs
  );
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
}

// Binary Operations
def TensorIR_AddOp : TensorIR_PointWiseOpBinary<"add", [Commutative]>;
def TensorIR_SubOp : TensorIR_PointWiseOpBinary<"sub", []>;
def TensorIR_MulOp : TensorIR_PointWiseOpBinary<"mul", [Commutative]>;
def TensorIR_DivOp : TensorIR_PointWiseOpBinary<"div", []>;
def TensorIR_ModOp : TensorIR_PointWiseOpBinary<"mod", []>;
def TensorIR_RemOp : TensorIR_PointWiseOpBinary<"rem", []> {
  let summary = "Perform truncated remainder operation pointwise on two tensors";
  let description = [{
    Computes `lhs - trunc(lhs / rhs) * rhs` elementwise.
    The result has the sign of the dividend (`lhs`), unlike `nv_tensor_ir.mod`
    which follows floored-remainder semantics.
  }];
}
def TensorIR_MaxOp : TensorIR_PointWiseOpBinary<"max", [Commutative]> {
  let summary = "Compute the pointwise maximum of two tensors";
  let description = [{
    Computes the maximum of corresponding elements in `$lhs` and `$rhs`.
    For floating-point element types, NaNs propagate: if either operand is
    NaN, the result is NaN. Integer element types use their signed or unsigned
    ordering and are not affected by NaN semantics.
  }];
}
def TensorIR_MinOp : TensorIR_PointWiseOpBinary<"min", [Commutative]> {
  let summary = "Compute the pointwise minimum of two tensors";
  let description = [{
    Computes the minimum of corresponding elements in `$lhs` and `$rhs`.
    For floating-point element types, NaNs propagate: if either operand is
    NaN, the result is NaN. Integer element types use their signed or unsigned
    ordering and are not affected by NaN semantics.
  }];
}
def TensorIR_AddSquareOp : TensorIR_PointWiseOpBinary<"add_square", []>;
def TensorIR_GeluBwdOp : TensorIR_PointWiseOpBinary<"gelu_bwd", [ResultsAreFloatLike]>;

def TensorIR_SigmoidBwdOp : TensorIR_PointWiseOpBinary<"sigmoid_bwd", [ResultsAreFloatLike]>;
def TensorIR_TanhBwdOp : TensorIR_PointWiseOpBinary<"tanh_bwd", [ResultsAreFloatLike]>;
def TensorIR_GeluApproxTanhBwdOp : TensorIR_PointWiseOpBinary<"gelu_approx_tanh_bwd", [ResultsAreFloatLike]>;
def TensorIR_ReluBwdOp : TensorIR_PointWiseOpBinary<"relu_bwd", [ResultsAreFloatLike]>;

// Unary Operations
def TensorIR_AbsOp : TensorIR_PointWiseOpUnary<"abs", []>;
def TensorIR_CeilOp : TensorIR_PointWiseOpUnary<"ceil", [ResultsAreFloatLike]>;
def TensorIR_CosOp : TensorIR_PointWiseOpUnary<"cos", [ResultsAreFloatLike]>;
def TensorIR_ExpOp : TensorIR_PointWiseOpUnary<"exp", [ResultsAreFloatLike]>;
def TensorIR_FloorOp : TensorIR_PointWiseOpUnary<"floor", [ResultsAreFloatLike]>;
def TensorIR_LogOp : TensorIR_PointWiseOpUnary<"log", [ResultsAreFloatLike]>;
def TensorIR_NegOp : TensorIR_PointWiseOpUnary<"neg", [ResultsAreFloatLike]>;
def TensorIR_RsqrtOp : TensorIR_PointWiseOpUnary<"rsqrt", [ResultsAreFloatLike]>;
def TensorIR_SinOp : TensorIR_PointWiseOpUnary<"sin", [ResultsAreFloatLike]>;
def TensorIR_SqrtOp : TensorIR_PointWiseOpUnary<"sqrt", [ResultsAreFloatLike]>;
def TensorIR_TanOp : TensorIR_PointWiseOpUnary<"tan", [ResultsAreFloatLike]>;
def TensorIR_Atan2Op : TensorIR_PointWiseOpBinary<"atan2", [ResultsAreFloatLike]>;
def TensorIR_ErfOp : TensorIR_PointWiseOpUnary<"erf", [ResultsAreFloatLike]>;
def TensorIR_ReciprocalOp : TensorIR_PointWiseOpUnary<"reciprocal", [ResultsAreFloatLike]>;
def TensorIR_ReluFwdOp : TensorIR_PointWiseOpUnary<"relu_fwd", [ResultsAreFloatLike]> {
  let summary = "Apply the rectified linear unit pointwise";
  let description = [{
    Computes `max(0, input)` for each floating-point element. NaNs propagate:
    a NaN input produces a NaN result.
  }];
}
def TensorIR_TanhFwdOp : TensorIR_PointWiseOpUnary<"tanh_fwd", [ResultsAreFloatLike]>;
def TensorIR_SigmoidFwdOp : TensorIR_PointWiseOpUnary<"sigmoid_fwd", [ResultsAreFloatLike]>;
def TensorIR_GeluFwdOp : TensorIR_PointWiseOpUnary<"gelu_fwd", [ResultsAreFloatLike]>;
def TensorIR_GeluApproxTanhFwdOp : TensorIR_PointWiseOpUnary<"gelu_approx_tanh_fwd", [ResultsAreFloatLike]>;

def TensorIR_PowOp : TensorIR_PointWiseOpBase<"pow",
  [AllTypesMatch<["lhs", "output"]>, AllShapesMatch<["lhs", "rhs"]>]> {
  let summary = "Compute the power of a tensor by raising each element to an exponent";
  let description = [{
    This operation raises each element of the tensor $lhs to the power
    specified by the corresponding element in the tensor $rhs.

    This operation supports two cases:
    1. Both $lhs and $rhs have the same element type (both float or both integer)
    2. $lhs is a floating-point tensor and $rhs is an integer tensor

    Any other combination of element types is not supported.
  }];
  let arguments = (ins
    Arg<TensorIR_TensorType, "The base of the power operation">:$lhs,
    Arg<TensorIR_TensorType, "The exponent of the power operation">:$rhs
  );
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}

// Activation Operations
// Base class for activation forward ops with beta parameter (Softplus, Swish, ELU)
class TensorIR_ActivationFwdWithBetaOp<string mnemonic, list<Trait> traits = [ResultsAreFloatLike]> :
    TensorIR_PointWiseOpBase<mnemonic,
        !listconcat([SameOperandsAndResultType], traits)> {
  let summary = "Perform " # mnemonic # " op.";
  let description = [{
    This operation takes a tensor and return a new tensor of the same
    shape. The beta parameter controls the shape of the activation function.
    Beta can be any TensorIR floating-point value (f16, bf16, f32, f64).
  }];
  let arguments = (ins
    TensorIR_TensorType:$x,
    TensorIR_FloatAttr:$beta
  );
  let assemblyFormat = " `<` `beta` `=` $beta `>` $x attr-dict `:` type($output)";
}

def TensorIR_SoftplusFwdOp : TensorIR_ActivationFwdWithBetaOp<"softplus_fwd">;

def TensorIR_SwishFwdOp : TensorIR_ActivationFwdWithBetaOp<"swish_fwd">;

def TensorIR_EluFwdOp : TensorIR_ActivationFwdWithBetaOp<"elu_fwd">;

// Base class for activation backward ops with beta parameter (Softplus, Swish, ELU)
class TensorIR_ActivationBwdWithBetaOp<string mnemonic, list<Trait> traits = [ResultsAreFloatLike]> :
    TensorIR_PointWiseOpBase<mnemonic,
        !listconcat([SameOperandsAndResultType], traits)> {
  let summary = "Perform " # mnemonic # " op.";
  let description = [{
    This operation takes three tensors and returns a new tensor of the same
    shape as the inputs. The 'x' and 'dL_by_dy' tensors must have the same type.
    The 'beta' parameter controls the shape of the activation function.
    Beta can be any TensorIR floating-point value (f16, bf16, f32, f64).

    Computes the gradient of the corresponding forward activation function
    with respect to the input, scaled by the upstream gradient dL_by_dy.
  }];
  let arguments = (ins
    TensorIR_TensorType:$x,
    TensorIR_TensorType:$dL_by_dy,
    TensorIR_FloatAttr:$beta
  );
  let assemblyFormat = " `<` `beta` `=` $beta `>` $x `,` $dL_by_dy attr-dict `:` type($output) ";
}

def TensorIR_SoftplusBwdOp : TensorIR_ActivationBwdWithBetaOp<"softplus_bwd">;

def TensorIR_SwishBwdOp : TensorIR_ActivationBwdWithBetaOp<"swish_bwd">;

def TensorIR_EluBwdOp : TensorIR_ActivationBwdWithBetaOp<"elu_bwd">;

def TensorIR_BinarySelectOp : TensorIR_PointWiseOpBase<"binary_select",
  [AllTypesMatch<["lhs", "rhs", "output"]>,
  TypesMatchWith<
    "selector type has i1 element type and same shape as other operands",
    "lhs", "selector", "::llvm::cast<TensorType>($_self).clone("
    "::mlir::IntegerType::get($_self.getContext(), 1))">]> {
  let summary = "Use a selector tensor to select between two inputs.";
  let description = [{
    The boolean selector tensor selects pointwise elements of the
    $lhs and $rhs tensors. All shapes must match and the element
    types of the two inputs must match.
  }];
  let arguments = (ins
    Arg<TensorIR_TensorType, "Boolean tensor used for selection">:$selector,
    Arg<TensorIR_TensorType, "Left-hand side tensor of the select">:$lhs,
    Arg<TensorIR_TensorType, "Right-hand side tensor of the select">:$rhs
  );
  let results = (outs
    TensorIR_TensorType:$output
  );
  let assemblyFormat = "$selector `` `,` $lhs `` `,` $rhs attr-dict `:` type($lhs)";
}

def TensorIR_CmpOp : TensorIR_PointWiseOpBase<"cmp",
  [SameTypeOperands, TypesMatchWith<
    "result type has i1 element type and same shape as operands",
    "lhs", "output", "::llvm::cast<TensorType>($_self).clone("
    "::mlir::IntegerType::get($_self.getContext(), 1))">]> {
  let summary = "Compare two tensors and produce the result in a new tensor.";
  let description = [{
    This operation takes two tensors of the same shape and returns a new
    tensor of the same shape as the inputs with boolean element type.

    This operation can perform both integer and floating point comparisons
    based on the operand tensor element types. It takes either an IntegerComparator or
    FloatComparator attribute:

    - For integer tensor inputs, the comparator must be an IntegerComparator:
      "eq", "neq", "gt", "ge", "lt", "le"

    - For floating point tensor inputs, the comparator must be a FloatComparator:
      "ueq", "une", "ugt", "uge", "ult", "ule", "oeq", "one", "ogt", "oge", "olt", "ole"

    Examples:
    ```mlir
    // Integer comparison
    %result = nv_tensor_ir.cmp %lhs eq %rhs : tensor<10x10xsi32>

    // Float comparison
    %result = nv_tensor_ir.cmp %lhs oeq %rhs : tensor<10x10xf32>
    ```
  }];
  let arguments = (ins
    Comparator:$comparator,
    Arg<TensorIR_TensorType, "Left-hand side tensor of the operation">:$lhs,
    Arg<TensorIR_TensorType, "Right-hand side tensor of the operation">:$rhs
  );
  let results = (outs
    TensorIR_BoolTensor:$output
  );
  let assemblyFormat = "$lhs $comparator $rhs attr-dict `:` type($lhs)";
  let hasVerifier = 1;
}

// Boolean Operations

class TensorIR_PointWiseOpBoolean<string mnemonic, list<Trait> traits = []> :
    TensorIR_PointWiseOpBase<mnemonic,
        !listconcat(traits, [SameTypeOperands, TypesMatchWith<
    "result type has i1 element type and same shape as operands",
    "lhs", "output", "::llvm::cast<TensorType>($_self).clone("
    "::mlir::IntegerType::get($_self.getContext(), 1))">,
    ])> {
  let summary = "Compare two tensors using the boolean operator " # mnemonic;
  let description = [{
    This operation takes two tensors of the same shape and return a new
    boolean tensor of the same shape as the inputs.
  }];
  let arguments = (ins
    TensorIR_TensorType:$lhs,
    TensorIR_TensorType:$rhs
  );
  let results = (outs
    TensorIR_BoolTensor:$output
  );
  let assemblyFormat = "$lhs `` `,` $rhs attr-dict `:` type($lhs)";
}

def TensorIR_LogicalAndOp : TensorIR_PointWiseOpBoolean<"and", [Commutative]>;
def TensorIR_LogicalOrOp : TensorIR_PointWiseOpBoolean<"or", [Commutative]>;
def TensorIR_LogicalNotOp : TensorIR_PointWiseOpUnary<"not", []>;

def TensorIR_ConvertOp : TensorIR_PointWiseOpBase<"convert",
        [SameOperandsAndResultShape]> {
  let summary = "Convert operation";
  let description = [{
    Performs an element-wise conversion from one element type to another on
    `input` tensor, producing a `result` tensor. This function supports only implicit conversions
    and does not support (de)quantization.

    Example:
    ```mlir
    %result = nv_tensor_ir.convert %input : tensor<3xf32> -> tensor<3xf64>
    ```
  }];
  let arguments = (ins
    Arg<TensorIR_TensorType, "Input tensor of the operation">:$input
  );
  let builders = [
    OpBuilder<(ins "Value":$input, "Type":$resultElementTy)>];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Reduce Operations
//===----------------------------------------------------------------------===//

def TensorIR_ReduceOp
    : TensorIR_Op<
          "reduce", [Pure, OpAsmOpInterface, IterationSpaceInfoInterface,
                     AllRanksMatch<["input", "output"]>,
                     DeclareOpInterfaceMethods<
                         LayoutPropInterface, ["inferResultLayout"]>]> {
  let summary = "Reduce operator";
  let description = [{
    Reduce operation represents reducing values of a tensor in one or more dimensions.
    - `$input` could either be virtual or non-virtual.
    - `$dimensions` indicates the axis to be reduced.
    - `$reduction_mode` indicates the operation to be used by the reduce, e.g. max, min, sum, etc.

    For floating-point inputs, the built-in `min`, `max`, and `amax` modes
    propagate NaNs. If any element in a reduction slice is NaN, the result for
    that slice is NaN. `amax` first takes the absolute value of each element
    and then applies the same NaN-propagating maximum reduction.

    ```mlir
    %reduce = nv_tensor_ir.reduce(%input)<
        dimensions = [1],
        reduction_mode = <add> >
        : tensor<16x32x64xf32> -> tensor<16x1x64xf32>
    ```

    Iteration-space note: reduce collapses `$dimensions` to extent 1 but
    does NOT implement `isIterationSpaceTransition` — contraction to
    unit is not a domain redefinition per the interface contract (see
    `IterationSpaceInfoInterface.isIterationSpaceTransition`). A follow-on
    `broadcast` that expands the reduced dimension back to non-unit is
    what triggers a transition.

    Every non-reduced result dimension is inherited from the corresponding
    input dimension, including its static/dynamic type information. At
    runtime, inherited dynamic extents equal their input extents. A result may
    leave a statically known reduced extent dynamic when it crosses a graph
    signature that has not yet been shape-refined; that runtime extent is 1.
  }];
  let arguments = (ins
    TensorIR_TensorType:$input,
    ConfinedAttr<DenseI32ArrayAttr, [DenseArrayMinCt<1>]>:$dimensions,
    ReductionModeAttr:$reduction_mode
  );
  let results = (outs TensorIR_TensorType:$output);
  let assemblyFormat = [{`(`$input`)` `<`
                         `dimensions` `=` $dimensions `,`
                         `reduction_mode` `=` $reduction_mode `>`
                         attr-dict `:` type($input) `->` type($output)}];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def TensorIR_ReduceUDOp : TensorIR_Op<"reduce_ud",
  [Pure, SingleBlockImplicitTerminator<"YieldOp">, OpAsmOpInterface,
    DeclareOpInterfaceMethods<LayoutPropInterface, ["inferResultLayout"]>
  ]> {
  let summary = "Reduce operator for user defined implementation";
  let description = [{
    Reduce operation represents reducing values of a tensor in one or more dimensions.
    - `$input` could either be virtual or non-virtual.
    - `$dimensions` indicates the axis to be reduced.
    - `$identity` is required. It is a property of the _reduction functor_ and
      is used in parallel algorithms to ensure correct parallel reduction. For
      example, the identities of `min` and `max` are `+inf` and `-inf`,
      respectively.
    Need to define a region to implement reduction functionality additionally.
    ```mlir
    %reduce = nv_tensor_ir.reduce_ud(%inputs)<
        dimensions = [1],
        identity = [0.000000e+00 : f32]>
        (%lhs: f32, %rhs: f32) {
          %0 = arith.addf %lhs, %rhs: f32
          nv_tensor_ir.yield %0 : f32
        } : (tensor<16x32x64xf32>) -> (tensor<16x1x64xf32>)
    ```
    The following example demonstrates using the `nv_tensor_ir.reduce_ud` operation to
    implement Welford's online algorithm for computing mean and variance statistics
    along a reduction dimension.
    The operation takes three input tensors (input values, m2 values, and weights)
    and produces three output tensors (mean, m2, and weight) after reduction.
    `%arg0`, `%arg1`, `%arg2` are the accumulated values of `%mean`,`%m2`,`%weight` respectively.
    `%arg3`, `%arg4`, `%arg5` are the current values of `%mean`,`%m2`,`%weight` respectively.
    Each element in `$identity` is the identity value of `%mean`, `%m2`, and
    `%weight`, respectively.
    All inputs have the same shape. Each result has the same shape as its
    corresponding input except that reduced dimensions have extent 1.
    ```mlir
    %reduce0, %reduce1, %reduce2 = nv_tensor_ir.reduce_ud(%input, %m2, %weight)<
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
      nv_tensor_ir.yield %4, %9, %1 : f32, f32, f32
    } : tensor<8x32x64xf32>, tensor<8x32x64xf32>,
    tensor<8x32x64xf32> -> tensor<8x1x64xf32>,
    tensor<8x1x64xf32>, tensor<8x1x64xf32>
  }];
  let arguments = (ins Variadic<TensorIR_TensorType>:$inputs,
      ConfinedAttr<DenseI32ArrayAttr, [DenseArrayMinCt<1>]>:$dimensions,
      TensorIR_TypedAttrArray:$identity);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<TensorIR_TensorType>:$output);
  let assemblyFormat = [{
    `(` $inputs `)` `<`
    `dimensions` `=` $dimensions `,`
    `identity` `=` $identity `>`
    custom<ReductionBody>($body)
    attr-dict `:` type($inputs) `->` type($output)
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def TensorIR_YieldOp: TensorIR_Op<"yield",
                            [ParentOneOf<["ReduceUDOp"
                            ]>,
                            Pure, Terminator, ReturnLike]> {
  let summary = "Yield operation";

  let description = [{
    `nv_tensor_ir.yield` is a special terminator operation for blocks inside regions
    in `nv_tensor_ir` ops. It returns values to the immediately enclosing `nv_tensor_ir` op.

    The operation takes variable number of operands and produces no results.

    Example:

    ```mlir
    nv_tensor_ir.yield %f0, %f1 : f32, f32
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Matmul Operations
//===----------------------------------------------------------------------===//

def TensorIR_MatmulOp : TensorIR_Op<"matmul", [
    Pure, OpAsmOpInterface, AnchorOpInterface, IterationSpaceInfoInterface,
    AllRanksMatch<["a", "b", "c"]>,
    DeclareOpInterfaceMethods<LayoutPropInterface, ["inferResultLayout"]>
  ]> {
  let summary = "Matrix multiplication";
  let description = [{
    Matrix multiplication computes `C = A * B` when no accumulator is
    provided, or `C = Acc + A * B` when the optional accumulator is present.
    The matmul operation supports the following arguments:
    * a: First matrix of type `TensorIR_TensorType`
    * b: Second matrix of type `TensorIR_TensorType`
    * acc: Optional initial accumulator with the same type as `C`

    The multiplication is performed with the type of the inputs, before
    being converted to the data type of `C` and accumulated.

    The two innermost dimensions of tensors `A` and `B` of `TensorIR_TensorType`
    need to satisfy the condition that number of columns of `A` is equal to the
    number of rows of `B`

    The other outer dimensions specify a batch matrix multiplication.

    Result batch dimensions are the broadcasted shape of the corresponding
    dimensions of `A` and `B`. Unit dimensions broadcast to the other extent,
    including when that extent is dynamic. A dynamic dimension paired with a
    non-unit static dimension is constrained to broadcast to that static
    extent. The result `M` dimension is inherited exactly from `A`, the result
    `N` dimension is inherited exactly from `B`, and the two runtime `K`
    extents must be equal.

    Examples of valid dimensions for A, B are -
    A - [m, k] and B - [k, n] producing C - [m, n]
    A - [b, m, k] and B - [b, k, n] producing C - [b, m, n]
    A - [b0, b1, m, k] and B - [b0, b1, k, n] producing C - [b0, b1, m, n]

  }];
  let arguments = (ins
    // Operands
    Arg<TensorIR_TensorType, "First Tensor of Matmul">:$a,
    Arg<TensorIR_TensorType, "Second Tensor of Matmul">:$b,
    Arg<Optional<TensorIR_TensorType>,
        "Optional initial accumulator for Matmul">:$acc
  );
  let results = (outs
    Arg<TensorIR_TensorType, "Output of Matmul">:$c
  );
  let builders = [
    OpBuilder<(ins "Type":$c, "Value":$a, "Value":$b)>
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), getOperation()->getName().stripDialect());
    }


    //===------------------------------------------------------------------===//
    // IterationSpaceInfoInterface.
    //===------------------------------------------------------------------===//
    unsigned getNumIterSpaceDims();

    AffineMap computeCanonicalInputIterSpaceMap(unsigned inputIndex,
                                                std::optional<unsigned> iterSpaceRank);
    AffineMap computeCanonicalOutputIterSpaceMap(std::optional<unsigned> iterSpaceRank);

    AffineMap inferInputFromOutputIterSpaceMap(unsigned inputIndex);
    AffineMap inferOutputFromInputIterSpaceMaps();
  }];

  let assemblyFormat = [{
    `(` $a `,` $b `)`
     (`accum` `(` $acc^ `:` type($acc) `)`)?
     attr-dict `:`
     `(` type($a) `,` type($b) `)` `->` type($c)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tensor Manipulation Operations
//===----------------------------------------------------------------------===//

def TensorIR_DimOp : TensorIR_Op<"dim", [Pure, OpAsmOpInterface]> {
  let summary = "Return a tensor dimension extent";
  let description = [{
    Returns the runtime extent of `input` dimension `dimension` as an index
    scalar. This provides scalar SSA values for dynamic shape operands without
    packaging dimensions in a shape tensor.

    ```mlir
    %rows = nv_tensor_ir.dim %input dimension = 0 : tensor<?x32xf32>
    ```
  }];
  let arguments = (ins TensorIR_TensorType:$input, I64Attr:$dimension);
  let results = (outs Index:$size);
  let assemblyFormat =
      "$input `dimension` `=` $dimension attr-dict `:` type($input)";
  let hasVerifier = 1;
}

// SplatOp inherits from TensorIR_Op directly (not TensorIR_PointWiseOpBase)
// because it takes a scalar input, not a tensor. This requires overriding
// several IterationSpaceInfoInterface methods that assume tensor operands.
def TensorIR_SplatOp : TensorIR_PointWiseOpBase<"splat", [
    DynamicDimsOpInterface,
    DeclareOpInterfaceMethods<IterationSpaceInfoInterface, [
      "getNumIterSpaceDims",
      "computeCanonicalInputIterSpaceMap",
      "inferInputFromOutputIterSpaceMap",
      "inferOutputFromInputIterSpaceMaps",
      "getInputIterSpaceMap"
    ]>,
    TypesMatchWith<"input type matches data type of output",
                   "output", "input",
                   "::llvm::isa<nv_tensor_ir::TensorType>($_self) ? "
                   "::llvm::cast<nv_tensor_ir::TensorType>($_self).getElementType() : $_self">
  ]> {
  let summary = "Create tensor filled with a scalar value";
  let description = [{
    Creates a tensor where every element is the given scalar operand. The operand
    is required to be of integer/float type.

    `dynamic_dims` may supply one index scalar for every dynamic result
    dimension, in dimension order. It may be omitted for compatibility with
    producers that have not yet materialized explicit shape metadata. Static
    result dimensions do not have corresponding operands.

    ```mlir
    %alpha_tensor = nv_tensor_ir.splat %alpha
        dynamic_dims(%batch, %rows)
        : tensor<?x?x1xf16>
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat]>:$input,
                       Variadic<Index>:$dynamic_sizes);
  // Note: results already defined in TensorIR_PointWiseOpBase

  let assemblyFormat = [{
    $input
    (`dynamic_dims` `(` $dynamic_sizes^ `)`)?
    attr-dict `:` type($output)
  }];
  let hasCanonicalizer = 1;
}

def TensorIR_BroadcastOp : TensorIR_PointWiseOpBase<"broadcast", [
    DynamicDimsOpInterface,
    DeclareOpInterfaceMethods<LayoutPropInterface, ["inferResultLayout"]>,
    DeclareOpInterfaceMethods<IterationSpaceInfoInterface, ["isIterationSpaceTransition"]>
  ]> {
  let summary = "Tensor broadcast operation";
  let description = [{
    This operation broadcasts a tensor from a shape with dimensions of size 1 to
    a (maybe) larger shape. E.g. broadcasting from shape (?, 1, 1) to (?, ?, ?).

    Every broadcast dim must change from a literal `1` in the input to either a
    dynamic `?` or a literal `>1` in the output (the "shape=1 + dynamic stride"
    convention). The legacy "shape=? + stride=0" convention is no longer
    accepted. Non-broadcast dims must be either identical or both dynamic; a
    no-op broadcast (no dim changes) is also rejected — remove the op instead.

    `dynamic_dims` may supply one index scalar for every dynamic result
    dimension, in dimension order. It may be omitted during migration from
    implicit shape metadata.

    Example:
    ```mlir
    %result = nv_tensor_ir.broadcast %input
        dynamic_dims(%batch, %rows, %cols)
        : tensor<?x1x?xf16> -> tensor<?x?x?xf16>
    ```
  }];

  let arguments = (ins
    Arg<TensorIR_TensorType, "Input tensor to broadcast">:$input,
    Variadic<Index>:$dynamic_sizes
  );
  let results = (outs TensorIR_TensorType:$output);
  let assemblyFormat = [{
    $input
    (`dynamic_dims` `(` $dynamic_sizes^ `)`)?
    attr-dict `:` type($input) `->` type($output)
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def TensorIR_ConstantOp : TensorIR_Op<"constant", [
    Pure, ConstantLike, OpAsmOpInterface,
    DeclareOpInterfaceMethods<IterationSpaceInfoInterface, [
      "getNumIterSpaceDims",
      "computeCanonicalOutputIterSpaceMap"
    ]>,
    InferTypeOpAdaptor,
  ]> {
  let summary = "tensor constant operation";
  let description = [{
    Define a tensor ir level constant

    Example:
    ```mlir
    %const = nv_tensor_ir.constant dense<100> : tensor<16x32xui32>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs TensorIR_TensorLike:$result);

  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def TensorIR_ReshapeOp : TensorIR_Op<"reshape", [
    Pure,
    OpAsmOpInterface,
    DynamicDimsOpInterface,
    AllElementTypesMatch<["input", "output"]>,
    DeclareOpInterfaceMethods<LayoutPropInterface, ["inferResultLayout"]>
  ]> {
  let summary = "Reshape a tensor to a new shape";
  let description = [{
    The reshape operation takes an immutable tensor value and produces a tensor
    with the same element type and number of elements but a different shape.

    For a shape `S` of rank `n` and a coordinate `i`, define its canonical
    linear position as:

    ```text
    linear(S, i):
      position = 0
      for d = 0, ..., n - 1:
        position = position * S[d] + i[d]
      return position
    ```

    For input shape `S`, result shape `T`, input coordinate `i`, and result
    coordinate `j`, `result[j] == input[i]` when
    `linear(T, j) == linear(S, i)`. Therefore, reshape preserves the canonical
    flattened sequence, in which the last dimension varies fastest.

    For example, consider a `(2,3)` input whose rows contain `[a,b,c]` and
    `[d,e,f]`. Reshaping it to `(3,2)` produces these values under canonical
    row-major indexing:

    ```text
    [[a,b], [c,d], [e,f]]
    ```

    Example:
    ```mlir
    %result = nv_tensor_ir.reshape %input
              : tensor<2x3xf32> -> tensor<3x2xf32>
    ```

    `dynamic_dims` may supply one index scalar for every dynamic result
    dimension, in dimension order. It may be omitted during migration from
    implicit shape metadata.
  }];

  let arguments = (ins
    Arg<TensorIR_TensorType, "Input tensor to reshape">:$input,
    Variadic<Index>:$dynamic_sizes
  );
  let results = (outs
    Arg<TensorIR_TensorType, "Reshaped output tensor">:$output
  );

  let assemblyFormat = [{
    $input
    (`dynamic_dims` `(` $dynamic_sizes^ `)`)?
    attr-dict `:` type($input) `->` type($output)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def TensorIR_TransposeOp
    : TensorIR_Op<"transpose",
                  [Pure, OpAsmOpInterface, SameOperandsAndResultElementType,
                   AllRanksMatch<["input", "output"]>,
                   DeclareOpInterfaceMethods<
                       LayoutPropInterface, ["inferResultLayout"]>]> {
  let summary = "Transpose a tensor by permuting its dimensions";
  let description = [{
    The transpose operation takes a tensor and produces a new tensor with the same
    element type and permuted dimensions according to the specified permutation.
    The permutation is specified as an array of dimension indices that define how
    the input dimensions map to output dimensions.

    For a tensor with rank N, the permutation must be a valid permutation of
    indices [0, 1, ..., N-1]. The output shape is determined by applying the
    permutation to the input shape.
    More formally, for all indices i of the input,  output[permutation(i)] = input[i]

    Examples:
    ```mlir
    // Transpose a 2x3 matrix (swap rows and columns)
    %transposed = nv_tensor_ir.transpose %input permutation = [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>

    // Transpose a 3D tensor
    %transposed = nv_tensor_ir.transpose %input permutation = [2, 0, 1] : tensor<2x3x4xf32> -> tensor<4x2x3xf32>
    ```
  }];

  let arguments = (ins
    Arg<TensorIR_TensorType, "Input tensor to transpose">:$input,
    DenseI64ArrayAttr:$permutation
  );
  let results = (outs
    Arg<TensorIR_TensorType, "Transposed output tensor">:$output
  );

  let assemblyFormat = "$input `permutation` `=` $permutation attr-dict `:` type($input) `->` type($output)";
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def TensorIR_SliceOp
    : TensorIR_Op<
          "slice", [Pure, OpAsmOpInterface, SameOperandsAndResultElementType,
                    AllRanksMatch<["input", "output"]>,
                    DeclareOpInterfaceMethods<
                        LayoutPropInterface, ["inferResultLayout"]>]> {
  let summary = "Extract a slice from a tensor";
  let description = [{
    The slice operation extracts a subtensor from the input tensor using
    per-dimension start indices, limit indices (exclusive), and strides. Since
    these are static attributes, the result shape is always statically known.
    When an input dimension is dynamic, satisfying the static slice bounds is
    a runtime precondition.

    For a tensor with rank N, `starts`, `limits`, and `strides` must each be
    arrays of length N. All offsets and sizes must be non-negative, and all
    strides must be positive. For each dimension i the output extent is
    computed as
    1 + floor((limit[i] - 1 - start[i]) / stride[i]) if limit[i] > start[i],
    else 0, and must match the output shape.

    Examples:
    ```mlir
    // Slice rows [1, 3) and cols [2, 5) with unit strides -> shape (2, 3)
    %s = nv_tensor_ir.slice %input starts = [1, 2] limits = [3, 5] strides = [1, 1]
         : tensor<4x6xf32> -> tensor<2x3xf32>
    ```
  }];

  let arguments = (ins
    Arg<TensorIR_TensorType, "Input tensor to slice">:$input,
    ConfinedAttr<DenseI64ArrayAttr,
                 [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$starts,
    ConfinedAttr<DenseI64ArrayAttr,
                 [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$limits,
    ConfinedAttr<DenseI64ArrayAttr,
                 [DenseArrayStrictlyPositive<DenseI64ArrayAttr>]>:$strides
  );

  let results = (outs
    Arg<TensorIR_TensorType, "Sliced output tensor">:$output
  );

  let assemblyFormat = "$input `starts` `=` $starts `limits` `=` $limits `strides` `=` $strides attr-dict `:` type($input) `->` type($output)";
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def TensorIR_ConcatenateOp : TensorIR_Op<"concatenate", [
    Pure,
    OpAsmOpInterface,
    SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<LayoutPropInterface, ["inferResultLayout"]>
  ]> {
  let summary = "Concatenate tensors along a specified dimension";
  let description = [{
    The concatenate operation combines multiple tensors along a specified dimension.
    All input tensors must have the same element type and the same shape except
    along the concatenation dimension. The result tensor has the same element type
    and shape as the inputs, except the concatenation dimension size is the sum
    of all input tensor sizes along that dimension.

    If every concatenation extent is static, the result concatenation extent
    is their static sum. Otherwise it is dynamic and its runtime extent is the
    sum of the runtime input extents. Every non-concatenation result extent is
    inherited unchanged from the inputs.

    The dimension attribute specifies which dimension to concatenate along, using
    zero-based indexing. For tensors with rank N, the dimension must be in the
    range [0, N-1].

    Examples:
    ```mlir
    // Concatenate along dimension 0 (rows)
    %result = nv_tensor_ir.concatenate %input1, %input2 dimension = 0
              : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>

    // Concatenate along dimension 1 (columns)
    %result = nv_tensor_ir.concatenate %input1, %input2, %input3 dimension = 1
              : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x6xf32>
    ```
  }];

  let arguments = (ins
    Variadic<TensorIR_TensorType>:$inputs,
    I64Attr:$dimension
  );

  let results = (outs
    TensorIR_TensorType:$output
  );

  let assemblyFormat = "$inputs `dimension` `=` $dimension attr-dict `:` functional-type(operands, results)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def TensorIR_IotaOp : TensorIR_Op<"iota", [
    Pure,
    DynamicDimsOpInterface,
    OpAsmOpInterface]> {
  let summary = [{
    Builds a constant literal on device, incrementing by 1 along dimension.
  }];
  let description = [{
    Builds a constant literal on device rather than a potentially large host
    transfer. Creates an array that has specified shape and holds values
    starting at zero and incrementing by one along the specified dimension. For
    floating-point types, the produced array is equivalent to
    ConvertElementType(Iota(...)) where the Iota is of integral type and the
    conversion is to the floating-point type.

    For example, Iota(s32[4, 8], 0) returns
    ```
    [[0, 0, 0, 0, 0, 0, 0, 0 ],
     [1, 1, 1, 1, 1, 1, 1, 1 ],
     [2, 2, 2, 2, 2, 2, 2, 2 ],
     [3, 3, 3, 3, 3, 3, 3, 3 ]]
    ```
    Iota(s32[4, 8], 1) returns
    ```
    [[0, 1, 2, 3, 4, 5, 6, 7 ],
     [0, 1, 2, 3, 4, 5, 6, 7 ],
     [0, 1, 2, 3, 4, 5, 6, 7 ],
     [0, 1, 2, 3, 4, 5, 6, 7 ]]
    ```

    `dynamic_dims` may supply one index scalar for every dynamic result
    dimension, in dimension order. It may be omitted during migration from
    implicit shape metadata.
  }];

  let arguments = (ins I64Attr:$dimension, Variadic<Index>:$dynamic_sizes);
  let results = (outs TensorIR_TensorType:$output);

  let assemblyFormat = [{
    `dimension` `=` $dimension
    (`dynamic_dims` `(` $dynamic_sizes^ `)`)?
    attr-dict `:` type($output)
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(
        getResult(),
        "iota_" + std::to_string(static_cast<int64_t>(getDimension())));
    }
  }];

}


#endif // TENSOR_OPS
