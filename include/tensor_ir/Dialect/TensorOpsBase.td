// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSORIR_DIALECT_TENSOROPSBASE_TD
#define TENSORIR_DIALECT_TENSOROPSBASE_TD

include "tensor_ir/Dialect/TensorDialect.td"
include "tensor_ir/Dialect/TensorTypes.td"
include "tensor_ir/Dialect/TensorAttrs.td"
include "tensor_ir/Dialect/TensorInterfaces.td"

include "mlir/Dialect/GPU/IR/GPUBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

// Common TensorIR operation bases.

class TensorIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<TensorIR_Dialect, mnemonic, traits> {}

class TensorIR_PointWiseOpBase<string mnemonic, list<Trait> traits = []>
    : TensorIR_Op<
          mnemonic,
          !listconcat([IterationSpaceInfoInterface, Pure, OpAsmOpInterface,
                       PointwiseOpInterface, LayoutPropInterface], traits)> {
  let results = (outs TensorIR_TensorType:$output);
  let extraClassDeclaration = [{
    /// Uses operation name as the base for SSA result value names in the
    /// printed textual IR.
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), getOperation()->getName().stripDialect());
    }
  }];
}

class TensorIR_PointWiseOpUnary<string mnemonic, list<Trait> traits = []>
    : TensorIR_PointWiseOpBase<
          mnemonic,
          !listconcat([SameOperandsAndResultType, ElementwiseMappable], traits)> {
  let summary = "Perform " # mnemonic # " operation pointwise on a tensor";
  let description = [{
    This operation takes one tensor and returns a new tensor of the same
    shape.

    TODO: Document each operation.
    - the signedness aspect of division
    - Invalid value behavior for pow or mod
    - Overflow behavior.
    - NaN propagation, infinity handling?
  }];
  let arguments = (ins
    Arg<TensorIR_TensorType, "Input tensor of the operation">:$input
  );
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

class TensorIR_PointWiseOpBinary<string mnemonic, list<Trait> traits = []>
    : TensorIR_PointWiseOpBase<
          mnemonic,
          !listconcat([SameOperandsAndResultType, ElementwiseMappable], traits)> {
  let summary = "Perform " # mnemonic # " operation pointwise on two tensors";
  let description = [{
    This operation takes two tensors of the same shape and returns a new
    tensor of the same shape as the inputs.

    TODO: Document each operation.
    - the signedness aspect of division
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

class TensorIR_ActivationFwdWithBetaOp<
    string mnemonic, list<Trait> traits = [ResultsAreFloatLike]>
    : TensorIR_PointWiseOpBase<
          mnemonic, !listconcat([SameOperandsAndResultType], traits)> {
  let summary = "Perform " # mnemonic # " op.";
  let description = [{
    This operation takes a tensor and returns a new tensor of the same
    shape. The beta parameter controls the shape of the activation function.
    Beta can be any TensorIR floating-point value (f16, bf16, f32, f64).
  }];
  let arguments = (ins
    TensorIR_TensorType:$x,
    TensorIR_FloatAttr:$beta
  );
  let assemblyFormat =
      " `<` `beta` `=` $beta `>` $x attr-dict `:` type($output)";
}

class TensorIR_ActivationBwdWithBetaOp<
    string mnemonic, list<Trait> traits = [ResultsAreFloatLike]>
    : TensorIR_PointWiseOpBase<
          mnemonic, !listconcat([SameOperandsAndResultType], traits)> {
  let summary = "Perform " # mnemonic # " op.";
  let description = [{
    This operation takes two tensors and returns a new tensor of the same
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
  let assemblyFormat =
      " `<` `beta` `=` $beta `>` $x `,` $dL_by_dy attr-dict `:` "
      "type($output) ";
}

class TensorIR_PointWiseOpBoolean<string mnemonic, list<Trait> traits = []>
    : TensorIR_PointWiseOpBase<
          mnemonic,
          !listconcat(traits, [
            SameTypeOperands,
            TypesMatchWith<
                "result type has i1 element type and same shape as operands",
                "lhs", "output", "::llvm::cast<RankedTensorType>($_self).clone("
                "::mlir::IntegerType::get($_self.getContext(), 1))">
          ])> {
  let summary = "Compare two tensors using the boolean operator " # mnemonic;
  let description = [{
    This operation takes two tensors of the same shape and returns a new
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

#endif // TENSORIR_DIALECT_TENSOROPSBASE_TD
