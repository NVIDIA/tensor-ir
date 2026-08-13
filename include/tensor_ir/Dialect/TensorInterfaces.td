// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- TensorInterfaces.td - Tensor IR Op Interfaces ------*- tablegen -*-===//
//
// This file defines the operation interfaces for the Tensor IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_IR_DIALECT_TENSOR_INTERFACES_TD
#define TENSOR_IR_DIALECT_TENSOR_INTERFACES_TD

include "mlir/IR/OpBase.td"

def PointwiseOpInterface : OpInterface<"PointwiseOpInterface"> {
  let cppNamespace = "::mlir::nv_tensor_ir";
  let description = [{
    This interface is used to represent pointwise operations.
  }];
}

def DynamicDimsOpInterface : OpInterface<"DynamicDimsOpInterface"> {
  let cppNamespace = "::mlir::nv_tensor_ir";
  let description = [{
    Interface for operations that carry the runtime extents of their dynamic
    result dimensions in a variadic operand segment named `dynamic_sizes`.

    The segment contains one scalar SSA value for each dynamic result
    dimension, in dimension order. During migration from implicit descriptor
    metadata, implementations may omit the entire segment.
  }];
  let verify = [{
    return ::mlir::nv_tensor_ir::detail::verifyDynamicDimsOpInterface($_op);
  }];

  // Keep interface methods vertically aligned with the established style in
  // this file. clang-format otherwise collapses the newly added method list.
  // clang-format off
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return true if `operand` belongs to the dynamic-dimension operand
        segment. Operand identity is determined by position, so the same SSA
        value may also appear in another operand segment.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isDynamicDimOperand",
      /*args=*/(ins "::mlir::OpOperand &":$operand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (operand.getOwner() != $_op.getOperation())
          return false;
        auto dynamicDims = getDynamicDims();
        unsigned operandNumber = operand.getOperandNumber();
        unsigned begin = dynamicDims.getBeginOperandIndex();
        return operandNumber >= begin &&
               operandNumber < begin + dynamicDims.size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the dynamic-dimension operand segment.",
      /*retTy=*/"::mlir::Operation::operand_range",
      /*methodName=*/"getDynamicDims",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getDynamicSizes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the SSA value for dynamic dimension `dynamicDimIndex`, where the
        index is relative to the dynamic-dimension segment.
      }],
      /*retTy=*/"::mlir::FailureOr<::mlir::Value>",
      /*methodName=*/"getDynamicDim",
      /*args=*/(ins "unsigned":$dynamicDimIndex),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto dynamicDims = getDynamicDims();
        if (dynamicDimIndex >= dynamicDims.size())
          return ::mlir::failure();
        return dynamicDims[dynamicDimIndex];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return dimension `dimension` of result `resultNumber` as a static
        integer attribute or as its corresponding dynamic-dimension SSA value.
      }],
      /*retTy=*/"::mlir::FailureOr<::mlir::OpFoldResult>",
      /*methodName=*/"getDimensionSize",
      /*args=*/(ins "unsigned":$resultNumber, "unsigned":$dimension),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (resultNumber >= $_op->getNumResults())
          return ::mlir::failure();
        auto resultType = ::mlir::cast<::mlir::ShapedType>(
            $_op->getResult(resultNumber).getType());
        if (dimension >= static_cast<unsigned>(resultType.getRank()))
          return ::mlir::failure();

        int64_t size = resultType.getDimSize(dimension);
        if (!::mlir::ShapedType::isDynamic(size)) {
          auto attr = ::mlir::IntegerAttr::get(
              ::mlir::IndexType::get($_op->getContext()), size);
          return ::mlir::OpFoldResult(attr);
        }

        unsigned dynamicDimIndex = 0;
        for (unsigned i = 0; i < resultNumber; ++i) {
          auto precedingResultType = ::mlir::cast<::mlir::ShapedType>(
              $_op->getResult(i).getType());
          dynamicDimIndex += precedingResultType.getNumDynamicDims();
        }
        for (unsigned i = 0; i < dimension; ++i)
          dynamicDimIndex += resultType.isDynamicDim(i);
        auto dynamicDim = getDynamicDim(dynamicDimIndex);
        if (::mlir::failed(dynamicDim))
          return ::mlir::failure();
        return ::mlir::OpFoldResult(*dynamicDim);
      }]
    >,
  ];
  // clang-format on
}

def IterationSpaceInfoInterface : OpInterface<"IterationSpaceInfoInterface"> {
  let cppNamespace = "::mlir::nv_tensor_ir";
  let description = [{
    Interface that defines properties and relations between a Tensor IR
    operation and the iteration space of the graph.

    ITERATION SPACE MAPS
    ====================

    For example, iteration space maps are functions that describe how to map the
    symbolic dimensions of the graph iteration space to the specific dimensions of
    its input and output tensors. For example, in a 3-D matrix multiplication
    C[m,n] = A[m,k] * B[k,n]:
      * The graph iteration space dimensions are (m, n, k), in that order.
      * The iteration space map for input A is (m, n, k) -> (m, k)
      * The iteration space map for input B is (m, n, k) -> (k, n)
      * The iteration space map for output C is (m, n, k) -> (m, n)

    We leverage AffineMap to represent the iteration space map information. Using
    the AffineMap notation for the previous example:
      * The graph iteration space dimensions are (d0, d1, d2), in that order.
      * The iteration space map for input A is (d0, d1, d2) -> (d0, d2)
      * The iteration space map for input B is (d0, d1, d2) -> (d2, d1)
      * The iteration space map for output C is (d0, d1, d2) -> (d0, d1)

    ITERATION SPACE TRANSITIONS
    ===========================

    Each dimension in the iteration space has a domain that defines its
    iteration range (e.g., [0, dimSize)). In general, operations can be fused
    together in the same loop nest when their domains are compatible. However,
    when an operation redefines a dimension's domain (e.g., a 1 -> N broadcast
    or N -> 1 reduction), it may require the creation of a new iteration space
    and define a transition between the two iteration spaces.

    To identify these transitions, we track whether the domain of each
    dimension has been defined at each operation in the graph. A dimension's
    domain is *defined* when it is used for the first time with a non-unit
    size (unit dimensions are "redundant" and do not establish a domain).
    Therefore, each dimension may have one of the two states:

      * Undef: The dimension's domain has not been *defined* yet. The
        dimension has only been seen in the graph with unit size (1) so far.

      * Def: The domain has been *defined*. The dimension was used with a
        non-unit size at some point in the graph. Once Def, a dimension stays
        Def until it is redefined or dropped.

    An iteration space transition occurs when an operation redefines a
    previously defined dimension's domain. For example:

      * Broadcast expanding an Undef dimension: The broadcast establishes the
        domain for the first time for that dimension. No transition.

      * Broadcast expanding a Def dimension: The dimension had a defined
        domain which is now being redefined. This creates a new iteration
        space and defines a transition.

    Example 1: No transition (broadcast defines the domain for the first time)

      ```mlir
      nv_tensor_ir.graph @no_transition(
          %A: tensor<16x1x64xf32>
              { iter_space_id = 0, iter_space_dim_domains = [undef, undef, undef] },
          %B: tensor<16x32x64xf32>
              { iter_space_id = 0, iter_space_dim_domains = [undef, undef, undef] })
        -> tensor<16x32x64xf32> {

        // Broadcast defines dims 0,1,2 for the first time -> no transition.
        %0 = nv_tensor_ir.broadcast %A : ... -> tensor<16x32x64xf32>
             { iter_space_id = 0, iter_space_dim_domains = [def, def, def] }

        // Add stays in same iteration space (id=0)
        %1 = nv_tensor_ir.add %0, %B : ... -> tensor<16x32x64xf32>
             { iter_space_id = 0, iter_space_dim_domains = [def, def, def] }
        nv_tensor_ir.yield %1
      }
      ```

    Example 2: Reduce -> Add -> Broadcast (two iteration spaces, one
    transition)

      ```mlir
      nv_tensor_ir.graph @reduce_add_broadcast(
          %A: tensor<16x32x64xf32>
              { iter_space_id = 0, iter_space_dim_domains = [undef, undef, undef] },
          %B: tensor<16x1x64xf32>
              { iter_space_id = 0, iter_space_dim_domains = [undef, undef, undef] })
        -> tensor<16x48x64xf32> {

        // Reduce defines all dims (first use), contracts dim 1 which stays Def.
        %0 = nv_tensor_ir.reduce %A { axis = 1 } : ... -> tensor<16x1x64xf32>
             { iter_space_id = 0, iter_space_dim_domains = [def, def, def] }

        // Add stays in same iteration space (id=0), doesn't redefine any domain.
        %1 = nv_tensor_ir.add %0, %B : ... -> tensor<16x1x64xf32>
             { iter_space_id = 0, iter_space_dim_domains = [def, def, def] }

        // Broadcast redefines dim 1 (was Def, now redefined) -> TRANSITION
        %2 = nv_tensor_ir.broadcast %1 : ... -> tensor<16x48x64xf32>
             { iter_space_id = 1, iter_space_dim_domains = [def, def, def] }
        nv_tensor_ir.yield %2
      }
      ```
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the rank of the iteration space for the operation.
        Default implementation assumes the rank of the first operand contains
        all the iteration space dimensions.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumIterSpaceDims",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto operand = $_op->getOperand(0);
        Type operandType = operand.getType();
        assert(::llvm::isa<::mlir::nv_tensor_ir::TensorType>(operandType) &&
          "Invalid default implementation of getNumIterSpaceDims");

        auto outputType = ::llvm::cast<::mlir::nv_tensor_ir::TensorType>(operandType);
        return outputType.getRank();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Retrieve the iteration space map for a specific input operand, if available.
        Return an empty AffineMap, otherwise. Note that this operation should not
        directly compute or store the iteration space maps of input operands but
        just try to retrieve them from the actual input operands.
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"getInputIterSpaceMap",
      /*args=*/(ins "unsigned":$inputIdx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Value operand = $_op->getOperand(inputIdx);
        if (auto defOp = operand.getDefiningOp()) {
          if (auto iterSpaceOp = dyn_cast<IterationSpaceInfoInterface>(defOp)) {
            return iterSpaceOp.getOutputIterSpaceMap();
          }

          return ::mlir::AffineMap();
        }

        // Retrieve attribute from block argument.
        auto funcOp = ::llvm::cast<::mlir::FunctionOpInterface>($_op->getParentOp());
        auto blockArg = ::llvm::cast<::mlir::BlockArgument>(operand);
        auto mapAttr = funcOp.getArgAttr(blockArg.getArgNumber(),
                      ::mlir::nv_tensor_ir::TensorIRDialect::getIterSpaceMapAttrName());
        if (!mapAttr)
          return ::mlir::AffineMap();
        return ::llvm::cast<::mlir::AffineMapAttr>(mapAttr).getAffineMap();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Retrieve the iteration space map for the output operand, if available.
        Return an empty AffineMap, otherwise. Note that this operation should not
        directly compute or store the iteration space map of the output operand but
        just try to retrieve it from the operation.
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"getOutputIterSpaceMap",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto outputMapAttr = $_op->template getAttrOfType<::mlir::AffineMapAttr>(
          ::mlir::nv_tensor_ir::TensorIRDialect::getIterSpaceMapAttrName());
        if (!outputMapAttr)
          return ::mlir::AffineMap();
        return outputMapAttr.getAffineMap();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Compute the canonical iteration space map for a specific input operand. Return
        an empty AffineMap if the iteration space map cannot be computed.

        `iterSpaceRank` is the rank of the iteration space and must be the number
        of input dimensions in the resulting iteration space map. If it's not provided,
        getNumIterSpaceDims() is used, producing a map that is local to the dimensions
        of the operation.

        The default implementation returns an identity mapping between the iteration space
        dimensions and the result dimensions. The rank of the input operand must
        match the rank of the iteration space. Otherwise, the iteration space map
        cannot be computed.
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"computeCanonicalInputIterSpaceMap",
      /*args=*/(ins "unsigned":$inputIdx, "std::optional<unsigned>":$iterSpaceRank),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Use provided rank or fall back to the operation's iteration space rank.
        int numIterSpaceDims = iterSpaceRank.value_or($_op.getNumIterSpaceDims());

        Value operand = $_op->getOperand(inputIdx);
        int operandRank = ::llvm::cast<::mlir::nv_tensor_ir::TensorType>(operand.getType()).getRank();
        if (numIterSpaceDims != operandRank)
          return ::mlir::AffineMap();

        return ::mlir::AffineMap::getMultiDimIdentityMap(numIterSpaceDims, $_op->getContext());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Compute the canonical iteration space map for the output result. Return an
        empty AffineMap if the iteration space map cannot be computed.

        `iterSpaceRank` is the rank of the iteration space and must be the number
        of input dimensions in the resulting iteration space map. If it's not provided,
        getNumIterSpaceDims() is used, producing a map that is local to the dimensions
        of the operation.

        The default implementation returns an identity mapping between the iteration space
        dimensions and the result dimensions. The rank of the result must match the
        rank of the iteration space. Otherwise, the iteration space map cannot be
        computed.
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"computeCanonicalOutputIterSpaceMap",
      /*args=*/(ins "std::optional<unsigned>":$iterSpaceRank),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op->getNumResults() == 1 && "Multiple outputs are not supported");

        // Use provided rank or fall back to intrinsic iteration space rank.
        int numIterSpaceDims = iterSpaceRank.value_or($_op.getNumIterSpaceDims());

        Value result = $_op->getResult(0);
        auto resultType = ::llvm::cast<::mlir::nv_tensor_ir::TensorType>(result.getType());
        int resultRank = resultType.getRank();
        if (numIterSpaceDims != resultRank)
          return ::mlir::AffineMap();

        return ::mlir::AffineMap::getMultiDimIdentityMap(numIterSpaceDims, $_op->getContext());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Infer the iteration space map for a specific input operand from the
        output iteration space map. Return an empty AffineMap if the iteration
        space map cannot be inferred.

        The default implementation propagates the output iteration space map
        to the input operand unmodified, if the rank of the output result
        matches the rank of the input operand. Otherwise, the
        iteration space map cannot be inferred.
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"inferInputFromOutputIterSpaceMap",
      /*args=*/(ins "unsigned":$inputIndex),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op->getNumResults() == 1 && "Multiple outputs are not supported");

        auto outputMap = getOutputIterSpaceMap();
        if (!outputMap)
          return ::mlir::AffineMap();

        Value operand = $_op->getOperand(inputIndex);
        auto operandType = ::llvm::cast<::mlir::nv_tensor_ir::TensorType>(operand.getType());
        if (outputMap.getNumResults() != static_cast<unsigned>(operandType.getRank()))
          return ::mlir::AffineMap();

        return outputMap;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Infer the iteration space map for the output result from the iteration
        space maps of the input operands. Return an empty AffineMap if the
        iteration space map cannot be inferred.

        The default implementation propagates unmodified the first iteration space
        map found in one of the inputs
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"inferOutputFromInputIterSpaceMaps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        for (int i = 0, numOperands = $_op->getNumOperands(); i < numOperands; ++i) {
          auto inputMap = getInputIterSpaceMap(i);
          if (!inputMap)
            continue;

          return inputMap;
        }

        return ::mlir::AffineMap();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if this operation creates an iteration space transition.
        A transition occurs when an operation *redefines* a dimension's
        domain that was previously defined (Def state) — i.e. the op takes
        a Def dimension and assigns it a new, non-unit value.

        Contraction to a unit dimension is not a transition. Example:
        `reduce` collapses a dimension to size 1; the domain stays Def in
        the iteration-space tracking but the extent shrinks to unit, so
        reduce does NOT override this method. A subsequent `broadcast`
        that expands that unit dimension back to N, however, redefines
        the domain to a new non-unit extent — `broadcast` therefore DOES
        override this method to return true on the redefined dimension.

        The default implementation returns false (no transition), which
        is correct for pointwise and pure shape-preserving ops.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isIterationSpaceTransition",
      /*args=*/(ins "::llvm::ArrayRef<::mlir::nv_tensor_ir::DimState>":$inputDimStates),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Compute the canonical output dimension domain states for this operation.

        Returns dimension states for the operation's output shape. Non-unit
        dimensions are marked as Def, unit dimensions as Undef.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::nv_tensor_ir::DimState>",
      /*methodName=*/"getCanonicalOutputDimDomains",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(!$_op->getResults().empty() && "expected at least one result");
        auto outputType = ::mlir::dyn_cast<::mlir::nv_tensor_ir::TensorType>(
            $_op->getResult(0).getType());
        if (!outputType)
          return ::llvm::SmallVector<::mlir::nv_tensor_ir::DimState>{};
        auto outputShape = outputType.getShape();
        unsigned outputRank = outputShape.size();

        ::llvm::SmallVector<::mlir::nv_tensor_ir::DimState> outputDimStates(
            outputRank, ::mlir::nv_tensor_ir::DimState::Undef);
        for (unsigned i = 0; i < outputRank; ++i) {
          if (outputShape[i] > 1)
            outputDimStates[i] = ::mlir::nv_tensor_ir::DimState::Def;
        }
        return outputDimStates;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the iteration space ID for this operation.",
      /*retTy=*/"unsigned",
      /*methodName=*/"getIterSpaceId",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto attr = $_op->template getAttrOfType<::mlir::IntegerAttr>(
                TensorIRDialect::getIterSpaceIdAttrName());
        assert(attr && "expected iter_space_id attribute");
        return attr.getInt();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the dimension domain states for this operation.",
      /*retTy=*/"::llvm::SmallVector<::mlir::nv_tensor_ir::DimState>",
      /*methodName=*/"getIterSpaceDimDomains",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto attr = $_op->template getAttrOfType<IterSpaceDimDomainsAttr>(
            TensorIRDialect::getIterSpaceDimDomainsAttrName());
        assert(attr && "expected iter_space_dim_domains attribute");
        return ::llvm::SmallVector<DimState>(attr.getValue());
      }]
    >,
  ];
}

def LayoutPropInterface : OpInterface<"LayoutPropInterface"> {
  let cppNamespace = "::mlir::nv_tensor_ir";
  let description = [{
    Interface for operations that infer a result layout source from operand
    layout sources. Implementations may transform a single source or combine
    multiple sources.

    Layout-preserving operations can use the default implementation, which
    returns the first operand layout unchanged.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Infer the result layout source from `inputLayouts`, which contains one
        layout source per operand in operand order. Returns `nullptr` if the
        result layout cannot be represented.
      }],
      /*retTy=*/"::mlir::nv_tensor_ir::LayoutSourceAttrInterface",
      /*methodName=*/"inferResultLayout",
      /*args=*/(ins "::llvm::ArrayRef<::mlir::nv_tensor_ir::LayoutSourceAttrInterface>":$inputLayouts),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return inputLayouts[0];
      }]
    >,
  ];
}

def AnchorOpInterface : OpInterface<"AnchorOpInterface"> {
  let cppNamespace = "::mlir::nv_tensor_ir";
  let description = [{
    Interface for anchor operations (e.g., MatmulOp, ScaledMatmulOp, ConvFpropOp,
    ConvDgradOp, ConvWgradOp) that defines standard accessors for the primary
    operands A and B. This interface is used to identify the anchor operations
    in the graph.

    Semantic mapping:
      - Matmul ops: A → A, B → B (direct mapping)
      - Conv fprop ops: X → A, W → B, Y → C
      - Conv dgrad ops: Dy → A, W → B, Dx → C
      - Conv wgrad ops: Dy → A, X → B, Dw → C
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get A operand of the operation.
      }],
      /*retTy=*/"::mlir::TypedValue<::mlir::nv_tensor_ir::TensorType>",
      /*methodName=*/"getA",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*desc=*/[{
        Get B operand of the operation.
      }],
      /*retTy=*/"::mlir::TypedValue<::mlir::nv_tensor_ir::TensorType>",
      /*methodName=*/"getB",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*desc=*/[{
        Get C output of the operation.
      }],
      /*retTy=*/"::mlir::TypedValue<::mlir::nv_tensor_ir::TensorType>",
      /*methodName=*/"getC",
      /*args=*/(ins)
    >,
  ];
}

#endif // TENSOR_IR_DIALECT_TENSOR_INTERFACES_TD
