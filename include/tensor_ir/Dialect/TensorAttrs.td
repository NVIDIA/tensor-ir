// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_ATTRS
#define TENSOR_ATTRS

include "tensor_ir/Dialect/TensorDialect.td"
include "tensor_ir/Dialect/TensorEnums.td"
include "tensor_ir/Dialect/TensorAttrInterfaces.td"
include "mlir/IR/TensorEncoding.td"

def ReductionModeAttr : EnumAttr<TensorIR_Dialect, ReductionMode, "reduction_mode"> {
  let assemblyFormat = "`<` $value `>`";
}


class TensorIRAttr<string name, string attrMnemonic, list<Trait> traits=[]> : AttrDef<TensorIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}


def TensorSourceAttr : TensorIRAttr<"TensorSource", "tensor_source",
    [DeclareAttrInterfaceMethods<LayoutSourceAttrInterface>]> {
  let summary = [{
    Represents a simple layout annotation for an input (source) tensor.
  }];
  let description = [{
    Represents a simple layout annotation for an input (source) tensor.

    `tensor_id` is the index of the input argument that this layout applies to.

    `offset` is the offset of the current layout in the input tensor. This is
    primarily used for slice-like operations.

    `layout` is the actual layout annotation for the Tensor. This represents
    how the tensor is laid out in the memory.

    `layout` should be a valid string representation of a tcutegen::Layout.

    `dynamicValueMapping` specifies the mapping from dynamic dimensions in the layout
    to their corresponding runtime values. For a layout with N dynamic dimensions
    (marked by `?` in either shape or strides), this must be a permutation of
    [0, 1, ..., N-1], indicating which runtime argument provides each dynamic
    value. The permutation allows flexible ordering when multiple tensors share
    dynamic dimensions in different orders. Empty for tensors with static shapes.

    Example: tensor_source<1, 0, "(?,?):(?,1)", [0, 1, 2]>
    - References block argument #1 (graph input index)
    - The element offset is 0 (slice op may update this)
    - Layout has rank 2 with shape=(?, ?), strides=(?, 1)
    - Three dynamic values: shape[0], shape[1], stride[0]
    - dynamicValueMapping=[0,1,2] means the sequential runtime values following the
      tensor pointer define the dynamic sizes (transpose op may update this)
  }];
  let parameters = (ins "int32_t":$tensor_id,
                        "int64_t":$offset,
                        StringRefParameter<"">:$layout,
                        OptionalArrayRefParameter<"int32_t">:$dynamicValueMapping);
  let assemblyFormat = "`<` $tensor_id `,` $offset `,` $layout (`,` `[` $dynamicValueMapping^ `]`)? `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Get the CuTe layout from the layout string.
    tcutegen::Layout getCuteLayout() const;
  }];
}

def CompositeSourceAttr : TensorIRAttr<"CompositeSource", "composite_source",
    [DeclareAttrInterfaceMethods<LayoutSourceAttrInterface>]> {
  let summary = "Represents a collection of layouts for a tensor.";
  let description = [{
    Represents a collection of layouts for a tensor.

    `sources` is a list of underlying source attributes, each of which are
    needed for the operation that this layout annotation is applied to.
  }];
  let parameters = (ins ArrayRefParameter<"LayoutSourceAttrInterface">:$sources);
  let assemblyFormat = "`<` $sources `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Get the number of sources.
    size_t size() const { return getSources().size(); }

    /// Get a specific source attribute.
    LayoutSourceAttrInterface getSource(size_t index) const { return getSources()[index]; }
  }];
}

def ConcatSourceAttr : TensorIRAttr<"ConcatSource", "concat_source",
    [DeclareAttrInterfaceMethods<LayoutSourceAttrInterface>]> {
  let summary = "Represents a concatenation of multiple layouts in a single dimension.";
  let description = [{
    Represents a concatenation of multiple layouts in a single dimension.

    `dimension` is the index of the concatenated dimension in the layout sources.
    Note that the layout rank might be different from the tensor rank, so this
    index is not the same as the op dimension.

    `sources` is a list of underlying source attributes concatenated in a single
    dimension with index `dimension`. The shapes of the sources should be equal,
    except for the concatenation dimension.

    `argumentIndex` is a list of argument indices corresponding to the sources
    (has the same rank, when present). This is only set in case some of the
    operands are removed by the slice operation.
  }];
  let parameters = (ins "int64_t":$dimension,
                        ArrayRefParameter<"LayoutSourceAttrInterface">:$sources,
                        OptionalArrayRefParameter<"int32_t">:$argumentIndex);
  let assemblyFormat = "`<` `dim` `=` $dimension `,` $sources (`,` `[` $argumentIndex^ `]`)? `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Get the number of sources.
    size_t size() const { return getSources().size(); }

    /// Get a specific source attribute.
    LayoutSourceAttrInterface getSource(size_t index) const { return getSources()[index]; }
  }];
}

def ReductionSourceAttr : TensorIRAttr<"ReductionSource", "reduction_source",
    [DeclareAttrInterfaceMethods<LayoutSourceAttrInterface>]> {
  let summary = "Represents a reduction view on the underlying source.";
  let description = [{
    Represents a reduction view on the underlying source.

    `source` is the underlying source attribute and `view` is a CuTe layout that
    maps the dimensions of the input to the output. The last dimension of the
    view is the reduction dimension.
  }];
  let parameters = (ins StringRefParameter<"">:$view, "LayoutSourceAttrInterface":$source);
  let assemblyFormat = "`<` $view `,` $source `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Get the CuTe layout from the view layout string.
    tcutegen::Layout getCuteLayout() const;

    /// Get the reduction size.
    int64_t getReductionSize() const;

    /// Get the reduction shape.
    SmallVector<int64_t> getReductionShape() const;
  }];
}

def MatmulSourceAttr : TensorIRAttr<"MatmulSource", "matmul_source",
    [DeclareAttrInterfaceMethods<LayoutSourceAttrInterface>]> {
  let summary = "Represents a matmul layout.";
  let description = [{
    Represents a matmul layout.

    `lhs` and `rhs` are the underlying source layouts, and `view` defines a
    logical view for [B, M, N, K] shape, where batch dimensions are optional.
    The shape of the layout matches the operation output shape, and the strides
    define the logical dimension type:
    - [M * N * K, inf) range: batch dimensions;
    - [N * K, M * N * K) range: LHS noncontracting dimension;
    - [K, N * K) range: RHS noncontracting dimension;
    - [1, K) range: contracting dimension.
  }];
  let parameters = (ins StringRefParameter<"">:$view,
                        "int64_t":$b, "int64_t":$m, "int64_t":$n, "int64_t":$k,
                        "LayoutSourceAttrInterface":$lhs,
                        "LayoutSourceAttrInterface":$rhs);
  let assemblyFormat = "`<` $view `,` $b `,` $m `,` $n `,` $k `,` $lhs `,` $rhs `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Get the CuTe layout from the view layout string.
    tcutegen::Layout getCuteLayout() const;

    /// Get the contracting shape.
    SmallVector<int64_t> getContractingShape() const;

    /// Get LHS dimension mapping.
    llvm::FailureOr<SmallVector<size_t>> getLhsDimensionMap() const;

    /// Get RHS dimension mapping.
    llvm::FailureOr<SmallVector<size_t>> getRhsDimensionMap() const;
  }];
}

def IterSpaceDimDomainsAttr : TensorIRAttr<"IterSpaceDimDomains", "iter_space_dim_domains"> {
  let summary = "Array of dimension domain states for the iteration space";
  let description = [{
    Represents the domain state of each dimension in the iteration space.
    Used to track whether each dimension's domain has been defined.
  }];

  let parameters = (ins ArrayRefParameter<"::mlir::nv_tensor_ir::DimState">:$value);
  let hasCustomAssemblyFormat = 1;
}

def TensorIR_FloatAttr : Attr<CPred<"::llvm::isa<::mlir::FloatAttr>($_self)">, "TensorIR floating-point attribute"> {
  let storageType = [{ ::mlir::FloatAttr }];
  let returnType = [{ ::llvm::APFloat }];
  let convertFromStorage = "$_self.getValue()";
  let constBuilderCall = "::mlir::FloatAttr::get($_builder.getF32Type(), $0)";
}

#endif // TENSOR_ATTRS