// RUN: tensor_ir-opt -materialize-default-strides \
// RUN:   -layout-propagation-annotation -layout-propagation-normalization \
// RUN:   -tile-analyzer -split-input-file \
// RUN:   -verify-diagnostics=only-expected %s

// Independent result branches with no shared nonconstant SSA value require
// separate TensorIR graphs.
// expected-error @+1 {{multi-output results have no compatible nonconstant common SSA ancestor; split the results into multiple TensorIR graphs}}
nv_tensor_ir.graph @no_common_ancestor(
    %x: tensor<8xf32>, %y: tensor<8xf32>
    ) -> (tensor<8xf32>, tensor<8xf32>) {
  %a = abs %x : tensor<8xf32>
  %b = neg %y : tensor<8xf32>
  results %a, %b : tensor<8xf32>, tensor<8xf32>
}

// -----

// Dynamic multi-output reconciliation is deliberately rejected until its
// launch-coordinate and projected-store ABI is defined.
// expected-error @+1 {{dynamic multi-output layout propagation is not yet supported}}
nv_tensor_ir.graph @dynamic_multi_output(
    %x: tensor<?x16xf32>
    ) -> (tensor<?x16xf32>, tensor<?x16xf32>) {
  %a = abs %x : tensor<?x16xf32>
  %b = neg %x : tensor<?x16xf32>
  results %a, %b : tensor<?x16xf32>, tensor<?x16xf32>
}

// -----

// A common SSA ancestor is insufficient when its domain cannot be represented
// by every result branch. Partial slice domains need either predicated stores
// or separate kernels and are rejected by this initial implementation.
// expected-error @+1 {{multi-output results have no compatible nonconstant common SSA ancestor; split the results into multiple TensorIR graphs}}
nv_tensor_ir.graph @incompatible_partial_domains(
    %x: tensor<8xf32>
    ) -> (tensor<4xf32>, tensor<4xf32>) {
  %left = slice %x starts = [0] limits = [4] strides = [1]
      : tensor<8xf32> -> tensor<4xf32>
  %right = slice %x starts = [4] limits = [8] strides = [1]
      : tensor<8xf32> -> tensor<4xf32>
  results %left, %right : tensor<4xf32>, tensor<4xf32>
}

// -----

// A concatenation needs a unit tile on its concatenated dimension, while the
// projected reduction result needs the entire same dimension in one CTA. A
// single kernel cannot satisfy both requirements.
// expected-error @+1 {{iteration-space dimension 1 has incompatible tile requirements: size 1 for an operation-local iteration space and full size 2 for a projected graph result; split the results into multiple TensorIR graphs}}
nv_tensor_ir.graph @incompatible_concat_and_projection(
    %x: tensor<8x4xf32>, %y: tensor<8x4xf32>
    ) -> (tensor<8x8xf32>, tensor<8x1xf32>) {
  %joined = concatenate %x, %y dimension = 1
      : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x8xf32>
  %sum = reduce(%joined)<dimensions = [1], reduction_mode = <add>>
      : tensor<8x8xf32> -> tensor<8x1xf32>
  results %joined, %sum : tensor<8x8xf32>, tensor<8x1xf32>
}
