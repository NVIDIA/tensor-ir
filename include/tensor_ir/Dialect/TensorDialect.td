// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_DIALECT
#define TENSOR_IR_DIALECT

include "mlir/IR/OpBase.td"

def TensorIR_Dialect : Dialect {
  let name = "nv_tensor_ir";

  let cppNamespace = "::mlir::nv_tensor_ir";

  let summary = "Nvidia Tensor Graph IR Dialect";

  let description = [{
    The Tensor Graph IR Dialect models a graph of operations at
    a high level of abstraction operating on tensor objects. The
    design is taken from cuDNN graphs.

    Shape relationships use three named policies. An *inherited* result extent
    exactly preserves its source extent, including whether it is static or
    dynamic. An *equal merge* between operand extents accepts a dynamic extent
    as an unknown that may be refined by a static extent; two unequal static
    extents are incompatible. A *broadcast merge* additionally allows a unit
    extent to broadcast. When an operation derives a result from multiple
    operands, the declared result extent must equal the merged extent.
  }];

  let extraClassDeclaration = [{
    static StringRef getAlignmentAttrName() {
      return "nv_tensor_ir.alignment";
    }

    static StringRef getStrideAttrName() {
      return "nv_tensor_ir.stride";
    }

    static StringRef getCostAttrName() {
      return "nv_tensor_ir.cost";
    }

    static StringRef getAllowTMAAttrName() {
      return "nv_tensor_ir.allow_tma";
    }

    static StringRef getIterSpaceMapAttrName() {
      return "nv_tensor_ir.iter_space_map";
    }

    static StringRef getIterSpaceIdAttrName() {
      return "nv_tensor_ir.iter_space_id";
    }

    /// Unlike operations (which belong to exactly one iteration space),
    /// graph arguments can be referenced by ops across multiple spaces;
    /// this key carries that list, whereas `iter_space_id` is the
    /// single-valued per-op key.
    static StringRef getIterSpaceIdsAttrName() {
      return "nv_tensor_ir.iter_space_ids";
    }

    static StringRef getIterSpaceDimDomainsAttrName() {
      return "nv_tensor_ir.iter_space_dim_domains";
    }

    static StringRef getIterationSpaceAttrName() {
      return "iteration_space";
    }

    static StringRef getLayoutAttrName() {
      return "layout";
    }

    static StringRef getTileSizeAttrName() {
      return "tile_size";
    }

    static StringRef getTileCandidatesAttrName() {
      return "tile_candidates";
    }
  }];

  // Add this line when we define attributes in this dialect.
  let useDefaultAttributePrinterParser = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
}

#endif // TENSOR_IR_DIALECT
