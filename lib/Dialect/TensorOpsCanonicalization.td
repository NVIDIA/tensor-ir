// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_OPS_CANONICALIZATION
#define TENSOR_IR_OPS_CANONICALIZATION

include "mlir/IR/PatternBase.td"
include "tensor_ir/Dialect/TensorOps.td"

//===----------------------------------------------------------------------===//
// Native constraints and builders
//===----------------------------------------------------------------------===//

def HasStaticShape : Constraint<CPred<"hasStaticShape($0)">,
                                "has a static ranked tensor shape">;

def HasNoLayoutAnnotation
    : Constraint<CPred<"!hasLayoutAnnotation($0.getDefiningOp())">,
                 "has no layout annotation">;

def IsLosslessConvert
    : Constraint<CPred<"isLosslessConvert($0, $1)">,
                 "the conversion preserves every source value">;

def CanComposeSlices
    : Constraint<
          CPred<"canComposeSlices($0)">,
          "the slice attributes can be composed without integer overflow">;

def BuildSplatElementsAttr : NativeCodeCall<"buildSplatElementsAttr($0, $1)">;
def ComposePermutations
    : NativeCodeCall<"composePermutations($_builder, $0, $1)">;
def ComposeSliceStarts
    : NativeCodeCall<"composeSliceStarts($_builder, $0, $1, $2)">;
def ComposeSliceLimits
    : NativeCodeCall<"composeSliceLimits($_builder, $0, $1, $2, $3, $4, $5)">;
def ComposeSliceStrides
    : NativeCodeCall<"composeSliceStrides($_builder, $0, $1)">;

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// broadcast(broadcast(x)) -> broadcast(x)
// Keep the outer dynamic_dims; they describe the final result shape.
def MergeSequentialBroadcasts
    : Pat<(TensorIR_BroadcastOp:$outer
              (TensorIR_BroadcastOp:$inner $input, $innerSizes),
              $outerSizes),
          (TensorIR_BroadcastOp $input, $outerSizes),
          [(HasNoLayoutAnnotation $inner), (HasNoLayoutAnnotation $outer)]>;

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

// convert(convert(x)) -> convert(x) when the first conversion is lossless.
def MergeSequentialConverts
    : Pat<(TensorIR_ConvertOp:$outer(TensorIR_ConvertOp:$inner $input)),
          (TensorIR_ConvertOp $input),
          [(HasNoLayoutAnnotation $inner), (HasNoLayoutAnnotation $outer),
           (IsLosslessConvert $input, $inner)]>;

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

// splat(constant) -> constant
// Static results have an empty dynamic_sizes range.
def SplatConstant
    : Pat<(TensorIR_SplatOp:$output(TensorIR_ConstantOp $value), $dynamicSizes),
          (TensorIR_ConstantOp(BuildSplatElementsAttr $output, $value)),
          [(HasNoLayoutAnnotation $output), (HasStaticShape $output)]>;

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// transpose(transpose(x, p1), p2) -> transpose(x, compose(p1, p2))
def MergeSequentialTransposes
    : Pat<(TensorIR_TransposeOp:$outer
              (TensorIR_TransposeOp:$inner $input, $innerPermutation),
              $outerPermutation),
          (TensorIR_TransposeOp $input, (ComposePermutations $innerPermutation,
                                            $outerPermutation)),
          [(HasNoLayoutAnnotation $inner), (HasNoLayoutAnnotation $outer)]>;

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// slice(slice(x, s1, l1, t1), s2, l2, t2) -> slice(x, s, l, t)
def MergeSequentialSlices
    : Pat<(TensorIR_SliceOp:$outer(TensorIR_SliceOp:$inner $input, $innerStarts,
               $innerLimits, $innerStrides),
              $outerStarts, $outerLimits, $outerStrides),
          (TensorIR_SliceOp $input,
              (ComposeSliceStarts $innerStarts, $innerStrides, $outerStarts),
              (ComposeSliceLimits $innerStarts, $innerStrides, $outerStarts,
                  $outerLimits, $outerStrides, $innerLimits),
              (ComposeSliceStrides $innerStrides, $outerStrides)),
          [(HasNoLayoutAnnotation $inner), (HasNoLayoutAnnotation $outer),
           (CanComposeSlices $outer)]>;

#endif // TENSOR_IR_OPS_CANONICALIZATION
