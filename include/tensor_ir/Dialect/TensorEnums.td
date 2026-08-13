// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSORIR_ENUMS
#define TENSORIR_ENUMS

include "mlir/IR/EnumAttr.td"
include "tensor_ir/Dialect/TensorDialect.td"

//
// Comparators
//
// Integer comparators (total order):
def EQ : I32EnumAttrCase<"eq", 0>;   // equal
def NEQ : I32EnumAttrCase<"neq", 1>; // not equal
def GT : I32EnumAttrCase<"gt", 2>;   // greater than
def GE : I32EnumAttrCase<"ge", 3>;   // greater or equal
def LT : I32EnumAttrCase<"lt", 4>;   // less than
def LE : I32EnumAttrCase<"le", 5>;   // less or equal
// Float comparators follow LLVM's IEEE-754 fcmp semantics. The `u*` family
// is UNORDERED — the result is true if either operand is NaN OR the
// underlying comparison holds. The `o*` family is ORDERED — the result is
// true only if neither operand is NaN AND the underlying comparison holds.
def UEQ : I32EnumAttrCase<"ueq", 6>;   // unordered or equal
def UNE : I32EnumAttrCase<"une", 7>;   // unordered or not equal
def UGT : I32EnumAttrCase<"ugt", 8>;   // unordered or greater than
def UGE : I32EnumAttrCase<"uge", 9>;   // unordered or greater or equal
def ULT : I32EnumAttrCase<"ult", 10>;  // unordered or less than
def ULE : I32EnumAttrCase<"ule", 11>;  // unordered or less or equal
def OEQ : I32EnumAttrCase<"oeq", 12>;  // ordered and equal
def ONE : I32EnumAttrCase<"one", 13>;  // ordered and not equal
def OGT : I32EnumAttrCase<"ogt", 14>;  // ordered and greater than
def OGE : I32EnumAttrCase<"oge", 15>;  // ordered and greater or equal
def OLT : I32EnumAttrCase<"olt", 16>;  // ordered and less than
def OLE : I32EnumAttrCase<"ole", 17>;  // ordered and less or equal

def Comparator: I32EnumAttr<
    "Comparator",
    "Comparator enum",
    [EQ, NEQ, GT, GE, LT, LE, UEQ, UNE, UGT, UGE, ULT, ULE, OEQ, ONE, OGT, OGE, OLT, OLE]> {
  let cppNamespace = TensorIR_Dialect.cppNamespace;
  let genSpecializedAttr = 1;
  let description = [{
    This enum describes the comparator that can be assigned
    to the "cmp" operation.
    Enum cases: }]
     # !interleave(!foreach(case, enumerants, " " # case.str), ",")
     # "\n";
}

//
// ReductionMode
//
// `customize` indicates a user-defined reduction; the per-element combiner
// is carried by the `reduce_ud` op's region body rather than selected from
// this enum. The other cases are built-in reductions with fixed semantics
// (see the case names: add/mul/min/max sums/products/extrema, amax = max of
// |x|, avg = mean, norm1 = sum |x|, norm2 = sqrt(sum x^2), mul_no_zeros =
// product ignoring zero inputs).
def CUSTOMIZE : I32EnumAttrCase<"customize", 0>;
def ADD : I32EnumAttrCase<"add", 1>;
def MUL : I32EnumAttrCase<"mul", 2>;
def MIN : I32EnumAttrCase<"min", 3>;
def MAX : I32EnumAttrCase<"max", 4>;
def AMAX : I32EnumAttrCase<"amax", 5>;
def AVG : I32EnumAttrCase<"avg", 6>;
def NORM1 : I32EnumAttrCase<"norm1", 7>;
def NORM2 : I32EnumAttrCase<"norm2", 8>;
def MUL_NO_ZEROS : I32EnumAttrCase<"mul_no_zeros", 9>;

def ReductionMode: I32EnumAttr<
    "ReductionMode",
    "ReductionMode enum",
    [CUSTOMIZE, ADD, MUL, MIN, MAX, AMAX, AVG, NORM1, NORM2, MUL_NO_ZEROS]> {
  let cppNamespace = TensorIR_Dialect.cppNamespace;
  let genSpecializedAttr = 0;
  let description = [{
    This enum describes the ReductionMode that can be assigned
    to the "reduction" operation.
    Enum cases: }]
     # !interleave(!foreach(case, enumerants, " " # case.str), ",")
     # "\n";
}

def ComputeCapabilityEnum : I32EnumAttr<
  "ComputeCapability",
  "Target GPU compute capability",
  [
    I32EnumAttrCase<"Sm80",              80,  "sm_80">,
    I32EnumAttrCase<"Sm86",              86,  "sm_86">,
    I32EnumAttrCase<"Sm87",              87,  "sm_87">,
    I32EnumAttrCase<"Sm88",              88,  "sm_88">,
    I32EnumAttrCase<"Sm89",              89,  "sm_89">,
    I32EnumAttrCase<"Sm90",              90,  "sm_90">,
    I32EnumAttrCase<"Sm100",             100, "sm_100">,
    I32EnumAttrCase<"Sm103",             103, "sm_103">,
    I32EnumAttrCase<"Sm110",             110, "sm_110">,
    I32EnumAttrCase<"Sm120",             120, "sm_120">,
    I32EnumAttrCase<"Sm121",             121, "sm_121">,
  ]> {
  let description = [{
    Closed TensorIR compute capability set used for canonical target identity and
    chip-string round-tripping.
  }];

  let cppNamespace = TensorIR_Dialect.cppNamespace;
  let genSpecializedAttr = 0;
}

def ArchPortabilityEnum : I32EnumAttr<
  "ArchPortability",
  "Arch portability flag",
  [
    I32EnumAttrCase<"portable",             0>,
    I32EnumAttrCase<"family_portable",      1>,
    I32EnumAttrCase<"arch_conditional",     2>,
  ]> {
  let description = [{
    - portable          : Compilation target is portable to future archs        (eg. sm100 )
    - family_portable   : Compilation target is portable within the arch family (eg. sm100f)
    - arch_conditional  : Compilation target is not portable to any other archs (eg. sm100a)
  }];

  let cppNamespace = TensorIR_Dialect.cppNamespace;
  let genSpecializedAttr = 0;
}


//===----------------------------------------------------------------------===//
// Iteration Space Dimension Domain State
//===----------------------------------------------------------------------===//
//
// States to models whether a dimension's domain has been defined in the
// iteration space. This information is used to determine if an operation
// leads to an iteration space transition (i.e., a previously-defined dimension
// domain is redefined).
//
def DimStateUndef : I32EnumAttrCase<"Undef", 0, "undef">;
def DimStateDef   : I32EnumAttrCase<"Def", 1, "def">;

def DimState : I32EnumAttr<
    "DimState",
    "Models whether a dimension's domain has been defined",
    [DimStateUndef, DimStateDef]> {

  let cppNamespace = TensorIR_Dialect.cppNamespace;
  let genSpecializedAttr = 0;

  let description = [{
    Models whether a dimension's domain has been defined in the iteration space.
  }];
}


#endif  // TENSORIR_ENUMS
