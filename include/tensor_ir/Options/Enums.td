// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSORIR_OPTIONS_ENUMS
#define TENSORIR_OPTIONS_ENUMS

include "mlir/IR/EnumAttr.td"

// The option generator consumes the standard EnumInfo records.  This small
// case extension keeps CLI help text local to option enums without adding a
// second enum representation.
class CompilerOptionEnumCase<string sym, int val, string str = sym,
                             string helpStr = "">
    : I32EnumCase<sym, val, str> {
  string help = helpStr;
}

class CompilerOptionEnum<string name, string summary, list<EnumCase> cases>
    : I32Enum<name, summary, cases> {
  let cppNamespace = "::mlir::nv_tensor_ir";
  let underlyingType = "int";
}

def CompilerBackendEnum : CompilerOptionEnum<
    "CompilerBackend", "Compiler backend",
    [CompilerOptionEnumCase<"Auto", 0, "auto">,
     CompilerOptionEnumCase<"CudaTile", 1, "cuda_tile">,
    ]> {
}

def HostTargetEnum : CompilerOptionEnum<
    "HostTarget", "Host execution target",
    [CompilerOptionEnumCase<"Native", 0, "native">,
     CompilerOptionEnumCase<"Interpreter", 1, "interpreter">,
     CompilerOptionEnumCase<"Auto", 2, "auto">]>;

def HostArchEnum : CompilerOptionEnum<
    "HostArch", "Host architecture",
    [CompilerOptionEnumCase<"Native", 0, "native", "current process architecture">,
     CompilerOptionEnumCase<"X86_64", 1, "x86_64", "x86_64">,
     CompilerOptionEnumCase<"AArch64", 2, "aarch64">]>;

def HostOSEnum : CompilerOptionEnum<
    "HostOS", "Host operating system",
    [CompilerOptionEnumCase<"Windows", 0, "windows">,
     CompilerOptionEnumCase<"Linux", 1, "linux">]>;

def CudaTileCodegenStrategyEnum : CompilerOptionEnum<
    "CudaTileCodegenStrategy", "TensorIR-to-CUDA-Tile lowering strategy",
    [CompilerOptionEnumCase<"AffineMap", 0, "affine-map">,
     CompilerOptionEnumCase<"LayoutPropagation", 1, "layout-propagation">]> {
}

def ArtifactKindEnum : CompilerOptionEnum<
    "ArtifactKind", "CUDA Tile device artifact",
    [CompilerOptionEnumCase<"TileIR", 0, "tileir",
                            "Return TileIR bytecode for the deployment driver to JIT">,
     CompilerOptionEnumCase<"Cubin", 1, "cubin",
                            "Request a cubin and fall back to TileIR when no compatible assembler is available">]> {
}

def PersistenceModeEnum : CompilerOptionEnum<
    "PersistenceMode", "Kernel persistence mode",
    [CompilerOptionEnumCase<"None", 0, "none">,
     CompilerOptionEnumCase<"Static", 1, "static">,
    ]> {
}

def AutoGdcInsertionEnum : CompilerOptionEnum<
    "AutoGdcInsertion",
    "Automatic grid dependency control insertion for Programmatic Dependent "
    "Launch",
    [CompilerOptionEnumCase<"None", 0, "none",
                            "Do not insert grid dependency control">,
     CompilerOptionEnumCase<
         "Conservative", 1, "conservative",
         "Wait for the preceding grid before any global memory access; "
         "dependent grids are triggered implicitly when predecessors exit">]> {
}


#endif // TENSORIR_OPTIONS_ENUMS
