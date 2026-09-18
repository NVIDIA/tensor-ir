// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source of generated option structs, parsers, bindings, and documentation.
include "tensor_ir/Options/OptionsBase.td"
include "tensor_ir/Options/Enums.td"

def DeviceTargetOptions : ComponentOptions<"DeviceTarget"> {
  let clCategory = "TensorIR Compiler Device Target Options";
  let invocationMember = "deviceTarget";
  let options = [
    // The C API exposes this through the structured compute-capability and
    // portability accessors rather than a generated raw-string accessor.
    StringOption<"kernelTarget", "kernel-target",
                 "Kernel target architecture and portability contract",
                 "\"sm_100f\"", 0>,
    Option<"codegenTarget", "codegen-target", "int32_t",
           "0", "Concrete compute capability used for code generation; zero "
                "derives it from kernel-target",
           1, [RangeTrait<0, 2147483647>]>,
  ];
}

def CompilerDebugOptions : ComponentOptions<"CompilerDebug"> {
  let clCategory = "TensorIR Compiler Debug Options";
  let invocationMember = "debug";
  let options = [
    FlagOption<"printIRAfterAll", "print-ir-after-all",
               "Print IR after every pass">,
    StringOption<"printIRTreeDir", "print-ir-tree-dir",
               "Directory to print per-pass IR snapshots">,
    StringOption<"reproducerDir", "reproducer-dir",
               "Directory to write MLIR pass pipeline crash reproducers">,
    FlagOption<"enableTiming", "timing", "Enable timing of compilation">,
    StringOption<"dumpCudaTileIRPath", "dump-ir",
               "Path to dump CUDA Tile IR">,
    StringOption<"dumpTileIRBCPath", "dump-tileir-bc",
               "Path to dump Tile IR bytecode">,
    StringOption<"loadTileIRBCPath", "load-tileir-bc",
               "Path to load Tile IR bytecode">,
    FlagOption<"printCudaTileIR", "print-cuda-tile-ir",
           "Print CUDA Tile IR">,
    StringOption<"cudaToolkitPath", "cuda-toolkit-path",
               "Path to the CUDA toolkit root (replaces the CUDA_ROOT env "
               "variable for compilation)">,
  ];
}

def CudaTileCodegenStrategyOpt
    : EnumOption<"codegenStrategy", "codegen-strategy",
                 CudaTileCodegenStrategyEnum, "LayoutPropagation",
                 "TensorIR-to-CUDA-Tile lowering strategy"> {
  let capiType = "MlirTensorIRCudaTileCodegenStrategy";
}

def CudaTileArtifactKindOpt
    : EnumOption<"artifactKind", "artifact-kind", ArtifactKindEnum, "TileIR",
                 "CUDA Tile device artifact">;

def PersistenceModeOpt
    : EnumOption<"persistenceMode", "persistence-mode", PersistenceModeEnum,
                 "None", "Kernel persistence mode">;

def CompilerBackendOpt : EnumOption<"compilerBackend", "compiler-backend",
                                    CompilerBackendEnum, "Auto",
                                    "Compiler backend">;

def CodegenOptions : ComponentOptions<"Codegen"> {
  let clCategory = "TensorIR Codegen Options";
  let invocationMember = "codegen";
  let options = [CompilerBackendOpt,
       Option<"ctaCount", "cta-count", "int32_t", "1",
              "Number of cooperative thread arrays", 1, [PositiveTrait<>]>,
       TileShapeOption<"tileSize", "tile-size", "int32_t",
                       "Per-dimension kernel tile sizes ('x' separated)">];
}

def CudaTileOptions : ComponentOptions<"CudaTile"> {
  let clCategory = "TensorIR CUDA Tile Options";
  let invocationMember = "cudaTile";
  let verifyFunction =
      "mlir::nv_tensor_ir::backend::cuda_tile::verifyCudaTileOptions";
  let options = [StringOption<
           "bytecodeVersion", "bytecode-version",
           "CUDA Tile bytecode target: default, current, compatibility, or a "
           "dotted version", "\"default\"", 0>,
       Option<"warpCount", "warp-count", "int32_t", "4",
              "Number of worker warps per CTA", 1, [PositiveTrait<>]>,
       FlagOption<"uniformSignature", "uniform-signature",
              "Pass sizes and strides for static and dynamic dimensions">,
       Option<"reductionTileSize", "reduction-tile-size", "int64_t", "128",
              "Tile size for contracting dimensions in layout-propagation "
              "lowering; must be a positive power of two",
              1, [PositiveTrait<>, PowerOfTwoTrait<>]>,
       Option<"maxTileCandidates", "max-tile-candidates", "int32_t", "1",
              "Maximum layout-propagation tile candidates",
              1, [RangeTrait<0, 2147483647>]>,
       Option<"occupancy", "occupancy", "int32_t", "1",
              "Target occupancy (CTAs per SM)", 1, [PositiveTrait<>]>,
       // 0 means "not supplied": a static persistent kernel needs the real SM
       // count of the device it will run on, which only the caller knows.
       Option<"smCount", "sm-count", "int32_t", "0",
              "SM count of the target device, for persistent kernels", 1,
              [RangeTrait<0, 2147483647>]>,
       CudaTileCodegenStrategyOpt, CudaTileArtifactKindOpt, PersistenceModeOpt];
}


def CompilerToolTestOptions : ComponentOptions<"CompilerToolTest"> {
  let clCategory = "TensorIR Compiler Tool Test Options";
  let options = [FlagOption<"launch", "launch",
              "Launch compiled runtime kernel on GPU">,
       FlagOption<"verify", "verify",
              "Launch and verify results against reference">,
       Option<"seed", "seed", "unsigned", "2024", "Random seed for test data">,
       Option<"iterations", "iterations", "int", "1",
              "Number of launch iterations">,
       Option<"tolerance", "tolerance", "float", "1e-2",
              "Tolerance for verification">,
       ShapeOption<
           "dynamicDims", "dynamic-dims",
           "Runtime extents of the dynamic '?' dimensions, by position">,
       ShapeOption<"dynamicStrides", "dynamic-strides",
                   "Runtime values of the dynamic '?' strides, by position">,
  ];
}

def CompilerToolDebugOptions : ComponentOptions<"CompilerToolDebug"> {
  let clCategory = "TensorIR Compiler Tool Debug Options";
  let options = [FlagOption<"verbose", "verbose",
                        "Enable verbose log output">,
                 StringOption<"dumpArtifactPath", "dump-artifact",
                        "Write compiled CudaTile device code to this path and "
                        "metadata to <path>.meta">,
  ];
}

def CompilerInvocation
    : CompilerInvocationOptions<[
        DeviceTargetOptions,
        CodegenOptions,
        CudaTileOptions,
        CompilerDebugOptions
      ]>;
