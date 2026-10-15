# TensorIR public compiler library.

add_tensor_ir_library(NVTensorIRLinalgPipelines
  Linalg/Pipelines.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRAffineToStandard
    MLIRBufferizationTransforms
    MLIRConvertToLLVMPass
    MLIRFuncDialect
    MLIRLinalgTransforms
    MLIRMathToFuncs
    MLIRMathToLibm
    MLIRMemRefTransforms
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRTransforms
    NVTensorIR::ToLinalgConversion
)

add_tensor_ir_library(NVTensorIRCudaTilePipelines
  CudaTile/Pipelines.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    CudaTileTransforms
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    MLIRSCFToControlFlow
    MLIRTransforms
    NVTensorIR::CompilerOptions
    NVTensorIR::Dialect
    NVTensorIR::ToCudaTileConversion
    NVTensorIR::Transform
)


# Assembles TileIR bytecode into a cubin via 'tileiras'; shared by Compiler
# and Runtime.
add_tensor_ir_library(NVTensorIRTileIRAssembly
  CudaTile/TileIRAssembly.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
)

add_tensor_ir_library(NVTensorIRCompiler STATIC
  Compiler.cpp
  CudaTile/CudaTileCompiler.cpp
  CudaTile/CudaTileFrontend.cpp
  CudaTile/KernelMetadataExtraction.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    NVTensorIR::CudaTilePipelines
    NVTensorIR::Registration
    NVTensorIR::Dialect
    NVTensorIR::Analysis
    MLIRParser
    MLIRIR
    MLIRPass
    NVTensorIR::Artifact
    NVTensorIR::CompilerOptions
    NVTensorIRTileIRAssembly
)
