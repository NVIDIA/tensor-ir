# TensorIR public compiler library.

add_tensor_ir_library(NVTensorIRLinalgPipelines
  Linalg/Pipelines.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRAffineToStandard
    MLIRArithTransforms
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
