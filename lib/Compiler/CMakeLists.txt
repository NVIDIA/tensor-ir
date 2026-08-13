# TensorIR public compiler library.

add_tensor_ir_library(NVTensorIRCudaTilePipelines
  CudaTile/Pipelines.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    NVTensorIR::Dialect
    NVTensorIR::ToCudaTileConversion
    NVTensorIR::Transform
)


add_tensor_ir_library(NVTensorIRCompiler STATIC
  Compiler.cpp
  CudaTile/CudaTileCompiler.cpp
  CudaTile/CudaTileFrontend.cpp
  CudaTile/TileIRAssembly.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    NVTensorIR::CudaTilePipelines
    NVTensorIR::Registration
    NVTensorIR::Dialect
    MLIRParser
    MLIRIR
    MLIRPass
    NVTensorIR::Runtime
)

