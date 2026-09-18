add_tensor_ir_public_c_api_library(NVTensorIRCompilerCAPIOptions
  Options.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS NVTensorIRCompilerOptionsIncGen
  LINK_LIBS PUBLIC NVTensorIRCompilerOptions
)

add_tensor_ir_public_c_api_library(NVTensorIRCAPI
  TensorIR.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
    MLIRCAPIIR
    NVTensorIR::Compiler
    NVTensorIR::Dialect
    NVTensorIR::Registration
    NVTensorIR::Runtime
    NVTensorIR::CompilerCAPIOptions
)
