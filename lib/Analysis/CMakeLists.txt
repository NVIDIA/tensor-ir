add_tensor_ir_library(NVTensorIRAnalysis
  PARTIAL_SOURCES_INTENDED
  TileAnalyzer.cpp
  TileCandidateGenerator.cpp

  DEPENDS
  NVTensorIRCompilerOptionsIncGen

  LINK_LIBS PUBLIC
  NVTensorIR::Dialect
  NVTensorIR::Utils
)

add_tensor_ir_library(NVTensorIRPolyExprs
  PARTIAL_SOURCES_INTENDED
  PolyExprs.cpp

  LINK_COMPONENTS Support
)
