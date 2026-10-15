add_subdirectory(Tiling)

add_tensor_ir_library(NVTensorIRAnalysis
  PARTIAL_SOURCES_INTENDED
  SymbolicLayout.cpp
  LayoutPropagation.cpp
  TileAnalyzer.cpp
  TileCandidateGenerator.cpp

  DEPENDS
  NVTensorIRCompilerOptionsIncGen

  LINK_LIBS PUBLIC
  NVTensorIR::Dialect
  NVTensorIR::Utils
  NVTensorIR::PolyExprs
)

add_tensor_ir_library(NVTensorIRPolyExprs
  PARTIAL_SOURCES_INTENDED
  PolyExprs.cpp

  LINK_COMPONENTS Support
)
