add_tensor_ir_library(NVTensorIRAnalysis
  KernelArgLayout.cpp
  TileAnalyzer.cpp
  TileCandidateGenerator.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  NVTensorIR::Dialect
  NVTensorIR::Utils
)
