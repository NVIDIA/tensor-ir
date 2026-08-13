add_tensor_ir_library(NVTensorIRTransform
  DiscoverIterationSpaceInfo.cpp
  GraphSplitting.cpp
  LayoutAnnotation.cpp
  MaterializeDefaultStrides.cpp
  TileAnalyzerPass.cpp
  TileSelectionPass.cpp

  DEPENDS
  NVTensorIRTransformPassesIncGen

  LINK_LIBS PUBLIC
  NVTensorIR::Dialect
  NVTensorIR::Analysis
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
