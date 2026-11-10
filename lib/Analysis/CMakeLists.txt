add_subdirectory(Tiling)

add_tensor_ir_library(NVTensorIRAnalysis
  PARTIAL_SOURCES_INTENDED
  SymbolicLayout.cpp
  ShapeOracle.cpp
  ShapeRelationModels.cpp
  LayoutPropagation.cpp
  TileAnalyzer.cpp
  TileCandidateGenerator.cpp

  DEPENDS
  NVTensorIRCompilerOptionsIncGen

  LINK_LIBS PUBLIC
  NVTensorIR::Dialect
  NVTensorIR::Utils
  NVTensorIR::PolyExprs
  MLIRAffineDialect
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRValueBoundsOpInterface
)

add_tensor_ir_library(NVTensorIRPolyExprs
  PARTIAL_SOURCES_INTENDED
  PolyExprs.cpp

  LINK_COMPONENTS Support
)
