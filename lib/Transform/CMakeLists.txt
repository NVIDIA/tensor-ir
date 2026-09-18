add_tensor_ir_library(NVTensorIRTransform
  Bufferize.cpp
  DiscoverIterationSpaceInfo.cpp
  FormGrid.cpp
  TileReductions.cpp
  GraphSplitting.cpp
  LayoutAnnotation.cpp
  MaterializeDefaultStrides.cpp
  OutlineKernel.cpp
  TileAnalyzerPass.cpp
  TileSelectionPass.cpp

  DEPENDS
  NVTensorIRTransformPassesIncGen

  LINK_LIBS PUBLIC
  NVTensorIR::Dialect
  NVTensorIR::Analysis
  NVTensorIR::Utils
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
)
