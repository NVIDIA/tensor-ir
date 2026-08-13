add_tensor_ir_conversion_library(NVTensorIRToCudaTileConversion
  AffineMapImpl.cpp
  EmitHelpers.cpp
  LayoutPropagationImpl.cpp
  MatmulOps.cpp
  PointwiseOps.cpp
  ReductionOps.cpp
  StructureBuilder.cpp
  TensorToCudaTile.cpp

  DEPENDS
  MLIRTensorToCudaTileConversionPassIncGen

  LINK_LIBS PUBLIC
  CudaTileDialect
  MLIRArithDialect
  MLIRTransformUtils

  # For TileAnalyzer and analyzeTileSizeForGraph
  NVTensorIR::Analysis
 )
