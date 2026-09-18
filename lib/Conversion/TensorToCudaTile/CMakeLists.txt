add_tensor_ir_conversion_library(NVTensorIRToCudaTileConversion
  AffineMapImpl.cpp
  EmitHelpers.cpp
  OutlinedComputeOps.cpp
  OutlinedKernelImpl.cpp
  PointwiseOps.cpp
  TensorToCudaTile.cpp

  DEPENDS
  MLIRTensorToCudaTileConversionPassIncGen

  LINK_LIBS PUBLIC
  CudaTileDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRTransformUtils

  # For TileAnalyzer and analyzeTileSizeForGraph
  NVTensorIR::Analysis
  NVTensorIR::CompilerOptions
 )
