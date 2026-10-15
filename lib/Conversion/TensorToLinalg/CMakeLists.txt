add_tensor_ir_library(NVTensorIRLinalgScalarBuilder
  ScalarBuilder.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithUtils
  MLIRMathDialect
  NVTensorIR::Dialect
)

add_tensor_ir_conversion_library(NVTensorIRToLinalgConversion
  TensorToLinalg.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRTensorToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRTransformUtils
  NVTensorIR::Dialect
  NVTensorIR::LinalgScalarBuilder
)
