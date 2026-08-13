add_tensor_ir_test_library(NVTensorIRTestDialect
  TestConvertOpBuilder.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRFuncDialect
  NVTensorIR::Dialect
)
