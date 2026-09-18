add_tensor_ir_test_library(NVTensorIRTestDialect
  TestConvertOpBuilder.cpp
  TestSideEffects.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRFuncDialect
  MLIRSideEffectInterfaces
  NVTensorIR::Dialect
)
