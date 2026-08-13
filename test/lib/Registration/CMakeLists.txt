
# Exclude from libMLIR.so because this has static options intended for
# opt-like tools only.
add_tensor_ir_test_library(NVTensorIRTestRegistration
  RegisterPasses.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  NVTensorIR::TestAnalysis
  NVTensorIR::TestDialect
)
