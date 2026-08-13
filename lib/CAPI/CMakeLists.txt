add_tensor_ir_public_c_api_library(NVTensorIRCAPI
  TensorIR.cpp

  LINK_LIBS PUBLIC
    MLIRCAPIIR
    NVTensorIR::Compiler
    NVTensorIR::Dialect
    NVTensorIR::Registration
    NVTensorIR::Runtime
)
