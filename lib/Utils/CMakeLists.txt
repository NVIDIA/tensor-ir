add_tensor_ir_library(NVTensorIRUtils
  ComputeCapability.cpp
  Utils.cpp
  DEPENDS

  LINK_LIBS PUBLIC
  NVTensorIR::Dialect
  MLIRBytecodeWriter
)
