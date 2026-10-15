add_tensor_ir_library(NVTensorIRTilingAnalysis
  Arch.cpp
  Enumerate.cpp
  Evaluate.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  NVTensorIR::Analysis
  NVTensorIR::Dialect
  NVTensorIR::Utils
)
