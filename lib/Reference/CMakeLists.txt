add_tensor_ir_library(NVTensorIRReference STATIC
  SimplifiedTensor.cpp
  TensorMemory.cpp
  ReferenceGraph.cpp
  ReferenceNode.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRParser
    NVTensorIR::Dialect
    NVTensorIR::Runtime
    NVTensorIR::Utils
  PRIVATE
    NVTensorIR::CudaApi
)
