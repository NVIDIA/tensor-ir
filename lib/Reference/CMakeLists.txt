add_library(NVTensorIRReference STATIC
  SimplifiedTensor.cpp
  TensorMemory.cpp
  ReferenceGraph.cpp
  ReferenceNode.cpp
)

tensor_ir_register_library(NVTensorIRReference)

target_include_directories(NVTensorIRReference
  PUBLIC
    ${TENSOR_IR_SOURCE_DIR}/include
)

target_link_libraries(NVTensorIRReference
  PUBLIC
    MLIRIR
    MLIRParser
    NVTensorIR::Dialect
    NVTensorIR::Runtime
    NVTensorIR::Utils
    LLVMSupport
  PRIVATE
    NVTensorIR::CudaApi
)
