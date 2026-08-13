add_library(NVTensorIRReference STATIC
  simplified_tensor.cpp
  tensor_memory.cpp
  reference_graph.cpp
  reference_node.cpp
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
