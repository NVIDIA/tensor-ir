# TensorIR Runtime library

add_library(NVTensorIRRuntime STATIC
  CudaTileRuntimeKernel.cpp
)

tensor_ir_register_library(NVTensorIRRuntime)

target_include_directories(NVTensorIRRuntime
  PUBLIC
    $<BUILD_INTERFACE:${TENSOR_IR_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${TENSOR_IR_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(NVTensorIRRuntime
  PUBLIC
    NVTensorIR::CudaApi
  PRIVATE
    CudaTileDialect
    CudaTileBytecodeWriter
    LLVMSupport
)
