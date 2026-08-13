add_tensor_ir_library(NVTensorIRSupport
  TCutegen.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

add_tensor_ir_library(NVTensorIRCudaApi STATIC
  CudaApi.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(NVTensorIRCudaApi
  PUBLIC
    CUDA::toolkit
  PRIVATE
    ${CMAKE_DL_LIBS}
)

if(TENSOR_IR_CUDART_LINKAGE STREQUAL "STATIC")
  if(NOT TARGET CUDA::cudart_static)
    message(FATAL_ERROR
      "TENSOR_IR_CUDART_LINKAGE=STATIC requires CUDA::cudart_static")
  endif()
  target_compile_definitions(NVTensorIRCudaApi
    PRIVATE TENSOR_IR_STATIC_LINK_CUDART=1)
  target_link_libraries(NVTensorIRCudaApi
    PRIVATE CUDA::cudart_static)
elseif(NOT TENSOR_IR_CUDART_LINKAGE STREQUAL "DYNAMIC")
  message(FATAL_ERROR
    "Invalid TENSOR_IR_CUDART_LINKAGE='${TENSOR_IR_CUDART_LINKAGE}'; "
    "expected DYNAMIC or STATIC")
endif()
