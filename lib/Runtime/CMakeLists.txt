# TensorIR Runtime library

add_tensor_ir_library(NVTensorIRKernelLaunchHelpers STATIC
  KernelLaunchHelpers.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    NVTensorIR::Artifact
)

add_library(NVTensorIRRuntime STATIC
  CudaTileRuntimeKernel.cpp
)

tensor_ir_register_library(NVTensorIRRuntime)

# KernelMetadata stores PersistenceMode from the generated Options header.
add_dependencies(NVTensorIRRuntime NVTensorIRCompilerOptionsIncGen)

target_include_directories(NVTensorIRRuntime
  PUBLIC
    $<BUILD_INTERFACE:${TENSOR_IR_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${TENSOR_IR_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(NVTensorIRRuntime
  PUBLIC
    NVTensorIR::KernelLaunchHelpers
    NVTensorIR::CudaApi
    NVTensorIR::Artifact
  PRIVATE
    CudaTileDialect
    CudaTileBytecodeWriter
    LLVMSupport
    NVTensorIRTileIRAssembly
)
