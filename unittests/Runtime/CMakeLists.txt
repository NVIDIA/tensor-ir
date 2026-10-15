add_tensor_ir_ut(TensorIRNativeHostModuleTests
  SRCS NativeHostModuleTests.cpp
  LINK_LIBS
  NVTensorIR::Runtime
  NVTensorIR::Compiler
)

target_include_directories(TensorIRNativeHostModuleTests PRIVATE
  ${TENSOR_IR_SOURCE_DIR}/lib/Runtime)
