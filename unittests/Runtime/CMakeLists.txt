add_tensor_ir_ut(TensorIRNativeHostModuleTests
  SRCS NativeHostModuleTests.cpp
  LINK_LIBS
  NVTensorIR::Runtime
  NVTensorIR::Compiler
)

target_include_directories(TensorIRNativeHostModuleTests PRIVATE
  ${TENSOR_IR_SOURCE_DIR}/lib/Runtime)


add_tensor_ir_ut(TensorIRRuntimeOperandAccessorTests
  SRCS RuntimeOperandAccessorTests.cpp
  LINK_LIBS
  NVTensorIR::Runtime
)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND _TIR_RUNTIME_TEST_GOLD)
  target_link_options(TensorIRRuntimeOperandAccessorTests PRIVATE
    -fuse-ld=gold -Wl,--stub-group-size=4194304)
endif()
