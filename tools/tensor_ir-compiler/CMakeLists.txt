# TensorIR Compiler Tool
# Compiles TensorIR MLIR programs to executable GPU kernels

add_mlir_tool(tensor_ir-compiler
  tensor_ir-compiler.cpp
  test_options.cpp
  PARTIAL_SOURCES_INTENDED
)

set(_tensor_ir_compiler_targets
  NVTensorIRCudaApi
  NVTensorIRRuntime
  NVTensorIRCompiler
  NVTensorIRReference
  tensor_ir-compiler
)

foreach(_target ${_tensor_ir_compiler_targets})
  llvm_update_compile_flags(${_target})
endforeach()

target_link_libraries(tensor_ir-compiler
  PRIVATE
    NVTensorIR::Compiler
    NVTensorIR::Reference
)

mlir_check_all_link_libraries(tensor_ir-compiler)

if(COMMAND tensor_ir_add_large_debug_link_options)
  tensor_ir_add_large_debug_link_options(tensor_ir-compiler)
endif()
