set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

get_property(TENSOR_IR_LIBS GLOBAL PROPERTY TENSOR_IR_LIBS)

set(TENSOR_IR_OPT_LIBS
  ${TENSOR_IR_LIBS}

  # TensorIR libraries
  NVTensorIR::CudaTilePipelines
  NVTensorIR::Dialect
  NVTensorIR::Transform
  NVTensorIR::ToCudaTileConversion

  # MLIR core
  MLIRIR
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  )


if(TENSOR_IR_INCLUDE_TESTS)
  list(APPEND TENSOR_IR_OPT_LIBS
    NVTensorIR::TestAnalysis
    NVTensorIR::TestDialect
    NVTensorIR::TestRegistration
  )
endif()

add_mlir_tool(tensor_ir-opt
  tensor_ir-opt.cpp

  DEPENDS
  ${TENSOR_IR_OPT_LIBS}
  )

target_link_libraries(tensor_ir-opt PRIVATE
  ${TENSOR_IR_OPT_LIBS}
)

if(TENSOR_IR_INCLUDE_TESTS)
  target_compile_definitions(tensor_ir-opt PRIVATE
    TENSOR_IR_INCLUDE_TESTS=1)
endif()

llvm_update_compile_flags(tensor_ir-opt)
mlir_check_all_link_libraries(tensor_ir-opt)
