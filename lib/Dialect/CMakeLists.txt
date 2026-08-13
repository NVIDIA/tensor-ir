set(LLVM_TARGET_DEFINITIONS TensorOpsCanonicalization.td)
mlir_tablegen(TensorOpsCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(TensorIROpsCanonicalizationIncGen)

add_tensor_ir_dialect_library(NVTensorIRDialect
  Canonicalization.cpp
  TensorAttrs.cpp
  TensorDialect.cpp
  TensorOps.cpp

  DEPENDS
  TensorIRIncGen
  TensorIROpsCanonicalizationIncGen

  LINK_LIBS PUBLIC
  NVTensorIR::Support
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
 )

target_include_directories(NVTensorIRDialect
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}

  INTERFACE
  ${TENSOR_IR_COMPILER_INCLUDE_DIR}
  ${TENSOR_IR_COMPILER_BINARY_INCLUDE_DIR}
  ${TENSOR_IR_COMPILER_BINARY_LIB_DIR}
)
