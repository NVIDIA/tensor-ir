# Keep compiler-wide registry composition out of libMLIR. Consumers opt in to
# this dependency when they need the complete TensorIR compiler environment.
add_tensor_ir_library(NVTensorIRRegistration
  Registration.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  CudaTileDialect
  CudaTileTransforms

  MLIRArithDialect
  MLIRArithToLLVM
  MLIRArithTransforms
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRCastInterfaces
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUTransforms
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRLLVMIRToLLVMTranslation
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRPtrDialect
  MLIRPtrToLLVM
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorToLLVM
  MLIRVectorTransforms

  NVTensorIR::CudaTilePipelines
  NVTensorIR::Dialect
  NVTensorIR::LinalgPipelines
  NVTensorIR::ToCudaTileConversion
  NVTensorIR::ToLinalgConversion
  NVTensorIR::Transform

)
