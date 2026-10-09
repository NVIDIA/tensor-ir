# Keep compiler-wide registry composition out of libMLIR. Consumers opt in to
# this dependency when they need the complete TensorIR compiler environment.
add_tensor_ir_library(NVTensorIRRegistration
  Registration.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  CudaTileDialect
  MLIRIR
  NVTensorIR::Dialect
  NVTensorIR::ToCudaTileConversion
  NVTensorIR::Transform
  MLIRArithToLLVM
  MLIRArithDialect
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRBufferizationDialect
  MLIRCastInterfaces
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUTransforms
  MLIRLLVMDialect
  MLIRLLVMIRToLLVMTranslation
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRPtrDialect
  MLIRPtrToLLVM
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorToLLVM
  MLIRVectorTransforms

)
