# Initialize and link the supported host backends present in this LLVM build.
set(host_codegen_components)
foreach(host_arch X86 AArch64)
  if(host_arch IN_LIST LLVM_TARGETS_TO_BUILD)
    set_property(SOURCE HostToObject.cpp APPEND PROPERTY
      COMPILE_DEFINITIONS TENSOR_IR_HOST_TARGET_${host_arch}=1)
    list(APPEND host_codegen_components ${host_arch}CodeGen)
  endif()
endforeach()

add_tensor_ir_library(NVTensorIRTransform
  HostAbiBridge.cpp
  HostLoweringUtils.cpp
  HostToObject.cpp
  HostToLLVM.cpp
  Bufferize.cpp
  CudaTileInsertGdcWait.cpp
  DiscoverIterationSpaceInfo.cpp
  FormGrid.cpp
  TileReductions.cpp
  MaterializeDefaultStrides.cpp
  OutlineKernel.cpp
  TileAnalyzerPass.cpp
  TileSelectionPass.cpp

  DEPENDS
  NVTensorIRTransformPassesIncGen

  LINK_COMPONENTS CodeGen Object ${host_codegen_components}

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRPass
  MLIRPtrDialect
  MLIRPtrToLLVM
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  MLIRUBDialect
  MLIRUBToLLVM
  CudaTileDialect
  MLIRTargetLLVMIRExport
  NVTensorIR::Analysis
  NVTensorIR::Dialect
  NVTensorIR::Utils
)
