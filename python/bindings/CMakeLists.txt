declare_mlir_python_extension(
    TensorIRPythonSources.Dialects.nv_tensor_ir.PublicCompiler
  MODULE_NAME _tensor_ir
  ADD_TO_PARENT TensorIRPythonSources.Dialects.nv_tensor_ir
  SOURCES
    TensorIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    NVTensorIR::CAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
    dlpack_includes
)

declare_mlir_python_extension(TensorIRPythonSources.SiteInitializer
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT TensorIRPythonSources
  SOURCES
    SiteInitializerModule.cpp
  EMBED_CAPI_LINK_LIBS
    NVTensorIR::CAPI
)

