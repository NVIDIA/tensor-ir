# Keep compiler-wide registry composition out of libMLIR. Consumers opt in to
# this dependency when they need the complete TensorIR compiler environment.
add_tensor_ir_library(NVTensorIRRegistration
  Registration.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRIR
  NVTensorIR::Dialect
  MLIRFuncInlinerExtension

)
