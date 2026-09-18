add_tensor_ir_library(NVTensorIRCompilerOptions
  Options.cpp
  CompilerInvocation.cpp
  BytecodeVersionOptions.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS
    NVTensorIRCompilerOptionsIncGen
    TensorIRIncGen
  LINK_LIBS PUBLIC
    CudaTileBytecodeWriter
    NVTensorIR::Utils
  LINK_COMPONENTS Support
)

# Tool-support library: the cl bindings TU. Linked by tools, never by the
# compiler options library or anything below it.
add_tensor_ir_library(NVTensorIRCompilerCLOptions
  OptionsCL.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS NVTensorIRCompilerOptionsIncGen
  LINK_LIBS PUBLIC NVTensorIRCompilerOptions
  LINK_COMPONENTS Support
)
