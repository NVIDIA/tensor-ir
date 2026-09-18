include("${TENSOR_IR_SOURCE_DIR}/cmake/TensorIRFetchFlatBuffers.cmake")
tensor_ir_fetch_flatbuffers()

add_subdirectory(Schema)

add_tensor_ir_library(NVTensorIRArtifact
  TensorIRArtifact.cpp
  TensorIRArtifactCodec.cpp

  LINK_LIBS
  PUBLIC
    NVTensorIR::Utils
    CudaTileDialect
    CudaTileBytecodeWriter
    NVTensorIR::ArtifactSchema
)
