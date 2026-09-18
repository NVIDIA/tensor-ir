set(TENSOR_IR_ARTIFACT_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/Artifact.fbs")
set(TENSOR_IR_ARTIFACT_SCHEMA_INCLUDE_DIR "${TENSOR_IR_BINARY_DIR}/include")
set(TENSOR_IR_ARTIFACT_SCHEMA_OUTPUT_DIR
    "${TENSOR_IR_ARTIFACT_SCHEMA_INCLUDE_DIR}/tensor_ir/Artifact/Schema")
set(TENSOR_IR_ARTIFACT_SCHEMA_HEADER
    "${TENSOR_IR_ARTIFACT_SCHEMA_OUTPUT_DIR}/Artifact_generated.h")

tensor_ir_flatc_command(TENSOR_IR_FLATC)

add_custom_command(
  OUTPUT "${TENSOR_IR_ARTIFACT_SCHEMA_HEADER}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${TENSOR_IR_ARTIFACT_SCHEMA_OUTPUT_DIR}"
  COMMAND ${TENSOR_IR_FLATC}
          --cpp
          --cpp-std c++17
          --scoped-enums
          --filename-suffix _generated
          -o "${TENSOR_IR_ARTIFACT_SCHEMA_OUTPUT_DIR}"
          "${TENSOR_IR_ARTIFACT_SCHEMA}"
  DEPENDS "${TENSOR_IR_ARTIFACT_SCHEMA}"
  COMMENT "Generating TensorIR artifact FlatBuffers schema"
  VERBATIM)

add_custom_target(NVTensorIRArtifactSchemaGen
                   DEPENDS "${TENSOR_IR_ARTIFACT_SCHEMA_HEADER}")

add_library(NVTensorIRArtifactSchema INTERFACE)
tensor_ir_register_library(NVTensorIRArtifactSchema)
add_dependencies(NVTensorIRArtifactSchema NVTensorIRArtifactSchemaGen)
target_include_directories(
  NVTensorIRArtifactSchema
  INTERFACE "$<BUILD_INTERFACE:${TENSOR_IR_ARTIFACT_SCHEMA_INCLUDE_DIR}>")
target_link_libraries(NVTensorIRArtifactSchema INTERFACE flatbuffers)
