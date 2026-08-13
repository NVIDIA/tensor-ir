# Defines TensorIR's canonical library targets and component aliases.

include_guard(GLOBAL)

set_property(GLOBAL PROPERTY TENSOR_IR_LIBS "")

function(tensor_ir_register_library name)
  cmake_parse_arguments(ARG "TEST_LIBRARY" "" "" ${ARGN})

  if(NOT TARGET "${name}")
    message(FATAL_ERROR
      "Cannot register TensorIR library '${name}': target does not exist")
  endif()

  string(REGEX REPLACE "^NVTensorIR" "" component "${name}")
  if(component STREQUAL name OR component STREQUAL "")
    message(FATAL_ERROR
      "TensorIR library '${name}' must be named NVTensorIR<Component>")
  endif()

  set(alias "NVTensorIR::${component}")
  if(TARGET "${alias}")
    message(FATAL_ERROR
      "Cannot register TensorIR library '${name}': alias '${alias}' exists")
  endif()

  add_library("${alias}" ALIAS "${name}")
  if(NOT ARG_TEST_LIBRARY)
    set_property(GLOBAL APPEND PROPERTY TENSOR_IR_LIBS "${name}")
  endif()
endfunction()
