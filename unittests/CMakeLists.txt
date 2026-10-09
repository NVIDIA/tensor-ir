add_custom_target(TensorIRUnitTests)

function(add_tensor_ir_ut ARG_TARGET)

  cmake_parse_arguments(ARG "" "" "SRCS;LINK_LIBS" ${ARGN})
  # The test/Unit lit suite discovers the binary that add_unittest places in
  # the build tree's unittests/ directory.
  add_unittest(TensorIRUnitTests ${ARG_TARGET} ${ARG_SRCS}
    PARTIAL_SOURCES_INTENDED)
  target_link_libraries(${ARG_TARGET} PRIVATE ${ARG_LINK_LIBS})
endfunction()

add_subdirectory(Analysis)

add_subdirectory(Support)
