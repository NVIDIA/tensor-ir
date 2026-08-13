if(TENSOR_IR_INCLUDE_TESTS)
  # Passed to lit.site.cfg.py.in to set up the path where to find libraries.
  set(MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  llvm_canonicalize_cmake_booleans(
    TENSOR_IR_INCLUDE_TESTS
    TENSOR_IR_ENABLE_BINDINGS_PYTHON
  )

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )


  set(MLIR_TEST_DEPENDS
    FileCheck
    not
    tensor_ir-compiler
    tensor_ir-opt
  )


  if(TENSOR_IR_ENABLE_BINDINGS_PYTHON)

    list(APPEND MLIR_TEST_DEPENDS
      TensorIRPythonModules
    )
  endif()

  # This target can be used to just build the dependencies
  # for the check-mlir target without executing the tests.
  # This is useful for bots when splitting the build step
  # from the testing step.
  add_custom_target(check-tensor-ir-build-only
    DEPENDS ${MLIR_TEST_DEPENDS}
  )

  add_lit_testsuite(check-tensor-ir "Running the TENSOR_IR regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MLIR_TEST_DEPENDS}
  )
  set_target_properties(check-tensor-ir PROPERTIES FOLDER "Tests")

endif()
