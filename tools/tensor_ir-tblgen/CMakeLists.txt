set(LLVM_LINK_COMPONENTS TableGen Support)
add_tablegen(tensor_ir-tblgen TensorIR
  tensor_ir-tblgen.cpp
  ComponentOptionsGen.cpp
)
# Non-exported TableGen tools are excluded from the default build, but this
# executable is installed even when package jobs disable LLVM_BUILD_UTILS.
set_target_properties(tensor_ir-tblgen PROPERTIES EXCLUDE_FROM_ALL FALSE)
install(TARGETS tensor_ir-tblgen
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(tensor_ir-tblgen PRIVATE -Wno-global-constructors)
endif()
target_link_libraries(tensor_ir-tblgen PRIVATE MLIRTblgenLib)
mlir_check_all_link_libraries(tensor_ir-tblgen)
