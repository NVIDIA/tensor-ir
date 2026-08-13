# The nv_tensor_ir package definition. Public compiler/runtime bindings are
# always present.

set(LLVM_INSTALL_TOOLCHAIN_ONLY ON)

include(AddMLIRPython)

# MLIR's Python headers suppress Clang-only diagnostics. GCC reports each
# unknown diagnostic name under -Wpragmas.
add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-pragmas>)


# All native extensions in this package share one relocated MLIR import prefix
# and one nanobind type-registration domain.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=nv_tensor_ir._mlir.")
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN nv_tensor_ir)


set(_tensor_ir_python_source_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nv_tensor_ir")
set(_tensor_ir_python_mlir_source_dir
    "${_tensor_ir_python_source_dir}/_mlir")

set(TENSOR_IR_PYTHON_PACKAGE_PREFIX
    "${TENSOR_IR_BINARY_DIR}/python_packages"
    CACHE PATH "Build-tree root for TensorIR Python packages")
set(_tensor_ir_python_package_dir
    "${TENSOR_IR_PYTHON_PACKAGE_PREFIX}/nv_tensor_ir")

################################################################################
# Package declarations
################################################################################

add_subdirectory(src)
add_subdirectory(bindings)

################################################################################
# One common C API DSO for every extension in the package
################################################################################

set(_tensor_ir_python_capi_sources
  MLIRPythonExtension.Core
  TensorIRPythonSources
)

add_mlir_python_common_capi_library(NVTensorIRPythonCAPI
  INSTALL_COMPONENT TensorIRPythonModules
  INSTALL_DESTINATION python_packages/nv_tensor_ir/_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${_tensor_ir_python_package_dir}/_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${_tensor_ir_python_capi_sources}
  EMBED_LIBS
    # Static C API archives belong here, not in COMMON_CAPI_LINK_LIBS, so MLIR
    # TypeID and registry state is present exactly once in the package.
    MLIRCAPITransforms
)
tensor_ir_register_library(NVTensorIRPythonCAPI)



################################################################################
# Package assembly
################################################################################

set(_tensor_ir_python_module_sources
  MLIRPythonSources.Core
  TensorIRPythonSources
)

if(NOT COMMAND add_project_mlir_python_modules)
  macro(add_project_mlir_python_modules)
    add_mlir_python_modules(${ARGV})
  endmacro()
endif()

add_project_mlir_python_modules(TensorIRPythonMLIRModules
  ROOT_PREFIX "${_tensor_ir_python_package_dir}/_mlir"
  INSTALL_PREFIX "python_packages/nv_tensor_ir/_mlir"
  MLIR_BINDINGS_PYTHON_NB_DOMAIN nv_tensor_ir
  DECLARED_SOURCES ${_tensor_ir_python_module_sources}
  COMMON_CAPI_LINK_LIBS NVTensorIR::PythonCAPI
)

# The TensorIR dialect declaration lives in the src subdirectory, while this
# packaging target is created here. With Unix Makefiles, explicitly order the
# generated-source copy after the cross-directory TableGen target.
add_dependencies(
  TensorIRPythonMLIRModules.sources.TensorIRPythonSources.Dialects.nv_tensor_ir.ops_gen
  TensorIRPythonSources.Dialects.nv_tensor_ir.tablegen
)

add_mlir_python_sources_target(TensorIRPythonPublicModules
  INSTALL_COMPONENT TensorIRPythonModules
  INSTALL_DIR "python_packages/nv_tensor_ir"
  OUTPUT_DIRECTORY "${_tensor_ir_python_package_dir}"
  SOURCES_TARGETS TensorIRPythonPublicSources
)

add_custom_target(TensorIRPythonModules ALL)
add_dependencies(TensorIRPythonModules
  NVTensorIRPythonCAPI
  TensorIRPythonMLIRModules
  TensorIRPythonPublicModules
)


# Keep the public build target name stable.
add_custom_target(tensor_ir_python_bindings
  DEPENDS TensorIRPythonModules)
