# Fetch and configure the LLVM project revision required by CUDA Tile.
#
# TensorIR records a compatible CUDA Tile and LLVM pair. This module configures
# the LLVM revision as an excluded FetchContent dependency and exposes MLIR's
# generated build-tree package through the same root CUDA Tile consumes.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/TensorIRDependencyPins.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/TensorIRFetchContent.cmake")

set(TENSOR_IR_LLVM_SOURCE_URL
    "https://github.com/llvm/llvm-project/archive/${TENSOR_IR_PINNED_LLVM_COMMIT}.tar.gz"
    CACHE STRING "LLVM source archive URL")

function(_tensor_ir_populate_llvm_project source_url binary_dir_var)
  tensor_ir_fetch_content(
    tensor_ir_llvm_project llvm_source_dir llvm_binary_dir
    SOURCE_SUBDIR llvm
    FETCH_CONTENT_ARGS
      URL "${source_url}"
      URL_HASH "SHA256=${TENSOR_IR_PINNED_LLVM_ARCHIVE_SHA256}")
  set(${binary_dir_var} "${llvm_binary_dir}" PARENT_SCOPE)
endfunction()

function(_tensor_ir_prepare_mlir_build_package llvm_binary_dir mlir_dir_var)
  unset(_tensor_ir_mlir_dir CACHE)
  find_path(_tensor_ir_mlir_dir NAMES MLIRConfig.cmake
            PATHS "${CMAKE_BINARY_DIR}/lib/cmake/mlir" REQUIRED NO_DEFAULT_PATH)

  # MLIR emits its build-tree package in the top-level build directory while
  # LLVM emits LLVMConfig.cmake in its FetchContent binary directory. CUDA Tile
  # expects both packages below one build/install root, so mirror the small MLIR
  # CMake package there.
  set(tensor_ir_llvm_mlir_dir "${llvm_binary_dir}/lib/cmake/mlir")
  file(MAKE_DIRECTORY "${tensor_ir_llvm_mlir_dir}")
  file(COPY "${_tensor_ir_mlir_dir}/" DESTINATION "${tensor_ir_llvm_mlir_dir}")

  set(${mlir_dir_var} "${_tensor_ir_mlir_dir}" PARENT_SCOPE)
  unset(_tensor_ir_mlir_dir CACHE)
endfunction()

macro(_tensor_ir_configure_llvm_options)
  # Macro for configuring LLVM options when this project is managing the LLVM
  # dependency.
  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
  set(LLVM_ENABLE_RUNTIMES "" CACHE STRING "" FORCE)
  set(LLVM_TARGETS_TO_BUILD "Native" CACHE STRING "" FORCE)

  # With CMP0077 NEW, option() doesn't clear normal variables.
  set(LLVM_APPEND_VC_REV OFF)
  set(LLVM_BUILD_EXAMPLES ON)
  set(LLVM_BUILD_UTILS ON)
  set(LLVM_INSTALL_UTILS ON)
  set(MLIR_INCLUDE_INTEGRATION_TESTS ON)

  # These are declared CACHE BOOL, so we must FORCE them.
  set(MLIR_ENABLE_BINDINGS_PYTHON "${TENSOR_IR_ENABLE_BINDINGS_PYTHON}"
      CACHE BOOL "Build MLIR Python bindings" FORCE)

  # If ccache is available, enable it.
  find_program(CCACHE_EXE NAMES ccache NO_CMAKE_FIND_ROOT_PATH)
  if(CCACHE_EXE)
    set(LLVM_CCACHE_BUILD ON CACHE BOOL "Use ccache" FORCE)
  endif()
endmacro()

macro(tensor_ir_configure_llvm_cmake_compatibility_policies)
  # LLVM revisions supported by CUDA Tile predate CMake 4.4's linker flag
  # parsing policy and expect its compatibility behavior.
  if(POLICY CMP0181)
    cmake_policy(SET CMP0181 OLD)
  endif()
  if(POLICY CMP0219)
    cmake_policy(SET CMP0219 OLD)
  endif()
endmacro()

function(tensor_ir_fetch_llvm_project llvm_root_var)

  tensor_ir_configure_llvm_cmake_compatibility_policies()
  if(TENSOR_IR_LLVM_SOURCE_URL STREQUAL "")
    message(FATAL_ERROR "TENSOR_IR_LLVM_SOURCE_URL must not be empty")
  endif()

  _tensor_ir_configure_llvm_options()
  message(STATUS "Fetching pinned LLVM ${TENSOR_IR_PINNED_LLVM_COMMIT} from "
                 "${TENSOR_IR_LLVM_SOURCE_URL}")
  _tensor_ir_populate_llvm_project("${TENSOR_IR_LLVM_SOURCE_URL}"
                                   llvm_binary_dir)
  _tensor_ir_prepare_mlir_build_package("${llvm_binary_dir}" mlir_dir)

  set(MLIR_DIR "${mlir_dir}"
      CACHE PATH "Path to the MLIR package configuration" FORCE)
  set(${llvm_root_var} "${llvm_binary_dir}" PARENT_SCOPE)
endfunction()

macro(tensor_ir_configure_apply_llvm_compat_module)
  # Apply an optional CMake module prior to finding the MLIR package
  # configuration. This is another extension point for build configurations that
  # need to inject testing logic prior to the find_package(MLIR) call. Populate
  # the module path using TENSOR_IR_LLVM_COMPAT_MODULE variable.
  if(DEFINED TENSOR_IR_LLVM_COMPAT_MODULE)
    include("${TENSOR_IR_LLVM_COMPAT_MODULE}")
  endif()
endmacro()

# Find the MLIR package configuration and add the necessary CMake modules to the
# search path.
macro(tensor_ir_find_mlir)
  tensor_ir_configure_apply_llvm_compat_module()

  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
endmacro()

macro(tensor_ir_configure_cxx_llvm_abi_compatibility)
  # Apply required LLVM/MLIR build configuration options within this scope. This
  # will apply to all targets/sub-projects from this point on.
  include(HandleLLVMOptions)
  if(NOT DEFINED LLVM_DEFINITIONS)
    message(FATAL_ERROR "LLVM_DEFINITIONS is not set")
  endif()
  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})
endmacro()
