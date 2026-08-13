cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(TensorIRCompiler VERSION 0.1.0 LANGUAGES C CXX)

# ==============================================================================
# Project Options
# ==============================================================================

option(TENSOR_IR_ENABLE_BINDINGS_PYTHON "Build the TensorIR Python bindings" ON)
option(TENSOR_IR_INCLUDE_TESTS "Generate build targets for TensorIR tests" ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_TENSOR_IR_DOWNLOAD_LLVM_DEFAULT ON)
else()
  set(_TENSOR_IR_DOWNLOAD_LLVM_DEFAULT OFF)
endif()
option(TENSOR_IR_DOWNLOAD_LLVM
       "Download and build the LLVM project revision required by CUDA Tile"
       "${_TENSOR_IR_DOWNLOAD_LLVM_DEFAULT}")
unset(_TENSOR_IR_DOWNLOAD_LLVM_DEFAULT)

set(TENSOR_IR_CUDART_LINKAGE "DYNAMIC"
    CACHE STRING "CUDA Runtime linkage for TensorIR: DYNAMIC or STATIC")
set_property(CACHE TENSOR_IR_CUDART_LINKAGE PROPERTY STRINGS DYNAMIC STATIC)
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

# ==============================================================================
# Project Configuration
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# Disable GCC analyses which produce lots of false positives in
# LLVM ADT headers.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-uninitialized>)
  else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-maybe-uninitialized>)
  endif()
endif()

# ==============================================================================
# CUDA Toolkit
# ==============================================================================

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit include: ${CUDAToolkit_INCLUDE_DIRS}")

if(TENSOR_IR_ENABLE_BINDINGS_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
else()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

# ==============================================================================
# LLVM/MLIR
# ==============================================================================
include(cmake/TensorIRLLVMUtils.cmake)

# Keep dependency install rules out of TensorIR's install as far as LLVM's own
# configuration permits.
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON)

if(TENSOR_IR_DOWNLOAD_LLVM)
  tensor_ir_fetch_llvm_project(TENSOR_IR_FETCHED_LLVM_ROOT)
endif()

tensor_ir_find_mlir()

include(TableGen)
include(AddLLVM)
include(AddMLIR)

tensor_ir_configure_llvm_cmake_compatibility_policies()
tensor_ir_configure_cxx_llvm_abi_compatibility()

# ==============================================================================
# CUDA Tile IR
# ==============================================================================

if(NOT CUDA_TILE_USE_LLVM_INSTALL_DIR)
  if(TENSOR_IR_FETCHED_LLVM_ROOT)
    set(CUDA_TILE_USE_LLVM_INSTALL_DIR "${TENSOR_IR_FETCHED_LLVM_ROOT}")
  else()
    get_filename_component(CUDA_TILE_USE_LLVM_INSTALL_DIR
                           "${LLVM_DIR}/../../.." ABSOLUTE)
  endif()
endif()

set(CUDA_TILE_ENABLE_TESTING OFF)
set(CUDA_TILE_ENABLE_BINDINGS_PYTHON OFF)
include(cmake/TensorIRFetchCudaTile.cmake)
tensor_ir_fetch_cuda_tile(CUDA_TILE_SOURCE_DIR CUDA_TILE_BINARY_DIR)
message(STATUS "Building CUDA Tile IR from: ${CUDA_TILE_SOURCE_DIR}")

# ==============================================================================
# Python
# ==============================================================================
if(TENSOR_IR_ENABLE_BINDINGS_PYTHON)
  if(NOT MLIR_ENABLE_BINDINGS_PYTHON)
    message(
      FATAL_ERROR
        "TensorIR Python bindings require MLIR Python bindings enabled. "
        "Build LLVM with MLIR_ENABLE_BINDINGS_PYTHON=ON.")
  endif()

  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

# ==============================================================================
# Helper Functions
# ==============================================================================

include(cmake/TensorIRLibrary.cmake)

function(add_tensor_ir_dialect_library name)
  add_mlir_library(${name} ${ARGN})
  tensor_ir_register_library(${name})
endfunction()

function(add_tensor_ir_library name)
  add_mlir_library(${name} ${ARGN})
  tensor_ir_register_library(${name})
endfunction()

function(add_tensor_ir_public_c_api_library name)
  add_mlir_public_c_api_library(${name} ${ARGN})
  tensor_ir_register_library(${name})
endfunction()

function(add_tensor_ir_conversion_library name)
  add_mlir_library(${name} ${ARGN})
  tensor_ir_register_library(${name})
endfunction()

function(add_tensor_ir_test_library name)
  add_mlir_library(${name} ${ARGN})
  tensor_ir_register_library(${name} TEST_LIBRARY)
endfunction()

# ==============================================================================
# TensorIR Configuration
# ==============================================================================
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

set(TENSOR_IR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TENSOR_IR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(TENSOR_IR_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")

if(TENSOR_IR_ENABLE_BINDINGS_PYTHON)
  include(cmake/TensorIRFetchDLPack.cmake)
  tensor_ir_fetch_dlpack(DLPACK_SOURCE_DIR DLPACK_BINARY_DIR)
endif()

message(STATUS "TensorIR source directory: ${TENSOR_IR_SOURCE_DIR}")
message(STATUS "TensorIR binary directory: ${TENSOR_IR_BINARY_DIR}")

include_directories(
  "${TENSOR_IR_SOURCE_DIR}/include" "${TENSOR_IR_BINARY_DIR}/include"
  "${CUDA_TILE_SOURCE_DIR}/include" "${CUDA_TILE_BINARY_DIR}/include")

set(TENSOR_IR_TOOLS_DIR "${LLVM_RUNTIME_OUTPUT_INTDIR}")
set(TENSOR_IR_LIBRARY_DIR "${LLVM_LIBRARY_OUTPUT_INTDIR}")

add_subdirectory(include/tensor_ir)
add_subdirectory(lib)
if(TENSOR_IR_INCLUDE_TESTS)
  add_subdirectory(test/lib)
endif()
add_subdirectory(tools)
if(TENSOR_IR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()
if(TENSOR_IR_INCLUDE_TESTS)
  add_subdirectory(test)
endif()

# ==============================================================================
# Installation
# ==============================================================================

include(GNUInstallDirs)

install(TARGETS tensor_ir-compiler tensor_ir-opt
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
