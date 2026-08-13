# Fetch the DLPack headers required by TensorIR's Python bindings.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/TensorIRDependencyPins.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/TensorIRFetchContent.cmake")

set(TENSOR_IR_DLPACK_SOURCE_URL ""
    CACHE STRING "Override the pinned DLPack source archive URL")

function(tensor_ir_fetch_dlpack source_dir_var binary_dir_var)
  if(TENSOR_IR_DLPACK_SOURCE_URL)
    set(source_url "${TENSOR_IR_DLPACK_SOURCE_URL}")
  else()
    set(source_url
        "https://github.com/dmlc/dlpack/archive/${TENSOR_IR_PINNED_DLPACK_COMMIT}.tar.gz"
    )
  endif()

  message(STATUS "Fetching DLPack from ${source_url}")
  # DLPack is consumed headers-only through dlpack_includes. SOURCE_SUBDIR
  # prevents FetchContent_MakeAvailable() from adding DLPack's install rules.
  tensor_ir_fetch_content(
    tensor_ir_dlpack dlpack_source_dir dlpack_binary_dir
    POPULATE_ONLY
    SOURCE_SUBDIR include
    FETCH_CONTENT_ARGS
      URL "${source_url}"
      URL_HASH "SHA256=${TENSOR_IR_PINNED_DLPACK_ARCHIVE_SHA256}")

  if(NOT EXISTS "${dlpack_source_dir}/include/dlpack/dlpack.h")
    message(FATAL_ERROR "Fetched DLPack source does not contain dlpack.h: "
                        "${dlpack_source_dir}")
  endif()

  if(NOT TARGET dlpack_includes)
    add_library(dlpack_includes INTERFACE)
    target_include_directories(
      dlpack_includes
      INTERFACE "$<BUILD_INTERFACE:${dlpack_source_dir}/include>")
  endif()

  set(${source_dir_var} "${dlpack_source_dir}" PARENT_SCOPE)
  set(${binary_dir_var} "${dlpack_binary_dir}" PARENT_SCOPE)
endfunction()
