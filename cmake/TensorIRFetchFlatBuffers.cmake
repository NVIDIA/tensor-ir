# Fetch and build FlatBuffers for TensorIR, reusing a parent project's
# flatc/flatbuffers targets when already present in the build.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/TensorIRDependencyPins.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/TensorIRFetchContent.cmake")

set(TENSOR_IR_FLATBUFFERS_SOURCE_URL ""
    CACHE STRING "Override the pinned FlatBuffers source archive URL")

function(tensor_ir_fetch_flatbuffers)
  # Reuse an existing `flatbuffers` target (e.g. from an enclosing project
  # that already fetched one) instead of fetching a second copy.
  if(TARGET flatbuffers)
    return()
  endif()

  if(TENSOR_IR_FLATBUFFERS_SOURCE_URL)
    set(source_url "${TENSOR_IR_FLATBUFFERS_SOURCE_URL}")
  else()
    set(source_url
        "https://github.com/google/flatbuffers/archive/${TENSOR_IR_PINNED_FLATBUFFERS_COMMIT}.tar.gz"
    )
  endif()

  set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(FLATBUFFERS_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(FLATBUFFERS_BUILD_GRPCTEST OFF CACHE BOOL "" FORCE)
  set(FLATBUFFERS_BUILD_SHAREDLIB OFF CACHE BOOL "" FORCE)
  set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)

  if(CMAKE_CROSSCOMPILING)
    # flatc has no cross-build story; tensor_ir_flatc_command() falls back
    # to a prebuilt host flatc when no `flatc` target exists.
    set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
  else()
    set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
  endif()

  message(STATUS "Fetching FlatBuffers from ${source_url}")
  tensor_ir_fetch_content(
    tensor_ir_flatbuffers flatbuffers_source_dir flatbuffers_binary_dir
    FETCH_CONTENT_ARGS
      URL "${source_url}"
      URL_HASH "SHA256=${TENSOR_IR_PINNED_FLATBUFFERS_ARCHIVE_SHA256}")
endfunction()

# Resolves how to invoke flatc: the `flatc` target if one exists, else a
# prebuilt host flatc. Resolved lazily (not in tensor_ir_fetch_flatbuffers())
# so it still works when that function short-circuits without building a
# `flatc` target itself.
function(tensor_ir_flatc_command OUTPUT_VAR)
  if(TARGET flatc)
    set(${OUTPUT_VAR} flatc PARENT_SCOPE)
    return()
  endif()
  if(NOT TENSOR_IR_HOST_FLATC_EXECUTABLE)
    find_program(TENSOR_IR_HOST_FLATC_EXECUTABLE flatc REQUIRED)
  endif()
  set(${OUTPUT_VAR} "${TENSOR_IR_HOST_FLATC_EXECUTABLE}" PARENT_SCOPE)
endfunction()
