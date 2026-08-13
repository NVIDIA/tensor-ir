# Fetch the CUDA Tile source required by TensorIR.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/TensorIRDependencyPins.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/TensorIRFetchContent.cmake")

set(TENSOR_IR_CUDA_TILE_SOURCE_URL ""
    CACHE STRING "Override the pinned CUDA Tile source archive URL")

function(tensor_ir_fetch_cuda_tile source_dir_var binary_dir_var)
  if(TENSOR_IR_CUDA_TILE_SOURCE_URL)
    set(source_url "${TENSOR_IR_CUDA_TILE_SOURCE_URL}")
  else()
    set(source_url
        "https://github.com/NVIDIA/cuda-tile/archive/${TENSOR_IR_PINNED_CUDA_TILE_COMMIT}.tar.gz"
    )
  endif()

  message(STATUS "Fetching CUDA Tile from ${source_url}")
  tensor_ir_fetch_content(
    tensor_ir_cuda_tile cuda_tile_source_dir cuda_tile_binary_dir
    FETCH_CONTENT_ARGS
      URL "${source_url}"
      URL_HASH "SHA256=${TENSOR_IR_PINNED_CUDA_TILE_ARCHIVE_SHA256}")

  if(NOT EXISTS "${cuda_tile_source_dir}/CMakeLists.txt")
    message(
      FATAL_ERROR "Fetched CUDA Tile source does not contain CMakeLists.txt: "
                  "${cuda_tile_source_dir}")
  endif()

  set(${source_dir_var} "${cuda_tile_source_dir}" PARENT_SCOPE)
  set(${binary_dir_var} "${cuda_tile_binary_dir}" PARENT_SCOPE)
endfunction()
