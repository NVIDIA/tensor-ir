# Common FetchContent handling for TensorIR dependencies.

include_guard(GLOBAL)

include(FetchContent)

# tensor_ir_fetch_content(
#   <name> <source-dir-var> <binary-dir-var>
#   [POPULATE_ONLY]
#   [SOURCE_SUBDIR <relative-directory>]
#   FETCH_CONTENT_ARGS <FetchContent_Declare arguments>...)
#
# CMake 3.28 added FetchContent's EXCLUDE_FROM_ALL support. Older versions need
# explicit population and add_subdirectory() to keep dependency install rules
# out of TensorIR's install. POPULATE_ONLY instead uses SOURCE_SUBDIR to prevent
# MakeAvailable from adding the dependency in every supported CMake version.
function(tensor_ir_fetch_content _tensor_ir_content_name
         _tensor_ir_source_dir_var _tensor_ir_binary_dir_var)
  set(_tensor_ir_options POPULATE_ONLY)
  set(_tensor_ir_one_value_arguments SOURCE_SUBDIR)
  set(_tensor_ir_multi_value_arguments FETCH_CONTENT_ARGS)
  cmake_parse_arguments(
    PARSE_ARGV 3 _tensor_ir_arg "${_tensor_ir_options}"
    "${_tensor_ir_one_value_arguments}" "${_tensor_ir_multi_value_arguments}")

  if(_tensor_ir_arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown arguments for ${_tensor_ir_content_name}: "
                        "${_tensor_ir_arg_UNPARSED_ARGUMENTS}")
  endif()
  if(_tensor_ir_arg_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "Missing values for ${_tensor_ir_content_name}: "
                        "${_tensor_ir_arg_KEYWORDS_MISSING_VALUES}")
  endif()
  if(NOT _tensor_ir_arg_FETCH_CONTENT_ARGS)
    message(
      FATAL_ERROR "${_tensor_ir_content_name} requires FETCH_CONTENT_ARGS")
  endif()
  if(_tensor_ir_arg_POPULATE_ONLY AND NOT _tensor_ir_arg_SOURCE_SUBDIR)
    message(
      FATAL_ERROR
        "${_tensor_ir_content_name} POPULATE_ONLY requires SOURCE_SUBDIR")
  endif()

  set(_tensor_ir_fetch_content_arguments
      ${_tensor_ir_arg_FETCH_CONTENT_ARGS})
  if(_tensor_ir_arg_SOURCE_SUBDIR)
    list(APPEND _tensor_ir_fetch_content_arguments SOURCE_SUBDIR
         "${_tensor_ir_arg_SOURCE_SUBDIR}")
  endif()
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28
     AND NOT _tensor_ir_arg_POPULATE_ONLY)
    list(APPEND _tensor_ir_fetch_content_arguments EXCLUDE_FROM_ALL)
  endif()
  FetchContent_Declare(${_tensor_ir_content_name}
                       ${_tensor_ir_fetch_content_arguments})

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    FetchContent_MakeAvailable(${_tensor_ir_content_name})
  else()
    FetchContent_GetProperties(${_tensor_ir_content_name})
    if(NOT "${${_tensor_ir_content_name}_POPULATED}")
      FetchContent_Populate(${_tensor_ir_content_name})
      if(NOT _tensor_ir_arg_POPULATE_ONLY)
        set(_tensor_ir_source_subdir
            "${${_tensor_ir_content_name}_SOURCE_DIR}")
        if(_tensor_ir_arg_SOURCE_SUBDIR)
          string(APPEND _tensor_ir_source_subdir
                 "/${_tensor_ir_arg_SOURCE_SUBDIR}")
        endif()
        add_subdirectory("${_tensor_ir_source_subdir}"
                         "${${_tensor_ir_content_name}_BINARY_DIR}"
                         EXCLUDE_FROM_ALL)
      endif()
    endif()
  endif()

  FetchContent_GetProperties(${_tensor_ir_content_name})
  set(${_tensor_ir_source_dir_var}
      "${${_tensor_ir_content_name}_SOURCE_DIR}" PARENT_SCOPE)
  set(${_tensor_ir_binary_dir_var}
      "${${_tensor_ir_content_name}_BINARY_DIR}" PARENT_SCOPE)
endfunction()
