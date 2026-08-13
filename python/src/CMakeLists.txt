# Declare the pure-Python and generated sources that make up nv_tensor_ir.
# Paths below are relative to src/nv_tensor_ir so the source tree mirrors the
# package layout materialized in the build and install trees.

declare_mlir_python_sources(TensorIRPythonSources)

declare_mlir_python_sources(TensorIRPythonSources.Dialects
  ADD_TO_PARENT TensorIRPythonSources
)

declare_mlir_python_sources(TensorIRPythonPublicSources
  ROOT_DIR "${_tensor_ir_python_source_dir}"
  SOURCES
    runtime.py
  SOURCES_GLOB
    dsl/*.py
)

set(_tensor_ir_python_dialect_dependencies
  "${TENSOR_IR_SOURCE_DIR}/include/tensor_ir/Dialect/TensorOps.td"
  "${TENSOR_IR_SOURCE_DIR}/include/tensor_ir/Dialect/TensorAttrs.td"
)

# GEN_ENUM_BINDINGS_TD_FILE is resolved against ROOT_DIR, so it has to be a
# relative path. Derive it rather than hand-counting `../`s.
file(RELATIVE_PATH _tensor_ir_python_enum_td_file
  "${_tensor_ir_python_mlir_source_dir}"
  "${TENSOR_IR_SOURCE_DIR}/include/tensor_ir/Dialect/TensorAttrs.td")

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TensorIRPythonSources.Dialects
  ROOT_DIR "${_tensor_ir_python_mlir_source_dir}"
  TD_FILE dialects/TensorOps.td
  GEN_ENUM_BINDINGS_TD_FILE "${_tensor_ir_python_enum_td_file}"
  SOURCES
    dialects/nv_tensor_ir.py
  DIALECT_NAME nv_tensor_ir
  DEPENDS
    ${_tensor_ir_python_dialect_dependencies}
)

