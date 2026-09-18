set(LLVM_TARGET_DEFINITIONS Enums.td)
mlir_tablegen(OptionsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OptionsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(TensorIR CompilerOptions.h.inc -gen-component-options)
add_public_tablegen_target(NVTensorIRCompilerOptionsIncGen)
add_dependencies(mlir-headers NVTensorIRCompilerOptionsIncGen)
