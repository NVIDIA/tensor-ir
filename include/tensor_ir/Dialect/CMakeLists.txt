set(LLVM_TARGET_DEFINITIONS TensorOps.td)
mlir_tablegen(TensorOps.h.inc -gen-op-decls)
mlir_tablegen(TensorOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS TensorEnums.td)
mlir_tablegen(TensorEnums.h.inc -gen-enum-decls)
mlir_tablegen(TensorEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS TensorDialect.td)
mlir_tablegen(TensorDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TensorDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS TensorInterfaces.td)
mlir_tablegen(TensorOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TensorOpInterfaces.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS TensorAttrInterfaces.td)
mlir_tablegen(TensorAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TensorAttrInterfaces.cpp.inc -gen-attr-interface-defs)

set(LLVM_TARGET_DEFINITIONS TensorAttrs.td)
mlir_tablegen(TensorAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=nv_tensor_ir)
mlir_tablegen(TensorAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=nv_tensor_ir)

add_public_tablegen_target(TensorIRIncGen)
add_dependencies(mlir-headers TensorIRIncGen)

