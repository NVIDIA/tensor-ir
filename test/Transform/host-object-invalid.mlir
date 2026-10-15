// RUN: tensor_ir-opt %s --allow-unregistered-dialect --split-input-file --tir-host-to-object --verify-diagnostics

// expected-error @below {{unsupported native host import: external}}
module attributes {nv_tensor_ir.host_entrypoints = ["entry"]} {
  llvm.func @external()
  llvm.func @entry() {
    llvm.call @external() : () -> ()
    llvm.return
  }
}

// -----

// expected-error @below {{native host LLVM IR translation failed}}
module attributes {nv_tensor_ir.host_entrypoints = ["entry"]} {
  llvm.func @entry() {
    // expected-error @below {{cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration}}
    "test.untranslatable"() : () -> ()
    llvm.return
  }
}
