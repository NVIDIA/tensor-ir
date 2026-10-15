// RUN: tensor_ir-opt %s --tir-linalg-pipeline='stop-after=linalg' \
// RUN:   | FileCheck %s --check-prefix=LINALG
// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(tir-linalg-pipeline{stop-after=linalg-opt})' \
// RUN:   | FileCheck %s --check-prefix=OPT
// RUN: tensor_ir-opt %s --tir-linalg-pipeline='stop-after=llvm' \
// RUN:   | FileCheck %s --check-prefix=LLVM --implicit-check-not=linalg. --implicit-check-not=memref.
// RUN: tensor_ir-opt %s --tir-linalg-pipeline \
// RUN:   | FileCheck %s --check-prefix=LLVM --implicit-check-not=linalg. --implicit-check-not=memref.
// RUN: tensor_ir-opt %s --tir-linalg-pipeline='stop-after=linalg' --dump-pass-pipeline -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EARLY-PIPELINE
// RUN: tensor_ir-opt %s --tir-linalg-pipeline --dump-pass-pipeline -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LLVM-PIPELINE --implicit-check-not=convert-arith-to-llvm --implicit-check-not=convert-func-to-llvm --implicit-check-not=convert-math-to-llvm --implicit-check-not=finalize-memref-to-llvm --implicit-check-not=convert-cf-to-llvm

// EARLY-PIPELINE: Pass Manager with 1 passes:
// EARLY-PIPELINE-NEXT: builtin.module(
// EARLY-PIPELINE-NEXT: convert-tensor-to-linalg
// EARLY-PIPELINE-NEXT: )
// LLVM-PIPELINE: convert-to-llvm

// LINALG-LABEL: func.func @fuse_pointwise(
// LINALG: linalg.generic
// LINALG: arith.addf
// LINALG: linalg.generic
// LINALG: arith.mulf
// LINALG: return

// OPT-LABEL: func.func @fuse_pointwise(
// OPT: linalg.generic
// OPT: arith.addf
// OPT: arith.mulf
// OPT: linalg.yield
// OPT-NOT: linalg.generic
// OPT: return

// LLVM-LABEL: llvm.func @fuse_pointwise(
// LLVM: llvm.fadd
// LLVM: llvm.fmul
// LLVM: llvm.return
nv_tensor_ir.graph @fuse_pointwise(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %sum = add %a, %b : tensor<4xf32>
  %product = mul %sum, %b : tensor<4xf32>
  results %product : tensor<4xf32>
}
