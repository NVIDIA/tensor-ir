// RUN: tensor_ir-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(tir-static-persistence{sm-count=2}))' --verify-diagnostics

func.func @rank_two() {
  // expected-error @below {{static persistence requires a rank-one destination-free forall}}
  scf.forall (%i, %j) in (4, 4) {
  }
  return
}

// -----

func.func @with_output(%init: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @below {{static persistence requires a rank-one destination-free forall}}
  %result = scf.forall (%i) in (4) shared_outs(%out = %init) -> (tensor<4xf32>) {
    scf.forall.in_parallel {
    }
  }
  return %result : tensor<4xf32>
}
