// RUN: tensor_ir-opt %s --outline-tensor-ir-kernel --split-input-file --verify-diagnostics

func.func @wrong_rank() {
  // expected-error @below {{expected a rank-one flattened grid}}
  scf.forall (%x, %y) in (2, 2) {
  } {mapping = [#gpu.block<x>, #gpu.block<y>]}
  return
}

// -----

func.func @wrong_mapping() {
  // expected-error @below {{expected mapping = [#gpu.block<x>] on the flattened grid}}
  scf.forall (%x) in (2) {
  } {mapping = [#gpu.block<y>]}
  return
}

// -----

func.func @shared_output(%init: tensor<4xf32>) {
  // expected-error @below {{expected destination passing through memref arguments, not scf.forall shared outputs}}
  %result = scf.forall (%x) in (4) shared_outs(%out = %init) -> tensor<4xf32> {
    %value = tensor.extract_slice %init[%x] [1] [1]
        : tensor<4xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %value into %out[%x] [1] [1]
          : tensor<1xf32> into tensor<4xf32>
    }
  } {mapping = [#gpu.block<x>]}
  return
}

// -----

func.func private @capture_source() -> index

func.func @unexpected_capture(%tiles: index) {
  // expected-error @below {{cannot outline capture because 'func.call' is not a pure, regionless setup operation}}
  %capture = func.call @capture_source() : () -> index
  scf.forall (%x) in (%tiles) {
    %use = arith.addi %x, %capture : index
  } {mapping = [#gpu.block<x>]}
  return
}

// -----

func.func @collision() {
  scf.forall (%x) in (1) {
  } {mapping = [#gpu.block<x>]}
  return
}

// expected-error @below {{cannot reserve kernel symbol @collision; it is already used by 'func.func'}}
func.func @collision_dispatch() {
  scf.forall (%x) in (1) {
  } {mapping = [#gpu.block<x>]}
  return
}

// -----

// expected-error @below {{expected exactly one mapped scf.forall to outline}}
func.func @multiple_foralls() {
  scf.forall (%x) in (1) {
  } {mapping = [#gpu.block<x>]}
  scf.forall (%x) in (1) {
  } {mapping = [#gpu.block<x>]}
  return
}
