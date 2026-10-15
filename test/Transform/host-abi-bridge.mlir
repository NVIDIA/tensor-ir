// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics --tir-host-abi-bridge | FileCheck %s


module attributes {gpu.container_module} {

  // CHECK-LABEL: func.func private @dispatch(
  // CHECK-SAME: %{{[^:]+}}: !llvm.ptr {nv_tensor_ir.runtime_context}, %{{[^:]+}}: memref<4xf32, #ptr.generic_space>, %{{[^:]+}}: f32) -> i32
  // CHECK: %[[OK:.*]] = arith.constant 0 : i32
  // CHECK: return %[[OK]] : i32
  func.func @dispatch(%a: memref<4xf32, #ptr.generic_space>, %scale: f32) {
    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@first blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
        args(%a : memref<4xf32, #ptr.generic_space>, %scale : f32)
    gpu.launch_func @kernels::@second blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
    gpu.launch_func @kernels::@first blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
        args(%a : memref<4xf32, #ptr.generic_space>, %scale : f32)
    return
  }

  // CHECK-LABEL: func.func private @cycle1(
  // CHECK-SAME: %[[CTX:[^:]+]]: !llvm.ptr {nv_tensor_ir.runtime_context}, %[[A:[^:]+]]: memref<4xf32, #ptr.generic_space>, %[[SCALE:[^:]+]]: f32) -> i32
  // CHECK: %[[S2:.*]] = call @cycle2(%[[CTX]], %[[A]], %[[SCALE]]) : (!llvm.ptr, memref<4xf32, #ptr.generic_space>, f32) -> i32
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK: %[[FAILED2:.*]] = arith.cmpi ne, %[[S2]], %[[ZERO]] : i32
  // CHECK: cf.cond_br %[[FAILED2]], ^[[FAIL:bb[0-9]+]](%[[S2]] : i32), ^[[NEXT:bb[0-9]+]]
  // CHECK: ^[[NEXT]]:
  // CHECK: %[[S1:.*]] = call @cycle1(%[[CTX]], %[[A]], %[[SCALE]]) : (!llvm.ptr, memref<4xf32, #ptr.generic_space>, f32) -> i32
  // CHECK: %[[FAILED1:.*]] = arith.cmpi ne, %[[S1]], %{{.*}} : i32
  // CHECK: cf.cond_br %[[FAILED1]], ^[[FAIL]](%[[S1]] : i32), ^[[DONE:bb[0-9]+]]
  // CHECK: ^[[DONE]]:
  // CHECK: %[[OK:.*]] = arith.constant 0 : i32
  // CHECK: return %[[OK]] : i32
  // CHECK: ^[[FAIL]](%[[STATUS:[^:]+]]: i32):
  // CHECK-NEXT: return %[[STATUS]] : i32
  func.func @cycle1(%a: memref<4xf32, #ptr.generic_space>, %scale: f32) {
    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@first blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
        args(%a : memref<4xf32, #ptr.generic_space>, %scale : f32)
    call @cycle2(%a, %scale) : (memref<4xf32, #ptr.generic_space>, f32) -> ()
    call @cycle1(%a, %scale) : (memref<4xf32, #ptr.generic_space>, f32) -> ()
    return
  }

  // CHECK-LABEL: func.func private @cycle2(
  // CHECK-SAME: %{{[^:]+}}: !llvm.ptr {nv_tensor_ir.runtime_context}, %{{[^:]+}}: memref<4xf32, #ptr.generic_space>, %{{[^:]+}}: f32) -> i32
  func.func @cycle2(%a: memref<4xf32, #ptr.generic_space>, %scale: f32) {
    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@second blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
    return
  }

  // Helpers take the context and return a status even without launches.
  // CHECK-LABEL: func.func private @helper(
  // CHECK-SAME: %{{[^:]+}}: !llvm.ptr {nv_tensor_ir.runtime_context}, %[[X:[^:]+]]: i32) -> (i32, i32)
  // CHECK: %[[OK:.*]] = arith.constant 0 : i32
  // CHECK: return %[[OK]], %[[X]] : i32, i32
  func.func private @helper(%x: i32) -> i32 { return %x : i32 }

  // Declarations keep the external ABI.
  // CHECK: func.func private @external(i32) -> i32
  func.func private @external(i32) -> i32

  // Existing results follow the status.
  // CHECK-LABEL: func.func private @compute(
  // CHECK-SAME: %{{[^:]+}}: !llvm.ptr {nv_tensor_ir.runtime_context}, %[[X:[^:]+]]: i32) -> (i32, i32)
  // CHECK: %[[OK:.*]] = arith.constant 0 : i32
  // CHECK: return %[[OK]], %[[X]] : i32, i32
  func.func private @compute(%x: i32) -> i32 {
    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@second blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
    return %x : i32
  }

  // The failure block returns poison for the caller's other results.
  // CHECK-LABEL: func.func private @forward(
  // CHECK-SAME: %[[CTX:[^:]+]]: !llvm.ptr {nv_tensor_ir.runtime_context}, %[[X:[^:]+]]: i32) -> (i32, i32)
  // CHECK: %[[RES:.*]]:2 = call @compute(%[[CTX]], %[[X]]) : (!llvm.ptr, i32) -> (i32, i32)
  // CHECK: %[[FAILED:.*]] = arith.cmpi ne, %[[RES]]#0, %{{.*}} : i32
  // CHECK: cf.cond_br %[[FAILED]], ^[[FAIL:bb[0-9]+]](%[[RES]]#0 : i32), ^[[NEXT:bb[0-9]+]]
  // CHECK: ^[[NEXT]]:
  // CHECK: %[[OK:.*]] = arith.constant 0 : i32
  // CHECK: return %[[OK]], %[[RES]]#1 : i32, i32
  // CHECK: ^[[FAIL]](%[[STATUS:[^:]+]]: i32):
  // CHECK: %[[POISON:.*]] = ub.poison : i32
  // CHECK: return %[[STATUS]], %[[POISON]] : i32, i32
  func.func private @forward(%x: i32) -> i32 {
    %y = call @compute(%x) : (i32) -> i32
    return %y : i32
  }

  // A public function takes the context and returns a status even without
  // launches.
  // CHECK-LABEL: func.func private @no_launch(
  // CHECK-SAME: %[[CTX:[^:]+]]: !llvm.ptr {nv_tensor_ir.runtime_context}, %[[X:[^:]+]]: i32) -> i32
  // CHECK: %[[RES:.*]]:2 = call @forward(%[[CTX]], %[[X]]) : (!llvm.ptr, i32) -> (i32, i32)
  // CHECK: %[[FAILED:.*]] = arith.cmpi ne, %[[RES]]#0, %{{.*}} : i32
  // CHECK: cf.cond_br %[[FAILED]], ^[[FAIL:bb[0-9]+]](%[[RES]]#0 : i32), ^[[NEXT:bb[0-9]+]]
  // CHECK: ^[[NEXT]]:
  // CHECK: %[[HELPER:.*]]:2 = call @helper(%[[CTX]], %[[RES]]#1) : (!llvm.ptr, i32) -> (i32, i32)
  // CHECK: %[[HELPER_FAILED:.*]] = arith.cmpi ne, %[[HELPER]]#0, %{{.*}} : i32
  // CHECK: cf.cond_br %[[HELPER_FAILED]], ^[[FAIL]](%[[HELPER]]#0 : i32), ^[[AFTER_HELPER:bb[0-9]+]]
  // CHECK: ^[[AFTER_HELPER]]:
  // CHECK-NEXT: call @external(%[[HELPER]]#1) : (i32) -> i32
  // CHECK: %[[OK:.*]] = arith.constant 0 : i32
  // CHECK: return %[[OK]] : i32
  // CHECK: ^[[FAIL]](%[[STATUS:[^:]+]]: i32):
  // CHECK-NEXT: return %[[STATUS]] : i32
  func.func @no_launch(%x: i32) {
    %y = call @forward(%x) : (i32) -> i32
    %z = call @helper(%y) : (i32) -> i32
    %w = call @external(%z) : (i32) -> i32
    return
  }

  // Device kernels keep their signatures.
  // CHECK-LABEL: gpu.module @kernels
  // CHECK: func.func @first(%{{[^:]+}}: memref<4xf32, #ptr.generic_space>, %{{[^:]+}}: f32) attributes {gpu.kernel}
  // CHECK: func.func @second() attributes {gpu.kernel}
  gpu.module @kernels {
    func.func @first(%a: memref<4xf32, #ptr.generic_space>, %scale: f32) attributes {gpu.kernel} { return }
    func.func @second() attributes {gpu.kernel} { return }
  }

}

// Each public function gets a C entry wrapper. The wrapper passes its runtime
// argument as the context, decodes packedArgs in argument order, and returns
// the function's status. A rank-1 memref takes base pointer, size, and stride
// slots; a scalar takes one slot.

// CHECK-LABEL: func.func @tensor_ir_host_entry_dispatch_v1(
// CHECK-SAME: %[[RUNTIME:[^:]+]]: !llvm.ptr, %[[PACKED:[^:]+]]: !llvm.ptr, %{{[^:]+}}: i32) -> i32
// CHECK: %[[BASE_SLOT:.*]] = llvm.getelementptr %[[PACKED]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK: %[[BASE_BITS:.*]] = llvm.load %[[BASE_SLOT]] : !llvm.ptr -> i64
// CHECK: %[[BASE:.*]] = llvm.inttoptr %[[BASE_BITS]] : i64 to !llvm.ptr
// CHECK: llvm.insertvalue %[[BASE]], %{{.*}}[0]
// CHECK: llvm.insertvalue %[[BASE]], %{{.*}}[1]
// CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : {{index|i64}}) : i64
// CHECK: llvm.insertvalue %[[OFFSET]], %{{.*}}[2]
// CHECK: %[[SIZE_SLOT:.*]] = llvm.getelementptr %[[PACKED]][1]
// CHECK: %[[SIZE:.*]] = llvm.load %[[SIZE_SLOT]] : !llvm.ptr -> i64
// CHECK: %[[STRIDE_SLOT:.*]] = llvm.getelementptr %[[PACKED]][2]
// CHECK: %[[STRIDE:.*]] = llvm.load %[[STRIDE_SLOT]] : !llvm.ptr -> i64
// CHECK: %[[WITH_SIZE:.*]] = llvm.insertvalue %[[SIZE]], %{{.*}}[3, 0]
// CHECK: %[[DESC:.*]] = llvm.insertvalue %[[STRIDE]], %[[WITH_SIZE]][4, 0]
// CHECK: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[DESC]] : {{.*}} to memref<4xf32, #ptr.generic_space>
// CHECK: %[[SCALE_SLOT:.*]] = llvm.getelementptr %[[PACKED]][3]
// CHECK: %[[SCALE_BITS:.*]] = llvm.load %[[SCALE_SLOT]] : !llvm.ptr -> i64
// CHECK: %[[SCALE_I32:.*]] = llvm.trunc %[[SCALE_BITS]] : i64 to i32
// CHECK: %[[SCALE:.*]] = llvm.bitcast %[[SCALE_I32]] : i32 to f32
// CHECK: %[[STATUS:.*]] = call @dispatch(%[[RUNTIME]], %[[MEMREF]], %[[SCALE]]) : (!llvm.ptr, memref<4xf32, #ptr.generic_space>, f32) -> i32
// CHECK: return %[[STATUS]] : i32

// CHECK-LABEL: func.func @tensor_ir_host_entry_cycle1_v1(
// CHECK-SAME: %[[RUNTIME:[^:]+]]: !llvm.ptr
// CHECK: %[[STATUS:.*]] = call @cycle1(%[[RUNTIME]], %{{.*}}, %{{.*}}) : (!llvm.ptr, memref<4xf32, #ptr.generic_space>, f32) -> i32
// CHECK: return %[[STATUS]] : i32

// CHECK-LABEL: func.func @tensor_ir_host_entry_cycle2_v1(
// CHECK-SAME: %[[RUNTIME:[^:]+]]: !llvm.ptr
// CHECK: %[[STATUS:.*]] = call @cycle2(%[[RUNTIME]], %{{.*}}, %{{.*}}) : (!llvm.ptr, memref<4xf32, #ptr.generic_space>, f32) -> i32
// CHECK: return %[[STATUS]] : i32
// CHECK-NOT: func.func

// CHECK-LABEL: func.func @tensor_ir_host_entry_no_launch_v1(
// CHECK-SAME: %[[RUNTIME:[^:]+]]: !llvm.ptr, %[[PACKED:[^:]+]]: !llvm.ptr
// CHECK: %[[X_SLOT:.*]] = llvm.getelementptr %[[PACKED]][0]
// CHECK: %[[X_BITS:.*]] = llvm.load %[[X_SLOT]] : !llvm.ptr -> i64
// CHECK: %[[X:.*]] = llvm.trunc %[[X_BITS]] : i64 to i32
// CHECK: %[[STATUS:.*]] = call @no_launch(%[[RUNTIME]], %[[X]]) : (!llvm.ptr, i32) -> i32
// CHECK: return %[[STATUS]] : i32
// CHECK-NOT: func.func

// -----

// expected-error @below {{expects a public host function to return nothing}}
func.func @bad_result(%x: f32) -> f32 { return %x : f32 }

// -----

func.func @leaf() { return }

func.func @nested(%cond: i1) {
  scf.if %cond {
    // expected-error @below {{calls a function that takes the runtime context, so it must be directly in a func.func body; lower SCF to CF first}}
    func.call @leaf() : () -> ()
  }
  return
}
