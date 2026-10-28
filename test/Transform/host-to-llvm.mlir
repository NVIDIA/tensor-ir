// RUN: tensor_ir-opt %s --split-input-file --tir-host-to-llvm --canonicalize --cse | FileCheck %s --check-prefix=ABI
// RUN: tensor_ir-opt %s --split-input-file --tir-host-to-llvm | FileCheck %s --implicit-check-not=gpu.launch_func --implicit-check-not=builtin.unrealized_conversion_cast

// CHECK-LABEL: llvm.func @negative_index_constant()
//       CHECK: %[[VALUE:.*]] = llvm.mlir.constant(-1 : i64) : i64
//       CHECK: llvm.return %[[VALUE]] : i64
func.func @negative_index_constant() -> index {
  %value = arith.constant -1 : index
  return %value : index
}

// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL: llvm.func @dispatch(
  //  CHECK-SAME: %[[RT:[^:]+]]: !llvm.ptr {nv_tensor_ir.runtime_context}, %[[GRID:[^:]+]]: i64, %[[BLOCK:[^:]+]]: i64, %[[SMEM:[^:]+]]: i32, %[[VALUE:[^:]+]]: f32) -> i32
  //       CHECK: %[[PARAMS_ONE:.*]] = llvm.mlir.constant(1 : i64)
  //       CHECK: %[[PARAMS:.*]] = llvm.alloca %[[PARAMS_ONE]] x !llvm.struct<(array<1 x ptr>, array<1 x i64>)> {alignment = 8 : i64}
  //       CHECK: %[[CONFIG_PTR:.*]] = llvm.alloca %[[PARAMS_ONE]] x !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)> {alignment = 4 : i64}
  //       CHECK: %[[SUCCESS:.*]] = llvm.mlir.constant(0 : i32)
  //       CHECK: %[[GRID32:.*]] = llvm.trunc %[[GRID]] : i64 to i32
  //       CHECK: llvm.insertvalue %[[GRID32]], %{{.*}}[0, 0]
  //       CHECK: %[[BLOCK32:.*]] = llvm.trunc %[[BLOCK]] : i64 to i32
  //       CHECK: llvm.insertvalue %[[BLOCK32]], %{{.*}}[1, 0]
  //       CHECK: %[[CONFIG:.*]] = llvm.insertvalue %[[SMEM]], %{{.*}}[2]
  //       CHECK: llvm.store %[[CONFIG]], %[[CONFIG_PTR]]
  //       CHECK: %[[VALUE_PTR:.*]] = llvm.getelementptr %[[PARAMS]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<1 x ptr>, array<1 x i64>)>
  //       CHECK: llvm.store %[[VALUE]], %[[VALUE_PTR]] : f32, !llvm.ptr
  //       CHECK: %[[PARAM_PTR:.*]] = llvm.getelementptr %[[PARAMS]][0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<1 x ptr>, array<1 x i64>)>
  //       CHECK: llvm.store %[[VALUE_PTR]], %[[PARAM_PTR]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[RUNTIME:.*]] = llvm.load %[[RT]] : !llvm.ptr -> !llvm.struct<(i32, i32, ptr, ptr)>
  //       CHECK: %[[CALLBACK:.*]] = llvm.extractvalue %[[RUNTIME]][3]
  //       CHECK: %[[CONTEXT:.*]] = llvm.extractvalue %[[RUNTIME]][2]
  //       CHECK: %[[ORDINAL:.*]] = llvm.mlir.constant(7 : i32)
  //       CHECK: %[[STATUS:.*]] = llvm.call %[[CALLBACK]](%[[CONTEXT]], %[[ORDINAL]], %[[CONFIG_PTR]], %[[PARAMS]]) : !llvm.ptr, (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
  //       CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
  //       CHECK: %[[OK:.*]] = llvm.icmp "eq" %[[STATUS]], %[[ZERO]]
  //       CHECK: llvm.cond_br %[[OK]], ^[[DONE:bb[0-9]+]], ^[[ERROR:bb[0-9]+]]
  //       CHECK: ^[[DONE]]:
  //       CHECK: llvm.return %[[SUCCESS]] : i32
  //       CHECK: ^[[ERROR]]:
  //       CHECK: %[[FAILED:.*]] = llvm.mlir.constant(-3 : i32)
  //       CHECK: llvm.return %[[FAILED]] : i32
  func.func private @dispatch(
      %runtime: !llvm.ptr {nv_tensor_ir.runtime_context}, %grid: index, %block: index, %smem: i32,
      %value: f32) -> i32 {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    gpu.launch_func @kernels::@kernel blocks in (%grid, %one, %one)
        threads in (%block, %one, %one)
        dynamic_shared_memory_size %smem args(%value : f32)
        {nv_tensor_ir.kernel_ordinal = 7 : i32}
    return %zero : i32
  }
  // CHECK-LABEL: gpu.module @kernels
  //       CHECK: func.func @kernel(%[[VALUE:[^:]+]]: f32) attributes {gpu.kernel, nv_tensor_ir.kernel_ordinal = 7 : i32}
  //  CHECK-NEXT: %{{.*}} = math.exp %[[VALUE]] : f32
  //  CHECK-NEXT: return
  gpu.module @kernels {
    func.func @kernel(%value: f32) attributes {gpu.kernel, nv_tensor_ir.kernel_ordinal = 7 : i32} {
      %x = math.exp %value : f32
      return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // ABI-LABEL: llvm.func @dispatch_identity(
  // ABI: llvm.alloca {{.*}} x !llvm.struct<(array<3 x ptr>, array<3 x i64>)>
  func.func private @dispatch_identity(%runtime: !llvm.ptr {nv_tensor_ir.runtime_context},
      %view: memref<?x?xf32>) -> i32 {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    gpu.launch_func @kernels::@kernel blocks in (%one, %one, %one)
        threads in (%one, %one, %one) args(%view : memref<?x?xf32>)
        {nv_tensor_ir.kernel_ordinal = 0 : i32}
    return %zero : i32
  }
  gpu.module @kernels {
    func.func @kernel(%view: memref<?x?xf32>) attributes {gpu.kernel} {
      return
    }
  }
}

// -----

// A signed-integer scalar kernel argument is converted to the signless
// integer type LLVM dialect ops require before being stored into the
// params slot.

// CHECK-LABEL: llvm.func @dispatch_scalar_int(
//  CHECK-SAME: {{.*}}%[[VALUE:[^:]+]]: i32) -> i32
//       CHECK: llvm.store %[[VALUE]], %{{.*}} : i32, !llvm.ptr
module attributes {gpu.container_module} {
  func.func private @dispatch_scalar_int(
      %runtime: !llvm.ptr {nv_tensor_ir.runtime_context}, %grid: index, %block: index, %smem: i32,
      %value: si32) -> i32 {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    gpu.launch_func @kernels::@kernel blocks in (%grid, %one, %one)
        threads in (%block, %one, %one)
        dynamic_shared_memory_size %smem args(%value : si32)
        {nv_tensor_ir.kernel_ordinal = 3 : i32}
    return %zero : i32
  }
  gpu.module @kernels {
    func.func @kernel(%value: si32) attributes {gpu.kernel, nv_tensor_ir.kernel_ordinal = 3 : i32} {
      return
    }
  }
}
