// RUN: tensor_ir-opt %s --split-input-file --pass-pipeline='builtin.module(tir-host-to-llvm-pipeline)' \
// RUN:  | FileCheck %s --check-prefix=CHECK --implicit-check-not=gpu.launch_func --implicit-check-not=builtin.unrealized_conversion_cast

module @dynamic attributes {gpu.container_module} {
  // CHECK-LABEL: llvm.func @copy(
  //  CHECK-SAME:   %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[ALLOCATED:.+]]: !llvm.ptr, %[[ALIGNED:.+]]: !llvm.ptr,
  //  CHECK-SAME:   %[[OFFSET:.+]]: i64, %[[SIZE:.+]]: i64, %[[STRIDE:.+]]: i64, %[[SCALE:.+]]: f32) -> i32
  //   CHECK-DAG: %[[LAUNCH_ERROR:.+]] = llvm.mlir.constant(-3 : i32) : i32
  //   CHECK-DAG: %[[OTHER_ID:.+]] = llvm.mlir.constant(1 : i32) : i32
  //   CHECK-DAG: %[[COPY_ID:.+]] = llvm.mlir.constant(0 : i32) : i32
  //   CHECK-DAG: %[[CONFIG_UNDEF:.+]] = llvm.mlir.poison : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //   CHECK-DAG: %[[TRIP_COUNT:.+]] = llvm.mlir.constant(3 : i64) : i64
  //   CHECK-DAG: %[[STEP:.+]] = llvm.mlir.constant(1 : i64) : i64
  //   CHECK-DAG: %[[START:.+]] = llvm.mlir.constant(0 : i64) : i64
  //   CHECK-DAG: %[[CONFIG_PTR:.+]] = llvm.alloca %[[STEP]] x !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  //   CHECK-DAG: %[[PARAMS:.+]] = llvm.alloca %{{.+}} x !llvm.struct<(array<4 x ptr>, array<4 x i64>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  //       CHECK: llvm.br ^[[LOOP:.+]](%[[START]] : i64)
  //       CHECK: ^[[LOOP]](%[[IV:.+]]: i64):
  //       CHECK: %[[IN_BOUNDS:.+]] = llvm.icmp "slt" %[[IV]], %[[TRIP_COUNT]] : i64
  //       CHECK: llvm.cond_br %[[IN_BOUNDS]], ^[[LAUNCH_COPY:.+]], ^[[DONE:.+]]
  //       CHECK: ^[[LAUNCH_COPY]]:
  //       CHECK: %[[SIZE_I32:.+]] = llvm.trunc %[[SIZE]] : i64 to i32
  //       CHECK: %[[GRID_X:.+]] = llvm.insertvalue %[[SIZE_I32]], %[[CONFIG_UNDEF]][0, 0] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[DIM_ONE:.+]] = llvm.trunc %[[STEP]] : i64 to i32
  //       CHECK: %[[GRID_Y:.+]] = llvm.insertvalue %[[DIM_ONE]], %[[GRID_X]][0, 1] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[GRID_Z:.+]] = llvm.insertvalue %[[DIM_ONE]], %[[GRID_Y]][0, 2] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[BLOCK_X:.+]] = llvm.insertvalue %[[DIM_ONE]], %[[GRID_Z]][1, 0] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[BLOCK_Y:.+]] = llvm.insertvalue %[[DIM_ONE]], %[[BLOCK_X]][1, 1] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[BLOCK_Z:.+]] = llvm.insertvalue %[[DIM_ONE]], %[[BLOCK_Y]][1, 2] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[CONFIG:.+]] = llvm.insertvalue %[[COPY_ID]], %[[BLOCK_Z]][2] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: llvm.store %[[CONFIG]], %[[CONFIG_PTR]] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>, !llvm.ptr
  //       CHECK: %[[DATA_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<4 x ptr>, array<4 x i64>)>
  //       CHECK: llvm.store %[[ALIGNED]], %[[DATA_SLOT]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[FIRST_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<4 x ptr>, array<4 x i64>)>
  //       CHECK: llvm.store %[[DATA_SLOT]], %[[FIRST_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[SIZE_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<4 x ptr>, array<4 x i64>)>
  //       CHECK: llvm.store %[[SIZE_I32]], %[[SIZE_SLOT]] : i32, !llvm.ptr
  //       CHECK: %[[SIZE_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<4 x ptr>, array<4 x i64>)>
  //       CHECK: llvm.store %[[SIZE_SLOT]], %[[SIZE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[STRIDE_I32:.+]] = llvm.trunc %[[STRIDE]] : i64 to i32
  //       CHECK: %[[STRIDE_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<4 x ptr>, array<4 x i64>)>
  //       CHECK: llvm.store %[[STRIDE_I32]], %[[STRIDE_SLOT]] : i32, !llvm.ptr
  //       CHECK: %[[STRIDE_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<4 x ptr>, array<4 x i64>)>
  //       CHECK: llvm.store %[[STRIDE_SLOT]], %[[STRIDE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[SCALE_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<4 x ptr>, array<4 x i64>)>
  //       CHECK: llvm.store %[[SCALE]], %[[SCALE_SLOT]] : f32, !llvm.ptr
  //       CHECK: %[[SCALE_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<4 x ptr>, array<4 x i64>)>
  //       CHECK: llvm.store %[[SCALE_SLOT]], %[[SCALE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[RUNTIME0:.+]] = llvm.load %[[RUNTIME_PTR]] : !llvm.ptr -> !llvm.struct<(i32, i32, ptr, ptr)>
  //   CHECK-DAG: %[[CALLBACK0:.+]] = llvm.extractvalue %[[RUNTIME0]][3] : !llvm.struct<(i32, i32, ptr, ptr)>
  //   CHECK-DAG: %[[CONTEXT0:.+]] = llvm.extractvalue %[[RUNTIME0]][2] : !llvm.struct<(i32, i32, ptr, ptr)>
  //       CHECK: %[[STATUS0:.+]] = llvm.call %[[CALLBACK0]](%[[CONTEXT0]], %[[COPY_ID]], %[[CONFIG_PTR]], %[[PARAMS]])
  //       CHECK: %[[OK0:.+]] = llvm.icmp "eq" %[[STATUS0]], %[[COPY_ID]] : i32
  //       CHECK: llvm.cond_br %[[OK0]], ^[[LAUNCH_OTHER:.+]], ^[[ERROR_COPY:.+]]
  //       CHECK: ^[[LAUNCH_OTHER]]:
  //       CHECK: llvm.store %[[CONFIG]], %[[CONFIG_PTR]] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[ALIGNED]], %[[DATA_SLOT]] : !llvm.ptr, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[DATA_SLOT]], %[[FIRST_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.store %[[SIZE_I32]], %[[SIZE_SLOT]] : i32, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[SIZE_SLOT]], %[[SIZE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.store %[[STRIDE_I32]], %[[STRIDE_SLOT]] : i32, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[STRIDE_SLOT]], %[[STRIDE_PARAM]] : !llvm.ptr, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[SCALE]], %[[SCALE_SLOT]] : f32, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[SCALE_SLOT]], %[[SCALE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[RUNTIME1:.+]] = llvm.load %[[RUNTIME_PTR]] : !llvm.ptr -> !llvm.struct<(i32, i32, ptr, ptr)>
  //   CHECK-DAG: %[[CALLBACK1:.+]] = llvm.extractvalue %[[RUNTIME1]][3] : !llvm.struct<(i32, i32, ptr, ptr)>
  //   CHECK-DAG: %[[CONTEXT1:.+]] = llvm.extractvalue %[[RUNTIME1]][2] : !llvm.struct<(i32, i32, ptr, ptr)>
  //       CHECK: %[[STATUS1:.+]] = llvm.call %[[CALLBACK1]](%[[CONTEXT1]], %[[OTHER_ID]], %[[CONFIG_PTR]], %[[PARAMS]])
  //       CHECK: %[[OK1:.+]] = llvm.icmp "eq" %[[STATUS1]], %[[COPY_ID]] : i32
  //       CHECK: llvm.cond_br %[[OK1]], ^[[LAUNCH_COPY_AGAIN:.+]], ^[[ERROR_OTHER:.+]]
  //       CHECK: ^[[LAUNCH_COPY_AGAIN]]:
  //       CHECK: llvm.store %[[CONFIG]], %[[CONFIG_PTR]] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[ALIGNED]], %[[DATA_SLOT]] : !llvm.ptr, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[DATA_SLOT]], %[[FIRST_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.store %[[SIZE_I32]], %[[SIZE_SLOT]] : i32, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[SIZE_SLOT]], %[[SIZE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.store %[[STRIDE_I32]], %[[STRIDE_SLOT]] : i32, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[STRIDE_SLOT]], %[[STRIDE_PARAM]] : !llvm.ptr, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[SCALE]], %[[SCALE_SLOT]] : f32, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[SCALE_SLOT]], %[[SCALE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[RUNTIME2:.+]] = llvm.load %[[RUNTIME_PTR]] : !llvm.ptr -> !llvm.struct<(i32, i32, ptr, ptr)>
  //   CHECK-DAG: %[[CALLBACK2:.+]] = llvm.extractvalue %[[RUNTIME2]][3] : !llvm.struct<(i32, i32, ptr, ptr)>
  //   CHECK-DAG: %[[CONTEXT2:.+]] = llvm.extractvalue %[[RUNTIME2]][2] : !llvm.struct<(i32, i32, ptr, ptr)>
  //       CHECK: %[[STATUS2:.+]] = llvm.call %[[CALLBACK2]](%[[CONTEXT2]], %[[COPY_ID]], %[[CONFIG_PTR]], %[[PARAMS]])
  //       CHECK: %[[OK2:.+]] = llvm.icmp "eq" %[[STATUS2]], %[[COPY_ID]] : i32
  //       CHECK: llvm.cond_br %[[OK2]], ^[[LATCH:.+]], ^[[ERROR_COPY_AGAIN:.+]]
  //       CHECK: ^[[LATCH]]:
  //       CHECK: %[[NEXT_IV:.+]] = llvm.add %[[IV]], %[[STEP]] : i64
  //       CHECK: llvm.br ^[[LOOP]](%[[NEXT_IV]] : i64)
  //       CHECK: ^[[DONE]]:
  //       CHECK: llvm.return %[[COPY_ID]] : i32
  //       CHECK: ^[[ERROR_COPY]]:
  //       CHECK: llvm.return %[[LAUNCH_ERROR]] : i32
  //       CHECK: ^[[ERROR_OTHER]]:
  //       CHECK: llvm.return %[[LAUNCH_ERROR]] : i32
  //       CHECK: ^[[ERROR_COPY_AGAIN]]:
  //       CHECK: llvm.return %[[LAUNCH_ERROR]] : i32
  func.func @copy(
      %a: memref<?xf32, strided<[?]>>, %scale: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %n = memref.dim %a, %c0 : memref<?xf32, strided<[?]>>
    scf.for %i = %c0 to %c3 step %c1 {

      gpu.launch_func @kernels::@copy blocks in (%n, %c1, %c1)
          threads in (%c1, %c1, %c1)
          args(%a : memref<?xf32, strided<[?]>>, %scale : f32)

      gpu.launch_func @kernels::@other blocks in (%n, %c1, %c1)
          threads in (%c1, %c1, %c1)
          args(%a : memref<?xf32, strided<[?]>>, %scale : f32)

      gpu.launch_func @kernels::@copy blocks in (%n, %c1, %c1)
          threads in (%c1, %c1, %c1)
          args(%a : memref<?xf32, strided<[?]>>, %scale : f32)
    }
    return
  }
  // CHECK-LABEL: gpu.module @kernels
  //       CHECK: func.func @copy(%[[A:[^:]+]]: memref<?xf32, strided<[?]>>, %[[SCALE:[^:]+]]: f32) attributes {gpu.kernel, nv_tensor_ir.kernel_ordinal = 0 : i32}
  //   CHECK-DAG: %[[ZERO:[^ ]+]] = arith.constant 0 : index
  //   CHECK-DAG: %[[ONE:[^ ]+]] = arith.constant 1 : index
  //       CHECK: scf.for %[[IV:[^ ]+]] = %[[ZERO]] to %[[ONE]] step %[[ONE]] {
  //       CHECK: %[[UNUSED:[^ ]+]] = arith.constant 7 : i32
  //       CHECK: return
  //       CHECK: func.func @other(%[[A:[^:]+]]: memref<?xf32, strided<[?]>>, %[[SCALE:[^:]+]]: f32) attributes {gpu.kernel, nv_tensor_ir.kernel_ordinal = 1 : i32}
  //       CHECK: return
  gpu.module @kernels {
    func.func @copy(%a: memref<?xf32, strided<[?]>>,
                    %scale: f32) attributes {gpu.kernel} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c1 step %c1 {
        %unused = arith.constant 7 : i32
      }
      return
    }
    func.func @other(%a: memref<?xf32, strided<[?]>>,
                     %scale: f32) attributes {gpu.kernel} {
      return
    }
  }
  // Decode host slots into dispatch arguments through the retained ABI bridge.
  // CHECK-LABEL: llvm.func @tensor_ir_host_entry_copy_v1(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[PACKED:.+]]: !llvm.ptr, %[[COUNT:.+]]: i32) -> i32
  //   CHECK-DAG: %[[OFFSET:.+]] = llvm.mlir.constant(0 : i64) : i64
  //   CHECK-NOT: llvm.alloca
  //       CHECK: %[[PTR_BITS:.+]] = llvm.load %[[PACKED]] : !llvm.ptr -> i64
  //       CHECK: %[[DATA:.+]] = llvm.inttoptr %[[PTR_BITS]] : i64 to !llvm.ptr
  //       CHECK: %[[SIZE_PTR:.+]] = llvm.getelementptr %[[PACKED]][1] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[SIZE:.+]] = llvm.load %[[SIZE_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[STRIDE_PTR:.+]] = llvm.getelementptr %[[PACKED]][2] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[STRIDE:.+]] = llvm.load %[[STRIDE_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[SCALE_PTR:.+]] = llvm.getelementptr %[[PACKED]][3] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[SCALE_BITS:.+]] = llvm.load %[[SCALE_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[SCALE_I32:.+]] = llvm.trunc %[[SCALE_BITS]] : i64 to i32
  //       CHECK: %[[SCALE:.+]] = llvm.bitcast %[[SCALE_I32]] : i32 to f32
  //       CHECK: %[[STATUS:.+]] = llvm.call @copy(%[[RUNTIME_PTR]], %[[DATA]], %[[DATA]], %[[OFFSET]], %[[SIZE]], %[[STRIDE]], %[[SCALE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32) -> i32
  //       CHECK: llvm.return %[[STATUS]] : i32
}

// -----

module @conditional attributes {gpu.container_module} {
  // CHECK-LABEL: llvm.func @conditional(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[COUNT:.+]]: i64) -> i32
  //   CHECK-DAG: %[[LAUNCH_ERROR:.+]] = llvm.mlir.constant(-3 : i32) : i32
  //   CHECK-DAG: %[[SUCCESS:.+]] = llvm.mlir.constant(0 : i32) : i32
  //   CHECK-DAG: %[[INDEX_ZERO:.+]] = llvm.mlir.constant(0 : i64) : i64
  //       CHECK: %[[ACTIVE:.+]] = llvm.icmp "sgt" %[[COUNT]], %[[INDEX_ZERO]] : i64
  //       CHECK: llvm.cond_br %[[ACTIVE]], ^[[LAUNCH:.+]], ^[[DONE:.+]]
  //       CHECK: ^[[LAUNCH]]:
  //       CHECK: llvm.store %[[COUNT]], %{{.*}} : i64, !llvm.ptr
  //       CHECK: %[[STATUS:.*]] = llvm.call %{{.*}}(%{{.*}})
  //       CHECK: %[[OK:.+]] = llvm.icmp "eq" %[[STATUS]], %[[SUCCESS]] : i32
  //       CHECK: llvm.cond_br %[[OK]], ^[[CONTINUE:.+]], ^[[ERROR:.+]]
  //       CHECK: ^[[CONTINUE]]:
  //       CHECK: llvm.br ^[[DONE]]
  //       CHECK: ^[[DONE]]:
  //       CHECK: llvm.return %[[SUCCESS]] : i32
  //       CHECK: ^[[ERROR]]:
  //       CHECK: llvm.return %[[LAUNCH_ERROR]] : i32
  func.func @conditional(%count: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %active = arith.cmpi sgt, %count, %c0 : index
    scf.if %active {
      gpu.launch_func @kernels::@empty blocks in (%count, %c1, %c1)
          threads in (%c1, %c1, %c1) args(%count : index)
    }
    return
  }
  // CHECK-LABEL: gpu.module @kernels
  //       CHECK: func.func @empty(%[[COUNT:[^:]+]]: index) attributes {gpu.kernel, nv_tensor_ir.kernel_ordinal = 0 : i32}
  gpu.module @kernels {
    func.func @empty(%count: index) attributes {gpu.kernel} { return }
  }
  // CHECK-LABEL: llvm.func @tensor_ir_host_entry_conditional_v1(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[PACKED:.+]]: !llvm.ptr, %[[COUNT:.+]]: i32) -> i32
  //       CHECK: %[[BITS:.+]] = llvm.load %[[PACKED]] : !llvm.ptr -> i64
  //       CHECK: %[[STATUS:.+]] = llvm.call @conditional(%[[RUNTIME_PTR]], %[[BITS]]) : (!llvm.ptr, i64) -> i32
  //       CHECK: llvm.return %[[STATUS]] : i32
}

// -----

module attributes {gpu.container_module} {
  // Kernel parameters follow the launch operand order, not the host order.
  // CHECK-LABEL: llvm.func @reordered(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[A:.+]]: f32, %[[B:.+]]: f32) -> i32
  //   CHECK-DAG: %[[PARAMS:.+]] = llvm.alloca %{{.+}} x !llvm.struct<(array<2 x ptr>, array<2 x i64>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  //       CHECK: %[[FIRST_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<2 x ptr>, array<2 x i64>)>
  //       CHECK: llvm.store %[[B]], %[[FIRST_SLOT]] : f32, !llvm.ptr
  //       CHECK: %[[FIRST_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<2 x ptr>, array<2 x i64>)>
  //       CHECK: llvm.store %[[FIRST_SLOT]], %[[FIRST_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[SECOND_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<2 x ptr>, array<2 x i64>)>
  //       CHECK: llvm.store %[[A]], %[[SECOND_SLOT]] : f32, !llvm.ptr
  //       CHECK: %[[SECOND_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<2 x ptr>, array<2 x i64>)>
  //       CHECK: llvm.store %[[SECOND_SLOT]], %[[SECOND_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.call %{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}, %[[PARAMS]])
  func.func @reordered(%a: f32, %b: f32) {
    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@reordered blocks in (%one, %one, %one)
        threads in (%one, %one, %one) args(%b : f32, %a : f32)
    return
  }
  gpu.module @kernels {
    func.func @reordered(%a: f32, %b: f32) attributes {gpu.kernel} { return }
  }
  // Decode both host slots in host argument order.
  // CHECK-LABEL: llvm.func @tensor_ir_host_entry_reordered_v1(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[PACKED:.+]]: !llvm.ptr, %[[COUNT:.+]]: i32) -> i32
  //       CHECK: %[[A_BITS:.+]] = llvm.load %[[PACKED]] : !llvm.ptr -> i64
  //       CHECK: %[[A_I32:.+]] = llvm.trunc %[[A_BITS]] : i64 to i32
  //       CHECK: %[[A:.+]] = llvm.bitcast %[[A_I32]] : i32 to f32
  //       CHECK: %[[B_PTR:.+]] = llvm.getelementptr %[[PACKED]][1] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[B_BITS:.+]] = llvm.load %[[B_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[B_I32:.+]] = llvm.trunc %[[B_BITS]] : i64 to i32
  //       CHECK: %[[B:.+]] = llvm.bitcast %[[B_I32]] : i32 to f32
  //       CHECK: %[[STATUS:.+]] = llvm.call @reordered(%[[RUNTIME_PTR]], %[[A]], %[[B]]) : (!llvm.ptr, f32, f32) -> i32
  //       CHECK: llvm.return %[[STATUS]] : i32
}

// -----

// One entrypoint launches the same kernel with different operands, including
// a value computed on the host. Each launch rewrites the shared kernel
// parameter table from its own operands at the same slot addresses.
module @relaunch attributes {gpu.container_module} {
  // CHECK-LABEL: llvm.func @relaunch(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[A_ALLOCATED:.+]]: !llvm.ptr, %[[A:.+]]: !llvm.ptr, %[[A_OFFSET:.+]]: i64, %[[A_SIZE:.+]]: i64, %[[A_STRIDE:.+]]: i64, %[[B_ALLOCATED:.+]]: !llvm.ptr, %[[B:.+]]: !llvm.ptr, %[[B_OFFSET:.+]]: i64, %[[B_SIZE:.+]]: i64, %[[B_STRIDE:.+]]: i64, %[[X:.+]]: f32, %[[Y:.+]]: f32) -> i32
  //   CHECK-DAG: %[[KERNEL_ID:.+]] = llvm.mlir.constant(0 : i32) : i32
  //   CHECK-DAG: %[[CONFIG_PTR:.+]] = llvm.alloca %{{.+}} x !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //   CHECK-DAG: %[[PARAMS:.+]] = llvm.alloca %{{.+}} x !llvm.struct<(array<3 x ptr>, array<3 x i64>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  //       CHECK: %[[PRODUCT:.+]] = llvm.fmul %[[X]], %[[Y]] : f32
  // First launch: (%a, %x).
  //       CHECK: llvm.store %{{.+}}, %[[CONFIG_PTR]] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>, !llvm.ptr
  //       CHECK: %[[DATA_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x ptr>, array<3 x i64>)>
  //       CHECK: llvm.store %[[A]], %[[DATA_SLOT]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[FIRST_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x ptr>, array<3 x i64>)>
  //       CHECK: llvm.store %[[DATA_SLOT]], %[[FIRST_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[A_SIZE_I32:.+]] = llvm.trunc %[[A_SIZE]] : i64 to i32
  //       CHECK: %[[SIZE_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x ptr>, array<3 x i64>)>
  //       CHECK: llvm.store %[[A_SIZE_I32]], %[[SIZE_SLOT]] : i32, !llvm.ptr
  //       CHECK: %[[SIZE_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x ptr>, array<3 x i64>)>
  //       CHECK: llvm.store %[[SIZE_SLOT]], %[[SIZE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[SCALE_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x ptr>, array<3 x i64>)>
  //       CHECK: llvm.store %[[X]], %[[SCALE_SLOT]] : f32, !llvm.ptr
  //       CHECK: %[[SCALE_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x ptr>, array<3 x i64>)>
  //       CHECK: llvm.store %[[SCALE_SLOT]], %[[SCALE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.call %{{.+}}(%{{.+}}, %[[KERNEL_ID]], %[[CONFIG_PTR]], %[[PARAMS]])
  //       CHECK: llvm.cond_br %{{.+}}, ^[[SECOND:.+]], ^{{.+}}
  // Second launch: (%b, %y) overwrites every slot.
  //       CHECK: ^[[SECOND]]:
  //       CHECK: llvm.store %{{.+}}, %[[CONFIG_PTR]] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[B]], %[[DATA_SLOT]] : !llvm.ptr, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[DATA_SLOT]], %[[FIRST_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[B_SIZE_I32:.+]] = llvm.trunc %[[B_SIZE]] : i64 to i32
  //  CHECK-NEXT: llvm.store %[[B_SIZE_I32]], %[[SIZE_SLOT]] : i32, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[SIZE_SLOT]], %[[SIZE_PARAM]] : !llvm.ptr, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[Y]], %[[SCALE_SLOT]] : f32, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[SCALE_SLOT]], %[[SCALE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.call %{{.+}}(%{{.+}}, %[[KERNEL_ID]], %[[CONFIG_PTR]], %[[PARAMS]])
  //       CHECK: llvm.cond_br %{{.+}}, ^[[THIRD:.+]], ^{{.+}}
  // Third launch: (%a, %x * %y) reuses the first launch's size conversion.
  //       CHECK: ^[[THIRD]]:
  //       CHECK: llvm.store %{{.+}}, %[[CONFIG_PTR]] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[A]], %[[DATA_SLOT]] : !llvm.ptr, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[DATA_SLOT]], %[[FIRST_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.store %[[A_SIZE_I32]], %[[SIZE_SLOT]] : i32, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[SIZE_SLOT]], %[[SIZE_PARAM]] : !llvm.ptr, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[PRODUCT]], %[[SCALE_SLOT]] : f32, !llvm.ptr
  //  CHECK-NEXT: llvm.store %[[SCALE_SLOT]], %[[SCALE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.call %{{.+}}(%{{.+}}, %[[KERNEL_ID]], %[[CONFIG_PTR]], %[[PARAMS]])
  func.func @relaunch(%a: memref<?xf32>,
                      %b: memref<?xf32>,
                      %x: f32, %y: f32) {
    %product = arith.mulf %x, %y : f32
    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@scale blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
        args(%a : memref<?xf32>, %x : f32)
    gpu.launch_func @kernels::@scale blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
        args(%b : memref<?xf32>, %y : f32)
    gpu.launch_func @kernels::@scale blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
        args(%a : memref<?xf32>, %product : f32)
    return
  }
  gpu.module @kernels {
    func.func @scale(%data: memref<?xf32>, %factor: f32)
        attributes {gpu.kernel} { return }
  }
  // FINAL-LABEL: cuda_tile.module @cuda_tile_relaunch
  //       FINAL: entry @scale(%{{[^:]+}}: tile<ptr<f32>>, %{{[^:]+}}: tile<i32>, %{{[^:]+}}: tile<f32>)
}

// -----

// i32 launch sizes and an explicit dynamic shared memory size are stored into
// the launch config without any width conversion.
module attributes {gpu.container_module} {
  // CHECK-LABEL: llvm.func @i32_dims(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[GRID_SIZE:.+]]: i32, %[[BLOCK_SIZE:.+]]: i32, %[[SMEM_BYTES:.+]]: i32) -> i32
  //   CHECK-DAG: %[[SUCCESS:.+]] = llvm.mlir.constant(0 : i32) : i32
  //   CHECK-DAG: %[[DIM_ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
  //   CHECK-DAG: %[[CONFIG_UNDEF:.+]] = llvm.mlir.poison : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //   CHECK-DAG: %[[CONFIG_PTR:.+]] = llvm.alloca %{{.+}} x !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //   CHECK-DAG: %[[PARAMS:.+]] = llvm.alloca %{{.+}} x !llvm.struct<(array<1 x ptr>, array<1 x i64>)>
  //   CHECK-NOT: llvm.trunc
  //       CHECK: %[[GRID_X:.+]] = llvm.insertvalue %[[GRID_SIZE]], %[[CONFIG_UNDEF]][0, 0] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[GRID_Y:.+]] = llvm.insertvalue %[[DIM_ONE]], %[[GRID_X]][0, 1] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[GRID_Z:.+]] = llvm.insertvalue %[[DIM_ONE]], %[[GRID_Y]][0, 2] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[BLOCK_X:.+]] = llvm.insertvalue %[[BLOCK_SIZE]], %[[GRID_Z]][1, 0] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[BLOCK_Y:.+]] = llvm.insertvalue %[[DIM_ONE]], %[[BLOCK_X]][1, 1] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[BLOCK_Z:.+]] = llvm.insertvalue %[[DIM_ONE]], %[[BLOCK_Y]][1, 2] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: %[[CONFIG:.+]] = llvm.insertvalue %[[SMEM_BYTES]], %[[BLOCK_Z]][2] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>
  //       CHECK: llvm.store %[[CONFIG]], %[[CONFIG_PTR]] : !llvm.struct<(array<3 x i32>, array<3 x i32>, i32)>, !llvm.ptr
  //       CHECK: %[[GRID_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<1 x ptr>, array<1 x i64>)>
  //       CHECK: llvm.store %[[GRID_SIZE]], %[[GRID_SLOT]] : i32, !llvm.ptr
  //       CHECK: llvm.call %{{.+}}(%{{.+}}, %[[SUCCESS]], %[[CONFIG_PTR]], %[[PARAMS]])
  func.func @i32_dims(%grid: i32, %block: i32, %smem: i32) {
    %one = arith.constant 1 : i32
    gpu.launch_func @kernels::@i32_dims blocks in (%grid, %one, %one)
        threads in (%block, %one, %one) : i32
        dynamic_shared_memory_size %smem args(%grid : i32)
    return
  }
  gpu.module @kernels {
    func.func @i32_dims(%n: i32) attributes {gpu.kernel} { return }
  }
}

// -----

// Even a fully static tensor consumes all five host slots. The private device
// signature needs only the pointer: no sizes or strides are launch parameters.
module @uniform attributes {gpu.container_module} {
  // CHECK-LABEL: llvm.func @copy(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[ALLOCATED:.+]]: !llvm.ptr, %[[ALIGNED:.+]]: !llvm.ptr, %[[OFFSET:.+]]: i64, %[[SIZE0:.+]]: i64, %[[SIZE1:.+]]: i64, %[[STRIDE0:.+]]: i64, %[[STRIDE1:.+]]: i64) -> i32
  //   CHECK-DAG: %[[PARAMS:.+]] = llvm.alloca %{{.+}} x !llvm.struct<(array<1 x ptr>, array<1 x i64>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  //       CHECK: %[[DATA_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<1 x ptr>, array<1 x i64>)>
  //       CHECK: llvm.store %[[ALIGNED]], %[[DATA_SLOT]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[FIRST_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<1 x ptr>, array<1 x i64>)>
  //       CHECK: llvm.store %[[DATA_SLOT]], %[[FIRST_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.call %{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}, %[[PARAMS]])
  func.func @copy(
      %a: memref<32x16xf32, strided<[16, 1]>>) {
    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@copy blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
        args(%a : memref<32x16xf32, strided<[16, 1]>>)
    return
  }
  gpu.module @kernels {
    func.func @copy(%a: memref<32x16xf32, strided<[16, 1]>>)
        attributes {gpu.kernel} { return }
  }
  // CHECK-LABEL: llvm.func @tensor_ir_host_entry_copy_v1(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[PACKED:.+]]: !llvm.ptr, %[[COUNT:.+]]: i32) -> i32
  //   CHECK-DAG: %[[OFFSET:.+]] = llvm.mlir.constant(0 : i64) : i64
  //   CHECK-NOT: llvm.alloca
  //       CHECK: %[[PTR_BITS:.+]] = llvm.load %[[PACKED]] : !llvm.ptr -> i64
  //       CHECK: %[[DATA:.+]] = llvm.inttoptr %[[PTR_BITS]] : i64 to !llvm.ptr
  //       CHECK: %[[SIZE0_PTR:.+]] = llvm.getelementptr %[[PACKED]][1] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[SIZE0:.+]] = llvm.load %[[SIZE0_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[SIZE1_PTR:.+]] = llvm.getelementptr %[[PACKED]][2] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[SIZE1:.+]] = llvm.load %[[SIZE1_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[STRIDE0_PTR:.+]] = llvm.getelementptr %[[PACKED]][3] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[STRIDE0:.+]] = llvm.load %[[STRIDE0_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[STRIDE1_PTR:.+]] = llvm.getelementptr %[[PACKED]][4] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[STRIDE1:.+]] = llvm.load %[[STRIDE1_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[STATUS:.+]] = llvm.call @copy(%[[RUNTIME_PTR]], %[[DATA]], %[[DATA]],
  //  CHECK-SAME:    %[[OFFSET]], %[[SIZE0]], %[[SIZE1]], %[[STRIDE0]], %[[STRIDE1]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
  //       CHECK: llvm.return %[[STATUS]] : i32
  // FINAL-LABEL: cuda_tile.module
  //       FINAL: entry @copy(%{{[^:]+}}: tile<ptr<f32>>)
}

// -----

// Only dynamic metadata reaches the device. The host entry decodes the pointer
// and scalar following a partly-static tensor from HOST slot offsets, while the
// launch packs them at the smaller device parameter indices.
module @mixed attributes {gpu.container_module} {
  // CHECK-LABEL: llvm.func @mixed(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[A_ALLOCATED:.+]]: !llvm.ptr, %[[A:.+]]: !llvm.ptr, %[[A_OFFSET:.+]]: i64, %[[A_SIZE0:.+]]: i64, %[[A_SIZE1:.+]]: i64, %[[A_STRIDE0:.+]]: i64, %[[A_STRIDE1:.+]]: i64, %[[B_ALLOCATED:.+]]: !llvm.ptr, %[[B:.+]]: !llvm.ptr, %[[B_OFFSET:.+]]: i64, %[[B_SIZE:.+]]: i64, %[[B_STRIDE:.+]]: i64, %[[SCALE:.+]]: i32) -> i32
  //   CHECK-DAG: %[[PARAMS:.+]] = llvm.alloca %{{.+}} x !llvm.struct<(array<5 x ptr>, array<5 x i64>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  //       CHECK: %[[A_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 0]
  //       CHECK: llvm.store %[[A]], %[[A_SLOT]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[FIRST_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 0]
  //       CHECK: llvm.store %[[A_SLOT]], %[[FIRST_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[SIZE1_I32:.+]] = llvm.trunc %[[A_SIZE1]] : i64 to i32
  //       CHECK: %[[SIZE1_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 1]
  //       CHECK: llvm.store %[[SIZE1_I32]], %[[SIZE1_SLOT]] : i32, !llvm.ptr
  //       CHECK: %[[SIZE1_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 1]
  //       CHECK: llvm.store %[[SIZE1_SLOT]], %[[SIZE1_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[STRIDE0_I32:.+]] = llvm.trunc %[[A_STRIDE0]] : i64 to i32
  //       CHECK: %[[STRIDE0_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 2]
  //       CHECK: llvm.store %[[STRIDE0_I32]], %[[STRIDE0_SLOT]] : i32, !llvm.ptr
  //       CHECK: %[[STRIDE0_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 2]
  //       CHECK: llvm.store %[[STRIDE0_SLOT]], %[[STRIDE0_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[B_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 3]
  //       CHECK: llvm.store %[[B]], %[[B_SLOT]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[B_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 3]
  //       CHECK: llvm.store %[[B_SLOT]], %[[B_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: %[[SCALE_SLOT:.+]] = llvm.getelementptr %[[PARAMS]][0, 1, 4]
  //       CHECK: llvm.store %[[SCALE]], %[[SCALE_SLOT]] : i32, !llvm.ptr
  //       CHECK: %[[SCALE_PARAM:.+]] = llvm.getelementptr %[[PARAMS]][0, 0, 4]
  //       CHECK: llvm.store %[[SCALE_SLOT]], %[[SCALE_PARAM]] : !llvm.ptr, !llvm.ptr
  //       CHECK: llvm.call %{{.+}}(%{{.+}}, %{{.+}}, %{{.+}}, %[[PARAMS]])
  func.func @mixed(
      %a: memref<4x?xf32, strided<[?, 1]>>,
      %b: memref<8xf32>, %scale: i32) {
    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@mixed blocks in (%one, %one, %one)
        threads in (%one, %one, %one)
        args(%a : memref<4x?xf32, strided<[?, 1]>>,
             %b : memref<8xf32>, %scale : i32)
    return
  }
  gpu.module @kernels {
    func.func @mixed(
        %a: memref<4x?xf32, strided<[?, 1]>>,
        %b: memref<8xf32>, %scale: i32)
        attributes {gpu.kernel} { return }
  }
  // CHECK-LABEL: llvm.func @tensor_ir_host_entry_mixed_v1(
  //  CHECK-SAME: %[[RUNTIME_PTR:.+]]: !llvm.ptr, %[[PACKED:.+]]: !llvm.ptr, %[[COUNT:.+]]: i32) -> i32
  //   CHECK-DAG: %[[OFFSET:.+]] = llvm.mlir.constant(0 : i64) : i64
  //   CHECK-NOT: llvm.alloca
  //       CHECK: %[[A_BITS:.+]] = llvm.load %[[PACKED]] : !llvm.ptr -> i64
  //       CHECK: %[[A:.+]] = llvm.inttoptr %[[A_BITS]] : i64 to !llvm.ptr
  //       CHECK: %[[SIZE0_PTR:.+]] = llvm.getelementptr %[[PACKED]][1] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[SIZE0:.+]] = llvm.load %[[SIZE0_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[SIZE1_PTR:.+]] = llvm.getelementptr %[[PACKED]][2] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[SIZE1:.+]] = llvm.load %[[SIZE1_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[STRIDE0_PTR:.+]] = llvm.getelementptr %[[PACKED]][3] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[STRIDE0:.+]] = llvm.load %[[STRIDE0_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[STRIDE1_PTR:.+]] = llvm.getelementptr %[[PACKED]][4] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[STRIDE1:.+]] = llvm.load %[[STRIDE1_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[B_PTR:.+]] = llvm.getelementptr %[[PACKED]][5] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[B_BITS:.+]] = llvm.load %[[B_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[B:.+]] = llvm.inttoptr %[[B_BITS]] : i64 to !llvm.ptr
  //       CHECK: %[[B_SIZE_PTR:.+]] = llvm.getelementptr %[[PACKED]][6] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[B_SIZE:.+]] = llvm.load %[[B_SIZE_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[B_STRIDE_PTR:.+]] = llvm.getelementptr %[[PACKED]][7] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[B_STRIDE:.+]] = llvm.load %[[B_STRIDE_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[SCALE_PTR:.+]] = llvm.getelementptr %[[PACKED]][8] : (!llvm.ptr) -> !llvm.ptr, i64
  //       CHECK: %[[SCALE_BITS:.+]] = llvm.load %[[SCALE_PTR]] : !llvm.ptr -> i64
  //       CHECK: %[[SCALE:.+]] = llvm.trunc %[[SCALE_BITS]] : i64 to i32
  //       CHECK: %[[STATUS:.+]] = llvm.call @mixed(%[[RUNTIME_PTR]], %[[A]], %[[A]], %[[OFFSET]], %[[SIZE0]],
  //  CHECK-SAME:    %[[SIZE1]], %[[STRIDE0]], %[[STRIDE1]], %[[B]], %[[B]], %[[OFFSET]], %[[B_SIZE]], %[[B_STRIDE]], %[[SCALE]])
  //       CHECK: llvm.return %[[STATUS]] : i32
}

// -----

// A private helper may launch and return a value. Launch failures return a
// status with poison for that value; its caller propagates the status.
module @helper_launch attributes {gpu.container_module} {
  // CHECK-LABEL: llvm.func @dispatch(
  //       CHECK: %[[RESULT:.*]] = llvm.call @helper(%{{.*}}) : (!llvm.ptr, i64) -> !llvm.struct<(i32, i32)>
  //       CHECK: %[[STATUS:.*]] = llvm.extractvalue %[[RESULT]][0]
  //       CHECK: llvm.icmp "ne" %[[STATUS]], %{{.*}} : i32
  //       CHECK: llvm.cond_br %{{.*}}, ^[[FAIL:bb[0-9]+]](%[[STATUS]] : i32),
  //       CHECK: ^[[FAIL]](%[[ERROR_STATUS:.*]]: i32):
  //       CHECK: llvm.return %[[ERROR_STATUS]] : i32
  func.func @dispatch(%grid: index) {
    %value = func.call @helper(%grid) : (index) -> i32
    return
  }
  // CHECK-LABEL: llvm.func @helper(
  //  CHECK-SAME: -> !llvm.struct<(i32, i32)>
  //   CHECK-DAG: %[[POISON:.*]] = llvm.mlir.poison : i32
  //   CHECK-DAG: %[[RANGE_ERROR:.*]] = llvm.mlir.constant(-5 : i32)
  //   CHECK-DAG: %[[CALLBACK_ERROR:.*]] = llvm.mlir.constant(-3 : i32)
  //       CHECK: llvm.call %{{.*}}(%{{.*}})
  //       CHECK: llvm.insertvalue %[[RANGE_ERROR]], %{{.*}}[0]
  //       CHECK: llvm.insertvalue %[[POISON]], %{{.*}}[1]
  //       CHECK: llvm.return %{{.*}} : !llvm.struct<(i32, i32)>
  //       CHECK: llvm.insertvalue %[[CALLBACK_ERROR]], %{{.*}}[0]
  //       CHECK: llvm.insertvalue %[[POISON]], %{{.*}}[1]
  //       CHECK: llvm.return %{{.*}} : !llvm.struct<(i32, i32)>
  func.func private @helper(%grid: index) -> i32 {
    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@empty blocks in (%grid, %one, %one)
        threads in (%one, %one, %one)
    %value = arith.constant 5 : i32
    return %value : i32
  }
  // CHECK-LABEL: gpu.module @kernels
  //       CHECK: func.func @empty() attributes {gpu.kernel, nv_tensor_ir.kernel_ordinal = 0 : i32}
  gpu.module @kernels {
    func.func @empty() attributes {gpu.kernel} { return }
  }
  // CHECK-LABEL: llvm.func @tensor_ir_host_entry_dispatch_v1(
  //       CHECK: llvm.call @dispatch
}
