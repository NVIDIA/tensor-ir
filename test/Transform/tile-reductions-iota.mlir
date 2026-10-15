// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(\
// RUN:   materialize-default-strides, \
// RUN:   tile-selection), \
// RUN:   tir-bufferize, \
// RUN:   func.func(tir-form-grid,tir-tile-reductions{reduction-tile-size=64}))' \
// RUN:   --split-input-file | FileCheck %s

// The column index must advance with the contraction loop, just like the load.
// This is a small counterpart of Mamba's chunked causal-mask producer.
// CHECK-LABEL: func.func @contracting_iota_dispatch
// CHECK: %[[LOOP:.*]] = scf.for %[[K:[^ ]+]] = {{.*}} iter_args(%[[ACC:[^ ]+]] = {{.*}})
// CHECK: %[[LOAD:.*]] = nv_tensor_ir.load {{.*}}[0, %[[K]]]
// CHECK: %[[LOCAL:.*]] = nv_tensor_ir.iota dimension = 1 : tensor<4x64xsi32>
// CHECK: %[[EXTENT:.*]] = arith.constant 64 : index
// CHECK: %[[OFFSET:.*]] = arith.muli %[[K]], %[[EXTENT]] : index
// CHECK: %[[CAST:.*]] = arith.index_cast %[[OFFSET]] : index to i32
// CHECK: %[[SIGNED:.*]] = builtin.unrealized_conversion_cast %[[CAST]] : i32 to si32
// CHECK: %[[OFFSET_TILE:.*]] = nv_tensor_ir.splat %[[SIGNED]] : tensor<4x64xsi32>
// CHECK: %[[SHIFTED:.*]] = nv_tensor_ir.add %[[LOCAL]], %[[OFFSET_TILE]]
// CHECK: %[[GLOBAL:.*]] = nv_tensor_ir.add %[[SHIFTED]], {{.*}}
// CHECK: %[[FLOAT:.*]] = nv_tensor_ir.convert %[[GLOBAL]]
// CHECK: %[[PRODUCT:.*]] = nv_tensor_ir.mul %[[LOAD]], %[[FLOAT]]
// CHECK: %[[NEXT:.*]] = nv_tensor_ir.add %[[ACC]], %[[PRODUCT]]
// CHECK: scf.yield %[[NEXT]]
// CHECK: nv_tensor_ir.reduce(%[[LOOP]])
nv_tensor_ir.graph @contracting_iota(%input: tensor<4x256xf32>)
    -> tensor<4x1xf32> attributes {tile_size = array<i32: 4>} {
  %index = iota dimension = 1 : tensor<4x256xsi32>
  %float_index = convert %index : tensor<4x256xsi32> -> tensor<4x256xf32>
  %weighted = mul %input, %float_index : tensor<4x256xf32>
  %sum = reduce(%weighted) <dimensions = [1], reduction_mode = <add>>
      : tensor<4x256xf32> -> tensor<4x1xf32>
  results %sum : tensor<4x1xf32>
}

// -----

// The MMA operand cloner must advance K indices as well as load addresses.
// CHECK-LABEL: func.func @matmul_iota_dispatch
// CHECK: %[[LOOP:.*]] = scf.for %[[K:[^ ]+]] = {{.*}} iter_args(%[[ACC:[^ ]+]] = {{.*}})
// CHECK: %[[LHS:.*]] = nv_tensor_ir.load {{.*}}[0, %[[K]]]
// CHECK: %[[LOCAL:.*]] = nv_tensor_ir.iota dimension = 1 : tensor<8x64xui32>
// CHECK: %[[EXTENT:.*]] = arith.constant 64 : index
// CHECK: %[[OFFSET:.*]] = arith.muli %[[K]], %[[EXTENT]] : index
// CHECK: %[[CAST:.*]] = arith.index_cast %[[OFFSET]] : index to i32
// CHECK: %[[UNSIGNED:.*]] = builtin.unrealized_conversion_cast %[[CAST]] : i32 to ui32
// CHECK: %[[OFFSET_TILE:.*]] = nv_tensor_ir.splat %[[UNSIGNED]] : tensor<8x64xui32>
// CHECK: %[[SHIFTED:.*]] = nv_tensor_ir.add %[[LOCAL]], %[[OFFSET_TILE]]
// CHECK: %[[GLOBAL:.*]] = nv_tensor_ir.add %[[SHIFTED]], {{.*}}
// CHECK: %[[FACTOR:.*]] = nv_tensor_ir.convert %[[GLOBAL]]
// CHECK: %[[WEIGHTED:.*]] = nv_tensor_ir.mul %[[LHS]], %[[FACTOR]]
// CHECK: %[[RHS:.*]] = nv_tensor_ir.load {{.*}}[%[[K]], 0]
// CHECK: %[[PARTIAL:.*]] = nv_tensor_ir.matmul(%[[WEIGHTED]], %[[RHS]]) accum(%[[ACC]]
// CHECK: scf.yield %[[PARTIAL]]
// CHECK: nv_tensor_ir.store %[[LOOP]]
nv_tensor_ir.graph @matmul_iota(%lhs: tensor<8x256xbf16>, %rhs: tensor<256x8xbf16>)
    -> tensor<8x8xf32> attributes {tile_size = array<i32: 8, 8>} {
  %index = iota dimension = 1 : tensor<8x256xui32>
  %factor = convert %index : tensor<8x256xui32> -> tensor<8x256xbf16>
  %weighted = mul %lhs, %factor : tensor<8x256xbf16>
  %result = matmul(%weighted, %rhs)
      : (tensor<8x256xbf16>, tensor<256x8xbf16>) -> tensor<8x8xf32>
  results %result : tensor<8x8xf32>
}

// -----

// An iota along a parallel dimension must not acquire the K-loop offset.
// CHECK-LABEL: func.func @parallel_iota_dispatch
// CHECK: %[[LOOP:.*]] = scf.for %[[K:[^ ]+]] = {{.*}} iter_args(%[[ACC:[^ ]+]] = {{.*}})
// CHECK: %[[LOAD:.*]] = nv_tensor_ir.load {{.*}}[%{{.*}}, %[[K]]]
// CHECK: %[[LOCAL:.*]] = nv_tensor_ir.iota dimension = 0 : tensor<4x64xsi32>
// CHECK-NOT: arith.muli %[[K]]
// CHECK: %[[ORIGIN:.*]] = nv_tensor_ir.splat %{{.*}} : tensor<4x64xsi32>
// CHECK-NEXT: %[[GLOBAL:.*]] = nv_tensor_ir.add %[[LOCAL]], %[[ORIGIN]]
// CHECK: %[[FACTOR:.*]] = nv_tensor_ir.convert %[[GLOBAL]]
// CHECK: %[[PRODUCT:.*]] = nv_tensor_ir.mul %[[LOAD]], %[[FACTOR]]
// CHECK: %[[NEXT:.*]] = nv_tensor_ir.add %[[ACC]], %[[PRODUCT]]
// CHECK: scf.yield %[[NEXT]]
// CHECK: nv_tensor_ir.reduce(%[[LOOP]])
nv_tensor_ir.graph @parallel_iota(%input: tensor<8x256xf32>)
    -> tensor<8x1xf32> attributes {tile_size = array<i32: 4>} {
  %index = iota dimension = 0 : tensor<8x256xsi32>
  %factor = convert %index : tensor<8x256xsi32> -> tensor<8x256xf32>
  %weighted = mul %input, %factor : tensor<8x256xf32>
  %sum = reduce(%weighted) <dimensions = [1], reduction_mode = <add>>
      : tensor<8x256xf32> -> tensor<8x1xf32>
  results %sum : tensor<8x1xf32>
}
