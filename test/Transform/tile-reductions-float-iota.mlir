// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,tile-selection),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{reduction-tile-size=128}))' --split-input-file | FileCheck %s
// RUN: tensor_ir-opt %s --layout-propagation-pipeline --split-input-file | FileCheck %s --check-prefix=LOWER

// Floating-point iotas lower to signless casts and arith.uitofp during
// grid formation. Their producer chain must be strip-mined with the mask.
// CHECK-LABEL: func.func @causal_mask_dispatch
// CHECK: %[[LOOP:.*]] = scf.for %[[K:[^ ]+]] = {{.*}} iter_args(%[[ACC:[^ ]+]] = {{.*}})
// CHECK: %[[ROWS:.*]] = arith.uitofp {{.*}} : tensor<32x128xi32> to tensor<32x128xf32>
// CHECK: %[[LOCAL:.*]] = nv_tensor_ir.iota dimension = 1 : tensor<32x128xui32>
// CHECK: %[[EXTENT:.*]] = arith.constant 128 : index
// CHECK: %[[OFFSET:.*]] = arith.muli %[[K]], %[[EXTENT]] : index
// CHECK: %[[INDEX:.*]] = arith.index_cast %[[OFFSET]] : index to i32
// CHECK: %[[UNSIGNED:.*]] = builtin.unrealized_conversion_cast %[[INDEX]] : i32 to ui32
// CHECK: %[[OFFSET_SPLAT:.*]] = nv_tensor_ir.splat %[[UNSIGNED]] : tensor<32x128xui32>
// CHECK: %[[SHIFTED:.*]] = nv_tensor_ir.add %[[LOCAL]], %[[OFFSET_SPLAT]] : tensor<32x128xui32>
// CHECK: %[[GLOBAL:.*]] = nv_tensor_ir.add %[[SHIFTED]], {{.*}} : tensor<32x128xui32>
// CHECK: %[[SIGNLESS:.*]] = builtin.unrealized_conversion_cast %[[GLOBAL]] : tensor<32x128xui32> to tensor<32x128xi32>
// CHECK: %[[COLS:.*]] = arith.uitofp %[[SIGNLESS]] : tensor<32x128xi32> to tensor<32x128xf32>
// CHECK: %[[MASK:.*]] = nv_tensor_ir.cmp %[[ROWS]] oge %[[COLS]] : tensor<32x128xf32>
// CHECK: %[[LHS:.*]] = nv_tensor_ir.load {{.*}} tile_indices [%{{.*}}, %[[K]]]
// CHECK: %[[ZERO:.*]] = nv_tensor_ir.constant dense<0.000000e+00> : tensor<32x128xf32>
// CHECK: %[[MASKED:.*]] = nv_tensor_ir.binary_select %[[MASK]], %[[LHS]], %[[ZERO]] : tensor<32x128xf32>
// CHECK: %[[RHS:.*]] = nv_tensor_ir.load {{.*}} tile_indices [%[[K]], %{{.*}}]
// CHECK: %[[PARTIAL:.*]] = nv_tensor_ir.matmul(%[[MASKED]], %[[RHS]]) accum(%[[ACC]]
// CHECK: scf.yield %[[PARTIAL]] : tensor<32x16xf32>
// CHECK: nv_tensor_ir.store %[[LOOP]],
// LOWER-LABEL: entry @causal_mask
// LOWER: mmaf
nv_tensor_ir.graph @causal_mask(%a: tensor<512x512xf32>, %b: tensor<512x48xf32>)
    -> tensor<512x48xf32> attributes {tile_size = array<i32: 32, 16>} {
  %rows = iota dimension = 0 : tensor<512x512xf32>
  %cols = iota dimension = 1 : tensor<512x512xf32>
  %mask = cmp %rows oge %cols : tensor<512x512xf32>
  %zero = constant dense<0.0> : tensor<512x512xf32>
  %causal = binary_select %mask, %a, %zero : tensor<512x512xf32>
  %c = matmul(%causal, %b)
      : (tensor<512x512xf32>, tensor<512x48xf32>) -> tensor<512x48xf32>
  results %c : tensor<512x48xf32>
}
