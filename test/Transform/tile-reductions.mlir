// RUN: tensor_ir-opt %s --pass-pipeline='builtin.module(nv_tensor_ir.graph(materialize-default-strides,layout-propagation-annotation,layout-propagation-normalization,tile-selection,graph-splitting),tir-bufferize,func.func(tir-form-grid,tir-tile-reductions{reduction-tile-size=128}))' --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @large_add_dispatch
// CHECK: %[[ZERO:.*]] = nv_tensor_ir.constant dense<0.000000e+00> : tensor<32x128xf32>
// CHECK: %[[LOOP:.*]] = scf.for %[[K:[^ ]+]] = {{.*}} to {{.*}} step {{.*}} iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (tensor<32x128xf32>) {
// CHECK:   %[[TILE:.*]] = nv_tensor_ir.load %{{.*}}[{{.*}}, %[[K]]]
// CHECK:   %[[SUM:.*]] = nv_tensor_ir.add %[[ACC]], %[[TILE]]
// CHECK:   scf.yield %[[SUM]]
// CHECK: }
// CHECK: nv_tensor_ir.reduce(%[[LOOP]]) <dimensions = [1], reduction_mode = <add>>
// CHECK-NOT: nv_tensor_ir.layout
nv_tensor_ir.graph @large_add(%input: tensor<64x1024xf32>)
    -> tensor<64x1xf32> attributes {tile_size = array<i32: 32>} {
  %result = reduce(%input) <dimensions = [1], reduction_mode = <add>>
      : tensor<64x1024xf32> -> tensor<64x1xf32>
  results %result : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: func.func @pointwise_prologue_dispatch
// CHECK: scf.for %[[K:[^ ]+]] =
// CHECK:   %[[TILE:.*]] = nv_tensor_ir.load {{.*}}[{{.*}}, %[[K]]]
// CHECK:   %[[SQUARE:.*]] = nv_tensor_ir.mul %[[TILE]], %[[TILE]]
// CHECK:   %[[SUM:.*]] = nv_tensor_ir.add {{.*}}, %[[SQUARE]]
// CHECK:   scf.yield %[[SUM]]
// CHECK: nv_tensor_ir.reduce
nv_tensor_ir.graph @pointwise_prologue(%input: tensor<64x1024xf32>)
    -> tensor<64x1xf32> attributes {tile_size = array<i32: 32>} {
  %square = mul %input, %input : tensor<64x1024xf32>
  %result = reduce(%square) <dimensions = [1], reduction_mode = <add>>
      : tensor<64x1024xf32> -> tensor<64x1xf32>
  results %result : tensor<64x1xf32>
}

// -----

// The selected contraction tile covers both dimensions, so no serial loop is
// needed and the coarse load/reduce pair remains directly inspectable.

// CHECK-LABEL: func.func @small_dispatch
// CHECK: scf.forall
// CHECK-NOT: scf.for %
// CHECK: nv_tensor_ir.load {{.*}} to tensor<32x8x8xf32>
// CHECK: nv_tensor_ir.reduce
nv_tensor_ir.graph @small(%input: tensor<8x64x8xf32>)
    -> tensor<1x64x1xf32> attributes {tile_size = array<i32: 32>} {
  %result = reduce(%input) <dimensions = [0, 2], reduction_mode = <add>>
      : tensor<8x64x8xf32> -> tensor<1x64x1xf32>
  results %result : tensor<1x64x1xf32>
}

// -----

// CHECK-LABEL: func.func @two_dimensions_dispatch
// CHECK: %[[ZERO:.*]] = nv_tensor_ir.constant dense<0.000000e+00> : tensor<32x8x16xf32>
// CHECK: scf.for %[[K:[^ ]+]] = {{.*}} to {{.*}} step {{.*}} iter_args(%[[ACC:.*]] = %[[ZERO]])
// CHECK:   nv_tensor_ir.load {{.*}}[{{.*}}, {{.*}}, %[[K]]]
// CHECK:   %[[SUM:.*]] = nv_tensor_ir.add %[[ACC]], {{.*}}
// CHECK:   scf.yield %[[SUM]]
// CHECK: nv_tensor_ir.reduce{{.*}}dimensions = [1, 2]
nv_tensor_ir.graph @two_dimensions(%input: tensor<8x64x8192xf32>)
    -> tensor<1x64x1xf32> attributes {tile_size = array<i32: 32>} {
  %result = reduce(%input) <dimensions = [0, 2], reduction_mode = <add>>
      : tensor<8x64x8192xf32> -> tensor<1x64x1xf32>
  results %result : tensor<1x64x1xf32>
}

// -----

// The nonlinear prologue is applied to each loaded chunk. The remaining
// reduction retains norm2 so conversion applies the square-root epilogue.

// CHECK-LABEL: func.func @large_norm2_dispatch
// CHECK: scf.for
// CHECK:   %[[TILE:.*]] = nv_tensor_ir.load
// CHECK:   %[[SQUARE:.*]] = nv_tensor_ir.mul %[[TILE]], %[[TILE]]
// CHECK:   nv_tensor_ir.add {{.*}}, %[[SQUARE]]
// CHECK: nv_tensor_ir.reduce{{.*}}reduction_mode = <norm2>{{.*}}nv_tensor_ir.reduction_prologue_applied
nv_tensor_ir.graph @large_norm2(%input: tensor<64x1024xf32>)
    -> tensor<64x1xf32> attributes {tile_size = array<i32: 32>} {
  %result = reduce(%input) <dimensions = [1], reduction_mode = <norm2>>
      : tensor<64x1024xf32> -> tensor<64x1xf32>
  results %result : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: func.func @user_defined_dispatch
// CHECK: %[[ZERO:.*]] = nv_tensor_ir.constant dense<0.000000e+00> : tensor<32x128xf32>
// CHECK: scf.for %{{.*}} iter_args(%[[ACC:.*]] = %[[ZERO]])
// CHECK:   %[[TILE:.*]] = nv_tensor_ir.load
// CHECK:   %[[SUM:.*]] = arith.addf %[[ACC]], %[[TILE]] {{.*}} : tensor<32x128xf32>
// CHECK:   scf.yield %[[SUM]]
// CHECK: nv_tensor_ir.reduce_ud
// CHECK:   arith.addf
// CHECK:   nv_tensor_ir.yield
nv_tensor_ir.graph @user_defined(%input: tensor<64x1024xf32>)
    -> tensor<64x1xf32> attributes {tile_size = array<i32: 32>} {
  %result = reduce_ud(%input) <dimensions = [1], identity = [0.0 : f32]>
      (%accumulator: f32, %value: f32) {
    %sum = arith.addf %accumulator, %value : f32
    nv_tensor_ir.yield %sum : f32
  } : tensor<64x1024xf32> -> tensor<64x1xf32>
  results %result : tensor<64x1xf32>
}

// -----

// Strip mining computes the sum across all chunks and divides once by the
// complete reduction extent.
// CHECK-LABEL: func.func @large_avg_dispatch
// CHECK: scf.for
// CHECK: nv_tensor_ir.reduce{{.*}}reduction_mode = <add>
// CHECK: %[[DIVISOR:.*]] = nv_tensor_ir.constant dense<1.024000e+03>
// CHECK: nv_tensor_ir.div {{.*}}, %[[DIVISOR]]
nv_tensor_ir.graph @large_avg(%input: tensor<64x1024xf32>)
    -> tensor<64x1xf32> attributes {tile_size = array<i32: 32>} {
  %result = reduce(%input) <dimensions = [1], reduction_mode = <avg>>
      : tensor<64x1024xf32> -> tensor<64x1xf32>
  results %result : tensor<64x1xf32>
}

// -----

// A partial final chunk still uses the full logical extent as the divisor.
// CHECK-LABEL: func.func @tail_avg_dispatch
// CHECK: scf.for
// CHECK: nv_tensor_ir.reduce{{.*}}reduction_mode = <add>
// CHECK: %[[DIVISOR:.*]] = nv_tensor_ir.constant dense<5.000000e+02>
// CHECK: nv_tensor_ir.div {{.*}}, %[[DIVISOR]]
nv_tensor_ir.graph @tail_avg(%input: tensor<64x500xf32>)
    -> tensor<64x1xf32> attributes {tile_size = array<i32: 32>} {
  %result = reduce(%input) <dimensions = [1], reduction_mode = <avg>>
      : tensor<64x500xf32> -> tensor<64x1xf32>
  results %result : tensor<64x1xf32>
}
