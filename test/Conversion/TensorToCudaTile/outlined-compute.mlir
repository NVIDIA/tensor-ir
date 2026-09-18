// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation" -split-input-file | FileCheck %s

module @pointwise_module {
  gpu.module @kernels {
  func.func @pointwise() attributes {gpu.kernel} {
    %lhs = nv_tensor_ir.constant dense<2.0> : tensor<2x4xf32>
    %rhs = nv_tensor_ir.constant dense<4.0> : tensor<2x4xf32>
    %sum = nv_tensor_ir.add %lhs, %rhs : tensor<2x4xf32>
    %exp = nv_tensor_ir.exp %sum : tensor<2x4xf32>
    %root = nv_tensor_ir.sqrt %exp : tensor<2x4xf32>
    %predicate = nv_tensor_ir.cmp %root olt %rhs : tensor<2x4xf32>
    %selected = nv_tensor_ir.binary_select %predicate, %root, %rhs : tensor<2x4xf32>
    %converted = nv_tensor_ir.convert %selected : tensor<2x4xf32> -> tensor<2x4xsi32>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_pointwise_module {
// CHECK: entry @pointwise()
// CHECK: %[[LHS:.*]] = constant <f32: 2.000000e+00> : tile<2x4xf32>
// CHECK: %[[RHS:.*]] = constant <f32: 4.000000e+00> : tile<2x4xf32>
// CHECK: %[[SUM:.*]] = addf %[[LHS]], %[[RHS]]
// CHECK: %[[EXP:.*]] = exp %[[SUM]]
// CHECK: %[[ROOT:.*]] = sqrt %[[EXP]]
// CHECK: %[[PREDICATE:.*]] = cmpf less_than ordered %[[ROOT]], %[[RHS]]
// CHECK: %[[SELECTED:.*]] = select %[[PREDICATE]], %[[ROOT]], %[[RHS]]
// CHECK: ftoi %[[SELECTED]] signed : tile<2x4xf32> -> tile<2x4xi32>
// CHECK: return

// -----

module @splat_module {
  gpu.module @kernels {
  func.func @constant_splat() attributes {gpu.kernel} {
    %value = nv_tensor_ir.constant 2.0 : f32
    %splat = nv_tensor_ir.splat %value : tensor<2x4xf32>
    return
  }
  func.func @runtime_splat(%value: f32) attributes {gpu.kernel} {
    %splat = nv_tensor_ir.splat %value : tensor<2x4xf32>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_splat_module {
// CHECK-LABEL: entry @constant_splat()
// CHECK: %[[SPLAT:.*]] = constant <f32: 2.000000e+00> : tile<2x4xf32>
// CHECK-NOT: reshape
// CHECK-NOT: broadcast
// CHECK: return
// CHECK-LABEL: entry @runtime_splat(
// CHECK-SAME: %[[VALUE:.*]]: tile<f32>)
// CHECK: %[[RESHAPED:.*]] = reshape %[[VALUE]] : tile<f32> -> tile<1x1xf32>
// CHECK: broadcast %[[RESHAPED]] : tile<1x1xf32> -> tile<2x4xf32>
// CHECK: return

// -----

module @shape_module {
  gpu.module @kernels {
  func.func @shape_ops() attributes {gpu.kernel} {
    %input = nv_tensor_ir.constant dense<1.0> : tensor<1x4xf32>
    %broadcast = nv_tensor_ir.broadcast %input : tensor<1x4xf32> -> tensor<2x4xf32>
    %reshape = nv_tensor_ir.reshape %broadcast : tensor<2x4xf32> -> tensor<4x2xf32>
    %transpose = nv_tensor_ir.transpose %reshape permutation = [1, 0]
        : tensor<4x2xf32> -> tensor<2x4xf32>
    %slice = nv_tensor_ir.slice %transpose starts = [0, 1] limits = [2, 3] strides = [1, 1]
        : tensor<2x4xf32> -> tensor<2x2xf32>
    %concat = nv_tensor_ir.concatenate %slice, %slice dimension = 0
        : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
    %iota = nv_tensor_ir.iota dimension = 0 : tensor<4x2xui32>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_shape_module {
// CHECK: entry @shape_ops()
// CHECK: %[[INPUT:.*]] = constant <f32: 1.000000e+00> : tile<1x4xf32>
// CHECK: %[[BROADCAST:.*]] = broadcast %[[INPUT]] : tile<1x4xf32> -> tile<2x4xf32>
// CHECK: %[[RESHAPE:.*]] = reshape %[[BROADCAST]] : tile<2x4xf32> -> tile<4x2xf32>
// CHECK: %[[TRANSPOSE:.*]] = permute %[[RESHAPE]] [1, 0]
// CHECK: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK: %[[ONE:.*]] = constant <i32: 1> : tile<i32>
// CHECK: %[[SLICE:.*]] = extract %[[TRANSPOSE]][%[[ZERO]], %[[ONE]]] : tile<2x4xf32> -> tile<2x2xf32>
// CHECK: cat %[[SLICE]], %[[SLICE]] dim = 0 : tile<2x2xf32>, tile<2x2xf32> -> tile<4x2xf32>
// CHECK: %[[IOTA:.*]] = iota : tile<4xi32>
// CHECK: reshape %[[IOTA]] : tile<4xi32> -> tile<4x1xi32>
// CHECK: broadcast {{.*}} : tile<4x1xi32> -> tile<4x2xi32>

// -----

module @reduction_module {
  gpu.module @kernels {
  func.func @reduction() attributes {gpu.kernel} {
    %input = nv_tensor_ir.constant dense<1.0> : tensor<4x8xf32>
    %result = nv_tensor_ir.reduce(%input)
        <dimensions = [1], reduction_mode = <add>>
        {nv_tensor_ir.reduction_extent = 8 : i64}
        : tensor<4x8xf32> -> tensor<4x1xf32>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_reduction_module {
// CHECK: entry @reduction()
// CHECK: %[[INPUT:.*]] = constant <f32: 1.000000e+00> : tile<4x8xf32>
// CHECK: %[[REDUCED:.*]] = reduce %[[INPUT]] dim=1 identities=[0.000000e+00 : f32]
// CHECK: addf
// CHECK: yield
// CHECK: reshape %[[REDUCED]] : tile<4xf32> -> tile<4x1xf32>

// -----

// Tile-shaped matmul lowering consumes the explicit operand and accumulator
// tiles directly; it does not require a layout source attribute.
module @matmul_module {
  gpu.module @kernels {
  func.func @matmul() attributes {gpu.kernel} {
    %lhs = nv_tensor_ir.constant dense<1.0> : tensor<16x32xf16>
    %rhs = nv_tensor_ir.constant dense<2.0> : tensor<32x8xf16>
    %product = nv_tensor_ir.matmul(%lhs, %rhs)
        : (tensor<16x32xf16>, tensor<32x8xf16>) -> tensor<16x8xf32>
    %accumulator = nv_tensor_ir.constant dense<3.0> : tensor<16x8xf32>
    %accumulated = nv_tensor_ir.matmul(%lhs, %rhs)
        accum(%accumulator : tensor<16x8xf32>)
        : (tensor<16x32xf16>, tensor<32x8xf16>) -> tensor<16x8xf32>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_matmul_module {
// CHECK: entry @matmul()
// CHECK: %[[LHS:.*]] = constant <f16: 1.000000e+00> : tile<16x32xf16>
// CHECK: %[[RHS:.*]] = constant <f16: 2.000000e+00> : tile<32x8xf16>
// CHECK: %[[ZERO:.*]] = constant <f32: 0.000000e+00> : tile<16x8xf32>
// CHECK: mmaf %[[LHS]], %[[RHS]], %[[ZERO]]
// CHECK: %[[ACC:.*]] = constant <f32: 3.000000e+00> : tile<16x8xf32>
// CHECK: mmaf %[[LHS]], %[[RHS]], %[[ACC]]

// -----

module @integer_matmul_module {
  gpu.module @kernels {
  func.func @integer_matmul() attributes {gpu.kernel} {
    %lhs = nv_tensor_ir.constant dense<1> : tensor<16x32xsi8>
    %rhs = nv_tensor_ir.constant dense<2> : tensor<32x8xsi8>
    %product = nv_tensor_ir.matmul(%lhs, %rhs)
        : (tensor<16x32xsi8>, tensor<32x8xsi8>) -> tensor<16x8xsi32>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_integer_matmul_module {
// CHECK: entry @integer_matmul()
// CHECK: %[[ILHS:.*]] = constant <i8: 1> : tile<16x32xi8>
// CHECK: %[[IRHS:.*]] = constant <i8: 2> : tile<32x8xi8>
// CHECK: %[[IZERO:.*]] = constant <i32: 0> : tile<16x8xi32>
// CHECK: mmai %[[ILHS]], %[[IRHS]], %[[IZERO]] signed signed

// -----

module @user_defined_reduction_module {
  gpu.module @kernels {
  func.func @user_defined_reduction() attributes {gpu.kernel} {
    %input = nv_tensor_ir.constant dense<1.0> : tensor<4x8xf32>
    %result = nv_tensor_ir.reduce_ud(%input)
        <dimensions = [1], identity = [0.0 : f32]>
        (%accumulator: f32, %value: f32) {
      %sum = arith.addf %accumulator, %value : f32
      nv_tensor_ir.yield %sum : f32
    } : tensor<4x8xf32> -> tensor<4x1xf32>
    return
  }
  func.func @user_defined_extrema() attributes {gpu.kernel} {
    %input = nv_tensor_ir.constant dense<1.0> : tensor<4x8xf32>
    %maximum = nv_tensor_ir.reduce_ud(%input)
        <dimensions = [1], identity = [0xFF800000 : f32]>
        (%accumulator: f32, %value: f32) {
      %result = arith.maximumf %accumulator, %value : f32
      nv_tensor_ir.yield %result : f32
    } : tensor<4x8xf32> -> tensor<4x1xf32>
    %minimum = nv_tensor_ir.reduce_ud(%input)
        <dimensions = [1], identity = [0x7F800000 : f32]>
        (%accumulator: f32, %value: f32) {
      %result = arith.minimumf %accumulator, %value : f32
      nv_tensor_ir.yield %result : f32
    } : tensor<4x8xf32> -> tensor<4x1xf32>
    return
  }
  func.func @user_defined_division() attributes {gpu.kernel} {
    %input = nv_tensor_ir.constant dense<1.0> : tensor<4x8xf32>
    %f32 = nv_tensor_ir.reduce_ud(%input)
        <dimensions = [1], identity = [1.0 : f32]>
        (%accumulator: f32, %value: f32) {
      %result = arith.divf %accumulator, %value : f32
      nv_tensor_ir.yield %result : f32
    } : tensor<4x8xf32> -> tensor<4x1xf32>
    %lowInput = nv_tensor_ir.constant dense<1.0> : tensor<4x8xf16>
    %f16 = nv_tensor_ir.reduce_ud(%lowInput)
        <dimensions = [1], identity = [1.0 : f16]>
        (%accumulator: f16, %value: f16) {
      %result = arith.divf %accumulator, %value : f16
      nv_tensor_ir.yield %result : f16
    } : tensor<4x8xf16> -> tensor<4x1xf16>
    return
  }
  }
}

// CHECK-LABEL: cuda_tile.module @cuda_tile_user_defined_reduction_module {
// CHECK: entry @user_defined_reduction()
// CHECK: %[[INPUT:.*]] = constant <f32: 1.000000e+00> : tile<4x8xf32>
// CHECK: %[[REDUCED:.*]] = reduce %[[INPUT]] dim=1 identities=[0.000000e+00 : f32]
// CHECK: addf
// CHECK: yield
// CHECK: reshape %[[REDUCED]] : tile<4xf32> -> tile<4x1xf32>
// CHECK-LABEL: entry @user_defined_extrema()
// CHECK: maxf {{.*}} propagate_nan
// CHECK: minf {{.*}} propagate_nan
// CHECK-LABEL: entry @user_defined_division()
// CHECK: divf {{.*}} rounding<full> : tile<f32>
// CHECK: %[[F16_LHS:.*]] = ftof {{.*}} : tile<f16> -> tile<f32>
// CHECK: %[[F16_RHS:.*]] = ftof {{.*}} : tile<f16> -> tile<f32>
// CHECK: %[[F16_DIV:.*]] = divf %[[F16_LHS]], %[[F16_RHS]] rounding<full>
// CHECK: ftof %[[F16_DIV]] : tile<f32> -> tile<f16>
