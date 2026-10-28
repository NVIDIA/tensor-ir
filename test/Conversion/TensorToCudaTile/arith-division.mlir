// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation" -split-input-file | FileCheck %s

// Unsigned floor division is arith.divui; Arith has no floordivui operation.

module {
  gpu.module @kernels {
    func.func @divide_index(%lhs: index, %rhs: index) attributes {gpu.kernel} {
      // CHECK-LABEL: entry @divide_index(
      // CHECK-SAME: %[[LHS:[^:]+]]: tile<i32>, %[[RHS:[^:]+]]: tile<i32>
      // CHECK: divi %[[LHS]], %[[RHS]] signed rounding<positive_inf> : tile<i32>
      %ceildivsi = arith.ceildivsi %lhs, %rhs : index
      // CHECK: divi %[[LHS]], %[[RHS]] unsigned rounding<positive_inf> : tile<i32>
      %ceildivui = arith.ceildivui %lhs, %rhs : index
      // CHECK: divi %[[LHS]], %[[RHS]] signed rounding<negative_inf> : tile<i32>
      %floordivsi = arith.floordivsi %lhs, %rhs : index
      // CHECK: divi %[[LHS]], %[[RHS]] unsigned : tile<i32>
      %divui = arith.divui %lhs, %rhs : index
      // CHECK: return
      return
    }
  }
}

// -----

module {
  gpu.module @kernels {
    func.func @divide_i32(%lhs: i32, %rhs: i32) attributes {gpu.kernel} {
      // CHECK-LABEL: entry @divide_i32(
      // CHECK-SAME: %[[LHS:[^:]+]]: tile<i32>, %[[RHS:[^:]+]]: tile<i32>
      // CHECK: divi %[[LHS]], %[[RHS]] signed rounding<positive_inf> : tile<i32>
      %ceildivsi = arith.ceildivsi %lhs, %rhs : i32
      // CHECK: divi %[[LHS]], %[[RHS]] unsigned rounding<positive_inf> : tile<i32>
      %ceildivui = arith.ceildivui %lhs, %rhs : i32
      // CHECK: divi %[[LHS]], %[[RHS]] signed rounding<negative_inf> : tile<i32>
      %floordivsi = arith.floordivsi %lhs, %rhs : i32
      // CHECK: divi %[[LHS]], %[[RHS]] unsigned : tile<i32>
      %divui = arith.divui %lhs, %rhs : i32
      // CHECK: return
      return
    }
  }
}

// -----

module {
  gpu.module @kernels {
    func.func @divide_i64(%lhs: i64, %rhs: i64) attributes {gpu.kernel} {
      // CHECK-LABEL: entry @divide_i64(
      // CHECK-SAME: %[[LHS:[^:]+]]: tile<i64>, %[[RHS:[^:]+]]: tile<i64>
      // CHECK: divi %[[LHS]], %[[RHS]] signed rounding<positive_inf> : tile<i64>
      %ceildivsi = arith.ceildivsi %lhs, %rhs : i64
      // CHECK: divi %[[LHS]], %[[RHS]] unsigned rounding<positive_inf> : tile<i64>
      %ceildivui = arith.ceildivui %lhs, %rhs : i64
      // CHECK: divi %[[LHS]], %[[RHS]] signed rounding<negative_inf> : tile<i64>
      %floordivsi = arith.floordivsi %lhs, %rhs : i64
      // CHECK: divi %[[LHS]], %[[RHS]] unsigned : tile<i64>
      %divui = arith.divui %lhs, %rhs : i64
      // CHECK: return
      return
    }
  }
}

// -----

module {
  gpu.module @kernels {
    func.func @divide_tensor(%input: memref<8xsi32>) attributes {gpu.kernel} {
      // CHECK-LABEL: entry @divide_tensor(
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // CHECK: %[[LHS:[^,]+]], %{{.*}} = load_view_tko
      %lhs_signed = nv_tensor_ir.load %input tile_indices [%c0], alignment 4, padding <zero>
          : memref<8xsi32> to tensor<4xsi32>
      // CHECK: %[[RHS:[^,]+]], %{{.*}} = load_view_tko
      %rhs_signed = nv_tensor_ir.load %input tile_indices [%c1], alignment 4, padding <zero>
          : memref<8xsi32> to tensor<4xsi32>
      %lhs = builtin.unrealized_conversion_cast %lhs_signed : tensor<4xsi32> to tensor<4xi32>
      %rhs = builtin.unrealized_conversion_cast %rhs_signed : tensor<4xsi32> to tensor<4xi32>
      // CHECK: divi %[[LHS]], %[[RHS]] signed rounding<positive_inf> : tile<4xi32>
      %ceildivsi = arith.ceildivsi %lhs, %rhs : tensor<4xi32>
      // CHECK: divi %[[LHS]], %[[RHS]] unsigned rounding<positive_inf> : tile<4xi32>
      %ceildivui = arith.ceildivui %lhs, %rhs : tensor<4xi32>
      // CHECK: divi %[[LHS]], %[[RHS]] signed rounding<negative_inf> : tile<4xi32>
      %floordivsi = arith.floordivsi %lhs, %rhs : tensor<4xi32>
      // CHECK: divi %[[LHS]], %[[RHS]] unsigned : tile<4xi32>
      %divui = arith.divui %lhs, %rhs : tensor<4xi32>
      // CHECK: return
      return
    }
  }
}
