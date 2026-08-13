// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_concat_1way
// CHECK-NOT: cmpi
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[ARG0]]
nv_tensor_ir.graph @test_concat_1way(
    %arg0: tensor<32x16xf32>
    ) -> (tensor<32x16xf32>)
    attributes {tile_size = array<i32: 1, 512>} {
  %out = concatenate %arg0 dimension = 0
    : (tensor<32x16xf32>) -> tensor<32x16xf32>
  results %out : tensor<32x16xf32>
}

// -----

// CHECK-LABEL: @test_concat_across_columns
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[TWO:.*]] = constant <i32: 2>
// CHECK: %[[CMP:.*]] = cmpi less_than %[[BLOCK:.*]], %[[TWO]], unsigned
// CHECK: %[[RESULT:.*]] = if %[[CMP]] -> (tile<1x512xf32>) {
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[ZERO]]]
// CHECK:   yield %[[ARG0]]
// CHECK: } else {
// CHECK:   %[[IDX:.*]] = subi %[[BLOCK]], %[[TWO]]
// CHECK:   %[[ARG1:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[IDX]], %[[ZERO]]]
// CHECK:   yield %[[ARG1]]
// CHECK: }
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_concat_across_columns(
    %arg0: tensor<32x32xf32>,
    %arg1: tensor<48x32xf32>
    ) -> (tensor<80x32xf32>)
    attributes {tile_size = array<i32: 1, 512>} {
  %out = concatenate %arg0, %arg1 dimension = 0
    : (tensor<32x32xf32>,
       tensor<48x32xf32>) -> tensor<80x32xf32>
  results %out : tensor<80x32xf32>
}

// -----

// CHECK-LABEL: @test_concat_across_rows
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[C16:.*]] = constant <i32: 16>
// CHECK-DAG: %[[BIDX:.+]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK-DAG: %[[C2:.*]] = constant <i32: 2>
// CHECK-DAG: %[[X_REM_16:.+]] = remi %[[BIDX]], %[[C16]] unsigned
// CHECK-DAG: %[[X_DIV_16:.+]] = divi %[[BIDX]], %[[C16]] unsigned
// CHECK-DAG: %[[CMP:.*]] = cmpi less_than %[[X_DIV_16]], %[[C2]], unsigned
// CHECK: %[[RESULT:.*]] = if %[[CMP]] -> (tile<2x1x16xf32>) {
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[X_REM_16]], %[[X_DIV_16]], %[[ZERO]]]
// CHECK:   yield %[[ARG0]]
// CHECK: } else {
// CHECK:   %[[IDX:.*]] = subi %[[X_DIV_16]], %[[C2]]
// CHECK:   %[[ARG1:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[X_REM_16]], %[[IDX]], %[[ZERO]]]
// CHECK:   yield %[[ARG1]]
// CHECK: }
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_concat_across_rows(
    %arg0: tensor<32x32xf32>,
    %arg1: tensor<32x48xf32>
    ) -> (tensor<32x80xf32>)
    attributes {tile_size = array<i32: 2, 1, 16>} {
  %out = concatenate %arg0, %arg1 dimension = 1
    : (tensor<32x32xf32>,
       tensor<32x48xf32>) -> tensor<32x80xf32>
  results %out : tensor<32x80xf32>
}

// -----

// CHECK-LABEL: @test_concat_3way
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[TWO:.*]] = constant <i32: 2>
// CHECK-DAG: %[[THREE:.*]] = constant <i32: 3>
// CHECK: %[[CMP1:.*]] = cmpi less_than %[[BLOCK:.*]], %[[TWO]], unsigned
// CHECK: %[[RESULT:.*]] = if %[[CMP1]] -> (tile<32x1x16xf32>) {
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[ZERO]], %[[BLOCK]], %[[ZERO]]]
// CHECK:   yield %[[ARG0]]
// CHECK: } else {
// CHECK:   %[[IDX1:.*]] = subi %[[BLOCK]], %[[TWO]]
// CHECK:   %[[CMP2:.*]] = cmpi less_than %[[IDX1]], %[[THREE]], unsigned
// CHECK:   %[[INNER:.*]] = if %[[CMP2]] -> (tile<32x1x16xf32>) {
// CHECK:     %[[ARG1:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[ZERO]], %[[IDX1]], %[[ZERO]]]
// CHECK:     yield %[[ARG1]]
// CHECK:   } else {
// CHECK:     %[[IDX2:.*]] = subi %[[IDX1]], %[[THREE]]
// CHECK:     %[[ARG2:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[ZERO]], %[[IDX2]], %[[ZERO]]]
// CHECK:     yield %[[ARG2]]
// CHECK:   }
// CHECK:   yield %[[INNER]]
// CHECK: }
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_concat_3way(
    %arg0: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"},
    %arg1: tensor<32x48xf32> {nv_tensor_ir.stride = "(1,32)"},
    %arg2: tensor<32x64xf32> {nv_tensor_ir.stride = "(1,32)"}
    ) -> (tensor<32x144xf32> {nv_tensor_ir.stride = "(1,32)"})
    attributes {tile_size = array<i32: 32, 1, 16>} {
  %out = concatenate %arg0, %arg1, %arg2 dimension = 1
    : (tensor<32x32xf32>,
       tensor<32x48xf32>,
       tensor<32x64xf32>) -> tensor<32x144xf32>
  results %out : tensor<32x144xf32>
}

// -----

// CHECK-LABEL: @test_concat_nested
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[TWO:.*]] = constant <i32: 2>
// CHECK-DAG: %[[FOUR:.*]] = constant <i32: 4>
// CHECK-DAG: %[[FIVE:.*]] = constant <i32: 5>
// CHECK: %[[CMP1:.*]] = cmpi less_than %[[BLOCK:.*]], %[[FIVE]], unsigned
// CHECK: %[[RESULT:.*]] = if %[[CMP1]] -> (tile<32x1x16xf32>) {
// CHECK:   %[[CMP2:.*]] = cmpi less_than %[[BLOCK]], %[[TWO]], unsigned
// CHECK:   %[[LHS:.*]] = if %[[CMP2]] -> (tile<32x1x16xf32>) {
// CHECK:     %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[ZERO]], %[[BLOCK]], %[[ZERO]]]
// CHECK:     yield %[[ARG0]]
// CHECK:   } else {
// CHECK:     %[[IDX1:.*]] = subi %[[BLOCK]], %[[TWO]]
// CHECK:     %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[ZERO]], %[[IDX1]], %[[ZERO]]]
// CHECK:     yield %[[ARG0]]
// CHECK:   }
// CHECK:   yield %[[LHS]]
// CHECK: } else {
// CHECK:   %[[IDX2:.*]] = subi %[[BLOCK]], %[[FIVE]]
// CHECK:   %[[CMP3:.*]] = cmpi less_than %[[IDX2]], %[[FOUR]], unsigned
// CHECK:   %[[RHS:.*]] = if %[[CMP3]] -> (tile<32x1x16xf32>) {
// CHECK:     %[[ARG2:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[ZERO]], %[[IDX2]], %[[ZERO]]]
// CHECK:     yield %[[ARG2]]
// CHECK:   } else {
// CHECK:     %[[IDX3:.*]] = subi %[[IDX2]], %[[FOUR]]
// CHECK:     %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[ZERO]], %[[IDX3]], %[[ZERO]]]
// CHECK:     yield %[[ARG0]]
// CHECK:   }
// CHECK:   yield %[[RHS]]
// CHECK: }
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @test_concat_nested(
    %arg0: tensor<32x32xf32> {nv_tensor_ir.stride = "(1,32)"},
    %arg1: tensor<32x48xf32> {nv_tensor_ir.stride = "(1,32)"},
    %arg2: tensor<32x64xf32> {nv_tensor_ir.stride = "(1,32)"},
    %arg3: tensor<32x80xf32> {nv_tensor_ir.stride = "(1,32)"}
    ) -> (tensor<32x224xf32> {nv_tensor_ir.stride = "(1,32)"})
    attributes {tile_size = array<i32: 32, 1, 16>} {
  %lhs = concatenate %arg0, %arg1 dimension = 1
    : (tensor<32x32xf32>,
       tensor<32x48xf32>) -> tensor<32x80xf32>
  %rhs = concatenate %arg2, %arg3 dimension = 1
    : (tensor<32x64xf32>,
       tensor<32x80xf32>) -> tensor<32x144xf32>
  %out = concatenate %lhs, %rhs dimension = 1
    : (tensor<32x80xf32>,
       tensor<32x144xf32>) -> tensor<32x224xf32>
  results %out : tensor<32x224xf32>
}

// -----

// CHECK-LABEL: @test_concat_transpose_col_major
// CHECK: make_tensor_view {{.*}}, shape = [8, 1, 8], strides = [8, 1, 1]
// CHECK: make_tensor_view {{.*}}, shape = [8, 1, 8], strides = [1, 1, 8]
// CHECK: make_tensor_view {{.*}}, shape = [8, 2, 8], strides = [1, 64, 8]
// CHECK: store_view_tko weak
nv_tensor_ir.graph @test_concat_transpose_col_major(
    %arg0: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"},
    %arg1: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}
    ) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(1,8)"})
    attributes {tile_size = array<i32: 8, 1, 1>} {
  %tr = transpose %arg0 permutation = [1, 0]
    : tensor<8x8xf32> -> tensor<8x8xf32>
  %out = concatenate %tr, %arg1 dimension = 1
    : (tensor<8x8xf32>,
       tensor<8x8xf32>) -> tensor<8x16xf32>
  results %out : tensor<8x16xf32>
}

// -----

// ============================================================================
// TEST 6: Self-referenced concat operand (regression, was LayoutPropInputValidation.
// SelfReferencedConcatOperandLowersWithoutCrashing). The block argument %arg0 is
// both a direct concat operand (first source) and the source of the two sliced
// operands, so the operand-to-iteration-space relation must be keyed by operand
// position. Data flow: source 0 (blocks < 20) loads the block argument directly;
// the two later sources recompute from slices of the same block argument.
// ============================================================================
// CHECK-LABEL: @fused_concatenate
// CHECK-DAG:   %[[CPOS:.*]] = constant <f32: 5.900000e-01> : tile<1xf32>
// CHECK-DAG:   %[[CNEG:.*]] = constant <f32: -5.900000e-01> : tile<1xf32>
// CHECK-DAG:   %[[C20:.*]] = constant <i32: 20> : tile<i32>
// CHECK:       %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK:       %[[CMP0:.*]] = cmpi less_than %[[BLOCK]], %[[C20]], unsigned
// CHECK:       %[[RESULT:.*]] = if %[[CMP0]] -> (tile<1xf32>) {
// CHECK:         make_tensor_view %[[IN:.*]], shape = [20]
// CHECK:         %[[SELF:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]]]
// CHECK:         yield %[[SELF]]
// CHECK:       } else {
// CHECK:         %{{.*}} = subi %[[BLOCK]], %[[C20]]
// CHECK:         %[[INNER:.*]] = if
// CHECK:           %{{.*}} = offset %[[IN]], %{{.*}}
// CHECK:           %[[S0:.*]], %{{.*}} = load_view_tko weak
// CHECK:           %[[A0:.*]] = addf %[[S0]], %[[CNEG]]
// CHECK:           yield %[[A0]]
// CHECK:         } else {
// CHECK:           make_tensor_view %[[IN]], shape = [1]
// CHECK:           %[[S1:.*]], %{{.*}} = load_view_tko weak
// CHECK:           %[[A1:.*]] = addf %[[S1]], %[[CPOS]]
// CHECK:           yield %[[A1]]
// CHECK:         }
// CHECK:         yield %[[INNER]]
// CHECK:       }
// CHECK:       store_view_tko weak %[[RESULT]], %{{.*}}[%[[BLOCK]]]
nv_tensor_ir.graph @fused_concatenate(
    %arg0: tensor<20xf32> {nv_tensor_ir.stride = "(1)"}
    ) -> (tensor<22xf32> {nv_tensor_ir.stride = "(1)"})
    attributes {tile_size = array<i32: 1>} {
  %s0 = slice %arg0 starts = [19] limits = [20] strides = [1] : tensor<20xf32> -> tensor<1xf32>
  %c0 = nv_tensor_ir.constant dense<-0.59> : tensor<1xf32>
  %a0 = add %s0, %c0 : tensor<1xf32>
  %s1 = slice %arg0 starts = [0] limits = [1] strides = [1] : tensor<20xf32> -> tensor<1xf32>
  %c1 = nv_tensor_ir.constant dense<0.59> : tensor<1xf32>
  %a1 = add %s1, %c1 : tensor<1xf32>
  %0 = concatenate %arg0, %a0, %a1 dimension = 0
    : (tensor<20xf32>,
       tensor<1xf32>,
       tensor<1xf32>) -> tensor<22xf32>
  results %0 : tensor<22xf32>
}

// -----

// ============================================================================
// TEST 7: Pruned concat operand (regression, was LayoutPropInputValidation.
// PrunedConcatOperandLowersWithoutCrashing). A slice prunes the first source, so
// the surviving sources carry argument index [1, 2] and the concat builds two
// iteration spaces from three operands. The relation must map through the
// argument index: the two kept operands (%in1, %in2) each load from their own
// space, and the pruned %in0 feeds no space (only two if-branches, %in0 is never
// viewed).
// ============================================================================
// CHECK-LABEL: @pruned_concatenate
// CHECK-SAME:  (%[[IN0:.+]]: tile<ptr<f32>>, %[[IN1:.+]]: tile<ptr<f32>>, %[[IN2:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
// CHECK-DAG:   %[[C2:.*]] = constant <i32: 2> : tile<i32>
// CHECK-DAG:   %[[C5:.*]] = constant <i32: 5> : tile<i32>
// CHECK:       %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK:       %[[SEL:.*]] = remi %[[BLOCK]], %[[C5]] unsigned
// CHECK:       %[[CMP:.*]] = cmpi less_than %[[SEL]], %[[C2]], unsigned
// CHECK:       %[[RESULT:.*]] = if %[[CMP]] -> (tile<8x1x1xf32>) {
// CHECK:         make_tensor_view %[[IN1]], shape = [8, 2, 2]
// CHECK:         %[[T1:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%{{.*}}, %[[SEL]], %{{.*}}]
// CHECK:         yield %[[T1]]
// CHECK:       } else {
// CHECK:         %[[IDX:.*]] = subi %[[SEL]], %[[C2]]
// CHECK:         make_tensor_view %[[IN2]], shape = [8, 3, 2]
// CHECK:         %[[T2:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%{{.*}}, %[[IDX]], %{{.*}}]
// CHECK:         yield %[[T2]]
// CHECK:       }
// CHECK:       store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @pruned_concatenate(
    %in0: tensor<8x2xf32> {nv_tensor_ir.stride = "(2,1)"},
    %in1: tensor<8x4xf32> {nv_tensor_ir.stride = "(4,1)"},
    %in2: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
    ) -> (tensor<8x10xf32> {nv_tensor_ir.stride = "(10,1)"})
    attributes {tile_size = array<i32: 8, 1, 1>} {
  %concat = concatenate %in0, %in1, %in2 dimension = 1
    : (tensor<8x2xf32>,
       tensor<8x4xf32>,
       tensor<8x8xf32>) -> tensor<8x14xf32>
  %out = slice %concat starts = [0, 2] limits = [8, 12] strides = [1, 1] : tensor<8x14xf32> -> tensor<8x10xf32>
  results %out : tensor<8x10xf32>
}

// -----

// ============================================================================
// TEST 8: Self-referenced pruned concat operand (regression, was
// LayoutPropInputValidation.SelfReferencedPrunedConcatOperandLowersWithoutCrashing).
// A computed value %v feeds the concat twice; the slice prunes the first copy.
// The pruned operand still carries %v's iteration-space id, so mapping by
// argument index is required: the kept plain source %arg1 loads on its own space
// and the surviving self-referenced operand recomputes %v (load %arg0 + 0.5),
// while the pruned copy contributes no branch.
// ============================================================================
// CHECK-LABEL: @self_ref_pruned_concatenate
// CHECK-SAME:  (%[[IN0:.+]]: tile<ptr<f32>>, %[[IN1:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
// CHECK-DAG:   %[[HALF:.*]] = constant <f32: 5.000000e-01> : tile<8x1x1xf32>
// CHECK-DAG:   %[[C4:.*]] = constant <i32: 4> : tile<i32>
// CHECK-DAG:   %[[C5:.*]] = constant <i32: 5> : tile<i32>
// CHECK:       %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK:       %[[SEL:.*]] = remi %[[BLOCK]], %[[C5]] unsigned
// CHECK:       %[[CMP:.*]] = cmpi less_than %[[SEL]], %[[C4]], unsigned
// CHECK:       %[[RESULT:.*]] = if %[[CMP]] -> (tile<8x1x1xf32>) {
// CHECK:         make_tensor_view %[[IN1]], shape = [8, 4, 2]
// CHECK:         %[[T1:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%{{.*}}, %[[SEL]], %{{.*}}]
// CHECK:         yield %[[T1]]
// CHECK:       } else {
// CHECK:         make_tensor_view %[[IN0]], shape = [8, 1, 2]
// CHECK:         %[[T0:.*]], %{{.*}} = load_view_tko weak
// CHECK:         %[[ADD:.*]] = addf %[[T0]], %[[HALF]]
// CHECK:         yield %[[ADD]]
// CHECK:       }
// CHECK:       store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @self_ref_pruned_concatenate(
    %arg0: tensor<8x2xf32> {nv_tensor_ir.stride = "(2,1)"},
    %arg1: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
    ) -> (tensor<8x10xf32> {nv_tensor_ir.stride = "(10,1)"})
    attributes {tile_size = array<i32: 8, 1, 1>} {
  %c = nv_tensor_ir.constant dense<0.5> : tensor<8x2xf32>
  %v = add %arg0, %c : tensor<8x2xf32>
  %concat = concatenate %v, %arg1, %v dimension = 1
    : (tensor<8x2xf32>,
       tensor<8x8xf32>,
       tensor<8x2xf32>) -> tensor<8x12xf32>
  %out = slice %concat starts = [0, 2] limits = [8, 12] strides = [1, 1] : tensor<8x12xf32> -> tensor<8x10xf32>
  results %out : tensor<8x10xf32>
}
