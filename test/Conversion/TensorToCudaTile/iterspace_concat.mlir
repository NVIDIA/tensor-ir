// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_concat_1way
// CHECK-NOT: cmpi
// CHECK: %[[TILE:.*]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[TILE]]
nv_tensor_ir.graph @test_concat_1way(
    %arg0: tensor<32x16xf32>
    ) -> (tensor<32x16xf32>)
    attributes {tile_size = array<i32: 1, 512>} {
  %out = concatenate %arg0 dimension = 0
    : (tensor<32x16xf32>) -> tensor<32x16xf32>
  results %out : tensor<32x16xf32>
}

// -----

// CHECK-LABEL: entry @test_concat_across_columns(
// CHECK-SAME: %[[IN0:[^,]*]]: tile<ptr<f32>>, %[[IN1:[^,]*]]: tile<ptr<f32>>, %[[OUT:[^)]*]]: tile<ptr<f32>>)
// CHECK-DAG: %[[TWO:.*]] = constant <i32: 2>
// CHECK-DAG: %[[FIVE:.*]] = constant <i32: 5>
// CHECK: %[[BIDX:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[BLOCK:.*]] = remi %[[BIDX]], %[[FIVE]] unsigned
// CHECK: %[[COL:.*]] = divi %[[BIDX]], %[[FIVE]] unsigned
// CHECK: %[[CMP:.*]] = cmpi less_than %[[BLOCK]], %[[TWO]], unsigned
// CHECK: %[[RESULT:.*]] = if %[[CMP]] -> (tile<1x512xf32>) {
// CHECK:   %[[VIEW0:.*]] = make_tensor_view %[[IN0]], shape = [2, 512], strides = [512, 1]
// CHECK:   %[[PVIEW0:.*]] = make_partition_view %[[VIEW0]]
// CHECK:   %[[TILE0:.*]], %{{.*}} = load_view_tko weak %[[PVIEW0]][%[[BLOCK]], %[[COL]]]
// CHECK:   yield %[[TILE0]]
// CHECK: } else {
// CHECK:   %[[IDX:.*]] = subi %[[BLOCK]], %[[TWO]]
// CHECK:   %[[VIEW1:.*]] = make_tensor_view %[[IN1]], shape = [3, 512], strides = [512, 1]
// CHECK:   %[[PVIEW1:.*]] = make_partition_view %[[VIEW1]]
// CHECK:   %[[TILE1:.*]], %{{.*}} = load_view_tko weak %[[PVIEW1]][%[[IDX]], %[[COL]]]
// CHECK:   yield %[[TILE1]]
// CHECK: }
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [5, 512], strides = [512, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[BLOCK]], %[[COL]]]
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

// CHECK-LABEL: entry @test_concat_across_rows(
// CHECK-SAME: %[[IN0:[^,]*]]: tile<ptr<f32>>, %[[IN1:[^,]*]]: tile<ptr<f32>>, %[[OUT:[^)]*]]: tile<ptr<f32>>)
// CHECK-DAG: %[[C16:.*]] = constant <i32: 16>
// CHECK-DAG: %[[C5:.*]] = constant <i32: 5>
// CHECK-DAG: %[[BIDX:.+]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK-DAG: %[[C2:.*]] = constant <i32: 2>
// CHECK: %[[X_REM_16:.+]] = remi %[[BIDX]], %[[C16]] unsigned
// CHECK: %[[X_DIV_16:.+]] = divi %[[BIDX]], %[[C16]] unsigned
// CHECK: %[[ROW:.+]] = remi %[[X_DIV_16]], %[[C5]] unsigned
// CHECK: %[[COL:.+]] = divi %[[X_DIV_16]], %[[C5]] unsigned
// CHECK: %[[CMP:.*]] = cmpi less_than %[[ROW]], %[[C2]], unsigned
// CHECK: %[[RESULT:.*]] = if %[[CMP]] -> (tile<2x1x16xf32>) {
// CHECK:   %[[VIEW0:.*]] = make_tensor_view %[[IN0]], shape = [32, 2, 16], strides = [32, 16, 1]
// CHECK:   %[[PVIEW0:.*]] = make_partition_view %[[VIEW0]]
// CHECK:   %[[TILE0:.*]], %{{.*}} = load_view_tko weak %[[PVIEW0]][%[[X_REM_16]], %[[ROW]], %[[COL]]]
// CHECK:   yield %[[TILE0]]
// CHECK: } else {
// CHECK:   %[[IDX:.*]] = subi %[[ROW]], %[[C2]]
// CHECK:   %[[VIEW1:.*]] = make_tensor_view %[[IN1]], shape = [32, 3, 16], strides = [48, 16, 1]
// CHECK:   %[[PVIEW1:.*]] = make_partition_view %[[VIEW1]]
// CHECK:   %[[TILE1:.*]], %{{.*}} = load_view_tko weak %[[PVIEW1]][%[[X_REM_16]], %[[IDX]], %[[COL]]]
// CHECK:   yield %[[TILE1]]
// CHECK: }
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [32, 5, 16], strides = [80, 16, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[X_REM_16]], %[[ROW]], %[[COL]]]
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

// CHECK-LABEL: entry @test_concat_3way(
// CHECK-SAME: %[[IN0:[^,]*]]: tile<ptr<f32>>, %[[IN1:[^,]*]]: tile<ptr<f32>>, %[[IN2:[^,]*]]: tile<ptr<f32>>, %[[OUT:[^)]*]]: tile<ptr<f32>>)
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[TWO:.*]] = constant <i32: 2>
// CHECK-DAG: %[[THREE:.*]] = constant <i32: 3>
// CHECK-DAG: %[[NINE:.*]] = constant <i32: 9>
// CHECK: %[[BIDX:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[BLOCK:.*]] = remi %[[BIDX]], %[[NINE]] unsigned
// CHECK: %[[COL:.*]] = divi %[[BIDX]], %[[NINE]] unsigned
// CHECK: %[[CMP1:.*]] = cmpi less_than %[[BLOCK]], %[[TWO]], unsigned
// CHECK: %[[RESULT:.*]] = if %[[CMP1]] -> (tile<32x1x16xf32>) {
// CHECK:   %[[VIEW0:.*]] = make_tensor_view %[[IN0]], shape = [32, 2, 16], strides = [1, 512, 32]
// CHECK:   %[[PVIEW0:.*]] = make_partition_view %[[VIEW0]]
// CHECK:   %[[TILE0:.*]], %{{.*}} = load_view_tko weak %[[PVIEW0]][%[[ZERO]], %[[BLOCK]], %[[COL]]]
// CHECK:   yield %[[TILE0]]
// CHECK: } else {
// CHECK:   %[[IDX1:.*]] = subi %[[BLOCK]], %[[TWO]]
// CHECK:   %[[CMP2:.*]] = cmpi less_than %[[IDX1]], %[[THREE]], unsigned
// CHECK:   %[[INNER:.*]] = if %[[CMP2]] -> (tile<32x1x16xf32>) {
// CHECK:     %[[VIEW1:.*]] = make_tensor_view %[[IN1]], shape = [32, 3, 16], strides = [1, 512, 32]
// CHECK:     %[[PVIEW1:.*]] = make_partition_view %[[VIEW1]]
// CHECK:     %[[TILE1:.*]], %{{.*}} = load_view_tko weak %[[PVIEW1]][%[[ZERO]], %[[IDX1]], %[[COL]]]
// CHECK:     yield %[[TILE1]]
// CHECK:   } else {
// CHECK:     %[[IDX2:.*]] = subi %[[IDX1]], %[[THREE]]
// CHECK:     %[[VIEW2:.*]] = make_tensor_view %[[IN2]], shape = [32, 4, 16], strides = [1, 512, 32]
// CHECK:     %[[PVIEW2:.*]] = make_partition_view %[[VIEW2]]
// CHECK:     %[[TILE2:.*]], %{{.*}} = load_view_tko weak %[[PVIEW2]][%[[ZERO]], %[[IDX2]], %[[COL]]]
// CHECK:     yield %[[TILE2]]
// CHECK:   }
// CHECK:   yield %[[INNER]]
// CHECK: }
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [32, 9, 16], strides = [1, 512, 32]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[ZERO]], %[[BLOCK]], %[[COL]]]
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

// CHECK-LABEL: entry @test_concat_nested(
// CHECK-SAME: %[[IN0:[^,]*]]: tile<ptr<f32>>, %[[IN1:[^,]*]]: tile<ptr<f32>>, %[[IN2:[^,]*]]: tile<ptr<f32>>, %[[IN3:[^,]*]]: tile<ptr<f32>>, %[[OUT:[^)]*]]: tile<ptr<f32>>)
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[TWO:.*]] = constant <i32: 2>
// CHECK-DAG: %[[FOUR:.*]] = constant <i32: 4>
// CHECK-DAG: %[[FIVE:.*]] = constant <i32: 5>
// CHECK-DAG: %[[FOURTEEN:.*]] = constant <i32: 14>
// CHECK: %[[BIDX:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[BLOCK:.*]] = remi %[[BIDX]], %[[FOURTEEN]] unsigned
// CHECK: %[[COL:.*]] = divi %[[BIDX]], %[[FOURTEEN]] unsigned
// CHECK: %[[CMP1:.*]] = cmpi less_than %[[BLOCK]], %[[FIVE]], unsigned
// CHECK: %[[RESULT:.*]] = if %[[CMP1]] -> (tile<32x1x16xf32>) {
// CHECK:   %[[CMP2:.*]] = cmpi less_than %[[BLOCK]], %[[TWO]], unsigned
// CHECK:   %[[LHS:.*]] = if %[[CMP2]] -> (tile<32x1x16xf32>) {
// CHECK:     %[[VIEW0:.*]] = make_tensor_view %[[IN0]], shape = [32, 2, 16], strides = [1, 512, 32]
// CHECK:     %[[PVIEW0:.*]] = make_partition_view %[[VIEW0]]
// CHECK:     %[[TILE0:.*]], %{{.*}} = load_view_tko weak %[[PVIEW0]][%[[ZERO]], %[[BLOCK]], %[[COL]]]
// CHECK:     yield %[[TILE0]]
// CHECK:   } else {
// CHECK:     %[[IDX1:.*]] = subi %[[BLOCK]], %[[TWO]]
// CHECK:     %[[VIEW1:.*]] = make_tensor_view %[[IN1]], shape = [32, 3, 16], strides = [1, 512, 32]
// CHECK:     %[[PVIEW1:.*]] = make_partition_view %[[VIEW1]]
// CHECK:     %[[TILE1:.*]], %{{.*}} = load_view_tko weak %[[PVIEW1]][%[[ZERO]], %[[IDX1]], %[[COL]]]
// CHECK:     yield %[[TILE1]]
// CHECK:   }
// CHECK:   yield %[[LHS]]
// CHECK: } else {
// CHECK:   %[[IDX2:.*]] = subi %[[BLOCK]], %[[FIVE]]
// CHECK:   %[[CMP3:.*]] = cmpi less_than %[[IDX2]], %[[FOUR]], unsigned
// CHECK:   %[[RHS:.*]] = if %[[CMP3]] -> (tile<32x1x16xf32>) {
// CHECK:     %[[VIEW2:.*]] = make_tensor_view %[[IN2]], shape = [32, 4, 16], strides = [1, 512, 32]
// CHECK:     %[[PVIEW2:.*]] = make_partition_view %[[VIEW2]]
// CHECK:     %[[TILE2:.*]], %{{.*}} = load_view_tko weak %[[PVIEW2]][%[[ZERO]], %[[IDX2]], %[[COL]]]
// CHECK:     yield %[[TILE2]]
// CHECK:   } else {
// CHECK:     %[[IDX3:.*]] = subi %[[IDX2]], %[[FOUR]]
// CHECK:     %[[VIEW3:.*]] = make_tensor_view %[[IN3]], shape = [32, 5, 16], strides = [1, 512, 32]
// CHECK:     %[[PVIEW3:.*]] = make_partition_view %[[VIEW3]]
// CHECK:     %[[TILE3:.*]], %{{.*}} = load_view_tko weak %[[PVIEW3]][%[[ZERO]], %[[IDX3]], %[[COL]]]
// CHECK:     yield %[[TILE3]]
// CHECK:   }
// CHECK:   yield %[[RHS]]
// CHECK: }
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [32, 14, 16], strides = [1, 512, 32]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[ZERO]], %[[BLOCK]], %[[COL]]]
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

// The middle dimension has extent one, so its stride never contributes to an
// address. It carries the pitch that clears the inner dimensions rather than a
// unit stride, which would otherwise collide with the genuinely contiguous
// dimension of these column-major operands.
// CHECK-LABEL: entry @test_concat_transpose_col_major(
// CHECK-SAME: %[[IN0:[^,]+]]: tile<ptr<f32>>, %[[IN1:[^,]+]]: tile<ptr<f32>>, %[[OUT:[^)]+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[TWO:.*]] = constant <i32: 2>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[SEL:.*]] = remi %[[BLOCK]], %[[TWO]] unsigned
// CHECK: %[[COL:.*]] = divi %[[BLOCK]], %[[TWO]] unsigned
// CHECK: %[[CMP:.*]] = cmpi less_than %[[SEL]], %[[ONE]], unsigned
// CHECK: %[[RESULT:.*]] = if %[[CMP]] -> (tile<8x1x1xf32>) {
// CHECK:   %[[VIEW0:.*]] = make_tensor_view %[[IN0]], shape = [8, 1, 8], strides = [8, 2, 1]
// CHECK:   %[[PVIEW0:.*]] = make_partition_view %[[VIEW0]]
// CHECK:   %[[TILE0:.*]], %{{.*}} = load_view_tko weak %[[PVIEW0]][%[[ZERO]], %[[ZERO]], %[[COL]]]
// CHECK:   yield %[[TILE0]]
// CHECK: } else {
// CHECK:   %[[VIEW1:.*]] = make_tensor_view %[[IN1]], shape = [8, 1, 8], strides = [1, 2, 8]
// CHECK:   %[[PVIEW1:.*]] = make_partition_view %[[VIEW1]]
// CHECK:   %[[TILE1:.*]], %{{.*}} = load_view_tko weak %[[PVIEW1]][%[[ZERO]], %[[ZERO]], %[[COL]]]
// CHECK:   yield %[[TILE1]]
// CHECK: }
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [8, 2, 8], strides = [1, 64, 8]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[ZERO]], %[[SEL]], %[[COL]]]
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
// CHECK-DAG:   %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK-DAG:   %[[C2:.*]] = constant <i32: 2> : tile<i32>
// CHECK-DAG:   %[[C5:.*]] = constant <i32: 5> : tile<i32>
// CHECK:       %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK:       %[[SEL:.*]] = remi %[[BLOCK]], %[[C5]] unsigned
// CHECK:       %[[COL:.*]] = divi %[[BLOCK]], %[[C5]] unsigned
// CHECK:       %[[CMP:.*]] = cmpi less_than %[[SEL]], %[[C2]], unsigned
// CHECK:       %[[RESULT:.*]] = if %[[CMP]] -> (tile<8x1x1xf32>) {
// CHECK:         %[[VIEW1:.*]] = make_tensor_view %[[IN1]], shape = [8, 2, 2], strides = [4, 2, 1]
// CHECK:         %[[PVIEW1:.*]] = make_partition_view %[[VIEW1]]
// CHECK:         %[[T1:.*]], %{{.*}} = load_view_tko weak %[[PVIEW1]][%[[ZERO]], %[[SEL]], %[[COL]]]
// CHECK:         yield %[[T1]]
// CHECK:       } else {
// CHECK:         %[[IDX:.*]] = subi %[[SEL]], %[[C2]]
// CHECK:         %[[VIEW2:.*]] = make_tensor_view %[[IN2]], shape = [8, 3, 2], strides = [8, 2, 1]
// CHECK:         %[[PVIEW2:.*]] = make_partition_view %[[VIEW2]]
// CHECK:         %[[T2:.*]], %{{.*}} = load_view_tko weak %[[PVIEW2]][%[[ZERO]], %[[IDX]], %[[COL]]]
// CHECK:         yield %[[T2]]
// CHECK:       }
// CHECK:       %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [8, 5, 2], strides = [10, 2, 1]
// CHECK:       %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
// CHECK:       store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[ZERO]], %[[SEL]], %[[COL]]]
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
// CHECK-DAG:   %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK-DAG:   %[[C4:.*]] = constant <i32: 4> : tile<i32>
// CHECK-DAG:   %[[C5:.*]] = constant <i32: 5> : tile<i32>
// CHECK:       %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK:       %[[SEL:.*]] = remi %[[BLOCK]], %[[C5]] unsigned
// CHECK:       %[[COL:.*]] = divi %[[BLOCK]], %[[C5]] unsigned
// CHECK:       %[[CMP:.*]] = cmpi less_than %[[SEL]], %[[C4]], unsigned
// CHECK:       %[[RESULT:.*]] = if %[[CMP]] -> (tile<8x1x1xf32>) {
// CHECK:         %[[VIEW1:.*]] = make_tensor_view %[[IN1]], shape = [8, 4, 2], strides = [8, 2, 1]
// CHECK:         %[[PVIEW1:.*]] = make_partition_view %[[VIEW1]]
// CHECK:         %[[T1:.*]], %{{.*}} = load_view_tko weak %[[PVIEW1]][%[[ZERO]], %[[SEL]], %[[COL]]]
// CHECK:         yield %[[T1]]
// CHECK:       } else {
// CHECK:         %[[VIEW0:.*]] = make_tensor_view %[[IN0]], shape = [8, 1, 2], strides = [2, 2, 1]
// CHECK:         %[[PVIEW0:.*]] = make_partition_view %[[VIEW0]]
// CHECK:         %[[T0:.*]], %{{.*}} = load_view_tko weak %[[PVIEW0]][%[[ZERO]], %[[ZERO]], %[[COL]]]
// CHECK:         %[[ADD:.*]] = addf %[[T0]], %[[HALF]]
// CHECK:         yield %[[ADD]]
// CHECK:       }
// CHECK:       %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [8, 5, 2], strides = [10, 2, 1]
// CHECK:       %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
// CHECK:       store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[ZERO]], %[[SEL]], %[[COL]]]
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
