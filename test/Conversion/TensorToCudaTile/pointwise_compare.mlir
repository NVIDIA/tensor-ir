// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s
// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// Verify comparison operation lowering for F32, SI32, UI32 types.
// Other types are verified in the E2E test.

// ============================================================================
// Floating point compare (ordered)
// ============================================================================

// CHECK-LABEL: @cmp_f32_oeq
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf equal ordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_oeq(%arg0: tensor<32x64xf32>,
                                %arg1: tensor<32x64xf32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 oeq %arg1 : tensor<32x64xf32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_f32_one
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf not_equal ordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_one(%arg0: tensor<16x32xf32>,
                                %arg1: tensor<16x32xf32>) -> (tensor<16x32xi1>) {
  %result = cmp %arg0 one %arg1 : tensor<16x32xf32>
  results %result : tensor<16x32xi1>
}

// -----

// CHECK-LABEL: @cmp_f32_ole
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf less_than_or_equal ordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_ole(%arg0: tensor<16x32xf32>,
                                %arg1: tensor<16x32xf32>) -> (tensor<16x32xi1>) {
  %result = cmp %arg0 ole %arg1 : tensor<16x32xf32>
  results %result : tensor<16x32xi1>
}

// -----

// CHECK-LABEL: @cmp_f32_oge
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf greater_than_or_equal ordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_oge(%arg0: tensor<16x32xf32>,
                                %arg1: tensor<16x32xf32>) -> (tensor<16x32xi1>) {
  %result = cmp %arg0 oge %arg1 : tensor<16x32xf32>
  results %result : tensor<16x32xi1>
}

// -----

// CHECK-LABEL: @cmp_f32_olt
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf less_than ordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_olt(%arg0: tensor<16x32xf32>,
                                %arg1: tensor<16x32xf32>) -> (tensor<16x32xi1>) {
  %result = cmp %arg0 olt %arg1 : tensor<16x32xf32>
  results %result : tensor<16x32xi1>
}

// -----

// CHECK-LABEL: @cmp_f32_ogt
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf greater_than ordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_ogt(%arg0: tensor<16x32xf32>,
                                %arg1: tensor<16x32xf32>) -> (tensor<16x32xi1>) {
  %result = cmp %arg0 ogt %arg1 : tensor<16x32xf32>
  results %result : tensor<16x32xi1>
}

// -----

// ============================================================================
// Floating point compare (unordered)
// ============================================================================

// CHECK-LABEL: @cmp_f32_ueq
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf equal unordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_ueq(%arg0: tensor<32x64xf32>,
                                %arg1: tensor<32x64xf32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 ueq %arg1 : tensor<32x64xf32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_f32_une
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf not_equal unordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_une(%arg0: tensor<16x32xf32>,
                                %arg1: tensor<16x32xf32>) -> (tensor<16x32xi1>) {
  %result = cmp %arg0 une %arg1 : tensor<16x32xf32>
  results %result : tensor<16x32xi1>
}

// -----

// CHECK-LABEL: @cmp_f32_ule
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf less_than_or_equal unordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_ule(%arg0: tensor<16x32xf32>,
                                %arg1: tensor<16x32xf32>) -> (tensor<16x32xi1>) {
  %result = cmp %arg0 ule %arg1 : tensor<16x32xf32>
  results %result : tensor<16x32xi1>
}

// -----

// CHECK-LABEL: @cmp_f32_uge
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf greater_than_or_equal unordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_uge(%arg0: tensor<16x32xf32>,
                                %arg1: tensor<16x32xf32>) -> (tensor<16x32xi1>) {
  %result = cmp %arg0 uge %arg1 : tensor<16x32xf32>
  results %result : tensor<16x32xi1>
}

// -----

// CHECK-LABEL: @cmp_f32_ult
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf less_than unordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_ult(%arg0: tensor<16x32xf32>,
                                %arg1: tensor<16x32xf32>) -> (tensor<16x32xi1>) {
  %result = cmp %arg0 ult %arg1 : tensor<16x32xf32>
  results %result : tensor<16x32xi1>
}

// -----

// CHECK-LABEL: @cmp_f32_ugt
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+f32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpf greater_than unordered %[[ARG0]], %[[ARG1]] : [[TILE]]
nv_tensor_ir.graph @cmp_f32_ugt(%arg0: tensor<16x32xf32>,
                                %arg1: tensor<16x32xf32>) -> (tensor<16x32xi1>) {
  %result = cmp %arg0 ugt %arg1 : tensor<16x32xf32>
  results %result : tensor<16x32xi1>
}

// -----

// ============================================================================
// Signed integer compare
// ============================================================================

// CHECK-LABEL: @cmp_si32_eq
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi equal %[[ARG0]], %[[ARG1]], signed : [[TILE]]
nv_tensor_ir.graph @cmp_si32_eq(%arg0: tensor<32x64xsi32>,
                                %arg1: tensor<32x64xsi32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 eq %arg1 : tensor<32x64xsi32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_si32_neq
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi not_equal %[[ARG0]], %[[ARG1]], signed : [[TILE]]
nv_tensor_ir.graph @cmp_si32_neq(%arg0: tensor<32x64xsi32>,
                                 %arg1: tensor<32x64xsi32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 neq %arg1 : tensor<32x64xsi32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_si32_le
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi less_than_or_equal %[[ARG0]], %[[ARG1]], signed : [[TILE]]
nv_tensor_ir.graph @cmp_si32_le(%arg0: tensor<32x64xsi32>,
                                %arg1: tensor<32x64xsi32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 le %arg1 : tensor<32x64xsi32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_si32_ge
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi greater_than_or_equal %[[ARG0]], %[[ARG1]], signed : [[TILE]]
nv_tensor_ir.graph @cmp_si32_ge(%arg0: tensor<32x64xsi32>,
                                %arg1: tensor<32x64xsi32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 ge %arg1 : tensor<32x64xsi32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_si32_lt
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi less_than %[[ARG0]], %[[ARG1]], signed : [[TILE]]
nv_tensor_ir.graph @cmp_si32_lt(%arg0: tensor<32x64xsi32>,
                                %arg1: tensor<32x64xsi32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 lt %arg1 : tensor<32x64xsi32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_si32_gt
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi greater_than %[[ARG0]], %[[ARG1]], signed : [[TILE]]
nv_tensor_ir.graph @cmp_si32_gt(%arg0: tensor<32x64xsi32>,
                                %arg1: tensor<32x64xsi32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 gt %arg1 : tensor<32x64xsi32>
  results %result : tensor<32x64xi1>
}

// -----

// ============================================================================
// Unsigned integer compare
// ============================================================================

// CHECK-LABEL: @cmp_ui32_eq
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi equal %[[ARG0]], %[[ARG1]], unsigned : [[TILE]]
nv_tensor_ir.graph @cmp_ui32_eq(%arg0: tensor<32x64xui32>,
                                %arg1: tensor<32x64xui32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 eq %arg1 : tensor<32x64xui32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_ui32_neq
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi not_equal %[[ARG0]], %[[ARG1]], unsigned : [[TILE]]
nv_tensor_ir.graph @cmp_ui32_neq(%arg0: tensor<32x64xui32>,
                                 %arg1: tensor<32x64xui32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 neq %arg1 : tensor<32x64xui32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_ui32_le
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi less_than_or_equal %[[ARG0]], %[[ARG1]], unsigned : [[TILE]]
nv_tensor_ir.graph @cmp_ui32_le(%arg0: tensor<32x64xui32>,
                                %arg1: tensor<32x64xui32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 le %arg1 : tensor<32x64xui32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_ui32_ge
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi greater_than_or_equal %[[ARG0]], %[[ARG1]], unsigned : [[TILE]]
nv_tensor_ir.graph @cmp_ui32_ge(%arg0: tensor<32x64xui32>,
                                %arg1: tensor<32x64xui32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 ge %arg1 : tensor<32x64xui32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_ui32_lt
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi less_than %[[ARG0]], %[[ARG1]], unsigned : [[TILE]]
nv_tensor_ir.graph @cmp_ui32_lt(%arg0: tensor<32x64xui32>,
                                %arg1: tensor<32x64xui32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 lt %arg1 : tensor<32x64xui32>
  results %result : tensor<32x64xi1>
}

// -----

// CHECK-LABEL: @cmp_ui32_gt
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE:tile<([0-9]+x)+i32>]]
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: cmpi greater_than %[[ARG0]], %[[ARG1]], unsigned : [[TILE]]
nv_tensor_ir.graph @cmp_ui32_gt(%arg0: tensor<32x64xui32>,
                                %arg1: tensor<32x64xui32>) -> (tensor<32x64xi1>) {
  %result = cmp %arg0 gt %arg1 : tensor<32x64xui32>
  results %result : tensor<32x64xi1>
}
