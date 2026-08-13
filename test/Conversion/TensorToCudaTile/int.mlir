// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s


// CHECK-LABEL: entry @test_integer_types
// CHECK-SAME: tile<ptr<i8>>
// CHECK-SAME: tile<ptr<i32>>
// CHECK-SAME: tile<ptr<f32>>
// CHECK-DAG: make_tensor_view {{.*}} : tensor_view<{{[0-9]+}}x{{[0-9]+}}x{{[0-9]+}}xi8
// CHECK-DAG: make_tensor_view {{.*}} : tensor_view<{{[0-9]+}}x{{[0-9]+}}xi32
// CHECK-DAG: make_tensor_view {{.*}} : tensor_view<{{[0-9]+}}x{{[0-9]+}}xf32
// CHECK-DAG: load_view_tko {{.*}} -> tile<{{[0-9]+}}x{{[0-9]+}}x{{[0-9]+}}xi8>
// CHECK-DAG: load_view_tko {{.*}} -> tile<{{[0-9]+}}x{{[0-9]+}}xi32>
// CHECK-DAG: load_view_tko {{.*}} -> tile<{{[0-9]+}}x{{[0-9]+}}xf32>
// CHECK-DAG: store_view_tko {{.*}} tile<{{[0-9]+}}x{{[0-9]+}}x{{[0-9]+}}xi8>
// CHECK-DAG: store_view_tko {{.*}} tile<{{[0-9]+}}x{{[0-9]+}}xi32>
// CHECK-DAG: store_view_tko {{.*}} tile<{{[0-9]+}}x{{[0-9]+}}xf32>
module {
  nv_tensor_ir.graph @test_integer_types(
    %a: tensor<8x16x32xsi8>,
    %b: tensor<64x128xsi32>,
    %c: tensor<64x128xf32>) -> (
      tensor<8x16x32xsi8>,
      tensor<64x128xsi32>,
      tensor<64x128xf32>) {
      results %a, %b, %c : tensor<8x16x32xsi8>, tensor<64x128xsi32>, tensor<64x128xf32>
  }
}

// -----

// Test signed and unsigned integers with a mix of static and dynamic tensors
// CHECK-LABEL: entry @test_signed_unsigned_dynamic
// CHECK-SAME: tile<ptr<i32>>
// CHECK-SAME: tile<ptr<i32>>
// CHECK-SAME: tile<i32>
// CHECK-SAME: tile<i32>
// CHECK-SAME: tile<i32>
// CHECK-SAME: tile<i32>
// Verify both tensor views are created correctly
// CHECK: make_tensor_view {{.*}} : tensor_view<32x64xi32
// CHECK: make_tensor_view {{.*}} -> tensor_view<?x?xi32
// CHECK: load_view_tko {{.*}} -> tile<{{[0-9]+}}x{{[0-9]+}}xi32>
// CHECK: load_view_tko {{.*}} -> tile<{{[0-9]+}}x{{[0-9]+}}xi32>
module {
  nv_tensor_ir.graph @test_signed_unsigned_dynamic(
    %static_signed: tensor<32x64xsi32>,
    %dynamic_unsigned: tensor<?x?xui32>) -> (
      tensor<32x64xsi32>,
      tensor<?x?xui32>) {
      results %static_signed, %dynamic_unsigned : 
        tensor<32x64xsi32>,
        tensor<?x?xui32>
  }
}
