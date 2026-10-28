// RUN: tensor_ir-opt %s -split-input-file -discover-iteration-space-info \
// RUN:   -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" \
// RUN:   | FileCheck %s

// Integer MMA must keep a zero i32 accumulator and preserve the signedness
// of both input tensors after converting their element types to signless i8.
// CHECK-LABEL: entry @signed_matmul
// CHECK: %[[ZERO:.+]] = constant <i32: 0> : tile<2x2xi32>
// CHECK: %[[LOOP:.+]] = for {{.*}} iter_values(%[[ACC:.+]] = %[[ZERO]])
// CHECK: %[[A:.+]], %{{.+}} = load_view_tko {{.*}} -> tile<2x2xi8>, token
// CHECK: %[[B:.+]], %{{.+}} = load_view_tko {{.*}} -> tile<2x2xi8>, token
// CHECK: %[[MMA:.+]] = mmai %[[A]], %[[B]], %[[ACC]] signed signed
// CHECK: continue %[[MMA]] : tile<2x2xi32>
// CHECK: store_view_tko weak %[[LOOP]],
nv_tensor_ir.graph @signed_matmul(%a: tensor<8x8xsi8>,
    %b: tensor<8x8xsi8>) -> tensor<8x8xsi32> {
  %r = "nv_tensor_ir.matmul"(%a, %b)
    : (tensor<8x8xsi8>, tensor<8x8xsi8>) -> tensor<8x8xsi32>
  results %r : tensor<8x8xsi32>
}

// -----

// CHECK-LABEL: entry @unsigned_matmul
// CHECK: %[[ZERO:.+]] = constant <i32: 0> : tile<2x2xi32>
// CHECK: %[[LOOP:.+]] = for {{.*}} iter_values(%[[ACC:.+]] = %[[ZERO]])
// CHECK: %[[A:.+]], %{{.+}} = load_view_tko {{.*}} -> tile<2x2xi8>, token
// CHECK: %[[B:.+]], %{{.+}} = load_view_tko {{.*}} -> tile<2x2xi8>, token
// CHECK: %[[MMA:.+]] = mmai %[[A]], %[[B]], %[[ACC]] unsigned unsigned
// CHECK: continue %[[MMA]] : tile<2x2xi32>
// CHECK: store_view_tko weak %[[LOOP]],
nv_tensor_ir.graph @unsigned_matmul(%a: tensor<8x8xui8>,
    %b: tensor<8x8xui8>) -> tensor<8x8xsi32> {
  %r = "nv_tensor_ir.matmul"(%a, %b)
    : (tensor<8x8xui8>, tensor<8x8xui8>) -> tensor<8x8xsi32>
  results %r : tensor<8x8xsi32>
}
