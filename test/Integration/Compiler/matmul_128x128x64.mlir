// RUN: tensor_ir-compiler %s --tile-size=8x8 --verbose --launch --verify
// RUN: tensor_ir-compiler %s --tile-size=32x32 --verbose --launch --verify
// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @matmul_f32_static(
    %a: tensor<128x64xf32>, // (M, K)
    %b: tensor<64x128xf32>) -> ( // (K, N)
      tensor<128x128xf32>) { // (M, N)
    %c = "nv_tensor_ir.matmul"(%a, %b) : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    results %c : tensor<128x128xf32>
  }
}
