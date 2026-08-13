// RUN: tensor_ir-compiler %s --tile-size=2x2 --verbose --launch --verify

module {
  nv_tensor_ir.graph @matmul_f32_static(
    %a: tensor<8x8xf32>, // (M, K)
    %b: tensor<8x8xf32>) -> ( // (K, N)
      tensor<8x8xf32>) { // (M, N)
    %c = "nv_tensor_ir.matmul"(%a, %b) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    results %c : tensor<8x8xf32>
  }
}
