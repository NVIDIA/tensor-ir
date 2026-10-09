// RUN: tensor_ir-compiler %s --verbose --launch --verify

// Keep the rank-zero input before the tensor used by the computation. Runtime
// argument packing must preserve its pointer-only ABI slot so the following
// input and output pointers remain aligned with the kernel signature.
module {
  nv_tensor_ir.graph @rank_zero_argument(
    %scalar: tensor<f32>,
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %result = abs %input : tensor<8x8xf32>
    results %result : tensor<8x8xf32>
  }
}
