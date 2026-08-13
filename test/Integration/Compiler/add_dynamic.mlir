// RUN: tensor_ir-compiler %s --dynamic-dims=16,8 --dynamic-strides=8 --tile-size=8x8 --verbose --launch --verify

// Fully dynamic shape: both dims unknown at compile time.
// Tests that the OSS compiler correctly extracts KernelArgLayout with
// dynamic sizes and strides, wires FlatArgPacker + TileBasedGridComputer,
// and that the test harness resolves both dynamic dims and dynamic strides.
module {
  nv_tensor_ir.graph @add_dynamic_shape(
    %a: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"},
    %b: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"}
  ) -> (tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"}) {
    %add = add %a, %b : tensor<?x?xf32>
    results %add : tensor<?x?xf32>
  }
}
