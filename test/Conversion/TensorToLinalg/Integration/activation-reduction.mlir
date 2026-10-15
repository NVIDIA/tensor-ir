// RUN: tensor_ir-opt %s --tir-linalg-pipeline='use-libm' \
// RUN:   | mlir-runner -e main -entry-point-result=void

// These reference values are evaluations of the documented mathematical
// formulas, with beta=2 and gradients [2, 3, 4]. They check complete results
// rather than duplicating the emitted operation sequences.
func.func @close(%actual: f32, %expected: f32, %tolerance: f32) {
  %diff = arith.subf %actual, %expected : f32
  %abs = math.absf %diff : f32
  %ok = arith.cmpf ole, %abs, %tolerance : f32
  cf.assert %ok, "activation or reduction numerical mismatch"
  return
}

func.func @relu_fwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.relu_fwd %x : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @relu_bwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.relu_bwd %x, %g : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @sigmoid_fwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.sigmoid_fwd %x : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @sigmoid_bwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.sigmoid_bwd %x, %g : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @tanh_fwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.tanh_fwd %x : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @tanh_bwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.tanh_bwd %x, %g : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @gelu_fwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.gelu_fwd %x : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @gelu_bwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.gelu_bwd %x, %g : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @gelu_approx_tanh_fwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.gelu_approx_tanh_fwd %x : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @gelu_approx_tanh_bwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.gelu_approx_tanh_bwd %x, %g : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @softplus_fwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.softplus_fwd<beta = 2.0 : f64> %x : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @softplus_bwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.softplus_bwd<beta = 2.0 : f64> %x, %g : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @swish_fwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.swish_fwd<beta = 2.0 : f64> %x : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @swish_bwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.swish_bwd<beta = 2.0 : f64> %x, %g : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @elu_fwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.elu_fwd<beta = 2.0 : f64> %x : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @elu_bwd(%x: tensor<3xf32>, %g: tensor<3xf32>) -> tensor<3xf32> {
  %r = nv_tensor_ir.elu_bwd<beta = 2.0 : f64> %x, %g : tensor<3xf32>
  return %r : tensor<3xf32>
}

func.func @reduce_add(%x: tensor<2x3xf32>) -> tensor<2x1xf32> {
  %r = nv_tensor_ir.reduce(%x)<dimensions = [1], reduction_mode = <add>> : tensor<2x3xf32> -> tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

func.func @reduce_mul(%x: tensor<2x3xf32>) -> tensor<2x1xf32> {
  %r = nv_tensor_ir.reduce(%x)<dimensions = [1], reduction_mode = <mul>> : tensor<2x3xf32> -> tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

func.func @reduce_min(%x: tensor<2x3xf32>) -> tensor<2x1xf32> {
  %r = nv_tensor_ir.reduce(%x)<dimensions = [1], reduction_mode = <min>> : tensor<2x3xf32> -> tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

func.func @reduce_max(%x: tensor<2x3xf32>) -> tensor<2x1xf32> {
  %r = nv_tensor_ir.reduce(%x)<dimensions = [1], reduction_mode = <max>> : tensor<2x3xf32> -> tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

func.func @reduce_amax(%x: tensor<2x3xf32>) -> tensor<2x1xf32> {
  %r = nv_tensor_ir.reduce(%x)<dimensions = [1], reduction_mode = <amax>> : tensor<2x3xf32> -> tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

func.func @reduce_avg(%x: tensor<2x3xf32>) -> tensor<2x1xf32> {
  %r = nv_tensor_ir.reduce(%x)<dimensions = [1], reduction_mode = <avg>> : tensor<2x3xf32> -> tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

func.func @reduce_norm1(%x: tensor<2x3xf32>) -> tensor<2x1xf32> {
  %r = nv_tensor_ir.reduce(%x)<dimensions = [1], reduction_mode = <norm1>> : tensor<2x3xf32> -> tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

func.func @reduce_norm2(%x: tensor<2x3xf32>) -> tensor<2x1xf32> {
  %r = nv_tensor_ir.reduce(%x)<dimensions = [1], reduction_mode = <norm2>> : tensor<2x3xf32> -> tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

func.func @reduce_mul_no_zeros(%x: tensor<2x3xf32>) -> tensor<2x1xf32> {
  %r = nv_tensor_ir.reduce(%x)<dimensions = [1], reduction_mode = <mul_no_zeros>> : tensor<2x3xf32> -> tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

func.func @main() {
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i2 = arith.constant 2 : index
  %tol = arith.constant 2.0e-6 : f32
  %x = arith.constant dense<[-1.0, 0.0, 1.0]> : tensor<3xf32>
  %g = arith.constant dense<[2.0, 3.0, 4.0]> : tensor<3xf32>
  %relu_fwd = func.call @relu_fwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %relu_fwd_0 = tensor.extract %relu_fwd[%i0] : tensor<3xf32>
  %relu_fwd_expected_0 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%relu_fwd_0, %relu_fwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %relu_fwd_1 = tensor.extract %relu_fwd[%i1] : tensor<3xf32>
  %relu_fwd_expected_1 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%relu_fwd_1, %relu_fwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %relu_fwd_2 = tensor.extract %relu_fwd[%i2] : tensor<3xf32>
  %relu_fwd_expected_2 = arith.constant 1.0000000000000000e+00 : f32
  func.call @close(%relu_fwd_2, %relu_fwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %relu_bwd = func.call @relu_bwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %relu_bwd_0 = tensor.extract %relu_bwd[%i0] : tensor<3xf32>
  %relu_bwd_expected_0 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%relu_bwd_0, %relu_bwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %relu_bwd_1 = tensor.extract %relu_bwd[%i1] : tensor<3xf32>
  %relu_bwd_expected_1 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%relu_bwd_1, %relu_bwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %relu_bwd_2 = tensor.extract %relu_bwd[%i2] : tensor<3xf32>
  %relu_bwd_expected_2 = arith.constant 4.0000000000000000e+00 : f32
  func.call @close(%relu_bwd_2, %relu_bwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %sigmoid_fwd = func.call @sigmoid_fwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %sigmoid_fwd_0 = tensor.extract %sigmoid_fwd[%i0] : tensor<3xf32>
  %sigmoid_fwd_expected_0 = arith.constant 2.6894142136999510e-01 : f32
  func.call @close(%sigmoid_fwd_0, %sigmoid_fwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %sigmoid_fwd_1 = tensor.extract %sigmoid_fwd[%i1] : tensor<3xf32>
  %sigmoid_fwd_expected_1 = arith.constant 5.0000000000000000e-01 : f32
  func.call @close(%sigmoid_fwd_1, %sigmoid_fwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %sigmoid_fwd_2 = tensor.extract %sigmoid_fwd[%i2] : tensor<3xf32>
  %sigmoid_fwd_expected_2 = arith.constant 7.3105857863000490e-01 : f32
  func.call @close(%sigmoid_fwd_2, %sigmoid_fwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %sigmoid_bwd = func.call @sigmoid_bwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %sigmoid_bwd_0 = tensor.extract %sigmoid_bwd[%i0] : tensor<3xf32>
  %sigmoid_bwd_expected_0 = arith.constant 3.9322386648296370e-01 : f32
  func.call @close(%sigmoid_bwd_0, %sigmoid_bwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %sigmoid_bwd_1 = tensor.extract %sigmoid_bwd[%i1] : tensor<3xf32>
  %sigmoid_bwd_expected_1 = arith.constant 7.5000000000000000e-01 : f32
  func.call @close(%sigmoid_bwd_1, %sigmoid_bwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %sigmoid_bwd_2 = tensor.extract %sigmoid_bwd[%i2] : tensor<3xf32>
  %sigmoid_bwd_expected_2 = arith.constant 7.8644773296592740e-01 : f32
  func.call @close(%sigmoid_bwd_2, %sigmoid_bwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %tanh_fwd = func.call @tanh_fwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %tanh_fwd_0 = tensor.extract %tanh_fwd[%i0] : tensor<3xf32>
  %tanh_fwd_expected_0 = arith.constant -7.6159415595576485e-01 : f32
  func.call @close(%tanh_fwd_0, %tanh_fwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %tanh_fwd_1 = tensor.extract %tanh_fwd[%i1] : tensor<3xf32>
  %tanh_fwd_expected_1 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%tanh_fwd_1, %tanh_fwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %tanh_fwd_2 = tensor.extract %tanh_fwd[%i2] : tensor<3xf32>
  %tanh_fwd_expected_2 = arith.constant 7.6159415595576485e-01 : f32
  func.call @close(%tanh_fwd_2, %tanh_fwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %tanh_bwd = func.call @tanh_bwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %tanh_bwd_0 = tensor.extract %tanh_bwd[%i0] : tensor<3xf32>
  %tanh_bwd_expected_0 = arith.constant 8.3994868322805227e-01 : f32
  func.call @close(%tanh_bwd_0, %tanh_bwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %tanh_bwd_1 = tensor.extract %tanh_bwd[%i1] : tensor<3xf32>
  %tanh_bwd_expected_1 = arith.constant 3.0000000000000000e+00 : f32
  func.call @close(%tanh_bwd_1, %tanh_bwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %tanh_bwd_2 = tensor.extract %tanh_bwd[%i2] : tensor<3xf32>
  %tanh_bwd_expected_2 = arith.constant 1.6798973664561045e+00 : f32
  func.call @close(%tanh_bwd_2, %tanh_bwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %gelu_fwd = func.call @gelu_fwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %gelu_fwd_0 = tensor.extract %gelu_fwd[%i0] : tensor<3xf32>
  %gelu_fwd_expected_0 = arith.constant -1.5865525393145707e-01 : f32
  func.call @close(%gelu_fwd_0, %gelu_fwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %gelu_fwd_1 = tensor.extract %gelu_fwd[%i1] : tensor<3xf32>
  %gelu_fwd_expected_1 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%gelu_fwd_1, %gelu_fwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %gelu_fwd_2 = tensor.extract %gelu_fwd[%i2] : tensor<3xf32>
  %gelu_fwd_expected_2 = arith.constant 8.4134474606854293e-01 : f32
  func.call @close(%gelu_fwd_2, %gelu_fwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %gelu_bwd = func.call @gelu_bwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %gelu_bwd_0 = tensor.extract %gelu_bwd[%i0] : tensor<3xf32>
  %gelu_bwd_expected_0 = arith.constant -1.6663094117537258e-01 : f32
  func.call @close(%gelu_bwd_0, %gelu_bwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %gelu_bwd_1 = tensor.extract %gelu_bwd[%i1] : tensor<3xf32>
  %gelu_bwd_expected_1 = arith.constant 1.5000000000000000e+00 : f32
  func.call @close(%gelu_bwd_1, %gelu_bwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %gelu_bwd_2 = tensor.extract %gelu_bwd[%i2] : tensor<3xf32>
  %gelu_bwd_expected_2 = arith.constant 4.3332618823507456e+00 : f32
  func.call @close(%gelu_bwd_2, %gelu_bwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %gelu_approx_tanh_fwd = func.call @gelu_approx_tanh_fwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %gelu_approx_tanh_fwd_0 = tensor.extract %gelu_approx_tanh_fwd[%i0] : tensor<3xf32>
  %gelu_approx_tanh_fwd_expected_0 = arith.constant -1.5880800939172324e-01 : f32
  func.call @close(%gelu_approx_tanh_fwd_0, %gelu_approx_tanh_fwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %gelu_approx_tanh_fwd_1 = tensor.extract %gelu_approx_tanh_fwd[%i1] : tensor<3xf32>
  %gelu_approx_tanh_fwd_expected_1 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%gelu_approx_tanh_fwd_1, %gelu_approx_tanh_fwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %gelu_approx_tanh_fwd_2 = tensor.extract %gelu_approx_tanh_fwd[%i2] : tensor<3xf32>
  %gelu_approx_tanh_fwd_expected_2 = arith.constant 8.4119199060827676e-01 : f32
  func.call @close(%gelu_approx_tanh_fwd_2, %gelu_approx_tanh_fwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %gelu_approx_tanh_bwd = func.call @gelu_approx_tanh_bwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %gelu_approx_tanh_bwd_0 = tensor.extract %gelu_approx_tanh_bwd[%i0] : tensor<3xf32>
  %gelu_approx_tanh_bwd_expected_0 = arith.constant -1.6592816769156510e-01 : f32
  func.call @close(%gelu_approx_tanh_bwd_0, %gelu_approx_tanh_bwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %gelu_approx_tanh_bwd_1 = tensor.extract %gelu_approx_tanh_bwd[%i1] : tensor<3xf32>
  %gelu_approx_tanh_bwd_expected_1 = arith.constant 1.5000000000000000e+00 : f32
  func.call @close(%gelu_approx_tanh_bwd_1, %gelu_approx_tanh_bwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %gelu_approx_tanh_bwd_2 = tensor.extract %gelu_approx_tanh_bwd[%i2] : tensor<3xf32>
  %gelu_approx_tanh_bwd_expected_2 = arith.constant 4.3318563353831303e+00 : f32
  func.call @close(%gelu_approx_tanh_bwd_2, %gelu_approx_tanh_bwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %softplus_fwd = func.call @softplus_fwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %softplus_fwd_0 = tensor.extract %softplus_fwd[%i0] : tensor<3xf32>
  %softplus_fwd_expected_0 = arith.constant 6.3464005521486247e-02 : f32
  func.call @close(%softplus_fwd_0, %softplus_fwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %softplus_fwd_1 = tensor.extract %softplus_fwd[%i1] : tensor<3xf32>
  %softplus_fwd_expected_1 = arith.constant 3.4657359027997264e-01 : f32
  func.call @close(%softplus_fwd_1, %softplus_fwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %softplus_fwd_2 = tensor.extract %softplus_fwd[%i2] : tensor<3xf32>
  %softplus_fwd_expected_2 = arith.constant 1.0634640055214863e+00 : f32
  func.call @close(%softplus_fwd_2, %softplus_fwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %softplus_bwd = func.call @softplus_bwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %softplus_bwd_0 = tensor.extract %softplus_bwd[%i0] : tensor<3xf32>
  %softplus_bwd_expected_0 = arith.constant 2.3840584404423509e-01 : f32
  func.call @close(%softplus_bwd_0, %softplus_bwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %softplus_bwd_1 = tensor.extract %softplus_bwd[%i1] : tensor<3xf32>
  %softplus_bwd_expected_1 = arith.constant 1.5000000000000000e+00 : f32
  func.call @close(%softplus_bwd_1, %softplus_bwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %softplus_bwd_2 = tensor.extract %softplus_bwd[%i2] : tensor<3xf32>
  %softplus_bwd_expected_2 = arith.constant 3.5231883119115293e+00 : f32
  func.call @close(%softplus_bwd_2, %softplus_bwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %swish_fwd = func.call @swish_fwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %swish_fwd_0 = tensor.extract %swish_fwd[%i0] : tensor<3xf32>
  %swish_fwd_expected_0 = arith.constant -1.1920292202211755e-01 : f32
  func.call @close(%swish_fwd_0, %swish_fwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %swish_fwd_1 = tensor.extract %swish_fwd[%i1] : tensor<3xf32>
  %swish_fwd_expected_1 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%swish_fwd_1, %swish_fwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %swish_fwd_2 = tensor.extract %swish_fwd[%i2] : tensor<3xf32>
  %swish_fwd_expected_2 = arith.constant 8.8079707797788231e-01 : f32
  func.call @close(%swish_fwd_2, %swish_fwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %swish_bwd = func.call @swish_bwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %swish_bwd_0 = tensor.extract %swish_bwd[%i0] : tensor<3xf32>
  %swish_bwd_expected_0 = arith.constant -1.8156849756979093e-01 : f32
  func.call @close(%swish_bwd_0, %swish_bwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %swish_bwd_1 = tensor.extract %swish_bwd[%i1] : tensor<3xf32>
  %swish_bwd_expected_1 = arith.constant 1.5000000000000000e+00 : f32
  func.call @close(%swish_bwd_1, %swish_bwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %swish_bwd_2 = tensor.extract %swish_bwd[%i2] : tensor<3xf32>
  %swish_bwd_expected_2 = arith.constant 4.3631369951395822e+00 : f32
  func.call @close(%swish_bwd_2, %swish_bwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %elu_fwd = func.call @elu_fwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %elu_fwd_0 = tensor.extract %elu_fwd[%i0] : tensor<3xf32>
  %elu_fwd_expected_0 = arith.constant -1.2642411176571153e+00 : f32
  func.call @close(%elu_fwd_0, %elu_fwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %elu_fwd_1 = tensor.extract %elu_fwd[%i1] : tensor<3xf32>
  %elu_fwd_expected_1 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%elu_fwd_1, %elu_fwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %elu_fwd_2 = tensor.extract %elu_fwd[%i2] : tensor<3xf32>
  %elu_fwd_expected_2 = arith.constant 1.0000000000000000e+00 : f32
  func.call @close(%elu_fwd_2, %elu_fwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %elu_bwd = func.call @elu_bwd(%x, %g) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %elu_bwd_0 = tensor.extract %elu_bwd[%i0] : tensor<3xf32>
  %elu_bwd_expected_0 = arith.constant 1.4715177646857693e+00 : f32
  func.call @close(%elu_bwd_0, %elu_bwd_expected_0, %tol) : (f32, f32, f32) -> ()
  %elu_bwd_1 = tensor.extract %elu_bwd[%i1] : tensor<3xf32>
  %elu_bwd_expected_1 = arith.constant 6.0000000000000000e+00 : f32
  func.call @close(%elu_bwd_1, %elu_bwd_expected_1, %tol) : (f32, f32, f32) -> ()
  %elu_bwd_2 = tensor.extract %elu_bwd[%i2] : tensor<3xf32>
  %elu_bwd_expected_2 = arith.constant 4.0000000000000000e+00 : f32
  func.call @close(%elu_bwd_2, %elu_bwd_expected_2, %tol) : (f32, f32, f32) -> ()
  %input = arith.constant dense<[[-2.0, 0.0, 3.0], [4.0, -5.0, 0.0]]> : tensor<2x3xf32>
  %reduce_add = func.call @reduce_add(%input) : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %reduce_add_0 = tensor.extract %reduce_add[%i0, %i0] : tensor<2x1xf32>
  %reduce_add_expected_0 = arith.constant 1.0000000000000000e+00 : f32
  func.call @close(%reduce_add_0, %reduce_add_expected_0, %tol) : (f32, f32, f32) -> ()
  %reduce_add_1 = tensor.extract %reduce_add[%i1, %i0] : tensor<2x1xf32>
  %reduce_add_expected_1 = arith.constant -1.0000000000000000e+00 : f32
  func.call @close(%reduce_add_1, %reduce_add_expected_1, %tol) : (f32, f32, f32) -> ()
  %reduce_mul = func.call @reduce_mul(%input) : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %reduce_mul_0 = tensor.extract %reduce_mul[%i0, %i0] : tensor<2x1xf32>
  %reduce_mul_expected_0 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%reduce_mul_0, %reduce_mul_expected_0, %tol) : (f32, f32, f32) -> ()
  %reduce_mul_1 = tensor.extract %reduce_mul[%i1, %i0] : tensor<2x1xf32>
  %reduce_mul_expected_1 = arith.constant 0.0000000000000000e+00 : f32
  func.call @close(%reduce_mul_1, %reduce_mul_expected_1, %tol) : (f32, f32, f32) -> ()
  %reduce_min = func.call @reduce_min(%input) : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %reduce_min_0 = tensor.extract %reduce_min[%i0, %i0] : tensor<2x1xf32>
  %reduce_min_expected_0 = arith.constant -2.0000000000000000e+00 : f32
  func.call @close(%reduce_min_0, %reduce_min_expected_0, %tol) : (f32, f32, f32) -> ()
  %reduce_min_1 = tensor.extract %reduce_min[%i1, %i0] : tensor<2x1xf32>
  %reduce_min_expected_1 = arith.constant -5.0000000000000000e+00 : f32
  func.call @close(%reduce_min_1, %reduce_min_expected_1, %tol) : (f32, f32, f32) -> ()
  %reduce_max = func.call @reduce_max(%input) : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %reduce_max_0 = tensor.extract %reduce_max[%i0, %i0] : tensor<2x1xf32>
  %reduce_max_expected_0 = arith.constant 3.0000000000000000e+00 : f32
  func.call @close(%reduce_max_0, %reduce_max_expected_0, %tol) : (f32, f32, f32) -> ()
  %reduce_max_1 = tensor.extract %reduce_max[%i1, %i0] : tensor<2x1xf32>
  %reduce_max_expected_1 = arith.constant 4.0000000000000000e+00 : f32
  func.call @close(%reduce_max_1, %reduce_max_expected_1, %tol) : (f32, f32, f32) -> ()
  %reduce_amax = func.call @reduce_amax(%input) : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %reduce_amax_0 = tensor.extract %reduce_amax[%i0, %i0] : tensor<2x1xf32>
  %reduce_amax_expected_0 = arith.constant 3.0000000000000000e+00 : f32
  func.call @close(%reduce_amax_0, %reduce_amax_expected_0, %tol) : (f32, f32, f32) -> ()
  %reduce_amax_1 = tensor.extract %reduce_amax[%i1, %i0] : tensor<2x1xf32>
  %reduce_amax_expected_1 = arith.constant 5.0000000000000000e+00 : f32
  func.call @close(%reduce_amax_1, %reduce_amax_expected_1, %tol) : (f32, f32, f32) -> ()
  %reduce_avg = func.call @reduce_avg(%input) : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %reduce_avg_0 = tensor.extract %reduce_avg[%i0, %i0] : tensor<2x1xf32>
  %reduce_avg_expected_0 = arith.constant 3.3333333333333331e-01 : f32
  func.call @close(%reduce_avg_0, %reduce_avg_expected_0, %tol) : (f32, f32, f32) -> ()
  %reduce_avg_1 = tensor.extract %reduce_avg[%i1, %i0] : tensor<2x1xf32>
  %reduce_avg_expected_1 = arith.constant -3.3333333333333331e-01 : f32
  func.call @close(%reduce_avg_1, %reduce_avg_expected_1, %tol) : (f32, f32, f32) -> ()
  %reduce_norm1 = func.call @reduce_norm1(%input) : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %reduce_norm1_0 = tensor.extract %reduce_norm1[%i0, %i0] : tensor<2x1xf32>
  %reduce_norm1_expected_0 = arith.constant 5.0000000000000000e+00 : f32
  func.call @close(%reduce_norm1_0, %reduce_norm1_expected_0, %tol) : (f32, f32, f32) -> ()
  %reduce_norm1_1 = tensor.extract %reduce_norm1[%i1, %i0] : tensor<2x1xf32>
  %reduce_norm1_expected_1 = arith.constant 9.0000000000000000e+00 : f32
  func.call @close(%reduce_norm1_1, %reduce_norm1_expected_1, %tol) : (f32, f32, f32) -> ()
  %reduce_norm2 = func.call @reduce_norm2(%input) : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %reduce_norm2_0 = tensor.extract %reduce_norm2[%i0, %i0] : tensor<2x1xf32>
  %reduce_norm2_expected_0 = arith.constant 3.6055512754639891e+00 : f32
  func.call @close(%reduce_norm2_0, %reduce_norm2_expected_0, %tol) : (f32, f32, f32) -> ()
  %reduce_norm2_1 = tensor.extract %reduce_norm2[%i1, %i0] : tensor<2x1xf32>
  %reduce_norm2_expected_1 = arith.constant 6.4031242374328485e+00 : f32
  func.call @close(%reduce_norm2_1, %reduce_norm2_expected_1, %tol) : (f32, f32, f32) -> ()
  %reduce_mul_no_zeros = func.call @reduce_mul_no_zeros(%input) : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %reduce_mul_no_zeros_0 = tensor.extract %reduce_mul_no_zeros[%i0, %i0] : tensor<2x1xf32>
  %reduce_mul_no_zeros_expected_0 = arith.constant -6.0000000000000000e+00 : f32
  func.call @close(%reduce_mul_no_zeros_0, %reduce_mul_no_zeros_expected_0, %tol) : (f32, f32, f32) -> ()
  %reduce_mul_no_zeros_1 = tensor.extract %reduce_mul_no_zeros[%i1, %i0] : tensor<2x1xf32>
  %reduce_mul_no_zeros_expected_1 = arith.constant -2.0000000000000000e+01 : f32
  func.call @close(%reduce_mul_no_zeros_1, %reduce_mul_no_zeros_expected_1, %tol) : (f32, f32, f32) -> ()
  return
}
