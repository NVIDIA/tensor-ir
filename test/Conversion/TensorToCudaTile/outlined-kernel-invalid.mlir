// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation" --verify-diagnostics -split-input-file

module {
  gpu.module @kernels {
  func.func @bad_axis() attributes {gpu.kernel} {
    // expected-error @+2 {{staged TensorIR kernels require the x grid axis}}
    // expected-error @+1 {{failed to legalize operation 'gpu.block_id'}}
    %unused = gpu.block_id y
    return
  }
  }
}

// -----

module {
  // expected-error @below {{outlined TensorIR kernels must be nested in a gpu.module}}
  func.func @top_level() attributes {gpu.kernel} {
    return
  }
}

// -----

module {
  gpu.module @kernels {
    // expected-error @below {{outlined TensorIR gpu.module may contain only gpu.kernel func.func entry points}}
    func.func @helper() {
      return
    }
    func.func @entry() attributes {gpu.kernel} {
      return
    }
  }
}

// -----

module {
  gpu.module @kernels0 {
    func.func @duplicate() attributes {gpu.kernel} {
      return
    }
  }
  gpu.module @kernels1 {
    // expected-error @below {{duplicate outlined kernel name @duplicate}}
    func.func @duplicate() attributes {gpu.kernel} {
      return
    }
  }
}

// -----

module {
  gpu.module @kernels {
  // expected-error @+2 {{outlined TensorIR kernels must return void}}
  // expected-error @+1 {{failed to legalize operation 'func.func'}}
  func.func @returns_value() -> i32 attributes {gpu.kernel} {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
  }
}

// -----

module {
  gpu.module @kernels {
  func.func @dynamic_dim(%arg: memref<?xf32>, %dim: index) attributes {gpu.kernel} {
    // expected-error @+2 {{staged kernels require a constant memref.dim index}}
    // expected-error @+1 {{failed to legalize operation 'memref.dim'}}
    %unused = memref.dim %arg, %dim : memref<?xf32>
    return
  }
  }
}

// -----

module {
  gpu.module @kernels {
  func.func @unlowered(
      %arg: memref<4xf32, #ptr.generic_space>) attributes {gpu.kernel} {
    %zero = arith.constant 0 : index
    %ptr = ptr.to_ptr %arg : memref<4xf32, #ptr.generic_space>
        -> !ptr.ptr<#ptr.generic_space>
    %tile = nv_tensor_ir.load %ptr[%zero]
        view offset: [0], sizes: [4], strides: [1], alignment: 4
        : !ptr.ptr<#ptr.generic_space> to tensor<4xf32>
    // expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.gelu_bwd'}}
    %unused = "nv_tensor_ir.gelu_bwd"(%tile, %tile) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return
  }
  }
}

// -----

module {
  gpu.module @kernels {
  func.func @strided_slice() attributes {gpu.kernel} {
    %input = nv_tensor_ir.constant dense<1.0> : tensor<8xf32>
    // expected-error @+2 {{staged tile slice requires unit strides}}
    // expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.slice'}}
    %slice = nv_tensor_ir.slice %input starts = [0] limits = [8] strides = [2]
        : tensor<8xf32> -> tensor<4xf32>
    return
  }
  }
}

// -----

module {
  gpu.module @kernels {
  // expected-error @+1 {{outlined TensorIR kernel arguments must be scalars or strided memrefs, but got 'tensor<4xf32>'}}
  func.func @tensor_argument(%arg: tensor<4xf32>) attributes {gpu.kernel} {
    return
  }
  }
}

// -----

module {
  gpu.module @kernels {
  func.func @unsupported_matmul_types() attributes {gpu.kernel} {
    %lhs = nv_tensor_ir.constant dense<1.0> : tensor<16x32xf32>
    %rhs = nv_tensor_ir.constant dense<2.0> : tensor<32x8xf32>
    // expected-error @+2 {{unsupported input/output type combination}}
    // expected-error @+1 {{failed to legalize operation 'nv_tensor_ir.matmul'}}
    %unused = nv_tensor_ir.matmul(%lhs, %rhs)
        : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf16>
    return
  }
  }
}

// -----

// expected-error @+1 {{layout-propagation conversion requires the outlined gpu.kernel form; run tir-bufferize, tir-form-grid, tir-tile-reductions, and outline-tensor-ir-kernel first}}
module {
  nv_tensor_ir.graph @unformed(%arg: tensor<4xf32>) -> tensor<4xf32> {
    results %arg : tensor<4xf32>
  }
}

// -----

module {
  // expected-error @+1 {{layout-propagation conversion does not accept analysis attribute 'tile_size'; run TensorIR tiled-program formation first}}
  func.func @analysis_attribute() attributes {
      gpu.kernel, tile_size = array<i32: 4>} {
    return
  }
}

// -----

// expected-error @+1 {{layout-propagation conversion requires at least one outlined gpu.kernel function nested in a gpu.module}}
module {
}

// -----

module {
  gpu.module @kernels {
    func.func @valid_kernel() attributes {gpu.kernel} {
      return
    }
    // expected-error @below {{outlined TensorIR gpu.module may contain only gpu.kernel func.func entry points}}
    func.func @unexpected_helper() {
      return
    }
  }
}
