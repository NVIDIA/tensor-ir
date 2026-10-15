// RUN: tensor_ir-opt %s --split-input-file --verify-diagnostics --tir-host-to-llvm

module {
  func.func @unsupported(%x: f32) -> f32 {
    // expected-error @below {{failed to legalize operation 'math.exp'}}
    %result = math.exp %x : f32
    return %result : f32
  }
}

// -----

module {
  llvm.func @unresolved(%x: i32) -> f32 {
    // expected-error @below {{unresolved cast in native host lowering}}
    %result = builtin.unrealized_conversion_cast %x : i32 to f32
    llvm.return %result : f32
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @unmarked_runtime(%runtime: !llvm.ptr) -> i32 {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // expected-error @below {{expected host function with runtime pointer argument}}
    gpu.launch_func @kernels::@copy
        blocks in (%one, %one, %one) threads in (%one, %one, %one)
    return %zero : i32
  }
  gpu.module @kernels {
    func.func @copy() attributes {gpu.kernel} { return }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @missing_runtime() -> i32 {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // expected-error @below {{expected host function with runtime pointer argument}}
    gpu.launch_func @kernels::@copy
        blocks in (%one, %one, %one) threads in (%one, %one, %one) {nv_tensor_ir.kernel_ordinal = 0 : i32}
    return %zero : i32
  }
  gpu.module @kernels {
    func.func @copy() attributes {gpu.kernel} { return }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @wrong_result(%runtime: !llvm.ptr {nv_tensor_ir.runtime_context})  {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // expected-error @below {{expected host function with runtime pointer argument}}
    gpu.launch_func @kernels::@copy
        blocks in (%one, %one, %one) threads in (%one, %one, %one) {nv_tensor_ir.kernel_ordinal = 0 : i32}
    return
  }
  gpu.module @kernels {
    func.func @copy() attributes {gpu.kernel} { return }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @async_launch(%runtime: !llvm.ptr {nv_tensor_ir.runtime_context}) -> i32 {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // expected-error @below {{native host execution requires synchronous launches}}
    %token = gpu.launch_func async [] @kernels::@copy
        blocks in (%one, %one, %one) threads in (%one, %one, %one) {nv_tensor_ir.kernel_ordinal = 0 : i32}
    return %zero : i32
  }
  gpu.module @kernels {
    func.func @copy() attributes {gpu.kernel} { return }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @cluster_launch(%runtime: !llvm.ptr {nv_tensor_ir.runtime_context}) -> i32 {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // expected-error @below {{native host ABI v1 does not support cluster dimensions}}
    gpu.launch_func @kernels::@copy clusters in (%one, %one, %one)
        blocks in (%one, %one, %one) threads in (%one, %one, %one) {nv_tensor_ir.kernel_ordinal = 0 : i32}
    return %zero : i32
  }
  gpu.module @kernels {
    func.func @copy() attributes {gpu.kernel} { return }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @async_dependency(%runtime: !llvm.ptr {nv_tensor_ir.runtime_context}, %dependency: !gpu.async.token) -> i32 {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // expected-error @below {{native host execution requires synchronous launches}}
    %token = gpu.launch_func async [%dependency] @kernels::@copy
        blocks in (%one, %one, %one) threads in (%one, %one, %one)
        {nv_tensor_ir.kernel_ordinal = 0 : i32}
    return %zero : i32
  }
  gpu.module @kernels {
    func.func @copy() attributes {gpu.kernel} { return }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @async_object(%runtime: !llvm.ptr {nv_tensor_ir.runtime_context}, %stream: !llvm.ptr) -> i32 {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // expected-error @below {{native host execution requires synchronous launches}}
    gpu.launch_func <%stream : !llvm.ptr> @kernels::@copy
        blocks in (%one, %one, %one) threads in (%one, %one, %one)
        {nv_tensor_ir.kernel_ordinal = 0 : i32}
    return %zero : i32
  }
  gpu.module @kernels {
    func.func @copy() attributes {gpu.kernel} { return }
  }
}
