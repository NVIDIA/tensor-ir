# TensorIR Compiler

## Early Release Status

TensorIR is being released early so the community can evaluate the project,
provide feedback, and engage with the API and development direction.

This release is not yet intended as a performance benchmark or
production-performance commitment. Performance may vary across workloads,
configurations, hardware, and compiler versions as we continue to mature code
generation, heuristics, autotuning, validation, and integration.

We welcome issue reports, feedback on usability and supported workflows, and
contributions that help improve the project.

## Building TensorIR Compiler

### Prerequisites

Required:
- CMake 3.20.0 or later
- C++17 compatible compiler
- CUDA Toolkit 13.3 or later and a compatible NVIDIA driver
- Python 3.10 or later
- Ninja build system

CUDA Toolkit 13.1 can be used for kernels supported by the compatibility
bytecode format by passing `--bytecode-version=compatibility`; the default
compiler and test workflow requires CUDA Toolkit 13.3 or later.

Python bindings (enabled by default):
- Python development headers
- nanobind 2.9 or later

Pass `-DTENSOR_IR_ENABLE_BINDINGS_PYTHON=OFF` to build only the compiler
without these Python binding dependencies.

Testing/examples:
- pytest and PyTorch, validated with pytest 8.3.4 and PyTorch 2.10

### Quick Start

TensorIR Compiler depends on [CUDA Tile IR](https://github.com/NVIDIA/cuda-tile)
for GPU code generation. LLVM and MLIR are located through CMake config packages
using `find_package(MLIR CONFIG)` and `find_package(LLVM CONFIG)`. When
`TENSOR_IR_DOWNLOAD_LLVM=OFF`, set `MLIR_DIR` to the directory containing
`MLIRConfig.cmake`, or override package discovery using another method supported
by CMake's
[`find_package` command](https://cmake.org/cmake/help/latest/command/find_package.html).

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tensor_ir-compiler tensor_ir-opt tensor_ir_python_bindings --parallel 32
```

### Build Options

| Option | Description |
|--------|-------------|
| `-DTENSOR_IR_CUDA_TILE_SOURCE_URL=<url>` | Override the pinned CUDA Tile source archive |
| `-DTENSOR_IR_DOWNLOAD_LLVM=ON/OFF` | Download the pinned LLVM source (default: `ON` for top-level builds) |
| `-DTENSOR_IR_LLVM_SOURCE_URL=<url>` | LLVM source archive URL; accepts a cached `file://` URL |
| `-DMLIR_DIR=<path>` | MLIR package directory used when LLVM downloading is disabled |
| `-DTENSOR_IR_ENABLE_BINDINGS_PYTHON=ON/OFF` | Build the TensorIR Python bindings (default: `ON`) |
| `-DTENSOR_IR_INCLUDE_TESTS=ON/OFF` | Build test-only passes and configure the lit suite (default: `ON`) |
| `-DTENSOR_IR_CUDART_LINKAGE=DYNAMIC/STATIC` | Dynamically load cudart or link `CUDA::cudart_static` (default: `DYNAMIC`) |
| `-DCMAKE_BUILD_TYPE=Release` | Build type (Release/Debug/RelWithDebInfo) |

To use an existing compatible LLVM installation:

```bash
cmake -S . -B build -G Ninja \
  -DTENSOR_IR_DOWNLOAD_LLVM=OFF \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
```

The build produces:

- `build/bin/tensor_ir-compiler`
- `build/bin/tensor_ir-opt`
- `build/python_packages/nv_tensor_ir` (when Python bindings are enabled)

To use the Python bindings or DSL directly from the build tree, run from the
repository root with:

```bash
export PYTHONPATH="$PWD/build/python_packages:$PYTHONPATH"
```

The Python package contains both the lower-level MLIR bindings under
`nv_tensor_ir._mlir` and the DSL under `nv_tensor_ir.dsl`.

`cmake --install build` installs:

- `bin/tensor_ir-compiler`
- `bin/tensor_ir-opt`
- `python_packages/nv_tensor_ir` (when Python bindings are enabled)

Use `cmake --install build --prefix <install-prefix>` to install into a
user-writable directory. Add `<install-prefix>/python_packages` to
`PYTHONPATH` to import the installed Python package.

## Usage

The compiler is compile-only by default: it does not build a reference graph or
launch the kernel. `--launch` and `--verify` require an NVIDIA GPU compatible
with the selected target architecture and a compatible driver; `--verify`
launches the kernel and compares its results with the reference implementation.

### Compile MLIR to GPU Kernel

```bash
build/bin/tensor_ir-compiler \
  test/Integration/Compiler/matmul_8x8x8.mlir \
  --verbose --print-ir-after-all
```

### Compile Dynamic-Shape MLIR

```bash
build/bin/tensor_ir-compiler \
  test/Integration/Compiler/add_dynamic.mlir \
  --dynamic-dims=16,8 --dynamic-strides=8 --tile-size=8x8 \
  --verify
```

`--dynamic-dims` and `--dynamic-strides` provide runtime values for `?`
dimensions and strides by tensor dimension position, and the same values are
reused across tensors. When a dynamic position exceeds the supplied list, the
last value is reused. If `--dynamic-strides` is omitted, dynamic strides are
inferred as packed strides. `--tile-size` controls compiler tiling; it is not a
runtime problem-size option.

### Command-Line Options

The options below cover the common compile and run workflow. For the complete
option list and defaults, run `build/bin/tensor_ir-compiler --help`.

**General:**
- `--verbose` - Enable verbose output
- `--dump-artifact=<path>` - Write compiled CUDA Tile device code to `<path>`
  and metadata to `<path>.meta`

**Compilation:**
- `--dump-ir=<path>` - Dump lowered CUDA Tile dialect MLIR
- `--dump-tileir-bc=<path>` - Dump Tile IR bytecode after compilation
- `--bytecode-version=<version>` - Tile IR bytecode target: `default`,
  `current`, or `compatibility`. The default target is
  `max(compatibility, 13.3)`.
- `--codegen-strategy=<strategy>` - TensorIR-to-CUDA-Tile lowering path:
  `layout-propagation` (default) or `affine-map`
- `--target-sm=<sm_XX[a|f]>` - Target GPU architecture (for example,
  `sm_100`, `sm_100a`, or `sm_100f`); defaults to the SM100 family target
  `sm_100f`
- `--tile-size=<MxN>` or `<MxNxK>` - Tile sizes for kernel tiling
- `--print-ir-after-all` - Print IR after each pass; ignored when
  `--print-ir-tree-dir` is set
- `--print-ir-tree-dir=<dir>` - Dump per-pass IR snapshots to a directory;
  takes precedence over `--print-ir-after-all`
- `--timing` - Show pass timing information
- `--uniform-signature` - Pass all sizes and strides as kernel arguments,
  including static sizes and strides

The IR debug options above also support environment variables that override CLI
values: `TENSOR_IR_DUMP_IR`, `TENSOR_IR_DUMP_TILEIR_BC`,
`TENSOR_IR_LOAD_TILEIR_BC`, `TENSOR_IR_PRINT_IR_AFTER_ALL`,
`TENSOR_IR_PRINT_IR_TREE_DIR`, and `TENSOR_IR_TIMING`. Boolean env vars accept
`1/0`, `true/false`, `on/off`, and `yes/no`; path env vars require a path, and
`0/false/off/no` disables them.

**Execution:**
- With no execution option, compile without building or running a reference graph
- `--launch` - Execute the compiled kernel on GPU
- `--verify` - Launch and verify results against the reference implementation
- `--iterations=<N>` - Number of execution iterations for benchmarking
- `--seed=<N>` - Random seed for test data
- `--tolerance=<value>` - Numerical tolerance for verification

**Dynamic shapes:**
- `--dynamic-dims=<d0,d1,...>` - Runtime values for `?` dimensions
- `--dynamic-strides=<s0,s1,...>` - Runtime values for `?` strides

### Example MLIR Inputs

Static-shape matmul:

```mlir
module {
  nv_tensor_ir.graph @matmul_f32_static(
    %a: tensor<8x8xf32>,
    %b: tensor<8x8xf32>) -> (
      tensor<8x8xf32>) {
    %c = "nv_tensor_ir.matmul"(%a, %b) :
      (tensor<8x8xf32>,
       tensor<8x8xf32>) ->
      tensor<8x8xf32>
    results %c : tensor<8x8xf32>
  }
}
```

Dynamic-shape pointwise add:

```mlir
module {
  nv_tensor_ir.graph @add_dynamic_shape(
    %a: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"},
    %b: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"}
  ) -> (tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"}) {
    %add = add %a, %b : tensor<?x?xf32>
    results %add : tensor<?x?xf32>
  }
}
```

More runnable MLIR examples are available under
`test/Integration/Compiler/`.

### Python Bindings

The OSS build packages TensorIR MLIR Python bindings under
`build/python_packages/nv_tensor_ir`. These bindings can register the TensorIR
dialect, build or parse TensorIR MLIR modules, and compile an MLIR module
operation through the OSS TensorIR extension. Runtime launch accepts
DLPack-compatible tensors; the example below uses PyTorch only as one tensor
provider.

At launch, callers must provide CUDA-accessible DLPack tensors that match the
compiled argument shapes, data types, and layouts. The runtime does not validate
every mismatch before launching the kernel.

```python
import torch

from nv_tensor_ir._mlir import ir
from nv_tensor_ir._mlir.dialects import nv_tensor_ir

mlir_text = """
module {
  nv_tensor_ir.graph @add_op(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %out = add %a, %b : tensor<8x8xf32>
    results %out : tensor<8x8xf32>
  }
}
"""

with ir.Context() as ctx, ir.Location.unknown(ctx):
    nv_tensor_ir.register_dialect(ctx, load=True)
    module = ir.Module.parse(mlir_text)

    assert nv_tensor_ir.can_compile(module, tile_sizes=[8])
    with nv_tensor_ir.compile(module, tile_sizes=[8]) as program:
        assert program.get_bytecode()

        a = torch.randn((8, 8), device="cuda", dtype=torch.float32)
        b = torch.randn((8, 8), device="cuda", dtype=torch.float32)
        out = torch.empty_like(a)

        program.launch(a, b, out)
```

By default, TensorIR targets `max(compatibility, 13.3)`. To request the
compatibility bytecode target explicitly, pass a `CompileOptions` object:

```python
options = nv_tensor_ir.CompileOptions()
options.tile_sizes = [8]
options.bytecode_version = nv_tensor_ir.BytecodeVersion.compatibility()
program = nv_tensor_ir.compile(module, options=options)
```

### Python DSL

The OSS build also packages a lightweight Python DSL under `nv_tensor_ir.dsl`.
The DSL traces a Python kernel into TensorIR MLIR, compiles it through the same
OSS compiler path, and runs it with DLPack-compatible tensors. Use
`tir.TensorSpec` with `tir.DataType` to compile from metadata without
allocating framework tensors.

```python
import torch

from nv_tensor_ir import dsl as tir


USE_GELU = True


@tir.kernel
def fused_gemm_epilogue(a, b, bias, residual):
    x = a @ b + bias
    if USE_GELU:  # Static control flow, resolved while tracing.
        x = tir.gelu_approx_tanh(x)
    return x + residual


a = torch.randn((128, 64), device="cuda")
b = torch.randn((64, 128), device="cuda")
bias = torch.randn((128, 128), device="cuda")
residual = torch.randn_like(bias)
output = torch.empty_like(bias)

compiled = tir.compile(
    fused_gemm_epilogue,
    a,
    b,
    bias,
    residual,
    output=output,
    tile_sizes=(64, 64),
)
compiled.run(a, b, bias, residual, output=output)
```

## Testing

After configuring the generated OSS tree, run the compiler sample MLIR, Python
binding, and DSL smoke tests with:

```bash
cmake --build build --target check-tensor-ir
```

The target sets the test `PYTHONPATH` and points the tests at the build-tree
`tensor_ir-compiler`. The smoke tests launch kernels and require a compatible
NVIDIA GPU and driver.

## License

TensorIR-owned source files are distributed under
`Apache-2.0 WITH LLVM-exception`, as identified by the SPDX headers in each
file. CUDA Tile is fetched from its upstream repository and distributed under
its own license.
