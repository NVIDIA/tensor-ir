# TensorIR Python package

This directory builds the `nv_tensor_ir` distribution. `src/nv_tensor_ir`
mirrors the installed package one-for-one, so moving a file within that tree is
an API change; build-only organization belongs in `bindings/` or the CMake
files. `src/CMakeLists.txt` declares pure and generated Python sources,
`bindings/CMakeLists.txt` declares native nanobind extensions, and the
top-level `CMakeLists.txt` assembles both with the vendored MLIR sources and
their shared C API library.

The package installs a site initializer that registers the TensorIR dialect
with every newly created MLIR context.
