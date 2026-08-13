// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Bindings/Python/NanobindAdaptors.h" // IWYU pragma: keep

#include "tensor_ir-c/TensorIR.h"

/// MLIR's `_mlir_libs/__init__.py` imports every `_site_initialize_<N>` module
/// it finds and calls `register_dialects` on the registry of each newly created
/// `ir.Context`. That is how the TensorIR dialect becomes available without an
/// explicit `register_dialect()` call.
NB_MODULE(_site_initialize_0, m) {
  m.doc() = "TensorIR dialect registration hook for the nv_tensor_ir package";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirTensorIRRegisterAllDialects(registry);
  });
}
