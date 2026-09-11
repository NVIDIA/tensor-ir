// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Thin driver for the self-registering TensorIR TableGen backends.
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"
int main(int argc, char **argv) { return mlir::MlirTblgenMain(argc, argv); }
