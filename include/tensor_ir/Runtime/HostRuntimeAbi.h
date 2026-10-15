// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_RUNTIME_HOST_RUNTIME_ABI_H_
#define TENSOR_IR_RUNTIME_HOST_RUNTIME_ABI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(modernize-use-using)

#define TENSOR_IR_HOST_ABI_VERSION_1 1

/// Launch configuration passed from compiled host code to the runtime callback.
typedef struct TensorIRLaunchConfigV1 {
  uint32_t grid[3];
  uint32_t block[3];
  uint32_t dynamicSharedMemoryBytes;
} TensorIRLaunchConfigV1;

/// Callback invoked by compiled host code to launch a device kernel.
///
/// Returns 0 on success, or a non-zero status code on failure.
/// Callbacks must not unwind through compiled host code.
typedef int32_t (*TensorIRLaunchFnV1)(void *userData, uint32_t kernelOrdinal,
                                      const TensorIRLaunchConfigV1 *config,
                                      void *const *kernelParams);

/// Host runtime context and dispatch table passed into the native entry point.
typedef struct TensorIRHostRuntimeV1 {
  uint32_t structSize;       // sizeof(TensorIRHostRuntimeV1)
  uint32_t abiVersion;       // TENSOR_IR_HOST_ABI_VERSION_1
  void *userData;            // Opaque pointer to runtime execution context
  TensorIRLaunchFnV1 launch; // Runtime callback for kernel launching
} TensorIRHostRuntimeV1;

/// Function pointer signature of the exported host entry function.
///
/// packedArgs uses a fixed layout in graph argument order. Each tensor occupies
/// 1 + 2 * rank int64_t slots: data pointer bits, every size, then every stride
/// (in elements). Static sizes and strides must also be provided and must agree
/// with the compiled tensor type. Each scalar occupies one slot containing its
/// value bits in the low bytes. The device launch ABI is private and specialized
/// by the generated host code; it does not change this public layout.
///
/// Returns 0 on success; -1 for ABI version/size mismatch; -2 for packed
/// argument count/layout mismatch or unrepresentable device size/stride;
/// -3 for callback failure; -4 for an invalid kernel ordinal; -5 for a negative
/// or unrepresentable launch configuration. Other nonzero values indicate
/// execution failure.
/// Neither the runtime context nor argument pointers may be retained after
/// returning. `runtime` and `runtime->launch` are required. `packedArgs` is
/// required when `packedArgCount` is nonzero.
typedef int32_t (*TensorIRHostEntryV1)(const TensorIRHostRuntimeV1 *runtime,
                                       const int64_t *packedArgs,
                                       uint32_t packedArgCount);

// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
}
#endif

#endif // TENSOR_IR_RUNTIME_HOST_RUNTIME_ABI_H_
