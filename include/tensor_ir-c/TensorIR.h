// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_C_TENSOR_IR_H
#define TENSOR_IR_C_TENSOR_IR_H

#include "mlir-c/IR.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(modernize-use-using)

//===----------------------------------------------------------------------===//
// Dialect and pass registration.
//===----------------------------------------------------------------------===//

/// Declares the C API registration hooks for the TensorIR dialect.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(nv_tensor_ir, nv_tensor_ir);

/// Adds all dialects used by TensorIR to the registry.
MLIR_CAPI_EXPORTED void
mlirTensorIRRegisterAllDialects(MlirDialectRegistry registry);

//===----------------------------------------------------------------------===//
// Bytecode version API.
//===----------------------------------------------------------------------===//

/// Identifies a supported CUDA Tile bytecode format version.
typedef struct MlirTensorIRBytecodeVersion {
  uint8_t major;
  uint8_t minor;
  uint16_t tag;
} MlirTensorIRBytecodeVersion;

/// Returns the most recent CUDA Tile bytecode version supported by this build.
MLIR_CAPI_EXPORTED MlirTensorIRBytecodeVersion
mlirTensorIRGetCurrentBytecodeVersion(void);

/// Returns the CUDA Tile bytecode version with the widest compatibility range.
MLIR_CAPI_EXPORTED MlirTensorIRBytecodeVersion
mlirTensorIRGetCompatibilityBytecodeVersion(void);

/// Returns the default bytecode target selected for TensorIR compilation.
MLIR_CAPI_EXPORTED MlirTensorIRBytecodeVersion
mlirTensorIRGetDefaultBytecodeVersion(void);

/// Formats `version` as `major.minor` or `major.minor.tag` and forwards the
/// resulting string and `userData` to `callback`.
///
/// The string reference passed to `callback` is valid only for the duration of
/// the callback. If `callback` is NULL or `version` is unsupported, the
/// function returns without invoking the callback.
MLIR_CAPI_EXPORTED void
mlirTensorIRFormatBytecodeVersion(MlirTensorIRBytecodeVersion version,
                                  MlirStringCallback callback, void *userData);

/// Returns true if `lhs` and `rhs` have identical version components.
MLIR_CAPI_EXPORTED bool
mlirTensorIRBytecodeVersionEqual(MlirTensorIRBytecodeVersion lhs,
                                 MlirTensorIRBytecodeVersion rhs);

/// Returns a hash of `version` that is consistent with
/// `mlirTensorIRBytecodeVersionEqual`.
MLIR_CAPI_EXPORTED uint32_t
mlirTensorIRBytecodeVersionHash(MlirTensorIRBytecodeVersion version);

//===----------------------------------------------------------------------===//
// CUDA Tile compilation options.
//===----------------------------------------------------------------------===//

/// Describes how narrowly generated code is tied to a GPU architecture.
typedef enum MlirTensorIRArchPortability {
  /// Generate code intended to remain portable to later architectures.
  MlirTensorIRArchPortabilityPortable = 0,
  /// Generate code portable within the selected architecture family.
  MlirTensorIRArchPortabilityFamilyPortable = 1,
  /// Generate code conditional on the selected concrete architecture.
  MlirTensorIRArchPortabilityArchConditional = 2,
} MlirTensorIRArchPortability;

/// Selects the TensorIR-to-CUDA-Tile lowering strategy.
typedef enum MlirTensorIRCudaTileCodegenStrategy {
  /// Derive iteration-space maps using affine analysis.
  MlirTensorIRCudaTileCodegenStrategyAffineMap = 0,
  /// Derive access patterns using layout propagation.
  MlirTensorIRCudaTileCodegenStrategyLayoutPropagation = 1,
} MlirTensorIRCudaTileCodegenStrategy;

/// Selects the requested CUDA Tile device artifact.
typedef enum MlirTensorIRCudaTileArtifactKind {
  /// Return TileIR bytecode for the deployment driver to JIT.
  MlirTensorIRCudaTileArtifactKindTileIR = 0,
  /// Request a cubin for an arch-conditional target. Internal builds use
  /// libtileiras; OSS builds use a compatible `tileiras` from PATH. Return
  /// TileIR bytecode when no compatible assembler is available.
  MlirTensorIRCudaTileArtifactKindCubin = 1,
} MlirTensorIRCudaTileArtifactKind;

/// Configures compilation of a TensorIR module for the CUDA Tile runtime.
///
/// Pointer and string fields are borrowed and need only remain valid for the
/// duration of the call that receives this structure. Empty optional paths
/// are represented by an `MlirStringRef` with length zero.
typedef struct MlirTensorIRCudaTileCompileOptions {
  /// Target compute capability encoded as an SM integer, for example 100 for
  /// `sm_100`.
  int32_t computeCapability;

  /// Preferred portability mode applied to `computeCapability`. If the target
  /// architecture does not support this mode, compilation uses the closest
  /// supported mode with lower specificity. For example, arch-conditional
  /// `sm_80` is lowered to portable `sm_80`.
  MlirTensorIRArchPortability archPortability;

  /// Number of cooperative thread arrays used by each MMA instruction.
  int32_t ctaCount;

  /// Number of warps in each cooperative thread array.
  int32_t warpCount;

  /// Per-dimension tile sizes. Must be non-NULL when `numTileSizes` is nonzero.
  const int32_t *tileSizes;

  /// Number of elements in `tileSizes`.
  size_t numTileSizes;

  /// Whether the kernel signature includes size and stride arguments for every
  /// tensor dimension, including static dimensions.
  bool uniformSignature;

  /// TensorIR-to-CUDA-Tile lowering strategy.
  MlirTensorIRCudaTileCodegenStrategy codegenStrategy;

  /// Maximum number of tile candidates retained by layout propagation.
  /// Ignored by the affine-map strategy.
  int32_t maxTileCandidates;

  /// Optional path to which the lowered CUDA Tile IR is written.
  MlirStringRef dumpCudaTileIRPath;

  /// Optional path to which generated CUDA Tile bytecode is written.
  MlirStringRef dumpTileIRBCPath;

  /// Optional path from which CUDA Tile bytecode is loaded instead of using
  /// the bytecode generated from the lowered IR.
  MlirStringRef loadTileIRBCPath;

  /// Optional directory for per-pass IR snapshots. When non-empty, this takes
  /// precedence over `printIRAfterAll`.
  MlirStringRef printIRTreeDir;

  /// Whether to print IR after every compiler pass.
  bool printIRAfterAll;

  /// Whether to collect and print compiler pass timing information.
  bool enableTiming;

  /// CUDA Tile bytecode version targeted by compilation.
  MlirTensorIRBytecodeVersion bytecodeVersion;

  /// Selects the requested device artifact. Cubin requires
  /// MlirTensorIRArchPortabilityArchConditional. Zero-initialization requests
  /// TileIR bytecode.
  MlirTensorIRCudaTileArtifactKind artifactKind;
} MlirTensorIRCudaTileCompileOptions;

//===----------------------------------------------------------------------===//
// Runtime argument API.
//===----------------------------------------------------------------------===//

/// Identifies the active member of `MlirTensorIRArgumentValue`.
typedef enum MlirTensorIRArgumentKind {
  /// An argument with no value.
  MlirTensorIRArgumentKindNone = 0,
  /// A signed 64-bit integer.
  MlirTensorIRArgumentKindInt64 = 1,
  /// A 64-bit floating-point value.
  MlirTensorIRArgumentKindFloat64 = 2,
  /// An opaque pointer.
  MlirTensorIRArgumentKindPointer = 3,
  /// A strided tensor view.
  MlirTensorIRArgumentKindTensor = 4,
} MlirTensorIRArgumentKind;

/// Describes a strided tensor passed to a compiled TensorIR program.
///
/// The view does not own `data`, `shape`, or `strides`. A non-NULL metadata
/// array contains `rank` elements. All provided storage must remain valid for
/// the duration of the runtime call receiving the view.
typedef struct MlirTensorIRTensorView {
  /// Pointer to the first tensor element visible through this view.
  void *data;

  /// Number of tensor dimensions.
  int32_t rank;

  /// Pointer to an array containing the extent of each dimension. May be NULL
  /// when the compiled kernel does not require runtime shape metadata.
  int64_t *shape;

  /// Pointer to an array containing the element stride of each dimension. May
  /// be NULL when the compiled kernel does not require runtime stride metadata.
  int64_t *strides;
} MlirTensorIRTensorView;

/// Stores the value of one runtime argument.
///
/// The active member is selected by the corresponding
/// `MlirTensorIRArgumentKind`.
typedef union MlirTensorIRArgumentValue {
  /// Value for `MlirTensorIRArgumentKindInt64`.
  int64_t int64Value;
  /// Value for `MlirTensorIRArgumentKindFloat64`.
  double float64Value;
  /// Value for `MlirTensorIRArgumentKindPointer`.
  void *pointerValue;
  /// Value for `MlirTensorIRArgumentKindTensor`.
  MlirTensorIRTensorView tensor;
} MlirTensorIRArgumentValue;

/// Associates a runtime argument value with its kind.
typedef struct MlirTensorIRArgument {
  /// Selects the active member of `value`.
  MlirTensorIRArgumentKind kind;

  /// Argument storage interpreted according to `kind`.
  MlirTensorIRArgumentValue value;
} MlirTensorIRArgument;

/// A borrowed contiguous sequence of runtime arguments.
///
/// `args` may be NULL when `numArgs` is zero. The array and all storage
/// referenced by its elements must remain valid for the duration of the
/// runtime call receiving this structure.
typedef struct MlirTensorIRPackedArgs {
  /// Pointer to the first runtime argument.
  const MlirTensorIRArgument *args;

  /// Number of elements in `args`.
  size_t numArgs;
} MlirTensorIRPackedArgs;

//===----------------------------------------------------------------------===//
// Compiled program API.
//===----------------------------------------------------------------------===//

/// An opaque handle to an owning compiled TensorIR program.
///
/// Handles returned by `mlirTensorIRProgramCompile` must eventually be passed
/// to `mlirTensorIRProgramDelete`.
///
/// Functions returning `MlirLogicalResult` reject a NULL program handle and
/// optionally report the error through their error callback. Destruction and
/// state-query APIs handle NULL as documented below. Whenever an optional
/// error callback is invoked, its `MlirStringRef` is valid only for the
/// duration of the callback and the associated user data is forwarded
/// unchanged.
typedef struct MlirTensorIRProgram {
  /// Opaque implementation pointer.
  void *ptr;
} MlirTensorIRProgram;

/// Compiles `module` using `options` and returns a new owning program handle.
///
/// `module` and all borrowed storage in `options` need only remain valid for
/// the duration of this call. On failure, returns a handle whose `ptr` is NULL
/// and, when `errorCallback` is non-NULL, forwards an error message and
/// `errorUserData` to it. The message is valid only during the callback.
MLIR_CAPI_EXPORTED MlirTensorIRProgram mlirTensorIRProgramCompile(
    MlirModule module, MlirTensorIRCudaTileCompileOptions options,
    MlirStringCallback errorCallback, void *errorUserData);

/// Checks whether `module` can be compiled using `options` without compiling
/// it.
///
/// On success, writes the result to `canCompile`. On failure, leaves
/// `canCompile` unspecified and optionally reports an error through
/// `errorCallback`. `canCompile` must not be NULL.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRProgramCanCompile(
    MlirModule module, MlirTensorIRCudaTileCompileOptions options,
    bool *canCompile, MlirStringCallback errorCallback, void *errorUserData);

/// Deletes `program` and releases both the handle and its owned resources.
///
/// The handle is invalid after this call and must not be used again. Each
/// non-NULL owning handle must be deleted exactly once. A handle whose `ptr` is
/// NULL may also be passed and is ignored.
MLIR_CAPI_EXPORTED void mlirTensorIRProgramDelete(MlirTensorIRProgram program);

/// Releases the resources owned by `program` without deleting the handle.
///
/// This operation is idempotent. A destroyed program cannot be initialized or
/// executed, but its handle must still be passed to
/// `mlirTensorIRProgramDelete`. A NULL handle is ignored.
MLIR_CAPI_EXPORTED void mlirTensorIRProgramDestroy(MlirTensorIRProgram program);

/// Returns true if the resources owned by `program` have been destroyed or if
/// the handle is NULL.
MLIR_CAPI_EXPORTED bool
mlirTensorIRProgramIsDestroyed(MlirTensorIRProgram program);

/// Returns true if the runtime state for `program` has been initialized.
/// Returns false if the handle is NULL.
MLIR_CAPI_EXPORTED bool
mlirTensorIRProgramIsInitialized(MlirTensorIRProgram program);

/// Initializes the runtime state for `program`.
///
/// Initialization is idempotent and is also performed lazily by
/// `mlirTensorIRProgramLaunch`. On failure, an error is optionally reported
/// through `errorCallback`.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRProgramInitialize(
    MlirTensorIRProgram program, MlirStringCallback errorCallback,
    void *errorUserData);

/// Checks whether `program` supports `args`.
///
/// On success, writes the result to `supported`. A runtime constraint mismatch
/// is reported as a successful call with `supported` set to false. Other
/// errors produce a failure and are optionally reported through
/// `errorCallback`. `supported` must not be NULL.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRProgramCheckSupport(
    MlirTensorIRProgram program, MlirTensorIRPackedArgs args, bool *supported,
    MlirStringCallback errorCallback, void *errorUserData);

/// Queries the number of workspace bytes required to launch `program` with
/// `args`.
///
/// On success, writes the required size to `workspaceSize`. On failure, leaves
/// `workspaceSize` unspecified and optionally reports an error through
/// `errorCallback`. `workspaceSize` must not be NULL.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRProgramQueryWorkspaceSize(
    MlirTensorIRProgram program, MlirTensorIRPackedArgs args,
    size_t *workspaceSize, MlirStringCallback errorCallback,
    void *errorUserData);

/// Launches `program` with `args` on the CUDA stream represented by `stream`.
///
/// `workspaceData` points to `workspaceSize` bytes of caller-owned device
/// storage. It may be NULL when no workspace is required. A NULL `stream`
/// selects the default CUDA stream. On failure, an error is optionally reported
/// through `errorCallback`.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRProgramLaunch(
    MlirTensorIRProgram program, MlirTensorIRPackedArgs args,
    void *workspaceData, size_t workspaceSize, void *stream,
    MlirStringCallback errorCallback, void *errorUserData);

/// Sends the selected TileIR bytecode or cuBin artifact to `bytecodeCallback`.
///
/// The artifact is binary data and is not NUL-terminated. Its `MlirStringRef`
/// is valid only for the duration of the callback. `bytecodeCallback` must not
/// be NULL. On failure, an error is optionally reported through
/// `errorCallback`.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRProgramGetBytecode(
    MlirTensorIRProgram program, MlirStringCallback bytecodeCallback,
    void *bytecodeUserData, MlirStringCallback errorCallback,
    void *errorUserData);

// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
}
#endif

#endif // TENSOR_IR_C_TENSOR_IR_H
