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

#define GEN_COMPILERINVOCATION_CAPI_DECLS
#include "tensor_ir/Options/CompilerOptions.h.inc"

/// Validates cross-component option combinations.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRCompilerInvocationValidate(
    MlirTensorIRCompilerInvocation invocation, MlirStringCallback errorCallback,
    void *errorUserData);

/// Sets the target the kernel is compiled for from the compute capability and
/// portability pair callers choose it by.
///
/// The invocation stores the target as the single `kernel-target` spelling, so
/// these accessors are the C mirror of that spelling's parse and print. Fails
/// on a combination no target supports, leaving the invocation unchanged.
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirTensorIRCompilerInvocationSetKernelTarget(
    MlirTensorIRCompilerInvocation invocation, int32_t computeCapability,
    MlirTensorIRArchPortability portability);

/// Returns the compute capability of the target `invocation` names, or 0 if it
/// names no valid target.
MLIR_CAPI_EXPORTED int32_t
mlirTensorIRCompilerInvocationGetKernelComputeCapability(
    MlirTensorIRCompilerInvocation invocation);

/// Returns the portability of the target `invocation` names, or
/// `MlirTensorIRArchPortabilityFamilyPortable` if it names no valid target.
MLIR_CAPI_EXPORTED MlirTensorIRArchPortability
mlirTensorIRCompilerInvocationGetKernelArchPortability(
    MlirTensorIRCompilerInvocation invocation);

/// The CUDA Tile bytecode version to target. Unsupported versions are retained
/// so that validation, cache-key calculation, and compilation reject the
/// request instead of silently using the previous version. `tile-size` is the
/// other CUDA Tile option the generated typed accessors cannot cover; these
/// are its C API.
MLIR_CAPI_EXPORTED void mlirTensorIRCompilerInvocationSetBytecodeVersion(
    MlirTensorIRCompilerInvocation invocation,
    MlirTensorIRBytecodeVersion version);

/// Returns the configured bytecode version, including a complete numeric tuple
/// for an unsupported version. Returns `{0, 0, 0}` for a null invocation or a
/// malformed stored spelling.
MLIR_CAPI_EXPORTED MlirTensorIRBytecodeVersion
mlirTensorIRCompilerInvocationGetBytecodeVersion(
    MlirTensorIRCompilerInvocation invocation);

/// Sets the per-dimension kernel tile sizes to the `numSizes` extents `sizes`
/// points at. `sizes` may be NULL when `numSizes` is zero.
MLIR_CAPI_EXPORTED void mlirTensorIRCompilerInvocationSetTileSizes(
    MlirTensorIRCompilerInvocation invocation, const int32_t *sizes,
    size_t numSizes);

/// Returns the per-dimension kernel tile sizes and writes their count to
/// `numSizes`, which must not be NULL.
///
/// The returned array borrows storage `invocation` keeps owning: it is valid
/// until the tile sizes are set again or the invocation is destroyed. Returns
/// NULL when no tile size is set.
MLIR_CAPI_EXPORTED const int32_t *mlirTensorIRCompilerInvocationGetTileSizes(
    MlirTensorIRCompilerInvocation invocation, size_t *numSizes);

//===----------------------------------------------------------------------===//
// Runtime argument API.
//===----------------------------------------------------------------------===//

/// Identifies the active member of `MlirTensorIRArgumentValue`.
typedef enum MlirTensorIRArgumentKind {
  /// An argument with no value.
  MlirTensorIRArgumentKindNone = 0,
  /// An opaque pointer.
  MlirTensorIRArgumentKindPointer = 1,
  /// A strided tensor view.
  MlirTensorIRArgumentKindTensor = 2,
  /// A by-value scalar with an explicit element type (see
  /// `MlirTensorIRScalarView`); the caller always declares the true element
  /// type.
  MlirTensorIRArgumentKindScalar = 3,
} MlirTensorIRArgumentKind;

/// Identifies the element type of a `MlirTensorIRScalarView`.
typedef enum MlirTensorIRElementType {
  MlirTensorIRElementTypeBool = 0,
  MlirTensorIRElementTypeF16 = 1,
  MlirTensorIRElementTypeBF16 = 2,
  MlirTensorIRElementTypeF32 = 3,
  MlirTensorIRElementTypeF64 = 4,
  MlirTensorIRElementTypeSI8 = 5,
  MlirTensorIRElementTypeSI16 = 6,
  MlirTensorIRElementTypeSI32 = 7,
  MlirTensorIRElementTypeSI64 = 8,
  MlirTensorIRElementTypeUI8 = 9,
  MlirTensorIRElementTypeUI16 = 10,
  MlirTensorIRElementTypeUI32 = 11,
  MlirTensorIRElementTypeUI64 = 12,
  MlirTensorIRElementTypeF8E4M3FN = 13,
  MlirTensorIRElementTypeF8E5M2 = 14,
} MlirTensorIRElementType;

/// Parses an MLIR assembly element type spelling (e.g. "f32") into a
/// `MlirTensorIRElementType`.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRElementTypeFromSpelling(
    MlirStringRef spelling, MlirTensorIRElementType *result,
    MlirStringCallback errorCallback, void *errorUserData);

/// A by-value scalar argument with an explicit element type. `bits` holds the
/// raw value, left-aligned in the low-order bytes for the type's byte width;
/// the high-order bytes are unspecified and must never be read.
typedef struct MlirTensorIRScalarView {
  MlirTensorIRElementType type;
  int64_t bits;
} MlirTensorIRScalarView;

/// Describes a strided tensor passed to a TensorIR runtime kernel.
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
  /// Value for `MlirTensorIRArgumentKindPointer`.
  void *pointerValue;
  /// Value for `MlirTensorIRArgumentKindTensor`.
  MlirTensorIRTensorView tensor;
  /// Value for `MlirTensorIRArgumentKindScalar`.
  MlirTensorIRScalarView scalar;
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
// Compilation artifact and runtime kernel API.
//===----------------------------------------------------------------------===//

/// Builds the unhashed cache key for compiling `module` with `options`.
///
/// The key is forwarded to `keyCallback`. On failure, the key callback is not
/// invoked and the error is optionally forwarded to `errorCallback`.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRCalculateCacheKey(
    MlirModule module, MlirTensorIRCompilerInvocation invocation,
    MlirStringCallback keyCallback, void *keyUserData,
    MlirStringCallback errorCallback, void *errorUserData);

/// Receives one tile shape. `shape` contains `rank` extents and is valid only
/// for the duration of the callback.
typedef void (*MlirTensorIRTileCandidateCallback)(const int32_t *shape,
                                                  size_t rank, void *userData);

/// Analyzes exactly one graph in place using the CUDA Tile compiler and sends
/// tile candidates to `candidateCallback` in best-first order. Requires a
/// positive maxTileCandidates or a tile hint. An explicit invocation tile size
/// produces no callbacks. The module may also be modified on failure.
///
/// `module`, `invocation`, and `candidateCallback` must not be NULL. On failure,
/// no candidates are delivered and an error is optionally sent to
/// `errorCallback`; its string is valid only during the callback.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRCudaTileAnalyzeModule(
    MlirModule module, MlirTensorIRCompilerInvocation invocation,
    MlirTensorIRTileCandidateCallback candidateCallback,
    void *candidateUserData, MlirStringCallback errorCallback,
    void *errorUserData);

/// An owning immutable compilation artifact, independent of runtime state.
/// Delete each non-null handle exactly once with mlirTensorIRArtifactDestroy.
typedef struct MlirTensorIRArtifact {
  void *ptr;
} MlirTensorIRArtifact;

/// An owning loaded runtime kernel. Delete each non-null handle exactly once
/// with mlirTensorIRRuntimeKernelDestroy. Loading retains an independent copy
/// of the artifact, so the original artifact can be destroyed immediately.
typedef struct MlirTensorIRRuntimeKernel {
  void *ptr;
} MlirTensorIRRuntimeKernel;

/// Compile without creating or loading a runtime kernel. The module and
/// invocation need only live for this call. Returns null on failure.
/// Error callback strings borrow storage valid only during the callback.
MLIR_CAPI_EXPORTED MlirTensorIRArtifact mlirTensorIRArtifactCompile(
    MlirModule module, MlirTensorIRCompilerInvocation invocation,
    MlirStringCallback errorCallback, void *errorUserData);

/// Destroy the artifact handle and its storage. A null handle is ignored.
MLIR_CAPI_EXPORTED void
mlirTensorIRArtifactDestroy(MlirTensorIRArtifact artifact);

/// Send the original device binary to callback (required). The binary string
/// borrows storage valid only during the callback. Rejects a null artifact.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRArtifactGetBytecode(
    MlirTensorIRArtifact artifact, MlirStringCallback callback, void *userData,
    MlirStringCallback errorCallback, void *errorUserData);

/// Serialize device code, host code, and metadata without loading any code.
/// Callback (required) borrows the bytes for its duration. Rejects null artifacts.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRArtifactSerialize(
    MlirTensorIRArtifact artifact, MlirStringCallback callback, void *userData,
    MlirStringCallback errorCallback, void *errorUserData);

/// Deserialize an owning artifact without loading code. Input bytes need only
/// live for this call. Returns null on failure.
MLIR_CAPI_EXPORTED MlirTensorIRArtifact mlirTensorIRArtifactDeserialize(
    MlirStringRef bytes, MlirStringCallback errorCallback, void *errorUserData);

/// Copy the artifact into a new runtime kernel and initialize host and device
/// state. Returns a ready-to-launch kernel, or null on failure (including null
/// input). The artifact remains valid and can be loaded more than once.
MLIR_CAPI_EXPORTED MlirTensorIRRuntimeKernel
mlirTensorIRArtifactLoad(MlirTensorIRArtifact artifact,
                         MlirStringCallback errorCallback, void *errorUserData);

/// Unload and destroy the runtime kernel. A null handle is ignored.
MLIR_CAPI_EXPORTED void
mlirTensorIRRuntimeKernelDestroy(MlirTensorIRRuntimeKernel kernel);

/// Runtime operations reject null kernel handles. Arguments and workspace
/// borrow caller-owned storage for the duration of the call. Output pointers
/// must be non-null. Optional error callbacks borrow their diagnostic strings.
/// A constraint mismatch succeeds with supported=false; other errors fail.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRRuntimeKernelCheckSupport(
    MlirTensorIRRuntimeKernel kernel, MlirTensorIRPackedArgs args,
    bool *supported, MlirStringCallback errorCallback, void *errorUserData);

MLIR_CAPI_EXPORTED MlirLogicalResult
mlirTensorIRRuntimeKernelQueryWorkspaceSize(MlirTensorIRRuntimeKernel kernel,
                                            MlirTensorIRPackedArgs args,
                                            size_t *workspaceSize,
                                            MlirStringCallback errorCallback,
                                            void *errorUserData);

/// Launch on stream (null selects the default stream). Workspace may be null
/// when its size is zero. No implicit loading is performed.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRRuntimeKernelLaunch(
    MlirTensorIRRuntimeKernel kernel, MlirTensorIRPackedArgs args,
    void *workspaceData, size_t workspaceSize, void *stream,
    MlirStringCallback errorCallback, void *errorUserData);

/// Checks whether `module` can be compiled using `invocation` without
/// compiling it.
///
/// On success, writes the result to `canCompile`. On failure, leaves
/// `canCompile` unspecified and optionally reports an error through
/// `errorCallback`. `canCompile` must not be NULL.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTensorIRCanCompile(
    MlirModule module, MlirTensorIRCompilerInvocation invocation,
    bool *canCompile, MlirStringCallback errorCallback, void *errorUserData);

// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
}
#endif

#endif // TENSOR_IR_C_TENSOR_IR_H
