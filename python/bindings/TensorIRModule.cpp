// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

// MLIR's Nanobind.h trampoline suppresses the warnings nanobind's headers
// trigger, but does not cover stl/unique_ptr.h. Repeat the suppressions here.
// The pragmas also pin this include in place: .clang-format uses
// IncludeBlocks: Regroup, which would otherwise sort it away from Nanobind.h.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-length-array"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#pragma GCC diagnostic ignored "-Wc++98-compat-extra-semi"
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#endif
#include "nanobind/stl/unique_ptr.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include "dlpack/dlpack.h"
#include "tensor_ir-c/TensorIR.h"
#include <Python.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

namespace {

struct PyBytecodeVersion {
  MlirTensorIRBytecodeVersion value;
};

void appendStringRef(MlirStringRef value, void *userData) {
  auto *result = static_cast<std::string *>(userData);
  result->append(value.data, value.length);
}

MlirStringRef wrapStringRef(const std::string &value) {
  return mlirStringRefCreate(value.data(), value.size());
}

std::string unwrapStringRef(MlirStringRef value) {
  return std::string(value.data, value.length);
}

std::string formatBytecodeVersion(const PyBytecodeVersion &bytecodeVersion) {
  std::string formatted;
  mlirTensorIRFormatBytecodeVersion(bytecodeVersion.value, appendStringRef,
                                    &formatted);
  return formatted;
}

[[noreturn]] void throwCAPIError(llvm::StringRef api,
                                 const std::string &error) {
  llvm::StringRef message = error;
  if (message.empty()) {
    message = "TensorIR C API call failed";
  }
  std::string fullMessage = api.str();
  fullMessage += ": ";
  fullMessage.append(message.data(), message.size());
  throw std::runtime_error(fullMessage);
}

void checkCAPIResult(MlirLogicalResult result, llvm::StringRef api,
                     const std::string &error) {
  if (mlirLogicalResultIsFailure(result)) {
    throwCAPIError(api, error);
  }
}

/// Shared ownership for the opaque invocation used by CompileOptions and its
/// InternalOptions view.
struct InvocationStorage {
  InvocationStorage() : value(mlirTensorIRCompilerInvocationCreate()) {}
  ~InvocationStorage() { mlirTensorIRCompilerInvocationDestroy(value); }

  MlirTensorIRCompilerInvocation value;
};

/// The options a `compile` call takes: an owning compiler invocation handle
/// plus the defaults this module has always presented.
///
/// Every option is reached through `tensor_ir-c`, never through the C++ options
/// structs: the typed accessors the C API declares per component for the
/// options whose type has a C spelling, and the invocation-level accessors for
/// the target and for the two (a version and a list) that do not.
class CompileOptions {
public:
  CompileOptions() : invocation(std::make_shared<InvocationStorage>()) {
    mlirTensorIRCompilerInvocationSetCompilerBackend(
        handle(), MlirTensorIRCodegenCompilerBackendCudaTile);
    mlirTensorIRCompilerInvocationSetBytecodeVersion(
        handle(), mlirTensorIRGetDefaultBytecodeVersion());
    // Cannot fail: sm_100 supports family portability, so this is a target.
    (void)mlirTensorIRCompilerInvocationSetKernelTarget(
        handle(), 100, MlirTensorIRArchPortabilityFamilyPortable);
  }

  // Move-only: the handle is owning, and nothing in the Python API copies an
  // options object. `parse` returns one by value, which the move covers.
  CompileOptions(CompileOptions &&) noexcept = default;
  CompileOptions &operator=(CompileOptions &&) noexcept = default;
  CompileOptions(const CompileOptions &) = delete;
  CompileOptions &operator=(const CompileOptions &) = delete;
  ~CompileOptions() = default;

  MlirTensorIRCompilerInvocation handle() const { return invocation->value; }

  static CompileOptions parse(const std::string &text) {
    CompileOptions options;
    if (mlirLogicalResultIsFailure(mlirTensorIRCompilerInvocationSetFromString(
            options.handle(), wrapStringRef(text)))) {
      throw std::invalid_argument("Failed to parse TensorIR compile options: " +
                                  text);
    }
    return options;
  }

  std::string toString() const {
    std::string text;
    mlirTensorIRCompilerInvocationPrint(handle(), appendStringRef, &text);
    return text;
  }

private:
  std::shared_ptr<InvocationStorage> invocation;
};

/// Retargets `options`, reporting a compute capability and portability pair no
/// target supports as \p what.
void setKernelTargetOrThrow(MlirTensorIRCompilerInvocation options,
                            int32_t computeCapability,
                            MlirTensorIRArchPortability portability,
                            const std::string &what) {
  if (mlirLogicalResultIsFailure(mlirTensorIRCompilerInvocationSetKernelTarget(
          options, computeCapability, portability))) {
    throw std::invalid_argument("Unsupported TensorIR " + what);
  }
}

/// Exposes a compile option as the read/write Python property \p name, reading
/// it with \p get and writing it with \p set. Both receive the invocation
/// handle, so that a binding names the C API accessors that reach its option
/// and nothing else. \p Value is the Python-visible type, which nanobind needs
/// spelled out.
template <typename Value, typename Class, typename Get, typename Set>
void bindOption(Class &cls, const char *name, Get get, Set set) {
  using Self = typename Class::Type;
  cls.def_prop_rw(
      name, [get](const Self &self) { return get(self.handle()); },
      [set](const Self &self, Value value) {
        set(self.handle(), std::move(value));
      });
}

/// Same, for a string option whose C getter returns borrowed storage.
template <typename Class>
void bindStringOption(Class &cls, const char *name,
                      MlirStringRef (*get)(MlirTensorIRCompilerInvocation),
                      void (*set)(MlirTensorIRCompilerInvocation,
                                  MlirStringRef)) {
  bindOption<std::string>(
      cls, name,
      [get](MlirTensorIRCompilerInvocation options) {
        return unwrapStringRef(get(options));
      },
      [set](MlirTensorIRCompilerInvocation options, const std::string &value) {
        set(options, wrapStringRef(value));
      });
}

struct PackedPythonArgs {
  std::vector<MlirTensorIRArgument> args;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::vector<int64_t>> strides;
  std::vector<nb::object> dlpackCapsules;

  MlirTensorIRPackedArgs packed() const { return {args.data(), args.size()}; }
};

struct PackedWorkspace {
  void *data = nullptr;
  size_t size = 0;
};

void checkCAPIResult(MlirLogicalResult result, llvm::StringRef api,
                     const std::string &error);

MlirModule validateModule(MlirModule module) {
  MlirOperation operation = mlirModuleGetOperation(module);
  MlirModule moduleFromOperation = mlirModuleFromOperation(operation);
  if (mlirModuleIsNull(moduleFromOperation)) {
    throw nb::type_error("expected an MLIR builtin.module");
  }
  return moduleFromOperation;
}

std::string calculateCacheKey(MlirModule module,
                              const CompileOptions &options) {
  std::string key;
  std::string error;
  checkCAPIResult(mlirTensorIRCalculateCacheKey(
                      validateModule(module), options.handle(), appendStringRef,
                      &key, appendStringRef, &error),
                  "mlirTensorIRCalculateCacheKey", error);
  return key;
}

bool supportsDLPack(nb::handle value) {
  return nb::hasattr(value, "__dlpack__") &&
         nb::hasattr(value, "__dlpack_device__");
}

template <typename T>
int64_t bitcastAndZExtTo64(T value) {
  int64_t bits = 0;
  static_assert(sizeof(T) <= sizeof(bits));
  std::memcpy(&bits, &value, sizeof(T));
  return bits;
}

// nb::cast<int64_t> throws an unfriendly, nanobind-internal exception for a
// Python int outside int64_t's range (e.g. >= 2**63) -- report a clean
// TypeError instead.
int64_t castToInt64OrThrow(nb::handle value) {
  try {
    return nb::cast<int64_t>(value);
  } catch (const std::exception &) {
    throw nb::type_error(
        "TensorIR scalar value does not fit in the declared dtype");
  }
}

// A bare Python int/float scalar argument gets a fixed default element type
// from its Python type: float -> F32, int -> SI64.
MlirTensorIRArgument packBareNumberArg(nb::handle value) {
  MlirTensorIRArgument argument{};
  argument.kind = MlirTensorIRArgumentKindScalar;
  if (nb::isinstance<nb::float_>(value)) {
    argument.value.scalar.type = MlirTensorIRElementTypeF32;
    argument.value.scalar.bits =
        bitcastAndZExtTo64<float>(static_cast<float>(nb::cast<double>(value)));
  } else {
    argument.value.scalar.type = MlirTensorIRElementTypeSI64;
    argument.value.scalar.bits =
        bitcastAndZExtTo64<int64_t>(castToInt64OrThrow(value));
  }
  return argument;
}

// A bare Python bool is an unambiguous i1 scalar (unlike int/float, which
// have no fixed width) -- accepted directly.
MlirTensorIRArgument packBoolArg(nb::handle value) {
  MlirTensorIRArgument argument{};
  argument.kind = MlirTensorIRArgumentKindScalar;
  argument.value.scalar.type = MlirTensorIRElementTypeBool;
  argument.value.scalar.bits = bitcastAndZExtTo64<bool>(nb::cast<bool>(value));
  return argument;
}

std::string pythonTypeName(nb::handle value) {
  const char *name = Py_TYPE(value.ptr())->tp_name;
  return name ? std::string(name) : std::string("<unknown>");
}

nb::object createDLPackCapsule(nb::handle value) {
  nb::object object = nb::borrow<nb::object>(value);
  return object.attr("__dlpack__")();
}

std::string dlpackDTypeDescription(DLDataType dtype) {
  return "DLPack code=" + std::to_string(static_cast<int>(dtype.code)) +
         " bits=" + std::to_string(static_cast<int>(dtype.bits)) +
         " lanes=" + std::to_string(static_cast<int>(dtype.lanes));
}

DLTensor *getDLTensor(nb::object &capsule) {
  if (!PyObject_TypeCheck(capsule.ptr(), &PyCapsule_Type)) {
    throw nb::type_error("__dlpack__() did not return a PyCapsule");
  }

  const char *name = PyCapsule_GetName(capsule.ptr());
  void *ptr = PyCapsule_GetPointer(capsule.ptr(), name);
  if (!ptr) {
    throw nb::type_error("DLPack capsule data is null");
  }

  llvm::StringRef capsuleName(name ? name : "");
  if (capsuleName == "dltensor_versioned") {
    auto *versioned = static_cast<DLManagedTensorVersioned *>(ptr);
    if (versioned->version.major != DLPACK_MAJOR_VERSION) {
      throw nb::type_error(
          ("DLPack major version " +
           std::to_string(static_cast<int>(versioned->version.major)) +
           " is not supported; expected " +
           std::to_string(DLPACK_MAJOR_VERSION))
              .c_str());
    }
    return &versioned->dl_tensor;
  }
  if (capsuleName == "dltensor") {
    return &static_cast<DLManagedTensor *>(ptr)->dl_tensor;
  }
  throw nb::type_error("DLPack capsule must be named dltensor or "
                       "dltensor_versioned");
}

std::vector<int64_t> contiguousStrides(llvm::ArrayRef<int64_t> shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t running = 1;
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    strides[dim] = running;
    running *= shape[dim];
  }
  return strides;
}

std::vector<int64_t> dlpackShape(const DLTensor &tensor) {
  if (tensor.ndim < 0) {
    throw nb::type_error("DLPack tensor rank must be non-negative");
  }
  return std::vector<int64_t>(tensor.shape, tensor.shape + tensor.ndim);
}

std::vector<int64_t> dlpackStrides(const DLTensor &tensor,
                                   llvm::ArrayRef<int64_t> shape) {
  if (tensor.strides) {
    return std::vector<int64_t>(tensor.strides, tensor.strides + tensor.ndim);
  }
  return contiguousStrides(shape);
}

size_t checkedElementCount(llvm::ArrayRef<int64_t> shape) {
  size_t elementCount = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw nb::type_error(
          "DLPack tensor shape dimensions must be non-negative");
    }
    size_t sizeDim = static_cast<size_t>(dim);
    if (sizeDim != 0 &&
        elementCount > std::numeric_limits<size_t>::max() / sizeDim) {
      throw nb::type_error("DLPack tensor element count overflow");
    }
    elementCount *= sizeDim;
  }
  return elementCount;
}

size_t checkedByteSize(const DLTensor &tensor, llvm::ArrayRef<int64_t> shape) {
  size_t elementCount = checkedElementCount(shape);
  size_t elementBytes =
      static_cast<size_t>((tensor.dtype.bits * tensor.dtype.lanes + 7) / 8);
  if (elementBytes != 0 &&
      elementCount > std::numeric_limits<size_t>::max() / elementBytes) {
    throw nb::type_error("DLPack tensor byte size overflow");
  }
  return elementCount * elementBytes;
}

std::string tensorIRTypeFromDLPack(DLDataType dtype) {
  if (dtype.lanes != 1) {
    throw nb::type_error(
        ("TensorIR DSL does not support DLPack vector dtypes (" +
         dlpackDTypeDescription(dtype) + ")")
            .c_str());
  }
  if (dtype.code == kDLBool) {
    return "i1";
  }
  if (dtype.code == kDLFloat) {
    if (dtype.bits == 16) {
      return "f16";
    }
    if (dtype.bits == 32) {
      return "f32";
    }
    if (dtype.bits == 64) {
      return "f64";
    }
  }
  if (dtype.code == kDLBfloat && dtype.bits == 16) {
    return "bf16";
  }
  if (dtype.code == kDLFloat8_e4m3fn && dtype.bits == 8) {
    return "f8E4M3FN";
  }
  if (dtype.code == kDLFloat8_e5m2 && dtype.bits == 8) {
    return "f8E5M2";
  }
  if (dtype.code == kDLInt || dtype.code == kDLUInt) {
    if (dtype.bits == 8 || dtype.bits == 16 || dtype.bits == 32 ||
        dtype.bits == 64) {
      return std::string(dtype.code == kDLInt ? "si" : "ui") +
             std::to_string(dtype.bits);
    }
  }
  throw nb::type_error(("Unsupported TensorIR tensor dtype (" +
                        dlpackDTypeDescription(dtype) + ")")
                           .c_str());
}

DLTensor *keepDLPackTensorAlive(nb::handle value, PackedPythonArgs &packed) {
  // The capsule owns the producer's DLManagedTensor. Keep it alive until the
  // packed argument object is destroyed so tensor data/shape/stride
  // pointers remain valid for the whole runtime call.
  packed.dlpackCapsules.push_back(createDLPackCapsule(value));
  return getDLTensor(packed.dlpackCapsules.back());
}

void appendDLPackTensorView(const DLTensor &tensor, PackedPythonArgs &packed) {
  std::vector<int64_t> shape = dlpackShape(tensor);
  std::vector<int64_t> stride = dlpackStrides(tensor, shape);
  if (shape.size() != stride.size()) {
    throw nb::type_error(
        "DLPack tensor must have matching shape and stride ranks");
  }
  packed.shapes.push_back(std::move(shape));
  packed.strides.push_back(std::move(stride));
}

// A 0-d DLPack object used as a scalar argument; its element type comes
// from its own dtype. GPU-resident 0-d objects are rejected below (see
// the device_type check) -- express those as a real tensor operand
// instead.
MlirTensorIRArgument packDLPackScalarArg(const DLTensor &tensor) {
  if (tensor.device.device_type != kDLCPU &&
      tensor.device.device_type != kDLCUDAHost) {
    throw nb::type_error(
        "TensorIR 0-d DLPack scalar arguments must be host-resident (RTK's "
        "by-value scalar ABI only supports the host argument mode)");
  }
  std::string spelling = tensorIRTypeFromDLPack(tensor.dtype);
  MlirTensorIRElementType type;
  std::string error;
  if (mlirLogicalResultIsFailure(mlirTensorIRElementTypeFromSpelling(
          mlirStringRefCreate(spelling.data(), spelling.size()), &type,
          appendStringRef, &error))) {
    throw nb::type_error(error.c_str());
  }
  size_t width =
      static_cast<size_t>((tensor.dtype.bits * tensor.dtype.lanes + 7) / 8);
  // tensorIRTypeFromDLPack validates dtype.bits for every code except
  // kDLBool (which it accepts unconditionally), so an untrusted DLPack
  // producer reporting an oversized bool width could otherwise make the
  // memcpy below overrun `bits`.
  if (width == 0 || width > sizeof(int64_t)) {
    throw nb::type_error(
        "TensorIR 0-d DLPack scalar dtype width must be between 1 and 64 "
        "bits");
  }
  int64_t bits = 0;
  std::memcpy(&bits, static_cast<char *>(tensor.data) + tensor.byte_offset,
              width);
  MlirTensorIRArgument argument{};
  argument.kind = MlirTensorIRArgumentKindScalar;
  argument.value.scalar.type = type;
  argument.value.scalar.bits = bits;
  return argument;
}

// Packs an N-d DLPack object as a Tensor argument. `capsule` must be the
// result of createDLPackCapsule(value); ownership moves into
// `packed.dlpackCapsules` so the data/shape/stride pointers below stay
// valid for the whole runtime call.
MlirTensorIRArgument packDLPackTensorArg(nb::object capsule,
                                         PackedPythonArgs &packed) {
  packed.dlpackCapsules.push_back(std::move(capsule));
  const DLTensor &tensor = *getDLTensor(packed.dlpackCapsules.back());
  appendDLPackTensorView(tensor, packed);

  std::vector<int64_t> &shape = packed.shapes.back();
  std::vector<int64_t> &stride = packed.strides.back();
  char *data = static_cast<char *>(tensor.data) + tensor.byte_offset;
  MlirTensorIRArgument argument{};
  argument.kind = MlirTensorIRArgumentKindTensor;
  argument.value.tensor = {data, static_cast<int32_t>(shape.size()),
                           shape.data(), stride.data()};
  return argument;
}

PackedPythonArgs packArgs(nb::tuple pyArgs) {
  PackedPythonArgs packed;
  packed.args.reserve(pyArgs.size());
  packed.shapes.reserve(pyArgs.size());
  packed.strides.reserve(pyArgs.size());
  for (nb::handle arg : pyArgs) {
    MlirTensorIRArgument argument{};
    if (arg.is_none()) {
      argument.kind = MlirTensorIRArgumentKindNone;
    } else if (nb::isinstance<nb::bool_>(arg)) {
      argument = packBoolArg(arg);
    } else if (nb::isinstance<nb::int_>(arg) ||
               nb::isinstance<nb::float_>(arg)) {
      argument = packBareNumberArg(arg);
    } else if (supportsDLPack(arg)) {
      nb::object capsule = createDLPackCapsule(arg);
      const DLTensor &tensor = *getDLTensor(capsule);
      argument = tensor.ndim == 0
                     ? packDLPackScalarArg(tensor)
                     : packDLPackTensorArg(std::move(capsule), packed);
    } else {
      throw nb::type_error(
          ("TensorIR arguments must be a bool, a number, or a "
           "DLPack-compatible tensor with __dlpack__ and __dlpack_device__ "
           "(got " +
           pythonTypeName(arg) + ")")
              .c_str());
    }
    packed.args.push_back(argument);
  }
  return packed;
}

PackedWorkspace packWorkspace(nb::object workspace, PackedPythonArgs &packed) {
  if (workspace.is_none()) {
    return {};
  }
  if (supportsDLPack(workspace)) {
    DLTensor *tensor = keepDLPackTensorAlive(workspace, packed);
    std::vector<int64_t> shape = dlpackShape(*tensor);
    char *data = static_cast<char *>(tensor->data) + tensor->byte_offset;
    return {data, checkedByteSize(*tensor, shape)};
  }
  throw nb::type_error(("workspace must be None or a DLPack-compatible tensor "
                        "with __dlpack__ and __dlpack_device__ (got " +
                        pythonTypeName(workspace) + ")")
                           .c_str());
}

nb::tuple tensorMetadataFromDLPack(nb::handle value) {
  if (!supportsDLPack(value)) {
    throw nb::type_error(("Expected a DLPack-compatible tensor with __dlpack__ "
                          "and __dlpack_device__ (got " +
                          pythonTypeName(value) + ")")
                             .c_str());
  }
  nb::object capsule = createDLPackCapsule(value);
  DLTensor *tensor = getDLTensor(capsule);
  std::vector<int64_t> shape = dlpackShape(*tensor);
  std::vector<int64_t> stride = dlpackStrides(*tensor, shape);
  // Match the effective address used by packDLPackTensorArg, including views.
  // Return zero for null data; alignment policy belongs to the DSL backend.
  uintptr_t pointer = tensor->data ? reinterpret_cast<uintptr_t>(tensor->data) +
                                         tensor->byte_offset
                                   : 0;
  return nb::make_tuple(shape, stride, tensorIRTypeFromDLPack(tensor->dtype),
                        pointer);
}

void *packStream(nb::object stream) {
  if (stream.is_none()) {
    return nullptr;
  }
  if (nb::isinstance<nb::int_>(stream)) {
    auto streamPtr = static_cast<uintptr_t>(nb::cast<uint64_t>(stream));
    return reinterpret_cast<void *>(streamPtr);
  }
  if (nb::hasattr(stream, "cuda_stream")) {
    auto streamPtr =
        static_cast<uintptr_t>(nb::cast<uint64_t>(stream.attr("cuda_stream")));
    return reinterpret_cast<void *>(streamPtr);
  }
  throw nb::type_error(
      "stream must be None, an integer pointer, or a CUDA stream");
}

class PyRuntimeKernel {
public:
  explicit PyRuntimeKernel(MlirTensorIRRuntimeKernel kernel)
      : kernel_(kernel) {}
  PyRuntimeKernel(const PyRuntimeKernel &) = delete;
  PyRuntimeKernel &operator=(const PyRuntimeKernel &) = delete;
  ~PyRuntimeKernel() { mlirTensorIRRuntimeKernelDestroy(kernel_); }

  bool checkSupport(nb::tuple pyArgs) {
    PackedPythonArgs args = packArgs(pyArgs);
    MlirTensorIRPackedArgs packedArgs = args.packed();
    bool supported = false;
    std::string error;
    checkCAPIResult(
        mlirTensorIRRuntimeKernelCheckSupport(kernel_, packedArgs, &supported,
                                              appendStringRef, &error),
        "mlirTensorIRRuntimeKernelCheckSupport", error);
    return supported;
  }

  size_t queryWorkspaceSize(nb::tuple pyArgs) {
    PackedPythonArgs args = packArgs(pyArgs);
    MlirTensorIRPackedArgs packedArgs = args.packed();
    size_t workspaceSize = 0;
    std::string error;
    checkCAPIResult(
        mlirTensorIRRuntimeKernelQueryWorkspaceSize(
            kernel_, packedArgs, &workspaceSize, appendStringRef, &error),
        "mlirTensorIRRuntimeKernelQueryWorkspaceSize", error);
    return workspaceSize;
  }

  void launch(nb::tuple pyArgs, nb::object workspace, nb::object stream) {
    PackedPythonArgs args = packArgs(pyArgs);
    MlirTensorIRPackedArgs packedArgs = args.packed();
    PackedWorkspace packedWorkspace = packWorkspace(std::move(workspace), args);
    void *packedStream = packStream(std::move(stream));
    std::string error;
    checkCAPIResult(mlirTensorIRRuntimeKernelLaunch(
                        kernel_, packedArgs, packedWorkspace.data,
                        packedWorkspace.size, packedStream, appendStringRef,
                        &error),
                    "mlirTensorIRRuntimeKernelLaunch", error);
  }

private:
  MlirTensorIRRuntimeKernel kernel_;
};

class PyArtifact {
public:
  explicit PyArtifact(MlirTensorIRArtifact artifact) : artifact_(artifact) {}
  PyArtifact(const PyArtifact &) = delete;
  PyArtifact &operator=(const PyArtifact &) = delete;
  ~PyArtifact() { mlirTensorIRArtifactDestroy(artifact_); }

  std::unique_ptr<PyRuntimeKernel> load() const {
    std::string error;
    auto kernel = mlirTensorIRArtifactLoad(artifact_, appendStringRef, &error);
    if (!kernel.ptr) {
      throwCAPIError("mlirTensorIRArtifactLoad", error);
    }
    return std::make_unique<PyRuntimeKernel>(kernel);
  }

  nb::bytes getBytecode() const {
    std::string bytecode;
    std::string error;
    checkCAPIResult(mlirTensorIRArtifactGetBytecode(artifact_, appendStringRef,
                                                    &bytecode, appendStringRef,
                                                    &error),
                    "mlirTensorIRArtifactGetBytecode", error);
    return nb::bytes(bytecode.data(), bytecode.size());
  }

  nb::bytes serialize() const {
    std::string bytes;
    std::string error;
    checkCAPIResult(mlirTensorIRArtifactSerialize(artifact_, appendStringRef,
                                                  &bytes, appendStringRef,
                                                  &error),
                    "mlirTensorIRArtifactSerialize", error);
    return nb::bytes(bytes.data(), bytes.size());
  }

  static std::unique_ptr<PyArtifact> deserialize(nb::bytes bytes) {
    MlirStringRef data = mlirStringRefCreate(bytes.c_str(), bytes.size());
    std::string error;
    MlirTensorIRArtifact artifact =
        mlirTensorIRArtifactDeserialize(data, appendStringRef, &error);
    if (!artifact.ptr) {
      throwCAPIError("mlirTensorIRArtifactDeserialize", error);
    }
    return std::make_unique<PyArtifact>(artifact);
  }

private:
  MlirTensorIRArtifact artifact_;
};

} // namespace

NB_MODULE(_tensor_ir, m) {
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__nv_tensor_ir__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  nb::enum_<MlirTensorIRArchPortability>(m, "ArchPortability")
      .value("portable", MlirTensorIRArchPortabilityPortable)
      .value("family_portable", MlirTensorIRArchPortabilityFamilyPortable)
      .value("arch_conditional", MlirTensorIRArchPortabilityArchConditional);

  nb::enum_<MlirTensorIRCudaTileCodegenStrategy>(m, "CodegenStrategy")
      .value("AffineMap", MlirTensorIRCudaTileCodegenStrategyAffineMap)
      .value("LayoutPropagation",
             MlirTensorIRCudaTileCodegenStrategyLayoutPropagation);

  nb::enum_<MlirTensorIRCudaTileArtifactKind>(m, "CudaTileArtifactKind")
      .value("TileIR", MlirTensorIRCudaTileArtifactKindTileIR)
      .value("Cubin", MlirTensorIRCudaTileArtifactKindCubin);

  nb::enum_<MlirTensorIRCudaTileAutoGdcInsertion>(m, "AutoGdcInsertion")
      .value("NONE", MlirTensorIRCudaTileAutoGdcInsertionNone)
      .value("CONSERVATIVE", MlirTensorIRCudaTileAutoGdcInsertionConservative);

  nb::enum_<MlirTensorIRCodegenHostTarget>(m, "HostTarget")
      .value("Auto", MlirTensorIRCodegenHostTargetAuto)
      .value("Native", MlirTensorIRCodegenHostTargetNative)
      .value("Interpreter", MlirTensorIRCodegenHostTargetInterpreter);
  nb::enum_<MlirTensorIRCodegenHostArch>(m, "HostArch")
      .value("Native", MlirTensorIRCodegenHostArchNative)
      .value("X86_64", MlirTensorIRCodegenHostArchX86_64)
      .value("AArch64", MlirTensorIRCodegenHostArchAArch64);
  nb::enum_<MlirTensorIRCodegenHostOS>(m, "HostOS")
      .value("Windows", MlirTensorIRCodegenHostOSWindows)
      .value("Linux", MlirTensorIRCodegenHostOSLinux);

  nb::class_<PyBytecodeVersion>(m, "BytecodeVersion")
      .def_static("current",
                  [] {
                    return PyBytecodeVersion{
                        mlirTensorIRGetCurrentBytecodeVersion()};
                  })
      .def_static("compatibility",
                  [] {
                    return PyBytecodeVersion{
                        mlirTensorIRGetCompatibilityBytecodeVersion()};
                  })
      .def_static("default",
                  [] {
                    return PyBytecodeVersion{
                        mlirTensorIRGetDefaultBytecodeVersion()};
                  })
      .def("__str__", &formatBytecodeVersion)
      .def("__eq__",
           [](const PyBytecodeVersion &lhs, const PyBytecodeVersion &rhs) {
             return mlirTensorIRBytecodeVersionEqual(lhs.value, rhs.value);
           })
      .def("__hash__",
           [](const PyBytecodeVersion &version) {
             return static_cast<Py_hash_t>(
                 mlirTensorIRBytecodeVersionHash(version.value));
           })
      .def("__repr__", [](const PyBytecodeVersion &version) {
        return "BytecodeVersion('" + formatBytecodeVersion(version) + "')";
      });

  auto compileOptions =
      nb::class_<CompileOptions>(m, "CompileOptions")
          .def(nb::init<>())
          .def_static("parse", &CompileOptions::parse, nb::arg("text"))
          .def("__str__", &CompileOptions::toString);

  // The target reaches Python as the compute capability and portability pair a
  // caller chooses; the invocation stores the `kernel-target` spelling they name.
  // Setting one preserves the other.
  bindOption<int32_t>(
      compileOptions, "compute_capability",
      [](auto options) {
        return mlirTensorIRCompilerInvocationGetKernelComputeCapability(
            options);
      },
      [](auto options, int32_t computeCapability) {
        setKernelTargetOrThrow(
            options, computeCapability,
            mlirTensorIRCompilerInvocationGetKernelArchPortability(options),
            "compute capability: " + std::to_string(computeCapability));
      });
  bindOption<MlirTensorIRArchPortability>(
      compileOptions, "arch_portability",
      [](auto options) {
        return mlirTensorIRCompilerInvocationGetKernelArchPortability(options);
      },
      [](auto options, MlirTensorIRArchPortability portability) {
        setKernelTargetOrThrow(
            options,
            mlirTensorIRCompilerInvocationGetKernelComputeCapability(options),
            portability, "arch portability");
      });

  // A version and a list have no C spelling among the typed accessors, so both
  // are reached through the invocation instead of a component handle.
  bindOption<PyBytecodeVersion>(
      compileOptions, "bytecode_version",
      [](auto options) {
        return PyBytecodeVersion{
            mlirTensorIRCompilerInvocationGetBytecodeVersion(options)};
      },
      [](auto options, PyBytecodeVersion version) {
        mlirTensorIRCompilerInvocationSetBytecodeVersion(options,
                                                         version.value);
      });
  bindOption<std::vector<int32_t>>(
      compileOptions, "tile_sizes",
      [](auto options) {
        size_t numSizes = 0;
        const int32_t *sizes =
            mlirTensorIRCompilerInvocationGetTileSizes(options, &numSizes);
        return std::vector<int32_t>(sizes, sizes + numSizes);
      },
      [](auto options, const std::vector<int32_t> &sizes) {
        mlirTensorIRCompilerInvocationSetTileSizes(options, sizes.data(),
                                                   sizes.size());
      });

#define TIR_BIND_OPTION(type, name, field)                                     \
  bindOption<type>(compileOptions, name,                                       \
                   mlirTensorIRCompilerInvocationGet##field,                   \
                   mlirTensorIRCompilerInvocationSet##field)
#define TIR_BIND_STRING_OPTION(name, field)                                    \
  bindStringOption(compileOptions, name,                                       \
                   mlirTensorIRCompilerInvocationGet##field,                   \
                   mlirTensorIRCompilerInvocationSet##field)
  TIR_BIND_OPTION(int32_t, "cta_count", CtaCount);
  TIR_BIND_OPTION(int32_t, "warp_count", WarpCount);
  TIR_BIND_OPTION(MlirTensorIRCodegenHostTarget, "host_target", HostTarget);
  TIR_BIND_OPTION(MlirTensorIRCodegenHostArch, "host_arch", HostArch);
  TIR_BIND_OPTION(MlirTensorIRCodegenHostOS, "host_os", HostOS);
  TIR_BIND_OPTION(int32_t, "max_tile_candidates", MaxTileCandidates);
  TIR_BIND_OPTION(MlirTensorIRCudaTileAutoGdcInsertion, "auto_gdc_insertion",
                  AutoGdcInsertion);
  TIR_BIND_OPTION(MlirTensorIRCudaTileCodegenStrategy, "codegen_strategy",
                  CodegenStrategy);
  TIR_BIND_OPTION(MlirTensorIRCudaTileArtifactKind, "artifact_kind",
                  ArtifactKind);
  TIR_BIND_OPTION(bool, "print_ir_after_all", PrintIRAfterAll);
  TIR_BIND_OPTION(bool, "enable_timing", EnableTiming);
  TIR_BIND_OPTION(bool, "print_cuda_tile_ir", PrintCudaTileIR);
  TIR_BIND_STRING_OPTION("cuda_toolkit_path", CudaToolkitPath);
  TIR_BIND_STRING_OPTION("dump_cuda_tile_ir_path", DumpCudaTileIRPath);
  TIR_BIND_STRING_OPTION("dump_tileir_bc_path", DumpTileIRBCPath);
  TIR_BIND_STRING_OPTION("load_tileir_bc_path", LoadTileIRBCPath);
  TIR_BIND_STRING_OPTION("print_ir_tree_dir", PrintIRTreeDir);
#undef TIR_BIND_STRING_OPTION
#undef TIR_BIND_OPTION

  nb::class_<PyRuntimeKernel>(m, "_RuntimeKernel")
      .def("check_support", &PyRuntimeKernel::checkSupport, nb::arg("args"))
      .def("query_workspace_size", &PyRuntimeKernel::queryWorkspaceSize,
           nb::arg("args"))
      .def("launch", &PyRuntimeKernel::launch, nb::arg("args"),
           nb::arg("workspace") = nb::none(), nb::arg("stream") = nb::none());

  nb::class_<PyArtifact>(m, "_Artifact")
      .def("load", &PyArtifact::load)
      .def("get_bytecode", &PyArtifact::getBytecode)
      .def("serialize", &PyArtifact::serialize)
      .def_static("deserialize", &PyArtifact::deserialize, nb::arg("bytes"));

  m.def(
      "_tensor_metadata_from_dlpack", &tensorMetadataFromDLPack,
      nb::arg("tensor"),
      "Return (shape, stride, dtype, pointer) metadata for a DLPack-compatible "
      "tensor.");

  m.def("calculate_cache_key", &calculateCacheKey, nb::arg("module"),
        nb::arg("options"),
        "Return the unhashed MLIR module-to-kernel cache key.");

  m.def(
      "compile",
      [](MlirModule module,
         const CompileOptions &compileOptions) -> std::unique_ptr<PyArtifact> {
        std::string error;
        MlirTensorIRArtifact artifact = mlirTensorIRArtifactCompile(
            validateModule(module), compileOptions.handle(), appendStringRef,
            &error);
        if (!artifact.ptr) {
          throwCAPIError("mlirTensorIRArtifactCompile", error);
        }
        return std::make_unique<PyArtifact>(artifact);
      },
      nb::arg("module"), nb::arg("options"));

  m.def(
      "analyze_module",
      [](MlirModule module, const CompileOptions &compileOptions) {
        std::vector<std::vector<int32_t>> candidates;
        std::string error;
        auto collect = [](const int32_t *shape, size_t rank, void *userData) {
          auto &result =
              *static_cast<std::vector<std::vector<int32_t>> *>(userData);
          result.emplace_back(shape, shape + rank);
        };
        checkCAPIResult(mlirTensorIRCudaTileAnalyzeModule(
                            validateModule(module), compileOptions.handle(),
                            collect, &candidates, appendStringRef, &error),
                        "mlirTensorIRCudaTileAnalyzeModule", error);
        return candidates;
      },
      nb::arg("module"), nb::arg("options"));

  m.def(
      "can_compile",
      [](MlirModule module, const CompileOptions &compileOptions) -> bool {
        bool canCompile = false;
        std::string error;
        checkCAPIResult(mlirTensorIRCanCompile(
                            validateModule(module), compileOptions.handle(),
                            &canCompile, appendStringRef, &error),
                        "mlirTensorIRCanCompile", error);
        return canCompile;
      },
      nb::arg("module"), nb::arg("options"));
}
