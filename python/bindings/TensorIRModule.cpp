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

#include "dlpack/dlpack.h"
#include "tensor_ir-c/TensorIR.h"
#include <Python.h>
#include <cstdint>
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

std::string formatBytecodeVersion(const PyBytecodeVersion &bytecodeVersion) {
  std::string formatted;
  mlirTensorIRFormatBytecodeVersion(bytecodeVersion.value, appendStringRef,
                                    &formatted);
  return formatted;
}

struct CompileOptions {
  int32_t computeCapability = 100;
  MlirTensorIRArchPortability archPortability =
      MlirTensorIRArchPortabilityFamilyPortable;
  int32_t ctaCount = 1;
  int32_t warpCount = 4;
  std::vector<int32_t> tileSizes;
  bool uniformSignature = false;
  MlirTensorIRCudaTileCodegenStrategy codegenStrategy =
      MlirTensorIRCudaTileCodegenStrategyLayoutPropagation;
  int32_t maxTileCandidates = 1;
  std::string dumpCudaTileIRPath;
  std::string dumpTileIRBCPath;
  std::string loadTileIRBCPath;
  std::string printIRTreeDir;
  bool printIRAfterAll = false;
  bool enableTiming = false;
  PyBytecodeVersion bytecodeVersion{mlirTensorIRGetDefaultBytecodeVersion()};
  MlirTensorIRCudaTileArtifactKind artifactKind =
      MlirTensorIRCudaTileArtifactKindTileIR;
};

MlirStringRef wrapStringRef(const std::string &value) {
  return mlirStringRefCreate(value.data(), value.size());
}

MlirTensorIRCudaTileCompileOptions
wrapCompileOptions(const CompileOptions &options) {
  MlirTensorIRCudaTileCompileOptions cOptions{};
  cOptions.computeCapability = options.computeCapability;
  cOptions.archPortability = options.archPortability;
  cOptions.ctaCount = options.ctaCount;
  cOptions.warpCount = options.warpCount;
  cOptions.tileSizes = options.tileSizes.data();
  cOptions.numTileSizes = options.tileSizes.size();
  cOptions.uniformSignature = options.uniformSignature;
  cOptions.codegenStrategy = options.codegenStrategy;
  cOptions.maxTileCandidates = options.maxTileCandidates;
  cOptions.dumpCudaTileIRPath = wrapStringRef(options.dumpCudaTileIRPath);
  cOptions.dumpTileIRBCPath = wrapStringRef(options.dumpTileIRBCPath);
  cOptions.loadTileIRBCPath = wrapStringRef(options.loadTileIRBCPath);
  cOptions.printIRTreeDir = wrapStringRef(options.printIRTreeDir);
  cOptions.printIRAfterAll = options.printIRAfterAll;
  cOptions.enableTiming = options.enableTiming;
  cOptions.bytecodeVersion = options.bytecodeVersion.value;
  cOptions.artifactKind = options.artifactKind;
  return cOptions;
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

MlirModule validateModule(MlirModule module) {
  MlirOperation operation = mlirModuleGetOperation(module);
  MlirModule moduleFromOperation = mlirModuleFromOperation(operation);
  if (mlirModuleIsNull(moduleFromOperation)) {
    throw nb::type_error("expected an MLIR builtin.module");
  }
  return moduleFromOperation;
}

bool supportsDLPack(nb::handle value) {
  return nb::hasattr(value, "__dlpack__") &&
         nb::hasattr(value, "__dlpack_device__");
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

MlirTensorIRArgument packDLPackTensorArg(nb::handle value,
                                         PackedPythonArgs &packed) {
  DLTensor *tensor = keepDLPackTensorAlive(value, packed);
  appendDLPackTensorView(*tensor, packed);

  std::vector<int64_t> &shape = packed.shapes.back();
  std::vector<int64_t> &stride = packed.strides.back();
  char *data = static_cast<char *>(tensor->data) + tensor->byte_offset;
  MlirTensorIRArgument argument{};
  argument.kind = MlirTensorIRArgumentKindTensor;
  argument.value.tensor = {data, static_cast<int32_t>(shape.size()),
                           shape.data(), stride.data()};
  return argument;
}

MlirTensorIRArgument packTensorArg(nb::handle value, PackedPythonArgs &packed) {
  if (supportsDLPack(value)) {
    return packDLPackTensorArg(value, packed);
  }
  throw nb::type_error(("TensorIR tensor arguments must be DLPack-compatible "
                        "tensors with __dlpack__ and __dlpack_device__ (got " +
                        pythonTypeName(value) + ")")
                           .c_str());
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
    } else if (nb::isinstance<nb::int_>(arg)) {
      argument.kind = MlirTensorIRArgumentKindInt64;
      argument.value.int64Value = nb::cast<int64_t>(arg);
    } else if (nb::isinstance<nb::float_>(arg)) {
      argument.kind = MlirTensorIRArgumentKindFloat64;
      argument.value.float64Value = nb::cast<double>(arg);
    } else {
      argument = packTensorArg(arg, packed);
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
  return nb::make_tuple(shape, stride, tensorIRTypeFromDLPack(tensor->dtype));
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

class PyProgram {
public:
  explicit PyProgram(MlirTensorIRProgram program) : program_(program) {}

  PyProgram(const PyProgram &) = delete;
  PyProgram &operator=(const PyProgram &) = delete;
  PyProgram(PyProgram &&) = delete;
  PyProgram &operator=(PyProgram &&) = delete;
  ~PyProgram() { mlirTensorIRProgramDelete(program_); }

  void destroy() { mlirTensorIRProgramDestroy(program_); }

  bool isDestroyed() const { return mlirTensorIRProgramIsDestroyed(program_); }

  bool isInitialized() const {
    return mlirTensorIRProgramIsInitialized(program_);
  }

  void initialize() {
    std::string error;
    checkCAPIResult(
        mlirTensorIRProgramInitialize(program_, appendStringRef, &error),
        "mlirTensorIRProgramInitialize", error);
  }

  bool checkSupport(nb::tuple pyArgs) {
    PackedPythonArgs args = packArgs(pyArgs);
    MlirTensorIRPackedArgs packedArgs = args.packed();
    bool supported = false;
    std::string error;
    checkCAPIResult(mlirTensorIRProgramCheckSupport(program_, packedArgs,
                                                    &supported, appendStringRef,
                                                    &error),
                    "mlirTensorIRProgramCheckSupport", error);
    return supported;
  }

  size_t queryWorkspaceSize(nb::tuple pyArgs) {
    PackedPythonArgs args = packArgs(pyArgs);
    MlirTensorIRPackedArgs packedArgs = args.packed();
    size_t workspaceSize = 0;
    std::string error;
    checkCAPIResult(
        mlirTensorIRProgramQueryWorkspaceSize(
            program_, packedArgs, &workspaceSize, appendStringRef, &error),
        "mlirTensorIRProgramQueryWorkspaceSize", error);
    return workspaceSize;
  }

  void launch(nb::tuple pyArgs, nb::object workspace, nb::object stream) {
    PackedPythonArgs args = packArgs(pyArgs);
    MlirTensorIRPackedArgs packedArgs = args.packed();
    PackedWorkspace packedWorkspace = packWorkspace(std::move(workspace), args);
    void *packedStream = packStream(std::move(stream));
    std::string error;
    checkCAPIResult(
        mlirTensorIRProgramLaunch(program_, packedArgs, packedWorkspace.data,
                                  packedWorkspace.size, packedStream,
                                  appendStringRef, &error),
        "mlirTensorIRProgramLaunch", error);
  }

  nb::bytes getBytecode() const {
    std::string bytecode;
    std::string error;
    checkCAPIResult(mlirTensorIRProgramGetBytecode(program_, appendStringRef,
                                                   &bytecode, appendStringRef,
                                                   &error),
                    "mlirTensorIRProgramGetBytecode", error);
    return nb::bytes(bytecode.data(), bytecode.size());
  }

  std::string repr() const {
    if (isDestroyed()) {
      return "Program(state='destroyed')";
    }
    return isInitialized() ? "Program(state='initialized')"
                           : "Program(state='uninitialized')";
  }

private:
  MlirTensorIRProgram program_;
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

  nb::class_<CompileOptions>(m, "CompileOptions")
      .def(nb::init<>())
      .def_rw("compute_capability", &CompileOptions::computeCapability)
      .def_rw("arch_portability", &CompileOptions::archPortability)
      .def_rw("cta_count", &CompileOptions::ctaCount)
      .def_rw("warp_count", &CompileOptions::warpCount)
      .def_rw("tile_sizes", &CompileOptions::tileSizes)
      .def_rw("uniform_signature", &CompileOptions::uniformSignature)
      .def_rw("codegen_strategy", &CompileOptions::codegenStrategy)
      .def_rw("max_tile_candidates", &CompileOptions::maxTileCandidates)
      .def_rw("dump_cuda_tile_ir_path", &CompileOptions::dumpCudaTileIRPath)
      .def_rw("dump_tileir_bc_path", &CompileOptions::dumpTileIRBCPath)
      .def_rw("load_tileir_bc_path", &CompileOptions::loadTileIRBCPath)
      .def_rw("print_ir_tree_dir", &CompileOptions::printIRTreeDir)
      .def_rw("print_ir_after_all", &CompileOptions::printIRAfterAll)
      .def_rw("enable_timing", &CompileOptions::enableTiming)
      .def_rw("bytecode_version", &CompileOptions::bytecodeVersion)
      .def_rw("artifact_kind", &CompileOptions::artifactKind);

  nb::class_<PyProgram>(m, "_Program")
      .def("destroy", &PyProgram::destroy)
      .def("initialize", &PyProgram::initialize)
      .def("check_support", &PyProgram::checkSupport, nb::arg("args"))
      .def("query_workspace_size", &PyProgram::queryWorkspaceSize,
           nb::arg("args"))
      .def("launch", &PyProgram::launch, nb::arg("args"),
           nb::arg("workspace") = nb::none(), nb::arg("stream") = nb::none())
      .def("get_bytecode", &PyProgram::getBytecode)
      .def_prop_ro("is_destroyed", &PyProgram::isDestroyed)
      .def_prop_ro("is_initialized", &PyProgram::isInitialized)
      .def("__repr__", &PyProgram::repr);

  m.def("_tensor_metadata_from_dlpack", &tensorMetadataFromDLPack,
        nb::arg("tensor"),
        "Return (shape, stride, dtype) metadata for a DLPack-compatible "
        "tensor.");

  m.def(
      "compile",
      [](MlirModule module,
         const CompileOptions &compileOptions) -> std::unique_ptr<PyProgram> {
        MlirTensorIRCudaTileCompileOptions cOptions =
            wrapCompileOptions(compileOptions);
        std::string error;
        MlirTensorIRProgram program = mlirTensorIRProgramCompile(
            validateModule(module), cOptions, appendStringRef, &error);
        if (!program.ptr) {
          throwCAPIError("mlirTensorIRProgramCompile", error);
        }
        return std::make_unique<PyProgram>(program);
      },
      nb::arg("module"), nb::arg("options"));

  m.def(
      "can_compile",
      [](MlirModule module, const CompileOptions &compileOptions) -> bool {
        MlirTensorIRCudaTileCompileOptions cOptions =
            wrapCompileOptions(compileOptions);
        bool canCompile = false;
        std::string error;
        checkCAPIResult(mlirTensorIRProgramCanCompile(validateModule(module),
                                                      cOptions, &canCompile,
                                                      appendStringRef, &error),
                        "mlirTensorIRProgramCanCompile", error);
        return canCompile;
      },
      nb::arg("module"), nb::arg("options"));
}
