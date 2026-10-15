// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Artifact/TensorIRArtifactCodec.h"

#include "tensor_ir/Artifact/Schema/Artifact_generated.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "flatbuffers/flatbuffers.h"

using mlir::FailureOr;
using mlir::nv_tensor_ir::ArchPortability;
using mlir::nv_tensor_ir::PersistenceMode;
using mlir::nv_tensor_ir::SmTarget;
using mlir::nv_tensor_ir::Status;
using mlir::nv_tensor_ir::StatusOr;

namespace tensor_ir::rt {

namespace {
namespace schema = ::tensor_ir::rt::schema;

constexpr uint32_t kTensorIRArtifactVersion = 3;

schema::ArchPortability toSchema(ArchPortability portability) {
  switch (portability) {
  case ArchPortability::portable:
    return schema::ArchPortability::Portable;
  case ArchPortability::family_portable:
    return schema::ArchPortability::FamilyPortable;
  case ArchPortability::arch_conditional:
    return schema::ArchPortability::ArchConditional;
  }
  llvm_unreachable("unrecognized ArchPortability");
}

schema::Persistence toSchema(PersistenceMode mode) {
  switch (mode) {
  case PersistenceMode::None:
    return schema::Persistence::None;
  case PersistenceMode::Static:
    return schema::Persistence::Static;
  }
  llvm_unreachable("unrecognized PersistenceMode");
}

StatusOr<ArchPortability> fromSchema(schema::ArchPortability portability) {
  switch (portability) {
  case schema::ArchPortability::Portable:
    return ArchPortability::portable;
  case schema::ArchPortability::FamilyPortable:
    return ArchPortability::family_portable;
  case schema::ArchPortability::ArchConditional:
    return ArchPortability::arch_conditional;
  }
  return Status::InvalidArgument(
      "serialized TensorIRArtifact has an invalid arch portability");
}

StatusOr<PersistenceMode> fromSchema(schema::Persistence persistence) {
  switch (persistence) {
  case schema::Persistence::None:
    return PersistenceMode::None;
  case schema::Persistence::Static:
    return PersistenceMode::Static;
  }
  return Status::InvalidArgument(
      "serialized TensorIRArtifact has an invalid persistence mode");
}

// schema::ElementType mirrors ElementType 1:1 (see Artifact.fbs), so this is
// total -- every ElementType has a schema counterpart.
schema::ElementType toSchema(ElementType type) {
  switch (type) {
  case ElementType::Bool:
    return schema::ElementType::Bool;
  case ElementType::F16:
    return schema::ElementType::F16;
  case ElementType::BF16:
    return schema::ElementType::BF16;
  case ElementType::F32:
    return schema::ElementType::F32;
  case ElementType::F64:
    return schema::ElementType::F64;
  case ElementType::F8E4M3FN:
    return schema::ElementType::F8E4M3FN;
  case ElementType::F8E5M2:
    return schema::ElementType::F8E5M2;
  case ElementType::SI8:
    return schema::ElementType::SI8;
  case ElementType::SI16:
    return schema::ElementType::SI16;
  case ElementType::SI32:
    return schema::ElementType::SI32;
  case ElementType::SI64:
    return schema::ElementType::SI64;
  case ElementType::UI8:
    return schema::ElementType::UI8;
  case ElementType::UI16:
    return schema::ElementType::UI16;
  case ElementType::UI32:
    return schema::ElementType::UI32;
  case ElementType::UI64:
    return schema::ElementType::UI64;
  }
  llvm_unreachable("unrecognized ElementType");
}

StatusOr<ElementType> fromSchema(schema::ElementType type) {
  switch (type) {
  case schema::ElementType::Bool:
    return ElementType::Bool;
  case schema::ElementType::F16:
    return ElementType::F16;
  case schema::ElementType::BF16:
    return ElementType::BF16;
  case schema::ElementType::F32:
    return ElementType::F32;
  case schema::ElementType::F64:
    return ElementType::F64;
  case schema::ElementType::F8E4M3FN:
    return ElementType::F8E4M3FN;
  case schema::ElementType::F8E5M2:
    return ElementType::F8E5M2;
  case schema::ElementType::SI8:
    return ElementType::SI8;
  case schema::ElementType::SI16:
    return ElementType::SI16;
  case schema::ElementType::SI32:
    return ElementType::SI32;
  case schema::ElementType::SI64:
    return ElementType::SI64;
  case schema::ElementType::UI8:
    return ElementType::UI8;
  case schema::ElementType::UI16:
    return ElementType::UI16;
  case schema::ElementType::UI32:
    return ElementType::UI32;
  case schema::ElementType::UI64:
    return ElementType::UI64;
  }
  return Status::InvalidArgument(
      "serialized TensorIRArtifact has an invalid element type");
}

StatusOr<ArgDesc> decodeArgDesc(schema::ArgDesc kind, const void *value) {
  switch (kind) {
  case schema::ArgDesc::TensorArgDesc: {
    const auto *tensorArg = static_cast<const schema::TensorArgDesc *>(value);
    TIR_ASSIGN_OR_RETURN(ElementType elementType,
                         fromSchema(tensorArg->element_type()));
    llvm::SmallVector<int64_t> staticShape;
    if (const auto *shape = tensorArg->shape()) {
      staticShape.assign(shape->begin(), shape->end());
    }
    std::optional<llvm::SmallVector<int64_t>> staticStrides;
    if (const auto *strides = tensorArg->strides()) {
      staticStrides.emplace(strides->begin(), strides->end());
    }
    TIR_ASSIGN_OR_RETURN(
        auto desc, TensorArgDesc::create(
                       elementType, staticShape,
                       staticStrides ? std::optional<llvm::ArrayRef<int64_t>>(
                                           *staticStrides)
                                     : std::nullopt));
    return ArgDesc(std::move(desc));
  }
  case schema::ArgDesc::ScalarArgDesc: {
    const auto *scalarArg = static_cast<const schema::ScalarArgDesc *>(value);
    TIR_ASSIGN_OR_RETURN(ElementType elementType,
                         fromSchema(scalarArg->element_type()));
    TIR_ASSIGN_OR_RETURN(auto desc, ScalarArgDesc::create(elementType));
    return ArgDesc(std::move(desc));
  }
  case schema::ArgDesc::NONE:
  default:
    return Status::InvalidArgument(
        "serialized TensorIRArtifact has a malformed argument descriptor");
  }
}

} // namespace

// Stays out of the anonymous namespace above so its name matches
// TensorIRArtifact.h's `friend class TensorIRArtifactEncoder;` declaration.

// Builds the TensorIRArtifact FlatBuffers table -- the only place
// TensorIRArtifact and its generated schema type meet. All nested pieces
// share one builder, passed in by the caller and held by reference here
// instead of threaded through every helper argument.
class TensorIRArtifactEncoder {
public:
  explicit TensorIRArtifactEncoder(::flatbuffers::FlatBufferBuilder &builder)
      : builder_(builder) {}

  ::flatbuffers::Offset<schema::TensorIRArtifact>
  encode(const TensorIRArtifact &artifact) {
    const KernelMetadata &argLayout = artifact.argLayout;

    llvm::SmallVector<::flatbuffers::Offset<void>> argOffsets;
    llvm::SmallVector<schema::ArgDesc> argKinds;
    argOffsets.reserve(argLayout.args.size());
    argKinds.reserve(argLayout.args.size());
    for (const ArgDesc &desc : argLayout.args) {
      schema::ArgDesc kind;
      argOffsets.push_back(encodeArgDesc(desc, kind));
      argKinds.push_back(kind);
    }
    auto args = builder_.CreateVector(argOffsets.data(), argOffsets.size());
    auto argsType = builder_.CreateVector(argKinds.data(), argKinds.size());
    auto tileSizes = builder_.CreateVector(argLayout.tileSizes.data(),
                                           argLayout.tileSizes.size());
    auto gridShape = builder_.CreateVector(argLayout.gridShape.data(),
                                           argLayout.gridShape.size());
    auto gridShapeDimMapping =
        builder_.CreateVector(argLayout.gridShapeDimMapping.data(),
                              argLayout.gridShapeDimMapping.size());

    std::optional<schema::GridDimensions> staticGridStruct;
    if (artifact.staticGrid) {
      staticGridStruct.emplace((*artifact.staticGrid)[0],
                               (*artifact.staticGrid)[1],
                               (*artifact.staticGrid)[2]);
    }

    auto metadata = schema::CreateKernelMetadata(
        builder_, builder_.CreateString(artifact.funcName), argsType, args,
        tileSizes, argLayout.numInputs, argLayout.gridShapeTensorIdx, gridShape,
        gridShapeDimMapping, toSchema(argLayout.persistence), argLayout.smCount,
        argLayout.occupancy, staticGridStruct ? &*staticGridStruct : nullptr,
        artifact.usesPDL_);

    // The current single-entrypoint ABI uses the same logical arguments as
    // the sole device kernel.
    auto kernelNameOffset = builder_.CreateString(artifact.kernelName);
    auto entrypoint = schema::CreateEntrypoint(
        builder_, builder_.CreateString(artifact.entrypointName),
        argLayout.numInputs, argsType, args);
    auto kernels = builder_.CreateVector(&metadata, 1);
    auto entrypoints = builder_.CreateVector(&entrypoint, 1);

    schema::SmTarget archStruct(artifact.arch.getComputeCapabilityVersion(),
                                toSchema(artifact.arch.getPortability()));

    schema::DeviceModule binaryType;
    ::flatbuffers::Offset<void> binaryOffset =
        encodeBinary(artifact.binary, binaryType);

    ::flatbuffers::Offset<schema::HostModule> hostModule;
    if (artifact.hostModule) {
      const auto &host = *artifact.hostModule;
      auto bytes = builder_.CreateVector(
          reinterpret_cast<const uint8_t *>(host.bytes.data()),
          host.bytes.size());
      hostModule = schema::CreateHostModule(builder_, host.abiVersion, bytes,
                                            builder_.CreateString(host.triple));
    }

    return schema::CreateTensorIRArtifact(
        builder_, kTensorIRArtifactVersion, kernelNameOffset, &archStruct,
        binaryType, binaryOffset, kernels, hostModule, entrypoints);
  }

private:
  ::flatbuffers::Offset<void> encodeBinary(const Binary &binary,
                                           schema::DeviceModule &binaryType) {
    return std::visit(
        [&](const auto &b) -> ::flatbuffers::Offset<void> {
          auto bytesOffset = builder_.CreateVector(
              reinterpret_cast<const uint8_t *>(b.bytes.data()),
              b.bytes.size());
          using T = std::decay_t<decltype(b)>;
          if constexpr (std::is_same_v<T, Cubin>) {
            binaryType = schema::DeviceModule::Cubin;
            return schema::CreateCubin(builder_, bytesOffset).Union();
          } else {
            static_assert(std::is_same_v<T, TileIRBytecode>);
            binaryType = schema::DeviceModule::TileIRBytecode;
            std::optional<schema::BcVersion> versionStruct;
            if (b.version) {
              versionStruct.emplace(b.version->getMajor(),
                                    b.version->getMinor(), b.version->getTag());
            }
            return schema::CreateTileIRBytecode(builder_, bytesOffset,
                                                versionStruct ? &*versionStruct
                                                              : nullptr)
                .Union();
          }
        },
        binary);
  }

  ::flatbuffers::Offset<void> encodeArgDesc(const ArgDesc &arg,
                                            schema::ArgDesc &kind) {
    schema::ElementType elementType = toSchema(getElementType(arg));
    if (std::holds_alternative<ScalarArgDesc>(arg)) {
      kind = schema::ArgDesc::ScalarArgDesc;
      return schema::CreateScalarArgDesc(builder_, elementType).Union();
    }
    const auto &desc = std::get<TensorArgDesc>(arg);
    kind = schema::ArgDesc::TensorArgDesc;
    auto staticShape =
        builder_.CreateVector(desc.shape().data(), desc.shape().size());
    // The strides vector's presence (null vs. non-null, not emptiness) is
    // the hasExplicitStrides signal (see decodeArgDesc).
    ::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> staticStrides;
    if (desc.hasExplicitStrides()) {
      staticStrides =
          builder_.CreateVector(desc.strides().data(), desc.strides().size());
    }
    return schema::CreateTensorArgDesc(builder_, elementType, staticShape,
                                       staticStrides)
        .Union();
  }

  ::flatbuffers::FlatBufferBuilder &builder_;
};

namespace {

StatusOr<Binary> decodeBinary(schema::DeviceModule kind, const void *value) {
  switch (kind) {
  case schema::DeviceModule::Cubin: {
    const auto *cubin = static_cast<const schema::Cubin *>(value);
    if (!cubin || !cubin->bytes()) {
      return Status::InvalidArgument(
          "serialized TensorIRArtifact is missing cubin bytes");
    }
    llvm::SmallVector<char, 0> bytes(cubin->bytes()->begin(),
                                     cubin->bytes()->end());
    return Binary(Cubin{std::move(bytes)});
  }
  case schema::DeviceModule::TileIRBytecode: {
    const auto *tileir = static_cast<const schema::TileIRBytecode *>(value);
    if (!tileir || !tileir->bytes()) {
      return Status::InvalidArgument(
          "serialized TensorIRArtifact is missing TileIR bytecode bytes");
    }
    llvm::SmallVector<char, 0> bytes(tileir->bytes()->begin(),
                                     tileir->bytes()->end());
    std::optional<::mlir::cuda_tile::BytecodeVersion> version;
    if (const schema::BcVersion *v = tileir->version()) {
      version = ::mlir::cuda_tile::BytecodeVersion::fromVersion(
          v->major_version(), v->minor_version(), v->tag());
      if (!version) {
        return Status::InvalidArgument(
            "serialized TensorIRArtifact has an unsupported bytecode "
            "version");
      }
    }
    return Binary(TileIRBytecode{std::move(bytes), version});
  }
  case schema::DeviceModule::NONE:
  default:
    return Status::InvalidArgument(
        "serialized TensorIRArtifact has a malformed binary");
  }
}

StatusOr<std::unique_ptr<TensorIRArtifact>>
decodeTensorIRArtifact(const schema::TensorIRArtifact &payload) {
  if (payload.version() != kTensorIRArtifactVersion) {
    return Status::NotSupported("serialized TensorIRArtifact version " +
                                std::to_string(payload.version()) +
                                " is not supported");
  }
  if (payload.kernels()->size() != 1 || payload.entrypoints()->size() != 1) {
    return Status::NotSupported(
        "serialized TensorIRArtifact must have exactly one kernel and one "
        "entrypoint");
  }
  std::optional<HostModule> hostModule;
  if (const auto *host = payload.host_module()) {
    hostModule = HostModule{llvm::SmallVector<char, 0>(host->bytes()->begin(),
                                                       host->bytes()->end()),
                            host->abi_version(), host->triple()->str()};
  }
  const schema::KernelMetadata *metadata = payload.kernels()->Get(0);
  const schema::Entrypoint *entrypoint = payload.entrypoints()->Get(0);
  if (entrypoint->name()->size() == 0) {
    return Status::InvalidArgument(
        "serialized TensorIRArtifact has an empty entrypoint name");
  }
  KernelMetadataBuilder builder;
  if (const auto *args = metadata->args()) {
    const auto *argTypes = metadata->args_type();
    builder.args.reserve(args->size());
    for (::flatbuffers::uoffset_t i = 0; i < args->size(); ++i) {
      TIR_ASSIGN_OR_RETURN(ArgDesc desc,
                           decodeArgDesc(argTypes->Get(i), args->Get(i)));
      builder.args.push_back(std::move(desc));
    }
  }
  if (const auto *tileSizes = metadata->tile_sizes()) {
    builder.tileSizes.assign(tileSizes->begin(), tileSizes->end());
  }
  builder.numInputs = metadata->num_inputs();
  builder.gridShapeTensorIdx = metadata->grid_shape_tensor_idx();
  if (const auto *gridShape = metadata->grid_shape()) {
    builder.gridShape.assign(gridShape->begin(), gridShape->end());
  }
  if (const auto *dimMapping = metadata->grid_shape_dim_mapping()) {
    builder.gridShapeDimMapping.assign(dimMapping->begin(), dimMapping->end());
  }
  TIR_ASSIGN_OR_RETURN(builder.persistence,
                       fromSchema(metadata->persistence()));
  builder.smCount = metadata->sm_count();
  builder.occupancy = metadata->occupancy();
  TIR_ASSIGN_OR_RETURN(KernelMetadata argLayout,
                       KernelMetadata::create(std::move(builder)));

  // The current runtime passes entrypoint arguments directly to the kernel.
  if (entrypoint->num_inputs() != argLayout.numInputs ||
      (entrypoint->args() ? entrypoint->args()->size() : 0) !=
          argLayout.args.size()) {
    return Status::InvalidArgument(
        "serialized TensorIRArtifact entrypoint and kernel signatures differ");
  }
  for (::flatbuffers::uoffset_t i = 0; i < argLayout.args.size(); ++i) {
    TIR_ASSIGN_OR_RETURN(ArgDesc desc,
                         decodeArgDesc(entrypoint->args_type()->Get(i),
                                       entrypoint->args()->Get(i)));
    if (!(desc == argLayout.args[i])) {
      return Status::InvalidArgument("serialized TensorIRArtifact entrypoint "
                                     "and kernel signatures differ");
    }
  }

  std::optional<std::array<int32_t, 3>> staticGrid;
  if (const schema::GridDimensions *grid = metadata->static_grid()) {
    staticGrid = {grid->x(), grid->y(), grid->z()};
  }

  const schema::SmTarget *archStruct = payload.arch();
  if (!archStruct) {
    return Status::InvalidArgument(
        "serialized TensorIRArtifact is missing its compute target");
  }
  TIR_ASSIGN_OR_RETURN(ArchPortability portability,
                       fromSchema(archStruct->arch_portability()));
  FailureOr<SmTarget> arch =
      SmTarget::fromCc(archStruct->compute_capability(), portability);
  if (failed(arch)) {
    return Status::InvalidArgument(
        "serialized TensorIRArtifact has an invalid compute target");
  }

  TIR_ASSIGN_OR_RETURN(Binary binary, decodeBinary(payload.device_module_type(),
                                                   payload.device_module()));

  std::string funcName = metadata->name()->str();
  std::string kernelName = payload.module_name()->str();

  return TensorIRArtifact::create(
      std::move(kernelName), std::move(funcName), std::move(binary), *arch,
      std::move(argLayout), staticGrid, metadata->uses_pdl(),
      std::move(hostModule), entrypoint->name()->str());
}

class FlatBufferMemoryBuffer final : public llvm::MemoryBuffer {
public:
  explicit FlatBufferMemoryBuffer(::flatbuffers::DetachedBuffer buffer)
      : buffer_(std::move(buffer)) {
    const char *begin = reinterpret_cast<const char *>(buffer_.data());
    init(begin, begin + buffer_.size(), /*RequiresNullTerminator=*/false);
  }

  BufferKind getBufferKind() const override { return MemoryBuffer_Malloc; }

  llvm::StringRef getBufferIdentifier() const override { return "Artifact"; }

private:
  ::flatbuffers::DetachedBuffer buffer_;
};

} // namespace

StatusOr<std::unique_ptr<TensorIRArtifact>>
deserializeArtifact(llvm::MemoryBufferRef bytes) {
  llvm::StringRef data = bytes.getBuffer();
  if (data.empty()) {
    return Status::InvalidArgument("cannot deserialize an empty buffer");
  }
  // VerifyTensorIRArtifactBuffer() below checks both the file identifier and
  // overall structure safely, without risking a short-buffer read.
  ::flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t *>(data.data()), data.size());
  if (!schema::VerifyTensorIRArtifactBuffer(verifier)) {
    return Status::InvalidArgument(
        "serialized artifact is not a valid FlatBuffer");
  }
  const schema::TensorIRArtifact *payload =
      schema::GetTensorIRArtifact(data.data());
  return decodeTensorIRArtifact(*payload);
}

StatusOr<std::unique_ptr<llvm::MemoryBuffer>>
serializeArtifact(const TensorIRArtifact &artifact) {
  ::flatbuffers::FlatBufferBuilder builder;
  auto payload = TensorIRArtifactEncoder(builder).encode(artifact);
  schema::FinishTensorIRArtifactBuffer(builder, payload);
  return std::unique_ptr<llvm::MemoryBuffer>(
      std::make_unique<FlatBufferMemoryBuffer>(builder.Release()));
}

Status saveArtifact(const TensorIRArtifact &artifact, llvm::raw_ostream &os) {
  StatusOr<std::unique_ptr<llvm::MemoryBuffer>> bytes =
      serializeArtifact(artifact);
  TIR_RETURN_IF_ERROR(bytes.status());
  os << (*bytes)->getBuffer();
  return Status::Ok();
}

Status saveArtifact(const TensorIRArtifact &artifact, llvm::StringRef path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    return Status::InvalidArgument("failed to open '" + path.str() +
                                   "' for writing: " + ec.message());
  }
  TIR_RETURN_IF_ERROR(saveArtifact(artifact, out));
  out.flush();
  if (out.has_error()) {
    // raw_fd_ostream's destructor treats an unacknowledged error as fatal;
    // clear it now that we've captured it in the returned Status.
    out.clear_error();
    return Status::InvalidArgument("failed to write to '" + path.str() + "'");
  }
  return Status::Ok();
}

StatusOr<std::unique_ptr<TensorIRArtifact>> loadArtifact(llvm::StringRef path) {
  // getFileOrSTDIN (not getFile): matches saveArtifact's path overload,
  // where raw_fd_ostream already treats "-" as stdout.
  auto bufferOrErr = llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (!bufferOrErr) {
    return Status::InvalidArgument(
        "failed to open '" + path.str() +
        "' for reading: " + bufferOrErr.getError().message());
  }
  return deserializeArtifact((*bufferOrErr)->getMemBufferRef());
}

} // namespace tensor_ir::rt
