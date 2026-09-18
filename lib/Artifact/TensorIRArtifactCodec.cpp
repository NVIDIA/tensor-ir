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

constexpr uint32_t kTensorIRArtifactVersion = 1;

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

StatusOr<ElementTypeInfo> decodeElementType(schema::ElementType wireType) {
  TIR_ASSIGN_OR_RETURN(ElementType type, fromSchema(wireType));
  return ElementTypeInfo(type);
}

StatusOr<TensorArgDesc> decodeArgDesc(schema::ArgDesc kind, const void *value) {
  switch (kind) {
  case schema::ArgDesc::TensorArgDesc: {
    const auto *tensorArg = static_cast<const schema::TensorArgDesc *>(value);
    TIR_ASSIGN_OR_RETURN(ElementTypeInfo elementInfo,
                         decodeElementType(tensorArg->element_type()));
    llvm::SmallVector<int64_t> staticShape;
    if (const auto *shape = tensorArg->shape()) {
      staticShape.assign(shape->begin(), shape->end());
    }
    int32_t rank = static_cast<int32_t>(staticShape.size());
    const auto *strides = tensorArg->strides();
    llvm::SmallVector<int64_t> staticStrides;
    bool hasExplicitStrides = strides != nullptr;
    if (strides) {
      staticStrides.assign(strides->begin(), strides->end());
    }
    return TensorArgDesc{rank,
                         std::move(staticShape),
                         std::move(staticStrides),
                         /*numDynSizes=*/0,
                         /*numDynStrides=*/0,
                         hasExplicitStrides,
                         /*isScalar=*/false,
                         elementInfo};
  }
  case schema::ArgDesc::ScalarArgDesc: {
    const auto *scalarArg = static_cast<const schema::ScalarArgDesc *>(value);
    TIR_ASSIGN_OR_RETURN(ElementTypeInfo elementInfo,
                         decodeElementType(scalarArg->element_type()));
    return TensorArgDesc{/*rank=*/0,
                         /*staticShape=*/{},
                         /*staticStrides=*/{},
                         /*numDynSizes=*/0,
                         /*numDynStrides=*/0,
                         /*hasExplicitStrides=*/false,
                         /*isScalar=*/true,
                         elementInfo};
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
    const KernelArgLayout &argLayout = artifact.argLayout;

    llvm::SmallVector<::flatbuffers::Offset<void>> argOffsets;
    llvm::SmallVector<schema::ArgDesc> argKinds;
    argOffsets.reserve(argLayout.tensorDescs.size());
    argKinds.reserve(argLayout.tensorDescs.size());
    for (const TensorArgDesc &desc : argLayout.tensorDescs) {
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

    auto metadata = schema::CreateLaunchMetadata(
        builder_, builder_.CreateString(artifact.funcName), argsType, args,
        tileSizes, argLayout.numInputs, argLayout.gridShapeTensorIdx, gridShape,
        gridShapeDimMapping, argLayout.uniformSignature,
        toSchema(argLayout.persistence), argLayout.smCount, argLayout.occupancy,
        staticGridStruct ? &*staticGridStruct : nullptr);

    auto kernelNameOffset = builder_.CreateString(artifact.kernelName);

    schema::SmTarget archStruct(artifact.arch.getComputeCapabilityVersion(),
                                toSchema(artifact.arch.getPortability()));

    schema::Binary binaryType;
    ::flatbuffers::Offset<void> binaryOffset =
        encodeBinary(artifact.binary, binaryType);

    return schema::CreateTensorIRArtifact(builder_, kTensorIRArtifactVersion,
                                          kernelNameOffset, &archStruct,
                                          binaryType, binaryOffset, metadata);
  }

private:
  ::flatbuffers::Offset<void> encodeBinary(const Binary &binary,
                                           schema::Binary &binaryType) {
    return std::visit(
        [&](const auto &b) -> ::flatbuffers::Offset<void> {
          auto bytesOffset = builder_.CreateVector(
              reinterpret_cast<const uint8_t *>(b.bytes.data()),
              b.bytes.size());
          using T = std::decay_t<decltype(b)>;
          if constexpr (std::is_same_v<T, Cubin>) {
            binaryType = schema::Binary::Cubin;
            return schema::CreateCubin(builder_, bytesOffset).Union();
          } else {
            static_assert(std::is_same_v<T, TileIRBytecode>);
            binaryType = schema::Binary::TileIRBytecode;
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

  ::flatbuffers::Offset<void> encodeArgDesc(const TensorArgDesc &desc,
                                            schema::ArgDesc &kind) {
    schema::ElementType elementType = toSchema(desc.elementInfo.type());
    if (desc.isScalar) {
      kind = schema::ArgDesc::ScalarArgDesc;
      return schema::CreateScalarArgDesc(builder_, elementType).Union();
    }
    kind = schema::ArgDesc::TensorArgDesc;
    auto staticShape =
        builder_.CreateVector(desc.staticShape.data(), desc.staticShape.size());
    // The strides vector's presence (null vs. non-null, not emptiness) is
    // the hasExplicitStrides signal (see decodeArgDesc).
    ::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> staticStrides;
    if (desc.hasExplicitStrides) {
      staticStrides = builder_.CreateVector(desc.staticStrides.data(),
                                            desc.staticStrides.size());
    }
    return schema::CreateTensorArgDesc(builder_, elementType, staticShape,
                                       staticStrides)
        .Union();
  }

  ::flatbuffers::FlatBufferBuilder &builder_;
};

namespace {

StatusOr<Binary> decodeBinary(schema::Binary kind, const void *value) {
  switch (kind) {
  case schema::Binary::Cubin: {
    const auto *cubin = static_cast<const schema::Cubin *>(value);
    if (!cubin || !cubin->bytes()) {
      return Status::InvalidArgument(
          "serialized TensorIRArtifact is missing cubin bytes");
    }
    llvm::SmallVector<char, 0> bytes(cubin->bytes()->begin(),
                                     cubin->bytes()->end());
    return Binary(Cubin{std::move(bytes)});
  }
  case schema::Binary::TileIRBytecode: {
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
          v->major(), v->minor(), v->tag());
      if (!version) {
        return Status::InvalidArgument(
            "serialized TensorIRArtifact has an unsupported bytecode "
            "version");
      }
    }
    return Binary(TileIRBytecode{std::move(bytes), version});
  }
  case schema::Binary::NONE:
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
  const schema::LaunchMetadata *metadata = payload.metadata();
  KernelArgLayout argLayout;
  if (const auto *args = metadata->args()) {
    const auto *argTypes = metadata->args_type();
    argLayout.tensorDescs.reserve(args->size());
    for (::flatbuffers::uoffset_t i = 0; i < args->size(); ++i) {
      TIR_ASSIGN_OR_RETURN(TensorArgDesc desc,
                           decodeArgDesc(argTypes->Get(i), args->Get(i)));
      argLayout.tensorDescs.push_back(std::move(desc));
    }
  }
  if (const auto *tileSizes = metadata->tile_sizes()) {
    argLayout.tileSizes.assign(tileSizes->begin(), tileSizes->end());
  }
  argLayout.numInputs = metadata->num_inputs();
  argLayout.gridShapeTensorIdx = metadata->grid_shape_tensor_idx();
  if (const auto *gridShape = metadata->grid_shape()) {
    argLayout.gridShape.assign(gridShape->begin(), gridShape->end());
  }
  if (const auto *dimMapping = metadata->grid_shape_dim_mapping()) {
    argLayout.gridShapeDimMapping.assign(dimMapping->begin(),
                                         dimMapping->end());
  }
  argLayout.uniformSignature = metadata->uniform_signature();
  for (TensorArgDesc &desc : argLayout.tensorDescs) {
    if (desc.isScalar) {
      continue;
    }
    desc.numDynSizes = argLayout.uniformSignature
                           ? desc.rank
                           : countDynamicDims(desc.staticShape);
    desc.numDynStrides = desc.hasExplicitStrides
                             ? (argLayout.uniformSignature
                                    ? desc.rank
                                    : countDynamicDims(desc.staticStrides))
                             : 0;
  }
  argLayout.totalKernelArgs = 0;
  for (const TensorArgDesc &desc : argLayout.tensorDescs) {
    argLayout.totalKernelArgs += desc.totalArgs();
  }
  TIR_ASSIGN_OR_RETURN(argLayout.persistence,
                       fromSchema(metadata->persistence()));
  argLayout.smCount = metadata->sm_count();
  argLayout.occupancy = metadata->occupancy();

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

  TIR_ASSIGN_OR_RETURN(Binary binary,
                       decodeBinary(payload.binary_type(), payload.binary()));

  std::string funcName = metadata->func_name()->str();
  std::string kernelName = payload.kernel_name()->str();

  return TensorIRArtifact::create(std::move(kernelName), std::move(funcName),
                                  std::move(binary), *arch,
                                  std::move(argLayout), staticGrid);
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
