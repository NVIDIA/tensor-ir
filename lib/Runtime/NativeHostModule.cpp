// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "NativeHostModule.h"

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/TargetParser/Host.h"

#include <atomic>
#include <limits>

namespace tensor_ir::rt {

// Shared by all modules, with no platform plugins or process-symbol search.
// Our objects have no imports and never unwind across the host ABI.
class NativeHostLoader {
public:
  NativeHostLoader() {
    auto memory = llvm::jitlink::InProcessMemoryManager::Create();
    if (!memory) {
      error = llvm::toString(memory.takeError());
      return;
    }
    auto control = llvm::orc::SelfExecutorProcessControl::Create();
    if (!control) {
      error = llvm::toString(control.takeError());
      return;
    }
    session =
        std::make_unique<llvm::orc::ExecutionSession>(std::move(*control));
    layer = std::make_unique<llvm::orc::ObjectLinkingLayer>(*session,
                                                            std::move(*memory));
  }
  ~NativeHostLoader() {
    if (session) {
      if (auto error = session->endSession()) {
        session->reportError(std::move(error));
      }
    }
  }

  std::unique_ptr<llvm::orc::ExecutionSession> session;
  std::unique_ptr<llvm::orc::ObjectLinkingLayer> layer;
  std::atomic<uint64_t> nextId{0};
  std::string error;
};

NativeHostModule::NativeHostModule(std::shared_ptr<NativeHostLoader> loader,
                                   llvm::orc::JITDylib &dylib)
    : loader_(std::move(loader)), dylib_(dylib),
      tracker_(dylib.createResourceTracker()) {}
NativeHostModule::~NativeHostModule() {
  if (auto error = tracker_->remove()) {
    loader_->session->reportError(std::move(error));
  }
  if (auto error = loader_->session->removeJITDylib(dylib_)) {
    loader_->session->reportError(std::move(error));
  }
}

StatusOr<std::unique_ptr<NativeHostModule>>
NativeHostModule::load(const HostModule &host, llvm::StringRef entrySymbol) {
  if (host.abiVersion != TENSOR_IR_HOST_ABI_VERSION_1) {
    return Status::NotSupported("Unsupported native host ABI version");
  }
  llvm::Triple process(llvm::sys::getProcessTriple());
  llvm::Triple target(host.triple);
  // ELF does not encode the calling convention's OS. Check persisted metadata
  // before inspecting the object's architecture and container.
  if (target.getArch() != process.getArch() ||
      target.getOS() != process.getOS() ||
      (process.isOSWindows() &&
       target.getEnvironment() != process.getEnvironment()) ||
      target.getObjectFormat() != llvm::Triple::ELF) {
    return Status::NotSupported(
        "Native host target triple is incompatible with this process");
  }
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(host.bytes.data(), host.bytes.size()), entrySymbol);
  auto object =
      llvm::object::ObjectFile::createObjectFile(buffer->getMemBufferRef());
  if (!object) {
    return Status::InvalidArgument("Invalid native host object: " +
                                   llvm::toString(object.takeError()));
  }
  if (!(*object)->isRelocatableObject() ||
      (*object)->getArch() != target.getArch() || !(*object)->isELF() ||
      (*object)->isLittleEndian() != process.isLittleEndian()) {
    return Status::NotSupported(
        "Native host object is incompatible with this process");
  }
  // Keep the import allow-list empty, including for externally supplied objects.
  for (auto symbol : (*object)->symbols()) {
    auto flags = symbol.getFlags();
    if (!flags) {
      return Status::InvalidArgument(llvm::toString(flags.takeError()));
    }
    if (*flags & llvm::object::SymbolRef::SF_Undefined) {
      return Status::NotSupported("Native host object has unsupported imports");
    }
  }
  static auto loader = std::make_shared<NativeHostLoader>();
  if (!loader->error.empty()) {
    return Status::NotSupported("Cannot create native host loader: " +
                                loader->error);
  }
  auto &dylib = loader->session->createBareJITDylib(
      "tensor_ir_host_" + std::to_string(loader->nextId++));
  auto result =
      std::unique_ptr<NativeHostModule>(new NativeHostModule(loader, dylib));
  if (auto error = loader->layer->add(result->tracker_, std::move(buffer))) {
    return Status::InvalidArgument("Cannot load native host object: " +
                                   llvm::toString(std::move(error)));
  }
  auto entry = loader->session->lookup({&dylib}, entrySymbol);
  if (!entry) {
    return Status::NotFound("Cannot resolve native host entry: " +
                            llvm::toString(entry.takeError()));
  }
  result->entry_ = entry->getAddress().toPtr<TensorIRHostEntryV1>();
  return result;
}

StatusOr<llvm::SmallVector<int64_t>>
packNativeHostArgs(const KernelMetadata &metadata, PackedArgs args) {
  if (args.size() < 0 ||
      static_cast<size_t>(args.size()) != metadata.args.size()) {
    return Status::InvalidArgument("Native host argument count mismatch");
  }
  llvm::SmallVector<int64_t> packed;
  for (auto [i, argument] : llvm::enumerate(metadata.args)) {
    const Any *value = args[i];
    if (!value) {
      return Status::InvalidArgument(
          "Native host argument index out of bounds");
    }
    if (const auto *scalarDesc = std::get_if<ScalarArgDesc>(&argument)) {
      if (value->typeIndex != TypeIndex::kScalar ||
          value->scalar.type != scalarDesc->elementType) {
        return Status::InvalidArgument("Native host scalar type mismatch");
      }
      packed.push_back(value->scalar.bits);
      continue;
    }
    const auto *desc = std::get_if<TensorArgDesc>(&argument);
    if (!desc) {
      return Status::InvalidArgument("Unsupported native host argument kind");
    }
    const TensorView *view =
        value->typeIndex == TypeIndex::kTensor ? &value->tensorView : nullptr;
    if ((!view && value->typeIndex != TypeIndex::kPointer) ||
        (view &&
         (view->ndim != desc->rank() || (view->ndim && !view->shape)))) {
      return Status::InvalidArgument(
          "Native host tensor rank or type mismatch");
    }
    packed.push_back(
        reinterpret_cast<intptr_t>(view ? view->data : value->ptr));
    llvm::SmallVector<int64_t> shape, strides(desc->rank());
    for (int d = 0; d < desc->rank(); ++d) {
      int64_t size = view ? view->shape[d] : desc->shape()[d];
      if (size < 0 || (desc->shape()[d] != TensorArgDesc::kDynamic &&
                       size != desc->shape()[d])) {
        return Status::InvalidArgument("Native host tensor shape mismatch");
      }
      shape.push_back(size);
    }
    int64_t contiguousStride = 1;
    for (int d = desc->rank() - 1; d >= 0; --d) {
      int64_t stride =
          view && view->stride
              ? view->stride[d]
              : (!view && desc->hasExplicitStrides() ? desc->strides()[d]
                                                     : contiguousStride);
      // Singleton dimensions never advance the address, so their strides may
      // differ from the compiled layout (for example after a reshape).
      if (stride < 0 ||
          (shape[d] != 1 &&
           ((desc->hasExplicitStrides() &&
             desc->strides()[d] != TensorArgDesc::kDynamic &&
             stride != desc->strides()[d]) ||
            (!desc->hasExplicitStrides() && stride != contiguousStride)))) {
        return Status::InvalidArgument("Native host tensor stride mismatch");
      }
      strides[d] = stride;
      if (d > 0) {
        if (shape[d] &&
            contiguousStride > std::numeric_limits<int64_t>::max() / shape[d]) {
          return Status::InvalidArgument("Native host tensor stride overflow");
        }
        contiguousStride *= shape[d];
      }
    }
    packed.append(shape);
    packed.append(strides);
  }
  if (packed.size() > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument(
        "Native host packed argument count overflow");
  }
  return packed;
}
} // namespace tensor_ir::rt
