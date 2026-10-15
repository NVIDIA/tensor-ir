// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_LIB_RUNTIME_NATIVEHOSTMODULE_H
#define TENSOR_IR_LIB_RUNTIME_NATIVEHOSTMODULE_H

#include "tensor_ir/Artifact/TensorIRArtifact.h"
#include "tensor_ir/Runtime/HostRuntimeAbi.h"
#include "tensor_ir/Runtime/Types.h"

#include "llvm/ExecutionEngine/Orc/Core.h"

namespace tensor_ir::rt {

class NativeHostLoader;

/// Owns the executable mappings backing a native host entrypoint.
class NativeHostModule {
public:
  static StatusOr<std::unique_ptr<NativeHostModule>>
  load(const HostModule &host, llvm::StringRef entrySymbol);
  ~NativeHostModule();
  int32_t invoke(const TensorIRHostRuntimeV1 &runtime,
                 llvm::ArrayRef<int64_t> args) const {
    return entry_(&runtime, args.data(), args.size());
  }

private:
  NativeHostModule(std::shared_ptr<NativeHostLoader> loader,
                   llvm::orc::JITDylib &dylib);
  std::shared_ptr<NativeHostLoader> loader_;
  llvm::orc::JITDylib &dylib_;
  llvm::orc::ResourceTrackerSP tracker_;
  TensorIRHostEntryV1 entry_ = nullptr;
};

StatusOr<llvm::SmallVector<int64_t>>
packNativeHostArgs(const KernelMetadata &metadata, PackedArgs args);

} // namespace tensor_ir::rt
#endif
