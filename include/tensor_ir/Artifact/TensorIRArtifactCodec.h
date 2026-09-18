// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TensorIRArtifactCodec - serializes/deserializes a TensorIRArtifact to/from
// a single self-describing FlatBuffer.

#ifndef TENSOR_IR_ARTIFACT_TENSORIRARTIFACTCODEC_H
#define TENSOR_IR_ARTIFACT_TENSORIRARTIFACTCODEC_H

#include "tensor_ir/Artifact/TensorIRArtifact.h"
#include "tensor_ir/Runtime/Types.h"
#include "tensor_ir/Support/Status.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace tensor_ir::rt {

/// The returned TensorIRArtifact owns its own copies of every field; it does
/// not retain a reference to `bytes`, which can be freed as soon as this
/// returns.
StatusOr<std::unique_ptr<TensorIRArtifact>>
deserializeArtifact(llvm::MemoryBufferRef bytes);

StatusOr<std::unique_ptr<llvm::MemoryBuffer>>
serializeArtifact(const TensorIRArtifact &artifact);

Status saveArtifact(const TensorIRArtifact &artifact, llvm::raw_ostream &os);

/// `path == "-"` writes to stdout instead of opening a file.
Status saveArtifact(const TensorIRArtifact &artifact, llvm::StringRef path);

/// `path == "-"` reads from stdin instead of opening a file.
StatusOr<std::unique_ptr<TensorIRArtifact>> loadArtifact(llvm::StringRef path);

} // namespace tensor_ir::rt

#endif // TENSOR_IR_ARTIFACT_TENSORIRARTIFACTCODEC_H
