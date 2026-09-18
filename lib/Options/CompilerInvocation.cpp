// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Options/Options.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

namespace mlir::nv_tensor_ir {

FailureOr<ComputeCapability>
CompilerInvocation::getCodegenComputeCapability() const {
  FailureOr<SmTarget> kernelTarget = getKernelSmTarget();
  if (failed(kernelTarget)) {
    return failure();
  }
  if (deviceTarget.codegenTarget == 0) {
    return kernelTarget->getComputeCapability();
  }
  std::optional<ComputeCapability> codegenTarget =
      symbolizeComputeCapability(deviceTarget.codegenTarget);
  if (!codegenTarget || !kernelTarget->validateCodegenTargetCompatibility(
                            deviceTarget.codegenTarget)) {
    return failure();
  }
  return *codegenTarget;
}

std::string CompilerInvocation::toShaderNameSuffix() const {
  CompilerInvocation cacheOptions = *this;
  cacheOptions.debug = {};
  if (FailureOr<ComputeCapability> codegenTarget =
          getCodegenComputeCapability();
      succeeded(codegenTarget)) {
    cacheOptions.deviceTarget.codegenTarget = toCcInt(*codegenTarget);
  }
  std::string serialized = cacheOptions.toString();
  std::string suffix;
  llvm::raw_string_ostream os(suffix);
  os << deviceTarget.kernelTarget << "_opts_"
     << llvm::format_hex_no_prefix(
            llvm::xxh3_64bits(llvm::StringRef(serialized)), 16);
  return os.str();
}

Status CompilerInvocation::validate() const {
  FailureOr<SmTarget> target = getKernelSmTarget();
  if (failed(target)) {
    return Status::InvalidArgument("Invalid target SM '" +
                                   deviceTarget.kernelTarget + "'");
  }

  if (auto error = validateComponents()) {
    return Status::InvalidArgument(*error);
  }

  FailureOr<ComputeCapability> codegenTarget = getCodegenComputeCapability();
  if (failed(codegenTarget)) {
    return Status::InvalidArgument("Codegen compute capability " +
                                   std::to_string(deviceTarget.codegenTarget) +
                                   " is incompatible with kernel target '" +
                                   deviceTarget.kernelTarget + "'");
  }

  if (cudaTile.persistenceMode == PersistenceMode::Static &&
      cudaTile.smCount <= 0) {
    return Status::InvalidArgument(
        "Static persistent kernels require smCount > 0, but got " +
        std::to_string(cudaTile.smCount));
  }

  if (cudaTile.artifactKind != ArtifactKind::TileIR &&
      cudaTile.artifactKind != ArtifactKind::Cubin) {
    return Status::InvalidArgument("Unrecognized CUDA Tile artifact kind");
  }
  if (cudaTile.artifactKind == ArtifactKind::Cubin &&
      target->getPortability() != ArchPortability::arch_conditional) {
    return Status::InvalidArgument("Cubin requires an arch-conditional target");
  }

  return Status::Ok();
}

} // namespace mlir::nv_tensor_ir
