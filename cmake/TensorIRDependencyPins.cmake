# Source revisions used by the standalone TensorIR build.
#
# Validate the CUDA Tile and LLVM revisions together with the standalone build
# and test suite. LLVM must provide FP8 cast expansion for CPU execution.

# CUDA Tile and its LLVM compatibility pin.
set(TENSOR_IR_PINNED_CUDA_TILE_COMMIT
    "e11729ed4e228af005dc8cc113d7b84157e9d654")
set(TENSOR_IR_PINNED_CUDA_TILE_ARCHIVE_SHA256
    "56c03d732ac5f23e0ae9c0c184ff65aa282c24bd30a1bde723457f8e986e855e")
set(TENSOR_IR_PINNED_LLVM_COMMIT "80017ba18d8c8866713586fa19245f0878b4f8a2")
set(TENSOR_IR_PINNED_LLVM_ARCHIVE_SHA256
    "d3d6ae9c477c47a8efaf17d76419cd80d06eb53c8bf6d65ac46054be3551aed9"
    CACHE STRING "LLVM source archive SHA256")

# DLPack is independent of the CUDA Tile and LLVM compatibility pair.
set(TENSOR_IR_PINNED_DLPACK_COMMIT "6ea9b3eb64c881f614cd4537f95f0e125a35555c")
set(TENSOR_IR_PINNED_DLPACK_ARCHIVE_SHA256
    "c87782e5edd06ce2f1c88841e8fcde7b88aa65fe6158be74337bd93eaa468331")

# FlatBuffers is independent of the CUDA Tile and LLVM compatibility pair.
set(TENSOR_IR_PINNED_FLATBUFFERS_COMMIT "v24.3.25")
set(TENSOR_IR_PINNED_FLATBUFFERS_ARCHIVE_SHA256
    "4157c5cacdb59737c5d627e47ac26b140e9ee28b1102f812b36068aab728c1ed")
