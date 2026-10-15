# Source revisions used by the standalone TensorIR build.
#
# Update the CUDA Tile and LLVM revisions together. The LLVM revision must match
# the compatibility pin for the selected CUDA Tile revision.

# CUDA Tile v13.4.0 and its cmake/IncludeLLVM.cmake compatibility pin.
set(TENSOR_IR_PINNED_CUDA_TILE_COMMIT
    "7e8e2e68fa219716103824c01f7303367cf7df8d")
set(TENSOR_IR_PINNED_CUDA_TILE_ARCHIVE_SHA256
    "2398f88dd0dbcda3732934c8527485c3941d3a6453c98b6c249461e6309ea44c")
set(TENSOR_IR_PINNED_LLVM_COMMIT "9ebb067a8a2b4b0705f06d59c77d36dfab98333f")
set(TENSOR_IR_PINNED_LLVM_ARCHIVE_SHA256
    "84f6624d9f6a32e180d7cb3b91d9e47b3f51da7f1f79ef216614dbff8b928145"
    CACHE STRING "LLVM source archive SHA256")

# DLPack is independent of the CUDA Tile and LLVM compatibility pair.
set(TENSOR_IR_PINNED_DLPACK_COMMIT "6ea9b3eb64c881f614cd4537f95f0e125a35555c")
set(TENSOR_IR_PINNED_DLPACK_ARCHIVE_SHA256
    "c87782e5edd06ce2f1c88841e8fcde7b88aa65fe6158be74337bd93eaa468331")

# FlatBuffers is independent of the CUDA Tile and LLVM compatibility pair.
set(TENSOR_IR_PINNED_FLATBUFFERS_COMMIT "v24.3.25")
set(TENSOR_IR_PINNED_FLATBUFFERS_ARCHIVE_SHA256
    "4157c5cacdb59737c5d627e47ac26b140e9ee28b1102f812b36068aab728c1ed")
