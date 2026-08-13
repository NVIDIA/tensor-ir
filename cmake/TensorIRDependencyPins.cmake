# Source revisions used by the standalone TensorIR build.
#
# Update the CUDA Tile and LLVM revisions together. The LLVM revision must match
# the compatibility pin for the selected CUDA Tile revision.

set(TENSOR_IR_PINNED_CUDA_TILE_COMMIT
    "af2417041cc939b87ef56d92cfdcf61737c5457e")
set(TENSOR_IR_PINNED_CUDA_TILE_ARCHIVE_SHA256
    "81597e49469171bf8fa7319fbd44ebe133001521f484589e3dd3fb3fad282dc0")
set(TENSOR_IR_PINNED_LLVM_COMMIT "57109befac92811d2253109242ca6fa69c961fb2")
set(TENSOR_IR_PINNED_LLVM_ARCHIVE_SHA256
    "725cc90dc8221d9f8fd66da076c0a129441ec54f081eb7fbe029b6a8d9171e94")

# DLPack is independent of the CUDA Tile and LLVM compatibility pair.
set(TENSOR_IR_PINNED_DLPACK_COMMIT "6ea9b3eb64c881f614cd4537f95f0e125a35555c")
set(TENSOR_IR_PINNED_DLPACK_ARCHIVE_SHA256
    "c87782e5edd06ce2f1c88841e8fcde7b88aa65fe6158be74337bd93eaa468331")
