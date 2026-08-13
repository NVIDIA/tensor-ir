// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"

#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#define DEBUG_TYPE "convert-tensor-to-cuda-tile"

namespace mlir::nv_tensor_ir {
#define GEN_PASS_DEF_TENSORTOCUDATILECONVERSIONPASS
#include "tensor_ir/Conversion/TensorToCudaTile/Passes.h.inc"
} // namespace mlir::nv_tensor_ir

using namespace mlir;
using namespace mlir::nv_tensor_ir;

namespace {

/// Type converter for transforming Tensor IR types to CUDA Tile types.
/// Note: Tensor IR operates on signed/unsigned integers, while CUDA Tile
/// operates on signless integers and signedness is defined by the operations.
class TensorToCudaTileTypeConverter : public TypeConverter {
public:
  TensorToCudaTileTypeConverter(MLIRContext *ctx) {
    // Pass-through conversion for CUDA Tile types.
    addConversion([=](cuda_tile::TileType tileTy) -> Type { return tileTy; });

    // Convert scalar float types to 0-D tiles.
    addConversion([=](FloatType floatTy) -> Type {
      return cuda_tile::TileType::get(/*shape=*/llvm::ArrayRef<int64_t>{},
                                      floatTy);
    });

    // Convert scalar integer types to 0-D tiles (signless).
    addConversion([=](IntegerType intTy) -> Type {
      if (intTy.isSignedInteger() || intTy.isUnsignedInteger()) {
        intTy = IntegerType::get(ctx, intTy.getIntOrFloatBitWidth(),
                                 IntegerType::SignednessSemantics::Signless);
      }
      return cuda_tile::TileType::get(/*shape=*/llvm::ArrayRef<int64_t>{},
                                      intTy);
    });

    // Convert tensor types to 0-D tiles, as the actual tile shape is not
    // available during the type conversion (depends on the iteration space).
    addConversion([=](nv_tensor_ir::TensorType tensorTy) -> Type {
      return convertType(tensorTy.getElementType());
    });
  }
};

/// Create the type converter that the TensorIR-to-CudaTile conversion uses.
/// File-local: the conversion pass below is the only construction site.
std::unique_ptr<TypeConverter>
createTensorToCudaTileTypeConverter(MLIRContext *context) {
  return std::make_unique<TensorToCudaTileTypeConverter>(context);
}

class TensorToCudaTileConversionPass
    : public nv_tensor_ir::impl::TensorToCudaTileConversionPassBase<
          TensorToCudaTileConversionPass> {

public:
  using nv_tensor_ir::impl::TensorToCudaTileConversionPassBase<
      TensorToCudaTileConversionPass>::TensorToCudaTileConversionPassBase;

  void runOnOperation() override {
    LLVM_DEBUG({
      llvm::dbgs() << "Running TensorToCudaTileConversionPass\n"
                   << "  num_ctas: " << num_ctas << "\n"
                   << "  occupancy: " << occupancy << "\n"
                   << "  num_warps: " << num_warps << "\n"
                   << "  persistence: "
                   << static_cast<int>(
                          static_cast<PersistenceMode>(persistence))
                   << "\n"
                   << "  sm_count: " << sm_count << "\n"
                   << "  reduction_tile_size: "
                   << static_cast<int64_t>(reduction_tile_size) << "\n"
                   << "  codegen_strategy: "
                   << static_cast<int>(static_cast<CudaTileCodegenStrategy>(
                          codegen_strategy))
                   << "\n"
                   << "  uniform_signature: " << uniform_signature << "\n";
    });

    // Verify the pass options and the IR state.
    // The error (if any) is emitted by the validation functions.
    if (failed(validatePassOptions()) || failed(validateModule())) {
      return signalPassFailure();
    }

    // Create the type converter (shared with the pre-flight feasibility check).
    MLIRContext *ctx = &getContext();
    std::unique_ptr<TypeConverter> typeConverterPtr =
        createTensorToCudaTileTypeConverter(ctx);
    TypeConverter &typeConverter = *typeConverterPtr;

    bool enableExperimentalCudaTileOps = false;

    // Create the main conversion state object.
    std::unique_ptr<tensor_to_cuda_tile::ConversionState> state;
    const bool useLayoutPropagation =
        codegen_strategy == CudaTileCodegenStrategy::LayoutPropagation;
    if (useLayoutPropagation) {
      auto optimizationHints =
          tensor_to_cuda_tile::createEntryOptimizationHints(
              ctx, num_ctas, occupancy, num_warps);
      state = tensor_to_cuda_tile::createLayoutPropagationConversionState(
          ctx, typeConverter, optimizationHints, uniform_signature,
          reduction_tile_size, enableExperimentalCudaTileOps);
    } else {
      TensorToCudaTilePipelineOptions options;
      options.tileSize.assign(tile_size.begin(), tile_size.end());
      options.numCTAs = num_ctas;
      options.occupancy = occupancy;
      options.numWarps = num_warps;
      options.smCount = sm_count;
      options.uniformSignature = uniform_signature;
      options.persistence = persistence;
      state = tensor_to_cuda_tile::createAffineMapConversionState(
          ctx, typeConverter, options);
    }

    // Preserve launch metadata for backends that run analysis and conversion
    // as a single pipeline. Conversion erases the GraphOp that carries both the
    // selected tile and normalized layout-propagation iteration space before
    // runtime launch metadata can read them.
    SmallVector<int32_t> resolvedTileSize;
    SmallVector<int64_t> resolvedIterationSpaceShape;

    // Convert all graph operations.
    auto result = getOperation()->walk([&](GraphOp graphOp) {
      LLVM_DEBUG(llvm::dbgs() << "Converting graph operation: "
                              << graphOp.getName() << "\n");
      if (useLayoutPropagation && resolvedIterationSpaceShape.empty()) {
        Operation *terminator = graphOp.getBody()->getTerminator();
        if (terminator) {
          if (auto iterationSpace =
                  terminator->getAttrOfType<LayoutSourceAttrInterface>(
                      TensorIRDialect::getIterationSpaceAttrName())) {
            auto shape = iterationSpace.getShape();
            resolvedIterationSpaceShape.assign(shape.begin(), shape.end());
          }
        }
      }
      if (failed(state->start(graphOp))) {
        return WalkResult::interrupt();
      }
      if (resolvedTileSize.empty()) {
        ArrayRef<int64_t> resolved = state->getResolvedTileSize();
        resolvedTileSize.reserve(resolved.size());
        for (int64_t tile : resolved) {
          resolvedTileSize.push_back(static_cast<int32_t>(tile));
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return signalPassFailure();
    }

    if (!resolvedTileSize.empty()) {
      getOperation()->setAttr(
          mlir::nv_tensor_ir::kResolvedTileSizeAttrName,
          DenseI32ArrayAttr::get(&getContext(), resolvedTileSize));
    }

    if (!resolvedIterationSpaceShape.empty()) {
      getOperation()->setAttr(
          mlir::nv_tensor_ir::kResolvedIterationSpaceShapeAttrName,
          DenseI64ArrayAttr::get(&getContext(), resolvedIterationSpaceShape));
    }

    // Create the CUDA Tile module and move converted functions into it.
    cuda_tile::ModuleOp newModuleOp = createModuleOp();
    Block &body = newModuleOp.getBody().front();

    LLVM_DEBUG(llvm::dbgs() << "Moving converted graphs into module: "
                            << newModuleOp.getSymName() << "\n");
    getOperation().walk([&](cuda_tile::EntryOp entryOp) {
      if (entryOp->getParentOp() != newModuleOp) {
        entryOp->moveBefore(&body, body.end());
      }
    });
  }

private:
  /// Verify the correctness of the pass options.
  /// If any of the options are not supported, emit an error and return failure.
  LogicalResult validatePassOptions() {
    auto loc = UnknownLoc::get(&getContext());

    if (num_ctas == 16) {
      return emitError(loc)
             << "num_ctas = 16 is currently not supported, requires driver to "
                "support `cudaFuncAttributeNonPortableClusterSizeAllowed`";
    }
    if (num_ctas < 1 || num_ctas > 8) {
      return emitError(loc)
             << "num_ctas must be in range [1, 8], got " << num_ctas;
    }
    if (occupancy < 1) {
      return emitError(loc) << "occupancy must be >= 1, got " << occupancy;
    }
    if (num_warps < 1) {
      return emitError(loc) << "num_warps must be >= 1, got " << num_warps;
    }
    if (codegen_strategy == CudaTileCodegenStrategy::LayoutPropagation &&
        (reduction_tile_size < 1 ||
         !llvm::isPowerOf2_64(static_cast<uint64_t>(reduction_tile_size)))) {
      return emitError(loc)
             << "reduction_tile_size must be a positive power of two, got "
             << static_cast<int64_t>(reduction_tile_size);
    }

    if (persistence == PersistenceMode::Static) {
      if (num_ctas > 1) {
        return emitError(loc)
               << "Static persistent kernels are incompatible with CGA "
                  "clusters (num_ctas > 1). Use num_ctas=1 for persistent "
                  "kernels or disable persistence for CGA kernels.";
      }
      if (sm_count < 1) {
        return emitError(loc)
               << "Static persistent kernels require sm_count >= 1, but got "
               << sm_count;
      }
    }

    return success();
  }

  /// Verify the assumptions about the IR state.
  LogicalResult validateModule() {
    mlir::ModuleOp op = getOperation();

    if (!op.getBodyRegion().hasOneBlock()) {
      return op->emitError("expect only one block in the module");
    }
    if (!op.getOps<cuda_tile::ModuleOp>().empty()) {
      return op->emitError("module has already been converted to CUDA Tile");
    }
    WalkResult result = op.walk([&](MatmulOp matmulOp) {
      if (!matmulOp.getAcc()) {
        return WalkResult::advance();
      }
      matmulOp.emitError(
          "custom accumulator is not supported by TensorToCudaTile");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      return failure();
    }

    return success();
  }

  /// Create a new `cuda_tile::ModuleOp` in the built-in module.
  cuda_tile::ModuleOp createModuleOp() {
    mlir::ModuleOp op = getOperation();

    // Derive the new module name from the built-in module name.
    auto tensorModuleName = op.getName();
    std::string cudaTileModuleName =
        tensorModuleName.has_value() ? "cuda_tile_" + tensorModuleName->str()
                                     : "cuda_tile_module";

    // Insert the new module at the end of the built-in module.
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPointToEnd(&op.getBodyRegion().front());
    return cuda_tile::ModuleOp::create(rewriter, op.getLoc(),
                                       cudaTileModuleName);
  }
};

} // namespace
