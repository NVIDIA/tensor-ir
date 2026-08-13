// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"
#include "tensor_ir/Support/TCutegen.h"
#include "tensor_ir/Utils/Utils.h"

#include "llvm/Support/Debug.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace tcg = mlir::nv_tensor_ir::tcutegen;

//===----------------------------------------------------------------------===//
// Tensor IR to CUDA Tile conversion implementation using layout propagation.
//
// The state interface implementation is reset for each graph conversion.
// For each graph:
// 1. `normalizedLayout` is set from the "iteration_space" attribute.
// 2. `tileShape` (in the default iteration space) is set from the "tile_size"
//     attribute.
// 3. `inputDescriptors` and `outputDescriptors` are set from the graph function
//     signature attributes (SSA values are undefined at this point).
// 4. `applyFullConversion` is called to convert the graph operations.
//
// The `GraphOp` gets converted first:
// 1. `cuda_tile::EntryOp` is created and the graph region is moved into it.
// 2. `applySignatureConversion` is called to update the block arguments.
// 3.  At this point, the SSA values for the block arguments are defined.
//     Update the input and output descriptors: set tensor pointers values,
//     dynamic sizes values and (if explicitly set) dynamic strides values.
//     If the strides are missing, calculate them from the sizes assuming
//     row-major ordering (may involve computation for dynamic sizes).
// 4. `cuda_tile::GetTileBlockIdOp` is created and the linear index is
//     transformed into dimensional indices (used for loads and stores).
// 5. `buildSkeleton` is called to build the code blocks for the operations
//     that modify the iteration space (concatenation, reduction, matmul).
// 6.  For each generated iteration space, emit the tile loads for all the
//     tensor sources in that iteration space.
//
// The rest of the operations are converted by the dialect conversion engine
// in the topological order. The adaptor operands are the tiles produced by
// the previous operations, except for block arguments which are replaced with
// the loaded tiles.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "convert-tensor-to-cuda-tile-layout-propagation"

namespace mlir::nv_tensor_ir {
namespace tensor_to_cuda_tile {
namespace {

class ConversionStateImpl : public ConversionState {
public:
  ConversionStateImpl(MLIRContext *context, const TypeConverter &typeConverter,
                      cuda_tile::OptimizationHintsAttr optimizationHints,
                      bool uniformSignature, int64_t reductionTileSize,
                      bool enableExperimentalCudaTileOps)
      : typeConverter(typeConverter), conversionTarget(*context),
        optimizationHints(optimizationHints),
        uniformSignature(uniformSignature),
        reductionTileSize(reductionTileSize),
        enableExperimentalCudaTileOps(enableExperimentalCudaTileOps) {
    conversionTarget.addLegalDialect<cuda_tile::CudaTileDialect>();
    patterns = registerPatterns(context);
  }

  /// Prepare the conversion state for a graph conversion.
  LogicalResult start(GraphOp graphOp) override;

  /// Prepare the conversion state for an operation conversion.
  LogicalResult update(ConversionPatternRewriter &rewriter,
                       Operation *op) override;

  /// Get the tile for an operand.
  Value getTile(Value operand) override;

  /// Get the tile shape for the current iteration space.
  ArrayRef<int64_t> getTileShape() override {
    return iterationSpace->tileShape;
  }

  ArrayRef<Value> getIndexValues() const override {
    if (iterationSpace) {
      return iterationSpace->indexValues;
    }
    return {};
  }

  /// Get the resolved tile size for launch metadata synchronization.
  ArrayRef<int64_t> getResolvedTileSize() override { return getTileShape(); }

  /// Get the block structure for the current operation.
  const BlockStructure *getBlockStructure() const override {
    return blockStructure;
  }

private:
  const TypeConverter &typeConverter;
  ConversionTarget conversionTarget;
  FrozenRewritePatternSet patterns;

  /// Optimization hints (common for all graphs).
  cuda_tile::OptimizationHintsAttr optimizationHints;

  /// When `uniformSignature` is true, all the dimension sizes are passed to the
  /// kernel as arguments, even for statically-known dimensions (such values
  /// are ignored).
  bool uniformSignature;

  int64_t reductionTileSize;

  bool enableExperimentalCudaTileOps;

  /// Normalized layout for the graph output.
  LayoutSourceAttrInterface normalizedLayout;

  /// Tensor descriptors for the graph inputs and outputs.
  SmallVector<TensorDescriptor> inputDescriptors;
  SmallVector<TensorDescriptor> outputDescriptors;

  /// Zero index value for the broadcasted dimensions.
  Value zeroIndex;

  /// Keep track of all block structures generated by the ops that modify the
  /// iteration space. This is needed as some iteration spaces have zero
  /// operations, i.e. the ID never appears in the graph.
  DenseMap<Operation *, BlockStructure> blockStructures;

  /// Mapping from iteration space ID to the iteration space pointer.
  DenseMap<int, IterationSpace *> iterationSpaces;

  /// Block structure pointer is set for ops that modify the iteration space.
  const BlockStructure *blockStructure = nullptr;
  /// Iteration space pointer is set for every operation.
  const IterationSpace *iterationSpace = nullptr;

  /// Layouts for the current operation's operands.
  using OperandLayouts = SmallVector<LayoutSourceAttrInterface>;
  OperandLayouts layouts;
  /// Index of the next operand to process, calling `getTile` updates this.
  size_t operandIndex = 0;

  /// Build the signature conversion for the graph function.
  /// Every tensor argument becomes a pointer, followed by size/stride arguments
  /// for dynamic dimensions (all dimensions, if `uniformSignature` is set).
  FailureOr<TypeConverter::SignatureConversion>
  buildSignatureConversion(FunctionType funcTy) const;

  /// Conversion pattern registration will be moved out of the state class
  /// once the refactoring is complete.
  RewritePatternSet registerPatterns(MLIRContext *context);

  /// Allow access to the conversion state from the conversion patterns.
  friend class GraphOpConversion;
  friend class ResultsOpConversion;
};

template <typename OpTy>
class NoOpConversion : public ConversionPattern<OpTy> {
  using Base = ConversionPattern<OpTy>;
  using Base::ConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(this->state.update(rewriter, op));
    Value tile = this->state.getTile(adaptor.getInput());

    rewriter.replaceOp(op, tile);
    return success();
  }
};

using ReshapeOpConversion = NoOpConversion<ReshapeOp>;
using BroadcastOpConversion = NoOpConversion<BroadcastOp>;
using TransposeOpConversion = NoOpConversion<TransposeOp>;
using SliceOpConversion = NoOpConversion<SliceOp>;

static Value broadcastAlongDimension(ConversionPatternRewriter &rewriter,
                                     Location loc, Value tile,
                                     ArrayRef<int64_t> targetShape,
                                     size_t dim) {
  auto tileTy = cast<ShapedType>(tile.getType());
  if (tileTy.getShape() == targetShape) {
    return tile;
  }
  SmallVector<int64_t> unitShape(targetShape.size(), 1);
  unitShape[dim] = tileTy.getShape()[0];
  auto targetTy = tileTy.clone(targetShape);
  Value reshaped = cuda_tile::ReshapeOp::create(rewriter, loc,
                                                tileTy.clone(unitShape), tile);
  if (cast<ShapedType>(reshaped.getType()) == targetTy) {
    return reshaped;
  }
  return cuda_tile::BroadcastOp::create(rewriter, loc, targetTy, reshaped);
}

//===----------------------------------------------------------------------===//
// IotaOp conversion
//===----------------------------------------------------------------------===//

/// Lowers `IotaOp` using propagated layout strides and block indices.
class IotaOpConversion : public ConversionPattern<IotaOp> {
public:
  using ConversionPattern<IotaOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(IotaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validateIotaOpLowerable(op, state.getTileShape()));

    ArrayRef<int64_t> tileShape = state.getTileShape();
    ArrayRef<Value> indexValues = state.getIndexValues();
    assert(!indexValues.empty() &&
           "layout-propagation codegen must provide iteration-space indices");
    assert(indexValues.size() == tileShape.size() &&
           "iteration-space index rank must match tile shape rank");

    auto layoutSource =
        cast<TensorSourceAttr>(op->getAttrOfType<LayoutSourceAttrInterface>(
            TensorIRDialect::getLayoutAttrName()));
    auto cuteLayout = layoutSource.getCuteLayout();
    const auto &cgStride = cuteLayout.stride();
    size_t rank = tcg::rank(cgStride);

    Location loc = op.getLoc();
    Type resultElemType = cast<TensorType>(op.getType()).getElementType();
    Type convertedElemType =
        cast<ShapedType>(getTypeConverter()->convertType(resultElemType))
            .getElementType();
    Type iotaElemType =
        resultElemType.isFloat() ? rewriter.getI32Type() : convertedElemType;

    auto fullTileType = cuda_tile::TileType::get(tileShape, convertedElemType);
    auto intTileType = fullTileType.clone(iotaElemType);

    auto scalarIndexType = cast<ShapedType>(indexValues[0].getType());

    Value blockSum;
    for (size_t dim = 0; dim < rank; ++dim) {
      int64_t stride = tcg::static_size(cgStride, dim);
      int64_t scale = tileShape[dim] * stride;
      if (scale == 0) {
        continue;
      }
      Value term = cuda_tile::MulIOp::create(
          rewriter, loc, indexValues[dim],
          createConstant(rewriter, loc, scalarIndexType, scale));
      blockSum = blockSum
                     ? cuda_tile::AddIOp::create(rewriter, loc, blockSum, term)
                     : term;
    }

    if (layoutSource.getOffset() != 0) {
      Value offsetVal = createConstant(rewriter, loc, scalarIndexType,
                                       layoutSource.getOffset());
      blockSum = blockSum ? cuda_tile::AddIOp::create(rewriter, loc, blockSum,
                                                      offsetVal)
                          : offsetVal;
    }

    Value result;
    if (blockSum) {
      SmallVector<int64_t> unitShape(tileShape.size(), 1);
      auto unitType = cast<ShapedType>(intTileType).clone(unitShape);
      blockSum =
          cuda_tile::ReshapeOp::create(rewriter, loc, unitType, blockSum);
      result =
          cuda_tile::BroadcastOp::create(rewriter, loc, intTileType, blockSum);
    } else {
      result = createConstant(rewriter, loc, intTileType, int64_t{0});
    }

    for (size_t dim = 0; dim < rank; ++dim) {
      int64_t stride = tcg::static_size(cgStride, dim);
      if (stride == 0) {
        continue;
      }
      int64_t dimTile = tileShape[dim];

      auto iota1DType =
          cuda_tile::TileType::get(ArrayRef<int64_t>{dimTile}, iotaElemType);
      Value localIota = cuda_tile::IotaOp::create(rewriter, loc, iota1DType);
      Value scaled =
          stride == 1
              ? localIota
              : cuda_tile::MulIOp::create(
                    rewriter, loc, localIota,
                    createConstant(rewriter, loc, cast<ShapedType>(iota1DType),
                                   stride));
      Value ranked =
          broadcastAlongDimension(rewriter, loc, scaled, tileShape, dim);
      result = cuda_tile::AddIOp::create(rewriter, loc, result, ranked);
    }

    if (resultElemType.isFloat()) {
      auto floatTileType = fullTileType.clone(resultElemType);
      result = cuda_tile::IToFOp::create(rewriter, loc, floatTileType, result,
                                         cuda_tile::Signedness::Unsigned,
                                         cuda_tile::RoundingMode::NEAREST_EVEN);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GraphOp conversion
//===----------------------------------------------------------------------===//

class GraphOpConversion : public ConversionPattern<GraphOp> {
public:
  using ConversionPattern<GraphOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(GraphOp graphOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &state = static_cast<ConversionStateImpl &>(this->state);

    // Convert the function signature.
    auto originalFuncType = graphOp.getFunctionType();
    auto signatureConversion = state.buildSignatureConversion(originalFuncType);
    if (failed(signatureConversion)) {
      return graphOp.emitError("Failed to convert graph signature");
    }
    auto convertedFuncType =
        FunctionType::get(rewriter.getContext(),
                          /*inputs=*/signatureConversion->getConvertedTypes(),
                          /*results=*/{});
    LLVM_DEBUG({
      llvm::dbgs() << "Original function type: " << originalFuncType << "\n";
      llvm::dbgs() << "Converted function type: " << convertedFuncType << "\n";
    });

    // Create the new function and move the graph region into it.
    auto entryOp = cuda_tile::EntryOp::create(
        rewriter, graphOp.getLoc(), graphOp.getName(), convertedFuncType,
        /*arg_attrs=*/{}, /*res_attrs=*/{}, state.optimizationHints);
    rewriter.inlineRegionBefore(graphOp.getRegion(), entryOp.getBody(),
                                entryOp.getBody().end());

    // Apply the signature conversion to update block arguments.
    Block *entryBlock = rewriter.applySignatureConversion(
        &entryOp.getBody().front(), *signatureConversion);
    rewriter.setInsertionPointToStart(entryBlock);

    // Update SSA values in the tensor descriptors.
    // Compute default strides if they're not explicitly set.
    ArrayRef<BlockArgument> args = entryBlock->getArguments();
    for (TensorDescriptor &inputDesc : state.inputDescriptors) {
      updateDescriptorValues(inputDesc, args, state.uniformSignature);
      setDefaultStrides(rewriter, inputDesc);
    }
    for (TensorDescriptor &outputDesc : state.outputDescriptors) {
      updateDescriptorValues(outputDesc, args, state.uniformSignature);
      setDefaultStrides(rewriter, outputDesc);
    }
    assert(args.empty() && "expected all arguments to be processed");

    // Create the zero index value.
    ShapedType indexType = cuda_tile::TileType::get(
        /*shape=*/llvm::ArrayRef<int64_t>{}, rewriter.getI32Type());
    state.zeroIndex =
        createConstant(rewriter, graphOp.getLoc(), indexType, int64_t{0});

    // Build virtual descriptor for the output tensor.
    // Dynamic dimensions sizes follow the tensor pointer (block argument).
    TensorDescriptor outputDesc;
    outputDesc.pointer = state.outputDescriptors[0].pointer;
    size_t argIdx = cast<BlockArgument>(outputDesc.pointer).getArgNumber();

    for (int64_t size : state.normalizedLayout.getShape()) {
      Value dynamicSize = size == ShapedType::kDynamic
                              ? entryBlock->getArgument(++argIdx)
                              : nullptr;
      outputDesc.sizes.push_back({size, dynamicSize});
      outputDesc.strides.push_back({1, nullptr});
    }

    // Build index values for the current tile.
    auto tileBlockIds =
        cuda_tile::GetTileBlockIdOp::create(rewriter, graphOp.getLoc());
    SmallVector<int64_t> tileShape =
        state.blockStructures[nullptr].iterationSpaces.front().tileShape;
    MLIR_ASSIGN_OR_RETURN(SmallVector<Value> indexValues,
                          calculateIndex(rewriter, tileBlockIds.getResult(0),
                                         outputDesc, tileShape));

    // Build the block structures and the iteration spaces.
    IterationSpace initial{
        entryBlock, std::move(tileShape), std::move(indexValues), {}};
    MLIR_ASSIGN_OR_RETURN(state.blockStructures,
                          buildSkeleton(rewriter, std::move(initial),
                                        *getTypeConverter(),
                                        state.reductionTileSize));

    // Process all the generated block structures.
    for (auto &[op, blockStructure] : state.blockStructures) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Block structure for " << op->getName() << "\n");

      // Process all the iteration spaces in the block structure.
      for (IterationSpace &iterationSpace : blockStructure.iterationSpaces) {
        LLVM_DEBUG({
          llvm::dbgs() << "Iteration space: (";
          llvm::interleaveComma(iterationSpace.tileShape, llvm::dbgs());
          llvm::dbgs() << ")\n";
        });

        // Create loads for the tensor sources in the iteration space.
        rewriter.setInsertionPointToEnd(iterationSpace.insertionBlock);
        for (auto &[source, loadedTile] : iterationSpace.loadedTiles) {
          LLVM_DEBUG({
            llvm::dbgs() << "Loading tensor #" << source.getTensorId()
                         << " layout=" << source.getLayout()
                         << " offset=" << source.getOffset() << "\n";
          });

          // Verify that the tensor ID is valid and refers to an input tensor.
          size_t tensorId = source.getTensorId();
          if (tensorId >= state.inputDescriptors.size() ||
              state.inputDescriptors[tensorId].sizes.empty()) {
            return graphOp.emitError("Invalid tensor ID: ") << tensorId;
          }

          // Emit load and store the loaded tile.
          auto layoutDesc =
              applyLayout(rewriter, state.inputDescriptors[tensorId], source);
          loadedTile = emitLoadMaybeBroadcast(rewriter, layoutDesc,
                                              iterationSpace, state.zeroIndex);
        }
      }

      // Record the iteration-space-id to iteration-space mapping from the
      // operand-to-iteration-space relation stored on the block structure
      // (`buildSkeleton` derived it once for every owner op). An operand with
      // no space (a concatenate operand pruned by a slice) is skipped, so a
      // pruned operand can never index past the iteration spaces that exist.
      for (auto [operandIdx, operand] : llvm::enumerate(op->getOperands())) {
        auto definingOp = operand.getDefiningOp();
        if (!definingOp) {
          continue;
        }
        auto iterSpaceId = definingOp->getAttrOfType<IntegerAttr>(
            TensorIRDialect::getIterSpaceIdAttrName());
        if (!iterSpaceId) {
          continue;
        }
        int spaceIdx = blockStructure.iterationSpaceIndexForOperand[operandIdx];
        if (spaceIdx == BlockStructure::kNoIterationSpace) {
          continue;
        }
        LLVM_DEBUG(llvm::dbgs() << "Adding iteration space ID "
                                << iterSpaceId.getInt() << "\n");
        IterationSpace *space = &blockStructure.iterationSpaces[spaceIdx];

        // Defensively reject a conflicting assignment: an iteration-space id
        // must never resolve to two different iteration spaces; fail with a
        // clear error rather than silently overwrite.
        auto [it, inserted] =
            state.iterationSpaces.try_emplace(iterSpaceId.getInt(), space);
        if (!inserted && it->second != space) {
          return op->emitError()
                 << "Conflicting iteration spaces for iteration space id "
                 << iterSpaceId.getInt();
        }
      }
    }

    // Replace `GraphOp` with `cuda_tile::EntryOp`.
    rewriter.replaceOp(graphOp, entryOp);
    return success();
  }

private:
  /// Load tile and optionally broadcast to the tile shape.
  static Value emitLoadMaybeBroadcast(ConversionPatternRewriter &rewriter,
                                      const TensorDescriptor &desc,
                                      const IterationSpace &iterationSpace,
                                      Value zeroIndex) {
    // Infer the tile type for the result.
    Type ptrType = cast<ShapedType>(desc.pointer.getType()).getElementType();
    Type elemType = cast<cuda_tile::PointerType>(ptrType).getPointeeType();
    auto tileType =
        cuda_tile::TileType::get(iterationSpace.tileShape, elemType);

    // Collect the broadcasted dimensions.
    SmallVector<size_t> broadcastDims;
    for (auto [idx, stride] : llvm::enumerate(desc.strides)) {
      if (stride.staticValue == 0) {
        broadcastDims.push_back(idx);
      }
    }

    // If there are no broadcasted dimensions, use simple load.
    if (broadcastDims.empty()) {
      return emitLoad(rewriter, desc, tileType, iterationSpace.indexValues);
    }

    // Update tile shape and index values for broadcasted dimensions.
    SmallVector<int64_t> tileShape = iterationSpace.tileShape;
    SmallVector<Value> indexValues = iterationSpace.indexValues;
    for (size_t dim : broadcastDims) {
      tileShape[dim] = 1;
      indexValues[dim] = zeroIndex;
    }

    // Emit load and broadcast.
    auto loadType = cast<ShapedType>(tileType).clone(tileShape);
    Value result = emitLoad(rewriter, desc, loadType, indexValues);
    if (loadType != tileType) {
      result = cuda_tile::BroadcastOp::create(rewriter, result.getLoc(),
                                              tileType, result);
    }
    return result;
  }

  /// Update SSA values in the tensor descriptor (pointer and dynamic sizes).
  /// Advance the arguments slice by the number of processed values.
  static void updateDescriptorValues(TensorDescriptor &desc,
                                     ArrayRef<BlockArgument> &args,
                                     bool uniformSignature) {
    // Set the pointer value.
    assert(!args.empty() && "missing pointer argument");
    desc.pointer = args[0];

    // Set the dynamic size values.
    size_t argIdx = 1;
    for (auto &size : desc.sizes) {
      if (size.staticValue == ShapedType::kDynamic) {
        assert(argIdx < args.size() && "missing dynamic size argument");
        size.dynamicValue = args[argIdx++];
      } else if (uniformSignature) {
        ++argIdx;
      }
    }

    // Set the dynamic stride values (if explicitly set).
    for (auto &stride : desc.strides) {
      if (stride.staticValue == ShapedType::kDynamic) {
        assert(argIdx < args.size() && "missing dynamic stride argument");
        stride.dynamicValue = args[argIdx++];
      } else if (uniformSignature) {
        ++argIdx;
      }
    }
    if (desc.strides.empty() && uniformSignature) {
      argIdx += desc.sizes.size();
    }

    // `argIdx` is the total number of arguments, including the pointer.
    LLVM_DEBUG({
      llvm::dbgs() << "Argument at position " << args[0].getArgNumber();
      if (argIdx == 1) {
        llvm::dbgs() << " is static\n";
      } else {
        llvm::dbgs() << " has " << (argIdx - 1) << " extra block args\n";
      }
    });

    // Update the arguments slice (advance).
    args = args.drop_front(argIdx);
  }

  /// Calculate the strides for tensor descriptor, if not explicitly set.
  /// TensorIR uses row-major as the default layout:
  ///   for tensor dimension sizes: [D0, D1, ..., Dn]
  ///   the calculated strides are: [D1*...*Dn, ..., Dn, 1]
  static void setDefaultStrides(ConversionPatternRewriter &rewriter,
                                TensorDescriptor &desc) {
    if (!desc.strides.empty() || desc.sizes.empty()) {
      return;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Calculating strides for " << desc.pointer << "\n");

    // Compose stride value with static or dynamic size.
    auto compose = [&](Value lhs,
                       TensorDescriptor::StaticOrDynamic size) -> Value {
      Value rhs = size.dynamicValue;
      if (!rhs && size.staticValue != 1) {
        rhs = createConstant(rewriter, lhs.getLoc(),
                             cast<ShapedType>(lhs.getType()), size.staticValue);
      }
      return rhs ? cuda_tile::MulIOp::create(rewriter, lhs.getLoc(), lhs, rhs)
                 : lhs; // Skip multiplication for unit size.
    };

    TensorDescriptor::StaticOrDynamic currentStride = {1, nullptr};
    desc.strides.resize(desc.sizes.size());

    for (size_t dim = desc.sizes.size(); dim > 0; --dim) {
      size_t idx = dim - 1;
      desc.strides[idx] = currentStride;

      auto &size = desc.sizes[idx];

      if (currentStride.dynamicValue) {
        // If current stride is dynamic, multiply it by current dimension size.
        currentStride.dynamicValue = compose(currentStride.dynamicValue, size);
      } else if (size.dynamicValue) {
        // If current size is dynamic, all preceding strides are also dynamic.
        currentStride.dynamicValue = compose(size.dynamicValue, currentStride);
        currentStride.staticValue = ShapedType::kDynamic;
      } else {
        // If current size is static, update the current stride.
        currentStride.staticValue *= size.staticValue;
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// ResultsOp conversion
//===----------------------------------------------------------------------===//

class ResultsOpConversion : public ConversionPattern<ResultsOp> {
public:
  using ConversionPattern<ResultsOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(ResultsOp resultsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &state = static_cast<ConversionStateImpl &>(this->state);

    // The `ResultsOp` has no "layout" attribute set, so the state must be
    // updated here instead of calling the `update` method.
    state.blockStructure = &state.blockStructures[resultsOp.getOperation()];
    state.iterationSpace = &state.blockStructure->iterationSpaces[0];
    state.layouts = ConversionStateImpl::OperandLayouts{state.normalizedLayout};
    state.operandIndex = 0;

    rewriter.setInsertionPointToEnd(state.iterationSpace->insertionBlock);

    // Reshape the output to the iteration space shape.
    const TensorDescriptor &desc = state.outputDescriptors[0];
    TensorSourceAttr outputSource = buildTensorSource(desc);
    outputSource = dyn_cast_if_present<TensorSourceAttr>(
        outputSource.reshape(state.normalizedLayout.getShape()));
    if (!outputSource) {
      return resultsOp.emitError("Failed to calculate the output layout");
    }

    // Emit store for the output tensor.
    auto layoutDesc = applyLayout(rewriter, desc, outputSource);
    Value tile = state.getTile(adaptor.getOperands()[0]);
    emitStore(rewriter, layoutDesc, tile, state.iterationSpace->indexValues);

    // Remove the zero index if it has no users.
    if (state.zeroIndex.use_empty()) {
      rewriter.eraseOp(state.zeroIndex.getDefiningOp());
    }

    // Replace `ResultsOp` with `cuda_tile::ReturnOp`.
    rewriter.replaceOpWithNewOp<cuda_tile::ReturnOp>(resultsOp);
    return success();
  }

private:
  /// Build tensor source from the tensor descriptor.
  static TensorSourceAttr buildTensorSource(const TensorDescriptor &desc) {
    tcg::Shape cgShape;
    for (auto &size : desc.sizes) {
      if (size.staticValue != ShapedType::kDynamic) {
        cgShape.append(size.staticValue);
      } else {
        cgShape.appendDynamic();
      }
    }
    tcg::Stride cgStride;
    for (auto &stride : desc.strides) {
      if (stride.staticValue != ShapedType::kDynamic) {
        cgStride.append(stride.staticValue);
      } else {
        cgStride.appendDynamic();
      }
    }
    tcg::Layout layout(cgShape, cgStride);
    return TensorSourceAttr::get(desc.pointer.getContext(), /*tensorId=*/-1,
                                 /*offset=*/0, layout.toString(),
                                 getDynamicValueMapping(layout));
  }
};

//===----------------------------------------------------------------------===//
// ConcatenateOp conversion
//===----------------------------------------------------------------------===//

class ConcatenateOpConversion : public ConversionPattern<ConcatenateOp> {
public:
  using ConversionPattern<ConcatenateOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(ConcatenateOp concatOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, concatOp));

    // Retrieve the block structure.
    const BlockStructure *blockStructure = state.getBlockStructure();
    assert(blockStructure && "expected block structure for concatenation");

    Value result;
    {
      // Restore the insertion point after processing the nested blocks.
      RewriterBase::InsertionGuard guard(rewriter);

      // Yield each kept operand's tile into the iteration space it feeds, from
      // the operand-to-iteration-space relation stored on the block structure.
      // A pruned operand feeds no space and is skipped. Operands are visited in
      // position order, matching the source order of `getTile`'s per-source
      // layout counter.
      for (auto [operandIdx, spaceIdx] :
           llvm::enumerate(blockStructure->iterationSpaceIndexForOperand)) {
        if (spaceIdx == BlockStructure::kNoIterationSpace) {
          continue;
        }
        Value tile = state.getTile(adaptor.getOperands()[operandIdx]);

        if (!blockStructure->yieldValues.empty()) {
          // Normal case: separate block for each source.
          rewriter.setInsertionPointToEnd(
              blockStructure->iterationSpaces[spaceIdx].insertionBlock);
          cuda_tile::YieldOp::create(rewriter, concatOp.getLoc(), tile);
          result = blockStructure->yieldValues[0];
        } else {
          // Special case: single source with no blocks.
          result = tile;
        }
      }
    }

    // Replace `ConcatenateOp` with the result of the select.
    rewriter.replaceOp(concatOp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion state implementation
//===----------------------------------------------------------------------===//

/// Prepare the conversion state for a graph conversion.
/// Read the IR attributes and convert them to internal structures.
///
/// Layout propagation pipeline consists of the following passes:
/// 1. "layout-propagation-annotation": Sets the "layout" attribute on each
///     operation in the graph.
/// 2. "layout-propagation-normalization": Sets the "iteration_space" attribute
///     on the graph output (also a layout, but may have a different rank).
/// 3. "tile-analyzer": Sets the "tile_candidates" attribute on the graph op
///     based on a heuristic.
/// 4. "tile-selection": Sets the "tile_size" attribute on the graph op.
/// 5. "graph-splitting": Back-propagates the normalized layout through the
///     graph, updates the "layout" attribute on each operation.
LogicalResult ConversionStateImpl::start(GraphOp graphOp) {
  LLVM_DEBUG(llvm::dbgs() << "Processing graph: " << graphOp.getName() << "\n");

  // Validate graph-level preconditions; on failure the helper has already
  // emitted the diagnostic.
  if (failed(verifyGraphLevelLayoutPropLowerable(graphOp))) {
    return failure();
  }

  // Cache validated attributes / descriptors on this state for use by the
  // per-op conversion patterns (accessed via friend declarations).
  Operation *resultsOp = graphOp.getBody()->getTerminator();
  normalizedLayout = resultsOp->getAttrOfType<LayoutSourceAttrInterface>(
      TensorIRDialect::getIterationSpaceAttrName());

  auto tileSizeAttr = graphOp->getAttrOfType<mlir::DenseI32ArrayAttr>(
      nv_tensor_ir::TensorIRDialect::getTileSizeAttrName());
  SmallVector<int64_t> tileShape(tileSizeAttr.asArrayRef().begin(),
                                 tileSizeAttr.asArrayRef().end());
  LLVM_DEBUG({
    llvm::dbgs() << "Tile shape: (";
    llvm::interleaveComma(tileShape, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  inputDescriptors = std::move(*getTensorDescriptors(graphOp.getArgumentTypes(),
                                                     graphOp.getAllArgAttrs()));
  outputDescriptors = std::move(*getTensorDescriptors(
      graphOp.getResultTypes(), graphOp.getAllResultAttrs()));

  // Create the default iteration space.
  IterationSpace defaultIterationSpace;
  defaultIterationSpace.tileShape = std::move(tileShape);
  BlockStructure root{{std::move(defaultIterationSpace)},
                      /*yieldValues=*/{},
                      /*iterationSpaceIndexForOperand=*/{}};

  blockStructures.clear();
  blockStructures.insert({nullptr, std::move(root)});

  // Reset other state components.
  iterationSpaces.clear();
  blockStructure = nullptr;
  iterationSpace = nullptr;
  layouts.clear();
  operandIndex = 0;

  // Start the conversion for the current graph.
  return applyFullConversion(graphOp, conversionTarget, patterns);
}

/// Prepare the conversion state for an operation conversion.
/// This must be called by every conversion pattern class implementation.
LogicalResult ConversionStateImpl::update(ConversionPatternRewriter &rewriter,
                                          Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Converting: " << op->getName() << "\n");

  // Verify the attribute-level legality (layout / iteration-space-id presence
  // and validity). This is shared with the pre-flight feasibility check. Here,
  // during the real conversion, the set of iteration spaces the driver has
  // actually discovered for this graph is live, so the membership check is
  // meaningful and is performed by passing that map directly.
  MLIR_RETURN_IF_ERROR(verifyOpStateAttrs(op, &iterationSpaces));

  // Decompose the layout attribute into the per-operand layouts. The attribute
  // kind has already been validated by `verifyOpStateAttrs` above.
  auto layout = op->getAttrOfType<LayoutSourceAttrInterface>(
      TensorIRDialect::getLayoutAttrName());
  if (isa<PointwiseOpInterface>(op) && op->getNumOperands() > 1) {
    layouts = OperandLayouts(cast<CompositeSourceAttr>(layout).getSources());
  } else if (isa<ConcatenateOp>(op)) {
    layouts = OperandLayouts(cast<ConcatSourceAttr>(layout).getSources());
  } else if (isa<ReduceOp>(op)) {
    layouts = OperandLayouts{cast<ReductionSourceAttr>(layout).getSource()};
  } else if (isa<ReduceUDOp>(op)) {
    // User-defined reduction layout has one source for each input. The
    // composite-source legality below is kept inline here (rather than lifted
    // into `verifyOpStateAttrs`): `ReduceUDOp` reaches this real-conversion
    // path but its per-op legality is not mirrored in the driver-free
    // pre-flight verifier, consistent with deferring such per-op failures to
    // `compile()`.
    auto reduce = cast<ReductionSourceAttr>(layout);
    if (op->getNumOperands() > 1) {
      auto composite = dyn_cast<CompositeSourceAttr>(reduce.getSource());
      if (!composite || composite.size() != op->getNumOperands()) {
        return op->emitError("Expected composite layout for user-defined "
                             "reduction with multiple operands.");
      }
      layouts = OperandLayouts(composite.getSources());
    } else {
      layouts = OperandLayouts{reduce.getSource()};
    }
  } else if (isa<MatmulOp>(op)) {
    // Matmul layout has two input sources.
    auto matmul = cast<MatmulSourceAttr>(layout);
    layouts = OperandLayouts{matmul.getLhs(), matmul.getRhs()};
  } else {
    // Other operations have a single input layout.
    layouts = OperandLayouts{layout};
  }

  // For ops that modify the iteration space, set the block structure pointer.
  auto it1 = blockStructures.find(op);
  blockStructure = it1 != blockStructures.end() ? &it1->second : nullptr;

  // Find the iteration space for the operation (validated above).
  auto iterSpaceId =
      op->getAttrOfType<IntegerAttr>(TensorIRDialect::getIterSpaceIdAttrName());
  iterationSpace = iterationSpaces.find(iterSpaceId.getInt())->second;
  rewriter.setInsertionPointToEnd(iterationSpace->insertionBlock);

  // Reset the operand index.
  operandIndex = 0;
  return success();
}

/// Get the tile for an operand.
/// This must be called once for each operand, instead of using the adaptor
/// value directly. This should not fail if the implementation is correct.
Value ConversionStateImpl::getTile(Value operand) {
  // Get the layout for the operand.
  assert(operandIndex < layouts.size() && "operand index out of bounds");
  LayoutSourceAttrInterface layout = layouts[operandIndex++];

  // Automatic type conversions result in type casts, skip them.
  if (auto unrealizedCast = dyn_cast_if_present<UnrealizedConversionCastOp>(
          operand.getDefiningOp())) {
    operand = unrealizedCast.getOperand(0);
  }

  // For block arguments, find the loaded tile for the layout.
  if (isa<BlockArgument>(operand)) {
    auto tensorSource = dyn_cast<TensorSourceAttr>(layout);
    assert(tensorSource && "expected tensor layout");

    // For owner ops that build several spaces (concatenate, matmul) resolve
    // which one this block-argument operand feeds. `operandIndex` counts the
    // per-source layouts consumed so far and the consumers drive `getTile` in
    // source order, so `operandIndex - 1` is that source's iteration space --
    // the same slot the stored `iterationSpaceIndexForOperand` relation assigns
    // to the operand. (`iterationSpaceIndexForOperand` is keyed by operand
    // position, which for a pruned concatenate differs from this per-source
    // counter, so it is not indexed directly here.)
    const IterationSpace *iterationSpace = this->iterationSpace;
    if (blockStructure) {
      size_t idx =
          blockStructure->iterationSpaces.size() > 1 ? operandIndex - 1 : 0;
      iterationSpace = &blockStructure->iterationSpaces[idx];
    }

    auto it = iterationSpace->loadedTiles.find(tensorSource);
    assert(it != iterationSpace->loadedTiles.end() &&
           "layout has no associated value");
    return it->second;
  }

  // The operand must have a tile type at this point.
  assert(isa<cuda_tile::TileType>(operand.getType()) && "expected tile type");
  return operand;
}

/// Build the signature conversion for the graph function.
/// For each tensor argument, use the pointer type argument followed by integer
/// arguments for each dynamic dimension and _explicit_ dynamic stride.
/// When `uniformSignature` is true, all dimensions and strides are passed.
FailureOr<TypeConverter::SignatureConversion>
ConversionStateImpl::buildSignatureConversion(FunctionType funcTy) const {
  TypeConverter::SignatureConversion result(funcTy.getNumInputs() +
                                            funcTy.getNumResults());

  // Type used for size/stride arguments.
  Type sizeType =
      typeConverter.convertType(IntegerType::get(funcTy.getContext(), 32));

  // Count the number of dynamic arguments (helper).
  auto countDynamic = [](ArrayRef<TensorDescriptor::StaticOrDynamic> items) {
    return llvm::count_if(items, [](const auto &item) {
      return item.staticValue == ShapedType::kDynamic;
    });
  };

  // Add kernel arguments (pointer and optional size/stride arguments).
  auto addKernelArguments = [&](int index, Type type,
                                const TensorDescriptor &desc) -> LogicalResult {
    // Convert the argument type to the tile type.
    auto converted = dyn_cast_if_present<cuda_tile::TileType>(
        typeConverter.convertType(type));
    if (!converted) {
      return failure();
    }

    // If the argument is not a tensor (e.g. a scalar), use the converted type.
    auto tensorTy = dyn_cast<nv_tensor_ir::TensorType>(type);
    if (!tensorTy) {
      result.addInputs(index, converted);
      return success();
    }

    // Convert the tile type to the pointer type.
    auto ptrType = cuda_tile::PointerType::get(converted.getElementType());
    SmallVector<Type> arguments{converted.clone(ptrType)};

    // Add optional size/stride arguments.
    int dynamicArgs = uniformSignature ? tensorTy.getRank() * 2
                                       : countDynamic(desc.sizes) +
                                             countDynamic(desc.strides);
    arguments.append(dynamicArgs, sizeType);

    // Add the arguments to the signature conversion.
    result.addInputs(index, arguments);
    return success();
  };

  // Add kernel arguments for graph inputs and outputs.
  int index = 0;
  for (const auto &[type, desc] :
       llvm::zip_equal(funcTy.getInputs(), inputDescriptors)) {
    MLIR_RETURN_IF_ERROR(addKernelArguments(index++, type, desc));
  }
  for (const auto &[type, desc] :
       llvm::zip_equal(funcTy.getResults(), outputDescriptors)) {
    MLIR_RETURN_IF_ERROR(addKernelArguments(index++, type, desc));
  }
  return result;
}

/// Register conversion patterns for the conversion state.
RewritePatternSet ConversionStateImpl::registerPatterns(MLIRContext *context) {
  RewritePatternSet patterns(context);

  // Conversion patterns defined in this file.
  patterns.add<GraphOpConversion, ResultsOpConversion, ReshapeOpConversion,
               BroadcastOpConversion, TransposeOpConversion, SliceOpConversion,
               ConcatenateOpConversion, IotaOpConversion>(*this, typeConverter,
                                                          context);

  // Pointwise operations.
  registerPointwisePatterns(patterns, *this, typeConverter,
                            enableExperimentalCudaTileOps);
  // Reduction operations.
  registerReductionPatterns(patterns, *this, typeConverter);
  // Matmul operation.
  registerMatmulPatterns(patterns, *this, typeConverter);

  return patterns;
}

} // namespace

LogicalResult validateIotaOpLowerable(IotaOp op, ArrayRef<int64_t> tileShape) {
  auto layout = op->getAttrOfType<LayoutSourceAttrInterface>(
      TensorIRDialect::getLayoutAttrName());
  auto layoutSource = dyn_cast<TensorSourceAttr>(layout);
  if (!layoutSource) {
    return op.emitError("expected tensor_source layout for iota");
  }

  auto cuteLayout = layoutSource.getCuteLayout();
  const auto &stride = cuteLayout.stride();
  if (!tcg::is_static(stride)) {
    return op.emitError("iota layout stride must be static");
  }

  size_t strideRank = tcg::rank(stride);
  if (strideRank != tileShape.size()) {
    return op.emitError("layout stride rank ")
           << strideRank << " does not match iteration space tile rank "
           << tileShape.size();
  }
  return success();
}

std::unique_ptr<ConversionState> createLayoutPropagationConversionState(
    MLIRContext *context, const TypeConverter &typeConverter,
    Attribute optimizationHints, bool uniformSignature,
    int64_t reductionTileSize, bool enableExperimentalCudaTileOps) {
  return std::make_unique<ConversionStateImpl>(
      context, typeConverter,
      cast_if_present<cuda_tile::OptimizationHintsAttr>(optimizationHints),
      uniformSignature, reductionTileSize, enableExperimentalCudaTileOps);
}

LogicalResult
verifyOpStateAttrs(Operation *op,
                   const DenseMap<int, IterationSpace *> *iterationSpaces) {
  // The `layout` attribute is produced by layout-propagation-annotation and
  // is required for every op the driver converts.
  auto layout = op->getAttrOfType<LayoutSourceAttrInterface>(
      TensorIRDialect::getLayoutAttrName());
  if (!layout) {
    return op->emitError(
        "Missing layout attribute. Make sure the "
        "\"layout-propagation-annotation\" pass has been run.");
  }

  // The layout must have the right kind for the op: a composite layout (one
  // source per operand) for multi-operand pointwise ops, a concat layout for
  // concatenation, and a reduction layout for reduction. Use `dyn_cast` so a
  // malformed graph produces a diagnostic instead of crashing.
  if (isa<PointwiseOpInterface>(op) && op->getNumOperands() > 1) {
    auto composite = dyn_cast<CompositeSourceAttr>(layout);
    if (!composite || composite.size() != op->getNumOperands()) {
      return op->emitError("Expected composite layout. Make sure the "
                           "\"graph-splitting\" pass has been run.");
    }
  } else if (isa<ConcatenateOp>(op)) {
    if (!dyn_cast<ConcatSourceAttr>(layout)) {
      return op->emitError("Expected concat layout. Make sure the "
                           "\"graph-splitting\" pass has been run.");
    }
  } else if (isa<ReduceOp, ReduceUDOp>(op)) {
    auto reduction = dyn_cast<ReductionSourceAttr>(layout);
    if (!reduction) {
      return op->emitError("Expected reduction layout. Make sure the "
                           "\"graph-splitting\" pass has been run.");
    }
    if (!tcg::is_static(reduction.getCuteLayout())) {
      return op->emitError(
          "reduction view layout must have static shape and stride");
    }
  } else if (isa<MatmulOp>(op)) {
    auto matmul = dyn_cast<MatmulSourceAttr>(layout);
    if (!matmul) {
      return op->emitError("Expected matmul layout. Make sure the "
                           "\"graph-splitting\" pass has been run.");
    }
    if (!tcg::is_static(matmul.getCuteLayout())) {
      return op->emitError(
          "matmul view layout must have static shape and stride");
    }
  }

  // The `iteration_space_id` attribute is produced by graph-splitting and must
  // refer to one of the iteration spaces that exist for this graph.
  auto iterSpaceId =
      op->getAttrOfType<IntegerAttr>(TensorIRDialect::getIterSpaceIdAttrName());
  if (!iterSpaceId) {
    return op->emitError("Missing iteration space id attribute. Make sure the "
                         "\"graph-splitting\" pass has been run.");
  }
  // The id-membership check only applies during the real conversion, where the
  // driver has populated the true set of iteration spaces. The driver-free
  // pre-flight verifier cannot faithfully reconstruct that set (the conversion
  // builds it recursively while materializing the skeleton), so it passes a
  // null map to skip this check; the presence checks above are still enforced.
  if (iterationSpaces && !iterationSpaces->contains(iterSpaceId.getInt())) {
    return op->emitError() << "Invalid iteration space id: "
                           << iterSpaceId.getInt();
  }

  return success();
}

LogicalResult verifyLayoutPropLowerable(GraphOp graphOp) {
  if (failed(verifyGraphLevelLayoutPropLowerable(graphOp))) {
    return failure();
  }

  auto tileSizeAttr = graphOp->getAttrOfType<mlir::DenseI32ArrayAttr>(
      nv_tensor_ir::TensorIRDialect::getTileSizeAttrName());
  assert(tileSizeAttr && "verified by verifyGraphLevelLayoutPropLowerable");
  SmallVector<int64_t> tileShape;
  tileShape.reserve(tileSizeAttr.size());
  for (int32_t size : tileSizeAttr.asArrayRef()) {
    tileShape.push_back(size);
  }

  // Per-op legality. Walk every op the conversion driver would convert (the
  // terminator's results op is handled separately by the driver and carries no
  // layout attribute, so it is skipped here). For each op verify the shared
  // attribute-level legality and dispatch to the per-op validator.
  //
  // The iteration-space-id *membership* check is intentionally skipped here (a
  // null valid-id set). The driver builds the true set of iteration spaces
  // recursively while materializing the conversion skeleton, and that set
  // cannot be faithfully reconstructed by a driver-free walk: multi-iteration-
  // space graphs (concat / reshape / reduce fusions) legitimately use ids the
  // reconstruction would miss, which would false-reject valid graphs. The
  // presence of the attribute is still verified; the membership check runs only
  // on the real-conversion path in `ConversionStateImpl::update()`.
  Operation *resultsOp = graphOp.getBody()->getTerminator();
  WalkResult walkResult = graphOp.getBody()->walk([&](Operation *op) {
    if (op == resultsOp) {
      return WalkResult::advance();
    }
    // Skip the ops nested inside a user-defined reduction's combiner body. The
    // conversion consumes that body as a unit (`ReduceUDOpConversion`); its
    // inner ops (arith ops + the `YieldOp`) are never individually layout-
    // propagated, so they legitimately carry no layout attribute and must not
    // be subjected to the per-op `verifyOpStateAttrs` layout check below (which
    // would false-reject an otherwise lowerable graph). The walk is post-order,
    // so these body ops are visited *before* the enclosing `ReduceUDOp`; a
    // `WalkResult::skip()` on the parent cannot protect them. Guard them here,
    // independent of walk order. The `ReduceUDOp` itself is still validated by
    // `verifyOpStateAttrs` (its top-level layout attribute) when the walk
    // reaches it.
    if (op->getParentOfType<ReduceUDOp>()) {
      return WalkResult::advance();
    }
    if (failed(verifyOpStateAttrs(op, /*iterationSpaces=*/nullptr))) {
      return WalkResult::interrupt();
    }
    if (isa<PointwiseOpInterface>(op)) {
      if (failed(validatePointwiseOpLowerable(op))) {
        return WalkResult::interrupt();
      }
    } else if (auto constantOp = dyn_cast<ConstantOp>(op)) {
      // ConstantOp is not a PointwiseOpInterface op, so dispatch its
      // pre-rewrite legality check explicitly.
      if (failed(validateConstantOpLowerable(constantOp))) {
        return WalkResult::interrupt();
      }
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      if (failed(validateReductionOpLowerable(reduceOp))) {
        return WalkResult::interrupt();
      }
    } else if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
      if (failed(validateMatmulOpLowerable(matmulOp))) {
        return WalkResult::interrupt();
      }
    } else if (auto iotaOp = dyn_cast<IotaOp>(op)) {
      if (failed(validateIotaOpLowerable(iotaOp, tileShape))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

LogicalResult verifyGraphLevelLayoutPropLowerable(GraphOp graphOp) {
  // Check that the graph has exactly one result.
  Operation *resultsOp = graphOp.getBody()->getTerminator();
  if (resultsOp->getNumOperands() != 1) {
    return graphOp.emitError("Only single output graphs are supported");
  }

  // Get the normalized layout from the graph attribute.
  auto normalizedLayout = resultsOp->getAttrOfType<LayoutSourceAttrInterface>(
      TensorIRDialect::getIterationSpaceAttrName());
  if (!normalizedLayout) {
    return graphOp.emitError(
        "Iteration space attribute must be set. Make sure the "
        "\"layout-propagation-normalization\" pass has been run.");
  }

  // Get the tile size from the graph attribute.
  auto tileSizeAttr = graphOp->getAttrOfType<mlir::DenseI32ArrayAttr>(
      nv_tensor_ir::TensorIRDialect::getTileSizeAttrName());
  if (!tileSizeAttr) {
    return graphOp.emitError("Tile size attribute must be set. Make sure the "
                             "\"tile-selection\" pass has been run.");
  }
  ArrayRef<int32_t> tileShape = tileSizeAttr.asArrayRef();

  // Check that the tile size is valid and matches the iteration space rank.
  // This could only fail if the attributes are set manually, or if the tiling
  // heuristic gives wrong results.
  if (llvm::any_of(tileShape, [](int32_t size) { return size <= 0; })) {
    return graphOp.emitError("Incorrect tile size, expected positive values.");
  }
  if (tileShape.size() != normalizedLayout.getShape().size()) {
    return graphOp.emitError("Incorrect tile size, expected rank ")
           << normalizedLayout.getShape().size();
  }

  // Check that the input/output tensor descriptors can be derived from the
  // graph signature.
  if (failed(getTensorDescriptors(graphOp.getArgumentTypes(),
                                  graphOp.getAllArgAttrs()))) {
    return graphOp.emitError("Failed to get input tensor descriptors");
  }
  if (failed(getTensorDescriptors(graphOp.getResultTypes(),
                                  graphOp.getAllResultAttrs()))) {
    return graphOp.emitError("Failed to get output tensor descriptors");
  }

  return success();
}

} // namespace tensor_to_cuda_tile
} // namespace mlir::nv_tensor_ir
