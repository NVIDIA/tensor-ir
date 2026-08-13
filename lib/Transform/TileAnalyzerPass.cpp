// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/TileAnalyzer.h"
#include "tensor_ir/Analysis/TileCandidateGenerator.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Dialect/TensorIRAttrs.h"
#include "tensor_ir/Support/TCutegen.h"
#include "tensor_ir/Transform/Passes.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <climits>
#include <utility>

#define DEBUG_TYPE "tile-analyzer"

namespace mlir::nv_tensor_ir {

// Include the generated pass definitions
#define GEN_PASS_DEF_TILEANALYZERPASS
#include "tensor_ir/Transform/Passes.h.inc"

namespace tcg = mlir::nv_tensor_ir::tcutegen;

namespace {

/// Tile-size limits for the conservative fallback path used when the
/// candidate generator cannot produce any valid tile (e.g., small or
/// awkwardly shaped iteration spaces, dynamic dims).
constexpr int32_t kDefaultDynamicTileSize = 128;
constexpr int32_t kMaxFallbackTileSize = 128;

/// Generate a conservative fallback tile from the iteration_space shape.
/// Static dims get bit_floor clamped to kMaxFallbackTileSize; dynamic
/// dims (<= 0, i.e. kDynamic) get kDefaultDynamicTileSize. Dims constrained
/// to a fixed tile size (e.g., a concat dim) override the result.
static SmallVector<int32_t>
generateFallbackTile(ArrayRef<DimSize> iterShape,
                     const llvm::DenseMap<size_t, DimSize> &fixedTileDims) {
  SmallVector<int32_t> tile(iterShape.size());
  for (size_t i = 0; i < iterShape.size(); ++i) {
    DimSize dim = iterShape[i];
    if (dim <= 0) {
      tile[i] = kDefaultDynamicTileSize;
    } else if (dim == 1) {
      tile[i] = 1;
    } else {
      uint64_t po2 = llvm::bit_floor(static_cast<uint64_t>(dim));
      tile[i] =
          static_cast<int32_t>(std::min<uint64_t>(po2, kMaxFallbackTileSize));
    }
  }
  for (auto [dim, size] : fixedTileDims) {
    tile[dim] = static_cast<int32_t>(size);
  }
  return tile;
}

/// Information about the iteration space extracted from layout attributes.
struct IterationSpaceInfo {
  SmallVector<DimSize>
      shape; ///< Common iteration space shape (shared by all sources)
  SmallVector<SmallVector<DimSize>>
      allStrides; ///< Per-source strides (one per input tensor)
  SmallVector<DimSize>
      allElementSizeBytes; ///< Per-source element sizes in bytes

  /// Dimensions with a fixed tile size (e.g. concat dim → 1).
  llvm::DenseMap<size_t, DimSize> fixedTileDims;
};

/// Helper: resolve element size in bytes for a TensorSourceAttr by looking up
/// its tensorId in the graph's argument types.
static FailureOr<DimSize> getElementSizeBytesForSource(GraphOp graphOp,
                                                       TensorSourceAttr src) {
  int32_t tid = src.getTensorId();
  if (tid < 0) {
    // Broadcast/constant sources (tensorId == -1) have all-zero strides and
    // don't generate real memory accesses. Return 0 so the caller can skip
    // them in coalescing analysis.
    return DimSize{0};
  }
  auto argTypes = graphOp.getArgumentTypes();
  if (static_cast<size_t>(tid) >= argTypes.size()) {
    // tensorId may refer to an output (appended after inputs). Check result
    // types as well.
    auto resTypes = graphOp.getResultTypes();
    size_t resIdx = static_cast<size_t>(tid) - argTypes.size();
    if (resIdx < resTypes.size()) {
      if (auto tensorType = dyn_cast<TensorType>(resTypes[resIdx])) {
        unsigned bits = tensorType.getElementType().getIntOrFloatBitWidth();
        return static_cast<DimSize>(std::max(1u, bits / CHAR_BIT));
      }
    }
    return failure();
  }
  auto tensorType = dyn_cast<TensorType>(argTypes[tid]);
  if (!tensorType) {
    return failure();
  }
  unsigned bits = tensorType.getElementType().getIntOrFloatBitWidth();
  return static_cast<DimSize>(std::max(1u, bits / CHAR_BIT));
}

/// Helper: extract shape and strides from a single TensorSourceAttr's CuTe
/// layout.
static FailureOr<std::pair<SmallVector<DimSize>, SmallVector<DimSize>>>
extractShapeAndStridesFromLayout(TensorSourceAttr source,
                                 Operation *diagnosticOp) {
  auto cuteLayout = source.getCuteLayout();
  if (!tcg::is_static(cuteLayout)) {
    return diagnosticOp->emitError()
           << "tile analysis requires static tensor-source layouts, but "
              "tensor source "
           << source.getTensorId() << " has a dynamic shape or stride";
  }

  const auto &cgShape = cuteLayout.shape();
  const auto &cgStride = cuteLayout.stride();
  size_t rank = tcg::rank(cgShape);

  SmallVector<DimSize> shape, strides;
  shape.reserve(rank);
  strides.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    shape.push_back(tcg::static_size(cgShape, i));
    strides.push_back(tcg::static_size(cgStride, i));
  }
  return std::make_pair(std::move(shape), std::move(strides));
}

/// Recursively walk the source tree to collect leaf TensorSourceAttr nodes
/// and dimensions requiring a fixed tile size (concat dim → 1).
static void collectSourceInfo(LayoutSourceAttrInterface source,
                              SmallVectorImpl<TensorSourceAttr> &leaves,
                              llvm::DenseMap<size_t, DimSize> &fixedDims) {
  if (auto ts = dyn_cast<TensorSourceAttr>(source)) {
    leaves.push_back(ts);
  } else if (auto concat = dyn_cast<ConcatSourceAttr>(source)) {
    fixedDims[concat.getDimension()] = 1;
    for (auto child : concat.getSources()) {
      collectSourceInfo(child, leaves, fixedDims);
    }
  } else if (auto composite = dyn_cast<CompositeSourceAttr>(source)) {
    for (auto child : composite.getSources()) {
      collectSourceInfo(child, leaves, fixedDims);
    }
  }
}

/// Extract iteration space info from the "iteration_space" attribute set by
/// LayoutPropagationNormalizationPass. That pass annotates the
/// terminator (results op) and ops that modify the iteration space (reductions,
/// matmuls) with a normalized TensorSourceAttr or CompositeSourceAttr.
///
/// We read from the terminator first (the graph's output iteration space).
/// The shape comes from the common iteration space; strides are extracted
/// per-source so the tile heuristic can evaluate coalescing for each input.
///
/// TODO: Multiple output support when available in the layout attribute.
///
/// Returns failure if no "iteration_space" attributes are found (Phase 1 was
/// not run, or LayoutPropagationNormalizationPass was not run).
FailureOr<IterationSpaceInfo> extractIterationSpace(GraphOp graphOp) {
  // Find the "iteration_space" attr on the terminator (results op).
  // MR #17719 annotates the terminator with the graph's output iteration space.
  Attribute iterSpaceAttr;
  Operation *terminator = graphOp.getBody()->getTerminator();
  if (terminator) {
    iterSpaceAttr =
        terminator->getAttr(TensorIRDialect::getIterationSpaceAttrName());
  }

  if (!iterSpaceAttr) {
    return graphOp.emitError("No \"iteration_space\" attribute found: "
                             "LayoutPropagationAnnotation and "
                             "LayoutPropagationNormalization must run first");
  }

  auto sourceAttr = dyn_cast<LayoutSourceAttrInterface>(iterSpaceAttr);
  if (!sourceAttr) {
    /// TODO: Multiple output support when available in the layout attribute.
    return graphOp.emitError(
        "\"iteration_space\" attribute must be a layout source attribute");
  }

  IterationSpaceInfo info;

  SmallVector<TensorSourceAttr> leafSources;
  collectSourceInfo(sourceAttr, leafSources, info.fixedTileDims);
  // Filter out constant/splat sources (tensorId < 0) — they don't correspond
  // to graph arguments.
  llvm::erase_if(leafSources,
                 [](TensorSourceAttr s) { return s.getTensorId() < 0; });
  if (leafSources.empty()) {
    info.shape = sourceAttr.getShape();
  }

  for (auto &leaf : leafSources) {
    auto elemSize = getElementSizeBytesForSource(graphOp, leaf);
    if (failed(elemSize)) {
      return failure();
    }
    if (*elemSize == 0) {
      continue;
    }
    MLIR_ASSIGN_OR_RETURN(auto shapeAndStrides,
                          extractShapeAndStridesFromLayout(leaf, graphOp));
    auto [shape, strides] = std::move(shapeAndStrides);
    if (info.shape.empty()) {
      info.shape = std::move(shape);
    }
    info.allStrides.push_back(std::move(strides));
    info.allElementSizeBytes.push_back(*elemSize);
  }

  if (info.shape.empty()) {
    return failure();
  }

  // The "iteration_space" attribute contains only input layouts.  Derive
  // output strides from the graph's result stride attributes (which may be
  // explicitly non-contiguous, e.g. nv_tensor_ir.stride="(8,2)") and reshape
  // them into the iteration_space coordinate system — same approach as
  // codegen in TensorToCudaTile.cpp.
  auto resTypes = graphOp.getResultTypes();
  auto resultDescriptors = getTensorDescriptors(graphOp.getResultTypes(),
                                                graphOp.getAllResultAttrs());
  if (failed(resultDescriptors)) {
    terminator->emitError(
        "Failed to get tensor descriptors for output tensors");
    return failure();
  }

  if (!resTypes.empty() && !resultDescriptors->empty()) {
    if (auto tensorType = dyn_cast<TensorType>(resTypes[0])) {
      unsigned bits = tensorType.getElementType().getIntOrFloatBitWidth();
      DimSize outputElemBytes =
          static_cast<DimSize>(std::max(1u, bits / CHAR_BIT));

      tcg::Layout outputLayout(getShapeRef(tensorType));
      auto &explicitStrides = (*resultDescriptors)[0].strides;
      if (!explicitStrides.empty()) {
        tcg::Stride outputStrides;
        for (auto stride : explicitStrides) {
          outputStrides.append(stride.staticValue);
        }
        outputLayout = tcg::Layout(getShapeRef(tensorType), outputStrides);
      }
      auto outputSource =
          TensorSourceAttr::get(graphOp.getContext(), /*tensorId=*/-1,
                                /*offset=*/0, outputLayout.toString(),
                                getDynamicValueMapping(outputLayout));
      auto reshaped =
          dyn_cast_or_null<TensorSourceAttr>(outputSource.reshape(info.shape));
      if (reshaped) {
        MLIR_ASSIGN_OR_RETURN(
            auto shapeAndStrides,
            extractShapeAndStridesFromLayout(reshaped, graphOp));
        auto [shape, strides] = std::move(shapeAndStrides);
        info.allStrides.push_back(std::move(strides));
        info.allElementSizeBytes.push_back(outputElemBytes);
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "TileAnalyzerPass: extracted iteration space from "
                    "\"iteration_space\" attr: shape="
                 << vectorToString(info.shape) << " (" << info.allStrides.size()
                 << " source(s))\n";
    for (size_t i = 0; i < info.allStrides.size(); ++i) {
      llvm::dbgs() << "  source[" << i
                   << "] strides=" << vectorToString(info.allStrides[i])
                   << " elemBytes=" << info.allElementSizeBytes[i] << "\n";
    }
  });

  return info;
}

} // namespace

struct TileAnalyzerPass : public impl::TileAnalyzerPassBase<TileAnalyzerPass> {

  using TileAnalyzerPassBase::TileAnalyzerPassBase;

  void runOnOperation() override {
    GraphOp graphOp = getOperation();

    // Validate pass options. max_candidates must be >= 1 — a value of 0 (or
    // negative) would silently produce zero candidates from the generator and
    // then mask the misconfiguration via the fallback path. Reject it up front
    // so users get a clear error instead of opaque downstream behavior.
    if (max_candidates <= 0) {
      graphOp.emitError() << "Invalid pass option: max-candidates must be > 0, "
                             "got "
                          << max_candidates.getValue();
      return signalPassFailure();
    }

    // Step 1: Early exit — tile already provided.
    // Skip candidate generation when a tile or candidate list is already known
    // (from tensor-ir attributes or compilation options). TileSelectionPass
    // will handle verification and annotation.
    if (graphOp->hasAttr(TensorIRDialect::getTileSizeAttrName())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "TileAnalyzerPass: tile_size already set on GraphOp "
                 << graphOp.getSymName()
                 << " (from tensor-ir), skipping candidate generation\n");
      return;
    }
    if (graphOp->hasAttr(TensorIRDialect::getTileCandidatesAttrName())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "TileAnalyzerPass: tile_candidates already set on GraphOp "
                 << graphOp.getSymName()
                 << " (from tensor-ir), skipping candidate generation\n");
      return;
    }
    if (!tile_size.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "TileAnalyzerPass: tile_size provided via pass options for "
                 << graphOp.getSymName()
                 << ", skipping candidate generation\n");
      return;
    }

    // Step 2: Extract iteration space from "iteration_space" attribute
    // (set by LayoutPropagationNormalizationPass).
    auto iterSpaceInfo = extractIterationSpace(graphOp);
    if (failed(iterSpaceInfo)) {
      return signalPassFailure();
    }

    auto &info = *iterSpaceInfo;

    auto cc = symbolizeComputeCapability(computeCapability);
    if (!cc) {
      graphOp.emitError("Invalid compute capability for tile analysis: ")
          << computeCapability;
      return signalPassFailure();
    }
    GpuArchitecture gpu(/*warp_size=*/32, /*cache_line=*/128, *cc,
                        /*num_sms=*/sm_count);
    TileCandidateGenerator generator(gpu, max_candidates);

    // Step 3: Generate tile candidates using all operands' strides
    // for per-operand coalescing analysis.
    auto candidates = generator.generateCandidates(info.shape, info.allStrides,
                                                   info.allElementSizeBytes,
                                                   info.fixedTileDims);

    // When the byte-weighted/smem tracks produce no valid candidates (e.g.
    // small or awkwardly shaped iteration spaces), synthesize a single
    // conservative fallback tile derived from the iteration_space shape.
    // This guarantees that TileAnalyzerPass always emits at least one
    // candidate, as long as the tile size is not already specified via options
    // or tile_size attribute on graph. So downstream passes (TileSelectionPass)
    // and external consumers (Compiler::analyzeGraph) never have to handle a
    // missing "tile_candidates" attribute.
    if (candidates.empty()) {
      auto fallback = generateFallbackTile(info.shape, info.fixedTileDims);
      LLVM_DEBUG(llvm::dbgs()
                 << "TileAnalyzerPass: no candidates from generator for "
                 << graphOp.getSymName() << "; using fallback tile "
                 << vectorToString(fallback) << "\n");
      candidates.push_back({std::move(fallback), 0.0f, "Fallback"});
    }

    // Step 4: Annotate IR with results.
    MLIRContext *ctx = graphOp.getContext();
    SmallVector<Attribute> candidateAttrs;
    for (const auto &c : candidates) {
      candidateAttrs.push_back(DenseI32ArrayAttr::get(ctx, c.tileShape));
    }
    graphOp->setAttr(TensorIRDialect::getTileCandidatesAttrName(),
                     ArrayAttr::get(ctx, candidateAttrs));

    LLVM_DEBUG({
      llvm::dbgs() << "TileAnalyzerPass: generated " << candidates.size()
                   << " candidates for " << graphOp.getSymName() << ":\n";
      for (size_t i = 0; i < candidates.size(); ++i) {
        llvm::dbgs() << "  [" << i << "] "
                     << vectorToString(candidates[i].tileShape)
                     << " (score: " << candidates[i].score << ", "
                     << candidates[i].rationale << ")\n";
      }
    });
  }
};

} // namespace mlir::nv_tensor_ir
