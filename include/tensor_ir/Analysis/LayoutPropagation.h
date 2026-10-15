// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_ANALYSIS_LAYOUTPROPAGATION_H
#define TENSOR_IR_ANALYSIS_LAYOUTPROPAGATION_H

#include "tensor_ir/Dialect/TensorIR.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::nv_tensor_ir {

/// The tensor boundary of a layout-propagation computation.
///
/// A TensorIR `GraphOp` and the bufferized function formed from it describe
/// the same computation with different boundary representations, so both
/// analyses are expressed in terms of this unified boundary summary.
///
/// Inputs seed forward propagation; results select the values to normalize.
/// Outputs describe physical destination storage and constrain normalization,
/// but are not sources of the forward computation.
///
/// For a `GraphOp`, inputs are block arguments and results are the terminator's
/// operands. For a bufferized function, inputs are tensor values produced by
/// boundary loads (`bufferization.to_tensor`) and results are tensor values
/// consumed by boundary stores (`bufferization.materialize_in_destination`),
/// with those bridging operations recorded in `boundaryOps` to exclude them
/// from analysis.
struct LayoutBoundary {
  /// Operation receiving whole-computation diagnostics.
  Operation *op = nullptr;
  /// Block holding the tensor computation.
  Block *body = nullptr;
  /// Physical storage layout of each boundary input tensor
  /// (`TensorSourceAttr`), seeding forward layout propagation. Tensor IDs
  /// follow ABI argument order.
  llvm::DenseMap<Value, TensorSourceAttr> inputs;
  /// Operations in `body` that bridge the boundary and are not analyzed.
  llvm::SmallPtrSet<Operation *, 8> boundaryOps;
  /// Computation results, in ABI order.
  SmallVector<Value> results;
  /// Physical storage layout of each result buffer (`TensorSourceAttr`). Tensor
  /// IDs follow the inputs. Null for results without a ranked tensor layout.
  /// These constrain normalization but do not feed forward propagation.
  SmallVector<TensorSourceAttr> outputs;
  /// Whether any boundary tensor has a dynamic shape.
  bool hasDynamicShapes = false;

  /// Build the boundary of `graph` from its signature attributes.
  static FailureOr<LayoutBoundary> get(GraphOp graph);
};

/// Maps logical tensor coordinates to the boundary input accesses needed to
/// compute them. Intermediate SSA tensors have no allocated storage: their
/// "layout" describes coordinate mappings and provenance. Only boundary input
/// and output buffers have physical layouts, represented by `TensorSourceAttr`
/// wrapping a CuTe `tcg::Layout` with strides and offsets.
///
/// Layout sources
///
/// Each SSA value maps to an immutable `LayoutSourceAttrInterface`:
/// - `TensorSourceAttr` records a boundary tensor ID, CuTe shape and strides,
///   element offset, and runtime values for dynamic layout components.
///   Synthetic sources use ID -1: constants and splats have zero strides;
///   `IotaOp` has unit stride along its varying dimension and zero elsewhere.
/// - `CompositeSourceAttr` combines distinct operand sources, preserving the
///   association between each operand and its constituent sources.
/// - `ReductionSourceAttr` pairs a source with a CuTe reduction view separating
///   spatial (surviving) and contracting (reduced) dimensions.
/// - `MatmulSourceAttr` records LHS/RHS sources, batch extents `B`, spatial
///   extents `M` and `N`, contracting extent `K`, and the logical MMA view.
/// - `ConcatSourceAttr` records operand sources and their concatenation axis.
/// - `ScanSliceSourceAttr` records the carry layout, streaming side inputs, and
///   sequential output dimensions of a `ScanSliceOp`.
///
/// Forward propagation
///
/// Starting from `LayoutBoundary::inputs`, propagation visits operations in
/// forward topological (block) order and invokes
/// `LayoutPropInterface::inferResultLayout` on operand layouts. Views
/// (`ReshapeOp`, `TransposeOp`, `BroadcastOp`, `SliceOp`) transform coordinate
/// mappings while retaining source IDs; multi-input pointwise operations
/// combine distinct sources into composites. ScanSlice bodies are excluded:
/// `ScanSliceSourceAttr` summarizes the scan_slice, and scan_slice conversion separately lowers
/// its body against loop-carried tiles. This is one forward pass, without
/// fixed-point iteration or control-flow joins.
///
/// Use by `tir-form-grid`
///
/// Forward propagation tracks how views and dataflow merges transform boundary
/// accesses. `tir-form-grid` uses that provenance while recursively requesting
/// tiles backward from result stores through SSA definitions:
/// - `LayoutNormalization` reconciles result layouts into an iteration-space
///   carrier. Grid formation partitions its tiles across a 1D `scf.forall`
///   mapped to `#gpu.block<x>`.
/// - `TileMaterializer` starts at `bufferization.materialize_in_destination`
///   stores with result tiles in carrier coordinates and recursively calls
///   `materialize` on defining operations.
/// - For multi-input pointwise operations, `remapPointwiseOperandLayouts`
///   matches forward operand provenance from `propagation.getLayout(...)`
///   against the requested composite, routing sub-slices to each operand.
/// - For `ReduceOp` and `MatmulOp`, forward sources supply contracting dimensions
///   and extents. `planReductionTile` and `materializeMatmul` expand the spatial
///   output demand along those axes to obtain operand tiles.
/// - For `ConcatenateOp`, `ConcatSourceAttr` supplies partition bounds for
///   `scf.if` branches that request tiles from active operands.
/// - Views pass the demanded layout to their input without runtime compute,
///   folding reshapes and permutations into upstream address arithmetic.
/// - At boundary `bufferization.to_tensor` inputs, the accumulated layout and
///   tile coordinates determine the slice, strided addresses, and vector
///   alignment for a hardware `LoadOp`.
class LayoutPropagationAnalysis {
public:
  /// Propagate layouts forward through `boundary.body`. Operand layouts must be
  /// available when their users are visited. Emits diagnostics and fails for
  /// unsupported operations (such as custom accumulator matmuls), missing
  /// operand layouts, or unrepresentable layout transformations; no partial
  /// analysis is returned.
  static FailureOr<LayoutPropagationAnalysis>
  compute(const LayoutBoundary &boundary);

  /// Return the recorded source layout of `value`, including seeded boundary
  /// inputs, or null if it has no state. This is an O(1) lookup in the
  /// precomputed analysis map, not lazy inference.
  /// Supported multi-result operations (ReduceUDOp and ScanSliceOp) share one
  /// attribute across their results.
  LayoutSourceAttrInterface getLayout(Value value) const {
    return layouts.lookup(value);
  }

  /// Return the boundary specification analyzed by this pass.
  const LayoutBoundary &getBoundary() const { return boundary; }

private:
  explicit LayoutPropagationAnalysis(const LayoutBoundary &boundary)
      : boundary(boundary) {}

  LayoutBoundary boundary;
  llvm::DenseMap<Value, LayoutSourceAttrInterface> layouts;
};

/// Reconciles forward result layouts with physical output views into a common
/// iteration-space carrier. Joint normalization preserves both load and store
/// mappings when coalescing or factoring dimensions; simplifying input accesses
/// alone could conflict with destination strides.
///
/// Single result without a scan_slice (`normalizeSingleResult`)
///
/// Bundle the forward result layout and physical output layout into a
/// two-element `CompositeSourceAttr`, using a synthetic output source with
/// `tensorId=0` and `offset=0`. Calling `.normalize()` recursively normalizes
/// sources and structured views; `TensorSourceAttr` uses `tcg::coalesce` to
/// merge contiguous dimensions with compatible strides. `getCommonIterationSpace`
/// examines dimension factors from right to left (last dimension fastest),
/// finds a shape refining both access and storage shapes, and reshapes both
/// sources into it.
///
/// Apply `rebuildLayoutFixReductions` to the normalized computation source so
/// reduction and matmul contracting dimensions match the reshaped space. This
/// source becomes `iterationSpace`; the physical output view is excluded from
/// the input sources to tile. `resultLayouts` and `resultViews` remain empty.
///
/// Multiple results or a scan_slice (`normalizeMultipleResults`)
///
/// 1. Identify the top-level `ScanSliceOp` (at most one) and separate active results
///    from streaming scan_slice results. Streaming results are stored inside the
///    scan_slice carry loop and excluded from carrier selection and the source union,
///    but retain metadata in ABI result order. Collect each active result's
///    forward layout (`getResultLayout`) and output view
///    (`getPhysicalResultSource`).
/// 2. Derive a common carrier (`deriveCommonCarrier`). A sole active result
///    supplies its own shape and anchor. Otherwise, `collectAncestors` finds
///    common SSA ancestors, excluding constants, splats, and rank-0 scalars.
///    A candidate is compatible if `liftResultToCarrier` can lift every active
///    result's layout and physical view into its shape.
/// 3. Select compatible candidates with maximal element count, grouped by shape.
///    For a single maximal shape, `selectCarrierAnchor` chooses the most
///    downstream candidate. With multiple shapes, an anchor must be downstream
///    of candidates in every other shape group; otherwise normalization fails
///    because the carrier is ambiguous.
/// 4. Lift each active result's layout and view into the selected carrier.
///    `liftResultToCarrier` walks backward to the anchor through the terminal
///    coordinate-preserving suffix: pointwise operations preserve coordinates,
///    transposes apply inverse permutations, and reshapes transport shape
///    changes. At the anchor, `liftLayoutToCarrier` tries rank expansion with
///    unit dimensions and broadcasting/reshaping.
/// 5. Bundle all lifted layouts followed by all lifted views in a composite and
///    normalize jointly, reconciling input accesses and output storage. Unpack
///    layouts into `resultLayouts` after `rebuildLayoutFixReductions`, and views
///    into `resultViews`, both indexed by ABI result position. For scans,
///    `populateStreamingResultMetadata` resolves streaming results against the
///    normalized carry.
/// 6. Sort active results' normalized terminal input layouts by printed
///    representation and deduplicate them, so shared inputs and multi-result
///    operations such as `ReduceUDOp` are materialized once. `iterationSpace`
///    is the sole root or a composite of these roots; its shape is the carrier
///    that `tir-form-grid` partitions into tiles.
struct LayoutNormalization {
  /// Union of the normalized input layouts across all results. Its shape
  /// defines the global multi-dimensional iteration-space carrier that
  /// `tir-form-grid` partitions into a grid of tiles.
  LayoutSourceAttrInterface iterationSpace;
  /// Per-result terminal layouts in the carrier, in ABI order. Empty for a
  /// single-result computation without a scan_slice, whose terminal layout is
  /// `iterationSpace`.
  SmallVector<LayoutSourceAttrInterface> resultLayouts;
  /// Per-result physical output views in the carrier, in ABI order. Used by
  /// `tir-form-grid` to decode destination logical descriptors and verify
  /// output store mapping. Empty exactly when `resultLayouts` is empty.
  SmallVector<TensorSourceAttr> resultViews;

  /// Normalize the forward layouts of the results against destination output
  /// views. Dispatches to single-result or multi-result normalization based on
  /// result count and scan_slice presence. Emits diagnostics on failure.
  static FailureOr<LayoutNormalization>
  compute(const LayoutPropagationAnalysis &propagation);

  /// Convenience entry point: extract the `LayoutBoundary` of `graph`, compute
  /// `LayoutPropagationAnalysis`, and normalize the resulting layouts.
  static FailureOr<LayoutNormalization> compute(GraphOp graph);
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_ANALYSIS_LAYOUTPROPAGATION_H
