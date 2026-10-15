// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Unit tests for the symbolic layout algebra underpinning graph normalization.
// Verifies algebraic primitives directly in isolation from compiler passes.
//===----------------------------------------------------------------------===//

#include "tensor_ir/Analysis/SymbolicLayout.h"

#include <gtest/gtest.h>
#include <limits>

using namespace mlir::nv_tensor_ir;

namespace {

class SymbolicLayoutTest : public testing::Test {
protected:
  PolyExprContext context;
  PolyExpr s(unsigned id) { return PolyExpr::getSymbol(context, id); }
  PolyExpr c(int64_t value) { return PolyExpr::getConstant(context, value); }

  llvm::SmallVector<PolyExpr> shape(std::initializer_list<int64_t> extents) {
    llvm::SmallVector<PolyExpr> result;
    for (int64_t extent : extents) {
      result.push_back(c(extent));
    }
    return result;
  }
};

//===----------------------------------------------------------------------===//
// SymbolicLayout coalescing
//===----------------------------------------------------------------------===//

TEST_F(SymbolicLayoutTest, CoalesceMergesContiguousRuns) {
  // In row-major layouts, dimension `i` fuses into `i+1` when its stride equals the total volume of `i+1`.
  auto layout = SymbolicLayout::getCompact(context, shape({2, 4, 6}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto coalesced = layout->coalesce();
  EXPECT_EQ(coalesced.toString(), "(48):(1)");

  auto strided =
      SymbolicLayout::get(context, shape({2, 4, 6}), shape({1, 2, 8}));
  ASSERT_TRUE(strided.ok()) << strided.status().message();
  auto coalesced2 = strided->coalesce();
  EXPECT_EQ(coalesced2.toString(), "(2,4,6):(1,2,8)");
}

TEST_F(SymbolicLayoutTest, CoalesceDropsUnitDimensions) {
  auto layout =
      SymbolicLayout::get(context, shape({2, 1, 3}), shape({3, 17, 1}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto coalesced = layout->coalesce();
  EXPECT_EQ(coalesced.toString(), "(6):(1)");

  auto allUnit = SymbolicLayout::get(context, shape({1, 1}), shape({4, 9}));
  ASSERT_TRUE(allUnit.ok()) << allUnit.status().message();
  auto coalesced2 = allUnit->coalesce();
  EXPECT_EQ(coalesced2.toString(), "(1):(0)");
}

TEST_F(SymbolicLayoutTest, CoalesceMergesAcrossASymbolicExtent) {
  // Solves algebraic equivalence `16*s0 == s0 * 16` symbolically, allowing the three dimensions to coalesce into a single run.
  auto layout = SymbolicLayout::get(context, {c(4), s(0), c(16)},
                                    {c(16) * s(0), c(16), c(1)});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto coalesced = layout->coalesce();
  ASSERT_EQ(coalesced.getRank(), 1u);
  EXPECT_EQ(coalesced.getDim(0).getSize(), c(64) * s(0));
  EXPECT_TRUE(coalesced.getDim(0).getStride().getConstantValue() == 1);
}

TEST_F(SymbolicLayoutTest, CoalesceKeepsUnrelatedSymbolicStrides) {
  auto layout = SymbolicLayout::get(context, {c(4), c(16)}, {s(0), s(1)});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto coalesced = layout->coalesce();
  EXPECT_EQ(coalesced.getRank(), 2u);
}

TEST_F(SymbolicLayoutTest, CoalesceKeepsLeavesWhenMergedExtentOverflows) {
  PolyExpr::Storage storage;
  storage.coefficients = {llvm::DynamicAPInt(1)};
  storage.offsets = {0, 1};
  storage.powers = {{0, std::numeric_limits<Exponent>::max()}};
  PolyExpr extent = PolyExpr::get(context, storage);
  auto layout = SymbolicLayout::get(context, {extent, s(0)}, {s(0), c(1)});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_EQ(layout->coalesce(), *layout);
}

TEST_F(SymbolicLayoutTest, CoalescePreservesLeavesWhenDefinitionsOverflow) {
  auto layout = SymbolicLayout::get(context, {s(0) * s(0)}, {c(1)});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  PolyExpr::Storage storage;
  storage.coefficients = {llvm::DynamicAPInt(1)};
  storage.offsets = {0, 1};
  storage.powers = {{1, std::numeric_limits<Exponent>::max()}};
  ASSERT_TRUE(context.addDefinition(0, PolyExpr::get(context, storage)));
  ASSERT_FALSE(layout->getDim(0).leaves.front().size.normalize());
  EXPECT_EQ(layout->coalesce(), *layout);
  EXPECT_EQ(layout->transpose({0}), *layout);
  EXPECT_EQ(layout->take(0, 1), *layout);
}

//===----------------------------------------------------------------------===//
// SymbolicLayout composition
//===----------------------------------------------------------------------===//

TEST_F(SymbolicLayoutTest, ComposeSplitsAndMerges) {
  auto layout = SymbolicLayout::getCompact(context, shape({4, 16}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto composed = layout->compose(shape({64}));
  ASSERT_TRUE(composed.ok()) << composed.status().message();
  EXPECT_EQ(composed->toString(), "(64):(1)");
  auto composed2 = layout->compose(shape({8, 8}));
  ASSERT_TRUE(composed2.ok()) << composed2.status().message();
  EXPECT_EQ(composed2->toString(), "(8,8):(8,1)");
  auto composed3 = layout->compose(shape({2, 2, 16}));
  ASSERT_TRUE(composed3.ok()) << composed3.status().message();
  EXPECT_EQ(composed3->toString(), "(2,2,16):(32,16,1)");
}

TEST_F(SymbolicLayoutTest, ComposeGroupsWhenATargetSpansSeveralRuns) {
  // Transposed access creates discontinuous segments, preserving both runs grouped within the target dimension.
  auto layout = SymbolicLayout::get(context, shape({16, 4}), shape({1, 16}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto composed = layout->compose(shape({64}));
  ASSERT_TRUE(composed.ok()) << composed.status().message();
  EXPECT_EQ(composed->toString(), "((16,4)):((1,16))");
}

TEST_F(SymbolicLayoutTest, ComposeNestsRatherThanLosingARun) {
  // Combining non-adjacent memory segments represents valid re-indexing by bundling the runs into a grouped dimension.
  auto layout = SymbolicLayout::get(context, shape({4, 4}), shape({1, 8}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto composed = layout->compose(shape({16}));
  ASSERT_TRUE(composed.ok()) << composed.status().message();
  EXPECT_EQ(composed->toString(), "((4,4)):((1,8))");
}

TEST_F(SymbolicLayoutTest, ComposeRejectsABoundaryInsideARun) {
  // The fastest segment has extent 3, preventing an exact partition for a target axis of size 2.
  auto layout = SymbolicLayout::get(context, shape({4, 3}), shape({1, 8}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_FALSE(layout->compose(shape({6, 2})).ok());
}

TEST_F(SymbolicLayoutTest, ComposeKeepsLeadingUnitDimensions) {
  auto layout = SymbolicLayout::getCompact(context, shape({64}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto composed = layout->compose(shape({1, 1, 64}));
  ASSERT_TRUE(composed.ok()) << composed.status().message();
  EXPECT_EQ(composed->toString(), "(1,1,64):(0,0,1)");
}

TEST_F(SymbolicLayoutTest, ComposeSplitsASymbolicExtentByALiteralFactor) {
  auto layout = SymbolicLayout::get(context, {c(4) * s(0)}, {c(1)});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto composed = layout->compose({s(0), c(4)});
  ASSERT_TRUE(composed.ok()) << composed.status().message();
  ASSERT_EQ(composed->getRank(), 2u);
  EXPECT_EQ(composed->getDim(0).getSize(), s(0));
  EXPECT_EQ(composed->getDim(0).getStride(), c(4));
  EXPECT_EQ(composed->getDim(1).getSize(), c(4));
  EXPECT_TRUE(composed->getDim(1).getStride().getConstantValue() == 1);
}

TEST_F(SymbolicLayoutTest, ComposeRejectsAnUnprovableSymbolicSplit) {
  auto layout = SymbolicLayout::get(context, {s(0)}, {c(1)});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_FALSE(layout->compose({s(1), c(4)}).ok());
}

//===----------------------------------------------------------------------===//
// Broadcast, transpose, and slice operations
//===----------------------------------------------------------------------===//

TEST_F(SymbolicLayoutTest, BroadcastExpandsUnitsWithZeroStride) {
  auto layout = SymbolicLayout::get(context, shape({1, 16}), shape({0, 1}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto broadcast = layout->broadcastTo(shape({8, 16}));
  ASSERT_TRUE(broadcast.ok()) << broadcast.status().message();
  EXPECT_EQ(broadcast->toString(), "(8,16):(0,1)");
  EXPECT_FALSE(layout->broadcastTo(shape({8, 32})).ok());
}

TEST_F(SymbolicLayoutTest, TransposeSelectsSourceDimensions) {
  auto layout = SymbolicLayout::getCompact(context, shape({2, 3, 4}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  // Output element result[k] maps to input[permutation[k]].
  auto transposed = layout->transpose({2, 0, 1});
  EXPECT_EQ(transposed.toString(), "(4,2,3):(1,12,4)");
}

TEST_F(SymbolicLayoutTest, SliceScalesStridesAndAccumulatesTheOffset) {
  auto layout = SymbolicLayout::getCompact(context, shape({16}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  PolyExpr offset = c(0);
  auto sliced = layout->slice({4}, {12}, {1}, offset);
  ASSERT_TRUE(sliced.ok()) << sliced.status().message();
  EXPECT_EQ(sliced->toString(), "(8):(1)");
  EXPECT_EQ(offset.getConstantValue(), 4);

  offset = c(0);
  auto sliced2 = layout->slice({0}, {16}, {2}, offset);
  ASSERT_TRUE(sliced2.ok()) << sliced2.status().message();
  EXPECT_EQ(sliced2->toString(), "(8):(2)");
  EXPECT_EQ(offset.getConstantValue(), 0);
}

TEST_F(SymbolicLayoutTest, SliceOfADynamicExtentIsStatic) {
  // Constant slice bounds form an in-bounds runtime precondition, yielding static extents even from dynamic inputs.
  auto layout = SymbolicLayout::get(context, {s(0)}, {c(1)});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  PolyExpr offset = c(0);
  auto sliced = layout->slice({2}, {10}, {1}, offset);
  ASSERT_TRUE(sliced.ok()) << sliced.status().message();
  EXPECT_EQ(sliced->getDim(0).getSize().getConstantValue(), 8);
}

//===----------------------------------------------------------------------===//
// Common iteration space derivation
//===----------------------------------------------------------------------===//

TEST_F(SymbolicLayoutTest, RefinesMisalignedFactorizations) {
  llvm::SmallVector<llvm::SmallVector<PolyExpr>> shapes{shape({128, 4}),
                                                        shape({4, 128})};
  std::optional<llvm::SmallVector<PolyExpr>> refined =
      getCommonIterationSpace(context, shapes);
  ASSERT_TRUE(refined.has_value());
  EXPECT_EQ(*refined, shape({4, 32, 4}));
}

TEST_F(SymbolicLayoutTest, KeepsTheCoarsestCommonShape) {
  std::optional<llvm::SmallVector<PolyExpr>> refined =
      getCommonIterationSpace(context, shape({8, 16}), shape({8, 16}));
  ASSERT_TRUE(refined.has_value());
  EXPECT_EQ(*refined, shape({8, 16}));
}

TEST_F(SymbolicLayoutTest, SplitsASymbolicExtentByALiteralFactor) {
  llvm::SmallVector<PolyExpr> lhs{c(4) * s(0)};
  llvm::SmallVector<PolyExpr> rhs{s(0), c(4)};
  std::optional<llvm::SmallVector<PolyExpr>> refined =
      getCommonIterationSpace(context, lhs, rhs);
  ASSERT_TRUE(refined.has_value());
  EXPECT_EQ(*refined, (llvm::SmallVector<PolyExpr>{s(0), c(4)}));
}

TEST_F(SymbolicLayoutTest, RejectsUnrelatedSymbolicExtents) {
  llvm::SmallVector<PolyExpr> lhs{s(0)};
  llvm::SmallVector<PolyExpr> rhs{s(1)};
  EXPECT_FALSE(getCommonIterationSpace(context, lhs, rhs).has_value());
}

TEST_F(SymbolicLayoutTest, RejectsMismatchedElementCounts) {
  EXPECT_FALSE(
      getCommonIterationSpace(context, shape({8}), shape({16})).has_value());
}

TEST_F(SymbolicLayoutTest, LocatesAGroupBoundary) {
  llvm::SmallVector<PolyExpr> extents = shape({2, 3, 4});
  EXPECT_EQ(findDimensionWithSuffixProduct(extents, c(12)), 1u);
  EXPECT_EQ(findDimensionWithSuffixProduct(extents, c(24)), 0u);
  EXPECT_EQ(findDimensionWithSuffixProduct(extents, c(1)), 3u);
  EXPECT_FALSE(findDimensionWithSuffixProduct(extents, c(6)).has_value());
}

TEST_F(SymbolicLayoutTest, UsesContextDefinitions) {
  auto layout = SymbolicLayout::get(context, {c(2), s(0)}, {s(1), c(1)});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto coalesced = layout->coalesce();
  EXPECT_EQ(coalesced.getRank(), 2u);
  ASSERT_TRUE(context.addDefinition(1, s(0)));
  auto coalesced2 = layout->coalesce();
  EXPECT_EQ(coalesced2.getSize(), c(2) * s(0));
  auto coalesced3 = layout->coalesce();
  EXPECT_EQ(coalesced3.getRank(), 1u);
  ASSERT_TRUE(context.addDefinition(0, c(4)));
  auto composed = layout->compose(shape({4, 2}));
  ASSERT_TRUE(composed.ok()) << composed.status().message();
  EXPECT_EQ(composed->toString(), "(4,2):(2,1)");
  auto compact = SymbolicLayout::getCompact(context, shape({2, 4}));
  ASSERT_TRUE(compact.ok());
  EXPECT_EQ(*layout, *compact);
  EXPECT_EQ(layout->toDebugString(), "(2,4):(4,1)");
  auto refined = getCommonIterationSpace(context, {s(1), c(2)}, {c(8)});
  ASSERT_TRUE(refined);
  EXPECT_EQ(*refined, shape({4, 2}));
}

TEST_F(SymbolicLayoutTest, UsesDefinedUnitExtents) {
  auto layout =
      SymbolicLayout::get(context, {s(0), c(2), s(1)}, {c(99), c(3), c(77)});
  ASSERT_TRUE(layout.ok());
  ASSERT_TRUE(context.addDefinition(0, c(1)));
  ASSERT_TRUE(context.addDefinition(1, c(1)));
  EXPECT_EQ(layout->coalesce().toString(), "(2):(3)");
  auto composed = layout->compose({s(0), c(2), s(1)});
  ASSERT_TRUE(composed.ok()) << composed.status().message();
  EXPECT_EQ(composed->toString(), "(1,2,1):(0,3,0)");
  auto coordinate = delinearize(1, {s(0), c(2), s(1)});
  ASSERT_TRUE(coordinate);
  EXPECT_EQ(*coordinate, (llvm::SmallVector<int64_t>{0, 1, 0}));
}

TEST_F(SymbolicLayoutTest, RejectsNullAndForeignExpressions) {
  PolyExprContext other;
  PolyExpr foreign = PolyExpr::getConstant(other, 1);
  EXPECT_FALSE(SymbolicLayout::get(context, {nullptr}, {c(1)}).ok());
  EXPECT_FALSE(SymbolicLayout::get(context, {foreign}, {c(1)}).ok());
  EXPECT_FALSE(SymbolicLayout::getCompact(context, {foreign}).ok());
  auto layout = SymbolicLayout::getCompact(context, shape({1}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_FALSE(layout->compose({foreign}).ok());
  EXPECT_FALSE(layout->broadcastTo({foreign}).ok());
  EXPECT_FALSE(getCommonIterationSpace(context, {foreign}, {foreign}));
  PolyExpr offset = foreign;
  EXPECT_FALSE(layout->slice({0}, {1}, {1}, offset).ok());
  EXPECT_EQ(offset, foreign);
  EXPECT_FALSE(layout->compose({nullptr}).ok());
}

TEST_F(SymbolicLayoutTest, ConstructionRejectsMalformedDimensions) {
  auto mismatch = SymbolicLayout::get(context, {c(2)}, {});
  ASSERT_FALSE(mismatch.ok());
  EXPECT_EQ(mismatch.status().code(), StatusCode::kInvalidArgument);
  EXPECT_FALSE(mismatch.status().message().empty());

  EXPECT_FALSE(SymbolicLayout::get(context, {LayoutDim()}).ok());
  EXPECT_FALSE(SymbolicLayout::get(context, {LayoutDim(c(2), nullptr)}).ok());
  PolyExprContext other;
  PolyExpr foreign = PolyExpr::getConstant(other, 1);
  EXPECT_FALSE(SymbolicLayout::get(context, {LayoutDim(c(2), foreign)}).ok());

  auto grouped = SymbolicLayout::get(
      context, {LayoutDim::getGroup({{c(2), c(1)}, {c(3), c(8)}})});
  ASSERT_TRUE(grouped.ok()) << grouped.status().message();
  EXPECT_EQ(grouped->toString(), "((2,3)):((1,8))");
  EXPECT_EQ(*grouped, *grouped);
}

TEST_F(SymbolicLayoutTest, FailuresPreserveLayoutAndSliceOffset) {
  auto layout = SymbolicLayout::getCompact(context, shape({4, 8}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  PolyExpr offset = c(7);
  // The first axis contributes an offset before the second axis fails.
  auto sliced = layout->slice({1, 0}, {3, 9}, {1, 1}, offset);
  ASSERT_FALSE(sliced.ok());
  EXPECT_EQ(sliced.status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(offset, c(7));
  EXPECT_EQ(layout->toString(), "(4,8):(8,1)");

  auto composed = layout->compose({s(0), s(1)});
  ASSERT_FALSE(composed.ok());
  EXPECT_EQ(composed.status().code(), StatusCode::kNotSupported);
  EXPECT_FALSE(composed.status().message().empty());
}

TEST_F(SymbolicLayoutTest, PreservesArbitraryPrecisionConstants) {
  llvm::DynamicAPInt large(std::numeric_limits<int64_t>::max());
  large += 1;
  PolyExpr extent = PolyExpr::getConstant(context, large);
  auto layout = SymbolicLayout::getCompact(context, {c(2), extent});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_EQ(layout->toString(),
            "(2,9223372036854775808):(9223372036854775808,1)");
  auto coalesced = layout->coalesce();
  EXPECT_EQ(coalesced.getSize().getConstantValue(), large * 2);
  auto composed = layout->compose({extent, c(2)});
  ASSERT_TRUE(composed.ok()) << composed.status().message();
  EXPECT_EQ(composed->getDim(0).getStride(), c(2));
  EXPECT_FALSE(delinearize(0, {extent}));
  PolyExpr offset = c(0);
  auto subset = layout->take(1, 2);
  auto sliced = subset.slice({1}, {5}, {2}, offset);
  ASSERT_TRUE(sliced.ok()) << sliced.status().message();
  EXPECT_EQ(sliced->toString(), "(2):(2)");
  EXPECT_EQ(offset, c(1));
}

TEST_F(SymbolicLayoutTest, EmptyLayoutsRetainTheirContext) {
  auto layout = SymbolicLayout::getCompact(context, {});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_EQ(&layout->getContext(), &context);
  EXPECT_EQ(layout->getSize(), c(1));
  EXPECT_EQ(layout->toString(), "():()");
  auto coalesced = layout->coalesce();
  EXPECT_EQ(coalesced.toString(), "(1):(0)");
  auto subset = layout->take(0, 0);
  EXPECT_EQ(subset.getSize(), c(1));
  EXPECT_EQ(layout->transpose({}), *layout);
  auto refined = getCommonIterationSpace(context, {}, {});
  ASSERT_TRUE(refined);
  EXPECT_TRUE(refined->empty());
}

TEST_F(SymbolicLayoutTest, RefinesPolynomialFactors) {
  PolyExpr factor = s(0) + c(1);
  PolyExpr volume = factor * factor;
  auto layout = SymbolicLayout::getCompact(context, {volume});
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  auto composed = layout->compose({factor, factor});
  ASSERT_TRUE(composed.ok()) << composed.status().message();
  EXPECT_EQ(composed->getDim(0).getStride(), factor);
  EXPECT_EQ(composed->toDebugString(), "(s0 + 1,s0 + 1):(s0 + 1,1)");
  auto refined = getCommonIterationSpace(context, {volume}, {factor, factor});
  ASSERT_TRUE(refined);
  EXPECT_EQ(*refined, (llvm::SmallVector<PolyExpr>{factor, factor}));
}

TEST_F(SymbolicLayoutTest, RejectsInvalidRefinementVolumes) {
  EXPECT_FALSE(getCommonIterationSpace(context, shape({16}), shape({8})));
  EXPECT_FALSE(getCommonIterationSpace(context, shape({0}), shape({0})));
  EXPECT_FALSE(getCommonIterationSpace(context, shape({-1}), shape({-1})));
  auto layout = SymbolicLayout::getCompact(context, shape({4}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_FALSE(layout->compose(shape({0})).ok());
}

TEST_F(SymbolicLayoutTest, SlicesGroupedDimensions) {
  auto layout = SymbolicLayout::get(context, shape({8, 2}), shape({1, 8}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  layout = layout->compose(shape({16}));
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  PolyExpr offset = c(0);
  auto sliced = layout->slice({1}, {16}, {2}, offset);
  ASSERT_TRUE(sliced.ok()) << sliced.status().message();
  EXPECT_EQ(sliced->toString(), "(8):(1)");
  EXPECT_EQ(offset, c(8));
  EXPECT_EQ(layout->flatten().toString(), "(8,2):(1,8)");
  offset = c(0);
  auto sliced2 = layout->slice({0}, {8}, {1}, offset);
  ASSERT_TRUE(sliced2.ok()) << sliced2.status().message();
  EXPECT_EQ(sliced2->toString(), "((4,2)):((1,8))");
}

} // namespace
