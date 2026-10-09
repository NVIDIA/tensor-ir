// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Support/TCutegen.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace tcg = mlir::nv_tensor_ir::tcutegen;

static_assert(
    std::is_same_v<decltype(std::declval<const tcg::Shape &>().get(0)),
                   const tcg::detail::RecursiveDimValue &>);
static_assert(
    std::is_same_v<decltype(std::declval<const tcg::Stride &>().get(0)),
                   const tcg::detail::RecursiveDimValue &>);

TEST(TCutegenDimValueTest, ParsesScalarDynamicMetadata) {
  std::optional<tcg::detail::DimValue> wideDivisible =
      tcg::detail::DimValue::fromString(" ? { i64 div = 8 } ");
  ASSERT_TRUE(wideDivisible);
  EXPECT_EQ(wideDivisible->toString(), "?{i64 div=8}");

  std::optional<tcg::detail::DimValue> reordered =
      tcg::detail::DimValue::fromString(" ?{div=8 i64} ");
  ASSERT_TRUE(reordered);
  EXPECT_EQ(reordered->toString(), "?{i64 div=8}");
}

TEST(TCutegenDimValueTest, RejectsMalformedValues) {
  for (const char *text :
       {"?{div=4 div=8}", "?{i128}", "?{i7}", "?{i64 div=0}", "?{i64div=4}",
        "?{div 4}", "?{i64 i32}", "?{i}", "", "abc", "-1", "?{}", "? { }",
        "?{div=}", "?{div=0}", "?{div=-4}", "?{div=4", "?{div=4}x"}) {
    SCOPED_TRACE(text);
    EXPECT_FALSE(tcg::detail::DimValue::fromString(text));
  }
}

TEST(TCutegenDimValueTest, ReportsStaticOrDynamic) {
  EXPECT_TRUE(tcg::detail::DimValue::getStatic(42).isStatic());
  EXPECT_FALSE(tcg::detail::DimValue::getDynamic(1).isStatic());
}

TEST(TCutegenDimValueTest, ReportsDivisibility) {
  EXPECT_EQ(tcg::detail::DimValue::getStatic(42).getDivisibility(), 42);
  EXPECT_EQ(tcg::detail::DimValue::getStatic(0).getDivisibility(), 0);
  EXPECT_EQ(tcg::detail::DimValue::getDynamic(8).getDivisibility(), 8);
  EXPECT_EQ(tcg::get_divisibility(tcg::detail::RecursiveDimValue(
                tcg::detail::DimValue::getDynamic(16))),
            16);
  EXPECT_EQ(tcg::get_divisibility(tcg::detail::RecursiveDimValue(
                tcg::detail::DimValue::getStatic(64))),
            64);
}

TEST(TCutegenDimValueTest, ConvertsStaticValueToInteger) {
  EXPECT_EQ(tcg::detail::DimValue::getStatic(42).as_int(), 42);
}

TEST(TCutegenDimValueTest, ComparesValues) {
  EXPECT_EQ(tcg::detail::DimValue::getStatic(42),
            tcg::detail::DimValue::getStatic(42));
  EXPECT_NE(tcg::detail::DimValue::getStatic(42),
            tcg::detail::DimValue::getStatic(7));
  EXPECT_EQ(tcg::detail::DimValue::getStatic(42), 42);
  EXPECT_NE(tcg::detail::DimValue::getDynamic(1), 42);
  EXPECT_NE(tcg::detail::DimValue::getDynamic(1),
            tcg::detail::DimValue::getStatic(1));
  EXPECT_EQ(tcg::detail::DimValue::getDynamic(8),
            tcg::detail::DimValue::getDynamic(8));
  EXPECT_NE(tcg::detail::DimValue::getDynamic(8),
            tcg::detail::DimValue::getDynamic(16));
  EXPECT_EQ(tcg::detail::DimValue::getDynamic(8, 64),
            tcg::detail::DimValue::getDynamic(8, 64));
  EXPECT_NE(tcg::detail::DimValue::getDynamic(8, 64),
            tcg::detail::DimValue::getDynamic(8));
}

TEST(TCutegenDimValueTest, PrintsToLlvmStream) {
  std::string text;
  llvm::raw_string_ostream os(text);

  os << tcg::detail::DimValue::getDynamic(8);

  EXPECT_EQ(os.str(), "?{div=8}");
}

TEST(TCutegenDimValueTest, ConstructorsCreateErrorValuesForInvalidInput) {
  EXPECT_TRUE(tcg::detail::has_error(tcg::detail::DimValue::getStatic(-1)));
  EXPECT_TRUE(tcg::detail::has_error(tcg::detail::DimValue::getDynamic(0)));
  EXPECT_TRUE(
      tcg::detail::has_error(tcg::detail::DimValue::getDynamic(1, 128)));
  EXPECT_EQ(tcg::detail::DimValue::getStatic(-1).toString(), "error_t");

  EXPECT_DEATH_IF_SUPPORTED(
      (void)(tcg::detail::DimValue::getDynamic(1).as_int()),
      "cannot convert dynamic dimension to integer");
}

TEST(TCutegenDimValueTest, ErrorValuesCompareAndReportState) {
  tcg::detail::DimValue error(tcg::CgErrorT{});
  tcg::detail::DimValue otherError(tcg::CgErrorT{});
  tcg::detail::DimValue valid = tcg::detail::DimValue::getStatic(1);

  EXPECT_TRUE(tcg::detail::has_error(error));
  EXPECT_FALSE(error.isStatic());
  EXPECT_EQ(error, otherError);
  EXPECT_NE(error, valid);
  EXPECT_NE(valid, error);
  EXPECT_NE(error, 0);
  EXPECT_DEATH_IF_SUPPORTED((void)error.as_int(),
                            "cannot convert error dimension to integer");
  EXPECT_DEATH_IF_SUPPORTED((void)error.getDivisibility(),
                            "cannot query error dimension divisibility");
}

TEST(TCutegenDimValueTest, RecursiveValuesCompareAndReportState) {
  using RecursiveDimValue = tcg::detail::RecursiveDimValue;
  RecursiveDimValue error(tcg::CgErrorT{});
  RecursiveDimValue otherError(tcg::CgErrorT{});
  RecursiveDimValue leaf(tcg::detail::DimValue::getStatic(1));
  RecursiveDimValue nested(std::vector<RecursiveDimValue>{leaf});
  RecursiveDimValue sameNested(std::vector<RecursiveDimValue>{leaf});
  RecursiveDimValue otherNested(std::vector<RecursiveDimValue>{
      RecursiveDimValue(tcg::detail::DimValue::getStatic(2))});
  tcg::detail::DimValue errorDim(tcg::CgErrorT{});

  EXPECT_EQ(error.toString(), "error_t");
  EXPECT_FALSE(error.isStatic());
  EXPECT_EQ(error, otherError);
  EXPECT_NE(error, leaf);
  EXPECT_NE(leaf, error);
  EXPECT_NE(leaf, nested);
  EXPECT_EQ(nested, sameNested);
  EXPECT_NE(nested, otherNested);
  EXPECT_EQ(leaf.rank(), 1u);
  EXPECT_EQ(nested.rank(), 1u);
  EXPECT_EQ(leaf, tcg::detail::DimValue::getStatic(1));
  EXPECT_NE(leaf, tcg::detail::DimValue::getStatic(2));
  EXPECT_NE(nested, tcg::detail::DimValue::getStatic(1));
  EXPECT_NE(error, errorDim);
  EXPECT_NE(leaf, errorDim);
  EXPECT_EQ(leaf, 1);
  EXPECT_NE(leaf, 2);
  EXPECT_NE(nested, 1);
  EXPECT_NE(error, 0);
  EXPECT_FALSE(tcg::detail::is_congruent(error, leaf));
  EXPECT_FALSE(tcg::detail::is_congruent(leaf, error));
  EXPECT_TRUE(tcg::detail::is_congruent(leaf, leaf));
  EXPECT_DEATH_IF_SUPPORTED((void)error.as_int(),
                            "cannot convert error dimension to integer");
  EXPECT_DEATH_IF_SUPPORTED((void)nested.as_int(),
                            "cannot convert recursive dimension to integer");
  EXPECT_DEATH_IF_SUPPORTED((void)error.getDivisibility(),
                            "cannot query error dimension divisibility");
  EXPECT_DEATH_IF_SUPPORTED((void)nested.getDivisibility(),
                            "cannot query recursive dimension divisibility");
}

template <typename T>
class TCutegenSequenceTest : public ::testing::Test {};
using SequenceTypes = ::testing::Types<tcg::Shape, tcg::Stride>;
TYPED_TEST_SUITE(TCutegenSequenceTest, SequenceTypes);

TYPED_TEST(TCutegenSequenceTest, RoundTripsAndTransforms) {
  struct Case {
    const char *input, *canonical, *reversed, *flat;
    size_t rank, depth, flatRank;
    bool isStatic;
    const char *firstMode = nullptr;
  };
  const Case cases[] = {
      {" (1, ?, ?{div=8}) ", "(1,?,?{div=8})", "(?{div=8},?,1)",
       "(1,?,?{div=8})", 3, 1, 3, false},
      {"(64,?{div=4},1)", "(64,?{div=4},1)", "(1,?{div=4},64)",
       "(64,?{div=4},1)", 3, 1, 3, false},
      {"(64,?{i64},?{i64 div=4},1)", "(64,?{i64},?{i64 div=4},1)",
       "(1,?{i64 div=4},?{i64},64)", "(64,?{i64},?{i64 div=4},1)", 4, 1, 4,
       false},
      {" ?{div=8} ", "?{div=8}", "?{div=8}", "?{div=8}", 1, 0, 1, false},
      {" 64 ", "64", "64", "64", 1, 0, 1, true},
      {"4", "4", "4", "4", 1, 0, 1, true},
      {"16", "16", "16", "16", 1, 0, 1, true},
      {" ((1, ?), (?{div=8}, 4), 2) ", "((1,?),(?{div=8},4),2)",
       "(2,(?{div=8},4),(1,?))", "(1,?,?{div=8},4,2)", 3, 2, 5, false, "(1,?)"},
      {" ((1, 2), (?{div=4}, 8), 32) ", "((1,2),(?{div=4},8),32)",
       "(32,(?{div=4},8),(1,2))", "(1,2,?{div=4},8,32)", 3, 2, 5, false,
       "(1,2)"},
      {"((1,2),(3,4))", "((1,2),(3,4))", "((3,4),(1,2))", "(1,2,3,4)", 2, 2, 4,
       true, "(1,2)"},
      {"()", "()", "()", "()", 0, 1, 0, true},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.input);
    auto value = TypeParam::fromString(test.input);
    ASSERT_TRUE(value);
    EXPECT_FALSE(tcg::has_error(*value));
    EXPECT_EQ(value->toString(), test.canonical);
    EXPECT_EQ(tcg::rank(*value), test.rank);
    EXPECT_EQ(tcg::depth(*value), test.depth);
    EXPECT_EQ(tcg::is_static(*value), test.isStatic);
    EXPECT_EQ(tcg::reverse(*value).toString(), test.reversed);
    auto flat = tcg::flatten(*value);
    EXPECT_EQ(flat.toString(), test.flat);
    EXPECT_EQ(tcg::rank(flat), test.flatRank);
    EXPECT_EQ(tcg::depth(flat), test.depth == 0 ? 0u : 1u);
    if (test.depth == 2) {
      EXPECT_EQ(tcg::rank(value->get(0)), 2u);
      EXPECT_EQ(tcg::depth(value->get(0)), 1u);
      EXPECT_EQ(value->get(0).toString(), test.firstMode);
      EXPECT_EQ(tcg::is_static(value->get(0)),
                std::string(test.firstMode) == "(1,2)");
    }
  }
  EXPECT_EQ(TypeParam().toString(), "()");
  EXPECT_FALSE(tcg::has_error(TypeParam()));
  for (const char *text : {"((),1)", "(1,())"}) {
    EXPECT_TRUE(TypeParam::fromString(text));
  }
  EXPECT_EQ(TypeParam(64).as_int(), 64);
  EXPECT_EQ(tcg::get_divisibility(*TypeParam::fromString("?{div=8}")), 8);
}

TYPED_TEST(TCutegenSequenceTest, TakesRanges) {
  struct Case {
    const char *input;
    int begin, end;
    const char *expected;
  };
  for (const auto &test :
       {Case{"((1,?),(?{div=8},4),2)", 1, 3, "((?{div=8},4),2)"},
        Case{"((1,?),(?{div=8},4),2)", 1, -1, "((?{div=8},4),2)"},
        Case{"((1,2),(?{div=4},8),32)", 0, 2, "((1,2),(?{div=4},8))"},
        Case{"((1,2),(?{div=4},8),32)", 2, 2, "()"}, Case{"4", 0, 1, "(4)"},
        Case{"(2,3)", 0, 8, "(2,3)"}, Case{"(3,1)", 0, 8, "(3,1)"},
        Case{"(2,3)", -1, 1, "error_t"}, Case{"(3,1)", -1, 1, "error_t"}}) {
    SCOPED_TRACE(test.input);
    auto value = TypeParam::fromString(test.input);
    ASSERT_TRUE(value);
    EXPECT_EQ(tcg::take(test.begin, test.end, *value).toString(),
              test.expected);
  }
}

TYPED_TEST(TCutegenSequenceTest, AccessesAndMutatesDimensions) {
  struct Case {
    int64_t first, divisor, last, nestedLast, scalar;
  };
  for (auto test : {Case{1, 8, 4, 3, 4}, Case{64, 4, 1, 4, 8}}) {
    auto dynamic = "?{div=" + std::to_string(test.divisor) + "}";
    auto value =
        TypeParam::fromString("(" + std::to_string(test.first) + "," + dynamic +
                              "," + std::to_string(test.last) + ")");
    ASSERT_TRUE(value);
    EXPECT_EQ(value->get(0), test.first);
    EXPECT_EQ(value->get(0).as_int(), test.first);
    EXPECT_EQ(value->get(1).toString(), dynamic);
    EXPECT_FALSE(tcg::is_static(value->get(1)));
    EXPECT_EQ(tcg::get_divisibility(value->get(1)), test.divisor);
    EXPECT_EQ(value->get(2), test.last);
    EXPECT_EQ((*value)[0], test.first);
    EXPECT_EQ((*value)[1].toString(), dynamic);
    EXPECT_EQ((*value)[2], test.last);
    for (const auto &dims :
         {llvm::SmallVector<TypeParam>(value->begin(), value->end()),
          llvm::SmallVector<TypeParam>(value->cbegin(), value->cend())}) {
      ASSERT_EQ(dims.size(), 3u);
      EXPECT_EQ(dims[0], test.first);
      EXPECT_EQ(dims[1].toString(), dynamic);
      EXPECT_EQ(dims[2], test.last);
    }
    auto nestedText = "(2," + std::to_string(test.nestedLast) + ")";
    auto nested = TypeParam::fromString(nestedText);
    auto replacement = TypeParam::fromString("?{div=16}");
    ASSERT_TRUE(nested);
    ASSERT_TRUE(replacement);
    auto it = value->begin();
    *it = *nested;
    *++it = *replacement;
    EXPECT_EQ(value->toString(), "(" + nestedText + ",?{div=16}," +
                                     std::to_string(test.last) + ")");
    TypeParam scalar(test.scalar);
    *scalar.begin() = *nested;
    EXPECT_EQ(scalar.toString(), nestedText);
    EXPECT_EQ(tcg::rank(scalar), 2u);
  }
}

TYPED_TEST(TCutegenSequenceTest, AppendsModes) {
  TypeParam value;
  value.append(2);
  value.appendDynamic(8);
  value.append(4);
  EXPECT_FALSE(tcg::has_error(value));
  EXPECT_EQ(value.toString(), "(2,?{div=8},4)");
  EXPECT_EQ(tcg::rank(value), 3u);
  EXPECT_EQ(tcg::get_divisibility(value[1]), 8);
  for (int64_t scalar : {2, 64}) {
    TypeParam emptyMode(scalar);
    emptyMode.append(TypeParam());
    EXPECT_FALSE(tcg::has_error(emptyMode));
    EXPECT_EQ(emptyMode.toString(), "(" + std::to_string(scalar) + ",())");
  }
  auto nested = TypeParam::fromString("(3,4)");
  ASSERT_TRUE(nested);
  TypeParam scalar(2);
  scalar.append(*nested);
  EXPECT_FALSE(tcg::has_error(scalar));
  EXPECT_EQ(scalar.toString(), "(2,(3,4))");
  auto strideMode = TypeParam::fromString("(8,1)");
  ASSERT_TRUE(strideMode);
  TypeParam stride;
  stride.append(64);
  stride.appendDynamic(4);
  stride.append(*strideMode);
  EXPECT_FALSE(tcg::has_error(stride));
  EXPECT_EQ(stride.toString(), "(64,?{div=4},(8,1))");
  EXPECT_EQ(tcg::rank(stride), 3u);
}

TYPED_TEST(TCutegenSequenceTest, ReplacingInvalidDimensionRestoresValidity) {
  TypeParam value;
  value.append(2);
  value.append(std::is_same_v<TypeParam, tcg::Shape> ? 0 : -1);
  EXPECT_TRUE(tcg::has_error(value));

  value.set(1, TypeParam(3));

  EXPECT_FALSE(tcg::has_error(value));
  EXPECT_EQ(value.toString(), "(2,3)");
}

TEST(TCutegenVectorValueTest, ShapeConstructsFromMlirShape) {
  llvm::SmallVector<int64_t> mlirShape{2, mlir::ShapedType::kDynamic, 4};
  tcg::Shape shape(mlirShape);

  EXPECT_EQ(shape.toString(), "(2,?,4)");
}

TEST(TCutegenVectorValueTest, ShapeAppendReportsInvalidDimensions) {
  tcg::Shape shape;

  shape.append(2);
  EXPECT_FALSE(tcg::has_error(shape));

  shape.append(0);
  EXPECT_TRUE(tcg::has_error(shape));
  EXPECT_EQ(shape.toString(), "error_t");
}

TEST(TCutegenCoordTest, ConvertsLogicalPositions) {
  struct Case {
    std::string shape;
    int64_t index;
    const char *expected;
  };
  for (const auto &test :
       {Case{"(8,2)", 10, "(5,0)"}, Case{"((2,3),4)", 7, "((0,1),3)"},
        Case{"(2,1,3)", 4, "(1,0,1)"}, Case{"(2,?,3)", 4, "error_t"},
        Case{"(" + std::to_string(std::numeric_limits<int64_t>::max()) + ",2)",
             0, "error_t"}}) {
    SCOPED_TRACE(test.shape);
    auto shape = tcg::Shape::fromString(test.shape);
    ASSERT_TRUE(shape);
    auto coord = tcg::idx2crd(test.index, *shape);
    EXPECT_EQ(coord.toString(), test.expected);
    EXPECT_EQ(tcg::has_error(coord), std::string(test.expected) == "error_t");
    if (test.shape == "((2,3),4)") {
      EXPECT_EQ(tcg::rank(coord), 2u);
      EXPECT_EQ(tcg::depth(coord), 2u);
    }
  }
}

TEST(TCutegenCoordTest, PreservesExplicitColumnMajorOffsets) {
  std::optional<tcg::Layout> layout = tcg::Layout::fromString("(2,4):(1,2)");
  ASSERT_TRUE(layout);

  tcg::Coord coord = tcg::idx2crd(5, layout->shape());

  EXPECT_EQ(coord.toString(), "(1,1)");
  EXPECT_EQ((*layout)(coord).as_int(), 3);
}

TEST(TCutegenCoordTest, AppendsStaticAndNestedCoordinateModes) {
  tcg::Coord coord;
  EXPECT_TRUE(tcg::has_error(coord));
  std::optional<tcg::Coord> nested = tcg::Coord::fromString("(1,0)");
  ASSERT_TRUE(nested);

  coord.append(2);
  coord.append(*nested);

  EXPECT_FALSE(tcg::has_error(coord));
  EXPECT_EQ(coord.toString(), "(2,(1,0))");
  EXPECT_EQ(tcg::rank(coord), 2u);
}

TEST(TCutegenCoordTest, TakesExplicitEmptyTuple) {
  std::optional<tcg::Coord> parsed = tcg::Coord::fromString("()");
  ASSERT_TRUE(parsed);

  tcg::Coord empty = tcg::take(0, 0, tcg::Coord(1));

  EXPECT_FALSE(tcg::has_error(empty));
  EXPECT_EQ(empty, *parsed);
  EXPECT_EQ(empty.toString(), "()");
  EXPECT_TRUE(tcg::has_error(tcg::Coord()));
}

TEST(TCutegenCoordTest, AppendPropagatesDefaultValueErrors) {
  tcg::Coord coord(2);

  coord.append(tcg::Coord());

  EXPECT_TRUE(tcg::has_error(coord));
  EXPECT_EQ(coord.toString(), "error_t");
}

TEST(TCutegenCoordTest, LayoutEvaluatesLogicalCoordinateOffset) {
  std::optional<tcg::Layout> layout = tcg::Layout::fromString("(8,2):(1,8)");
  ASSERT_TRUE(layout);

  tcg::Coord coord = tcg::idx2crd(10, layout->shape());

  EXPECT_EQ(coord.toString(), "(5,0)");
  EXPECT_EQ((*layout)(coord).as_int(), 5);
}

TEST(TCutegenCoordTest, LayoutOffsetReportsAdditionOverflow) {
  std::optional<tcg::Layout> layout = tcg::Layout::fromString(
      "(2,2):(" + std::to_string(std::numeric_limits<int64_t>::max()) + ",1)");
  std::optional<tcg::Coord> coord = tcg::Coord::fromString("(1,1)");
  ASSERT_TRUE(layout);
  ASSERT_TRUE(coord);

  EXPECT_TRUE(tcg::detail::has_error((*layout)(*coord)));
}

TEST(TCutegenCoordTest, LayoutOffsetPropagatesNestedAndDynamicValues) {
  const std::string max = std::to_string(std::numeric_limits<int64_t>::max());
  std::optional<tcg::Layout> nested =
      tcg::Layout::fromString("((2,2),2):((" + max + ",1),1)");
  std::optional<tcg::Coord> nestedCoord = tcg::Coord::fromString("((1,1),1)");
  std::optional<tcg::Layout> flat =
      tcg::Layout::fromString("(2,2,2):(" + max + ",1,1)");
  std::optional<tcg::Coord> flatCoord = tcg::Coord::fromString("(1,1,1)");
  std::optional<tcg::Layout> dynamic =
      tcg::Layout::fromString("(2,2):(?{div=8},1)");
  std::optional<tcg::Coord> coord = tcg::Coord::fromString("(1,1)");
  ASSERT_TRUE(nested);
  ASSERT_TRUE(nestedCoord);
  ASSERT_TRUE(flat);
  ASSERT_TRUE(flatCoord);
  ASSERT_TRUE(dynamic);
  ASSERT_TRUE(coord);

  EXPECT_TRUE(tcg::detail::has_error((*nested)(*nestedCoord)));
  EXPECT_TRUE(tcg::detail::has_error((*flat)(*flatCoord)));
  EXPECT_EQ((*dynamic)(*coord).toString(), "?");
  EXPECT_FALSE(tcg::is_static(*dynamic));
}

TEST(TCutegenCoordTest, RejectsInvalidIndicesShapesAndCoordinates) {
  std::optional<tcg::Shape> shape = tcg::Shape::fromString("(2,3)");
  std::optional<tcg::Layout> layout = tcg::Layout::fromString("(2,3):(3,1)");
  std::optional<tcg::Coord> wrongRank = tcg::Coord::fromString("(1)");
  ASSERT_TRUE(shape);
  ASSERT_TRUE(layout);
  ASSERT_TRUE(wrongRank);

  EXPECT_TRUE(tcg::has_error(tcg::idx2crd(-1, *shape)));
  EXPECT_TRUE(tcg::has_error(tcg::idx2crd(0, tcg::Shape())));
  EXPECT_TRUE(tcg::detail::has_error((*layout)(tcg::Coord(0))));
  EXPECT_TRUE(tcg::detail::has_error((*layout)(*wrongRank)));
  EXPECT_TRUE(tcg::detail::has_error((*layout)(tcg::Coord())));
  EXPECT_TRUE(tcg::has_error(layout->get(2)));
}

TEST(TCutegenCongruenceTest, MatchesScalarProfiles) {
  std::optional<tcg::Shape> scalarShape = tcg::Shape::fromString("4");
  std::optional<tcg::Stride> scalarStride = tcg::Stride::fromString("16");
  std::optional<tcg::Coord> scalarCoord = tcg::Coord::fromString("3");
  std::optional<tcg::Shape> rankOneShape = tcg::Shape::fromString("(4)");
  ASSERT_TRUE(scalarShape);
  ASSERT_TRUE(scalarStride);
  ASSERT_TRUE(scalarCoord);
  ASSERT_TRUE(rankOneShape);

  EXPECT_TRUE(tcg::is_congruent(*scalarShape, *scalarStride));
  EXPECT_TRUE(tcg::is_congruent(*scalarCoord, *scalarShape));
  EXPECT_FALSE(tcg::is_congruent(*scalarShape, *rankOneShape));
}

TEST(TCutegenCongruenceTest, MatchesRecursiveProfiles) {
  std::optional<tcg::Shape> shape =
      tcg::Shape::fromString("((2,?),(?{div=8},4),1)");
  std::optional<tcg::Stride> stride =
      tcg::Stride::fromString("((1,?{div=4}),(64,8),0)");
  std::optional<tcg::Coord> coord = tcg::Coord::fromString("((1,0),(3,2),0)");
  std::optional<tcg::Stride> incongruentStride =
      tcg::Stride::fromString("(1,(64,8),0)");
  ASSERT_TRUE(shape);
  ASSERT_TRUE(stride);
  ASSERT_TRUE(coord);
  ASSERT_TRUE(incongruentStride);

  EXPECT_TRUE(tcg::is_congruent(*shape, *stride));
  EXPECT_TRUE(tcg::is_congruent(*coord, *shape));
  EXPECT_FALSE(tcg::is_congruent(*shape, *incongruentStride));
}

TEST(TCutegenCongruenceTest, LayoutCongruenceUsesShapeProfile) {
  std::optional<tcg::Layout> lhs = tcg::Layout::fromString("(2,3):(1,2)");
  std::optional<tcg::Layout> rhs = tcg::Layout::fromString("(5,7):(100,1)");
  std::optional<tcg::Layout> nested =
      tcg::Layout::fromString("(5,(7,1)):(1,(5,35))");
  std::optional<tcg::Stride> matchingStride = tcg::Stride::fromString("(9,8)");
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  ASSERT_TRUE(nested);
  ASSERT_TRUE(matchingStride);

  EXPECT_TRUE(tcg::is_congruent(*lhs, *rhs));
  EXPECT_TRUE(tcg::is_congruent(*lhs, *matchingStride));
  EXPECT_FALSE(tcg::is_congruent(*lhs, *nested));
}

TEST(TCutegenVectorValueTest, CreatesCompactLogicalStrideFromShape) {
  for (const auto &test :
       {std::pair{"(2,3,4)", "(12,4,1)"}, std::pair{"(2,?,4)", "(?,4,1)"},
        std::pair{"(?,3,4)", "(12,4,1)"},
        std::pair{"(2,(3,4),1,?)", "(?,(?,?),?,1)"}}) {
    SCOPED_TRACE(test.first);
    auto shape = tcg::Shape::fromString(test.first);
    ASSERT_TRUE(shape);
    auto stride = tcg::Layout(*shape).stride();
    EXPECT_EQ(stride.toString(), test.second);
    EXPECT_EQ(tcg::depth(stride), tcg::depth(*shape));
  }
}

TEST(TCutegenVectorValueTest, DefaultStrideErrorsAtOverflow) {
  const auto max = std::to_string(std::numeric_limits<int64_t>::max());
  for (const auto &text : {"(2," + max + ",4)", "(2,(" + max + ",2))"}) {
    SCOPED_TRACE(text);
    auto shape = tcg::Shape::fromString(text);
    ASSERT_TRUE(shape);
    auto stride = tcg::Stride::getDefault(*shape);
    EXPECT_TRUE(tcg::has_error(stride));
    EXPECT_EQ(stride.toString(), "error_t");
    tcg::Layout layout(*shape);
    EXPECT_TRUE(tcg::has_error(layout));
    EXPECT_EQ(layout.toString(), "error_t:error_t");
  }
}

TYPED_TEST(TCutegenSequenceTest, ComparesSequences) {
  for (const auto &texts :
       {std::array{"(1,?,?{div=8})", "(1,?,?{div=16})", "(1,?)"},
        std::array{"(64,?{div=4},1)", "(64,?{div=8},1)", "(64,?{div=4})"}}) {
    auto value = TypeParam::fromString(texts[0]);
    ASSERT_TRUE(value);
    EXPECT_EQ(*value, *TypeParam::fromString(texts[0]));
    EXPECT_NE(*value, *TypeParam::fromString(texts[1]));
    EXPECT_NE(*value, *TypeParam::fromString(texts[2]));
  }
}

TYPED_TEST(TCutegenSequenceTest, PropagatesErrors) {
  TypeParam error(tcg::CgErrorT{});
  TypeParam storedError;
  storedError.append(std::is_same_v<TypeParam, tcg::Shape> ? 0 : -1);
  storedError.append(TypeParam(1));
  EXPECT_TRUE(tcg::has_error(storedError));
  EXPECT_EQ(error, TypeParam(tcg::CgErrorT{}));
  EXPECT_NE(error, TypeParam(1));
  EXPECT_NE(TypeParam(1), error);
  EXPECT_NE(error, 1);
  EXPECT_EQ(tcg::depth(error), 0u);
  EXPECT_FALSE(tcg::is_static(error));
  EXPECT_TRUE(tcg::has_error(tcg::flatten(error)));
  EXPECT_TRUE(tcg::has_error(tcg::reverse(error)));
  EXPECT_TRUE(tcg::has_error(tcg::take(0, 1, error)));
  auto tuple = TypeParam::fromString("(1,2)");
  ASSERT_TRUE(tuple);
  const std::string name =
      std::is_same_v<TypeParam, tcg::Shape> ? "shape" : "stride";
  EXPECT_DEATH_IF_SUPPORTED((void)tuple->as_int(),
                            "cannot convert non-scalar " + name +
                                " to integer");
  EXPECT_DEATH_IF_SUPPORTED((void)tcg::get_divisibility(*tuple),
                            "cannot query non-scalar " + name +
                                " divisibility");
  EXPECT_DEATH_IF_SUPPORTED((void)error.as_int(),
                            "cannot convert error " + name + " to integer");
  EXPECT_DEATH_IF_SUPPORTED((void)tcg::get_divisibility(error),
                            "cannot query error " + name + " divisibility");
}

TEST(TCutegenVectorValueTest, ErrorInputsPropagateToLayoutsAndCoordinates) {
  tcg::Shape error(tcg::CgErrorT{});
  EXPECT_TRUE(tcg::has_error(tcg::Stride::getDefault(error)));
  EXPECT_TRUE(tcg::has_error(tcg::idx2crd(0, error)));
  EXPECT_TRUE(tcg::has_error(tcg::Layout(error)));
  EXPECT_TRUE(
      tcg::has_error(tcg::Layout(tcg::Shape(1), tcg::Stride(tcg::CgErrorT{}))));
}

TEST(TCutegenVectorValueTest, RejectsMalformedVectors) {
  for (const char *text : {"", "1,2,3", "(1,,3)", "0", "(0,1)", "(-1,1)",
                           "(1,(2,),3)", "(1 2)", "(1,)", "((1,2)"}) {
    SCOPED_TRACE(text);
    EXPECT_FALSE(tcg::Shape::fromString(text));
  }
  for (const char *text : {"(1,?{div=})"}) {
    SCOPED_TRACE(text);
    EXPECT_FALSE(tcg::Stride::fromString(text));
  }
}

TEST(TCutegenVectorValueTest,
     ConstructorCreatesErrorShapeForInvalidDimensions) {
  llvm::SmallVector<int64_t> mlirShape{0, 2};
  llvm::SmallVector<int64_t> negativeMlirShape{-2, 2};
  tcg::Shape shape(mlirShape);
  tcg::Shape negativeShape(negativeMlirShape);

  EXPECT_TRUE(tcg::has_error(shape));
  EXPECT_EQ(shape.toString(), "error_t");
  EXPECT_TRUE(tcg::has_error(negativeShape));
  EXPECT_EQ(negativeShape.toString(), "error_t");
}

TEST(TCutegenLayoutTest, LayoutRoundTrips) {
  std::optional<tcg::Layout> layout =
      tcg::Layout::fromString("(1,?,?{div=8}):(64,?{div=4},1)");
  ASSERT_TRUE(layout);
  EXPECT_EQ(layout->toString(), "(1,?,?{div=8}):(64,?{div=4},1)");
  EXPECT_EQ(layout->shape()[0], 1);
  EXPECT_EQ(layout->shape()[1].toString(), "?");
  EXPECT_EQ(layout->shape()[2].toString(), "?{div=8}");
  EXPECT_EQ(layout->stride()[0], 64);
  EXPECT_EQ(layout->stride()[1].toString(), "?{div=4}");
  EXPECT_EQ(layout->stride()[2], 1);
}

TEST(TCutegenLayoutTest, ScalarLayoutRoundTrips) {
  std::optional<tcg::Layout> layout = tcg::Layout::fromString("4:16");
  ASSERT_TRUE(layout);

  EXPECT_EQ(layout->toString(), "4:16");
  EXPECT_EQ(tcg::rank(*layout), 1u);
  EXPECT_EQ(tcg::depth(*layout), 0u);
  EXPECT_EQ(layout->shape().as_int(), 4);
  EXPECT_EQ(layout->stride().as_int(), 16);
}

TEST(TCutegenLayoutTest, ScalarIntegerConstructorCreatesScalarLayout) {
  tcg::Shape shape(4);
  tcg::Stride stride(16);
  tcg::Layout layout(4, 16);

  EXPECT_EQ(shape.toString(), "4");
  EXPECT_EQ(shape.as_int(), 4);
  EXPECT_EQ(stride.toString(), "16");
  EXPECT_EQ(stride.as_int(), 16);
  EXPECT_EQ(layout.toString(), "4:16");
  EXPECT_EQ(tcg::rank(layout), 1u);
  EXPECT_EQ(tcg::depth(layout), 0u);
  EXPECT_EQ(layout.shape().as_int(), 4);
  EXPECT_EQ(layout.stride().as_int(), 16);
}

TEST(TCutegenLayoutTest, RecursiveLayoutRoundTrips) {
  std::optional<tcg::Layout> layout =
      tcg::Layout::fromString("((2,3),4):((1,2),6)");
  ASSERT_TRUE(layout);
  EXPECT_EQ(tcg::rank(*layout), 2u);
  EXPECT_EQ(tcg::depth(*layout), 2u);
  EXPECT_TRUE(tcg::is_static(*layout));
  EXPECT_EQ(layout->get(0).shape().toString(), "(2,3)");
  EXPECT_EQ(layout->get(0).stride().toString(), "(1,2)");
  EXPECT_EQ(layout->get(1).shape(), 4);
  EXPECT_EQ(layout->get(1).stride(), 6);
  EXPECT_EQ(layout->toString(), "((2,3),4):((1,2),6)");
}

TEST(TCutegenLayoutTest, StaticSizeMatchesShapeProduct) {
  struct Case {
    const char *text;
    int64_t shapeSize, strideSize;
    size_t index;
    int64_t shapeDim, strideDim;
  };
  for (const auto &test : {Case{"4:16", 4, 16, 0, 4, 16},
                           Case{"(2,3,4):(12,4,1)", 24, 48, 1, 3, 4},
                           Case{"((2,3),4):((1,2),6)", 24, 12, 0, 6, 2}}) {
    SCOPED_TRACE(test.text);
    auto layout = tcg::Layout::fromString(test.text);
    ASSERT_TRUE(layout);
    EXPECT_EQ(tcg::static_size(*layout), test.shapeSize);
    EXPECT_EQ(tcg::static_size(layout->shape()), test.shapeSize);
    EXPECT_EQ(tcg::static_size(layout->stride()), test.strideSize);
    EXPECT_EQ(tcg::static_size(*layout, test.index), test.shapeDim);
    EXPECT_EQ(tcg::static_size(layout->shape(), test.index), test.shapeDim);
    EXPECT_EQ(tcg::static_size(layout->stride(), test.index), test.strideDim);
  }
}

TEST(TCutegenLayoutTest, LayoutGetReturnsSubLayout) {
  std::optional<tcg::Layout> layout =
      tcg::Layout::fromString("(1,?,?{div=8}):(64,?{div=4},1)");
  ASSERT_TRUE(layout);

  tcg::Layout dim = layout->get(1);

  EXPECT_EQ(dim.shape().toString(), "?");
  EXPECT_EQ(dim.stride().toString(), "?{div=4}");
  EXPECT_EQ(dim.toString(), "?:?{div=4}");
  EXPECT_EQ(tcg::depth(dim), 0u);
  EXPECT_EQ(tcg::get_divisibility(dim.stride()), 4);
}

TEST(TCutegenLayoutTest, LayoutIterationReturnsSubLayouts) {
  std::optional<tcg::Layout> layout =
      tcg::Layout::fromString("(1,?,?{div=8}):(64,?{div=4},1)");
  ASSERT_TRUE(layout);

  llvm::SmallVector<tcg::Layout> dims(layout->begin(), layout->end());

  ASSERT_EQ(dims.size(), 3u);
  EXPECT_EQ(dims[0].toString(), "1:64");
  EXPECT_EQ(dims[1].toString(), "?:?{div=4}");
  EXPECT_EQ(dims[2].toString(), "?{div=8}:1");
}

TEST(TCutegenLayoutTest, TransformsShapeAndStrideTogether) {
  auto layout = tcg::Layout::fromString("((2,3),4,5):((1,2),6,24)");
  ASSERT_TRUE(layout);
  EXPECT_EQ(tcg::reverse(*layout).toString(), "(5,4,(2,3)):(24,6,(1,2))");
  EXPECT_EQ(tcg::reverse(tcg::Layout(4, 16)).toString(), "4:16");
  EXPECT_EQ(tcg::take(1, 3, *layout).toString(), "(4,5):(6,24)");
  EXPECT_EQ(tcg::take(3, 3, *layout).toString(), "():()");
  auto flat = tcg::flatten(*layout);
  EXPECT_EQ(flat.toString(), "(2,3,4,5):(1,2,6,24)");
  EXPECT_EQ(tcg::rank(flat), 4u);
  EXPECT_EQ(tcg::depth(flat), 1u);
  EXPECT_EQ(tcg::get(flat, 1).toString(), "3:2");
  EXPECT_EQ(tcg::flatten(tcg::Layout(4, 16)).toString(), "4:16");
}

TEST(TCutegenLayoutTest, EmptyLayoutIsValid) {
  EXPECT_FALSE(tcg::has_error(tcg::Layout()));
  EXPECT_EQ(tcg::Layout().toString(), "():()");

  std::optional<tcg::Layout> layout = tcg::Layout::fromString("():()");
  ASSERT_TRUE(layout);
  EXPECT_FALSE(tcg::has_error(*layout));
}

TEST(TCutegenLayoutTest, MakeLayoutWrapsScalarLayoutAsRankOne) {
  std::optional<tcg::Layout> scalar = tcg::Layout::fromString("4:16");
  ASSERT_TRUE(scalar);

  tcg::Layout layout = tcg::make_layout(std::vector{*scalar});

  EXPECT_EQ(layout.toString(), "(4):(16)");
  EXPECT_EQ(tcg::rank(layout), 1u);
  EXPECT_EQ(tcg::depth(layout), 1u);
}

TEST(TCutegenLayoutTest, MakeLayoutConcatenatesLayouts) {
  std::optional<tcg::Layout> first = tcg::Layout::fromString("2:1");
  std::optional<tcg::Layout> nested = tcg::Layout::fromString("(3,4):(4,1)");
  std::optional<tcg::Layout> broadcast = tcg::Layout::fromString("5:0");
  ASSERT_TRUE(first);
  ASSERT_TRUE(nested);
  ASSERT_TRUE(broadcast);

  tcg::Layout layout =
      tcg::make_layout(std::vector{*first, *nested, *broadcast});

  EXPECT_EQ(layout.toString(), "(2,(3,4),5):(1,(4,1),0)");
  EXPECT_EQ(tcg::rank(layout), 3u);
  EXPECT_EQ(tcg::depth(layout), 2u);
  EXPECT_EQ(tcg::get(layout, 1).toString(), "(3,4):(4,1)");
}

TEST(TCutegenLayoutTest, AppendsNestedLayouts) {
  tcg::Layout layout;
  EXPECT_FALSE(tcg::has_error(layout));
  EXPECT_EQ(layout.toString(), "():()");
  std::optional<tcg::Layout> nested = tcg::Layout::fromString("(3,4):(4,1)");
  ASSERT_TRUE(nested);

  layout.append(tcg::Layout(2, 1));
  layout.append(*nested);

  EXPECT_FALSE(tcg::has_error(layout));
  EXPECT_EQ(layout.toString(), "(2,(3,4)):(1,(4,1))");
  EXPECT_EQ(tcg::rank(layout), 2u);
}

TEST(TCutegenLayoutTest, ShapeOnlyConstructorUsesCompactLogicalStride) {
  std::optional<tcg::Shape> shape = tcg::Shape::fromString("(2,?,4)");
  ASSERT_TRUE(shape);

  tcg::Layout layout(*shape);
  EXPECT_EQ(layout.toString(), "(2,?,4):(?,4,1)");
}

TEST(TCutegenLayoutTest,
     ShapeOnlyConstructorUsesRecursiveCompactLogicalStride) {
  std::optional<tcg::Shape> shape = tcg::Shape::fromString("(2,(3,4),1,?)");
  ASSERT_TRUE(shape);

  tcg::Layout layout(*shape);
  EXPECT_EQ(layout.toString(), "(2,(3,4),1,?):(?,(?,?),?,1)");
}

TEST(TCutegenLayoutTest, ComparesLayouts) {
  EXPECT_EQ(*tcg::Layout::fromString("(1,?,?{div=8}):(64,?{div=4},1)"),
            *tcg::Layout::fromString("(1,?,?{div=8}):(64,?{div=4},1)"));
  EXPECT_NE(*tcg::Layout::fromString("(1,?,?{div=8}):(64,?{div=4},1)"),
            *tcg::Layout::fromString("(1,?,?{div=16}):(64,?{div=4},1)"));
  EXPECT_NE(*tcg::Layout::fromString("(1,?,?{div=8}):(64,?{div=4},1)"),
            *tcg::Layout::fromString("(1,?,?{div=8}):(64,?{div=8},1)"));
}

TEST(TCutegenLayoutTest, RejectsMalformedLayouts) {
  for (const char *text :
       {"", "(1,2,3)", "(1,2):(4,,8)", "(1,2):(8)", "((1,2),3):(1,(2,6))",
        "1):1", "1}:1", "1:1:1", "(1:1", "1:?{div=4", "?{tag(1)}:1"}) {
    SCOPED_TRACE(text);
    EXPECT_FALSE(tcg::Layout::fromString(text));
  }
}

TEST(TCutegenLayoutTest, ConstructorRejectsMismatchedRanks) {
  std::optional<tcg::Shape> shape = tcg::Shape::fromString("(1,2)");
  std::optional<tcg::Stride> stride = tcg::Stride::fromString("(1)");
  ASSERT_TRUE(shape);
  ASSERT_TRUE(stride);

  tcg::Layout layout(*shape, *stride);

  EXPECT_TRUE(layout.has_error());
  EXPECT_TRUE(tcg::has_error(layout));
  EXPECT_EQ(layout.toString(), "error_t:error_t");
}

TEST(TCutegenLayoutTest, ConstructorRejectsIncongruentRecursiveStructures) {
  std::optional<tcg::Shape> shape = tcg::Shape::fromString("((1,2),3)");
  std::optional<tcg::Stride> stride = tcg::Stride::fromString("(1,(2,6))");
  ASSERT_TRUE(shape);
  ASSERT_TRUE(stride);

  tcg::Layout layout(*shape, *stride);

  EXPECT_TRUE(tcg::has_error(layout));
  EXPECT_EQ(layout.toString(), "error_t:error_t");

  std::optional<tcg::Shape> nestedShape = tcg::Shape::fromString("((1,2),3)");
  std::optional<tcg::Stride> shorterNestedStride =
      tcg::Stride::fromString("((1),2)");
  std::optional<tcg::Stride> rankOneStride = tcg::Stride::fromString("(1)");
  ASSERT_TRUE(nestedShape);
  ASSERT_TRUE(shorterNestedStride);
  ASSERT_TRUE(rankOneStride);

  EXPECT_TRUE(tcg::has_error(tcg::Layout(*nestedShape, *shorterNestedStride)));
  EXPECT_TRUE(tcg::has_error(tcg::Layout(tcg::Shape(4), *rankOneStride)));
}

TEST(TCutegenLayoutTest, CoalescesLogicalLayouts) {
  auto expectCoalesced = [](llvm::StringRef input, llvm::StringRef expected) {
    std::optional<tcg::Layout> layout = tcg::Layout::fromString(input);
    ASSERT_TRUE(layout);

    EXPECT_EQ(tcg::coalesce(*layout).toString(), expected);
  };

  expectCoalesced("1:1", "1:0");
  expectCoalesced("(2,4,6):(24,6,1)", "48:1");
  expectCoalesced("((2,3),4):((12,4),1)", "24:1");
  expectCoalesced("(2,1,3):(3,17,1)", "6:1");
  expectCoalesced("(2,?,3):(?,3,1)", "(2,?,3):(?,3,1)");
  expectCoalesced("(2,4,6):(1,2,8)", "(2,4,6):(1,2,8)");
}

TEST(TCutegenLayoutTest, ReportsLogicalCoalesceOverflow) {
  std::optional<tcg::Layout> layout = tcg::Layout::fromString(
      "(" + std::to_string(std::numeric_limits<int64_t>::max()) + ",2):(2,1)");
  ASSERT_TRUE(layout);

  EXPECT_TRUE(tcg::has_error(tcg::coalesce(*layout)));
}

TEST(TCutegenLayoutTest, CoalescesEmptyAndDynamicLayoutsSafely) {
  auto expectUnchanged = [](llvm::StringRef input) {
    std::optional<tcg::Layout> layout = tcg::Layout::fromString(input);
    ASSERT_TRUE(layout);
    EXPECT_EQ(tcg::coalesce(*layout), *layout);
  };

  EXPECT_TRUE(tcg::has_error(tcg::coalesce(tcg::Layout())));
  std::optional<tcg::Layout> emptyLayout = tcg::Layout::fromString("():()");
  ASSERT_TRUE(emptyLayout);
  EXPECT_TRUE(tcg::has_error(tcg::coalesce(*emptyLayout)));
  std::optional<tcg::Layout> dynamicShape =
      tcg::Layout::fromString("(2,?):(?{div=8},1)");
  ASSERT_TRUE(dynamicShape);
  EXPECT_FALSE(tcg::is_static(*dynamicShape));
  expectUnchanged("(?,2):(?{div=8},1)");
  expectUnchanged("(2,?):(?{div=8},1)");
  expectUnchanged("(2,2):(?{div=8},1)");
  expectUnchanged("(2,2):(2,?)");
}

TEST(TCutegenLayoutTest, ErrorInputsPropagateThroughLayoutOperations) {
  tcg::Layout error(tcg::CgErrorT{});
  tcg::Layout appended(2, 1);
  tcg::Layout strideOnlyError(tcg::Shape(1), tcg::Stride(tcg::CgErrorT{}));
  tcg::Layout appendedStrideOnlyError(2, 1);
  appended.append(error);
  appendedStrideOnlyError.append(strideOnlyError);

  EXPECT_EQ(error, tcg::Layout(tcg::CgErrorT{}));
  EXPECT_NE(error, tcg::Layout(1, 1));
  EXPECT_NE(tcg::Layout(1, 1), error);
  EXPECT_TRUE(tcg::detail::has_error(error(tcg::Coord(0))));
  EXPECT_FALSE(tcg::is_static(error));
  EXPECT_TRUE(tcg::has_error(tcg::flatten(error)));
  EXPECT_TRUE(tcg::has_error(tcg::reverse(error)));
  EXPECT_TRUE(tcg::has_error(tcg::take(0, 1, error)));
  EXPECT_TRUE(tcg::has_error(error.get(0)));
  EXPECT_TRUE(tcg::has_error(appended));
  EXPECT_TRUE(tcg::has_error(appendedStrideOnlyError));
  EXPECT_TRUE(tcg::has_error(tcg::make_layout({})));
  EXPECT_TRUE(tcg::has_error(tcg::make_layout({tcg::Layout(2, 1), error})));
}

TEST(TCutegenCoordTest, ErrorCoordinatesReportState) {
  tcg::Coord error(tcg::CgErrorT{});
  tcg::Coord negative(-1);
  tcg::Coord coordWithStoredError(0);
  coordWithStoredError.append(-1);
  coordWithStoredError.append(tcg::Coord(0));
  std::optional<tcg::Coord> dynamic = tcg::Coord::fromString("?");
  std::optional<tcg::Coord> vector = tcg::Coord::fromString("(0,1)");
  std::optional<tcg::Coord> sameVector = tcg::Coord::fromString("(0,1)");
  std::optional<tcg::Coord> otherVector = tcg::Coord::fromString("(0,2)");
  ASSERT_TRUE(dynamic);
  ASSERT_TRUE(vector);
  ASSERT_TRUE(sameVector);
  ASSERT_TRUE(otherVector);

  EXPECT_EQ(error.toString(), "error_t");
  EXPECT_EQ(error, tcg::Coord(tcg::CgErrorT{}));
  EXPECT_NE(error, tcg::Coord(0));
  EXPECT_NE(tcg::Coord(0), error);
  EXPECT_EQ(tcg::depth(error), 0u);
  EXPECT_FALSE(tcg::is_static(error));
  EXPECT_TRUE(tcg::has_error(negative));
  EXPECT_TRUE(tcg::has_error(coordWithStoredError));
  EXPECT_FALSE(tcg::Coord::fromString("-1"));
  EXPECT_EQ(tcg::depth(*dynamic), 0u);
  EXPECT_FALSE(tcg::is_static(*dynamic));
  EXPECT_EQ(*vector, *sameVector);
  EXPECT_NE(*vector, *otherVector);
  EXPECT_NE(tcg::Coord(0), *vector);
  EXPECT_DEATH_IF_SUPPORTED((void)error.as_int(),
                            "cannot convert error coordinate to integer");
  EXPECT_DEATH_IF_SUPPORTED((void)vector->as_int(),
                            "cannot convert non-scalar coordinate to integer");
}

TEST(TCutegenLayoutTest, LogicalCoalescePreservesEverySmallStaticOffset) {
  for (int64_t dim0 = 1; dim0 <= 3; ++dim0) {
    for (int64_t dim1 = 1; dim1 <= 3; ++dim1) {
      for (int64_t dim2 = 1; dim2 <= 3; ++dim2) {
        for (int64_t stride0 = 0; stride0 <= 6; ++stride0) {
          for (int64_t stride1 = 0; stride1 <= 6; ++stride1) {
            for (int64_t stride2 = 0; stride2 <= 6; ++stride2) {
              tcg::Shape shape;
              shape.append(dim0);
              shape.append(dim1);
              shape.append(dim2);
              tcg::Stride stride;
              stride.append(stride0);
              stride.append(stride1);
              stride.append(stride2);
              tcg::Layout layout(shape, stride);
              tcg::Layout coalesced = tcg::coalesce(layout);
              ASSERT_FALSE(tcg::has_error(coalesced));

              int64_t elementCount = dim0 * dim1 * dim2;
              for (int64_t position = 0; position < elementCount; ++position) {
                tcg::Coord originalCoord =
                    tcg::idx2crd(position, layout.shape());
                tcg::Coord coalescedCoord =
                    tcg::idx2crd(position, coalesced.shape());
                ASSERT_FALSE(tcg::has_error(originalCoord));
                ASSERT_FALSE(tcg::has_error(coalescedCoord));
                EXPECT_EQ(layout(originalCoord), coalesced(coalescedCoord));
              }
            }
          }
        }
      }
    }
  }
}

TEST(TCutegenFacadeTest, TensorIRValuesUseCutegenCompatibleHelpers) {
  std::string shapeStr = "(1,?{div=8},4)";
  std::string strideStr = "(64,?{div=4},1)";
  std::optional<tcg::Shape> shape = tcg::from_string<tcg::Shape>(shapeStr);
  std::optional<tcg::Stride> stride = tcg::from_string<tcg::Stride>(strideStr);
  ASSERT_TRUE(shape);
  ASSERT_TRUE(stride);

  tcg::Layout layout(*shape, *stride);

  EXPECT_EQ(tcg::rank(*shape), 3u);
  EXPECT_EQ(tcg::rank(*stride), 3u);
  EXPECT_EQ(tcg::rank(layout), 3u);
  EXPECT_EQ(tcg::depth(*shape), 1u);
  EXPECT_EQ(tcg::depth(*stride), 1u);
  EXPECT_EQ(tcg::depth(layout), 1u);
  EXPECT_FALSE(tcg::is_static(*shape));
  EXPECT_FALSE(tcg::is_static(*stride));
  EXPECT_FALSE(tcg::is_static(layout));
  EXPECT_EQ(tcg::to_string(*shape), "(1,?{div=8},4)");
  EXPECT_EQ(tcg::get(*shape, 1).toString(), "?{div=8}");
  EXPECT_EQ(tcg::get(*stride, 1).toString(), "?{div=4}");

  tcg::Layout dim = tcg::get(layout, 1);
  EXPECT_EQ(dim.shape().toString(), "?{div=8}");
  EXPECT_EQ(dim.stride().toString(), "?{div=4}");
}

TEST(TCutegenFacadeTest, RepeatUsesTensorIRAppendForTensorIRValues) {
  std::optional<tcg::Shape> nestedShape = tcg::Shape::fromString("(2,3)");
  ASSERT_TRUE(nestedShape);

  EXPECT_EQ(tcg::repeat<tcg::Shape>(3, 1).toString(), "(1,1,1)");
  EXPECT_EQ(tcg::repeat<tcg::Shape>(2, *nestedShape).toString(),
            "((2,3),(2,3))");
  EXPECT_EQ(tcg::repeat<tcg::Stride>(3, 0).toString(), "(0,0,0)");
  EXPECT_EQ(tcg::repeat<tcg::Layout>(2, tcg::Layout(4, 16)).toString(),
            "(4,4):(16,16)");
  EXPECT_EQ(tcg::repeat<tcg::Coord>(2, tcg::Coord(1)).toString(), "(1,1)");
}
