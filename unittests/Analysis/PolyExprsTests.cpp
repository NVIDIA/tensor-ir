// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/PolyExprs.h"

#include "gtest/gtest.h"
#include <limits>

using namespace mlir::nv_tensor_ir;

/// \file
/// Unit tests for sparse integer polynomials and symbol evaluation contexts in TensorIR.
///
/// Test Suite Scope:
/// - Canonical Forms: zero polynomial, constants, indeterminates, CSR offset structure.
/// - Canonicalization: unsorted powers, duplicate variables, duplicate terms, zero cancellation.
/// - Defensive Validation: rejection of malformed CSR storage offsets.
/// - Ring Arithmetic: addition, subtraction, multiplication, identities, distributivity, interning.
/// - Multivariate Long Division: exact division over Z, division by zero, non-divisible indeterminates,
///   fractional coefficients, non-zero remainder, MonomialOrder tie-breaking.
/// - Arbitrary-Precision Coefficients: DynamicAPInt coefficients exceeding 64 bits, preserving precision.
/// - Exponent Limits: boundary values (UINT32_MAX) and exponent overflow detection.
/// - Symbol Definitions & DAG Normalization: transitive substitution, diamond DAGs, binary exponentiation.
/// - Cycle Detection & Atomicity: rejection of direct and transitive dependency cycles without state corruption.
/// - Range Boundaries: symbol definitions at the extremes of VariableId (DenseMap sentinel avoidance).
/// - Memoization & Cache Invalidation: caching normalization failures and clearing caches on updates.
/// - Context Isolation: cross-context rejection for operations, definitions, and queries.

/// Helper to construct a `PolyExpr` from small 64-bit integer coefficients for convenient test setup.
static const PolyExpr *getSmall(PolyExprContext &context,
                                const SparsePolynomialStorage<int64_t> &input) {
  PolyExpr::Storage storage;
  for (int64_t coefficient : input.coefficients) {
    storage.coefficients.emplace_back(coefficient);
  }
  storage.offsets = input.offsets;
  storage.powers = input.powers;
  return PolyExpr::get(context, storage);
}

namespace {
/// Test fixture providing a shared `PolyExprContext` and commonly used expressions:
/// - zero = 0
/// - one  = 1
/// - x    = x_0
/// - y    = x_1
class PolyExprTest : public testing::Test {
protected:
  PolyExprContext context;
  const PolyExpr *zero = PolyExpr::getZero(context);
  const PolyExpr *one = PolyExpr::getConstant(context, 1);
  const PolyExpr *x = PolyExpr::getSymbol(context, 0);
  const PolyExpr *y = PolyExpr::getSymbol(context, 1);
};

/// Tests construction of zero, constants, symbols, and canonicalization of raw sparse storage.
///
/// Verifies:
/// - Structural queries: isZero, isConstant, isMonomial, isSymbol, getConstantValue.
/// - Unique representation of zero with empty coefficients and powers.
/// - Canonicalization of unsimplified storage: merges duplicate variables, sorts powers,
///   eliminates zero exponents, cancels duplicate opposite terms, and drops zero coefficients.
/// - Support for maximum VariableId (UINT32_MAX).
TEST_F(PolyExprTest, ConstructorsAndCanonicalStorage) {
  EXPECT_TRUE(zero->isZero());
  EXPECT_TRUE(zero->isConstant());
  EXPECT_FALSE(zero->isMonomial());
  EXPECT_EQ(zero->getConstantValue(), 0);
  EXPECT_EQ(zero->getStorage().offsets, std::vector<Offset>({0}));
  EXPECT_EQ(zero, PolyExpr::getConstant(context, 0));
  EXPECT_TRUE(one->isConstant());
  EXPECT_TRUE(one->isMonomial());
  EXPECT_FALSE(one->isSymbol());
  EXPECT_TRUE(x->isSymbol());
  EXPECT_FALSE(x->isConstant());
  EXPECT_FALSE(x->getConstantValue());
  EXPECT_EQ(x, PolyExpr::getSymbol(context, 0));

  // Verify canonicalization of complex raw input:
  // - Term 0: 2 * (x_1^1 * x_0^1 * x_0^2) = 2 * x_0^3 * x_1^1
  // - Term 1: 3 * (x_7^0) = 3 (constant factor, zero exponent stripped)
  // - Term 2: -2 * (x_0^3 * x_1^1)
  // - Term 3: 0 * (x_9^5) = 0 (zero coefficient term, discarded)
  // Term 0 and Term 2 cancel out (2 - 2 = 0), leaving only the constant 3.
  auto *p = getSmall(
      context, {{2, 3, -2, 0},
                {0, 3, 4, 6, 7},
                {{1, 1}, {0, 1}, {0, 2}, {7, 0}, {0, 3}, {1, 1}, {9, 5}}});
  EXPECT_EQ(p, PolyExpr::getConstant(context, 3));

  // Verify power sorting and exponent accumulation within a single term:
  // 2 * (x_1^2 * x_0^1 * x_0^2) -> 2 * x_0^3 * x_1^2
  auto *q = getSmall(context, {{2}, {0, 3}, {{1, 2}, {0, 1}, {0, 2}}});
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(q->getStorage().powers, (std::vector<Power>{{0, 3}, {1, 2}}));
  EXPECT_EQ(q->getStorage().offsets, (std::vector<Offset>{0, 2}));
  constexpr VariableId maxId = std::numeric_limits<VariableId>::max();
  EXPECT_EQ(
      PolyExpr::getSymbol(context, maxId)->getStorage().powers[0].variable,
      maxId);
}

/// Verifies defensive validation of CSR offset arrays in PolyExpr::get.
/// Rejects:
/// - Missing offsets (offsets.size() != coefficients.size() + 1).
/// - Non-zero initial offset.
/// - Final offset not equal to total powers.
/// - Non-monotonic (unsorted) offsets.
TEST_F(PolyExprTest, RejectsMalformedStorage) {
  EXPECT_EQ(getSmall(context, {{1}, {}, {}}), nullptr);
  EXPECT_EQ(getSmall(context, {{1}, {1, 1}, {{0, 1}}}), nullptr);
  EXPECT_EQ(getSmall(context, {{1}, {0, 2}, {{0, 1}}}), nullptr);
  EXPECT_EQ(getSmall(context, {{1, 1, 1}, {0, 2, 1, 2}, {{0, 1}, {1, 1}}}),
            nullptr);
}

/// Verifies basic ring arithmetic, algebraic identities, and hash-consing interning.
///
/// Identities tested:
/// - Commutativity: x + y == y + x
/// - Additive inverse cancellation: x - x == 0
/// - Multiplicative annihilator: x * 0 == 0
/// - Multiplicative identity: x * 1 == x
/// - Difference of squares: (x + y)(x - y) == x^2 - y^2
/// - Binomial expansion: (x + y)^2 == x^2 + 2xy + y^2
/// - Interning: structurally identical expressions share pointer equality.
TEST_F(PolyExprTest, ArithmeticAndInterning) {
  auto *sum = x->add(*y);
  auto *difference = x->subtract(*y);
  ASSERT_NE(sum, nullptr);
  ASSERT_NE(difference, nullptr);
  EXPECT_EQ(sum, y->add(*x));
  EXPECT_EQ(x->subtract(*x), zero);
  EXPECT_EQ(x->multiply(*zero), zero);
  EXPECT_EQ(x->multiply(*one), x);
  auto *xx = x->multiply(*x);
  auto *yy = y->multiply(*y);
  EXPECT_EQ(sum->multiply(*difference), xx->subtract(*yy));
  EXPECT_EQ(sum->multiply(*sum),
            xx->add(*x->multiply(*y))->add(*x->multiply(*y))->add(*yy));
  EXPECT_EQ(sum->getStorage().powers, (std::vector<Power>{{0, 1}, {1, 1}}));
}

/// Verifies multivariate polynomial long division over Z without symbol definitions.
///
/// Cases tested:
/// - Exact quotient: (x^2 - y^2) / (x + y) == x - y
/// - Self-division: p / p == 1
/// - Zero dividend: 0 / x == 0
/// - Division by zero: returns nullptr
/// - Non-divisible indeterminates: x / y == nullptr
/// - Fractional coefficients over Z: x / 2 == nullptr
/// - Negative quotient coefficients: -3(x + y) / (x + y) == -3
/// - Non-zero remainder: (x^2 - y^2 + 1) / (x + y) == nullptr
/// - Monomial order with missing lower-ID variables in divisor terms.
TEST_F(PolyExprTest, ExactPolynomialDivision) {
  auto *sum = x->add(*y);
  auto *difference = x->subtract(*y);
  auto *p = sum->multiply(*difference);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(p->exactDiv(*sum), difference);
  EXPECT_EQ(p->exactDiv(*p), one);
  EXPECT_EQ(zero->exactDiv(*x), zero);
  EXPECT_EQ(x->exactDiv(*zero), nullptr);
  EXPECT_EQ(x->exactDiv(*y), nullptr);
  EXPECT_EQ(x->exactDiv(*PolyExpr::getConstant(context, 2)), nullptr);
  auto *negative = PolyExpr::getConstant(context, -3)->multiply(*sum);
  EXPECT_EQ(negative->exactDiv(*sum), PolyExpr::getConstant(context, -3));
  // Nonzero remainder after successfully canceling the leading term.
  EXPECT_EQ(p->add(*one)->exactDiv(*sum), nullptr);
  // Missing lower-ID variables must compare as exponent zero.
  auto *z = PolyExpr::getSymbol(context, 2);
  auto *divisor = x->multiply(*z)->add(*y);
  auto *quotient = x->add(*z)->subtract(*one);
  EXPECT_EQ(divisor->multiply(*quotient)->exactDiv(*divisor), quotient);
}

/// Verifies exact arbitrary-precision integer arithmetic (`llvm::DynamicAPInt`).
///
/// Ensures that large coefficients (e.g. (2^63 - 1)^4) do not overflow, truncate,
/// or lose precision during:
/// - Addition, subtraction, and multiplication.
/// - Canonicalization and term cancellation in PolyExpr::get.
/// - Dense polynomial squaring and exact division.
/// - Substitution and normalization under symbol definitions.
TEST_F(PolyExprTest, ArbitraryPrecisionCoefficients) {
  llvm::DynamicAPInt large(std::numeric_limits<int64_t>::max());
  large *= large;
  large *= large;
  auto *constant = PolyExpr::getConstant(context, large);
  EXPECT_EQ(constant->getConstantValue(), large);
  EXPECT_EQ(constant, PolyExpr::getConstant(context, large));
  EXPECT_EQ(constant->add(*one)->getConstantValue(), large + 1);
  EXPECT_EQ(zero->subtract(*constant)->getConstantValue(), -large);
  EXPECT_EQ(constant->multiply(*constant)->getConstantValue(), large * large);

  // Large coefficients must also participate in canonicalization and interning.
  EXPECT_EQ(PolyExpr::get(context, {{large, large, -large}, {0, 0, 0, 0}, {}}),
            constant);
  EXPECT_EQ(
      PolyExpr::get(context, {{large, -large}, {0, 1, 2}, {{0, 1}, {0, 1}}}),
      zero);
  auto *dense = PolyExpr::get(context, {{large, large, large, large},
                                        {0, 1, 2, 3, 3},
                                        {{0, 3}, {0, 2}, {0, 1}}});
  ASSERT_NE(dense, nullptr);
  auto *square = dense->multiply(*dense);
  ASSERT_NE(square, nullptr);
  EXPECT_EQ(square->exactDiv(*dense), dense);
  // Check the coefficient of x^3, where all four products contribute.
  EXPECT_EQ(square->getStorage().coefficients[3], 4 * large * large);
  ASSERT_TRUE(context.addDefinition(0, *constant));
  EXPECT_EQ(context.normalize(*x->multiply(*x)),
            PolyExpr::getConstant(context, large * large));
}

/// Verifies boundary condition handling for Exponent (uint32_t).
///
/// Ensures:
/// - Expressions with exponent UINT32_MAX can be created and divided.
/// - Exponent overflow on multiplication is safely caught and returns nullptr.
/// - PolyExpr::get detects exponent overflow when accumulating repeated powers.
TEST_F(PolyExprTest, CheckedExponents) {
  constexpr Exponent maxExponent = std::numeric_limits<Exponent>::max();
  auto *power = getSmall(context, {{1}, {0, 1}, {{0, maxExponent}}});
  ASSERT_NE(power, nullptr);
  EXPECT_EQ(power->multiply(*x), nullptr);
  EXPECT_EQ(power->exactDiv(*x)->multiply(*x), power);
  EXPECT_EQ(getSmall(context, {{1}, {0, 2}, {{0, maxExponent}, {0, 1}}}),
            nullptr);
}

/// Verifies symbol definitions, transitive normalization, and cache invalidation.
///
/// Evaluates:
/// - Normalizing an unmapped symbol y returns y.
/// - y and x + 1 compare unequal initially, but compare equal once y -> x + 1 is added.
/// - Transitive evaluation: z -> y^2, then x -> 2 leads to z normalizing to (2 + 1)^2 = 9.
/// - Cache invalidation: adding the definition for x invalidates cached normalizations for z.
/// - Undefined symbols return nullptr from getDefinition.
TEST_F(PolyExprTest, DefinitionsNormalizeAndInvalidateCache) {
  auto *z = PolyExpr::getSymbol(context, 2);
  auto *sum = x->add(*one);
  EXPECT_EQ(context.normalize(*y), y);
  EXPECT_EQ(context.areEqual(*y, *sum), false);
  ASSERT_TRUE(context.addDefinition(1, *sum));
  EXPECT_EQ(context.normalize(*y), sum);
  EXPECT_EQ(context.areEqual(*y, *sum), true);
  ASSERT_TRUE(context.addDefinition(2, *y->multiply(*y)));
  EXPECT_EQ(context.normalize(*z), sum->multiply(*sum));
  ASSERT_TRUE(context.addDefinition(0, *PolyExpr::getConstant(context, 2)));
  EXPECT_EQ(context.normalize(*z), PolyExpr::getConstant(context, 9));
  EXPECT_EQ(context.getDefinition(999), nullptr);
}

/// Verifies that circular dependencies and redefinitions are atomically rejected.
///
/// Checks:
/// - Direct self-cycle: x_0 -> x_0 + 1 is rejected.
/// - Transitive cycle: x_0 -> x_1 -> x_2 -> x_0 is rejected.
/// - Atomicity: failed additions leave all previous definitions and normalizations intact.
/// - Immutability: attempting to redefine an already-bound symbol is rejected.
TEST_F(PolyExprTest, RejectsCyclesAndRedefinitionsAtomically) {
  auto *z = PolyExpr::getSymbol(context, 2);
  EXPECT_FALSE(context.addDefinition(0, *x->add(*one)));
  EXPECT_EQ(context.getDefinition(0), nullptr);
  ASSERT_TRUE(context.addDefinition(0, *y));
  ASSERT_TRUE(context.addDefinition(1, *z));
  EXPECT_EQ(context.normalize(*x), z);
  EXPECT_FALSE(context.addDefinition(2, *x));
  EXPECT_FALSE(context.addDefinition(0, *one));
  EXPECT_EQ(context.getDefinition(0), y);
  EXPECT_EQ(context.getDefinition(2), nullptr);
  EXPECT_EQ(context.normalize(*x), z);
  ASSERT_TRUE(context.addDefinition(2, *one));
  EXPECT_EQ(context.normalize(*x), one);
}

/// Verifies that symbol definitions and lookups work across the entire VariableId range.
///
/// Specifically tests UINT32_MAX and UINT32_MAX - 1 to ensure that DenseMap key widening
/// prevents collisions with LLVM DenseMap sentinel empty and tombstone markers.
TEST_F(PolyExprTest, DefinitionsSupportFullVariableIdRange) {
  constexpr VariableId maxId = std::numeric_limits<VariableId>::max();
  for (VariableId id : {maxId, maxId - 1}) {
    ASSERT_TRUE(context.addDefinition(id, *one));
    EXPECT_EQ(context.getDefinition(id), one);
    EXPECT_EQ(context.normalize(*PolyExpr::getSymbol(context, id)), one);
  }
}

/// Verifies normalization over diamond Directed Acyclic Graphs (DAGs) and power expansion.
///
/// Evaluates:
/// - Diamond DAG: a = x + 1, b = a + y, c = a - y.
///   Then b * c = (a + y)(a - y) = a^2 - y^2 = (x + 1)^2 - y^2.
/// - Power normalization: evaluates (x + 1)^5 using binary exponentiation when x = 1,
///   yielding 2^5 = 32.
TEST_F(PolyExprTest, SharedDefinitionDagAndPowers) {
  auto *a = PolyExpr::getSymbol(context, 2);
  auto *b = PolyExpr::getSymbol(context, 3);
  auto *c = PolyExpr::getSymbol(context, 4);
  ASSERT_TRUE(context.addDefinition(2, *x->add(*one)));
  ASSERT_TRUE(context.addDefinition(3, *a->add(*y)));
  ASSERT_TRUE(context.addDefinition(4, *a->subtract(*y)));
  auto *sum = x->add(*one);
  auto *expected = sum->multiply(*sum)->subtract(*y->multiply(*y));
  EXPECT_EQ(context.areEqual(*b->multiply(*c), *expected), true);
  auto *power = getSmall(context, {{1}, {0, 1}, {{2, 5}}});
  ASSERT_TRUE(context.addDefinition(0, *one));
  EXPECT_EQ(context.normalize(*power), PolyExpr::getConstant(context, 32));
}

/// Verifies memoization of normalization failures and cache recovery upon new definitions.
///
/// Checks:
/// - A definition causing exponent overflow yields nullptr on normalize, which is memoized.
/// - areEqual returns nullopt when expansion fails.
/// - areEqual returns true via pointer equality for identical expressions without normalizing.
/// - Adding a new definition invalidates the failure cache, allowing successful normalization.
TEST_F(PolyExprTest, NormalizationFailureIsCachedAndInvalidated) {
  constexpr Exponent maxExponent = std::numeric_limits<Exponent>::max();
  auto *power = getSmall(context, {{1}, {0, 1}, {{2, maxExponent}}});
  ASSERT_TRUE(context.addDefinition(0, *power));
  auto *expression = x->multiply(*x);
  EXPECT_EQ(context.normalize(*expression), nullptr);
  EXPECT_EQ(context.areEqual(*expression, *one), std::nullopt);
  EXPECT_EQ(context.areEqual(*expression, *expression), true);
  ASSERT_TRUE(context.addDefinition(2, *one));
  EXPECT_EQ(context.normalize(*expression), one);
  EXPECT_EQ(context.areEqual(*expression, *one), true);
}

/// Verifies strict isolation between distinct PolyExprContext instances.
///
/// Expressions from differing contexts passed to add, subtract, multiply, exactDiv,
/// addDefinition, normalize, or areEqual are safely rejected.
TEST_F(PolyExprTest, RejectsForeignContexts) {
  PolyExprContext other;
  auto *foreign = PolyExpr::getSymbol(other, 0);
  EXPECT_EQ(x->add(*foreign), nullptr);
  EXPECT_EQ(x->subtract(*foreign), nullptr);
  EXPECT_EQ(x->multiply(*foreign), nullptr);
  EXPECT_EQ(x->exactDiv(*foreign), nullptr);
  EXPECT_FALSE(context.addDefinition(1, *foreign));
  EXPECT_EQ(context.normalize(*foreign), nullptr);
  EXPECT_EQ(context.areEqual(*x, *foreign), std::nullopt);
}
} // namespace
