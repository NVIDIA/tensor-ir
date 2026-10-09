// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_ANALYSIS_POLYEXPRS_H
#define TENSOR_IR_ANALYSIS_POLYEXPRS_H

#include "llvm/ADT/DynamicAPInt.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace mlir::nv_tensor_ir {

/// \file
/// Multi-variate integer polynomial analysis and symbol evaluation context for
/// TensorIR.
///
/// Mathematical Model:
/// Represents multi-variate polynomials in \f$\mathbb{Z}[x_0, x_1, \dots]\f$,
/// where symbols \f$x_i\f$ are non-negative integer variables. Coefficients are
/// stored as arbitrary-precision integers (`llvm::DynamicAPInt`), preventing
/// coefficient overflow during polynomial expansion, multiplication, and power
/// evaluation.
///
/// Storage & Canonical Form:
/// - Flat CSR-style storage (`SparsePolynomialStorage`): Monomials and powers are
///   stored in contiguous flat vectors, minimizing heap allocations and maximizing
///   cache locality.
/// - Invariants:
///   1. Powers within each monomial have strictly ascending `VariableId` order
///      with non-zero exponents (\f$e \ge 1\f$).
///   2. Terms have non-zero coefficients and are ordered in descending lexicographic
///      monomial order (smaller variable IDs have higher precedence).
///   3. Zero has no terms (empty coefficients, offsets = {0}, empty powers).
///   4. Flyweight Interning: Expressions are immutable and interned per
///      `PolyExprContext`. Pointer equality within a context implies canonical
///      polynomial equivalence.
///
/// Symbol Definitions & Normalization:
/// - `PolyExprContext` acts as an acyclic symbol table mapping variable IDs to
///   defining polynomials.
/// - Normalization recursively substitutes symbol definitions using binary
///   exponentiation, memoizing intermediate expressions for reuse across terms.

/// Unique 32-bit identifier for an indeterminate symbol variable \f$x_i\f$.
using VariableId = std::uint32_t;
/// Exponent for a variable factor \f$x_i^e\f$ in a monomial (must be >= 1 in canonical terms).
using Exponent = std::uint32_t;
/// Index offset into the flattened `powers` vector of `SparsePolynomialStorage`.
using Offset = std::uint32_t;

/// Represents a single variable factor raised to a power: \f$x_{\text{variable}}^{\text{exponent}}\f$.
struct Power {
  VariableId variable;
  Exponent exponent;

  /// Structural equality of variable indeterminate and exponent power.
  bool operator==(const Power &other) const {
    return variable == other.variable && exponent == other.exponent;
  }
};

/// Compressed Sparse Row (CSR) style flat storage for a multi-variate polynomial.
///
/// Monomial powers for term `i` span `powers[offsets[i] ... offsets[i + 1])`.
/// An empty power range (`offsets[i] == offsets[i + 1]`) represents a constant term's
/// empty monomial product (1). The zero polynomial has no terms (`coefficients` is
/// empty, `offsets = {0}`, and `powers` is empty).
template <class ExactInteger>
struct SparsePolynomialStorage {
  // Term i has one coefficient and powers[offsets[i] ... offsets[i + 1]).
  // Equal adjacent offsets represent a constant term's empty monomial.
  std::vector<ExactInteger> coefficients;
  // The trailing offset makes this size coefficients.size() + 1, even for zero.
  std::vector<Offset> offsets{0};
  std::vector<Power> powers;
};

class PolyExprContext;

/// An immutable integer polynomial in non-negative integer symbols. Expressions
/// are uniqued and owned by their context; all returned pointers expire with it.
/// Powers within a term have increasing variable IDs and nonzero exponents.
/// Terms have nonzero coefficients and are in descending lexicographic monomial
/// order (smaller variable IDs have precedence). Zero has no terms.
/// Pointer equality within a context compares canonical polynomials; use
/// PolyExprContext::areEqual to also account for symbol definitions.
class PolyExpr {
public:
  /// Arbitrary-precision integer storage allows coefficients to grow during
  /// polynomial arithmetic without overflow.
  using Storage = SparsePolynomialStorage<llvm::DynamicAPInt>;

  /// Returns the canonical zero polynomial (0) for the given context.
  static const PolyExpr *getZero(PolyExprContext &context);
  /// Returns the canonical constant polynomial for `value`. If `value == 0`,
  /// returns the canonical zero polynomial.
  static const PolyExpr *getConstant(PolyExprContext &context,
                                     const llvm::DynamicAPInt &value);
  /// Convenience overload constructing a constant polynomial from a 64-bit integer.
  static const PolyExpr *getConstant(PolyExprContext &context,
                                     std::int64_t value) {
    return getConstant(context, llvm::DynamicAPInt(value));
  }
  /// Returns the canonical atomic indeterminate symbol \f$1 \cdot x_{\text{variable}}^1\f$.
  static const PolyExpr *getSymbol(PolyExprContext &context,
                                   VariableId variable);
  /// Canonicalizes arbitrary sparse polynomial storage and returns the interned expression.
  ///
  /// Validates CSR offset structure, combines repeated variables within terms, combines
  /// like monomials, eliminates cancelled zero terms, and sorts terms into canonical
  /// descending lexicographic order. Returns `nullptr` on malformed offsets or exponent/offset
  /// overflow.
  static const PolyExpr *get(PolyExprContext &context, const Storage &storage);

  /// Returns the canonical underlying CSR storage.
  const Storage &getStorage() const { return storage; }
  /// Returns the enclosing context that owns this expression.
  PolyExprContext &getContext() const { return context; }
  /// Returns true if this is the zero polynomial (no terms).
  bool isZero() const { return storage.coefficients.empty(); }
  /// Returns true if this is a constant polynomial (no variable powers, including zero).
  bool isConstant() const { return storage.powers.empty(); }
  /// Returns true if this polynomial has exactly one non-zero term (including non-zero constants).
  bool isMonomial() const { return storage.coefficients.size() == 1; }
  /// Returns true if this expression is an atomic indeterminate symbol \f$1 \cdot x_i^1\f$.
  bool isSymbol() const;
  /// Returns the integer value if this is a constant expression (including 0), or `std::nullopt`.
  std::optional<llvm::DynamicAPInt> getConstantValue() const;

  /// Computes polynomial addition (\f$\text{this} + \text{rhs}\f$).
  /// Operands must share a context. Coefficient arithmetic is exact. Returns
  /// `nullptr` for exponent/offset overflow or mismatched contexts.
  const PolyExpr *add(const PolyExpr &rhs) const;
  /// Computes polynomial subtraction (\f$\text{this} - \text{rhs}\f$).
  /// Operands must share a context. Coefficient arithmetic is exact. Returns
  /// `nullptr` for exponent/offset overflow or mismatched contexts.
  const PolyExpr *subtract(const PolyExpr &rhs) const;
  /// Computes polynomial multiplication (\f$\text{this} \times \text{rhs}\f$).
  /// Distributes term pairs and combines like monomials. Operands must share a
  /// context. Returns `nullptr` for exponent/offset overflow or mismatched contexts.
  const PolyExpr *multiply(const PolyExpr &rhs) const;
  /// Multivariate polynomial exact division over \f$\mathbb{Z}\f$ without consulting
  /// symbol definitions.
  ///
  /// Performs multivariate long division using leading terms in canonical monomial order.
  /// Returns `nullptr` for division by zero, non-integral quotient coefficients, non-zero
  /// remainder, or mismatched contexts.
  const PolyExpr *exactDiv(const PolyExpr &rhs) const;

  PolyExpr(const PolyExpr &) = delete;
  PolyExpr &operator=(const PolyExpr &) = delete;

private:
  friend class PolyExprContext;
  PolyExpr(PolyExprContext &context, Storage storage);
  PolyExprContext &context;
  Storage storage;
};

/// Manages the lifetime, interning, and acyclic symbol definitions of polynomial expressions.
///
/// PolyExprContext provides:
/// - Unique storage (hash-consing) for `PolyExpr` instances.
/// - An acyclic symbol definition registry mapping `VariableId -> PolyExpr`.
/// - Recursive normalization (substitution) with memoization.
/// - Semantic equivalence testing modulo symbol definitions.
///
/// Thread Safety: Not thread-safe. Concurrent access must be synchronized externally.
class PolyExprContext {
public:
  PolyExprContext();
  ~PolyExprContext();
  PolyExprContext(const PolyExprContext &) = delete;
  PolyExprContext &operator=(const PolyExprContext &) = delete;

  /// Add a definition once. Return false for redefinition, a foreign context or
  /// a cycle, leaving existing definitions and normalization caches unchanged.
  /// References to symbols that have not yet been defined are allowed.
  ///
  /// Cycle detection traverses the transitive dependency graph prior to mutating
  /// internal state, ensuring atomic rejection. Successfully adding a definition
  /// invalidates the normalization cache.
  bool addDefinition(VariableId variable, const PolyExpr &definition);
  /// Returns the bound definition for `variable`, or `nullptr` if undefined.
  const PolyExpr *getDefinition(VariableId variable) const;

  /// Recursively substitute definitions, caching intermediate expressions.
  /// Return nullptr on exponent or offset overflow (including intermediates)
  /// or a foreign context. Adding a definition invalidates this cache.
  ///
  /// Uses exponentiation by squaring to evaluate powers of substituted expressions.
  const PolyExpr *normalize(const PolyExpr &expression);
  /// Tests whether `lhs` and `rhs` are semantically equivalent under current definitions.
  /// Evaluates pointer equality as a fast path. If unequal, normalizes both expressions
  /// and compares canonical normalized pointers.
  /// Returns `std::nullopt` if normalization overflows or an operand has a foreign context.
  std::optional<bool> areEqual(const PolyExpr &lhs, const PolyExpr &rhs);

private:
  friend class PolyExpr;
  const PolyExpr *intern(PolyExpr::Storage storage);
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_ANALYSIS_POLYEXPRS_H
