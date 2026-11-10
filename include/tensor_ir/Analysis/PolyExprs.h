// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
///   defining polynomials, and holds a reduction basis of polynomials known to
///   be zero.
/// - Normalization recursively substitutes symbol definitions using binary
///   exponentiation, memoizing intermediate expressions for reuse across terms,
///   and then reduces the result modulo the reduction basis.
/// - A frozen context accepts no further definitions or basis elements, so
///   normalized forms, and the identities of interned normalized expressions,
///   stay fixed.

#ifndef TENSOR_IR_ANALYSIS_POLYEXPRS_H
#define TENSOR_IR_ANALYSIS_POLYEXPRS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace mlir::nv_tensor_ir {

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
  /// Term i has one coefficient and powers[offsets[i] ... offsets[i + 1]).
  /// Equal adjacent offsets represent a constant term's empty monomial.
  std::vector<ExactInteger> coefficients;
  /// The trailing offset makes this size coefficients.size() + 1, even for zero.
  std::vector<Offset> offsets{0};
  std::vector<Power> powers;
};

class PolyExprContext;
class PolyExprImpl;

/// A copyable, pointer-sized handle to an immutable, context-owned polynomial.
/// The context must outlive its handles. A default-constructed handle is invalid;
/// use operator bool to test validity. Invalid is distinct from the zero polynomial.
/// Equality compares interned identity, without substituting symbol definitions;
/// use PolyExprContext::areEqual for semantic equality under the definitions
/// and the reduction basis of the context.
class PolyExpr {
public:
  using Storage = SparsePolynomialStorage<llvm::DynamicAPInt>;

  constexpr PolyExpr(std::nullptr_t = nullptr) {}
  explicit operator bool() const { return impl != nullptr; }
  friend bool operator==(PolyExpr lhs, PolyExpr rhs) {
    return lhs.impl == rhs.impl;
  }
  friend bool operator!=(PolyExpr lhs, PolyExpr rhs) { return !(lhs == rhs); }

  static PolyExpr getZero(PolyExprContext &context);
  static PolyExpr getConstant(PolyExprContext &context,
                              const llvm::DynamicAPInt &value);
  static PolyExpr getConstant(PolyExprContext &context, int64_t value) {
    return getConstant(context, llvm::DynamicAPInt(value));
  }
  static PolyExpr getSymbol(PolyExprContext &context, VariableId variable);
  /// Canonicalizes CSR storage, combining repeated powers and like terms,
  /// removing zero terms, and sorting. Returns invalid for malformed offsets
  /// or exponent/offset overflow.
  static PolyExpr get(PolyExprContext &context, const Storage &storage);

  /// A read-only term view. The context owns its coefficient and powers.
  struct Term {
    const llvm::DynamicAPInt &coefficient;
    llvm::ArrayRef<Power> powers;
  };

  class TermIterator
      : public llvm::iterator_facade_base<TermIterator,
                                          std::random_access_iterator_tag, Term,
                                          std::ptrdiff_t, const Term *, Term> {
  public:
    TermIterator() = default;
    TermIterator &operator+=(std::ptrdiff_t offset) {
      index += offset;
      return *this;
    }
    TermIterator &operator-=(std::ptrdiff_t offset) {
      index -= offset;
      return *this;
    }
    using iterator_facade_base::operator-;
    std::ptrdiff_t operator-(const TermIterator &other) const {
      assert(impl == other.impl &&
             "expected iterators into the same polynomial");
      return static_cast<std::ptrdiff_t>(index) -
             static_cast<std::ptrdiff_t>(other.index);
    }
    Term operator*() const;
    Term operator[](std::ptrdiff_t offset) const { return *(*this + offset); }
    bool operator==(const TermIterator &other) const {
      return impl == other.impl && index == other.index;
    }
    bool operator<(const TermIterator &other) const {
      assert(impl == other.impl &&
             "expected iterators into the same polynomial");
      return index < other.index;
    }

  private:
    friend class PolyExpr;
    TermIterator(const PolyExprImpl *impl, size_t index)
        : impl(impl), index(index) {}
    const PolyExprImpl *impl = nullptr;
    size_t index = 0;
  };

  /// Iterates canonical terms without substituting symbol definitions. Zero
  /// has no terms; constants have no powers. Requires a valid handle, and the
  /// context must outlive the iterators and term views.
  TermIterator begin() const;
  TermIterator end() const;

  /// Number of canonical terms. Requires a valid handle.
  size_t size() const;
  /// Returns a term without substituting definitions. Requires a valid handle
  /// and an index less than size(). The context owns the returned term view.
  Term operator[](size_t index) const;

  /// Requires a valid handle.
  PolyExprContext &getContext() const;

  /// Structural queries on the stored polynomial, without substituting symbol
  /// definitions. Use normalize() first to query after substitution. Invalid
  /// handles return false or std::nullopt. Integer extraction additionally
  /// checks the int64_t range.
  bool isZero() const;
  bool isConstant() const;
  bool isMonomial() const;
  bool isSymbol() const;
  std::optional<llvm::DynamicAPInt> getConstantValue() const;
  std::optional<int64_t> getConstantInt64() const;
  bool isOne() const;
  /// Sorted, unique variables of the stored terms, without substituting
  /// definitions. Empty for an invalid handle.
  llvm::SmallVector<VariableId> getVariables() const;
  /// Divides the stored polynomial by the gcd of its coefficients and makes
  /// its leading coefficient positive, without substituting definitions.
  /// Zero and invalid handles are returned unchanged.
  PolyExpr getPrimitivePart() const;

  /// Opaque pointer identifying the interned polynomial.
  const void *getAsOpaquePointer() const { return impl; }

  /// Substitutes definitions and reduces modulo the reduction basis; returns
  /// invalid on normalization failure.
  PolyExpr normalize() const;
  /// Prints with definitions substituted and arbitrary-precision coefficients,
  /// or <invalid> for an invalid handle or failed normalization.
  void print(llvm::raw_ostream &os) const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, PolyExpr expr) {
    expr.print(os);
    return os;
  }

  /// Raw polynomial arithmetic, without substituting symbol definitions.
  /// Invalid operands, different contexts and exponent/offset overflow return
  /// invalid. Division additionally requires an exact quotient over integers.
  PolyExpr add(PolyExpr rhs) const;
  PolyExpr subtract(PolyExpr rhs) const;
  PolyExpr multiply(PolyExpr rhs) const;
  PolyExpr exactDiv(PolyExpr rhs) const;

  /// Arithmetic after substituting current symbol definitions. Propagates
  /// invalid handles, normalization failures and the raw arithmetic failures
  /// above. Division is exact, not truncating; a nonzero remainder is invalid.
  PolyExpr operator+(PolyExpr rhs) const;
  PolyExpr operator-(PolyExpr rhs) const;
  PolyExpr operator*(PolyExpr rhs) const;
  PolyExpr operator/(PolyExpr rhs) const;
  PolyExpr operator-() const;
  PolyExpr &operator+=(PolyExpr rhs) { return *this = *this + rhs; }
  PolyExpr &operator-=(PolyExpr rhs) { return *this = *this - rhs; }
  PolyExpr &operator*=(PolyExpr rhs) { return *this = *this * rhs; }
  PolyExpr &operator/=(PolyExpr rhs) { return *this = *this / rhs; }

private:
  friend class PolyExprContext;
  explicit PolyExpr(const PolyExprImpl *impl) : impl(impl) {}
  const PolyExprImpl *impl = nullptr;
};

/// Manages the lifetime, interning, and acyclic symbol definitions of polynomial expressions.
///
/// PolyExprContext provides:
/// - Unique storage (hash-consing) for `PolyExprImpl` instances.
/// - An acyclic symbol definition registry mapping `VariableId -> PolyExpr`.
/// - A reduction basis of polynomials known to evaluate to zero.
/// - Recursive normalization (substitution, then reduction) with memoization.
/// - Three-valued semantic queries modulo definitions and the basis.
///
/// Symbols range over non-negative integers. Definitions precede basis
/// elements, and both precede `freeze()`; adding either afterwards is a
/// programming error.
///
/// Thread Safety: Not thread-safe. Concurrent access must be synchronized externally.
class PolyExprContext {
public:
  PolyExprContext();
  ~PolyExprContext();
  PolyExprContext(const PolyExprContext &) = delete;
  PolyExprContext &operator=(const PolyExprContext &) = delete;

  /// Add a definition once. Return false for redefinition, an invalid handle, a foreign context or
  /// a cycle, leaving existing definitions and normalization caches unchanged.
  /// References to symbols that have not yet been defined are allowed.
  /// Requires an unfrozen context with an empty reduction basis.
  ///
  /// Cycle detection traverses the transitive dependency graph prior to mutating
  /// internal state, ensuring atomic rejection. Successfully adding a definition
  /// invalidates the normalization cache.
  bool addDefinition(VariableId variable, PolyExpr definition);
  /// Returns the bound definition for `variable`, or an invalid handle if undefined.
  PolyExpr getDefinition(VariableId variable) const;

  /// Adds `zero`, a polynomial that evaluates to zero, to the reduction basis.
  /// The basis keeps the primitive part of `zero` after normalization, with a
  /// positive leading coefficient. Normalization then rewrites every term
  /// whose monomial and coefficient are multiples of the leading term of a
  /// basis element. Reduction is sound but not complete: without a completion
  /// of the basis, polynomials that are equal modulo the basis may keep
  /// different normal forms. A `zero` that already normalizes to zero adds
  /// nothing. Returns false, leaving the basis unchanged, for an invalid or
  /// foreign handle, a normalization failure, or a `zero` that normalizes to
  /// a nonzero constant. Requires an unfrozen context. Successfully adding an
  /// element invalidates the normalization cache.
  bool addReduction(PolyExpr zero);

  /// Fixes the definitions and the reduction basis. Afterwards normalization
  /// is a function of the expression alone.
  void freeze();
  bool isFrozen() const;

  /// Recursively substitute definitions, caching intermediate expressions, and
  /// reduce the result modulo the reduction basis.
  /// Return an invalid handle on exponent or offset overflow (including intermediates)
  /// or an invalid/foreign handle. Adding a definition or a basis element
  /// invalidates this cache.
  ///
  /// Uses exponentiation by squaring to evaluate powers of substituted expressions.
  PolyExpr normalize(PolyExpr expression);
  /// Three-valued equality modulo definitions and the reduction basis: true
  /// when `lhs - rhs` normalizes to zero, false when it normalizes to a
  /// nonzero constant, and `std::nullopt` otherwise. Identical handles are
  /// equal without normalizing. Also `std::nullopt` if normalization overflows
  /// or an operand is invalid or has a foreign context.
  std::optional<bool> areEqual(PolyExpr lhs, PolyExpr rhs);
  /// Three-valued test of `expression == 1`, as areEqual.
  std::optional<bool> isOne(PolyExpr expression);

private:
  friend class PolyExpr;
  friend class PolyExprImpl;
  const PolyExprImpl *intern(PolyExpr::Storage storage);
  bool addDefinitionImpl(VariableId variable, const PolyExprImpl &definition);
  const PolyExprImpl *getDefinitionImpl(VariableId variable) const;
  const PolyExprImpl *normalizeImpl(const PolyExprImpl &expression);
  const PolyExprImpl *reduceImpl(const PolyExprImpl &expression);
  std::optional<bool> areEqualImpl(const PolyExprImpl &lhs,
                                   const PolyExprImpl &rhs);
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::nv_tensor_ir

namespace llvm {
template <>
struct DenseMapInfo<mlir::nv_tensor_ir::PolyExpr> {
  using PolyExpr = mlir::nv_tensor_ir::PolyExpr;
  static unsigned getHashValue(PolyExpr expression) {
    return DenseMapInfo<const void *>::getHashValue(
        expression.getAsOpaquePointer());
  }
  static bool isEqual(PolyExpr lhs, PolyExpr rhs) { return lhs == rhs; }
};
} // namespace llvm

#endif // TENSOR_IR_ANALYSIS_POLYEXPRS_H
