// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/PolyExprs.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <set>
#include <utility>

using namespace mlir::nv_tensor_ir;

class mlir::nv_tensor_ir::PolyExprImpl {
public:
  /// Arbitrary-precision integer storage allows coefficients to grow during
  /// polynomial arithmetic without overflow.
  using Storage = SparsePolynomialStorage<llvm::DynamicAPInt>;

  /// Returns the canonical zero polynomial (0) for the given context.
  static const PolyExprImpl *getZero(PolyExprContext &context);
  /// Returns the canonical constant polynomial for `value`. If `value == 0`,
  /// returns the canonical zero polynomial.
  static const PolyExprImpl *getConstant(PolyExprContext &context,
                                         const llvm::DynamicAPInt &value);
  /// Convenience overload constructing a constant polynomial from a 64-bit integer.
  static const PolyExprImpl *getConstant(PolyExprContext &context,
                                         std::int64_t value) {
    return getConstant(context, llvm::DynamicAPInt(value));
  }
  /// Returns the canonical atomic indeterminate symbol \f$1 \cdot x_{\text{variable}}^1\f$.
  static const PolyExprImpl *getSymbol(PolyExprContext &context,
                                       VariableId variable);
  /// Canonicalizes arbitrary sparse polynomial storage and returns the interned expression.
  ///
  /// Validates CSR offset structure, combines repeated variables within terms, combines
  /// like monomials, eliminates cancelled zero terms, and sorts terms into canonical
  /// descending lexicographic order. Returns `nullptr` on malformed offsets or exponent/offset
  /// overflow.
  static const PolyExprImpl *get(PolyExprContext &context,
                                 const Storage &storage);

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

  /// Returns the stored constant value, or std::nullopt for non-constants or
  /// values outside int64_t, without substituting symbol definitions.
  std::optional<int64_t> getConstantInt64() const;
  /// Tests whether the stored polynomial is one, without substituting definitions.
  bool isOne() const;
  /// Prints the polynomial after substituting symbol definitions, or <invalid>
  /// on normalization failure. Coefficients retain arbitrary precision.
  void print(llvm::raw_ostream &os) const;

  /// Computes polynomial addition (\f$\text{this} + \text{rhs}\f$).
  /// Operands must share a context. Coefficient arithmetic is exact. Returns
  /// `nullptr` for exponent/offset overflow or mismatched contexts.
  const PolyExprImpl *add(const PolyExprImpl &rhs) const;
  /// Computes polynomial subtraction (\f$\text{this} - \text{rhs}\f$).
  /// Operands must share a context. Coefficient arithmetic is exact. Returns
  /// `nullptr` for exponent/offset overflow or mismatched contexts.
  const PolyExprImpl *subtract(const PolyExprImpl &rhs) const;
  /// Computes polynomial multiplication (\f$\text{this} \times \text{rhs}\f$).
  /// Distributes term pairs and combines like monomials. Operands must share a
  /// context. Returns `nullptr` for exponent/offset overflow or mismatched contexts.
  const PolyExprImpl *multiply(const PolyExprImpl &rhs) const;
  /// Multivariate polynomial exact division over \f$\mathbb{Z}\f$ without consulting
  /// symbol definitions.
  ///
  /// Performs multivariate long division using leading terms in canonical monomial order.
  /// Returns `nullptr` for division by zero, non-integral quotient coefficients, non-zero
  /// remainder, or mismatched contexts.
  const PolyExprImpl *exactDiv(const PolyExprImpl &rhs) const;

  PolyExprImpl(const PolyExprImpl &) = delete;
  PolyExprImpl &operator=(const PolyExprImpl &) = delete;

private:
  friend class PolyExprContext;
  PolyExprImpl(PolyExprContext &context, Storage storage);
  PolyExprContext &context;
  Storage storage;
};

/// \file
/// Implementation of multi-variate integer polynomial operations, canonicalization,
/// arithmetic, multivariate exact division, and symbol normalization in TensorIR.
///
/// Implementation Overview:
/// - Working Representation (`Terms`): Arithmetic operations (`add`, `subtract`,
///   `multiply`, `exactDiv`) unpack canonical CSR storage into an ordered associative
///   map `std::map<Monomial, llvm::DynamicAPInt, MonomialOrder>`.
/// - Leading Term Extraction: `MonomialOrder` arranges monomials in descending
///   lexicographic order. This ensures `terms.begin()` yields the leading monomial
///   in O(1), enabling the multivariate polynomial division algorithm over Z.
/// - Invariant Preservation: `accumulateTerm` immediately erases any term whose
///   coefficient sums to zero. Consequently, zero polynomials are uniquely represented
///   by an empty map, and `terms.begin()` is guaranteed to point to a non-zero term.
/// - Interning & Hash-Consing: Canonical expressions are uniqued in `PolyExprContext`
///   using `ExpressionOrder` with transparent lookup (`is_transparent`), avoiding
///   allocations during deduplication probes.
/// - DenseMap Sentinel Safety: Symbol IDs are widened to `uint64_t` in symbol tables
///   to avoid collisions with `DenseMap` empty (~0ULL) and tombstone (~0ULL - 1ULL) keys.
/// - Cycle Detection: Transitive dependency traversal prevents cyclic symbol definitions
///   from being inserted into the context.
/// - Normalization: Evaluates symbol substitutions using binary exponentiation
///   (exponentiation by squaring), avoiding unnecessary final squarings to protect
///   against exponent overflow.

/// A monomial is a sequence of variable powers sorted by strictly increasing VariableId.
using Monomial = std::vector<Power>;

namespace {
/// Strict weak ordering on monomials in descending graded/lexicographic order.
///
/// Within a monomial, variable powers are sorted by ascending `VariableId`.
/// When comparing two monomials:
/// - Absent variables are treated as having exponent zero.
/// - The smaller `VariableId` has higher precedence (lexicographical order). If variable IDs
///   differ, the monomial containing the smaller ID has a positive exponent where the other
///   has zero, so it is considered strictly greater (returns true).
/// - If variable IDs match, the monomial with the higher exponent is considered greater.
/// - If all shared variable powers match, the monomial with remaining variables has
///   higher total degree and is considered greater.
/// - The empty monomial (representing a constant factor of 1) is the minimum element.
///
/// This ordering guarantees that for non-zero polynomials, the leading term is unique
/// and strictly decreases during multivariate polynomial long division.
struct MonomialOrder {
  bool operator()(const Monomial &lhs, const Monomial &rhs) const {
    size_t i = 0, j = 0;
    while (i < lhs.size() && j < rhs.size()) {
      if (lhs[i].variable != rhs[j].variable) {
        // The smaller ID has a positive exponent where the other side has zero.
        return lhs[i].variable < rhs[j].variable;
      }
      if (lhs[i].exponent != rhs[j].exponent) {
        return lhs[i].exponent > rhs[j].exponent;
      }
      ++i;
      ++j;
    }
    return i < lhs.size();
  }
};
} // namespace

/// Working representation for polynomial arithmetic.
///
/// An ordered map maintains terms in descending canonical monomial order, combining
/// like terms upon insertion and providing constant-time access to the leading term
/// via `begin()`. Interned expressions use compact CSR storage (`PolyExprImpl::Storage`).
using Terms = std::map<Monomial, llvm::DynamicAPInt, MonomialOrder>;

/// Adds `coefficient` to `monomial` in the working `terms` map.
///
/// If `coefficient` cancels an existing term's coefficient to zero, the term is
/// immediately erased from `terms`. This preserves two critical invariants:
/// 1. The zero polynomial is uniquely represented as an empty map.
/// 2. `terms.begin()` never returns a term with a zero coefficient, ensuring the
///    correct leading term is selected during polynomial division.
static void accumulateTerm(Terms &terms, const Monomial &monomial,
                           const llvm::DynamicAPInt &coefficient) {
  auto it = terms.find(monomial);
  if (it == terms.end()) {
    if (coefficient != 0) {
      terms.emplace(monomial, coefficient);
    }
    return;
  }
  it->second += coefficient;
  // Remove cancellations immediately so zero has a unique representation and
  // division never selects a zero coefficient as its leading term.
  if (it->second == 0) {
    terms.erase(it);
  }
}

/// Decompresses flat CSR polynomial storage into the working `Terms` map.
///
/// Slices `powers[offsets[i] ... offsets[i + 1])` to reconstruct each term's monomial.
static Terms unpack(const PolyExprImpl &expression) {
  Terms terms;
  const auto &s = expression.getStorage();
  for (size_t i = 0; i < s.coefficients.size(); ++i) {
    terms.emplace(Monomial(s.powers.begin() + s.offsets[i],
                           s.powers.begin() + s.offsets[i + 1]),
                  s.coefficients[i]);
  }
  return terms;
}

/// Compresses a working `Terms` map into flat CSR polynomial storage.
///
/// Populates `coefficients`, `offsets`, and `powers`. Checks for index overflow
/// where total powers would exceed `std::numeric_limits<Offset>::max()`.
/// Returns `std::nullopt` on offset overflow.
static std::optional<PolyExprImpl::Storage> pack(const Terms &terms) {
  PolyExprImpl::Storage storage;
  for (const auto &term : terms) {
    // Coefficients are unbounded, but offsets into the flat power array are not.
    if (term.first.size() >
        std::numeric_limits<Offset>::max() - storage.powers.size()) {
      return std::nullopt;
    }
    storage.coefficients.push_back(term.second);
    storage.powers.insert(storage.powers.end(), term.first.begin(),
                          term.first.end());
    storage.offsets.push_back(static_cast<Offset>(storage.powers.size()));
  }
  return storage;
}

/// Multiplies two monomials: \f$\prod x_i^{a_i} \cdot \prod x_i^{b_i} = \prod x_i^{a_i + b_i}\f$.
///
/// Sums exponents for each variable using a widened 64-bit integer accumulator to detect
/// overflow before narrowing back to `Exponent`. Returns `std::nullopt` if any resulting
/// exponent exceeds `std::numeric_limits<Exponent>::max()`.
static std::optional<Monomial> multiplyMonomials(const Monomial &lhs,
                                                 const Monomial &rhs) {
  // Canonical inputs contain each variable once. Widen before adding exponents
  // so overflow is detected before narrowing back to the storage type.
  std::map<VariableId, uint64_t> exponents;
  for (const Power &power : lhs) {
    exponents[power.variable] = power.exponent;
  }
  for (const Power &power : rhs) {
    exponents[power.variable] += power.exponent;
  }
  Monomial result;
  for (auto [variable, exponent] : exponents) {
    if (exponent > std::numeric_limits<Exponent>::max()) {
      return std::nullopt;
    }
    result.push_back({variable, static_cast<Exponent>(exponent)});
  }
  return result;
}

/// Divides two monomials: \f$\prod x_i^{a_i} / \prod x_i^{b_i} = \prod x_i^{a_i - b_i}\f$.
///
/// Computes exact division of monomials over polynomials (requiring non-negative resulting
/// exponents). If `rhs` contains any variable absent from `lhs`, or if `rhs` has a strictly
/// greater exponent for any variable, exact division fails and returns `std::nullopt`.
/// Powers with exponent zero are omitted from the resulting canonical monomial.
static std::optional<Monomial> divideMonomials(const Monomial &lhs,
                                               const Monomial &rhs) {
  std::map<VariableId, Exponent> exponents;
  for (const Power &power : lhs) {
    exponents[power.variable] = power.exponent;
  }
  for (const Power &power : rhs) {
    if (exponents[power.variable] < power.exponent) {
      // A missing variable has exponent zero; negative exponents are not allowed.
      return std::nullopt;
    }
    exponents[power.variable] -= power.exponent;
  }
  Monomial result;
  for (auto [variable, exponent] : exponents) {
    if (exponent) {
      result.push_back({variable, exponent});
    }
  }
  return result;
}
namespace {
/// Comparator for uniquing interned `PolyExprImpl` instances in `PolyExprContext`.
///
/// Enables transparent lookup (`is_transparent`): `std::set::find` can search directly
/// with a `const PolyExprImpl::Storage &` without constructing or allocating a temporary
/// `PolyExprImpl` instance on the heap.
///
/// Note: This comparator defines a storage identity ordering (coefficients, offsets, powers)
/// for hash-consing and deduplication, distinct from `MonomialOrder` used in arithmetic.
struct ExpressionOrder {
  using is_transparent = void;

  static const PolyExprImpl::Storage &
  getStorage(const PolyExprImpl::Storage &storage) {
    return storage;
  }
  static const PolyExprImpl::Storage &
  getStorage(const std::unique_ptr<PolyExprImpl> &expr) {
    return expr->getStorage();
  }
  template <typename L, typename R>
  bool operator()(const L &lhs, const R &rhs) const {
    // This is a storage identity ordering, independent of the monomial order
    // used to select leading terms in arithmetic.
    const auto &left = getStorage(lhs);
    const auto &right = getStorage(rhs);
    if (left.coefficients != right.coefficients) {
      return left.coefficients < right.coefficients;
    }
    if (left.offsets != right.offsets) {
      return left.offsets < right.offsets;
    }
    return MonomialOrder{}(left.powers, right.powers);
  }
};
} // namespace

/// Internal implementation state for `PolyExprContext`.
struct PolyExprContext::Impl {
  /// Interning table storing unique canonical expressions. Pointers remain stable
  /// for the lifetime of the context.
  std::set<std::unique_ptr<PolyExprImpl>, ExpressionOrder> expressions;
  /// Symbol definitions mapping variable ID to defining polynomial.
  /// Keys are widened to `uint64_t` so that DenseMap's reserved empty (~0ULL) and
  /// tombstone (~0ULL - 1ULL) sentinel values can never collide with valid 32-bit `VariableId`s.
  llvm::DenseMap<uint64_t, const PolyExprImpl *> definitions;
  /// Memoization cache for `normalizeImpl()`. Maps unexpanded expression pointers to
  /// their canonical normalized forms. Stores `nullptr` to memoize failed expansions
  /// (e.g. exponent overflow). Cleared whenever definitions change.
  llvm::DenseMap<const PolyExprImpl *, const PolyExprImpl *> normalized;
};

PolyExprContext::PolyExprContext() : impl(std::make_unique<Impl>()) {}
PolyExprContext::~PolyExprContext() = default;

PolyExprImpl::PolyExprImpl(PolyExprContext &context, Storage storage)
    : context(context), storage(std::move(storage)) {}

/// Deduplicates and stores canonical storage in the context's flyweight set.
/// If an identical expression already exists, returns the existing pointer.
/// Otherwise, allocates and inserts a new `PolyExprImpl`.
const PolyExprImpl *PolyExprContext::intern(PolyExprImpl::Storage storage) {
  // Callers supply canonical storage, so an equal key reuses the same object.
  auto existing = impl->expressions.find(storage);
  if (existing != impl->expressions.end()) {
    return existing->get();
  }
  auto expression = std::unique_ptr<PolyExprImpl>(
      new PolyExprImpl(*this, std::move(storage)));
  auto *result = expression.get();
  impl->expressions.insert(std::move(expression));
  return result;
}

/// Zero is uniquely represented by empty coefficients, offsets {0}, and empty powers.
const PolyExprImpl *PolyExprImpl::getZero(PolyExprContext &context) {
  return context.intern({});
}

/// Constants have a single coefficient and an empty power range (offsets {0, 0}).
const PolyExprImpl *PolyExprImpl::getConstant(PolyExprContext &context,
                                              const llvm::DynamicAPInt &value) {
  if (value == 0) {
    return getZero(context);
  }
  return context.intern({{value}, {0, 0}, {}});
}

/// Symbols have coefficient 1 and a single power factor with exponent 1 (offsets {0, 1}).
const PolyExprImpl *PolyExprImpl::getSymbol(PolyExprContext &context,
                                            VariableId variable) {
  return context.intern({{llvm::DynamicAPInt(1)}, {0, 1}, {{variable, 1}}});
}

/// Canonicalizes arbitrary CSR storage into canonical form.
///
/// Performs comprehensive validation and normalization:
/// 1. Offset checks: verifies size, zero start, final offset, and monotonic sorting.
/// 2. Term canonicalization: iterates over terms, skipping zero coefficients.
/// 3. Power combination: groups powers by variable, sums exponents while checking for
///    overflow, sorts by variable ID, and strips zero exponents.
/// 4. Term aggregation: adds terms into `Terms`, combining like monomials and removing
///    coefficients that cancel to zero.
/// 5. Packing & interning: converts the normalized terms into canonical storage and interns it.
const PolyExprImpl *PolyExprImpl::get(PolyExprContext &context,
                                      const Storage &storage) {
  // Check the whole offset table before forming any ranges into powers.
  if (storage.offsets.size() != storage.coefficients.size() + 1 ||
      storage.offsets.front() != 0 ||
      storage.offsets.back() != storage.powers.size() ||
      !std::is_sorted(storage.offsets.begin(), storage.offsets.end())) {
    return nullptr;
  }
  Terms terms;
  for (size_t i = 0; i < storage.coefficients.size(); ++i) {
    if (storage.coefficients[i] == 0) {
      // Powers of a discarded term need no normalization or exponent checks.
      continue;
    }
    // Input powers may be unordered or repeated. Sort by ID while combining
    // exponents, then omit zero exponents before combining like terms.
    std::map<VariableId, Exponent> exponents;
    for (Offset j = storage.offsets[i]; j < storage.offsets[i + 1]; ++j) {
      const Power &power = storage.powers[j];
      auto &exponent = exponents[power.variable];
      if (power.exponent > std::numeric_limits<Exponent>::max() - exponent) {
        return nullptr;
      }
      exponent += power.exponent;
    }
    Monomial monomial;
    for (auto [variable, exponent] : exponents) {
      if (exponent) {
        monomial.push_back({variable, exponent});
      }
    }
    accumulateTerm(terms, monomial, storage.coefficients[i]);
  }
  auto result = pack(terms);
  return result ? context.intern(std::move(*result)) : nullptr;
}

/// Checks if expression is an atomic symbol: 1 * x_i^1.
bool PolyExprImpl::isSymbol() const {
  return isMonomial() && storage.coefficients[0] == 1 &&
         storage.powers.size() == 1 && storage.powers[0].exponent == 1;
}

/// Returns constant value if no variable powers exist, or nullopt for non-constants.
std::optional<llvm::DynamicAPInt> PolyExprImpl::getConstantValue() const {
  if (!isConstant()) {
    return std::nullopt;
  }
  return isZero() ? llvm::DynamicAPInt(0) : storage.coefficients[0];
}

std::optional<int64_t> PolyExprImpl::getConstantInt64() const {
  auto value = getConstantValue();
  if (!value || *value < std::numeric_limits<int64_t>::min() ||
      *value > std::numeric_limits<int64_t>::max()) {
    return std::nullopt;
  }
  return static_cast<int64_t>(*value);
}

bool PolyExprImpl::isOne() const { return getConstantValue() == 1; }

void PolyExprImpl::print(llvm::raw_ostream &os) const {
  const PolyExprImpl *expr = context.normalizeImpl(*this);
  if (!expr) {
    os << "<invalid>";
    return;
  }
  if (auto value = expr->getConstantValue()) {
    os << *value;
    return;
  }
  const auto &storage = expr->getStorage();
  for (size_t term = 0; term < storage.coefficients.size(); ++term) {
    const auto &coefficient = storage.coefficients[term];
    bool negative = coefficient < 0;
    if (term) {
      os << (negative ? " - " : " + ");
    } else if (negative) {
      os << '-';
    }
    auto magnitude = negative ? -coefficient : coefficient;
    bool hasPowers = storage.offsets[term] != storage.offsets[term + 1];
    if (magnitude != 1 || !hasPowers) {
      os << magnitude;
      if (hasPowers) {
        os << '*';
      }
    }
    for (Offset i = storage.offsets[term]; i < storage.offsets[term + 1]; ++i) {
      if (i != storage.offsets[term]) {
        os << '*';
      }
      const Power &power = storage.powers[i];
      os << 's' << power.variable;
      if (power.exponent != 1) {
        os << '^' << power.exponent;
      }
    }
  }
}

/// Adds two polynomials by unpacking into `Terms`, accumulating coefficients, and packing.
const PolyExprImpl *PolyExprImpl::add(const PolyExprImpl &rhs) const {
  if (&context != &rhs.context) {
    return nullptr;
  }
  Terms terms = unpack(*this);
  for (const auto &term : unpack(rhs)) {
    accumulateTerm(terms, term.first, term.second);
  }
  auto result = pack(terms);
  return result ? context.intern(std::move(*result)) : nullptr;
}

/// Subtracts two polynomials by unpacking into `Terms`, accumulating negated coefficients, and packing.
const PolyExprImpl *PolyExprImpl::subtract(const PolyExprImpl &rhs) const {
  if (&context != &rhs.context) {
    return nullptr;
  }
  Terms terms = unpack(*this);
  for (const auto &term : unpack(rhs)) {
    accumulateTerm(terms, term.first, -term.second);
  }
  auto result = pack(terms);
  return result ? context.intern(std::move(*result)) : nullptr;
}

/// Multiplies two polynomials using the Cauchy product.
/// Distributes every term of `lhs` against every term of `rhs`, multiplies coefficients
/// exactly, adds monomial exponents with overflow checking, and combines like terms.
const PolyExprImpl *PolyExprImpl::multiply(const PolyExprImpl &rhs) const {
  if (&context != &rhs.context) {
    return nullptr;
  }
  Terms terms;
  Terms right = unpack(rhs);
  // Distribute over term pairs and merge coincident monomials exactly.
  for (const auto &lhsTerm : unpack(*this)) {
    for (const auto &rhsTerm : right) {
      auto monomial = multiplyMonomials(lhsTerm.first, rhsTerm.first);
      if (!monomial) {
        return nullptr;
      }
      accumulateTerm(terms, *monomial, lhsTerm.second * rhsTerm.second);
    }
  }
  auto result = pack(terms);
  return result ? context.intern(std::move(*result)) : nullptr;
}

/// Multivariate polynomial long division over Z without consulting symbol definitions.
///
/// Algorithm:
/// - Divisor `rhs` must be non-zero and share this context.
/// - Iteratively cancels the leading term of the `remainder` (initially `*this`) using
///   the leading term of `divisor` (`rhs`):
///   1. Identifies leading terms `lead` from `remainder` and `divisorLead` from `divisor`.
///   2. Checks if `divisorLead.first` divides `lead.first` with non-negative exponents.
///   3. Checks if `lead.second` is exactly divisible by `divisorLead.second` over Z.
///   4. Computes quotient term `q = coefficient * monomial`.
///   5. Subtracts `q * divisor` from `remainder`. This exactly eliminates `lead` and leaves
///      all remaining terms strictly smaller in the monomial order.
/// - If `remainder` becomes empty, division succeeds and returns canonical quotient.
/// - If any division step fails, exact division over Z is impossible; returns `nullptr`.
const PolyExprImpl *PolyExprImpl::exactDiv(const PolyExprImpl &rhs) const {
  if (&context != &rhs.context || rhs.isZero()) {
    return nullptr;
  }
  Terms remainder = unpack(*this), divisor = unpack(rhs), quotient;
  // Single-divisor long division: an exact quotient must cancel each leading
  // term with non-negative exponents and an integral coefficient.
  while (!remainder.empty()) {
    const auto &lead = *remainder.begin();
    const auto &divisorLead = *divisor.begin();
    auto monomial = divideMonomials(lead.first, divisorLead.first);
    if (!monomial || lead.second % divisorLead.second != 0) {
      return nullptr;
    }
    auto coefficient = lead.second / divisorLead.second;
    accumulateTerm(quotient, *monomial, coefficient);
    // Subtract the entire divisor multiple. The leading term disappears and
    // every remaining term is smaller in the multiplicative monomial order.
    for (const auto &term : divisor) {
      auto powers = multiplyMonomials(*monomial, term.first);
      if (!powers) {
        return nullptr;
      }
      accumulateTerm(remainder, *powers, -(coefficient * term.second));
    }
  }
  auto result = pack(quotient);
  return result ? context.intern(std::move(*result)) : nullptr;
}

/// Looks up bound definition for `variable` in the context's symbol table.
const PolyExprImpl *
PolyExprContext::getDefinitionImpl(VariableId variable) const {
  auto it = impl->definitions.find(variable);
  return it == impl->definitions.end() ? nullptr : it->second;
}

/// Binds `variable` to `definition` if no cycle is created.
///
/// Cycle Detection:
/// Adding `variable -> definition` creates a cycle if and only if `variable` is
/// reachable from `definition` through existing transitive definitions.
/// Performs a depth-first search traversal through the expression dependency graph.
/// Visited symbols are tracked in `visited` to handle diamond/DAG dependencies in O(V + E).
///
/// Atomicity:
/// If a cycle or redefinition is detected, the method returns `false` before modifying
/// `definitions` or `normalized`.
///
/// Cache Invalidation:
/// When a definition is successfully added, `impl->normalized` is cleared because previously
/// normalized expressions may contain `variable` as an unexpanded leaf symbol.
bool PolyExprContext::addDefinitionImpl(VariableId variable,
                                        const PolyExprImpl &definition) {
  if (&definition.getContext() != this || getDefinitionImpl(variable)) {
    return false;
  }
  // Adding variable -> definition creates a cycle exactly when a dependency
  // reaches variable. Traverse first so rejected additions leave the context intact.
  std::vector<const PolyExprImpl *> pending{&definition};
  llvm::DenseSet<VariableId> visited;
  while (!pending.empty()) {
    const PolyExprImpl *expression = pending.back();
    pending.pop_back();
    for (const Power &power : expression->getStorage().powers) {
      if (power.variable == variable) {
        return false;
      }
      if (visited.insert(power.variable).second) {
        // Visit shared dependencies once; undefined symbols are leaves for now.
        if (const PolyExprImpl *dependency =
                getDefinitionImpl(power.variable)) {
          pending.push_back(dependency);
        }
      }
    }
  }
  impl->definitions.try_emplace(variable, &definition);
  // A formerly undefined symbol may occur anywhere in a cached expansion.
  // Clear all normalizations; the definition-independent interning table survives.
  impl->normalized.clear();
  return true;
}

/// Recursively substitutes defined symbols, expanding polynomials with binary exponentiation.
///
/// Substitution Algorithm:
/// - Checks `impl->normalized` for previously memoized results.
/// - Iterates over terms in `expression`:
///   - For each variable factor \f$x_v^e\f$, retrieves its definition if present and
///     recursively normalizes it; otherwise treats \f$x_v\f$ as an atomic symbol base.
///   - Computes \f$\text{base}^e\f$ via exponentiation by squaring in \f$O(\log e)\f$ multiplications.
///   - Optimization: skips the final squaring when \f$e == 0\f$ after right-shift, preventing
///     unnecessary multiplication that could trigger spurious exponent overflow.
///   - Multiplies term coefficient by the expanded power factor.
/// - Sums expanded terms into the result.
/// - Memoizes result in `impl->normalized` (including `nullptr` on overflow) before returning.
const PolyExprImpl *
PolyExprContext::normalizeImpl(const PolyExprImpl &expression) {
  if (&expression.getContext() != this) {
    return nullptr;
  }
  auto cached = impl->normalized.find(&expression);
  if (cached != impl->normalized.end()) {
    return cached->second;
  }
  const auto &storage = expression.getStorage();
  const PolyExprImpl *result = PolyExprImpl::getZero(*this);
  // Expand each coefficient times its substituted powers, then sum the terms.
  // Null checks stop further work after exponent or offset overflow.
  for (size_t i = 0; result && i < storage.coefficients.size(); ++i) {
    const PolyExprImpl *term =
        PolyExprImpl::getConstant(*this, storage.coefficients[i]);
    for (Offset j = storage.offsets[i]; term && j < storage.offsets[i + 1];
         ++j) {
      const Power &power = storage.powers[j];
      const PolyExprImpl *base = PolyExprImpl::getSymbol(*this, power.variable);
      if (const PolyExprImpl *definition = getDefinitionImpl(power.variable)) {
        // Acyclic definitions make recursion safe; memoization shares DAG work.
        base = normalizeImpl(*definition);
      }
      // Exponentiation by squaring avoids work proportional to the exponent.
      Exponent exponent = power.exponent;
      while (term && base && exponent) {
        if (exponent & 1) {
          term = term->multiply(*base);
        }
        exponent >>= 1;
        if (exponent) {
          // Avoid an unused final square, which could overflow unnecessarily.
          base = base->multiply(*base);
        }
      }
      if (!base) {
        // A failed substitution or square invalidates the containing term.
        term = nullptr;
      }
    }
    result = term ? result->add(*term) : nullptr;
  }
  impl->normalized.try_emplace(&expression, result);
  return result;
}

/// Tests semantic equality modulo symbol definitions.
///
/// Uses pointer equality as a fast path. If addresses differ, normalizes both expressions
/// and compares the resulting canonical pointers. Returns `std::nullopt` on overflow or
/// mismatched contexts.
std::optional<bool> PolyExprContext::areEqualImpl(const PolyExprImpl &lhs,
                                                  const PolyExprImpl &rhs) {
  if (&lhs.getContext() != this || &rhs.getContext() != this) {
    return std::nullopt;
  }
  if (&lhs == &rhs) {
    // Identical canonical expressions are equal even if expansion would fail.
    return true;
  }
  const PolyExprImpl *left = normalizeImpl(lhs);
  const PolyExprImpl *right = normalizeImpl(rhs);
  if (!left || !right) {
    return std::nullopt;
  }
  return left == right;
}

PolyExpr PolyExpr::getZero(PolyExprContext &context) {
  return PolyExpr(PolyExprImpl::getZero(context));
}

PolyExpr PolyExpr::getConstant(PolyExprContext &context,
                               const llvm::DynamicAPInt &value) {
  return PolyExpr(PolyExprImpl::getConstant(context, value));
}

PolyExpr PolyExpr::getSymbol(PolyExprContext &context, VariableId variable) {
  return PolyExpr(PolyExprImpl::getSymbol(context, variable));
}

PolyExpr PolyExpr::get(PolyExprContext &context, const Storage &storage) {
  return PolyExpr(PolyExprImpl::get(context, storage));
}

PolyExpr::Term PolyExpr::TermIterator::operator*() const {
  const auto &storage = impl->getStorage();
  return {storage.coefficients[index],
          llvm::ArrayRef<Power>(storage.powers)
              .slice(storage.offsets[index],
                     storage.offsets[index + 1] - storage.offsets[index])};
}

PolyExpr::TermIterator PolyExpr::begin() const {
  assert(impl && "expected a valid polynomial");
  return TermIterator(impl, 0);
}

PolyExpr::TermIterator PolyExpr::end() const {
  return TermIterator(impl, size());
}

size_t PolyExpr::size() const {
  assert(impl && "expected a valid polynomial");
  return impl->getStorage().coefficients.size();
}

PolyExpr::Term PolyExpr::operator[](size_t index) const {
  assert(impl && "expected a valid polynomial");
  assert(index < size() && "term index out of bounds");
  return *TermIterator(impl, index);
}

PolyExprContext &PolyExpr::getContext() const {
  assert(impl && "expected a valid polynomial");
  return impl->getContext();
}

bool PolyExpr::isZero() const { return impl && impl->isZero(); }

bool PolyExpr::isConstant() const { return impl && impl->isConstant(); }

bool PolyExpr::isMonomial() const { return impl && impl->isMonomial(); }

bool PolyExpr::isSymbol() const { return impl && impl->isSymbol(); }

bool PolyExpr::isOne() const { return impl && impl->isOne(); }

std::optional<llvm::DynamicAPInt> PolyExpr::getConstantValue() const {
  return impl ? impl->getConstantValue() : std::nullopt;
}

std::optional<int64_t> PolyExpr::getConstantInt64() const {
  return impl ? impl->getConstantInt64() : std::nullopt;
}

PolyExpr PolyExpr::normalize() const {
  return impl ? getContext().normalize(*this) : PolyExpr();
}

void PolyExpr::print(llvm::raw_ostream &os) const {
  if (!impl) {
    os << "<invalid>";
    return;
  }
  impl->print(os);
}

PolyExpr PolyExpr::add(PolyExpr rhs) const {
  return impl && rhs.impl ? PolyExpr(impl->add(*rhs.impl)) : PolyExpr();
}

PolyExpr PolyExpr::operator+(PolyExpr rhs) const {
  if (!impl || !rhs.impl || &getContext() != &rhs.getContext()) {
    return {};
  }
  return normalize().add(rhs.normalize());
}

PolyExpr PolyExpr::subtract(PolyExpr rhs) const {
  return impl && rhs.impl ? PolyExpr(impl->subtract(*rhs.impl)) : PolyExpr();
}

PolyExpr PolyExpr::operator-(PolyExpr rhs) const {
  if (!impl || !rhs.impl || &getContext() != &rhs.getContext()) {
    return {};
  }
  return normalize().subtract(rhs.normalize());
}

PolyExpr PolyExpr::multiply(PolyExpr rhs) const {
  return impl && rhs.impl ? PolyExpr(impl->multiply(*rhs.impl)) : PolyExpr();
}

PolyExpr PolyExpr::operator*(PolyExpr rhs) const {
  if (!impl || !rhs.impl || &getContext() != &rhs.getContext()) {
    return {};
  }
  return normalize().multiply(rhs.normalize());
}

PolyExpr PolyExpr::exactDiv(PolyExpr rhs) const {
  return impl && rhs.impl ? PolyExpr(impl->exactDiv(*rhs.impl)) : PolyExpr();
}

PolyExpr PolyExpr::operator/(PolyExpr rhs) const {
  if (!impl || !rhs.impl || &getContext() != &rhs.getContext()) {
    return {};
  }
  return normalize().exactDiv(rhs.normalize());
}

PolyExpr PolyExpr::operator-() const {
  return impl ? getZero(getContext()) - *this : PolyExpr();
}

bool PolyExprContext::addDefinition(VariableId variable, PolyExpr definition) {
  return definition && addDefinitionImpl(variable, *definition.impl);
}

PolyExpr PolyExprContext::getDefinition(VariableId variable) const {
  return PolyExpr(getDefinitionImpl(variable));
}

PolyExpr PolyExprContext::normalize(PolyExpr expression) {
  return expression ? PolyExpr(normalizeImpl(*expression.impl)) : PolyExpr();
}

std::optional<bool> PolyExprContext::areEqual(PolyExpr lhs, PolyExpr rhs) {
  return lhs && rhs ? areEqualImpl(*lhs.impl, *rhs.impl) : std::nullopt;
}
