// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Polynomial facts for index values and shaped dimensions. Construction
/// records the relations of every operation through ShapeRelationModels, in IR
/// order, solves them, turns the residuals it cannot solve into the reduction
/// basis of the context, and freezes the context; queries read the solved
/// state. Every symbol keeps its SSA origin; entry-block arguments form the
/// preferred basis of normalized expressions.

#include "tensor_ir/Analysis/ShapeOracle.h"

#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

#include "ShapeRelationModels.h"
#include <deque>
#include <limits>

using namespace mlir;
using namespace mlir::nv_tensor_ir;

/// Recursively translates an affine expression into polynomial arithmetic.
/// Floor, ceil, and mod require a positive constant divisor and, for symbolic
/// dividends, an exact division.
static FailureOr<PolyExpr> evaluateAffine(AffineExpr expression,
                                          ArrayRef<PolyExpr> operands,
                                          unsigned numDims,
                                          PolyExprContext &context) {
  if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
    return PolyExpr::getConstant(context, constant.getValue());
  }
  if (auto dim = dyn_cast<AffineDimExpr>(expression)) {
    return operands[dim.getPosition()];
  }
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expression)) {
    return operands[numDims + symbol.getPosition()];
  }
  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary) {
    return failure();
  }
  auto lhs = evaluateAffine(binary.getLHS(), operands, numDims, context);
  auto rhs = evaluateAffine(binary.getRHS(), operands, numDims, context);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  PolyExpr result;
  switch (binary.getKind()) {
  case AffineExprKind::Add:
    result = *lhs + *rhs;
    break;
  case AffineExprKind::Mul:
    result = *lhs * *rhs;
    break;
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    PolyExpr dividendExpr = lhs->normalize();
    PolyExpr divisorExpr = rhs->normalize();
    if (!dividendExpr || !divisorExpr) {
      return failure();
    }
    auto divisor = divisorExpr.getConstantValue();
    if (!divisor || *divisor <= 0) {
      return failure();
    }
    if (auto dividend = dividendExpr.getConstantValue()) {
      switch (binary.getKind()) {
      case AffineExprKind::FloorDiv:
        return PolyExpr::getConstant(context,
                                     llvm::floorDiv(*dividend, *divisor));
      case AffineExprKind::CeilDiv:
        return PolyExpr::getConstant(context,
                                     llvm::ceilDiv(*dividend, *divisor));
      default:
        return PolyExpr::getConstant(context, llvm::mod(*dividend, *divisor));
      }
    }
    if (binary.getKind() == AffineExprKind::Mod) {
      return (dividendExpr / divisorExpr)
                 ? FailureOr<PolyExpr>(PolyExpr::getZero(context))
                 : FailureOr<PolyExpr>(failure());
    }
    result = dividendExpr / divisorExpr;
    break;
  }
  default:
    return failure();
  }
  return result ? FailureOr<PolyExpr>(result) : FailureOr<PolyExpr>(failure());
}

/// State of the analysis: the columns and symbols of every index value and
/// shaped value dimension in the function, the relations between them, and
/// the classes of symbols that share a relation.
struct ShapeOracle::Impl : public nv_tensor_ir::detail::ShapeRelationBuilder {
  using ValueDim = std::pair<Value, std::optional<int64_t>>;

  /// `lhs == rhs`, recorded by `op`. A pending affine relation keeps `rhs`
  /// null until the divisions of `affine` become exact.
  struct Constraint {
    PolyExpr lhs;
    PolyExpr rhs;
    Operation *op = nullptr;
    /// Dimension of `op` reported when an assumption is contradicted.
    std::optional<int64_t> dim;
    /// Whether the constraint is an explicit shape assumption.
    bool assumption = false;
    /// Whether a side of the constraint cannot be represented.
    bool unrepresentable = false;
    AffineExpr affine;
    SmallVector<PolyExpr> affineOperands;
    unsigned numAffineDims = 0;
    /// Any symbol of the constraint; all of them share one class.
    std::optional<VariableId> anchor;
    /// Symbols whose binding retries the constraint.
    llvm::SmallSet<VariableId, 4> watched;
    bool queued = false;
    bool solved = false;
  };

  /// A failed constraint.
  struct Failure {
    unsigned constraint;
    /// The constant difference of a contradiction; null if unrepresentable.
    PolyExpr difference;
  };

  explicit Impl(FunctionOpInterface scope);

  // ShapeRelationBuilder.
  PolyExprContext &getContext() override { return context; }
  PolyExpr getExpr(Value value, std::optional<int64_t> dim) override {
    return getColumnExpr(value, dim);
  }
  using ShapeRelationBuilder::getExpr;
  void equate(PolyExpr lhs, PolyExpr rhs, Operation *op,
              std::optional<int64_t> dim, bool assumption) override;
  void equateAffine(Operation *op, Value result, std::optional<int64_t> dim,
                    AffineExpr expr, ArrayRef<PolyExpr> operands,
                    unsigned numDims) override;
  const ValueDimList &getArgumentDependencies() const override {
    return argumentDependencies;
  }
  bool hasOutOfRangeDim(Value value) override;

  // Symbols and columns.
  PolyExpr createSymbol(Value value, std::optional<int64_t> dim = std::nullopt);
  /// Expression of the column for `value` and `dim`, created on first use.
  PolyExpr getColumnExpr(Value value, std::optional<int64_t> dim);
  /// Creates the columns of an index value or of every dimension of a ranked
  /// shaped value.
  void createColumns(Value value);
  /// Expression of the extent that a dimension query with a constant,
  /// in-range index returns, or null for other values.
  PolyExpr getQueriedExtent(Value value);
  bool isArgument(VariableId symbol) const {
    return symbol < numArgumentSymbols;
  }
  /// Whether every symbol of `expression` stands for an entry-block argument.
  bool isOverArguments(PolyExpr expression) const;

  // Construction.
  void addConstraint(Constraint constraint);
  void populate(Operation *op);

  // Solver.
  void solve();
  void solve(unsigned index);
  void enqueue(unsigned index);
  void watch(unsigned index, PolyExpr expression);
  bool bind(VariableId variable, PolyExpr expression);
  /// Records that constraint `index` failed with `difference`.
  void fail(unsigned index, PolyExpr difference);
  /// Adds the residual of every unsolved constraint to the reduction basis.
  void addImpliedZeros();

  // Queries.
  /// Whether a symbol of `expression` belongs to a failed class.
  bool touchesFailure(PolyExpr expression) const;
  /// Normalized `expression`, or failure if it is invalid, foreign, touches a
  /// failed class, or fails to normalize.
  FailureOr<PolyExpr> normalize(PolyExpr expression) const;
  /// Normalized expression of a column, or failure.
  FailureOr<PolyExpr> getNormalizedColumn(ValueDim key) const;
  /// Emits the diagnostic of a failed constraint.
  void report(const Failure &failure) const;

  FunctionOpInterface scope;
  mutable PolyExprContext context;
  SmallVector<Symbol> symbols;
  /// Symbols below this bound stand for entry-block arguments.
  unsigned numArgumentSymbols = 0;
  DenseMap<ValueDim, PolyExpr> columns;
  ValueDimList argumentDependencies;
  /// Operations whose backward slices contain no out-of-range dimension.
  DenseSet<Operation *> validOps;

  SmallVector<Constraint> constraints;
  /// Constraints to retry when a symbol is bound.
  DenseMap<VariableId, SmallVector<unsigned>> watchers;
  std::deque<unsigned> worklist;

  /// Symbols related by a common constraint, compressed after population.
  llvm::IntEqClasses classes;
  /// The first failure of each failed class.
  DenseMap<unsigned, Failure> failures;
  /// Failures between constants, which belong to no class.
  SmallVector<Failure> constantFailures;
};

//===----------------------------------------------------------------------===//
// Symbols and columns
//===----------------------------------------------------------------------===//

PolyExpr ShapeOracle::Impl::createSymbol(Value value,
                                         std::optional<int64_t> dim) {
  assert(value && "every symbol must identify a value");
  VariableId id = symbols.size();
  symbols.push_back(Symbol{value, dim});
  classes.grow(symbols.size());
  return PolyExpr::getSymbol(context, id);
}

PolyExpr ShapeOracle::Impl::getColumnExpr(Value value,
                                          std::optional<int64_t> dim) {
  assert(value && "value must be non-null");
  ValueDim key(value, dim);
  if (auto it = columns.find(key); it != columns.end()) {
    return it->second;
  }
  PolyExpr expression;
  if (dim) {
    auto type = cast<ShapedType>(value.getType());
    assert(type.hasRank() && *dim >= 0 && *dim < type.getRank() &&
           "dimension must be within the rank");
    expression = type.isDynamicDim(*dim) ? createSymbol(value, *dim)
                                         : getConstant(type.getDimSize(*dim));
  } else {
    assert(value.getType().isIndex() && "value must be index-typed");
    if (auto constant = getConstantIntValue(value)) {
      expression = getConstant(*constant);
    } else if (PolyExpr source = getQueriedExtent(value)) {
      expression = source;
    } else {
      expression = createSymbol(value);
    }
  }
  columns.try_emplace(key, expression);
  return expression;
}

void ShapeOracle::Impl::createColumns(Value value) {
  if (value.getType().isIndex()) {
    (void)getColumnExpr(value, std::nullopt);
    return;
  }
  if (auto type = dyn_cast<ShapedType>(value.getType());
      type && type.hasRank()) {
    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      (void)getColumnExpr(value, dim);
    }
  }
}

PolyExpr ShapeOracle::Impl::getQueriedExtent(Value value) {
  Operation *op = value.getDefiningOp();
  auto match = op ? nv_tensor_ir::detail::matchConstantDim(op) : std::nullopt;
  if (!match) {
    return {};
  }
  auto [source, index] = *match;
  auto type = cast<ShapedType>(source.getType());
  if (!type.hasRank() || index < 0 || index >= type.getRank()) {
    return {};
  }
  return getColumnExpr(source, index);
}

bool ShapeOracle::Impl::isOverArguments(PolyExpr expression) const {
  assert(expression && "expression must be non-null");
  return llvm::all_of(expression.getVariables(), [&](VariableId variable) {
    return isArgument(variable);
  });
}

bool ShapeOracle::Impl::hasOutOfRangeDim(Value value) {
  SmallVector<Value> pending{value};
  SmallVector<Operation *> visitedOps;
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    Operation *op = current.getDefiningOp();
    if (!op) {
      // A block argument depends on the operands of its region's owner.
      op = cast<BlockArgument>(current).getOwner()->getParentOp();
      if (!op || op->getNumOperands() == 0) {
        continue;
      }
    }
    if (validOps.contains(op) || !visited.insert(op).second) {
      continue;
    }
    visitedOps.push_back(op);
    if (auto match = nv_tensor_ir::detail::matchConstantDim(op)) {
      auto [source, index] = *match;
      auto type = cast<ShapedType>(source.getType());
      if (index < 0 || (type.hasRank() && index >= type.getRank())) {
        return true;
      }
    }
    llvm::append_range(pending, op->getOperands());
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) {
          llvm::append_range(pending, block.getTerminator()->getOperands());
        }
      }
    }
  }
  validOps.insert(visitedOps.begin(), visitedOps.end());
  return false;
}

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

ShapeOracle::Impl::Impl(FunctionOpInterface scope) : scope(scope) {
  Region &body = scope.getFunctionBody();
  if (body.empty()) {
    return;
  }
  // Entry-block arguments come first, so they hold the lowest symbols and
  // form the basis that elimination preserves.
  for (BlockArgument argument : scope.getArguments()) {
    createColumns(argument);
    if (argument.getType().isIndex()) {
      argumentDependencies.emplace_back(argument, std::nullopt);
    } else if (auto type = dyn_cast<ShapedType>(argument.getType());
               type && type.hasRank()) {
      for (int64_t dim = 0; dim < type.getRank(); ++dim) {
        if (type.isDynamicDim(dim)) {
          argumentDependencies.emplace_back(argument, dim);
        }
      }
    }
  }
  numArgumentSymbols = symbols.size();
  for (Block &block : body) {
    if (&block != &body.front()) {
      for (BlockArgument argument : block.getArguments()) {
        createColumns(argument);
      }
    }
    for (Operation &op : block) {
      op.walk<WalkOrder::PreOrder>(
          [&](Operation *nested) { populate(nested); });
    }
  }
  classes.compress();
  solve();
  addImpliedZeros();
}

/// Creates the columns that `op` defines and records its relations.
void ShapeOracle::Impl::populate(Operation *op) {
  for (Value result : op->getResults()) {
    createColumns(result);
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument argument : block.getArguments()) {
        createColumns(argument);
      }
    }
  }
  nv_tensor_ir::detail::populateShapeRelations(op, *this);
}

void ShapeOracle::Impl::addConstraint(Constraint constraint) {
  for (PolyExpr expression : {constraint.lhs, constraint.rhs}) {
    if (!expression) {
      continue;
    }
    for (VariableId variable : expression.getVariables()) {
      if (constraint.anchor) {
        classes.join(*constraint.anchor, variable);
      } else {
        constraint.anchor = variable;
      }
    }
  }
  for (PolyExpr operand : constraint.affineOperands) {
    PolyExpr normalized = operand ? operand.normalize() : PolyExpr();
    if (!normalized) {
      constraint.unrepresentable = true;
      continue;
    }
    for (VariableId variable : normalized.getVariables()) {
      if (constraint.anchor) {
        classes.join(*constraint.anchor, variable);
      } else {
        constraint.anchor = variable;
      }
    }
  }
  constraints.push_back(std::move(constraint));
}

void ShapeOracle::Impl::equate(PolyExpr lhs, PolyExpr rhs, Operation *op,
                               std::optional<int64_t> dim, bool assumption) {
  Constraint constraint;
  constraint.lhs = lhs;
  constraint.rhs = rhs;
  constraint.op = op;
  constraint.dim = dim;
  constraint.assumption = assumption;
  constraint.unrepresentable = !lhs || !rhs;
  addConstraint(std::move(constraint));
}

void ShapeOracle::Impl::equateAffine(Operation *op, Value result,
                                     std::optional<int64_t> dim,
                                     AffineExpr expr,
                                     ArrayRef<PolyExpr> operands,
                                     unsigned numDims) {
  Constraint constraint;
  constraint.lhs = getColumnExpr(result, dim);
  constraint.op = op;
  constraint.affine = expr;
  constraint.affineOperands = llvm::to_vector(operands);
  constraint.numAffineDims = numDims;
  addConstraint(std::move(constraint));
}

//===----------------------------------------------------------------------===//
// Solver
//===----------------------------------------------------------------------===//

void ShapeOracle::Impl::solve() {
  for (unsigned index = 0, e = constraints.size(); index < e; ++index) {
    enqueue(index);
  }
  while (!worklist.empty()) {
    unsigned index = worklist.front();
    worklist.pop_front();
    solve(index);
  }
}

void ShapeOracle::Impl::enqueue(unsigned index) {
  Constraint &constraint = constraints[index];
  if (!constraint.queued && !constraint.solved) {
    constraint.queued = true;
    worklist.push_back(index);
  }
}

/// Retries constraint `index` whenever a symbol of `expression` is bound.
void ShapeOracle::Impl::watch(unsigned index, PolyExpr expression) {
  for (VariableId variable : expression.getVariables()) {
    if (constraints[index].watched.insert(variable).second) {
      watchers[variable].push_back(index);
    }
  }
}

/// Defines `variable` as the normalized `expression` and wakes the constraints
/// that watch it. An argument symbol may only be defined in terms of other
/// argument symbols, so expressions keep entry-block arguments as their basis.
bool ShapeOracle::Impl::bind(VariableId variable, PolyExpr expression) {
  expression = context.normalize(expression);
  if (!expression || (isArgument(variable) && !isOverArguments(expression)) ||
      !context.addDefinition(variable, expression)) {
    return false;
  }
  if (auto it = watchers.find(variable); it != watchers.end()) {
    for (unsigned index : it->second) {
      enqueue(index);
    }
  }
  return true;
}

void ShapeOracle::Impl::fail(unsigned index, PolyExpr difference) {
  const Constraint &constraint = constraints[index];
  if (!constraint.anchor) {
    // Both sides are constants, which only IR that fails verification
    // produces; no query can reach the contradiction.
    constantFailures.push_back(Failure{index, difference});
    return;
  }
  failures.try_emplace(classes[*constraint.anchor], Failure{index, difference});
}

/// Solves constraint `index` by isolating a linear symbol `c * v` of the
/// normalized difference and binding `v`. Symbols with higher IDs are
/// eliminated first, which keeps entry-block arguments as the basis. Only
/// constant coefficients are divided out: a runtime factor may be zero.
void ShapeOracle::Impl::solve(unsigned index) {
  constraints[index].queued = false;
  Constraint &constraint = constraints[index];
  if (constraint.anchor && failures.count(classes[*constraint.anchor])) {
    return;
  }
  if (constraint.unrepresentable) {
    fail(index, PolyExpr());
    return;
  }
  if (!constraint.rhs) {
    auto expression =
        evaluateAffine(constraint.affine, constraint.affineOperands,
                       constraint.numAffineDims, context);
    if (failed(expression)) {
      // Retry once a binding makes the divisions exact.
      for (PolyExpr operand : constraint.affineOperands) {
        watch(index, context.normalize(operand));
      }
      return;
    }
    constraint.rhs = *expression;
  }
  PolyExpr lhs = constraint.lhs;
  PolyExpr rhs = constraint.rhs;
  PolyExpr difference = context.normalize(lhs - rhs);
  if (!difference) {
    fail(index, PolyExpr());
    return;
  }
  if (difference.isZero()) {
    constraints[index].solved = true;
    return;
  }
  if (difference.isConstant()) {
    fail(index, difference);
    return;
  }
  SmallVector<std::pair<VariableId, PolyExpr>> candidates;
  for (PolyExpr::Term term : difference) {
    if (term.powers.size() != 1 || term.powers.front().exponent != 1) {
      continue;
    }
    VariableId variable = term.powers.front().variable;
    PolyExpr coefficient = PolyExpr::getConstant(context, term.coefficient);
    PolyExpr monomial = PolyExpr::getSymbol(context, variable) * coefficient;
    PolyExpr remainder = difference - monomial;
    PolyExpr expression = -remainder / coefficient;
    if (expression) {
      candidates.emplace_back(variable, expression);
    }
  }
  llvm::sort(candidates, [](const auto &lhs, const auto &rhs) {
    return lhs.first > rhs.first;
  });
  for (auto [variable, expression] : candidates) {
    if (bind(variable, expression)) {
      constraints[index].solved = true;
      return;
    }
  }
  // Retry once a symbol of the remaining residual is bound.
  watch(index, difference);
}

/// Adds the residual of every unsolved constraint in a class that has not
/// failed to the reduction basis, in constraint order. Each residual is
/// reduced by the residuals before it; one that reduces to a nonzero constant
/// contradicts them and fails its class.
void ShapeOracle::Impl::addImpliedZeros() {
  for (auto [index, constraint] : llvm::enumerate(constraints)) {
    if (constraint.solved || !constraint.rhs || !constraint.anchor ||
        failures.count(classes[*constraint.anchor])) {
      continue;
    }
    PolyExpr residual = context.normalize(constraint.lhs - constraint.rhs);
    if (!residual) {
      continue;
    }
    if (residual.isConstant()) {
      if (!residual.isZero()) {
        fail(index, residual);
      }
    } else {
      bool added = context.addReduction(residual);
      assert(added && "a normalized non-constant residual is a basis element");
      (void)added;
    }
  }
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

bool ShapeOracle::Impl::touchesFailure(PolyExpr expression) const {
  return llvm::any_of(expression.getVariables(), [&](VariableId variable) {
    return variable < symbols.size() && failures.count(classes[variable]);
  });
}

FailureOr<PolyExpr> ShapeOracle::Impl::normalize(PolyExpr expression) const {
  if (!expression || &expression.getContext() != &context ||
      touchesFailure(expression)) {
    return failure();
  }
  PolyExpr result = context.normalize(expression);
  if (!result) {
    return failure();
  }
  return result;
}

FailureOr<PolyExpr> ShapeOracle::Impl::getNormalizedColumn(ValueDim key) const {
  auto it = columns.find(key);
  if (it == columns.end()) {
    return failure();
  }
  return normalize(it->second);
}

void ShapeOracle::Impl::report(const Failure &failure) const {
  const Constraint &constraint = constraints[failure.constraint];
  if (!failure.difference) {
    constraint.op->emitError("cannot represent shape equality");
  } else if (constraint.assumption && constraint.dim) {
    constraint.op->emitOpError() << "dimension " << *constraint.dim
                                 << " contradicts the inferred extent";
  } else {
    constraint.op->emitOpError() << llvm::formatv(
        "contradictory shape equality: {0} != 0", failure.difference);
  }
}

ShapeOracle::ShapeOracle(Operation *scope)
    : impl(std::make_unique<Impl>(cast<FunctionOpInterface>(scope))) {
  impl->context.freeze();
}

ShapeOracle::~ShapeOracle() = default;

PolyExprContext &ShapeOracle::getContext() const { return impl->context; }

ArrayRef<ShapeOracle::Symbol> ShapeOracle::getSymbols() const {
  return impl->symbols;
}

unsigned ShapeOracle::getNumArgumentSymbols() const {
  return impl->numArgumentSymbols;
}

FailureOr<PolyExpr> ShapeOracle::getExtent(Value shaped, int64_t dim) const {
  return impl->getNormalizedColumn(Impl::ValueDim(shaped, dim));
}

FailureOr<SmallVector<PolyExpr>> ShapeOracle::getShape(Value shaped) const {
  auto type = shaped ? dyn_cast<ShapedType>(shaped.getType()) : ShapedType();
  if (!type || !type.hasRank()) {
    return failure();
  }
  SmallVector<PolyExpr> shape;
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    FailureOr<PolyExpr> extent = getExtent(shaped, dim);
    if (failed(extent)) {
      return failure();
    }
    shape.push_back(*extent);
  }
  return shape;
}

FailureOr<PolyExpr> ShapeOracle::getIndex(Value index) const {
  return impl->getNormalizedColumn(Impl::ValueDim(index, std::nullopt));
}

std::optional<bool> ShapeOracle::areEqual(PolyExpr lhs, PolyExpr rhs) const {
  if (failed(impl->normalize(lhs)) || failed(impl->normalize(rhs))) {
    return std::nullopt;
  }
  return impl->context.areEqual(lhs, rhs);
}

std::optional<bool> ShapeOracle::isOne(PolyExpr extent) const {
  if (failed(impl->normalize(extent))) {
    return std::nullopt;
  }
  return impl->context.isOne(extent);
}

int64_t ShapeOracle::getDivisor(PolyExpr extent) const {
  FailureOr<PolyExpr> normalized = impl->normalize(extent);
  if (failed(normalized)) {
    return 1;
  }
  if (normalized->isZero()) {
    return 0;
  }
  llvm::DynamicAPInt divisor = abs((*normalized)[0].coefficient);
  for (PolyExpr::Term term : *normalized) {
    divisor = gcd(divisor, abs(term.coefficient));
  }
  return divisor <= std::numeric_limits<int64_t>::max()
             ? static_cast<int64_t>(divisor)
             : 1;
}

std::optional<int64_t>
ShapeOracle::getConstantUpperBound(PolyExpr expression) const {
  FailureOr<PolyExpr> normalized = impl->normalize(expression);
  if (failed(normalized)) {
    return std::nullopt;
  }
  llvm::DynamicAPInt bound(0);
  for (PolyExpr::Term term : *normalized) {
    if (term.powers.empty()) {
      bound = term.coefficient;
    } else if (term.coefficient > 0) {
      return std::nullopt;
    }
  }
  if (bound < std::numeric_limits<int64_t>::min() ||
      bound > std::numeric_limits<int64_t>::max()) {
    return std::nullopt;
  }
  return static_cast<int64_t>(bound);
}

LogicalResult ShapeOracle::verify() const {
  // Report in constraint order, which follows the IR.
  SmallVector<const Impl::Failure *> reports;
  for (const Impl::Failure &failure : impl->constantFailures) {
    reports.push_back(&failure);
  }
  for (const auto &entry : impl->failures) {
    reports.push_back(&entry.second);
  }
  llvm::sort(reports, [](const Impl::Failure *lhs, const Impl::Failure *rhs) {
    return lhs->constraint < rhs->constraint;
  });
  for (const Impl::Failure *failure : reports) {
    impl->report(*failure);
  }
  return success(reports.empty());
}

bool ShapeOracle::isInvalidated(
    const AnalysisManager::PreservedAnalyses &preserved) const {
  return !preserved.isPreserved<ShapeOracle>();
}

//===----------------------------------------------------------------------===//
// Materialization
//===----------------------------------------------------------------------===//

FailureOr<OpFoldResult>
ShapeOracle::materialize(OpBuilder &builder, Location loc, PolyExpr extent,
                         MaterializationCache &cache) const {
  FailureOr<PolyExpr> normalized = impl->normalize(extent);
  if (failed(normalized) || !impl->isOverArguments(*normalized)) {
    return failure();
  }
  PolyExpr expression = *normalized;
  for (PolyExpr::Term term : expression) {
    if (term.coefficient < std::numeric_limits<int64_t>::min() ||
        term.coefficient > std::numeric_limits<int64_t>::max()) {
      return failure();
    }
  }
  if (std::optional<int64_t> constant = expression.getConstantInt64()) {
    return OpFoldResult(builder.getIndexAttr(*constant));
  }
  // Emit the polynomial as a sum of scaled monomials.
  Value total;
  for (PolyExpr::Term term : expression) {
    Value product;
    for (Power power : term.powers) {
      Value size = cache.symbols.lookup(power.variable);
      if (!size) {
        auto [value, dim] = impl->symbols[power.variable];
        if (!dim) {
          size = value;
        } else if (isa<MemRefType>(value.getType())) {
          size = memref::DimOp::create(builder, loc, value, *dim);
        } else {
          size = DimOp::create(builder, loc, value,
                               builder.getI64IntegerAttr(*dim));
        }
        cache.symbols[power.variable] = size;
      }
      for (Exponent exponent = 0; exponent < power.exponent; ++exponent) {
        product =
            product
                ? arith::MulIOp::create(builder, loc, product, size).getResult()
                : size;
      }
    }
    int64_t coefficient = static_cast<int64_t>(term.coefficient);
    if (coefficient != 1 || !product) {
      Value constant =
          arith::ConstantIndexOp::create(builder, loc, coefficient);
      product = product ? arith::MulIOp::create(builder, loc, product, constant)
                              .getResult()
                        : constant;
    }
    total =
        total ? arith::AddIOp::create(builder, loc, total, product).getResult()
              : product;
  }
  return OpFoldResult(total);
}
