// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Shape relations of individual operations. Each model reads operand and
/// result extents through a ShapeRelationBuilder and records equalities
/// between them.

#include "ShapeRelationModels.h"

#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;
using nv_tensor_ir::detail::ShapeRelationBuilder;

ShapeRelationBuilder::~ShapeRelationBuilder() = default;

PolyExpr ShapeRelationBuilder::getExpr(OpFoldResult value) {
  if (auto constant = getConstantIntValue(value)) {
    return getConstant(*constant);
  }
  return getExpr(cast<Value>(value));
}

PolyExpr ShapeRelationBuilder::getConstant(int64_t value) {
  return PolyExpr::getConstant(getContext(), value);
}

SmallVector<PolyExpr> ShapeRelationBuilder::getShape(Value value) {
  auto type = cast<ShapedType>(value.getType());
  assert(type.hasRank() && "value must be ranked");
  SmallVector<PolyExpr> shape;
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    shape.push_back(getExpr(value, dim));
  }
  return shape;
}

std::optional<std::pair<Value, int64_t>>
nv_tensor_ir::detail::matchConstantDim(Operation *op) {
  if (auto dim = dyn_cast<DimOp>(op)) {
    return std::make_pair(dim.getInput(), int64_t(dim.getDimension()));
  }
  if (auto dim = dyn_cast<tensor::DimOp>(op)) {
    if (auto index = dim.getConstantIndex()) {
      return std::make_pair(dim.getSource(), *index);
    }
  }
  if (auto dim = dyn_cast<memref::DimOp>(op)) {
    if (auto index = dim.getConstantIndex()) {
      return std::make_pair(dim.getSource(), *index);
    }
  }
  return std::nullopt;
}

/// Equates the extents of two ranked shaped values dimension by dimension.
static void equateShapes(ShapeRelationBuilder &builder, Operation *op,
                         Value lhs, Value rhs) {
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  if (!lhsType || !rhsType || !lhsType.hasRank() || !rhsType.hasRank()) {
    return;
  }
  assert(lhsType.getRank() == rhsType.getRank() &&
         "verified operations equate shapes of equal rank");
  for (int64_t dim = 0; dim < lhsType.getRank(); ++dim) {
    PolyExpr lhsExpr = builder.getExpr(lhs, dim);
    PolyExpr rhsExpr = builder.getExpr(rhs, dim);
    builder.equate(lhsExpr, rhsExpr, op, dim);
  }
}

/// Product of all extents of a ranked shaped value.
static PolyExpr getVolume(ShapeRelationBuilder &builder, Value value) {
  PolyExpr product = builder.getConstant(1);
  for (PolyExpr extent : builder.getShape(value)) {
    product = product * extent;
  }
  return product;
}

/// Equates dimension `dim` of `result` (or `result` itself) with its exact
/// value bound in terms of `dependencies`.
static LogicalResult equateValueBound(ShapeRelationBuilder &builder,
                                      Operation *op, Value result,
                                      std::optional<int64_t> dim,
                                      const ValueDimList &dependencies) {
  AffineMap map;
  ValueDimList mapOperands;
  if (failed(ValueBoundsConstraintSet::computeDependentBound(
          map, mapOperands, presburger::BoundType::EQ,
          ValueBoundsConstraintSet::Variable(result, dim), dependencies))) {
    return failure();
  }
  SmallVector<PolyExpr> operands;
  for (auto [value, operandDim] : mapOperands) {
    operands.push_back(builder.getExpr(value, operandDim));
  }
  builder.equateAffine(op, result, dim, map.getResult(0), operands,
                       map.getNumDims());
  return success();
}

/// Bounds every index result and dynamic result dimension of `op` through
/// its value-bounds model, preferring entry-block arguments over operands.
static void boundResults(ShapeRelationBuilder &builder, Operation *op) {
  ValueDimList operandDependencies;
  for (Value operand : op->getOperands()) {
    if (operand.getType().isIndex()) {
      operandDependencies.emplace_back(operand, std::nullopt);
    } else if (auto type = dyn_cast<ShapedType>(operand.getType());
               type && type.hasRank()) {
      for (int64_t dim = 0; dim < type.getRank(); ++dim) {
        operandDependencies.emplace_back(operand, dim);
      }
    }
  }
  auto bound = [&](Value result, std::optional<int64_t> dim) {
    if (succeeded(equateValueBound(builder, op, result, dim,
                                   builder.getArgumentDependencies()))) {
      return;
    }
    (void)equateValueBound(builder, op, result, dim, operandDependencies);
  };
  for (Value result : op->getResults()) {
    if (builder.hasOutOfRangeDim(result)) {
      continue;
    }
    if (result.getType().isIndex()) {
      bound(result, std::nullopt);
      continue;
    }
    auto type = dyn_cast<ShapedType>(result.getType());
    if (!type || !type.hasRank()) {
      continue;
    }
    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      if (type.isDynamicDim(dim)) {
        bound(result, dim);
      }
    }
  }
}

/// Equates an index arithmetic result with the polynomial of its operands.
static void modelIndexArithmetic(ShapeRelationBuilder &builder, Operation *op) {
  PolyExpr lhs = builder.getExpr(op->getOperand(0));
  PolyExpr rhs = builder.getExpr(op->getOperand(1));
  PolyExpr expression = isa<arith::AddIOp>(op)   ? lhs + rhs
                        : isa<arith::SubIOp>(op) ? lhs - rhs
                                                 : lhs * rhs;
  PolyExpr resultExpr = builder.getExpr(op->getResult(0));
  builder.equate(resultExpr, expression, op);
}

static void modelAffineApply(ShapeRelationBuilder &builder,
                             affine::AffineApplyOp apply) {
  SmallVector<PolyExpr> operands;
  for (Value operand : apply.getMapOperands()) {
    operands.push_back(builder.getExpr(operand));
  }
  builder.equateAffine(apply, apply.getResult(), std::nullopt,
                       apply.getMap().getResult(0), operands,
                       apply.getMap().getNumDims());
}

/// Equates the source and destination, and the destination with the result
/// when the operation has tensor semantics.
static void modelMaterializeInDestination(
    ShapeRelationBuilder &builder,
    bufferization::MaterializeInDestinationOp materialize) {
  equateShapes(builder, materialize, materialize.getSource(),
               materialize.getDest());
  if (Value result = materialize.getResult()) {
    equateShapes(builder, materialize, result, materialize.getDest());
  }
}

/// Equates region results with the values a single-block region yields.
/// Regions with several blocks relate nothing.
static void modelExecuteRegion(ShapeRelationBuilder &builder,
                               scf::ExecuteRegionOp region) {
  if (!region.getRegion().hasOneBlock()) {
    return;
  }
  ValueRange yielded =
      region.getRegion().front().getTerminator()->getOperands();
  for (auto [result, value] : llvm::zip_equal(region.getResults(), yielded)) {
    if (isa<RankedTensorType>(result.getType())) {
      equateShapes(builder, region, result, value);
    } else if (result.getType().isIndex()) {
      PolyExpr resultExpr = builder.getExpr(result);
      PolyExpr valueExpr = builder.getExpr(value);
      builder.equate(resultExpr, valueExpr, region);
    }
  }
}

static void modelAssumeShape(ShapeRelationBuilder &builder,
                             AssumeShapeOp assume) {
  for (auto [dim, size] : llvm::enumerate(assume.getMixedSizes())) {
    PolyExpr dimensionExpr = builder.getExpr(assume.getInput(), dim);
    PolyExpr sizeExpr = builder.getExpr(size);
    builder.equate(dimensionExpr, sizeExpr, assume, dim, /*assumption=*/true);
  }
}

/// Equates dynamic result dimensions with their `dynamic_dims` operands.
static void modelDynamicDims(ShapeRelationBuilder &builder,
                             DynamicDimsOpInterface dynamic) {
  Operation *op = dynamic.getOperation();
  for (OpResult result : op->getResults()) {
    auto type = dyn_cast<RankedTensorType>(result.getType());
    if (!type) {
      continue;
    }
    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      if (!type.isDynamicDim(dim)) {
        continue;
      }
      FailureOr<OpFoldResult> size =
          dynamic.getDimensionSize(result.getResultNumber(), dim);
      if (failed(size)) {
        continue;
      }
      PolyExpr dimensionExpr = builder.getExpr(result, dim);
      PolyExpr sizeExpr = builder.getExpr(*size);
      builder.equate(dimensionExpr, sizeExpr, op, dim);
    }
  }
}

static void modelTranspose(ShapeRelationBuilder &builder,
                           TransposeOp transpose) {
  for (auto [destination, source] :
       llvm::enumerate(transpose.getPermutation())) {
    PolyExpr resultExpr = builder.getExpr(transpose.getOutput(), destination);
    PolyExpr inputExpr = builder.getExpr(transpose.getInput(), source);
    builder.equate(resultExpr, inputExpr, transpose, destination);
  }
}

/// Sums extents along the concatenated dimension and equates the others.
static void modelConcatenate(ShapeRelationBuilder &builder,
                             ConcatenateOp concat) {
  auto type = concat.getOutput().getType();
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    PolyExpr result = builder.getExpr(concat.getOutput(), dim);
    if (static_cast<uint64_t>(dim) == concat.getDimension()) {
      PolyExpr sum = builder.getConstant(0);
      for (Value input : concat.getInputs()) {
        sum = sum + builder.getExpr(input, dim);
      }
      builder.equate(result, sum, concat, dim);
      continue;
    }
    for (Value input : concat.getInputs()) {
      PolyExpr inputExpr = builder.getExpr(input, dim);
      builder.equate(result, inputExpr, concat, dim);
    }
  }
}

/// Equates operand and result dimensions that index the same loop.
static void modelIterationDomain(ShapeRelationBuilder &builder,
                                 IterationDomainOpInterface domain) {
  Operation *op = domain.getOperation();
  SmallVector<AffineMap> maps = domain.getIterationDomainMaps();
  SmallVector<PolyExpr> extents(domain.getLoopIteratorTypes().size());
  ArrayRef<AffineMap> operandMaps =
      ArrayRef(maps).take_front(op->getNumOperands());
  for (auto [value, map] : llvm::zip_equal(op->getOperands(), operandMaps)) {
    for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
      auto axis = dyn_cast<AffineDimExpr>(expr);
      if (!axis) {
        continue;
      }
      PolyExpr next = builder.getExpr(value, dim);
      PolyExpr &extent = extents[axis.getPosition()];
      if (!extent) {
        extent = next;
        continue;
      }
      builder.equate(extent, next, op, dim);
    }
  }
  for (auto [result, map] : llvm::zip_equal(
           op->getResults(), ArrayRef(maps).drop_front(op->getNumOperands()))) {
    for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
      // Results that are not loop dimensions are static unit axes.
      auto axis = dyn_cast<AffineDimExpr>(expr);
      if (!axis) {
        continue;
      }
      PolyExpr extent = extents[axis.getPosition()];
      if (extent) {
        PolyExpr resultExpr = builder.getExpr(result, dim);
        builder.equate(resultExpr, extent, op, dim);
      }
    }
  }
}

static void modelReshape(ShapeRelationBuilder &builder, ReshapeOp reshape) {
  PolyExpr inputVolume = getVolume(builder, reshape.getInput());
  PolyExpr outputVolume = getVolume(builder, reshape.getOutput());
  builder.equate(inputVolume, outputVolume, reshape);
}

/// Equates the dimensions that do not broadcast.
static void modelBroadcast(ShapeRelationBuilder &builder,
                           BroadcastOp broadcast) {
  auto inputType = broadcast.getInput().getType();
  for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
    if (inputType.getDimSize(dim) != 1) {
      PolyExpr resultExpr = builder.getExpr(broadcast.getOutput(), dim);
      PolyExpr inputExpr = builder.getExpr(broadcast.getInput(), dim);
      builder.equate(resultExpr, inputExpr, broadcast, dim);
    }
  }
}

/// Equates all ranked shaped operand and result shapes of operations whose
/// traits require identical shapes. Returns true if a ranked shaped operand
/// determines the result shapes.
static bool modelShapePreservingTraits(ShapeRelationBuilder &builder,
                                       Operation *op) {
  if (!op->hasTrait<OpTrait::Elementwise>() &&
      !op->hasTrait<OpTrait::SameOperandsAndResultShape>() &&
      !op->hasTrait<OpTrait::SameOperandsAndResultType>()) {
    return false;
  }
  Value reference;
  for (Value operand : op->getOperands()) {
    auto type = dyn_cast<ShapedType>(operand.getType());
    if (!type || !type.hasRank()) {
      continue;
    }
    if (!reference) {
      reference = operand;
    } else {
      equateShapes(builder, op, reference, operand);
    }
  }
  if (!reference) {
    return false;
  }
  for (Value result : op->getResults()) {
    equateShapes(builder, op, reference, result);
  }
  return true;
}

void nv_tensor_ir::detail::populateShapeRelations(
    Operation *op, ShapeRelationBuilder &builder) {
  if (auto dynamic = dyn_cast<DynamicDimsOpInterface>(op)) {
    modelDynamicDims(builder, dynamic);
  }
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op) &&
      op->getResult(0).getType().isIndex()) {
    modelIndexArithmetic(builder, op);
    return;
  }
  bool modeled =
      TypeSwitch<Operation *, bool>(op)
          .Case([&](scf::ExecuteRegionOp region) {
            modelExecuteRegion(builder, region);
            return true;
          })
          // A dimension query shares its source extent's column.
          .Case<DimOp, tensor::DimOp, memref::DimOp>([&](Operation *dim) {
            (void)builder.getExpr(dim->getResult(0));
            return true;
          })
          .Case([&](affine::AffineApplyOp apply) {
            modelAffineApply(builder, apply);
            return true;
          })
          .Case([&](bufferization::MaterializeInDestinationOp materialize) {
            modelMaterializeInDestination(builder, materialize);
            return true;
          })
          .Case([&](AssumeShapeOp assume) {
            modelAssumeShape(builder, assume);
            return true;
          })
          .Case([&](TransposeOp transpose) {
            modelTranspose(builder, transpose);
            return true;
          })
          .Case([&](ConcatenateOp concat) {
            modelConcatenate(builder, concat);
            return true;
          })
          .Case([&](IterationDomainOpInterface domain) {
            modelIterationDomain(builder, domain);
            return true;
          })
          .Case([&](ReshapeOp reshape) {
            modelReshape(builder, reshape);
            return true;
          })
          .Case([&](BroadcastOp broadcast) {
            modelBroadcast(builder, broadcast);
            return true;
          })
          .Default(false);
  if (modeled) {
    return;
  }
  // Value-bounds models of shape-preserving operations, such as arith, may
  // bound only index values.
  if (modelShapePreservingTraits(builder, op)) {
    return;
  }
  if (isa<ValueBoundsOpInterface>(op)) {
    boundResults(builder, op);
  }
}
