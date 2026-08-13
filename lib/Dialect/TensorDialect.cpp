// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/TypeSwitch.h"

// ODS-generated definitions.
#include "tensor_ir/Dialect/TensorDialect.cpp.inc"
#include "tensor_ir/Dialect/TensorOpInterfaces.cpp.inc"

using namespace mlir;
using namespace mlir::nv_tensor_ir;
namespace tcg = mlir::nv_tensor_ir::tcutegen;

namespace mlir::nv_tensor_ir {

SmallVector<int64_t> detail::getFlatTensorShape(const tcutegen::Shape &shape) {
  SmallVector<int64_t> result;
  result.reserve(tcutegen::rank(shape));
  for (const auto &dimension : shape) {
    result.push_back(tcutegen::is_static(dimension) ? dimension.as_int()
                                                    : ShapedType::kDynamic);
  }
  return result;
}

tcutegen::Shape getShapeRef(TensorType tensorType) {
  tcutegen::Shape shape;
  for (int64_t dim : tensorType.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      shape.appendDynamic();
    } else {
      shape.append(dim);
    }
  }
  return shape;
}

} // namespace mlir::nv_tensor_ir

static LogicalResult verifyTensorSignatureAttribute(Operation *op, Type type,
                                                    NamedAttribute attr,
                                                    bool isInput) {
  StringRef attrName = attr.getName().strref();
  StringRef displayName = attrName.rsplit('.').second;
  auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(type);

  auto requireTensorType = [&]() -> LogicalResult {
    if (tensorType) {
      return success();
    }
    return op->emitError()
           << "expects " << displayName
           << " attribute to be applied to tensor type, but got " << type;
  };

  if (attrName == TensorIRDialect::getAlignmentAttrName()) {
    if (failed(requireTensorType())) {
      return failure();
    }
    auto alignment = dyn_cast<IntegerAttr>(attr.getValue());
    if (!alignment) {
      return op->emitOpError() << "expects alignment is `IntegerAttr`, but got "
                               << attr.getValue();
    }
    if (alignment.getInt() < 1) {
      return op->emitError()
             << "expects alignment >= 1, but got " << alignment.getInt();
    }
    return success();
  }

  if (attrName == TensorIRDialect::getStrideAttrName()) {
    if (failed(requireTensorType())) {
      return failure();
    }
    auto strideAttr = dyn_cast<StringAttr>(attr.getValue());
    if (!strideAttr) {
      return op->emitOpError()
             << "expects stride is `StringAttr`, but got " << attr.getValue();
    }

    auto stride = tcg::from_string<tcg::Stride>(strideAttr.str());
    if (!stride) {
      return op->emitError()
             << "expects stride to be tcutegen::Stride, but got " << strideAttr;
    }

    tcg::Shape shape = getShapeRef(tensorType);
    tcg::Layout layout(shape, *stride);
    if (tcg::has_error(layout)) {
      return op->emitError()
             << "expects stride is valid for corresponding shape, but got \""
             << tcg::to_string(*stride) << "\"";
    }

    // Reject stride=0 on input dimensions with shape != 1. Outputs are
    // unrestricted because their storage may intentionally represent a
    // broadcasted destination.
    if (isInput) {
      for (size_t i = 0; i < tcg::rank(*stride); ++i) {
        const auto &strideDim = (*stride)[i];
        if (!tcg::is_static(strideDim) || strideDim != 0) {
          continue;
        }
        const auto &shapeDim = shape[i];
        if (!tcg::is_static(shapeDim) || shapeDim != 1) {
          return op->emitError()
                 << "stride=0 is not allowed in graph input stride attribute "
                    "on a non-shape-1 dim (legacy broadcast convention is "
                    "deprecated; use shape=1 + an explicit broadcast op "
                    "instead), but got \""
                 << tcg::to_string(*stride) << "\" with shape \""
                 << tcg::to_string(shape) << "\"";
        }
      }
    }
    return success();
  }

  if (attrName == TensorIRDialect::getCostAttrName()) {
    if (failed(requireTensorType())) {
      return failure();
    }
    auto cost = dyn_cast<IntegerAttr>(attr.getValue());
    if (!cost) {
      return op->emitOpError()
             << "expects cost is `IntegerAttr`, but got " << attr.getValue();
    }
    int64_t costValue = cost.getInt();
    if (costValue != -1 && (costValue < 1 || costValue > 10)) {
      return op->emitError()
             << "expects cost to be -1 or between 1-10, but got " << costValue;
    }
    return success();
  }

  if (attrName == TensorIRDialect::getIterSpaceMapAttrName()) {
    if (failed(requireTensorType())) {
      return failure();
    }
    if (!isa<AffineMapAttr>(attr.getValue())) {
      return op->emitOpError()
             << "expects iter_space_map is `AffineMapAttr`, but got "
             << attr.getValue();
    }
    return success();
  }

  if (attrName == TensorIRDialect::getAllowTMAAttrName()) {
    if (failed(requireTensorType())) {
      return failure();
    }
    if (!isa<BoolAttr>(attr.getValue())) {
      return op->emitOpError()
             << "expects allow_tma is `BoolAttr`, but got " << attr.getValue();
    }
    return success();
  }

  if (attrName == TensorIRDialect::getIterSpaceIdAttrName()) {
    if (failed(requireTensorType())) {
      return failure();
    }
    if (!isa<IntegerAttr>(attr.getValue())) {
      return op->emitOpError()
             << "expects iter_space_id is `IntegerAttr`, but got "
             << attr.getValue();
    }
    return success();
  }

  if (attrName == TensorIRDialect::getIterSpaceIdsAttrName()) {
    if (failed(requireTensorType())) {
      return failure();
    }
    if (!isa<DenseI32ArrayAttr>(attr.getValue())) {
      return op->emitOpError()
             << "expects iter_space_ids is `DenseI32ArrayAttr`, but got "
             << attr.getValue();
    }
    return success();
  }

  if (attrName == TensorIRDialect::getIterSpaceDimDomainsAttrName()) {
    if (failed(requireTensorType())) {
      return failure();
    }
    if (!isa<IterSpaceDimDomainsAttr>(attr.getValue())) {
      return op->emitOpError() << "expects iter_space_dim_domains is "
                                  "`IterSpaceDimDomainsAttr`, but got "
                               << attr.getValue();
    }
    return success();
  }

  return op->emitOpError() << "does not support TensorIR signature attribute `"
                           << attrName << "`";
}

void TensorIRDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tensor_ir/Dialect/TensorAttrs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "tensor_ir/Dialect/TensorOps.cpp.inc"
      >();
}

LogicalResult
TensorIRDialect::verifyRegionArgAttribute(Operation *op, unsigned regionIndex,
                                          unsigned argIndex,
                                          NamedAttribute argAttr) {
  auto function = cast<FunctionOpInterface>(op);
  assert(regionIndex == 0 && "function arguments must belong to region zero");
  return verifyTensorSignatureAttribute(
      op, function.getArgumentTypes()[argIndex], argAttr, /*isInput=*/true);
}

LogicalResult TensorIRDialect::verifyRegionResultAttribute(
    Operation *op, unsigned regionIndex, unsigned resultIndex,
    NamedAttribute resultAttr) {
  auto function = cast<FunctionOpInterface>(op);
  assert(regionIndex == 0 && "function results must belong to region zero");
  return verifyTensorSignatureAttribute(op,
                                        function.getResultTypes()[resultIndex],
                                        resultAttr, /*isInput=*/false);
}

#define GET_ATTRDEF_CLASSES
#include "tensor_ir/Dialect/TensorAttrs.cpp.inc"
