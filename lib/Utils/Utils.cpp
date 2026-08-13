// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Utils/Utils.h"

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <climits>
#include <cstdint>

using namespace llvm;
using namespace mlir;
namespace tcg = mlir::nv_tensor_ir::tcutegen;

namespace mlir {
namespace nv_tensor_ir {
namespace {

FailureOr<SmallVector<TensorDescriptor::StaticOrDynamic>>
getStaticStrideInfo(StringRef strideText, size_t expectedRank) {
  std::optional<tcg::Stride> stride = tcg::Stride::fromString(strideText);
  if (!stride || tcg::rank(*stride) != expectedRank) {
    return failure();
  }

  SmallVector<TensorDescriptor::StaticOrDynamic> result;
  result.reserve(expectedRank);
  for (size_t i = 0; i < expectedRank; ++i) {
    const auto &dim = (*stride)[i];
    if (tcg::depth(dim) != 0) {
      return failure();
    }

    if (tcg::is_static(dim)) {
      result.push_back({tcg::static_size(dim), nullptr});
      continue;
    }

    int64_t divisibility = tcg::get_divisibility(dim);
    if (!llvm::isPowerOf2_64(divisibility)) {
      return failure();
    }
    result.push_back({ShapedType::kDynamic, nullptr, divisibility});
  }
  return result;
}

} // namespace

FailureOr<int64_t> getAlignmentFromGraph(Value tensor) {
  Location loc = tensor.getLoc();
  if (!isTensorType(tensor.getType())) {
    return emitError(loc) << "Expect value of TensorType as an input";
  }

  auto graphOp = dyn_cast<GraphOp>(tensor.getParentRegion()->getParentOp());
  if (!graphOp) {
    return emitError(loc) << "Expect GraphOp as a parent op of the tensor";
  }

  auto getAlignmentFromAttr = [&](size_t idx, ArrayAttr arrayAttr) -> size_t {
    if (arrayAttr) {
      if (auto argAttr = dyn_cast<DictionaryAttr>(arrayAttr[idx])) {
        if (auto alignAttr = argAttr.getAs<IntegerAttr>(
                TensorIRDialect::getAlignmentAttrName())) {
          return alignAttr.getInt();
        }
      }
    }
    // If alignment is not set, return default alignment
    return getDefaultAlignment(
        cast<nv_tensor_ir::TensorType>(tensor.getType()));
  };

  // Check if the tensor is input
  for (size_t i = 0; i < graphOp.getNumArguments(); ++i) {
    Value operand = graphOp.getArgument(i);
    if (operand == tensor) {
      return getAlignmentFromAttr(i, graphOp.getAllArgAttrs());
    }
  }

  // Check if the tensor is output
  for (size_t i = 0; i < graphOp.getNumResults(); ++i) {
    Value operand = graphOp.getResults()[i];
    if (operand == tensor) {
      return getAlignmentFromAttr(i, graphOp.getAllResultAttrs());
    }
  }

  return emitError(loc)
         << "Expect tensor value is either Graph input or output";
}

FailureOr<tcg::Stride> getStrideFromGraph(Value tensor) {
  Location loc = tensor.getLoc();
  auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(tensor.getType());
  if (!tensorType || !isTensorType(tensor.getType())) {
    return emitError(loc) << "Expect value of TensorType as an input";
  }

  auto graphOp = dyn_cast<GraphOp>(tensor.getParentRegion()->getParentOp());
  if (!graphOp) {
    return emitError(loc) << "Expect GraphOp as a parent op of the tensor";
  }

  auto getStridesFromAttr = [&](size_t idx,
                                ArrayAttr arrayAttr) -> FailureOr<tcg::Stride> {
    if (arrayAttr && idx < arrayAttr.size()) {
      if (auto argAttr = dyn_cast<DictionaryAttr>(arrayAttr[idx])) {
        if (auto strideAttr = argAttr.getAs<StringAttr>(
                TensorIRDialect::getStrideAttrName())) {
          std::optional<tcg::Stride> stride =
              tcg::Stride::fromString(strideAttr.strref());
          if (!stride) {
            return emitError(loc)
                   << "Invalid stride attribute: " << strideAttr.strref();
          }
          return *stride;
        }
      }
    }
    // If stride is not set, default to row-major (TensorIR's default layout).
    // MaterializeDefaultStridesPass materializes explicit row-major strides
    // on static-shape graph boundaries up front, so this fallback handles
    // the dynamic-shape case where the pass intentionally leaves strides
    // implicit to avoid changing the kernel-launch ABI.
    tcg::Shape shape = getShapeRef(tensorType);
    return tcg::Layout(shape).stride();
  };

  // Check if the tensor is input
  for (size_t i = 0; i < graphOp.getNumArguments(); ++i) {
    Value operand = graphOp.getArgument(i);
    if (operand == tensor) {
      return getStridesFromAttr(i, graphOp.getAllArgAttrs());
    }
  }

  // Check if the tensor is output
  for (size_t i = 0; i < graphOp.getNumResults(); ++i) {
    Value operand = graphOp.getResults()[i];
    if (operand == tensor) {
      return getStridesFromAttr(i, graphOp.getAllResultAttrs());
    }
  }

  return emitError(loc)
         << "Expect tensor value is either Graph input or output";
}

/// Returns true if the graph has no operations in the region other than the
/// `results` op. The graph may have inputs and outputs.
bool isEmptyRegionGraphOp(nv_tensor_ir::GraphOp graphOp) {
  Region &region = graphOp.getGraphBody();
  Block &block = region.front();
  return &block.front() == &block.back();
}

bool hasDynamicInputOrOutputTensor(nv_tensor_ir::GraphOp graphOp) {
  for (Type type : graphOp.getArgumentTypes()) {
    if (auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(type);
        tensorType && isTensorType(type)) {
      if (!tensorType.hasStaticShape()) {
        return true;
      }
    }
  }

  for (Type type : graphOp.getResultTypes()) {
    if (auto tensorType = dyn_cast<nv_tensor_ir::TensorType>(type);
        tensorType && isTensorType(type)) {
      if (!tensorType.hasStaticShape()) {
        return true;
      }
    }
  }

  return false;
}

/// Get CuTe layout for a graph input or output.
static std::optional<tcg::Layout> getLayoutForGraphInputOrOutput(Value value) {
  if (auto tensorTy = dyn_cast<TensorType>(value.getType());
      tensorTy && isTensorType(value.getType())) {
    tcg::Shape shape = getShapeRef(tensorTy);
    auto stride = getStrideFromGraph(value);
    if (succeeded(stride)) {
      return tcg::Layout(shape, *stride);
    }
  }
  return std::nullopt;
}

/// Get layout attribute for an operation or block argument.
Attribute getLayoutSourceAttr(Value value) {
  // If the value is a function argument, convert it to a tensor source.
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto layout = getLayoutForGraphInputOrOutput(value);
        layout.has_value()) {
      return TensorSourceAttr::get(value.getContext(), blockArg.getArgNumber(),
                                   /*offset=*/0, layout->toString(),
                                   getDynamicValueMapping(*layout));
    }
    return nullptr;
  }

  // Get layout attribute from the operation.
  return value.getDefiningOp()->getAttr(TensorIRDialect::getLayoutAttrName());
}

/// Get block argument index offsets for a dynamic layout.
SmallVector<int32_t> getDynamicValueMapping(const tcg::Layout &layout) {
  size_t dynamicCount = 0;
  for (size_t i = 0, n = tcg::rank(layout); i < n; ++i) {
    if (!tcg::get(layout.shape(), i).isStatic()) {
      ++dynamicCount;
    }
    if (!tcg::get(layout.stride(), i).isStatic()) {
      ++dynamicCount;
    }
  }
  SmallVector<int32_t> result(dynamicCount);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

/// Returns a vector of `TensorDescriptor`s for each tensor in the graph
/// signature (inputs or outputs).
FailureOr<SmallVector<TensorDescriptor>>
getTensorDescriptors(ArrayRef<Type> types, ArrayAttr attributes) {
  SmallVector<TensorDescriptor> result(types.size());

  for (size_t i = 0; i < types.size(); ++i) {
    TensorDescriptor &descriptor = result[i];
    if (auto tensorType = dyn_cast<TensorType>(types[i])) {
      // Set the sizes and the alignment from the tensor type.
      for (int64_t size : tensorType.getShape()) {
        descriptor.sizes.push_back(
            TensorDescriptor::StaticOrDynamic{size, nullptr});
      }
      descriptor.alignment = getDefaultAlignment(tensorType);

      // Get the attribute corresponding to the tensor (by index).
      DictionaryAttr attr = nullptr;
      if (attributes && attributes.size() > i) {
        attr = dyn_cast<DictionaryAttr>(attributes[i]);
      }
      if (!attr) {
        continue; // Dictionary attribute is not set, do not query for keys.
      }

      // Get the strides from the dictionary attribute.
      if (auto strideAttr =
              attr.getAs<StringAttr>(TensorIRDialect::getStrideAttrName())) {
        auto strides =
            getStaticStrideInfo(strideAttr.strref(), descriptor.sizes.size());
        if (failed(strides)) {
          return failure();
        }
        descriptor.strides = std::move(*strides);
      }

      // Get the alignment from the dictionary attribute.
      if (auto alignAttr = attr.getAs<IntegerAttr>(
              TensorIRDialect::getAlignmentAttrName())) {
        descriptor.alignment = alignAttr.getInt();
      }

      // Get the cost from the dictionary attribute.
      if (auto costAttr =
              attr.getAs<IntegerAttr>(TensorIRDialect::getCostAttrName())) {
        descriptor.cost = costAttr.getInt();
      }

      // Get the TMA flag from the dictionary attribute.
      if (auto tmaAttr =
              attr.getAs<BoolAttr>(TensorIRDialect::getAllowTMAAttrName())) {
        descriptor.allowTma = tmaAttr.getValue();
      }
    }
  }

  return result;
}

FailureOr<llvm::hash_code> getModuleHash(ModuleOp moduleOp) {
  OwningOpRef<Operation *> clonedModuleOp(moduleOp->clone());

  // Strip location information (ported from `StripDebugInfo::runOnOperation`)
  auto unknownLoc = UnknownLoc::get(clonedModuleOp->getContext());
  clonedModuleOp->walk([&](Operation *op) {
    op->setLoc(unknownLoc);
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (BlockArgument &arg : block.getArguments()) {
          arg.setLoc(unknownLoc);
        }
      }
    }
  });

  // Anonymize symbols to ignore symbol name discrepancy among modules
  size_t symNameCounter = 0;
  auto walkResult =
      clonedModuleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op->hasAttr(SymbolTable::getSymbolAttrName())) {
          StringAttr oldName = SymbolTable::getSymbolName(op);

          // Generate new anonymized name and replace all uses of the symbol
          std::string newNameStr = "__anno_" + std::to_string(symNameCounter++);
          StringAttr newName =
              StringAttr::get(clonedModuleOp->getContext(), newNameStr);
          if (failed(SymbolTable::replaceAllSymbolUses(oldName, newName,
                                                       *clonedModuleOp))) {
            op->emitError(
                "Failed to update symbol references during anonymization");
            return WalkResult::interrupt();
          }

          // Update the symbol name
          SymbolTable::setSymbolName(op, newName);
        }
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  // SSA variable names are not stored in the bytecode and thus nothing special
  // needed here.

  mlir::BytecodeWriterConfig config;
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);

  MLIR_RETURN_IF_ERROR(
      mlir::writeBytecodeToFile(*clonedModuleOp, ostream, config));
  return llvm::hash_value(buffer);
}

int64_t getDefaultAlignment(nv_tensor_ir::TensorType tensorType) {
  return std::max(1u, tensorType.getElementType().getIntOrFloatBitWidth() /
                          CHAR_BIT);
}

FailureOr<std::array<int32_t, 3>>
mapGridSizeTo3D(llvm::ArrayRef<DimSize> gridSize, Location loc) {
  // The CudaTile-backed launch path currently flattens work onto grid.x. Use
  // y/z coordinates once the runtime launch metadata supports them.
  int64_t size = 1;
  for (DimSize dimSize : gridSize) {
    size *= dimSize;
  }
  if (size < 1) {
    return mlir::emitError(loc)
           << "computed grid size must contain at least one tile (grid: "
           << vectorToString(gridSize) << ")";
  }
  if (size > INT32_MAX) {
    return mlir::emitError(loc)
           << "grid size " << size
           << " exceeds the representable 32-bit limit (grid: "
           << vectorToString(gridSize) << ")";
  }
  return std::array<int32_t, 3>{static_cast<int32_t>(size), 1, 1};
}

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

/// Convert a TensorIR integer comparator to Arith dialect integer predicate.
FailureOr<arith::CmpIPredicate>
convertToArithIPredicate(mlir::nv_tensor_ir::Comparator comparator) {
  switch (comparator) {
  // integer comparators
  case mlir::nv_tensor_ir::Comparator::eq:
    return mlir::arith::CmpIPredicate::eq;
  case mlir::nv_tensor_ir::Comparator::neq:
    return mlir::arith::CmpIPredicate::ne;
  case mlir::nv_tensor_ir::Comparator::gt:
    return mlir::arith::CmpIPredicate::sgt;
  case mlir::nv_tensor_ir::Comparator::ge:
    return mlir::arith::CmpIPredicate::sge;
  case mlir::nv_tensor_ir::Comparator::lt:
    return mlir::arith::CmpIPredicate::slt;
  case mlir::nv_tensor_ir::Comparator::le:
    return mlir::arith::CmpIPredicate::sle;
  default:
    return failure();
  }
}

/// Convert a TensorIR floating-point comparator to Arith dialect floating-point
/// predicate.
FailureOr<arith::CmpFPredicate>
convertToArithFPredicate(mlir::nv_tensor_ir::Comparator comparator) {
  switch (comparator) {
  // float comparators
  case mlir::nv_tensor_ir::Comparator::oeq:
    return mlir::arith::CmpFPredicate::OEQ;
  case mlir::nv_tensor_ir::Comparator::one:
    return mlir::arith::CmpFPredicate::ONE;
  case mlir::nv_tensor_ir::Comparator::ogt:
    return mlir::arith::CmpFPredicate::OGT;
  case mlir::nv_tensor_ir::Comparator::oge:
    return mlir::arith::CmpFPredicate::OGE;
  case mlir::nv_tensor_ir::Comparator::olt:
    return mlir::arith::CmpFPredicate::OLT;
  case mlir::nv_tensor_ir::Comparator::ole:
    return mlir::arith::CmpFPredicate::OLE;
  case mlir::nv_tensor_ir::Comparator::ueq:
    return mlir::arith::CmpFPredicate::UEQ;
  case mlir::nv_tensor_ir::Comparator::une:
    return mlir::arith::CmpFPredicate::UNE;
  case mlir::nv_tensor_ir::Comparator::ugt:
    return mlir::arith::CmpFPredicate::UGT;
  case mlir::nv_tensor_ir::Comparator::uge:
    return mlir::arith::CmpFPredicate::UGE;
  case mlir::nv_tensor_ir::Comparator::ult:
    return mlir::arith::CmpFPredicate::ULT;
  case mlir::nv_tensor_ir::Comparator::ule:
    return mlir::arith::CmpFPredicate::ULE;
  default:
    return failure();
  }
}

Type toSignless(Type ty) {
  if (ty && isa<IntegerType>(ty) && !ty.isSignlessInteger()) {
    auto ctx = ty.getContext();
    return IntegerType::get(ctx, ty.getIntOrFloatBitWidth(),
                            IntegerType::SignednessSemantics::Signless);
  }
  return ty;
}

} // namespace nv_tensor_ir
} // namespace mlir
