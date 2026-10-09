// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Support/TCutegen.h"

#include "mlir/IR/Diagnostics.h"
// Needed when OSS filtering removes cutegen's ShapedType include.
#include "mlir/IR/BuiltinTypes.h" // IWYU pragma: keep

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <utility>

namespace mlir::nv_tensor_ir::tcutegen {
namespace {

constexpr int32_t kDefaultDynamicWidth = 32;

using detail::DimValue;
using detail::RecursiveDimValue;
using detail::RecursiveDimValues;

std::optional<int64_t> parseInteger(llvm::StringRef str) {
  int64_t value = 0;
  if (!llvm::to_integer(str.trim(), value, 10)) {
    return std::nullopt;
  }
  return value;
}

bool isSupportedDynamicWidth(int64_t width) {
  switch (width) {
  case 1:
  case 8:
  case 16:
  case 32:
  case 64:
    return true;
  default:
    return false;
  }
}

std::optional<int64_t> consumeLeadingInteger(llvm::StringRef &str) {
  str = str.trim();
  size_t length = 0;
  while (length < str.size() &&
         std::isdigit(static_cast<unsigned char>(str[length]))) {
    ++length;
  }
  if (length == 0) {
    return std::nullopt;
  }

  llvm::StringRef valueStr = str.take_front(length);
  llvm::StringRef rest = str.drop_front(length);
  if (!rest.empty() &&
      !std::isspace(static_cast<unsigned char>(rest.front()))) {
    return std::nullopt;
  }

  str = rest;
  return parseInteger(valueStr);
}

RecursiveDimValue makeRecursiveValue(RecursiveDimValues values) {
  return RecursiveDimValue(
      std::vector<RecursiveDimValue>(std::make_move_iterator(values.begin()),
                                     std::make_move_iterator(values.end())));
}

std::string valueToString(const RecursiveDimValue &value);

std::string valuesToString(ArrayRef<RecursiveDimValue> values) {
  std::string result = "(";
  for (const RecursiveDimValue &value : values) {
    if (result.size() > 1) {
      result += ",";
    }
    result += valueToString(value);
  }
  result += ")";
  return result;
}

std::string valueToString(const RecursiveDimValue &value) {
  if (value.isLeaf()) {
    return value.getLeaf().toString();
  }
  return valuesToString(value.getValues());
}

class RecursiveValueParser {
public:
  RecursiveValueParser(llvm::StringRef str) : str(str) {}

  std::optional<RecursiveDimValue> parseTopLevelValue() {
    std::optional<RecursiveDimValue> value = parseValue();
    if (!value) {
      return std::nullopt;
    }
    skipWhitespace();
    if (pos != str.size()) {
      return std::nullopt;
    }
    return value;
  }

private:
  void skipWhitespace() {
    while (pos < str.size() &&
           std::isspace(static_cast<unsigned char>(str[pos]))) {
      ++pos;
    }
  }

  bool consume(char c) {
    skipWhitespace();
    if (pos >= str.size() || str[pos] != c) {
      return false;
    }
    ++pos;
    return true;
  }

  std::optional<RecursiveDimValues> parseTuple() {
    if (!consume('(')) {
      return std::nullopt;
    }

    RecursiveDimValues values;
    skipWhitespace();
    if (consume(')')) {
      return values;
    }

    while (true) {
      std::optional<RecursiveDimValue> value = parseValue();
      if (!value) {
        return std::nullopt;
      }
      values.push_back(std::move(*value));

      skipWhitespace();
      if (consume(')')) {
        return values;
      }
      if (!consume(',')) {
        return std::nullopt;
      }
    }
  }

  std::optional<RecursiveDimValue> parseValue() {
    skipWhitespace();
    if (pos >= str.size()) {
      return std::nullopt;
    }
    if (str[pos] == '(') {
      std::optional<RecursiveDimValues> values = parseTuple();
      if (!values) {
        return std::nullopt;
      }
      return makeRecursiveValue(std::move(*values));
    }

    size_t start = pos;
    while (pos < str.size() && str[pos] != ',' && str[pos] != ')') {
      ++pos;
    }
    if (start == pos) {
      return std::nullopt;
    }

    std::optional<DimValue> value = DimValue::fromString(str.slice(start, pos));
    if (!value) {
      return std::nullopt;
    }
    return RecursiveDimValue(*value);
  }

  llvm::StringRef str;
  size_t pos = 0;
};

std::optional<RecursiveDimValue> parseValue(llvm::StringRef str) {
  return RecursiveValueParser(str).parseTopLevelValue();
}

std::optional<std::pair<llvm::StringRef, llvm::StringRef>>
splitLayoutString(llvm::StringRef str) {
  size_t parenDepth = 0;
  size_t braceDepth = 0;
  std::optional<size_t> colonPos;
  for (size_t i = 0, e = str.size(); i < e; ++i) {
    switch (str[i]) {
    case '(':
      if (braceDepth == 0) {
        ++parenDepth;
      }
      break;
    case ')':
      if (braceDepth == 0) {
        if (parenDepth == 0) {
          return std::nullopt;
        }
        --parenDepth;
      }
      break;
    case '{':
      ++braceDepth;
      break;
    case '}':
      if (braceDepth == 0) {
        return std::nullopt;
      }
      --braceDepth;
      break;
    case ':':
      if (parenDepth == 0 && braceDepth == 0) {
        if (colonPos) {
          return std::nullopt;
        }
        colonPos = i;
      }
      break;
    default:
      break;
    }
  }
  if (!colonPos || parenDepth != 0 || braceDepth != 0) {
    return std::nullopt;
  }
  return std::make_pair(str.take_front(*colonPos),
                        str.drop_front(*colonPos + 1));
}

bool areCongruent(ArrayRef<RecursiveDimValue> lhs,
                  ArrayRef<RecursiveDimValue> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return std::equal(
      lhs.begin(), lhs.end(), rhs.begin(),
      [](const RecursiveDimValue &lhsValue, const RecursiveDimValue &rhsValue) {
        if (lhsValue.isLeaf() && rhsValue.isLeaf()) {
          return true;
        }
        if (lhsValue.isLeaf() != rhsValue.isLeaf()) {
          return false;
        }
        return areCongruent(lhsValue.getValues(), rhsValue.getValues());
      });
}

size_t getDepth(const RecursiveDimValue &value) {
  if (value.isLeaf()) {
    return 0;
  }
  size_t subDepth = 0;
  for (const RecursiveDimValue &nested : value.getValues()) {
    subDepth = std::max(subDepth, getDepth(nested));
  }
  return 1 + subDepth;
}

size_t getDepth(ArrayRef<RecursiveDimValue> values) {
  size_t subDepth = 0;
  for (const RecursiveDimValue &value : values) {
    subDepth = std::max(subDepth, getDepth(value));
  }
  return 1 + subDepth;
}

DimValue multiply(DimValue lhs, DimValue rhs) {
  if (detail::has_error(lhs) || detail::has_error(rhs)) {
    return DimValue(CgErrorT{});
  }
  if (!lhs.isStatic() || !rhs.isStatic()) {
    return DimValue::getDynamic(1);
  }
  std::optional<int64_t> result = llvm::checkedMul(lhs.as_int(), rhs.as_int());
  return result ? DimValue::getStatic(*result) : DimValue(CgErrorT{});
}

DimValue add(DimValue lhs, DimValue rhs) {
  if (detail::has_error(lhs) || detail::has_error(rhs)) {
    return DimValue(CgErrorT{});
  }
  if (!lhs.isStatic() || !rhs.isStatic()) {
    return DimValue::getDynamic(1);
  }
  std::optional<int64_t> result = llvm::checkedAdd(lhs.as_int(), rhs.as_int());
  return result ? DimValue::getStatic(*result) : DimValue(CgErrorT{});
}

DimValue product(const RecursiveDimValue &value) {
  if (detail::has_error(value)) {
    return DimValue(CgErrorT{});
  }
  if (value.isLeaf()) {
    return value.getLeaf();
  }
  DimValue result = DimValue::getStatic(1);
  for (const RecursiveDimValue &nested : value.getValues()) {
    result = multiply(result, product(nested));
  }
  return result;
}

void flattenRecursiveValueInto(const RecursiveDimValue &value,
                               RecursiveDimValues &flat) {
  if (value.isLeaf()) {
    flat.push_back(value);
    return;
  }

  for (const RecursiveDimValue &nested : value.getValues()) {
    flattenRecursiveValueInto(nested, flat);
  }
}

RecursiveDimValues flattenRecursiveValues(ArrayRef<RecursiveDimValue> values) {
  RecursiveDimValues flat;
  for (const RecursiveDimValue &value : values) {
    flattenRecursiveValueInto(value, flat);
  }
  return flat;
}

std::optional<bool> allStatic(const RecursiveDimValue &lhsShape,
                              const RecursiveDimValue &rhsShape,
                              const RecursiveDimValue &lhsStride,
                              const RecursiveDimValue &rhsStride) {
  if (!lhsShape.isLeaf() || !rhsShape.isLeaf() || !lhsStride.isLeaf() ||
      !rhsStride.isLeaf()) {
    assert(false && "coalesce expects flattened values");
    return std::nullopt;
  }
  return lhsShape.isStatic() && rhsShape.isStatic() && lhsStride.isStatic() &&
         rhsStride.isStatic();
}

// Coalesces flattened modes using logical last-dimension-fastest order.
// The left mode can be folded into the right mode exactly when advancing the
// left coordinate skips one complete right mode.
bool coalesceFlat(RecursiveDimValues &shape, RecursiveDimValues &stride) {
  if (shape.empty() || shape.size() != stride.size()) {
    return false;
  }

  int64_t index = static_cast<int64_t>(shape.size()) - 1;
  while (index >= 0) {
    if (index == 0) {
      return true;
    }

    if (shape[index - 1] == 1) {
      shape.erase(shape.begin() + index - 1);
      stride.erase(stride.begin() + index - 1);
      --index;
      continue;
    }

    const RecursiveDimValue &leftShape = shape[index - 1];
    const RecursiveDimValue &rightShape = shape[index];
    const RecursiveDimValue &leftStride = stride[index - 1];
    const RecursiveDimValue &rightStride = stride[index];
    std::optional<bool> areAllStatic =
        allStatic(leftShape, rightShape, leftStride, rightStride);
    if (!areAllStatic.has_value()) {
      return false;
    }
    if (*areAllStatic &&
        leftStride.getLeaf() ==
            multiply(rightShape.getLeaf(), rightStride.getLeaf())) {
      DimValue newShape = multiply(leftShape.getLeaf(), rightShape.getLeaf());
      if (detail::has_error(newShape)) {
        return false;
      }
      RecursiveDimValue newStride = rightStride;
      shape.erase(shape.begin() + index);
      stride.erase(stride.begin() + index);
      shape[index - 1] = RecursiveDimValue(newShape);
      stride[index - 1] = std::move(newStride);
      --index;
      continue;
    }

    --index;
  }
  return true;
}

RecursiveDimValue getDefaultStride(const RecursiveDimValue &shape,
                                   DimValue current) {
  if (detail::has_error(shape) || detail::has_error(current)) {
    return RecursiveDimValue(CgErrorT{});
  }
  if (shape.isLeaf()) {
    return RecursiveDimValue(current);
  }

  RecursiveDimValues values;
  values.reserve(shape.getValues().size());
  DimValue runningStride = current;
  for (const RecursiveDimValue &nested : llvm::reverse(shape.getValues())) {
    values.push_back(getDefaultStride(nested, runningStride));
    runningStride = multiply(runningStride, product(nested));
  }
  std::reverse(values.begin(), values.end());
  return makeRecursiveValue(std::move(values));
}

bool hasValidShapeDimensions(ArrayRef<RecursiveDimValue> values);

bool hasValidShapeDimensions(const RecursiveDimValue &value) {
  if (detail::has_error(value)) {
    return false;
  }
  if (value.isLeaf()) {
    const DimValue &dim = value.getLeaf();
    return !dim.isStatic() || dim.as_int() >= 1;
  }
  return hasValidShapeDimensions(value.getValues());
}

bool hasValidShapeDimensions(ArrayRef<RecursiveDimValue> values) {
  return llvm::all_of(values, [](const RecursiveDimValue &value) {
    return hasValidShapeDimensions(value);
  });
}

std::optional<RecursiveDimValues> takeValues(ArrayRef<RecursiveDimValue> values,
                                             int begin, int end) {
  if (begin < 0) {
    return std::nullopt;
  }

  int rank = static_cast<int>(values.size());
  end = end < 0 ? rank : std::min(end, rank);
  if (begin >= end) {
    return RecursiveDimValues{};
  }

  assert(end <= rank && "take end index must not exceed rank");
  RecursiveDimValues result;
  result.reserve(end - begin);
  for (int i = begin; i < end; ++i) {
    result.push_back(values[i]);
  }
  return result;
}

DimValue computeLayoutOffset(const RecursiveDimValue &coord,
                             const RecursiveDimValue &stride) {
  if (detail::has_error(coord) || detail::has_error(stride)) {
    return DimValue(CgErrorT{});
  }
  if (coord.isLeaf() && stride.isLeaf()) {
    return multiply(coord.getLeaf(), stride.getLeaf());
  }
  if (coord.isLeaf() != stride.isLeaf() ||
      coord.getValues().size() != stride.getValues().size()) {
    return DimValue(CgErrorT{});
  }

  DimValue result = DimValue::getStatic(0);
  for (auto [coordValue, strideValue] :
       llvm::zip_equal(coord.getValues(), stride.getValues())) {
    result = add(result, computeLayoutOffset(coordValue, strideValue));
  }
  return result;
}

DimValue computeLayoutOffset(ArrayRef<RecursiveDimValue> coord,
                             ArrayRef<RecursiveDimValue> stride) {
  if (coord.size() != stride.size()) {
    return DimValue(CgErrorT{});
  }

  DimValue result = DimValue::getStatic(0);
  for (auto [coordValue, strideValue] : llvm::zip_equal(coord, stride)) {
    result = add(result, computeLayoutOffset(coordValue, strideValue));
  }
  return result;
}

RecursiveDimValue idx2crdValue(int64_t index, const RecursiveDimValue &shape,
                               int64_t &stride) {
  if (detail::has_error(shape) || !shape.isStatic() || index < 0) {
    return RecursiveDimValue(CgErrorT{});
  }

  if (shape.isLeaf()) {
    int64_t shapeSize = shape.as_int();
    if (shapeSize <= 0 || stride <= 0) {
      return RecursiveDimValue(CgErrorT{});
    }
    int64_t coord = (index / stride) % shapeSize;
    std::optional<int64_t> nextStride = llvm::checkedMul(stride, shapeSize);
    if (!nextStride) {
      return RecursiveDimValue(CgErrorT{});
    }
    stride = *nextStride;
    return RecursiveDimValue(DimValue::getStatic(coord));
  }

  RecursiveDimValues coordValues;
  coordValues.reserve(shape.getValues().size());
  for (const RecursiveDimValue &nested : llvm::reverse(shape.getValues())) {
    coordValues.push_back(idx2crdValue(index, nested, stride));
    if (detail::has_error(coordValues.back())) {
      return RecursiveDimValue(CgErrorT{});
    }
  }
  std::reverse(coordValues.begin(), coordValues.end());
  return makeRecursiveValue(std::move(coordValues));
}

} // namespace

namespace detail {

bool has_error(const DimValue &dim) { return dim.has_error_; }

bool has_error(const RecursiveDimValue &dim) { return dim.has_error_; }

bool is_congruent(const RecursiveDimValue &lhs, const RecursiveDimValue &rhs) {
  if (has_error(lhs) || has_error(rhs)) {
    return false;
  }
  return areCongruent(ArrayRef<RecursiveDimValue>(&lhs, 1),
                      ArrayRef<RecursiveDimValue>(&rhs, 1));
}

RecursiveDimValue::RecursiveDimValue(DimValue value)
    : leaf(true), value(value), has_error_(has_error(this->value)) {}

RecursiveDimValue::RecursiveDimValue(std::vector<RecursiveDimValue> values)
    : leaf(false), value(DimValue::getStatic(0)), values(std::move(values)) {
  has_error_ = llvm::any_of(this->values, [](const RecursiveDimValue &nested) {
    return has_error(nested);
  });
}

RecursiveDimValue::RecursiveDimValue(CgErrorT)
    : leaf(true), value(DimValue(CgErrorT{})), has_error_(true) {}

const DimValue &RecursiveDimValue::getLeaf() const {
  assert(isLeaf() && "expected a scalar dimension value");
  return value;
}

llvm::ArrayRef<RecursiveDimValue> RecursiveDimValue::getValues() const {
  assert(!isLeaf() && "expected a recursive dimension value");
  return values;
}

std::string RecursiveDimValue::toString() const {
  return has_error(*this) ? "error_t" : valueToString(*this);
}

bool RecursiveDimValue::isStatic() const {
  if (has_error(*this)) {
    return false;
  }
  return isLeaf() ? value.isStatic()
                  : llvm::all_of(values, [](const RecursiveDimValue &nested) {
                      return nested.isStatic();
                    });
}

std::int64_t RecursiveDimValue::as_int() const {
  if (has_error(*this)) {
    llvm::report_fatal_error("cannot convert error dimension to integer");
  }
  if (!isLeaf()) {
    llvm::report_fatal_error("cannot convert recursive dimension to integer");
  }
  return value.as_int();
}

int64_t RecursiveDimValue::getDivisibility() const {
  if (has_error(*this)) {
    llvm::report_fatal_error("cannot query error dimension divisibility");
  }
  if (!isLeaf()) {
    llvm::report_fatal_error("cannot query recursive dimension divisibility");
  }
  return value.getDivisibility();
}

bool operator==(const RecursiveDimValue &lhs, const RecursiveDimValue &rhs) {
  if (has_error(lhs) || has_error(rhs)) {
    return has_error(lhs) == has_error(rhs);
  }
  if (lhs.isLeaf() != rhs.isLeaf()) {
    return false;
  }
  if (lhs.isLeaf()) {
    return lhs.getLeaf() == rhs.getLeaf();
  }
  return lhs.values == rhs.values;
}

bool operator!=(const RecursiveDimValue &lhs, const RecursiveDimValue &rhs) {
  return !(lhs == rhs);
}

bool operator==(const RecursiveDimValue &lhs, const DimValue &rhs) {
  if (has_error(lhs) || has_error(rhs)) {
    return false;
  }
  return lhs.isLeaf() && lhs.getLeaf() == rhs;
}

bool operator!=(const RecursiveDimValue &lhs, const DimValue &rhs) {
  return !(lhs == rhs);
}

bool operator==(const RecursiveDimValue &lhs, int64_t rhs) {
  if (has_error(lhs)) {
    return false;
  }
  return lhs.isLeaf() && lhs.getLeaf() == rhs;
}

bool operator!=(const RecursiveDimValue &lhs, int64_t rhs) {
  return !(lhs == rhs);
}

bool operator==(int64_t lhs, const RecursiveDimValue &rhs) {
  return rhs == lhs;
}

bool operator!=(int64_t lhs, const RecursiveDimValue &rhs) {
  return !(lhs == rhs);
}

DimValue DimValue::getStatic(int64_t value) {
  return DimValue(value, false, value, kDefaultDynamicWidth);
}

DimValue DimValue::getDynamic(int64_t divisibility, int32_t width) {
  return DimValue(0, true, divisibility, width);
}

DimValue::DimValue(CgErrorT)
    : value(0), isDynamic(false), divisibility(1), width(kDefaultDynamicWidth),
      has_error_(true) {}

std::string DimValue::toString() const {
  if (has_error(*this)) {
    return "error_t";
  }
  if (isDynamic) {
    if (divisibility == 1 && width == kDefaultDynamicWidth) {
      return "?";
    }

    std::string result = "?{";
    if (width != kDefaultDynamicWidth) {
      result += "i" + std::to_string(width);
      if (divisibility != 1) {
        result += " ";
      }
    }
    if (divisibility != 1) {
      result += "div=" + std::to_string(divisibility);
    }
    result += "}";
    return result;
  }
  return std::to_string(value);
}

std::optional<DimValue> DimValue::fromString(mlir::StringRef str) {
  llvm::StringRef s = str.trim();
  if (s == "?") {
    return DimValue::getDynamic(1);
  }

  if (s.consume_front("?")) {
    s = s.trim();
    if (!s.consume_front("{") || !s.consume_back("}")) {
      return std::nullopt;
    }
    if (s.trim().empty()) {
      return std::nullopt;
    }
    int64_t divisibility = 1;
    int32_t width = kDefaultDynamicWidth;
    bool hasDivisibility = false;
    bool hasWidth = false;
    while (!(s = s.trim()).empty()) {
      if (s.consume_front("div")) {
        if (hasDivisibility) {
          return std::nullopt;
        }
        s = s.trim();
        if (!s.consume_front("=")) {
          return std::nullopt;
        }
        std::optional<int64_t> parsedDivisibility = consumeLeadingInteger(s);
        if (!parsedDivisibility || *parsedDivisibility <= 0) {
          return std::nullopt;
        }
        divisibility = *parsedDivisibility;
        hasDivisibility = true;
        continue;
      }
      if (s.consume_front("i")) {
        if (hasWidth) {
          return std::nullopt;
        }
        std::optional<int64_t> parsedWidth = consumeLeadingInteger(s);
        if (!parsedWidth || !isSupportedDynamicWidth(*parsedWidth)) {
          return std::nullopt;
        }
        width = static_cast<int32_t>(*parsedWidth);
        hasWidth = true;
        continue;
      }
      return std::nullopt;
    }
    return DimValue::getDynamic(divisibility, width);
  }

  std::optional<int64_t> staticValue = parseInteger(s);
  if (!staticValue || *staticValue < 0) {
    return std::nullopt;
  }
  return DimValue::getStatic(*staticValue);
}

DimValue::DimValue(int64_t value, bool isDynamic, int64_t divisibility,
                   int32_t width)
    : value(value), isDynamic(isDynamic), divisibility(divisibility),
      width(width) {
  if (value < 0) {
    this->value = 0;
    has_error_ = true;
  }
  if (isDynamic && divisibility <= 0) {
    this->divisibility = 1;
    has_error_ = true;
  }
  if (!isSupportedDynamicWidth(width)) {
    this->width = kDefaultDynamicWidth;
    has_error_ = true;
  }
}

bool DimValue::isStatic() const { return !has_error(*this) && !isDynamic; }

std::int64_t DimValue::as_int() const {
  if (has_error(*this)) {
    llvm::report_fatal_error("cannot convert error dimension to integer");
  }
  if (isDynamic) {
    llvm::report_fatal_error("cannot convert dynamic dimension to integer");
  }
  return value;
}

int64_t DimValue::getDivisibility() const {
  if (has_error(*this)) {
    llvm::report_fatal_error("cannot query error dimension divisibility");
  }
  return divisibility;
}

bool operator==(const DimValue &lhs, const DimValue &rhs) {
  if (has_error(lhs) || has_error(rhs)) {
    return has_error(lhs) == has_error(rhs);
  }
  if (lhs.isDynamic != rhs.isDynamic) {
    return false;
  }
  if (lhs.isDynamic) {
    return lhs.divisibility == rhs.divisibility && lhs.width == rhs.width;
  }
  return lhs.value == rhs.value;
}

bool operator!=(const DimValue &lhs, const DimValue &rhs) {
  return !(lhs == rhs);
}

bool operator==(const DimValue &lhs, int64_t rhs) {
  if (has_error(lhs)) {
    return false;
  }
  return lhs.isStatic() && lhs.value == rhs;
}

bool operator!=(const DimValue &lhs, int64_t rhs) { return !(lhs == rhs); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DimValue &dim) {
  return os << dim.toString();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const RecursiveDimValue &dim) {
  return os << dim.toString();
}

mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const DimValue &dim) {
  return diag << dim.toString();
}

mlir::Diagnostic &operator<<(mlir::Diagnostic &diag,
                             const RecursiveDimValue &dim) {
  return diag << dim.toString();
}

} // namespace detail

template <SequenceKind K>
DimSequence<K>::DimSequence() : has_error_(K == SequenceKind::Coord) {}

template <SequenceKind K>
DimSequence<K>::DimSequence(detail::RecursiveDimValues values)
    : values(std::move(values)) {
  has_error_ = !validDimensions();
}

template <SequenceKind K>
DimSequence<K>::DimSequence(detail::RecursiveDimValue value)
    : scalarRoot(value.isLeaf()) {
  if (scalarRoot) {
    values.push_back(std::move(value));
  } else {
    values.assign(value.getValues().begin(), value.getValues().end());
  }
  has_error_ = !validDimensions();
}

template <SequenceKind K>
DimSequence<K>::DimSequence(llvm::ArrayRef<int64_t> dims) {
  for (int64_t dim : dims) {
    values.emplace_back(dim == mlir::ShapedType::kDynamic
                            ? DimValue::getDynamic(1)
                            : DimValue::getStatic(dim));
  }
  has_error_ = !validDimensions();
}

template <SequenceKind K>
DimSequence<K>::DimSequence(int64_t value)
    : DimSequence(RecursiveDimValue(DimValue::getStatic(value))) {}

template <SequenceKind K>
DimSequence<K>::DimSequence(CgErrorT)
    : values({RecursiveDimValue(CgErrorT{})}), scalarRoot(true),
      has_error_(true) {}

template <SequenceKind K>
bool DimSequence<K>::validDimensions() const {
  if constexpr (K == SequenceKind::Shape) {
    return hasValidShapeDimensions(values);
  }
  return llvm::none_of(values,
                       [](const auto &v) { return detail::has_error(v); });
}

template <SequenceKind K>
bool DimSequence<K>::hasError() const {
  return has_error_ || llvm::any_of(values, [](const auto &v) {
           return detail::has_error(v);
         });
}

template <SequenceKind K>
RecursiveDimValue DimSequence<K>::asRecursiveValue() && {
  return scalarRoot ? std::move(values.front())
                    : makeRecursiveValue(std::move(values));
}

template <SequenceKind K>
std::string DimSequence<K>::toString() const {
  return hasError()   ? "error_t"
         : scalarRoot ? values.front().toString()
                      : valuesToString(values);
}

template <SequenceKind K>
std::optional<DimSequence<K>> DimSequence<K>::fromString(mlir::StringRef str) {
  auto value = parseValue(str);
  if (!value) {
    return std::nullopt;
  }
  DimSequence result(std::move(*value));
  return result.hasError() ? std::nullopt : std::optional<DimSequence>(result);
}

template <SequenceKind K>
const RecursiveDimValue &
DimSequence<K>::requireScalar(const char *action) const {
  if (hasError() || !scalarRoot) {
    const char *name = K == SequenceKind::Shape    ? "shape"
                       : K == SequenceKind::Stride ? "stride"
                                                   : "coordinate";
    llvm::report_fatal_error(
        llvm::Twine("cannot ") + action +
        (hasError() ? " error " : " non-scalar ") + name +
        (std::string(action) == "convert" ? " to integer" : " divisibility"));
  }
  return values.front();
}

template <SequenceKind K>
int64_t DimSequence<K>::as_int() const {
  return requireScalar("convert").as_int();
}

template <SequenceKind K>
int64_t DimSequence<K>::getDivisibility() const {
  return requireScalar("query").getDivisibility();
}

template <SequenceKind K>
void DimSequence<K>::append(int64_t value) {
  append(DimSequence(value));
}

template <SequenceKind K>
void DimSequence<K>::appendDynamic(int64_t divisibility, int32_t width) {
  append(DimSequence(
      RecursiveDimValue(DimValue::getDynamic(divisibility, width))));
}

template <SequenceKind K>
void DimSequence<K>::append(DimSequence value) {
  scalarRoot = false;
  bool valueHasError = value.hasError();
  values.push_back(std::move(value).asRecursiveValue());
  has_error_ = valueHasError || !validDimensions();
}

template <SequenceKind K>
void DimSequence<K>::set(size_t index, DimSequence value) {
  if (index >= values.size()) {
    has_error_ = true;
  } else if (scalarRoot) {
    *this = std::move(value);
  } else {
    bool valueHasError = value.hasError();
    values[index] = std::move(value).asRecursiveValue();
    has_error_ = valueHasError || !validDimensions();
  }
}

template <SequenceKind K>
bool DimSequence<K>::isStatic() const {
  return !hasError() &&
         llvm::all_of(values, [](const auto &v) { return v.isStatic(); });
}

template <SequenceKind K>
size_t DimSequence<K>::getDepth() const {
  return hasError()   ? 0
         : scalarRoot ? tcutegen::getDepth(values.front())
                      : tcutegen::getDepth(ArrayRef<RecursiveDimValue>(values));
}

template <SequenceKind K>
int64_t DimSequence<K>::staticSize() const {
  return product(makeRecursiveValue(values)).as_int();
}

template <SequenceKind K>
DimSequence<K> DimSequence<K>::flattened() const {
  return hasError() || scalarRoot ? *this
                                  : DimSequence(flattenRecursiveValues(values));
}

template <SequenceKind K>
DimSequence<K> DimSequence<K>::reversed() const {
  return hasError() || scalarRoot ? *this
                                  : DimSequence(detail::RecursiveDimValues(
                                        values.rbegin(), values.rend()));
}

template <SequenceKind K>
DimSequence<K> DimSequence<K>::taken(int begin, int end) const {
  auto selected = takeValues(values, begin, end);
  return hasError() || !selected ? DimSequence(CgErrorT{})
                                 : DimSequence(std::move(*selected));
}

template <SequenceKind K>
Stride DimSequence<K>::getDefault(const Shape &shape) {
  if (shape.hasError()) {
    return Stride(CgErrorT{});
  }
  RecursiveDimValue root =
      shape.isScalar()
          ? shape.get(0)
          : makeRecursiveValue(detail::RecursiveDimValues(shape.getValues()));
  return Stride(getDefaultStride(root, DimValue::getStatic(1)));
}

template class DimSequence<SequenceKind::Shape>;
template class DimSequence<SequenceKind::Stride>;
template class DimSequence<SequenceKind::Coord>;

Layout::Layout() = default;

Layout::Layout(Shape shape) : Layout(shape, Stride::getDefault(shape)) {}

Layout::Layout(Shape shape, Stride stride)
    : shapeValue(std::move(shape)), strideValue(std::move(stride)) {
  has_error_ = !is_congruent(this->shapeValue, this->strideValue);
}

Layout::Layout(int64_t shape, int64_t stride)
    : Layout(Shape(shape), Stride(stride)) {}

Layout::Layout(CgErrorT)
    : shapeValue(Shape(CgErrorT{})), strideValue(Stride(CgErrorT{})),
      has_error_(true) {}

std::string Layout::toString() const {
  if (::mlir::nv_tensor_ir::tcutegen::has_error(*this)) {
    return "error_t:error_t";
  }
  return shapeValue.toString() + ":" + strideValue.toString();
}

detail::DimValue Layout::operator()(const Coord &coord) const {
  if (::mlir::nv_tensor_ir::tcutegen::has_error(*this) ||
      ::mlir::nv_tensor_ir::tcutegen::has_error(coord)) {
    return detail::DimValue(CgErrorT{});
  }
  if (shapeValue.isScalar() != coord.isScalar() ||
      !areCongruent(coord.getValues(), shapeValue.getValues())) {
    return detail::DimValue(CgErrorT{});
  }
  if (coord.isScalar()) {
    return computeLayoutOffset(coord.getValues().front(),
                               strideValue.getValues().front());
  }
  return computeLayoutOffset(coord.getValues(), strideValue.getValues());
}

Layout Layout::get(size_t index) const {
  if (::mlir::nv_tensor_ir::tcutegen::has_error(*this) ||
      index >= rank(*this)) {
    return Layout(CgErrorT{});
  }
  return Layout(Shape(shape()[index]), Stride(stride()[index]));
}

void Layout::append(Layout other) {
  shapeValue.append(std::move(other.shapeValue));
  strideValue.append(std::move(other.strideValue));
  has_error_ = !is_congruent(shapeValue, strideValue);
}

Layout::const_iterator Layout::begin() const {
  return const_iterator(*this, 0);
}

Layout::const_iterator Layout::end() const {
  return const_iterator(*this, rank(shape()));
}

bool operator==(const Layout &lhs, const Layout &rhs) {
  if (has_error(lhs) || has_error(rhs)) {
    return has_error(lhs) == has_error(rhs);
  }
  return lhs.shape() == rhs.shape() && lhs.stride() == rhs.stride();
}

bool operator!=(const Layout &lhs, const Layout &rhs) { return !(lhs == rhs); }

std::optional<Layout> Layout::fromString(mlir::StringRef str) {
  std::optional<std::pair<llvm::StringRef, llvm::StringRef>> pieces =
      splitLayoutString(str);
  if (!pieces) {
    return std::nullopt;
  }
  auto [shapeStr, strideStr] = *pieces;
  std::optional<Shape> shape = Shape::fromString(shapeStr);
  std::optional<Stride> stride = Stride::fromString(strideStr);
  if (!shape || !stride || !is_congruent(*shape, *stride)) {
    return std::nullopt;
  }
  return Layout(std::move(*shape), std::move(*stride));
}

size_t rank(const Layout &layout) { return rank(layout.shape()); }

size_t rank(const detail::RecursiveDimValue &value) { return value.rank(); }

bool has_error(const Layout &layout) {
  return layout.has_error_ || has_error(layout.shape()) ||
         has_error(layout.stride());
}

size_t depth(const Layout &layout) { return depth(layout.shape()); }

size_t depth(const detail::RecursiveDimValue &value) { return getDepth(value); }

bool is_static(const Layout &layout) {
  if (has_error(layout)) {
    return false;
  }
  return is_static(layout.shape()) && is_static(layout.stride());
}

bool is_static(const DimValue &dim) { return dim.isStatic(); }

bool is_static(const detail::RecursiveDimValue &value) {
  return value.isStatic();
}

int64_t get_divisibility(const DimValue &dim) { return dim.getDivisibility(); }

int64_t get_divisibility(const detail::RecursiveDimValue &value) {
  return value.getDivisibility();
}

int64_t static_size(const detail::RecursiveDimValue &value) {
  return product(value).as_int();
}

int64_t static_size(const Layout &layout) {
  return static_size(layout.shape());
}

int64_t static_size(const Layout &layout, size_t index) {
  assert(index < rank(layout));
  return static_size(layout.shape(), index);
}

Layout flatten(const Layout &layout) {
  if (has_error(layout)) {
    return Layout(CgErrorT{});
  }
  return Layout(flatten(layout.shape()), flatten(layout.stride()));
}

Layout coalesce(const Layout &layout) {
  if (has_error(layout)) {
    return Layout(CgErrorT{});
  }
  Layout flatLayout = flatten(layout);
  detail::RecursiveDimValues flatShape(flatLayout.shape().getValues());
  detail::RecursiveDimValues flatStride(flatLayout.stride().getValues());
  if (!coalesceFlat(flatShape, flatStride)) {
    return Layout(CgErrorT{});
  }

  if (llvm::all_of(flatShape, [](const detail::RecursiveDimValue &value) {
        return value == 1;
      })) {
    return Layout(
        Shape(detail::RecursiveDimValue(detail::DimValue::getStatic(1))),
        Stride(detail::RecursiveDimValue(detail::DimValue::getStatic(0))));
  }

  if (flatShape.back() == 1) {
    flatShape.pop_back();
    flatStride.pop_back();
  }

  assert(!flatShape.empty() && !flatStride.empty() &&
         "coalesced layout must not be empty");
  Shape coalescedShape = flatShape.size() == 1
                             ? Shape(std::move(flatShape.front()))
                             : Shape(std::move(flatShape));
  Stride coalescedStride = flatStride.size() == 1
                               ? Stride(std::move(flatStride.front()))
                               : Stride(std::move(flatStride));
  return Layout(std::move(coalescedShape), std::move(coalescedStride));
}

Layout make_layout(std::vector<Layout> lys) {
  if (lys.empty() || llvm::any_of(lys, [](const Layout &layout) {
        return has_error(layout);
      })) {
    return Layout(CgErrorT{});
  }

  Layout result;
  for (Layout &layout : lys) {
    result.append(std::move(layout));
  }
  return result;
}

Layout reverse(const Layout &layout) {
  if (has_error(layout)) {
    return Layout(CgErrorT{});
  }
  return Layout(reverse(layout.shape()), reverse(layout.stride()));
}

Layout take(int begin, int end, const Layout &layout) {
  if (has_error(layout)) {
    return Layout(CgErrorT{});
  }
  return Layout(take(begin, end, layout.shape()),
                take(begin, end, layout.stride()));
}

Coord idx2crd(int64_t index, const Shape &shape) {
  if (has_error(shape) || index < 0 || shape.getValues().empty()) {
    return Coord(CgErrorT{});
  }
  if (shape.isScalar()) {
    return Coord(index);
  }

  int64_t stride = 1;
  detail::RecursiveDimValues coordValues;
  coordValues.reserve(shape.getValues().size());
  for (const detail::RecursiveDimValue &dim :
       llvm::reverse(shape.getValues())) {
    coordValues.push_back(idx2crdValue(index, dim, stride));
    if (detail::has_error(coordValues.back())) {
      return Coord(CgErrorT{});
    }
  }
  std::reverse(coordValues.begin(), coordValues.end());
  return Coord(std::move(coordValues));
}

} // namespace mlir::nv_tensor_ir::tcutegen
