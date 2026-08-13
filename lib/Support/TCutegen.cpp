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

RecursiveDimValues makeRecursiveValues(ArrayRef<DimValue> values) {
  RecursiveDimValues recursiveValues;
  recursiveValues.reserve(values.size());
  for (DimValue value : values) {
    recursiveValues.emplace_back(value);
  }
  return recursiveValues;
}

RecursiveDimValue makeRecursiveValue(RecursiveDimValues values) {
  std::vector<RecursiveDimValue> nestedValues;
  nestedValues.reserve(values.size());
  for (RecursiveDimValue &value : values) {
    nestedValues.push_back(std::move(value));
  }
  return RecursiveDimValue(std::move(nestedValues));
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

bool hasValidCoordValues(const RecursiveDimValue &value) {
  if (detail::has_error(value)) {
    return false;
  }
  if (value.isLeaf()) {
    const DimValue &coord = value.getLeaf();
    return !coord.isStatic() || coord.as_int() >= 0;
  }
  return llvm::all_of(value.getValues(), [](const RecursiveDimValue &nested) {
    return hasValidCoordValues(nested);
  });
}

bool hasValidCoordValues(ArrayRef<RecursiveDimValue> values) {
  return llvm::all_of(values, [](const RecursiveDimValue &value) {
    return hasValidCoordValues(value);
  });
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

bool RecursiveDimValue::isLeaf() const { return leaf; }

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

size_t RecursiveDimValue::rank() const { return isLeaf() ? 1 : values.size(); }

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

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Shape::Shape(SmallVector<detail::DimValue> values)
    : Shape(makeRecursiveValues(values)) {}

Shape::Shape() = default;

Shape::Shape(detail::RecursiveDimValues values) : values(std::move(values)) {
  has_error_ = !hasValidShapeDimensions(this->values);
}

Shape::Shape(detail::RecursiveDimValue value) : scalarRoot(value.isLeaf()) {
  has_error_ = !hasValidShapeDimensions(value);
  if (scalarRoot) {
    values.push_back(std::move(value));
    return;
  }

  llvm::ArrayRef<detail::RecursiveDimValue> nestedValues = value.getValues();
  values.assign(nestedValues.begin(), nestedValues.end());
}

Shape::Shape(llvm::ArrayRef<int64_t> values)
    : Shape([&]() {
        SmallVector<detail::DimValue> dimValues;
        dimValues.reserve(values.size());
        for (int64_t value : values) {
          dimValues.push_back(value == mlir::ShapedType::kDynamic
                                  ? detail::DimValue::getDynamic(1)
                                  : detail::DimValue::getStatic(value));
        }
        return makeRecursiveValues(dimValues);
      }()) {}

Shape::Shape(int64_t value)
    : Shape(detail::RecursiveDimValue(detail::DimValue::getStatic(value))) {}

Shape::Shape(CgErrorT)
    : values({detail::RecursiveDimValue(CgErrorT{})}), scalarRoot(true),
      has_error_(true) {}

std::string Shape::toString() const {
  if (has_error(*this)) {
    return "error_t";
  }
  return scalarRoot ? values.front().toString() : valuesToString(values);
}

size_t Shape::rank() const { return values.size(); }

std::int64_t Shape::as_int() const {
  if (has_error(*this)) {
    llvm::report_fatal_error("cannot convert error shape to integer");
  }
  if (!scalarRoot) {
    llvm::report_fatal_error("cannot convert non-scalar shape to integer");
  }
  return values.front().as_int();
}

void Shape::append(int64_t value) {
  scalarRoot = false;
  values.emplace_back(detail::DimValue::getStatic(value));
  has_error_ = !hasValidShapeDimensions(values);
}

void Shape::append(Shape value) {
  bool valueHasError = has_error(value);
  scalarRoot = false;
  values.push_back(value.scalarRoot
                       ? std::move(value.values.front())
                       : makeRecursiveValue(std::move(value.values)));
  has_error_ = valueHasError || !hasValidShapeDimensions(values);
}

void Shape::appendDynamic(int64_t divisibility, int32_t width) {
  scalarRoot = false;
  values.emplace_back(detail::DimValue::getDynamic(divisibility, width));
  has_error_ = !hasValidShapeDimensions(values);
}

int64_t Shape::getDivisibility() const {
  if (has_error(*this)) {
    llvm::report_fatal_error("cannot query error shape divisibility");
  }
  if (!scalarRoot) {
    llvm::report_fatal_error("cannot query non-scalar shape divisibility");
  }
  return values.front().getDivisibility();
}

const detail::RecursiveDimValue &Shape::get(size_t index) const {
  return values[index];
}

const detail::RecursiveDimValue &Shape::operator[](size_t index) const {
  return values[index];
}

void Shape::set(size_t index, Shape value) {
  if (scalarRoot) {
    if (index != 0) {
      has_error_ = true;
      return;
    }
    *this = std::move(value);
    return;
  }

  if (index >= values.size()) {
    has_error_ = true;
    return;
  }
  has_error_ |= has_error(value);
  values[index] = value.scalarRoot
                      ? std::move(value.values.front())
                      : makeRecursiveValue(std::move(value.values));
}

Shape::iterator Shape::begin() { return iterator(*this, 0); }

Shape::iterator Shape::end() { return iterator(*this, rank()); }

Shape::const_iterator Shape::begin() const { return const_iterator(*this, 0); }

Shape::const_iterator Shape::end() const {
  return const_iterator(*this, rank());
}

Shape::const_iterator Shape::cbegin() const { return begin(); }

Shape::const_iterator Shape::cend() const { return end(); }

bool operator==(const Shape &lhs, const Shape &rhs) {
  if (has_error(lhs) || has_error(rhs)) {
    return has_error(lhs) == has_error(rhs);
  }
  return lhs.scalarRoot == rhs.scalarRoot && lhs.values == rhs.values;
}

bool operator!=(const Shape &lhs, const Shape &rhs) { return !(lhs == rhs); }

bool operator==(const Shape &lhs, int64_t rhs) {
  if (has_error(lhs)) {
    return false;
  }
  return lhs.scalarRoot && lhs.values.front() == rhs;
}

bool operator!=(const Shape &lhs, int64_t rhs) { return !(lhs == rhs); }

bool operator==(int64_t lhs, const Shape &rhs) { return rhs == lhs; }

bool operator!=(int64_t lhs, const Shape &rhs) { return !(lhs == rhs); }

std::optional<Shape> Shape::fromString(mlir::StringRef str) {
  std::optional<detail::RecursiveDimValue> value = parseValue(str);
  if (!value || !hasValidShapeDimensions(*value)) {
    return std::nullopt;
  }
  return Shape(std::move(*value));
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Stride::Stride(SmallVector<detail::DimValue> values)
    : Stride(makeRecursiveValues(values)) {}

Stride::Stride() = default;

Stride::Stride(detail::RecursiveDimValues values) : values(std::move(values)) {
  has_error_ =
      llvm::any_of(this->values, [](const detail::RecursiveDimValue &v) {
        return detail::has_error(v);
      });
}

Stride::Stride(detail::RecursiveDimValue value) : scalarRoot(value.isLeaf()) {
  has_error_ = detail::has_error(value);
  if (scalarRoot) {
    values.push_back(std::move(value));
    return;
  }

  llvm::ArrayRef<detail::RecursiveDimValue> nestedValues = value.getValues();
  values.assign(nestedValues.begin(), nestedValues.end());
}

Stride::Stride(int64_t value)
    : Stride(detail::RecursiveDimValue(detail::DimValue::getStatic(value))) {}

Stride::Stride(CgErrorT)
    : values({detail::RecursiveDimValue(CgErrorT{})}), scalarRoot(true),
      has_error_(true) {}

std::string Stride::toString() const {
  if (has_error(*this)) {
    return "error_t";
  }
  return scalarRoot ? values.front().toString() : valuesToString(values);
}

size_t Stride::rank() const { return values.size(); }

std::int64_t Stride::as_int() const {
  if (has_error(*this)) {
    llvm::report_fatal_error("cannot convert error stride to integer");
  }
  if (!scalarRoot) {
    llvm::report_fatal_error("cannot convert non-scalar stride to integer");
  }
  return values.front().as_int();
}

void Stride::append(int64_t value) {
  scalarRoot = false;
  values.emplace_back(detail::DimValue::getStatic(value));
  has_error_ = llvm::any_of(values, [](const detail::RecursiveDimValue &value) {
    return detail::has_error(value);
  });
}

void Stride::append(Stride value) {
  bool valueHasError = has_error(value);
  scalarRoot = false;
  values.push_back(value.scalarRoot
                       ? std::move(value.values.front())
                       : makeRecursiveValue(std::move(value.values)));
  has_error_ = valueHasError ||
               llvm::any_of(values, [](const detail::RecursiveDimValue &value) {
                 return detail::has_error(value);
               });
}

void Stride::appendDynamic(int64_t divisibility, int32_t width) {
  scalarRoot = false;
  values.emplace_back(detail::DimValue::getDynamic(divisibility, width));
  has_error_ = llvm::any_of(values, [](const detail::RecursiveDimValue &value) {
    return detail::has_error(value);
  });
}

int64_t Stride::getDivisibility() const {
  if (has_error(*this)) {
    llvm::report_fatal_error("cannot query error stride divisibility");
  }
  if (!scalarRoot) {
    llvm::report_fatal_error("cannot query non-scalar stride divisibility");
  }
  return values.front().getDivisibility();
}

const detail::RecursiveDimValue &Stride::get(size_t index) const {
  return values[index];
}

const detail::RecursiveDimValue &Stride::operator[](size_t index) const {
  return values[index];
}

void Stride::set(size_t index, Stride value) {
  if (scalarRoot) {
    if (index != 0) {
      has_error_ = true;
      return;
    }
    *this = std::move(value);
    return;
  }

  if (index >= values.size()) {
    has_error_ = true;
    return;
  }
  has_error_ |= has_error(value);
  values[index] = value.scalarRoot
                      ? std::move(value.values.front())
                      : makeRecursiveValue(std::move(value.values));
}

Stride::iterator Stride::begin() { return iterator(*this, 0); }

Stride::iterator Stride::end() { return iterator(*this, rank()); }

Stride::const_iterator Stride::begin() const {
  return const_iterator(*this, 0);
}

Stride::const_iterator Stride::end() const {
  return const_iterator(*this, rank());
}

Stride::const_iterator Stride::cbegin() const { return begin(); }

Stride::const_iterator Stride::cend() const { return end(); }

bool operator==(const Stride &lhs, const Stride &rhs) {
  if (has_error(lhs) || has_error(rhs)) {
    return has_error(lhs) == has_error(rhs);
  }
  return lhs.scalarRoot == rhs.scalarRoot && lhs.values == rhs.values;
}

bool operator!=(const Stride &lhs, const Stride &rhs) { return !(lhs == rhs); }

bool operator==(const Stride &lhs, int64_t rhs) {
  if (has_error(lhs)) {
    return false;
  }
  return lhs.scalarRoot && lhs.values.front() == rhs;
}

bool operator!=(const Stride &lhs, int64_t rhs) { return !(lhs == rhs); }

bool operator==(int64_t lhs, const Stride &rhs) { return rhs == lhs; }

bool operator!=(int64_t lhs, const Stride &rhs) { return !(lhs == rhs); }

std::optional<Stride> Stride::fromString(mlir::StringRef str) {
  std::optional<detail::RecursiveDimValue> value = parseValue(str);
  if (!value) {
    return std::nullopt;
  }
  return Stride(std::move(*value));
}

Stride Stride::getDefault(const Shape &shape) {
  if (has_error(shape)) {
    return Stride(CgErrorT{});
  }
  if (shape.scalarRoot) {
    return Stride(
        getDefaultStride(shape.values.front(), detail::DimValue::getStatic(1)));
  }

  detail::RecursiveDimValues values;
  values.reserve(shape.values.size());

  detail::DimValue runningStride = detail::DimValue::getStatic(1);
  for (const detail::RecursiveDimValue &dim : llvm::reverse(shape.values)) {
    values.push_back(getDefaultStride(dim, runningStride));
    runningStride = multiply(runningStride, product(dim));
  }
  std::reverse(values.begin(), values.end());

  return Stride(std::move(values));
}

Coord::Coord() { has_error_ = true; }

Coord::Coord(detail::RecursiveDimValues values) : values(std::move(values)) {
  if (this->values.empty()) {
    has_error_ = true;
    return;
  }
  has_error_ = !hasValidCoordValues(this->values);
}

Coord::Coord(detail::RecursiveDimValue value) : scalarRoot(value.isLeaf()) {
  has_error_ = !hasValidCoordValues(value);
  if (scalarRoot) {
    values.push_back(std::move(value));
    return;
  }

  llvm::ArrayRef<detail::RecursiveDimValue> nestedValues = value.getValues();
  values.assign(nestedValues.begin(), nestedValues.end());
}

Coord::Coord(int64_t value)
    : Coord(detail::RecursiveDimValue(detail::DimValue::getStatic(value))) {}

Coord::Coord(CgErrorT)
    : values({detail::RecursiveDimValue(CgErrorT{})}), scalarRoot(true),
      has_error_(true) {}

std::string Coord::toString() const {
  if (has_error(*this)) {
    return "error_t";
  }
  return scalarRoot ? values.front().toString() : valuesToString(values);
}

std::optional<Coord> Coord::fromString(mlir::StringRef str) {
  std::optional<detail::RecursiveDimValue> value = parseValue(str);
  if (!value || !hasValidCoordValues(*value)) {
    return std::nullopt;
  }
  return Coord(std::move(*value));
}

size_t Coord::rank() const { return values.size(); }

std::int64_t Coord::as_int() const {
  if (has_error(*this)) {
    llvm::report_fatal_error("cannot convert error coordinate to integer");
  }
  if (!scalarRoot) {
    llvm::report_fatal_error("cannot convert non-scalar coordinate to integer");
  }
  return values.front().as_int();
}

void Coord::append(int64_t value) {
  scalarRoot = false;
  values.emplace_back(detail::DimValue::getStatic(value));
  has_error_ = !hasValidCoordValues(values);
}

void Coord::append(Coord value) {
  bool valueHasError = has_error(value);
  scalarRoot = false;
  values.push_back(value.scalarRoot
                       ? std::move(value.values.front())
                       : makeRecursiveValue(std::move(value.values)));
  has_error_ = valueHasError || !hasValidCoordValues(values);
}

const detail::RecursiveDimValue &Coord::get(size_t index) const {
  return values[index];
}

const detail::RecursiveDimValue &Coord::operator[](size_t index) const {
  return values[index];
}

Coord::const_iterator Coord::begin() const { return const_iterator(*this, 0); }

Coord::const_iterator Coord::end() const {
  return const_iterator(*this, rank());
}

Coord::const_iterator Coord::cbegin() const { return begin(); }

Coord::const_iterator Coord::cend() const { return end(); }

bool operator==(const Coord &lhs, const Coord &rhs) {
  if (has_error(lhs) || has_error(rhs)) {
    return has_error(lhs) == has_error(rhs);
  }
  return lhs.scalarRoot == rhs.scalarRoot && lhs.values == rhs.values;
}

bool operator!=(const Coord &lhs, const Coord &rhs) { return !(lhs == rhs); }

Layout::Layout() = default;

Layout::Layout(Shape shape)
    : shapeValue(std::move(shape)),
      strideValue(Stride::getDefault(this->shapeValue)),
      has_error_(::mlir::nv_tensor_ir::tcutegen::has_error(shapeValue) ||
                 ::mlir::nv_tensor_ir::tcutegen::has_error(strideValue)) {}

Layout::Layout(Shape shape, Stride stride)
    : shapeValue(std::move(shape)), strideValue(std::move(stride)) {
  has_error_ = ::mlir::nv_tensor_ir::tcutegen::has_error(this->shapeValue) ||
               ::mlir::nv_tensor_ir::tcutegen::has_error(this->strideValue);
  if (has_error_) {
    return;
  }
  if (rank(this->shapeValue) != rank(this->strideValue)) {
    has_error_ = true;
    return;
  }
  if (this->shapeValue.scalarRoot != this->strideValue.scalarRoot ||
      !areCongruent(this->shapeValue.values, this->strideValue.values)) {
    has_error_ = true;
  }
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

const Shape &Layout::shape() const { return shapeValue; }

const Stride &Layout::stride() const { return strideValue; }

detail::DimValue Layout::operator()(const Coord &coord) const {
  if (::mlir::nv_tensor_ir::tcutegen::has_error(*this) ||
      ::mlir::nv_tensor_ir::tcutegen::has_error(coord)) {
    return detail::DimValue(CgErrorT{});
  }
  if (shapeValue.scalarRoot != coord.scalarRoot ||
      !areCongruent(coord.values, shapeValue.values)) {
    return detail::DimValue(CgErrorT{});
  }
  if (coord.scalarRoot) {
    return computeLayoutOffset(coord.values.front(),
                               strideValue.values.front());
  }
  return computeLayoutOffset(coord.values, strideValue.values);
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
  has_error_ = ::mlir::nv_tensor_ir::tcutegen::has_error(shapeValue) ||
               ::mlir::nv_tensor_ir::tcutegen::has_error(strideValue);
  if (has_error_) {
    return;
  }
  if (rank(shapeValue) != rank(strideValue) ||
      shapeValue.scalarRoot != strideValue.scalarRoot ||
      !areCongruent(shapeValue.values, strideValue.values)) {
    has_error_ = true;
  }
}

Layout::const_iterator Layout::begin() const {
  return const_iterator(*this, 0);
}

Layout::const_iterator Layout::end() const {
  return const_iterator(*this, rank(shape()));
}

Layout::const_iterator Layout::cbegin() const { return begin(); }

Layout::const_iterator Layout::cend() const { return end(); }

bool Layout::has_error() const { return has_error_; }

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
  if (!shape || !stride) {
    return std::nullopt;
  }
  if (shape->rank() != stride->rank()) {
    return std::nullopt;
  }
  if (shape->scalarRoot != stride->scalarRoot ||
      !areCongruent(shape->values, stride->values)) {
    return std::nullopt;
  }
  return Layout(std::move(*shape), std::move(*stride));
}

size_t rank(const Shape &shape) { return shape.rank(); }

size_t rank(const Stride &stride) { return stride.rank(); }

size_t rank(const Layout &layout) { return rank(layout.shape()); }

size_t rank(const Coord &coord) { return coord.rank(); }

size_t rank(const detail::RecursiveDimValue &value) { return value.rank(); }

bool has_error(const Shape &shape) {
  return shape.has_error_ ||
         llvm::any_of(shape.values, [](const detail::RecursiveDimValue &dim) {
           return detail::has_error(dim);
         });
}

bool has_error(const Stride &stride) {
  return stride.has_error_ ||
         llvm::any_of(stride.values, [](const detail::RecursiveDimValue &dim) {
           return detail::has_error(dim);
         });
}

bool has_error(const Layout &layout) {
  return layout.has_error_ || has_error(layout.shape()) ||
         has_error(layout.stride());
}

bool has_error(const Coord &coord) {
  return coord.has_error_ ||
         llvm::any_of(coord.values, [](const detail::RecursiveDimValue &dim) {
           return detail::has_error(dim);
         });
}

size_t depth(const Shape &shape) {
  if (has_error(shape)) {
    return 0;
  }
  if (shape.scalarRoot) {
    return getDepth(shape.values.front());
  }
  return getDepth(ArrayRef<detail::RecursiveDimValue>(shape.values));
}

size_t depth(const Stride &stride) {
  if (has_error(stride)) {
    return 0;
  }
  if (stride.scalarRoot) {
    return getDepth(stride.values.front());
  }
  return getDepth(ArrayRef<detail::RecursiveDimValue>(stride.values));
}

size_t depth(const Layout &layout) { return depth(layout.shape()); }

size_t depth(const Coord &coord) {
  if (has_error(coord)) {
    return 0;
  }
  if (coord.scalarRoot) {
    return getDepth(coord.values.front());
  }
  return getDepth(ArrayRef<detail::RecursiveDimValue>(coord.values));
}

size_t depth(const detail::RecursiveDimValue &value) { return getDepth(value); }

bool is_static(const Shape &shape) {
  if (has_error(shape)) {
    return false;
  }
  return llvm::all_of(shape.values, [](const detail::RecursiveDimValue &dim) {
    return dim.isStatic();
  });
}

bool is_static(const Stride &stride) {
  if (has_error(stride)) {
    return false;
  }
  return llvm::all_of(stride.values, [](const detail::RecursiveDimValue &dim) {
    return dim.isStatic();
  });
}

bool is_static(const Layout &layout) {
  if (has_error(layout)) {
    return false;
  }
  return is_static(layout.shape()) && is_static(layout.stride());
}

bool is_static(const Coord &coord) {
  if (has_error(coord)) {
    return false;
  }
  return llvm::all_of(coord.values, [](const detail::RecursiveDimValue &dim) {
    return dim.isStatic();
  });
}

bool is_static(const DimValue &dim) { return dim.isStatic(); }

bool is_static(const detail::RecursiveDimValue &value) {
  if (detail::has_error(value)) {
    return false;
  }
  return value.isStatic();
}

int64_t get_divisibility(const DimValue &dim) { return dim.getDivisibility(); }

int64_t get_divisibility(const Shape &shape) { return shape.getDivisibility(); }

int64_t get_divisibility(const Stride &stride) {
  return stride.getDivisibility();
}

int64_t get_divisibility(const detail::RecursiveDimValue &value) {
  return value.getDivisibility();
}

int64_t static_size(const detail::RecursiveDimValue &value) {
  return product(value).as_int();
}

int64_t static_size(const Shape &shape) {
  detail::DimValue result = detail::DimValue::getStatic(1);
  for (size_t i = 0, e = rank(shape); i < e; ++i) {
    result =
        multiply(result, detail::DimValue::getStatic(static_size(shape, i)));
  }
  return result.as_int();
}

int64_t static_size(const Shape &shape, size_t index) {
  assert(index < rank(shape));
  return static_size(shape[index]);
}

int64_t static_size(const Stride &stride) {
  detail::DimValue result = detail::DimValue::getStatic(1);
  for (size_t i = 0, e = rank(stride); i < e; ++i) {
    result =
        multiply(result, detail::DimValue::getStatic(static_size(stride, i)));
  }
  return result.as_int();
}

int64_t static_size(const Stride &stride, size_t index) {
  assert(index < rank(stride));
  return static_size(stride[index]);
}

int64_t static_size(const Layout &layout) {
  return static_size(layout.shape());
}

int64_t static_size(const Layout &layout, size_t index) {
  assert(index < rank(layout));
  return static_size(layout.shape(), index);
}

Shape flatten(const Shape &shape) {
  if (has_error(shape) || shape.scalarRoot) {
    return shape;
  }
  return Shape(flattenRecursiveValues(shape.values));
}

Stride flatten(const Stride &stride) {
  if (has_error(stride) || stride.scalarRoot) {
    return stride;
  }
  return Stride(flattenRecursiveValues(stride.values));
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
  detail::RecursiveDimValues flatShape = flatLayout.shape().values;
  detail::RecursiveDimValues flatStride = flatLayout.stride().values;
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

  detail::RecursiveDimValues shapeValues;
  detail::RecursiveDimValues strideValues;
  shapeValues.reserve(lys.size());
  strideValues.reserve(lys.size());
  for (const Layout &layout : lys) {
    const Shape &shape = layout.shape();
    const Stride &stride = layout.stride();
    shapeValues.push_back(shape.scalarRoot ? shape.values.front()
                                           : makeRecursiveValue(shape.values));
    strideValues.push_back(stride.scalarRoot
                               ? stride.values.front()
                               : makeRecursiveValue(stride.values));
  }

  return Layout(Shape(std::move(shapeValues)), Stride(std::move(strideValues)));
}

Shape reverse(const Shape &shape) {
  if (has_error(shape) || shape.scalarRoot) {
    return shape;
  }

  detail::RecursiveDimValues values;
  values.reserve(shape.values.size());
  for (const detail::RecursiveDimValue &value : llvm::reverse(shape.values)) {
    values.push_back(value);
  }
  return Shape(std::move(values));
}

Stride reverse(const Stride &stride) {
  if (has_error(stride) || stride.scalarRoot) {
    return stride;
  }

  detail::RecursiveDimValues values;
  values.reserve(stride.values.size());
  for (const detail::RecursiveDimValue &value : llvm::reverse(stride.values)) {
    values.push_back(value);
  }
  return Stride(std::move(values));
}

Layout reverse(const Layout &layout) {
  if (has_error(layout)) {
    return Layout(CgErrorT{});
  }
  return Layout(reverse(layout.shape()), reverse(layout.stride()));
}

Shape take(int begin, int end, const Shape &shape) {
  if (has_error(shape)) {
    return Shape(CgErrorT{});
  }

  std::optional<detail::RecursiveDimValues> values =
      takeValues(shape.values, begin, end);
  if (!values) {
    return Shape(CgErrorT{});
  }
  return Shape(std::move(*values));
}

Stride take(int begin, int end, const Stride &stride) {
  if (has_error(stride)) {
    return Stride(CgErrorT{});
  }

  std::optional<detail::RecursiveDimValues> values =
      takeValues(stride.values, begin, end);
  if (!values) {
    return Stride(CgErrorT{});
  }
  return Stride(std::move(*values));
}

Layout take(int begin, int end, const Layout &layout) {
  if (has_error(layout)) {
    return Layout(CgErrorT{});
  }
  return Layout(take(begin, end, layout.shape()),
                take(begin, end, layout.stride()));
}

Coord idx2crd(int64_t index, const Shape &shape) {
  if (has_error(shape) || index < 0) {
    return Coord(CgErrorT{});
  }
  if (shape.scalarRoot) {
    return Coord(index);
  }

  int64_t stride = 1;
  detail::RecursiveDimValues coordValues;
  coordValues.reserve(shape.values.size());
  for (const detail::RecursiveDimValue &dim : llvm::reverse(shape.values)) {
    coordValues.push_back(idx2crdValue(index, dim, stride));
    if (detail::has_error(coordValues.back())) {
      return Coord(CgErrorT{});
    }
  }
  std::reverse(coordValues.begin(), coordValues.end());
  return Coord(std::move(coordValues));
}

} // namespace mlir::nv_tensor_ir::tcutegen
