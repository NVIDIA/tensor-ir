// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_SUPPORT_TCUTEGEN_H_
#define TENSOR_IR_SUPPORT_TCUTEGEN_H_

// TensorIR-owned facade over cutegen. Public TensorIR headers should include
// this file instead of cutegen headers directly so cutegen can be replaced in
// one place later.

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir {
class Diagnostic;
} // namespace mlir
namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::nv_tensor_ir::tcutegen {

struct CgErrorT {};

enum class SequenceKind { Shape, Stride, Coord };
template <SequenceKind Kind>
class DimSequence;
using Shape = DimSequence<SequenceKind::Shape>;
using Stride = DimSequence<SequenceKind::Stride>;
using Coord = DimSequence<SequenceKind::Coord>;
class Layout;

namespace detail {

template <typename T>
inline constexpr bool isTensorIRCutegenValue =
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, Shape> ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, Stride> ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, Coord> ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, Layout>;

} // namespace detail

template <typename Lhs, typename Rhs,
          std::enable_if_t<detail::isTensorIRCutegenValue<Lhs> &&
                               detail::isTensorIRCutegenValue<Rhs>,
                           bool> = true>
bool is_congruent(const Lhs &lhs, const Rhs &rhs);

namespace detail {

template <typename T, bool Mutable>
class SequenceIterator;
template <typename T>
using IndexedIterator = SequenceIterator<T, false>;
template <typename T>
class IndexedReference;
template <typename T>
using MutableIndexedIterator = SequenceIterator<T, true>;

// Represents a single dimension of a shape or stride. Can be either a static
// integer or a dynamic DimValue, with divisibility and integer width metadata.
class DimValue {
public:
  // Creates a static DimValue. Value must be non-negative. Invalid input
  // creates an error DimValue.
  static DimValue getStatic(int64_t value);

  // Creates a dynamic DimValue. Divisibility must be positive and width must be
  // a cutegen-supported integer width; Invalid input creates an error DimValue.
  static DimValue getDynamic(int64_t divisibility, int32_t width = 32);

  // Creates an error DimValue.
  explicit DimValue(CgErrorT);

  // This method is composable with cutegen's string representation of dim
  // values.
  std::string toString() const;

  // This method is composable with cutegen's string representation of dim
  // values. If the input string is invalid, std::nullopt is returned.
  static std::optional<DimValue> fromString(mlir::StringRef str);

  // Returns true if the dimension is static.
  bool isStatic() const;

  // Cutegen API to ensure portability with cutegen.
  // Returns the static dimension value. Dynamic dimensions cause a crash.
  std::int64_t as_int() const;

  // Returns divisibility metadata matching cutegen: dynamic dimensions return
  // their encoded divisibility; static dimensions return abs(value), stored at
  // construction time by getStatic().
  int64_t getDivisibility() const;

  // Equality operators.
  friend bool operator==(const DimValue &lhs, const DimValue &rhs);
  friend bool operator!=(const DimValue &lhs, const DimValue &rhs);
  friend bool operator==(const DimValue &lhs, int64_t rhs);
  friend bool operator!=(const DimValue &lhs, int64_t rhs);

private:
  DimValue(int64_t value, bool isDynamic, int64_t divisibility, int32_t width);

  friend bool has_error(const DimValue &dim);

  int64_t value;
  bool isDynamic;
  int64_t divisibility;
  int32_t width;
  bool has_error_ = false;
};

// Recursive sequence of DimValues used to represent nested shape/stride modes.
class RecursiveDimValue {
public:
  explicit RecursiveDimValue(DimValue value);
  explicit RecursiveDimValue(std::vector<RecursiveDimValue> values);
  explicit RecursiveDimValue(CgErrorT);

  bool isLeaf() const { return leaf; }
  const DimValue &getLeaf() const;
  llvm::ArrayRef<RecursiveDimValue> getValues() const;
  std::string toString() const;
  size_t rank() const { return isLeaf() ? 1 : values.size(); }

  // Returns true if this scalar or recursive dimension value is fully static.
  bool isStatic() const;

  // Cutegen-like scalar APIs. Recursive values cause a crash.
  std::int64_t as_int() const;
  int64_t getDivisibility() const;

  // Equality operators.
  friend bool operator==(const RecursiveDimValue &lhs,
                         const RecursiveDimValue &rhs);
  friend bool operator!=(const RecursiveDimValue &lhs,
                         const RecursiveDimValue &rhs);
  friend bool operator==(const RecursiveDimValue &lhs, const DimValue &rhs);
  friend bool operator!=(const RecursiveDimValue &lhs, const DimValue &rhs);
  friend bool operator==(const RecursiveDimValue &lhs, int64_t rhs);
  friend bool operator!=(const RecursiveDimValue &lhs, int64_t rhs);
  friend bool operator==(int64_t lhs, const RecursiveDimValue &rhs);
  friend bool operator!=(int64_t lhs, const RecursiveDimValue &rhs);

private:
  friend bool has_error(const RecursiveDimValue &dim);

  bool leaf;
  DimValue value;
  std::vector<RecursiveDimValue> values;
  bool has_error_ = false;
};

using RecursiveDimValues = SmallVector<RecursiveDimValue>;

} // namespace detail

namespace detail {

template <typename T>
class IndexedReference {
public:
  IndexedReference() = default;
  IndexedReference(const IndexedReference &) = default;
  IndexedReference(T &value, size_t index) : value(&value), index(index) {}

  operator T() const { return T(value->get(index)); }

  IndexedReference &operator=(T newValue) {
    value->set(index, std::move(newValue));
    return *this;
  }

  IndexedReference &operator=(const IndexedReference &other) {
    return *this = static_cast<T>(other);
  }

  std::string toString() const { return static_cast<T>(*this).toString(); }

private:
  void reset(T &newValue, size_t newIndex) {
    value = &newValue;
    index = newIndex;
  }

  template <typename, bool>
  friend class SequenceIterator;

  T *value = nullptr;
  size_t index = 0;
};

/// Index-based traversal shared by value and mutable-proxy iterators.
template <typename T, bool Mutable>
class SequenceIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = std::conditional_t<Mutable, IndexedReference<T> &, T>;
  using Container = std::conditional_t<Mutable, T, const T>;

  SequenceIterator() = default;
  SequenceIterator(Container &value, size_t index)
      : value(&value), index(index) {}
  reference operator*() const {
    if constexpr (Mutable) {
      current.reset(*value, index);
      return current;
    } else {
      return T(value->get(index));
    }
  }
  SequenceIterator &operator++() {
    ++index;
    return *this;
  }
  SequenceIterator operator++(int) {
    auto copy = *this;
    ++*this;
    return copy;
  }
  friend bool operator==(const SequenceIterator &lhs,
                         const SequenceIterator &rhs) {
    return lhs.value == rhs.value && lhs.index == rhs.index;
  }
  friend bool operator!=(const SequenceIterator &lhs,
                         const SequenceIterator &rhs) {
    return !(lhs == rhs);
  }

private:
  Container *value = nullptr;
  size_t index = 0;
  mutable IndexedReference<T> current;
};

} // namespace detail
/// A recursive sequence with role-specific validation. Shapes require positive
/// static dimensions; strides and coordinates allow zero. Scalar roots remain
/// distinct from singleton tuples throughout parsing, iteration and comparison.
template <SequenceKind Kind>
class DimSequence {
public:
  using const_iterator = detail::IndexedIterator<DimSequence>;
  using iterator =
      std::conditional_t<Kind == SequenceKind::Coord, const_iterator,
                         detail::MutableIndexedIterator<DimSequence>>;

  /// Empty shapes/strides are valid; default coordinates are errors until
  /// populated. Parsing an empty tuple preserves its explicit tuple root.
  DimSequence();
  explicit DimSequence(int64_t value);
  explicit DimSequence(CgErrorT);
  DimSequence(llvm::ArrayRef<int64_t> values);
  explicit DimSequence(detail::RecursiveDimValue value);
  explicit DimSequence(detail::RecursiveDimValues values);

  static std::optional<DimSequence> fromString(mlir::StringRef str);
  static Stride getDefault(const Shape &shape);
  std::string toString() const;
  int64_t as_int() const;
  int64_t getDivisibility() const;
  void append(int64_t value);
  void append(DimSequence value);
  void appendDynamic(int64_t divisibility = 1, int32_t width = 32);
  void set(size_t index, DimSequence value);

  const detail::RecursiveDimValue &get(size_t index) const {
    return values[index];
  }
  const detail::RecursiveDimValue &operator[](size_t index) const {
    return get(index);
  }
  ArrayRef<detail::RecursiveDimValue> getValues() const { return values; }
  bool isScalar() const { return scalarRoot; }
  bool hasError() const;
  bool isStatic() const;
  size_t getDepth() const;
  int64_t staticSize() const;
  DimSequence flattened() const;
  DimSequence reversed() const;
  DimSequence taken(int begin, int end) const;

  iterator begin() { return iterator(*this, 0); }
  iterator end() { return iterator(*this, values.size()); }
  const_iterator begin() const { return const_iterator(*this, 0); }
  const_iterator end() const { return const_iterator(*this, values.size()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  friend bool operator==(const DimSequence &lhs, const DimSequence &rhs) {
    if (lhs.hasError() || rhs.hasError()) {
      return lhs.hasError() == rhs.hasError();
    }
    return lhs.scalarRoot == rhs.scalarRoot && lhs.values == rhs.values;
  }
  friend bool operator!=(const DimSequence &lhs, const DimSequence &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator==(const DimSequence &lhs, int64_t rhs) {
    return !lhs.hasError() && lhs.scalarRoot && lhs.values.front() == rhs;
  }
  friend bool operator!=(const DimSequence &lhs, int64_t rhs) {
    return !(lhs == rhs);
  }
  friend bool operator==(int64_t lhs, const DimSequence &rhs) {
    return rhs == lhs;
  }
  friend bool operator!=(int64_t lhs, const DimSequence &rhs) {
    return !(lhs == rhs);
  }

private:
  bool validDimensions() const;
  detail::RecursiveDimValue asRecursiveValue() &&;
  const detail::RecursiveDimValue &requireScalar(const char *action) const;
  detail::RecursiveDimValues values;
  bool scalarRoot = false;
  bool has_error_ = false;
};

class Layout {
public:
  using const_iterator = detail::IndexedIterator<Layout>;

  // Creates an empty layout. Append layouts to build a non-empty layout.
  Layout();

  // Constructs a compact logical layout for \p shape. The last dimension is
  // contiguous.
  Layout(Shape shape);

  // Shape and stride must be congruent. Invalid or non-congruent inputs create
  // an error layout; callers can check this with has_error().
  Layout(Shape shape, Stride stride);

  Layout(int64_t shape, int64_t stride);

  // Creates an error layout.
  explicit Layout(CgErrorT);

  // This method is composable with cutegen's string representation of layouts.
  std::string toString() const;

  // This method is composable with cutegen's string representation of layouts.
  // If the input string is invalid, std::nullopt is returned.
  static std::optional<Layout> fromString(mlir::StringRef str);

  // Gets the shape of the layout.
  const Shape &shape() const { return shapeValue; }

  // Gets the stride of the layout.
  const Stride &stride() const { return strideValue; }

  // Computes the linear layout offset for a coordinate congruent with this
  // layout's shape.
  detail::DimValue operator()(const Coord &coord) const;

  // Gets a sub-layout at the given index. Scalar sub-layouts remain scalar.
  Layout get(size_t index) const;

  // Append a nested layout mode.
  void append(Layout other);

  // Iterate over sub-layouts in storage order.
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  // Returns true if this layout was constructed with non-congruent shape and
  // stride inputs.
  bool has_error() const { return has_error_; }

  // Equality operators.
  friend bool operator==(const Layout &lhs, const Layout &rhs);
  friend bool operator!=(const Layout &lhs, const Layout &rhs);

private:
  friend bool has_error(const Layout &layout);
  friend Layout reverse(const Layout &layout);
  friend Layout take(int begin, int end, const Layout &layout);

  Shape shapeValue;
  Stride strideValue;
  bool has_error_ = false;
};

// Custom definitions of functions to be used with Shape, Stride and Layout
// defined above.
size_t depth(const Layout &layout);
size_t depth(const detail::RecursiveDimValue &value);
int64_t get_divisibility(const detail::DimValue &dim);
int64_t get_divisibility(const detail::RecursiveDimValue &value);
bool has_error(const Layout &layout);
bool is_static(const detail::DimValue &dim);
bool is_static(const Layout &layout);
bool is_static(const detail::RecursiveDimValue &value);
size_t rank(const Layout &layout);
size_t rank(const detail::RecursiveDimValue &value);
int64_t static_size(const detail::RecursiveDimValue &value);
int64_t static_size(const Layout &layout);
int64_t static_size(const Layout &layout, size_t index);
Layout coalesce(const Layout &layout);
Layout flatten(const Layout &layout);
Layout make_layout(std::vector<Layout> lys);
Layout reverse(const Layout &layout);
Layout take(int begin, int end, const Layout &layout);
Coord idx2crd(int64_t index, const Shape &shape);

template <SequenceKind K>
size_t depth(const DimSequence<K> &v) {
  return v.getDepth();
}
template <SequenceKind K>
int64_t get_divisibility(const DimSequence<K> &v) {
  return v.getDivisibility();
}
template <SequenceKind K>
bool has_error(const DimSequence<K> &v) {
  return v.hasError();
}
template <SequenceKind K>
bool is_static(const DimSequence<K> &v) {
  return v.isStatic();
}
template <SequenceKind K>
size_t rank(const DimSequence<K> &v) {
  return v.getValues().size();
}
template <SequenceKind K>
int64_t static_size(const DimSequence<K> &v) {
  return v.staticSize();
}
template <SequenceKind K>
int64_t static_size(const DimSequence<K> &v, size_t index) {
  assert(index < rank(v));
  return static_size(v[index]);
}
template <SequenceKind K>
DimSequence<K> flatten(const DimSequence<K> &v) {
  return v.flattened();
}
template <SequenceKind K>
DimSequence<K> reverse(const DimSequence<K> &v) {
  return v.reversed();
}
template <SequenceKind K>
DimSequence<K> take(int begin, int end, const DimSequence<K> &v) {
  return v.taken(begin, end);
}

namespace detail {
bool has_error(const DimValue &dim);
bool has_error(const RecursiveDimValue &dim);
bool is_congruent(const RecursiveDimValue &lhs, const RecursiveDimValue &rhs);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DimValue &dim);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const RecursiveDimValue &dim);
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const DimValue &dim);
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag,
                             const RecursiveDimValue &dim);
} // namespace detail

template <
    typename TRecVar, typename T,
    std::enable_if_t<detail::isTensorIRCutegenValue<TRecVar>, bool> = true>
TRecVar repeat(const size_t n, const T &value) {
  TRecVar result;
  for (size_t i = 0; i < n; ++i) {
    result.append(value);
  }
  return result;
}

template <typename Lhs, typename Rhs,
          std::enable_if_t<detail::isTensorIRCutegenValue<Lhs> &&
                               detail::isTensorIRCutegenValue<Rhs>,
                           bool>>
bool is_congruent(const Lhs &lhs, const Rhs &rhs) {
  if (has_error(lhs) || has_error(rhs)) {
    return false;
  }

  // Returns the shape/stride as-is. For layout, returns its shape. Layout shape
  // and stride are already congruent.
  auto getProfile = [](const auto &value) -> decltype(auto) {
    using Value = std::remove_cv_t<std::remove_reference_t<decltype(value)>>;
    if constexpr (std::is_same_v<Value, Layout>) {
      return (value.shape());
    } else {
      return (value);
    }
  };

  const auto &lhsProfile = getProfile(lhs);
  const auto &rhsProfile = getProfile(rhs);
  if (lhsProfile.isScalar() != rhsProfile.isScalar() ||
      lhsProfile.getValues().size() != rhsProfile.getValues().size()) {
    return false;
  }
  for (size_t i = 0, e = lhsProfile.getValues().size(); i < e; ++i) {
    if (!detail::is_congruent(lhsProfile.getValues()[i],
                              rhsProfile.getValues()[i])) {
      return false;
    }
  }
  return true;
}

template <typename T, typename TString,
          std::enable_if_t<detail::isTensorIRCutegenValue<T>, bool> = true>
std::optional<T> from_string(const TString &str) {
  return T::fromString(StringRef(str));
}

template <typename T,
          std::enable_if_t<detail::isTensorIRCutegenValue<T>, bool> = true>
std::string to_string(const T &t) {
  return t.toString();
}

template <typename T,
          std::enable_if_t<detail::isTensorIRCutegenValue<T>, bool> = true>
decltype(auto) get(const T &t, size_t index) {
  return t.get(index);
}

} // namespace mlir::nv_tensor_ir::tcutegen

#endif // TENSOR_IR_SUPPORT_TCUTEGEN_H_
