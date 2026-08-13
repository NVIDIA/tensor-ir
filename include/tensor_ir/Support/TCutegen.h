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

class Shape;
class Stride;
class Layout;
class Coord;

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

template <typename T>
class IndexedIterator;
template <typename T>
class IndexedReference;
template <typename T>
class MutableIndexedIterator;

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

  bool isLeaf() const;
  const DimValue &getLeaf() const;
  llvm::ArrayRef<RecursiveDimValue> getValues() const;
  std::string toString() const;
  size_t rank() const;

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

class Layout;
class Coord;

// Potentially recursive sequence of DimValues representing a shape.
class Shape {
public:
  using iterator = detail::MutableIndexedIterator<Shape>;
  using const_iterator = detail::IndexedIterator<Shape>;

  // Creates an empty shape. Append dimensions to build a non-empty shape.
  Shape();

  // Creates a shape from MLIR shaped-type dimensions. `ShapedType::kDynamic`
  // dimensions become dynamic DimValues. Static dimensions must be >= 1.
  // Invalid input creates an error Shape.
  Shape(llvm::ArrayRef<int64_t> values);

  // Creates a scalar shape. Invalid input creates an error Shape.
  explicit Shape(int64_t value);

  // Creates an error shape.
  explicit Shape(CgErrorT);

  // This method is composable with cutegen's string representation of shapes.
  std::string toString() const;

  // This method is composable with cutegen's string representation of shapes.
  // If the input string is invalid, std::nullopt is returned.
  static std::optional<Shape> fromString(mlir::StringRef str);

  // Cutegen-like scalar API. Non-scalar shapes cause a crash.
  std::int64_t as_int() const;

  // Append a static dimension. Static dimensions must be >= 1.
  void append(int64_t value);

  // Append a nested shape mode.
  void append(Shape value);

  // Append a dynamic dimension. Divisibility must be positive and width must be
  // a cutegen-supported integer width.
  void appendDynamic(int64_t divisibility = 1, int32_t width = 32);

  // Get the dimension at the given index. Index must be in [0, rank()).
  const detail::RecursiveDimValue &get(size_t index) const;

  // Get the dimension at the given index. Index must be in [0, rank()).
  const detail::RecursiveDimValue &operator[](size_t index) const;

  // Get the array of values.
  ArrayRef<detail::RecursiveDimValue> getValues() const { return values; }

  // Iterate over sub-shapes in storage order.
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;

  // Equality operators.
  friend bool operator==(const Shape &lhs, const Shape &rhs);
  friend bool operator!=(const Shape &lhs, const Shape &rhs);
  friend bool operator==(const Shape &lhs, int64_t rhs);
  friend bool operator!=(const Shape &lhs, int64_t rhs);
  friend bool operator==(int64_t lhs, const Shape &rhs);
  friend bool operator!=(int64_t lhs, const Shape &rhs);

private:
  // These mirror cutegen free-function APIs and are intentionally private on
  // the TensorIR replacement type.
  size_t rank() const;
  int64_t getDivisibility() const;

  Shape(SmallVector<detail::DimValue> values);
  Shape(detail::RecursiveDimValues values);
  explicit Shape(detail::RecursiveDimValue value);
  void set(size_t index, Shape value);

  friend class detail::IndexedReference<Shape>;
  friend class detail::IndexedIterator<Shape>;
  friend class Layout;
  friend class Stride;
  friend Layout coalesce(const Layout &layout);
  friend Shape flatten(const Shape &shape);
  friend Layout make_layout(std::vector<Layout> lys);
  friend Shape reverse(const Shape &shape);
  friend Shape take(int begin, int end, const Shape &shape);
  friend Coord idx2crd(int64_t index, const Shape &shape);
  friend size_t depth(const Shape &shape);
  friend int64_t get_divisibility(const Shape &shape);
  friend bool has_error(const Shape &shape);
  friend bool is_static(const Shape &shape);
  friend size_t rank(const Shape &shape);
  template <typename Lhs, typename Rhs,
            std::enable_if_t<detail::isTensorIRCutegenValue<Lhs> &&
                                 detail::isTensorIRCutegenValue<Rhs>,
                             bool>>
  friend bool is_congruent(const Lhs &lhs, const Rhs &rhs);

  detail::RecursiveDimValues values;
  bool scalarRoot = false;
  bool has_error_ = false;
};

// Potentially recursive sequence of DimValues representing a stride.
class Stride {
public:
  using iterator = detail::MutableIndexedIterator<Stride>;
  using const_iterator = detail::IndexedIterator<Stride>;

  // Creates an empty stride. Append dimensions to build a non-empty stride.
  Stride();

  // Creates a scalar stride. Invalid input creates an error Stride.
  explicit Stride(int64_t value);

  // Creates an error stride.
  explicit Stride(CgErrorT);

  // This method is composable with cutegen's string representation of strides.
  std::string toString() const;

  // This method is composable with cutegen's string representation of strides.
  // If the input string is invalid, std::nullopt is returned.
  static std::optional<Stride> fromString(mlir::StringRef str);

  // Creates the compact logical stride for \p shape. The last dimension is
  // contiguous and nested modes follow the same convention recursively.
  static Stride getDefault(const Shape &shape);

  // Cutegen-like scalar API. Non-scalar strides cause a crash.
  std::int64_t as_int() const;

  // Append a static dimension.
  void append(int64_t value);

  // Append a nested stride mode.
  void append(Stride value);

  // Append a dynamic dimension. Divisibility must be positive and width must be
  // a cutegen-supported integer width.
  void appendDynamic(int64_t divisibility = 1, int32_t width = 32);

  // Get the dimension at the given index. Index must be in [0, rank()).
  const detail::RecursiveDimValue &get(size_t index) const;

  // Get the dimension at the given index. Index must be in [0, rank()).
  const detail::RecursiveDimValue &operator[](size_t index) const;

  // Get the array of values.
  ArrayRef<detail::RecursiveDimValue> getValues() const { return values; }

  // Iterate over sub-strides in storage order.
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;

  // Equality operators.
  friend bool operator==(const Stride &lhs, const Stride &rhs);
  friend bool operator!=(const Stride &lhs, const Stride &rhs);
  friend bool operator==(const Stride &lhs, int64_t rhs);
  friend bool operator!=(const Stride &lhs, int64_t rhs);
  friend bool operator==(int64_t lhs, const Stride &rhs);
  friend bool operator!=(int64_t lhs, const Stride &rhs);

private:
  // These mirror cutegen free-function APIs and are intentionally private on
  // the TensorIR replacement type.
  size_t rank() const;
  int64_t getDivisibility() const;

  Stride(SmallVector<detail::DimValue> values);
  Stride(detail::RecursiveDimValues values);
  explicit Stride(detail::RecursiveDimValue value);
  void set(size_t index, Stride value);

  friend class detail::IndexedReference<Stride>;
  friend class detail::IndexedIterator<Stride>;
  friend class Layout;
  friend Layout coalesce(const Layout &layout);
  friend Stride flatten(const Stride &stride);
  friend Layout make_layout(std::vector<Layout> lys);
  friend Stride reverse(const Stride &stride);
  friend Stride take(int begin, int end, const Stride &stride);

  friend size_t depth(const Stride &stride);
  friend int64_t get_divisibility(const Stride &stride);
  friend bool has_error(const Stride &stride);
  friend bool is_static(const Stride &stride);
  friend size_t rank(const Stride &stride);
  template <typename Lhs, typename Rhs,
            std::enable_if_t<detail::isTensorIRCutegenValue<Lhs> &&
                                 detail::isTensorIRCutegenValue<Rhs>,
                             bool>>
  friend bool is_congruent(const Lhs &lhs, const Rhs &rhs);

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
  const Shape &shape() const;

  // Gets the stride of the layout.
  const Stride &stride() const;

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
  const_iterator cbegin() const;
  const_iterator cend() const;

  // Returns true if this layout was constructed with non-congruent shape and
  // stride inputs.
  bool has_error() const;

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

// Potentially recursive sequence of coordinate values. The structure mirrors a
// Shape and can be used as the domain coordinate for a Layout.
class Coord {
public:
  using const_iterator = detail::IndexedIterator<Coord>;

  // Creates an empty error coordinate. Append coordinates to build a valid
  // coordinate.
  Coord();

  // Creates a scalar coordinate. Invalid input creates an error Coord.
  explicit Coord(int64_t value);

  // Creates an error coordinate.
  explicit Coord(CgErrorT);

  // This method is composable with cutegen's string representation of
  // coordinates.
  std::string toString() const;

  // This method is composable with cutegen's string representation of
  // coordinates. If the input string is invalid, std::nullopt is returned.
  static std::optional<Coord> fromString(mlir::StringRef str);

  // Cutegen-like scalar API. Non-scalar coordinates cause a crash.
  std::int64_t as_int() const;

  // Append a static coordinate value. Static coordinates must be non-negative.
  void append(int64_t value);

  // Append a nested coordinate mode.
  void append(Coord value);

  // Get the coordinate at the given index. Index must be in [0, rank()).
  const detail::RecursiveDimValue &get(size_t index) const;

  // Get the coordinate at the given index. Index must be in [0, rank()).
  const detail::RecursiveDimValue &operator[](size_t index) const;

  // Iterate over sub-coordinates in storage order.
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;

  // Equality operators.
  friend bool operator==(const Coord &lhs, const Coord &rhs);
  friend bool operator!=(const Coord &lhs, const Coord &rhs);

private:
  size_t rank() const;

  Coord(detail::RecursiveDimValues values);
  explicit Coord(detail::RecursiveDimValue value);

  friend class detail::IndexedIterator<Coord>;
  friend detail::DimValue Layout::operator()(const Coord &coord) const;
  friend bool has_error(const Coord &coord);
  friend Coord idx2crd(int64_t index, const Shape &shape);
  friend size_t depth(const Coord &coord);
  friend bool is_static(const Coord &coord);
  friend size_t rank(const Coord &coord);
  template <typename Lhs, typename Rhs,
            std::enable_if_t<detail::isTensorIRCutegenValue<Lhs> &&
                                 detail::isTensorIRCutegenValue<Rhs>,
                             bool>>
  friend bool is_congruent(const Lhs &lhs, const Rhs &rhs);

  detail::RecursiveDimValues values;
  bool scalarRoot = false;
  bool has_error_ = false;
};

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

  template <typename>
  friend class MutableIndexedIterator;

  T *value = nullptr;
  size_t index = 0;
};

template <typename T>
class MutableIndexedIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = IndexedReference<T> &;

  MutableIndexedIterator() = default;
  MutableIndexedIterator(T &value, size_t index)
      : value(&value), index(index) {}

  reference operator*() const {
    current.reset(*value, index);
    return current;
  }

  MutableIndexedIterator &operator++() {
    ++index;
    return *this;
  }

  MutableIndexedIterator operator++(int) {
    MutableIndexedIterator copy = *this;
    ++*this;
    return copy;
  }

  friend bool operator==(const MutableIndexedIterator &lhs,
                         const MutableIndexedIterator &rhs) {
    return lhs.value == rhs.value && lhs.index == rhs.index;
  }

  friend bool operator!=(const MutableIndexedIterator &lhs,
                         const MutableIndexedIterator &rhs) {
    return !(lhs == rhs);
  }

private:
  T *value = nullptr;
  size_t index = 0;
  mutable IndexedReference<T> current;
};

template <typename T>
class IndexedIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = T;

  IndexedIterator() = default;
  IndexedIterator(const T &value, size_t index) : value(&value), index(index) {}

  T operator*() const {
    if constexpr (std::is_same_v<T, Layout>) {
      return value->get(index);
    } else {
      return T(value->get(index));
    }
  }

  IndexedIterator &operator++() {
    ++index;
    return *this;
  }

  IndexedIterator operator++(int) {
    IndexedIterator copy = *this;
    ++*this;
    return copy;
  }

  friend bool operator==(const IndexedIterator &lhs,
                         const IndexedIterator &rhs) {
    return lhs.value == rhs.value && lhs.index == rhs.index;
  }

  friend bool operator!=(const IndexedIterator &lhs,
                         const IndexedIterator &rhs) {
    return !(lhs == rhs);
  }

private:
  const T *value = nullptr;
  size_t index = 0;
};

} // namespace detail

// Custom definitions of functions to be used with Shape, Stride and Layout
// defined above.
size_t depth(const Shape &shape);
size_t depth(const Stride &stride);
size_t depth(const Layout &layout);
size_t depth(const Coord &coord);
size_t depth(const detail::RecursiveDimValue &value);
int64_t get_divisibility(const detail::DimValue &dim);
int64_t get_divisibility(const Shape &shape);
int64_t get_divisibility(const Stride &stride);
int64_t get_divisibility(const detail::RecursiveDimValue &value);
bool has_error(const Shape &shape);
bool has_error(const Stride &stride);
bool has_error(const Layout &layout);
bool has_error(const Coord &coord);
bool is_static(const detail::DimValue &dim);
bool is_static(const Shape &shape);
bool is_static(const Stride &stride);
bool is_static(const Layout &layout);
bool is_static(const Coord &coord);
bool is_static(const detail::RecursiveDimValue &value);
size_t rank(const Shape &shape);
size_t rank(const Stride &stride);
size_t rank(const Layout &layout);
size_t rank(const Coord &coord);
size_t rank(const detail::RecursiveDimValue &value);
int64_t static_size(const detail::RecursiveDimValue &value);
int64_t static_size(const Shape &shape);
int64_t static_size(const Shape &shape, size_t index);
int64_t static_size(const Stride &stride);
int64_t static_size(const Stride &stride, size_t index);
int64_t static_size(const Layout &layout);
int64_t static_size(const Layout &layout, size_t index);
Layout coalesce(const Layout &layout);
Shape flatten(const Shape &shape);
Stride flatten(const Stride &stride);
Layout flatten(const Layout &layout);
Layout make_layout(std::vector<Layout> lys);
Shape reverse(const Shape &shape);
Stride reverse(const Stride &stride);
Layout reverse(const Layout &layout);
Shape take(int begin, int end, const Shape &shape);
Stride take(int begin, int end, const Stride &stride);
Layout take(int begin, int end, const Layout &layout);
Coord idx2crd(int64_t index, const Shape &shape);

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
  if (lhsProfile.scalarRoot != rhsProfile.scalarRoot ||
      lhsProfile.values.size() != rhsProfile.values.size()) {
    return false;
  }
  for (size_t i = 0, e = lhsProfile.values.size(); i < e; ++i) {
    if (!detail::is_congruent(lhsProfile.values[i], rhsProfile.values[i])) {
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
