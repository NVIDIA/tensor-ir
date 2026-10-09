// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Reference/SimplifiedTensor.h"

#include "tensor_ir/Reference/TensorMemory.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "ConstantUtils.h"
#include "DataTypeDispatch.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>

namespace mlir::nv_tensor_ir::reference {

// ============ Utility Functions ============
bool isIntegerDataType(DataType dtype) {
  switch (dtype) {
  case DataType::BOOL:
  case DataType::INT32:
  case DataType::UINT32:
  case DataType::INT8:
  case DataType::UINT8:
    return true;
  default:
    return false;
  }
}

bool isUnsignedIntegerDataType(DataType dtype) {
  return dtype == DataType::UINT8 || dtype == DataType::UINT32;
}

bool isFloatDataType(DataType dtype) {
  switch (dtype) {
  case DataType::FLOAT32:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::F8E4M3FN:
  case DataType::F8E5M2:
  case DataType::F4E2M1FN:
  case DataType::DOUBLE:
    return true;
  default:
    return false;
  }
}

size_t getDataTypeSize(DataType dtype) {
  size_t size = 0;
  (void)detail::dispatchDataType(dtype, [&](auto tag) {
    size = sizeof(typename decltype(tag)::Type);
    return Status::Ok();
  });
  return size;
}

const char *getEnumName(DataType dtype) {
  switch (dtype) {
  case DataType::BOOL:
    return "BOOL";
  case DataType::FLOAT32:
    return "FLOAT32";
  case DataType::FLOAT16:
    return "FLOAT16";
  case DataType::BFLOAT16:
    return "BFLOAT16";
  case DataType::F8E4M3FN:
    return "F8E4M3FN";
  case DataType::F8E5M2:
    return "F8E5M2";
  case DataType::F4E2M1FN:
    return "F4E2M1FN";
  case DataType::INT32:
    return "INT32";
  case DataType::UINT32:
    return "UINT32";
  case DataType::INT8:
    return "INT8";
  case DataType::UINT8:
    return "UINT8";
  case DataType::DOUBLE:
    return "DOUBLE";
  default:
    return "UNKNOWN";
  }
}

// ============ TensorDesc Implementation ============
void TensorDesc::computeStrides() {
  if (dims.empty()) {
    strides.clear();
    totalElements = 1;
    return;
  }

  // TODO: we should infer the strides from the input mlir file
  strides.assign(dims.size(), 1);
  int64_t runningStride = 1;

  for (size_t reverseIdx = dims.size(); reverseIdx > 0; --reverseIdx) {
    size_t idx = reverseIdx - 1;
    strides[idx] = runningStride;
    runningStride *= dims[idx];
  }

  totalElements = std::accumulate(dims.begin(), dims.end(), 1LL,
                                  std::multiplies<int64_t>());
}

std::optional<size_t> TensorDesc::getSizeInBytes() const {
  if (dims.size() != strides.size()) {
    return std::nullopt;
  }

  auto getStorageSize = [&](size_t elementCount) -> std::optional<size_t> {
    if (dtype == DataType::F4E2M1FN) {
      return elementCount / 2 + elementCount % 2;
    }
    const size_t elementSize = getDataTypeSize(dtype);
    if (elementSize == 0 ||
        elementCount > std::numeric_limits<size_t>::max() / elementSize) {
      return std::nullopt;
    }
    return elementCount * elementSize;
  };

  if (dims.empty()) {
    if (totalElements != 1) {
      return std::nullopt;
    }
    return getStorageSize(1);
  }

  for (size_t dim = 0; dim < dims.size(); ++dim) {
    if (dims[dim] < 0 || strides[dim] < 0) {
      return std::nullopt;
    }
    if (dims[dim] == 0) {
      return size_t{0};
    }
  }

  size_t maxElementOffset = 0;
  for (size_t dim = 0; dim < dims.size(); ++dim) {
    const size_t extent = static_cast<size_t>(dims[dim] - 1);
    const size_t stride = static_cast<size_t>(strides[dim]);
    if (extent != 0 &&
        stride >
            (std::numeric_limits<size_t>::max() - maxElementOffset) / extent) {
      return std::nullopt;
    }
    maxElementOffset += extent * stride;
  }

  if (maxElementOffset == std::numeric_limits<size_t>::max()) {
    return std::nullopt;
  }
  return getStorageSize(maxElementOffset + 1);
}

Status TensorDesc::validateStridedOffsetsWithinAllocation(
    const std::string &tensorName) const {
  if (strides.size() != dims.size()) {
    return Status::InvalidArgument("Tensor " + tensorName +
                                   " stride rank does not match tensor rank");
  }

  for (size_t dim = 0; dim < dims.size(); ++dim) {
    if (dims[dim] <= 0) {
      return Status::InvalidArgument("Tensor " + tensorName +
                                     " must have positive dimensions");
    }
    if (strides[dim] < 0) {
      return Status::InvalidArgument("Tensor " + tensorName +
                                     " must have non-negative strides");
    }
  }

  if (!getSizeInBytes()) {
    return Status::InvalidArgument("Tensor " + tensorName +
                                   " strided storage size overflowed");
  }

  return Status::Ok();
}

void TensorDesc::print() const {
  llvm::outs() << "TensorDesc(dims: [";
  for (size_t i = 0; i < dims.size(); ++i) {
    llvm::outs() << dims[i];
    if (i < dims.size() - 1) {
      llvm::outs() << ", ";
    }
  }
  llvm::outs() << "], strides: [";
  for (size_t i = 0; i < strides.size(); ++i) {
    llvm::outs() << strides[i];
    if (i < strides.size() - 1) {
      llvm::outs() << ", ";
    }
  }
  llvm::outs() << "], dtype: " << getEnumName(dtype)
               << ", totalElements: " << totalElements << ")"
               << "\n";
}

// ============ SimplifiedTensor Implementation ============
SimplifiedTensor::SimplifiedTensor() : memory(nullptr) {}

SimplifiedTensor::~SimplifiedTensor() {
  // TensorMemory will be automatically cleaned up by unique_ptr
}

SimplifiedTensor::SimplifiedTensor(SimplifiedTensor &&other) noexcept
    : name(std::move(other.name)), desc(std::move(other.desc)),
      memory(std::move(other.memory)) {
  // unique_ptr move handles ownership transfer automatically
}

SimplifiedTensor &
SimplifiedTensor::operator=(SimplifiedTensor &&other) noexcept {
  if (this != &other) {
    // Move all members from other
    name = std::move(other.name);
    desc = std::move(other.desc);
    memory =
        std::move(other.memory); // unique_ptr handles cleanup automatically
  }
  return *this;
}

bool SimplifiedTensor::init(const std::string &name,
                            llvm::ArrayRef<int64_t> dims, DataType dtype) {
  this->name = name;
  desc.dims.assign(dims.begin(), dims.end());
  desc.dtype = dtype;

  desc.computeStrides();

  return true;
}

bool SimplifiedTensor::init(const std::string &name,
                            llvm::ArrayRef<int64_t> dims,
                            llvm::ArrayRef<int64_t> strides, DataType dtype) {
  if (dims.size() != strides.size()) {
    llvm::errs() << "Error: Tensor rank and stride rank must match\n";
    return false;
  }

  this->name = name;
  desc.dims.assign(dims.begin(), dims.end());
  desc.strides.assign(strides.begin(), strides.end());
  desc.dtype = dtype;

  desc.totalElements = std::accumulate(dims.begin(), dims.end(), 1LL,
                                       std::multiplies<int64_t>());

  return true;
}

bool SimplifiedTensor::allocate() {
  if (isAllocated()) {
    return true; // Already allocated
  }

  std::optional<size_t> sizeInBytes = desc.getSizeInBytes();
  if (!sizeInBytes) {
    llvm::errs() << "Tensor " << name << " storage size overflowed\n";
    return false;
  }
  memory = std::make_unique<TensorMemory>(*sizeInBytes);
  return memory->isAllocated();
}

bool SimplifiedTensor::isAllocated() const {
  return memory && memory->isAllocated();
}

Status SimplifiedTensor::validate(const std::string &role) const {
  if (!isAllocated()) {
    return Status::InvalidArgument(role + " tensor must be allocated");
  }
  if (!hostPtr()) {
    return Status::InvalidArgument(role +
                                   " tensor must have host memory allocated");
  }
  return desc.validateStridedOffsetsWithinAllocation(name);
}

// === Data Access ===
void *SimplifiedTensor::hostPtr() {
  return memory ? memory->hostPtr() : nullptr;
}

const void *SimplifiedTensor::hostPtr() const {
  return memory ? memory->hostPtr() : nullptr;
}

int64_t SimplifiedTensor::getStorageIndex(int64_t linearIndex) const {
  assert(desc.dims.size() == desc.strides.size() &&
         "tensor rank and stride rank must match");
  assert(linearIndex >= 0 && linearIndex < desc.totalElements &&
         "linear index must be within tensor bounds");

  int64_t storageIndex = 0;
  int64_t remaining = linearIndex;
  const size_t rank = desc.dims.size();
  for (size_t reverseIdx = rank; reverseIdx > 0; --reverseIdx) {
    const size_t dim = reverseIdx - 1;
    assert(desc.dims[dim] > 0 && "tensor dimensions must be positive");
    const int64_t dimIndex = remaining % desc.dims[dim];
    remaining /= desc.dims[dim];
    storageIndex += dimIndex * desc.strides[dim];
  }
  assert(remaining == 0 && "linear index must fully map into tensor rank");
  return storageIndex;
}

int64_t
SimplifiedTensor::getStorageIndex(llvm::ArrayRef<int64_t> indices) const {
  assert(indices.size() == desc.dims.size() &&
         "logical index rank must match tensor rank");
  assert(desc.dims.size() == desc.strides.size() &&
         "tensor rank and stride rank must match");

  int64_t storageIndex = 0;
  for (size_t dim = 0, rank = indices.size(); dim < rank; ++dim) {
    assert(desc.dims[dim] > 0 && "tensor dimensions must be positive");
    assert(indices[dim] >= 0 && indices[dim] < desc.dims[dim] &&
           "logical index must be within tensor bounds");
    storageIndex += indices[dim] * desc.strides[dim];
  }
  return storageIndex;
}

template <typename T>
static void fillLogicalElements(SimplifiedTensor &tensor, T value) {
  for (int64_t i = 0, e = tensor.getTotalElements(); i < e; ++i) {
    tensor.hostElement<T>(i) = value;
  }
}

template <typename T, typename Generator>
static void generateLogicalElements(SimplifiedTensor &tensor,
                                    Generator generator) {
  for (int64_t i = 0, e = tensor.getTotalElements(); i < e; ++i) {
    tensor.hostElement<T>(i) = generator();
  }
}

bool SimplifiedTensor::fillConstant(double value) {
  if (!hostPtr() || desc.dtype == DataType::UINT8) {
    return false;
  }
  Status status = detail::dispatchDataType(desc.dtype, [&](auto tag) -> Status {
    using T = typename decltype(tag)::Type;
    if constexpr (std::is_same_v<T, bool>) {
      fillLogicalElements<T>(*this, value != 0.0);
    } else if constexpr (std::is_integral_v<T>) {
      TIR_ASSIGN_OR_RETURN(auto converted,
                           detail::convertFiniteConstantToInteger<T>(
                               value, getEnumName(desc.dtype)));
      fillLogicalElements<T>(*this, converted);
    } else if constexpr (std::is_floating_point_v<T>) {
      fillLogicalElements<T>(*this, static_cast<T>(value));
    } else {
      // CUDA low-precision types round through float, including double inputs.
      fillLogicalElements<T>(*this, T(static_cast<float>(value)));
    }
    return Status::Ok();
  });
  return status.ok();
}

bool SimplifiedTensor::fillRandom(float minVal, float maxVal,
                                  std::mt19937 &gen) {
  if (!hostPtr() || desc.dtype == DataType::UINT8) {
    return false;
  }
  std::uniform_real_distribution<float> dis(minVal, maxVal);
  if (desc.dtype == DataType::F4E2M1FN) {
    constexpr std::array<float, 8> kPositiveValues = {0.0f, 0.5f, 1.0f, 1.5f,
                                                      2.0f, 3.0f, 4.0f, 6.0f};
    std::array<uint8_t, 16> candidates{};
    size_t candidateCount = 0;
    for (size_t magnitude = 0; magnitude < kPositiveValues.size();
         ++magnitude) {
      const float positive = kPositiveValues[magnitude];
      if (minVal <= positive && positive <= maxVal) {
        candidates[candidateCount++] = static_cast<uint8_t>(magnitude);
      }
      if (magnitude != 0 && minVal <= -positive && -positive <= maxVal) {
        candidates[candidateCount++] = static_cast<uint8_t>(magnitude | 0x8);
      }
    }
    std::optional<size_t> sizeInBytes = getSizeInBytes();
    if (candidateCount == 0 || !sizeInBytes) {
      return false;
    }

    std::memset(hostPtr(), 0, *sizeInBytes);
    auto *data = static_cast<uint8_t *>(hostPtr());
    std::uniform_int_distribution<size_t> fp4Dis(0, candidateCount - 1);
    for (int64_t i = 0; i < desc.totalElements; ++i) {
      const uint8_t encoded = candidates[fp4Dis(gen)];
      const int64_t storageIndex = getStorageIndex(i);
      const unsigned shift = 4 * (storageIndex % 2);
      uint8_t &storageByte = data[storageIndex / 2];
      storageByte = (storageByte & ~(0x0F << shift)) | (encoded << shift);
    }
    return true;
  }

  Status status = detail::dispatchDataType(desc.dtype, [&](auto tag) -> Status {
    using T = typename decltype(tag)::Type;
    if constexpr (std::is_same_v<T, bool>) {
      std::bernoulli_distribution boolDis(0.5);
      generateLogicalElements<T>(*this, [&]() { return boolDis(gen); });
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      const double lower =
          std::max(std::ceil(static_cast<double>(minVal)), 0.0);
      const double upper =
          std::min(std::floor(static_cast<double>(maxVal)),
                   static_cast<double>(std::numeric_limits<T>::max()));
      if (lower > upper) {
        return Status::InvalidArgument();
      }
      std::uniform_int_distribution<T> intDis(static_cast<T>(lower),
                                              static_cast<T>(upper));
      generateLogicalElements<T>(*this, [&]() { return intDis(gen); });
    } else if constexpr (std::is_integral_v<T>) {
      auto lower = static_cast<int32_t>(std::ceil(minVal));
      auto upper = static_cast<int32_t>(std::floor(maxVal));
      if constexpr (std::is_same_v<T, int8_t>) {
        lower = std::max(lower,
                         static_cast<int32_t>(std::numeric_limits<T>::min()));
        upper = std::min(upper,
                         static_cast<int32_t>(std::numeric_limits<T>::max()));
      }
      if (lower > upper) {
        return Status::InvalidArgument();
      }
      std::uniform_int_distribution<int32_t> intDis(lower, upper);
      generateLogicalElements<T>(*this,
                                 [&]() { return static_cast<T>(intDis(gen)); });
    } else {
      generateLogicalElements<T>(*this,
                                 [&]() { return static_cast<T>(dis(gen)); });
    }
    return Status::Ok();
  });
  return status.ok();
}

void SimplifiedTensor::print() const {
  llvm::outs() << "SimplifiedTensor(name: " << name << ", desc: ";
  desc.print();
  llvm::outs() << "hostPtr: " << hostPtr() << ", sizeInBytes: ";
  if (std::optional<size_t> sizeInBytes = getSizeInBytes()) {
    llvm::outs() << *sizeInBytes;
  } else {
    llvm::outs() << "overflow";
  }
  llvm::outs() << ", allocated: " << (isAllocated() ? "Yes" : "No") << ")"
               << "\n";
}

static void printElem(int8_t const *mem, size_t elementOffset, DataType type) {
  if (type == DataType::F4E2M1FN) {
    constexpr std::array<float, 8> kPositiveValues = {0.0f, 0.5f, 1.0f, 1.5f,
                                                      2.0f, 3.0f, 4.0f, 6.0f};
    const auto *data = reinterpret_cast<const uint8_t *>(mem);
    const uint8_t encoded =
        (data[elementOffset / 2] >> (4 * (elementOffset % 2))) & 0x0F;
    float value = kPositiveValues[encoded & 0x7];
    if ((encoded & 0x8) != 0) {
      value = -value;
    }
    llvm::outs() << llvm::format("%.3f", value);
    return;
  }

  Status status = detail::dispatchDataType(type, [&](auto tag) {
    using T = typename decltype(tag)::Type;
    T value = reinterpret_cast<const T *>(mem)[elementOffset];
    if constexpr (std::is_same_v<T, bool>) {
      llvm::outs() << (value ? "true" : "false");
    } else if constexpr (std::is_integral_v<T>) {
      llvm::outs() << +value;
    } else if constexpr (std::is_same_v<T, double>) {
      llvm::outs() << llvm::format("%.6f", value);
    } else {
      llvm::outs() << llvm::format("%.3f", static_cast<float>(value));
    }
    return Status::Ok();
  });
  if (!status.ok()) {
    llvm::outs() << "Unknown data type";
  }
}

// ============================================================================
// Internal helper for printing tensor data content
// ============================================================================

void SimplifiedTensor::printData(const void *dataPtr, const char *label,
                                 const char *ptrName,
                                 const void *ptrValue) const {
  llvm::outs() << label << ": SimplifiedTensor(name: " << name << ", desc: ";
  desc.print();
  llvm::outs() << ptrName << ": " << ptrValue << ", sizeInBytes: ";
  if (std::optional<size_t> sizeInBytes = getSizeInBytes()) {
    llvm::outs() << *sizeInBytes;
  } else {
    llvm::outs() << "overflow";
  }
  llvm::outs() << ", allocated: " << (isAllocated() ? "Yes" : "No") << ")"
               << "\n";

  // Print data content
  if (!dataPtr || !isAllocated()) {
    llvm::outs() << "  [No data - not allocated or null pointer]"
                 << "\n";
    return;
  }

  if (desc.totalElements == 0) {
    llvm::outs() << "  [No data - zero elements]"
                 << "\n";
    return;
  }

  int8_t const *data = static_cast<int8_t const *>(dataPtr);
  const auto &dims = desc.dims;
  const auto &strides = desc.strides;

  // Limit maximum elements to print to avoid too much output
  const int64_t maxPrintPerDim = 10;
  const int64_t maxTotalPrint = 1000;

  llvm::outs() << "  Data content (shape: [";
  for (size_t i = 0; i < dims.size(); ++i) {
    llvm::outs() << dims[i];
    if (i < dims.size() - 1) {
      llvm::outs() << ", ";
    }
  }
  llvm::outs() << "]):"
               << "\n";

  if (dims.empty()) {
    llvm::outs() << "  ";
    printElem(data, /*elementOffset=*/0, desc.dtype);
    llvm::outs() << "\n";
    return;
  }

  // Recursive lambda function to print multi-dimensional array
  std::function<void(int64_t, int64_t, const std::string &, int64_t &)>
      printRecursive;
  printRecursive = [&](int64_t dimIdx, int64_t offset,
                       const std::string &indent, int64_t &printedCount) {
    if (printedCount >= maxTotalPrint) {
      llvm::outs() << indent << "..."
                   << "\n";
      return;
    }

    if (dimIdx == static_cast<int64_t>(dims.size()) - 1) {
      // Innermost dimension, print actual data
      llvm::outs() << indent << "[";
      int64_t numToPrint = std::min(dims[dimIdx], maxPrintPerDim);
      bool truncated = dims[dimIdx] > maxPrintPerDim;

      for (int64_t i = 0; i < numToPrint && printedCount < maxTotalPrint; ++i) {
        int64_t elemOffset = offset + i * strides[dimIdx];
        printElem(data, elemOffset, desc.dtype);
        printedCount++;
        if (i < numToPrint - 1 || (truncated && i == numToPrint - 1)) {
          llvm::outs() << ", ";
        }
      }

      if (truncated) {
        llvm::outs() << "... (" << dims[dimIdx] << " total)";
      }
      llvm::outs() << "]";
    } else {
      // Outer dimension, recurse
      llvm::outs() << indent << "["
                   << "\n";
      int64_t numToPrint = std::min(dims[dimIdx], maxPrintPerDim);
      bool truncated = dims[dimIdx] > maxPrintPerDim;

      for (int64_t i = 0; i < numToPrint && printedCount < maxTotalPrint; ++i) {
        int64_t newOffset = offset + i * strides[dimIdx];
        printRecursive(dimIdx + 1, newOffset, indent + "  ", printedCount);
        if (i < numToPrint - 1 || (truncated && i == numToPrint - 1)) {
          llvm::outs() << ",";
        }
        llvm::outs() << "\n";
      }

      if (truncated) {
        llvm::outs() << indent << "  ... (" << dims[dimIdx]
                     << " total along this dimension)"
                     << "\n";
      }
      llvm::outs() << indent << "]";
    }
  };

  int64_t printedCount = 0;
  printRecursive(0, 0, "  ", printedCount);
  llvm::outs() << "\n";
}

// ============================================================================
// Public print methods
// ============================================================================

void SimplifiedTensor::printHostData() const {
  printData(hostPtr(), "Host Data", "hostPtr", hostPtr());
}

// ============================================================================
// Helper to convert values to double for comparison (internal)
// ============================================================================

template <typename T>
static inline double toDoubleForCheck(const T &val) {
  return static_cast<double>(val);
}

static inline double toDoubleForCheck(const half &val) {
  return static_cast<double>(__half2float(val));
}

static inline double toDoubleForCheck(const nv_bfloat16 &val) {
  return static_cast<double>(static_cast<float>(val));
}

static inline double toDoubleForCheck(const __nv_fp8_e4m3 &val) {
  return static_cast<double>(static_cast<float>(val));
}

static inline double toDoubleForCheck(const __nv_fp8_e5m2 &val) {
  return static_cast<double>(static_cast<float>(val));
}

// Unified tensor comparison kernel for checkRef
template <typename T>
static bool checkRefKernel(const SimplifiedTensor &tensor, const void *ref_ptr,
                           const void *test_ptr, int64_t numElements,
                           double rtol, double atol, bool verbose) {
  const T *ref = static_cast<const T *>(ref_ptr);
  const T *test = static_cast<const T *>(test_ptr);

  bool all_equal = true;
  int64_t numMismatches = 0;
  const int64_t maxPrint = 30;

  double maxSeenRtol = 0;
  double maxSeenAtol = 0;

  for (int64_t i = 0; i < numElements; ++i) {
    const int64_t storageIndex = tensor.getStorageIndex(i);
    double refVal = toDoubleForCheck(ref[storageIndex]);
    double testVal = toDoubleForCheck(test[storageIndex]);
    double nowRtol = std::abs(refVal - testVal) / (std::abs(refVal) + 1e-8);
    double nowAtol = std::abs(refVal - testVal);

    maxSeenRtol = std::max(maxSeenRtol, nowRtol);
    maxSeenAtol = std::max(maxSeenAtol, nowAtol);

    bool mismatch;
    if constexpr (std::is_integral_v<T>) {
      mismatch = ref[storageIndex] != test[storageIndex];
    } else {
      mismatch = (nowRtol > rtol) && (nowAtol > atol);
    }

    if (mismatch) {
      all_equal = false;
      numMismatches++;

      if (verbose && numMismatches <= maxPrint) {
        llvm::outs() << "  Mismatch at index " << i << ": " << refVal << " vs "
                     << testVal;
        if constexpr (std::is_integral_v<T>) {
          llvm::outs() << " (exact comparison)\n";
        } else {
          llvm::outs() << " (rtol=" << nowRtol << ", rtolThreshold=" << rtol
                       << ", absDiff=" << nowAtol << ", absThreshold=" << atol
                       << ")\n";
        }
      }
    }
  }

  if (verbose && numMismatches > 0) {
    llvm::outs() << "Total mismatches: " << numMismatches << " / "
                 << numElements << " (" << (100.0 * numMismatches / numElements)
                 << "%)";
    if constexpr (!std::is_integral_v<T>) {
      llvm::outs() << " (maxSeenRtol=" << maxSeenRtol
                   << ", maxSeenAtol=" << maxSeenAtol << ")";
    }
    llvm::outs() << "\n";
  }

  return all_equal;
}

bool SimplifiedTensor::checkRefAgainst(const void *gpuResultHost, double rtol,
                                       double atol, bool verbose) const {
  // Check if both memory regions are available
  if (!hostPtr() || !gpuResultHost) {
    if (verbose) {
      llvm::outs() << "SimplifiedTensor::checkRefAgainst - Missing memory: "
                      "cpuReferenceHost("
                   << hostPtr() << ") or gpuResultHost(" << gpuResultHost << ")"
                   << "\n";
    }
    return false;
  }

  int64_t numElements = getTotalElements();

  if (verbose) {
    llvm::outs()
        << "SimplifiedTensor::checkRefAgainst - Comparing CPU reference host "
           "vs GPU result host "
           "for tensor '"
        << name << "'"
        << "\n";
  }

  bool matches = false;
  Status status = detail::dispatchDataType(desc.dtype, [&](auto tag) {
    using T = typename decltype(tag)::Type;
    matches = checkRefKernel<T>(*this, hostPtr(), gpuResultHost, numElements,
                                rtol, atol, verbose);
    return Status::Ok();
  });
  if (!status.ok() && verbose) {
    llvm::outs() << "Unsupported data type: " << getEnumName(desc.dtype)
                 << "\n";
  }
  return matches;
}

// Factory functions
std::shared_ptr<SimplifiedTensor> createTensor(const std::string &name,
                                               llvm::ArrayRef<int64_t> dims,
                                               DataType dtype) {
  auto tensor = std::make_shared<SimplifiedTensor>();
  if (!tensor->init(name, dims, dtype)) {
    return nullptr;
  }
  if (!tensor->allocate()) {
    return nullptr;
  }
  return tensor;
}

std::shared_ptr<SimplifiedTensor> createTensor(const std::string &name,
                                               llvm::ArrayRef<int64_t> dims,
                                               llvm::ArrayRef<int64_t> strides,
                                               DataType dtype) {
  auto tensor = std::make_shared<SimplifiedTensor>();
  if (!tensor->init(name, dims, strides, dtype)) {
    return nullptr;
  }
  if (!tensor->allocate()) {
    return nullptr;
  }
  return tensor;
}

} // namespace mlir::nv_tensor_ir::reference
