// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Reference/simplified_tensor.h"

#include "tensor_ir/Reference/tensor_memory.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "constant_utils.h"
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
  case DataType::DOUBLE:
    return true;
  default:
    return false;
  }
}

size_t getDataTypeSize(DataType dtype) {
  switch (dtype) {
  case DataType::BOOL:
    return sizeof(bool);
  case DataType::FLOAT32:
    return sizeof(float);
  case DataType::FLOAT16:
    return sizeof(uint16_t);
  case DataType::BFLOAT16:
    return sizeof(nv_bfloat16);
  case DataType::F8E4M3FN:
    return sizeof(__nv_fp8_e4m3);
  case DataType::F8E5M2:
    return sizeof(__nv_fp8_e5m2);
  case DataType::INT32:
    return sizeof(int32_t);
  case DataType::UINT32:
    return sizeof(uint32_t);
  case DataType::INT8:
    return sizeof(int8_t);
  case DataType::UINT8:
    return sizeof(uint8_t);
  case DataType::DOUBLE:
    return sizeof(double);
  default:
    return 0;
  }
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
  if (dims.empty() || dims.size() != strides.size()) {
    return std::nullopt;
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

  const size_t elementSize = getDataTypeSize(dtype);
  if (elementSize == 0 ||
      maxElementOffset >= std::numeric_limits<size_t>::max() / elementSize) {
    return std::nullopt;
  }
  return (maxElementOffset + 1) * elementSize;
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
  if (dims.empty()) {
    llvm::errs() << "Error: Empty dimensions\n";
    return false;
  }

  this->name = name;
  desc.dims.assign(dims.begin(), dims.end());
  desc.dtype = dtype;

  desc.computeStrides();

  return true;
}

bool SimplifiedTensor::init(const std::string &name,
                            llvm::ArrayRef<int64_t> dims,
                            llvm::ArrayRef<int64_t> strides, DataType dtype) {
  if (dims.empty()) {
    llvm::errs() << "Error: Empty dimensions\n";
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
  void *hostData = hostPtr();
  if (!hostData) {
    return false;
  }

  // Fill according to data type
  switch (desc.dtype) {
  case DataType::BOOL: {
    fillLogicalElements<bool>(*this, value != 0.0);
    break;
  }
  case DataType::FLOAT32: {
    fillLogicalElements<float>(*this, static_cast<float>(value));
    break;
  }
  case DataType::FLOAT16: {
    fillLogicalElements<half>(*this, __float2half(static_cast<float>(value)));
    break;
  }
  case DataType::BFLOAT16: {
    fillLogicalElements<nv_bfloat16>(*this,
                                     nv_bfloat16(static_cast<float>(value)));
    break;
  }
  case DataType::F8E4M3FN: {
    fillLogicalElements<__nv_fp8_e4m3>(
        *this, __nv_fp8_e4m3(static_cast<float>(value)));
    break;
  }
  case DataType::F8E5M2: {
    fillLogicalElements<__nv_fp8_e5m2>(
        *this, __nv_fp8_e5m2(static_cast<float>(value)));
    break;
  }
  case DataType::INT32: {
    auto intValue =
        detail::convertFiniteConstantToInteger<int32_t>(value, "INT32");
    if (!intValue.ok()) {
      return false;
    }
    fillLogicalElements<int32_t>(*this, *intValue);
    break;
  }
  case DataType::UINT32: {
    auto intValue =
        detail::convertFiniteConstantToInteger<uint32_t>(value, "UINT32");
    if (!intValue.ok()) {
      return false;
    }
    fillLogicalElements<uint32_t>(*this, *intValue);
    break;
  }
  case DataType::INT8: {
    auto intValue =
        detail::convertFiniteConstantToInteger<int8_t>(value, "INT8");
    if (!intValue.ok()) {
      return false;
    }
    fillLogicalElements<int8_t>(*this, *intValue);
    break;
  }
  case DataType::DOUBLE: {
    fillLogicalElements<double>(*this, static_cast<double>(value));
    break;
  }
  default:
    return false;
  }

  return true;
}

bool SimplifiedTensor::fillRandom(float minVal, float maxVal,
                                  std::mt19937 &gen) {
  void *hostData = hostPtr();
  if (!hostData) {
    return false;
  }

  std::uniform_real_distribution<float> dis(minVal, maxVal);

  switch (desc.dtype) {
  case DataType::BOOL: {
    std::bernoulli_distribution bool_dis(0.5);
    generateLogicalElements<bool>(*this, [&]() { return bool_dis(gen); });
    break;
  }
  case DataType::FLOAT32: {
    generateLogicalElements<float>(*this, [&]() { return dis(gen); });
    break;
  }
  case DataType::FLOAT16: {
    generateLogicalElements<half>(*this,
                                  [&]() { return __float2half(dis(gen)); });
    break;
  }
  case DataType::BFLOAT16: {
    generateLogicalElements<nv_bfloat16>(
        *this, [&]() { return nv_bfloat16(dis(gen)); });
    break;
  }
  case DataType::F8E4M3FN: {
    generateLogicalElements<__nv_fp8_e4m3>(
        *this, [&]() { return __nv_fp8_e4m3(dis(gen)); });
    break;
  }
  case DataType::F8E5M2: {
    generateLogicalElements<__nv_fp8_e5m2>(
        *this, [&]() { return __nv_fp8_e5m2(dis(gen)); });
    break;
  }
  case DataType::DOUBLE: {
    generateLogicalElements<double>(
        *this, [&]() { return static_cast<double>(dis(gen)); });
    break;
  }
  case DataType::INT32: {
    auto minInt = static_cast<int32_t>(std::ceil(minVal));
    auto maxInt = static_cast<int32_t>(std::floor(maxVal));
    if (minInt > maxInt) {
      return false;
    }
    std::uniform_int_distribution<int32_t> intDis(minInt, maxInt);
    generateLogicalElements<int32_t>(*this, [&]() { return intDis(gen); });
    break;
  }
  case DataType::UINT32: {
    const double clampedMin =
        std::max(std::ceil(static_cast<double>(minVal)), 0.0);
    const double clampedMax =
        std::min(std::floor(static_cast<double>(maxVal)),
                 static_cast<double>(std::numeric_limits<uint32_t>::max()));
    if (clampedMin > clampedMax) {
      return false;
    }
    std::uniform_int_distribution<uint32_t> intDis(
        static_cast<uint32_t>(clampedMin), static_cast<uint32_t>(clampedMax));
    generateLogicalElements<uint32_t>(*this, [&]() { return intDis(gen); });
    break;
  }
  case DataType::INT8: {
    auto minInt = static_cast<int32_t>(std::ceil(minVal));
    auto maxInt = static_cast<int32_t>(std::floor(maxVal));
    minInt = std::max(minInt,
                      static_cast<int32_t>(std::numeric_limits<int8_t>::min()));
    maxInt = std::min(maxInt,
                      static_cast<int32_t>(std::numeric_limits<int8_t>::max()));
    if (minInt > maxInt) {
      return false;
    }
    std::uniform_int_distribution<int32_t> intDis(minInt, maxInt);
    generateLogicalElements<int8_t>(
        *this, [&]() { return static_cast<int8_t>(intDis(gen)); });
    break;
  }
  default:
    return false;
  }

  return true;
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

template <typename T_ELEM>
static T_ELEM getElem(int8_t const *mem, size_t elementOffset, DataType type) {
  int64_t elemSizeInBytes = getDataTypeSize(type);
  int64_t offsetInBytes = elementOffset * elemSizeInBytes;
  return *(reinterpret_cast<T_ELEM const *>(mem + offsetInBytes));
}

static void printElem(int8_t const *mem, size_t elementOffset, DataType type) {
  switch (type) {
  case DataType::BOOL: {
    llvm::outs() << (getElem<bool>(mem, elementOffset, type) ? "true"
                                                             : "false");
    break;
  }
  case DataType::FLOAT32: {
    llvm::outs() << llvm::format("%.3f",
                                 getElem<float>(mem, elementOffset, type));
    break;
  }
  case DataType::FLOAT16: {
    half elem = getElem<half>(mem, elementOffset, type);
    llvm::outs() << llvm::format("%.3f", static_cast<float>(elem));
    break;
  }
  case DataType::BFLOAT16: {
    nv_bfloat16 elem = getElem<nv_bfloat16>(mem, elementOffset, type);
    llvm::outs() << llvm::format("%.3f", static_cast<float>(elem));
    break;
  }
  case DataType::F8E4M3FN: {
    __nv_fp8_e4m3 elem = getElem<__nv_fp8_e4m3>(mem, elementOffset, type);
    llvm::outs() << llvm::format("%.3f", static_cast<float>(elem));
    break;
  }
  case DataType::F8E5M2: {
    __nv_fp8_e5m2 elem = getElem<__nv_fp8_e5m2>(mem, elementOffset, type);
    llvm::outs() << llvm::format("%.3f", static_cast<float>(elem));
    break;
  }
  case DataType::INT32: {
    llvm::outs() << getElem<int32_t>(mem, elementOffset, type);
    break;
  }
  case DataType::UINT32: {
    llvm::outs() << getElem<uint32_t>(mem, elementOffset, type);
    break;
  }
  case DataType::INT8: {
    llvm::outs() << static_cast<int>(getElem<int8_t>(mem, elementOffset, type));
    break;
  }
  case DataType::UINT8: {
    llvm::outs() << static_cast<unsigned>(
        getElem<uint8_t>(mem, elementOffset, type));
    break;
  }
  case DataType::DOUBLE: {
    llvm::outs() << llvm::format("%.6f",
                                 getElem<double>(mem, elementOffset, type));
    break;
  }
  default:
    llvm::outs() << "Unknown data type";
    break;
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

  switch (desc.dtype) {
  case DataType::BOOL:
    return checkRefKernel<bool>(*this, hostPtr(), gpuResultHost, numElements,
                                rtol, atol, verbose);
  case DataType::FLOAT32:
    return checkRefKernel<float>(*this, hostPtr(), gpuResultHost, numElements,
                                 rtol, atol, verbose);
  case DataType::FLOAT16:
    return checkRefKernel<half>(*this, hostPtr(), gpuResultHost, numElements,
                                rtol, atol, verbose);
  case DataType::BFLOAT16:
    return checkRefKernel<nv_bfloat16>(*this, hostPtr(), gpuResultHost,
                                       numElements, rtol, atol, verbose);
  case DataType::F8E4M3FN:
    return checkRefKernel<__nv_fp8_e4m3>(*this, hostPtr(), gpuResultHost,
                                         numElements, rtol, atol, verbose);
  case DataType::F8E5M2:
    return checkRefKernel<__nv_fp8_e5m2>(*this, hostPtr(), gpuResultHost,
                                         numElements, rtol, atol, verbose);
  case DataType::DOUBLE:
    return checkRefKernel<double>(*this, hostPtr(), gpuResultHost, numElements,
                                  rtol, atol, verbose);
  case DataType::INT32:
    return checkRefKernel<int32_t>(*this, hostPtr(), gpuResultHost, numElements,
                                   rtol, atol, verbose);
  case DataType::UINT32:
    return checkRefKernel<uint32_t>(*this, hostPtr(), gpuResultHost,
                                    numElements, rtol, atol, verbose);
  case DataType::INT8:
    return checkRefKernel<int8_t>(*this, hostPtr(), gpuResultHost, numElements,
                                  rtol, atol, verbose);
  case DataType::UINT8:
    return checkRefKernel<uint8_t>(*this, hostPtr(), gpuResultHost, numElements,
                                   rtol, atol, verbose);
  default:
    if (verbose) {
      llvm::outs() << "Unsupported data type: " << getEnumName(desc.dtype)
                   << "\n";
    }
    return false;
  }
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