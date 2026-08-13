// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Reference/reference_node.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"

#include "constant_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::nv_tensor_ir::reference {

// ============================================================================
// Private Kernel Implementations (Internal Use)
// ============================================================================

namespace {

/// Associates a tensor with the role name used in validation diagnostics.
struct NamedTensor {
  NamedTensor(std::string name, std::shared_ptr<SimplifiedTensor> tensor)
      : name(std::move(name)), tensor(std::move(tensor)) {}

  std::string name;
  std::shared_ptr<SimplifiedTensor> tensor;
};

/// Small validation sets normally contain only an operation's operands/results.
using NamedTensors = mlir::SmallVector<NamedTensor>;

/// Reject null tensor pointers before any validation dereferences them.
static Status validateTensorPointers(const NamedTensors &tensors) {
  for (const NamedTensor &namedTensor : tensors) {
    if (!namedTensor.tensor) {
      return Status::InvalidArgument(namedTensor.name + " tensor is null");
    }
  }
  return Status::Ok();
}

/// Validate the memory and descriptor invariants of every tensor.
static Status validateTensors(const NamedTensors &tensors) {
  TIR_RETURN_IF_ERROR(validateTensorPointers(tensors));
  for (const NamedTensor &namedTensor : tensors) {
    TIR_RETURN_IF_ERROR(namedTensor.tensor->validate(namedTensor.name));
  }
  return Status::Ok();
}

/// Require every tensor to have the same element type as the first tensor.
static Status validateSameType(const NamedTensors &tensors) {
  TIR_RETURN_IF_ERROR(validateTensorPointers(tensors));
  const size_t numTensors = tensors.size();
  if (numTensors < 2) {
    return Status::Ok();
  }

  const DataType expectedType = tensors.front().tensor->getDataType();
  for (size_t tensorIdx = 1; tensorIdx < numTensors; ++tensorIdx) {
    if (tensors[tensorIdx].tensor->getDataType() != expectedType) {
      return Status::InvalidArgument(tensors.front().name + " and " +
                                     tensors[tensorIdx].name +
                                     " tensors must have the same data type");
    }
  }
  return Status::Ok();
}

/// Require every tensor to have the same shape as the first tensor.
static Status validateSameShape(const NamedTensors &tensors) {
  TIR_RETURN_IF_ERROR(validateTensorPointers(tensors));
  const size_t numTensors = tensors.size();
  if (numTensors < 2) {
    return Status::Ok();
  }

  llvm::ArrayRef<int64_t> expectedShape = tensors.front().tensor->getDims();
  for (size_t tensorIdx = 1; tensorIdx < numTensors; ++tensorIdx) {
    if (tensors[tensorIdx].tensor->getDims() != expectedShape) {
      return Status::InvalidArgument(tensors.front().name + " and " +
                                     tensors[tensorIdx].name +
                                     " tensors must have the same shape");
    }
  }
  return Status::Ok();
}

/// Report the rejected type and operation in reference-node diagnostics.
static Status unsupportedDataType(DataType dtype, const char *operationName) {
  return Status::InvalidArgument("Unsupported data type " +
                                 std::string(getEnumName(dtype)) + " for " +
                                 operationName);
}

/// Tag type that carries the canonical C++ element type for a DataType.
template <typename T>
struct DataTypeTag {
  using Type = T;
};

template <typename T, typename... CandidateTypes>
constexpr bool isAnyOf = (std::is_same_v<T, CandidateTypes> || ...);

/// Canonical C++ element types supported by numeric Convert operations.
template <typename T>
constexpr bool isSupportedConvertNumericType =
    isAnyOf<T, float, half, nv_bfloat16, __nv_fp8_e4m3, __nv_fp8_e5m2, int32_t,
            double>;

/// Invoke \p callback with a tag for the canonical C++ type of \p dtype.
///
/// Callers retain responsibility for validating which data types their
/// operation supports. The callback must return a Status.
template <typename Callback>
static Status dispatchDataType(DataType dtype, Callback &&callback) {
  switch (dtype) {
  case DataType::BOOL:
    return std::forward<Callback>(callback)(DataTypeTag<bool>{});
  case DataType::FLOAT32:
    return std::forward<Callback>(callback)(DataTypeTag<float>{});
  case DataType::FLOAT16:
    return std::forward<Callback>(callback)(DataTypeTag<half>{});
  case DataType::BFLOAT16:
    return std::forward<Callback>(callback)(DataTypeTag<nv_bfloat16>{});
  case DataType::F8E4M3FN:
    return std::forward<Callback>(callback)(DataTypeTag<__nv_fp8_e4m3>{});
  case DataType::F8E5M2:
    return std::forward<Callback>(callback)(DataTypeTag<__nv_fp8_e5m2>{});
  case DataType::INT32:
    return std::forward<Callback>(callback)(DataTypeTag<int32_t>{});
  case DataType::UINT32:
    return std::forward<Callback>(callback)(DataTypeTag<uint32_t>{});
  case DataType::INT8:
    return std::forward<Callback>(callback)(DataTypeTag<int8_t>{});
  case DataType::UINT8:
    return std::forward<Callback>(callback)(DataTypeTag<uint8_t>{});
  case DataType::DOUBLE:
    return std::forward<Callback>(callback)(DataTypeTag<double>{});
  }
  return unsupportedDataType(dtype, "data type dispatch");
}

/// Associates a runtime data type with its type-specialized kernel.
template <typename Kernel>
struct DataTypeKernelEntry {
  DataType dtype;
  Kernel kernel;
};

/// Return the kernel registered for \p dtype, or nullptr when none exists.
template <typename Kernel, size_t NumEntries>
static Kernel
lookupDataTypeKernel(DataType dtype,
                     const DataTypeKernelEntry<Kernel> (&entries)[NumEntries]) {
  for (const DataTypeKernelEntry<Kernel> &entry : entries) {
    if (entry.dtype == dtype) {
      return entry.kernel;
    }
  }
  return nullptr;
}

/// Floating and integer element types supported by Constant (excludes BOOL).
///
/// Kept separate from isSupportedIotaDataType() even though both sets are
/// currently identical: each excludes BOOL for an unrelated reason, so the two
/// are expected to diverge.
///
/// TODO: Support BOOL here.
static bool isSupportedConstantDataType(DataType dtype) {
  switch (dtype) {
  case DataType::FLOAT32:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::F8E4M3FN:
  case DataType::F8E5M2:
  case DataType::DOUBLE:
  case DataType::INT32:
  case DataType::UINT32:
  case DataType::INT8:
  case DataType::UINT8:
    return true;
  case DataType::BOOL:
    return false;
  }
  return false;
}

/// Floating and integer element types supported by Iota (excludes BOOL).
///
/// BOOL is excluded because the iota sequence 0, 1, 2, ... collapses to
/// false, true, true, ..., which conveys no useful information.
static bool isSupportedIotaDataType(DataType dtype) {
  switch (dtype) {
  case DataType::FLOAT32:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::F8E4M3FN:
  case DataType::F8E5M2:
  case DataType::DOUBLE:
  case DataType::INT32:
  case DataType::UINT32:
  case DataType::INT8:
  case DataType::UINT8:
    return true;
  case DataType::BOOL:
    return false;
  }
  return false;
}

/// Numeric convert element types (excludes BOOL, UINT32, INT8, and UINT8).
static bool isSupportedConvertNumericDataType(DataType dtype) {
  bool isSupported = false;
  const Status status = dispatchDataType(dtype, [&](auto typeTag) {
    using T = typename decltype(typeTag)::Type;
    isSupported = isSupportedConvertNumericType<T>;
    return Status::Ok();
  });
  return status.ok() && isSupported;
}

// ============================================================================
// NumericConverter - Single-direction type conversion
// ============================================================================

template <typename To, typename From>
struct NumericConverter {
  using ResultType = To;
  using SourceType = From;

  static inline ResultType convert(SourceType const &val) {
    return static_cast<ResultType>(val);
  }
};

// Specialization: f8E4M3FN -> f8E5M2
template <>
struct NumericConverter<__nv_fp8_e5m2, __nv_fp8_e4m3> {
  static inline __nv_fp8_e5m2 convert(__nv_fp8_e4m3 const &val) {
    return __nv_fp8_e5m2(static_cast<float>(val));
  }
};

// Specialization: f8E5M2 -> f8E4M3FN
template <>
struct NumericConverter<__nv_fp8_e4m3, __nv_fp8_e5m2> {
  static inline __nv_fp8_e4m3 convert(__nv_fp8_e5m2 const &val) {
    return __nv_fp8_e4m3(static_cast<float>(val));
  }
};

// Specialization: half -> float
template <>
struct NumericConverter<float, half> {
  static inline float convert(half const &val) { return __half2float(val); }
};

// Specialization: float -> half
template <>
struct NumericConverter<half, float> {
  static inline half convert(float const &val) { return __float2half(val); }
};

// Specialization: half -> double
template <>
struct NumericConverter<double, half> {
  static inline double convert(half const &val) {
    return static_cast<double>(__half2float(val));
  }
};

// Specialization: double -> half
template <>
struct NumericConverter<half, double> {
  static inline half convert(double const &val) {
    return __float2half(static_cast<float>(val));
  }
};

// Specialization: bfloat16 -> float
template <>
struct NumericConverter<float, nv_bfloat16> {
  static inline float convert(nv_bfloat16 const &val) {
    return static_cast<float>(val);
  }
};

// Specialization: float -> bfloat16
template <>
struct NumericConverter<nv_bfloat16, float> {
  static inline nv_bfloat16 convert(float const &val) {
    return nv_bfloat16(val);
  }
};

// Specialization: bfloat16 -> double
template <>
struct NumericConverter<double, nv_bfloat16> {
  static inline double convert(nv_bfloat16 const &val) {
    return static_cast<double>(static_cast<float>(val));
  }
};

// Specialization: double -> bfloat16
template <>
struct NumericConverter<nv_bfloat16, double> {
  static inline nv_bfloat16 convert(double const &val) {
    return nv_bfloat16(static_cast<float>(val));
  }
};

// Specialization: bfloat16 -> half
template <>
struct NumericConverter<half, nv_bfloat16> {
  static inline half convert(nv_bfloat16 const &val) {
    return __float2half(static_cast<float>(val));
  }
};

// Specialization: half -> bfloat16
template <>
struct NumericConverter<nv_bfloat16, half> {
  static inline nv_bfloat16 convert(half const &val) {
    return nv_bfloat16(__half2float(val));
  }
};

// Specialization: int32 -> half
template <>
struct NumericConverter<half, int32_t> {
  static inline half convert(int32_t const &val) {
    return __float2half(static_cast<float>(val));
  }
};

// Specialization: float -> int32
template <>
struct NumericConverter<int32_t, float> {
  static inline int32_t convert(float const &val) {
    // Reference conversion rounds finite in-range values toward zero; Inf and
    // NaN inputs are undefined.
    return static_cast<int32_t>(std::trunc(val));
  }
};

// Specialization: double -> int32
template <>
struct NumericConverter<int32_t, double> {
  static inline int32_t convert(double const &val) {
    // Reference conversion rounds finite in-range values toward zero; Inf and
    // NaN inputs are undefined.
    return static_cast<int32_t>(std::trunc(val));
  }
};

// Specialization: half -> int32
template <>
struct NumericConverter<int32_t, half> {
  static inline int32_t convert(half const &val) {
    return NumericConverter<int32_t, float>::convert(__half2float(val));
  }
};

// Specialization: int32 -> bfloat16
template <>
struct NumericConverter<nv_bfloat16, int32_t> {
  static inline nv_bfloat16 convert(int32_t const &val) {
    return nv_bfloat16(static_cast<float>(val));
  }
};

// Specialization: bfloat16 -> int32
template <>
struct NumericConverter<int32_t, nv_bfloat16> {
  static inline int32_t convert(nv_bfloat16 const &val) {
    return NumericConverter<int32_t, float>::convert(static_cast<float>(val));
  }
};

// Specialization: bool -> any type (false -> 0, true -> 1 via float)
template <typename To>
struct NumericConverter<To, bool> {
  static inline To convert(bool const &val) {
    return NumericConverter<To, float>::convert(val ? 1.0F : 0.0F);
  }
};

/// Compute type shared by pointwise and reduction reference kernels.
template <typename T>
struct ReferenceComputeType {
  using Type = T;
};

template <>
struct ReferenceComputeType<half> {
  using Type = float;
};

template <>
struct ReferenceComputeType<nv_bfloat16> {
  using Type = float;
};

template <>
struct ReferenceComputeType<__nv_fp8_e4m3> {
  using Type = float;
};

template <>
struct ReferenceComputeType<__nv_fp8_e5m2> {
  using Type = float;
};

template <>
struct ReferenceComputeType<int8_t> {
  using Type = int32_t;
};

template <>
struct ReferenceComputeType<uint8_t> {
  using Type = int32_t;
};

template <typename T>
using ReferenceComputeTypeT = typename ReferenceComputeType<T>::Type;

} // anonymous namespace

// Unified matrix multiplication kernel supporting both 2D and 3D (batched)
// cases
// - For 2D: pass batchSize=1, and strides should have 2 elements [mStride,
// kStride]
// - For 3D: pass actual batchSize, and strides should have 3 elements
// [batchStride, mStride, kStride] Supports independent Input/Compute/Output
// types for mixed-precision operations
template <typename InputT, typename ComputeT, typename OutputT>
static void
matmulKernel(const void *aPtr, const void *bPtr, void *cPtr, int64_t batchSize,
             int64_t M, int64_t K, int64_t N, double alpha, double beta,
             llvm::ArrayRef<int64_t> aStrides, llvm::ArrayRef<int64_t> bStrides,
             llvm::ArrayRef<int64_t> cStrides) {
  using InputToCompute = NumericConverter<ComputeT, InputT>;
  using ComputeToOutput = NumericConverter<OutputT, ComputeT>;
  using OutputToCompute = NumericConverter<ComputeT, OutputT>;

  const InputT *A = static_cast<const InputT *>(aPtr);
  const InputT *B = static_cast<const InputT *>(bPtr);
  OutputT *C = static_cast<OutputT *>(cPtr);

  // Determine if this is 2D or 3D based on stride vector size
  bool is2D = (aStrides.size() == 2);

  // For 2D: batch loop runs once, offsets are 0
  // For 3D: batch loop runs batchSize times, offsets use stride[0]
  int64_t aBatchStride = is2D ? 0 : aStrides[0];
  int64_t bBatchStride = is2D ? 0 : bStrides[0];
  int64_t cBatchStride = is2D ? 0 : cStrides[0];

  int64_t aMStride = is2D ? aStrides[0] : aStrides[1];
  int64_t aKStride = is2D ? aStrides[1] : aStrides[2];
  int64_t bKStride = is2D ? bStrides[0] : bStrides[1];
  int64_t bNStride = is2D ? bStrides[1] : bStrides[2];
  int64_t cMStride = is2D ? cStrides[0] : cStrides[1];
  int64_t cNStride = is2D ? cStrides[1] : cStrides[2];

  for (int64_t b = 0; b < batchSize; ++b) {
    int64_t aBatchOffset = b * aBatchStride;
    int64_t bBatchOffset = b * bBatchStride;
    int64_t cBatchOffset = b * cBatchStride;

    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        ComputeT sum = 0;
        for (int64_t k = 0; k < K; ++k) {
          int64_t aIdx = aBatchOffset + m * aMStride + k * aKStride;
          int64_t bIdx = bBatchOffset + k * bKStride + n * bNStride;
          ComputeT aVal = InputToCompute::convert(A[aIdx]);
          ComputeT bVal = InputToCompute::convert(B[bIdx]);
          sum += aVal * bVal;
        }

        int64_t cIdx = cBatchOffset + m * cMStride + n * cNStride;
        if (beta != 0.0) {
          ComputeT cVal = OutputToCompute::convert(C[cIdx]);
          C[cIdx] =
              ComputeToOutput::convert(static_cast<ComputeT>(alpha) * sum +
                                       static_cast<ComputeT>(beta) * cVal);
        } else {
          C[cIdx] =
              ComputeToOutput::convert(static_cast<ComputeT>(alpha) * sum);
        }
      }
    }
  }
}

// TODO: Optimize matmul kernels - consider launching on GPU

// ============================================================================
// MatmulNode Implementation
// ============================================================================

// ============================================================================
// MatmulNode Type Combination Registry - Single Source of Truth
// ============================================================================
// This registry defines all supported (input, output) type combinations and
// their corresponding kernel function pointers for MatmulNode operations.

namespace {

// Hash function for (DataType, DataType) pair
struct DataTypePairHash {
  std::size_t operator()(const std::pair<DataType, DataType> &p) const {
    return std::hash<int>()(static_cast<int>(p.first)) ^
           (std::hash<int>()(static_cast<int>(p.second)) << 1);
  }
};

} // anonymous namespace

// Type alias for matmul kernel function pointer
using MatmulKernelFunc = void (*)(const void *, const void *, void *, int64_t,
                                  int64_t, int64_t, int64_t, double, double,
                                  llvm::ArrayRef<int64_t>,
                                  llvm::ArrayRef<int64_t>,
                                  llvm::ArrayRef<int64_t>);

// Build the registry as a compile-time initialized map
// To add a new combination: add one entry here, and the rest is automatic
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif
static const std::unordered_map<std::pair<DataType, DataType>, MatmulKernelFunc,
                                DataTypePairHash>
    kMatmulKernelRegistry = {
        // {inputType, outputType} -> kernelFunction
        {{DataType::FLOAT32, DataType::FLOAT32},
         &matmulKernel<float, float, float>},
        {{DataType::FLOAT16, DataType::FLOAT16},
         &matmulKernel<half, half, half>},
        {{DataType::DOUBLE, DataType::DOUBLE},
         &matmulKernel<double, double, double>},
        {{DataType::FLOAT16, DataType::FLOAT32},
         &matmulKernel<half, float, float>},

        // Future combinations - uncomment to enable:
        // {{DataType::FLOAT32, DataType::FLOAT16},
        // &matmulKernel<float, float, half>},
        // {{DataType::INT8,    DataType::INT32}, &matmulKernel<int8_t,
        // int32_t, int32_t>},
};
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

// Check if an (inputType, outputType) combination is supported for matmul
bool MatmulNode::isSupportedTypeCombination(DataType inputType,
                                            DataType outputType) {
  return kMatmulKernelRegistry.find({inputType, outputType}) !=
         kMatmulKernelRegistry.end();
}

Status MatmulNode::validate() const {
  TIR_RETURN_IF_ERROR(validateTensors({{"A", a}, {"B", b}, {"C", c}}));

  // Check data types - A and B must have the same type, C can differ
  // This allows mixed-precision operations (e.g., F16 input -> F32 output)
  TIR_RETURN_IF_ERROR(validateSameType({{"A", a}, {"B", b}}));

  // Validate that the (inputType, outputType) combination is supported
  DataType inputType = a->getDataType();
  DataType outputType = c->getDataType();

  if (!isSupportedTypeCombination(inputType, outputType)) {
    return Status::InvalidArgument(
        "Unsupported input/output type combination for matmul: " +
        std::string(getEnumName(inputType)) + " -> " +
        std::string(getEnumName(outputType)) +
        ". "
        "Supported: F32 -> F32, F16 -> F16, F64 -> F64, F16 -> F32");
  }

  const auto &aDims = a->getDims();
  const auto &bDims = b->getDims();
  const auto &cDims = c->getDims();

  // Support 2D and 3D (batched) matmul
  if (aDims.size() != bDims.size() || aDims.size() != cDims.size()) {
    return Status::InvalidArgument(
        "All tensors must have the same number of dimensions");
  }

  // Unified dimension check for 2D and 3D matmul
  // 2D: A[M, K] @ B[K, N] -> C[M, N]
  // 3D: A[BS, M, K] @ B[BS, K, N] -> C[BS, M, N]
  if (aDims.size() != 2 && aDims.size() != 3) {
    return Status::InvalidArgument(
        "Only 2D and 3D tensors are supported for matmul");
  }

  const bool is2D = (aDims.size() == 2);

  const int64_t aBatchSize = is2D ? 1 : aDims[0];
  const int64_t aM = is2D ? aDims[0] : aDims[1];
  const int64_t aK = is2D ? aDims[1] : aDims[2];

  const int64_t bBatchSize = is2D ? 1 : bDims[0];
  const int64_t bK = is2D ? bDims[0] : bDims[1];
  const int64_t bN = is2D ? bDims[1] : bDims[2];

  const int64_t cBatchSize = is2D ? 1 : cDims[0];
  const int64_t cM = is2D ? cDims[0] : cDims[1];
  const int64_t cN = is2D ? cDims[1] : cDims[2];

  if (aBatchSize != bBatchSize) {
    return Status::InvalidArgument(
        "Batch size mismatch between A and B tensors");
  }

  if (aK != bK) {
    return Status::InvalidArgument(
        "Matmul dimension mismatch: A's K != B's K " + std::to_string(aK) +
        " != " + std::to_string(bK));
  }

  if (cBatchSize != aBatchSize) {
    return Status::InvalidArgument("Output batch size mismatch");
  }
  if (cM != aM || cN != bN) {
    return Status::InvalidArgument("Output dimension mismatch for matmul");
  }

  return Status::Ok();
}

Status MatmulNode::computeValidated() {
  const auto &aDims = a->getDims();
  const auto &aStrides = a->getDesc().strides;
  const auto &bStrides = b->getDesc().strides;
  const auto &cStrides = c->getDesc().strides;

  // Use hostPtr() for all tensors (reference computation)
  // alpha = 1.0, beta = 0.0 (simple C = A @ B)
  const double alpha = 1.0;
  const double beta = 0.0;

  DataType inputType = a->getDataType();
  DataType outputType = c->getDataType();

  // Determine batch size and matrix dimensions
  const bool is2D = (aDims.size() == 2);
  const int64_t batchSize = is2D ? 1 : aDims[0];
  const int64_t M = is2D ? aDims[0] : aDims[1];
  const int64_t K = is2D ? aDims[1] : aDims[2];
  const int64_t N = is2D ? b->getDims()[1] : b->getDims()[2];

  auto it = kMatmulKernelRegistry.find({inputType, outputType});

  // Sanity check: should always find a kernel if validate() passed
  if (it == kMatmulKernelRegistry.end()) {
    return Status::InvalidArgument(
        "Unsupported input/output type combination for matmul");
  }

  MatmulKernelFunc kernelFunc = it->second;

  // Invoke the selected kernel once with all parameters
  kernelFunc(a->hostPtr(), b->hostPtr(), c->hostPtr(), batchSize, M, K, N,
             alpha, beta, aStrides, bStrides, cStrides);

  return Status::Ok();
}

// ============================================================================
// ConcatenateNode Implementation
// ============================================================================

namespace {

static std::vector<int64_t> unravelIndex(int64_t linearIndex,
                                         llvm::ArrayRef<int64_t> dims) {
  std::vector<int64_t> coord(dims.size(), 0);
  int64_t remaining = linearIndex;
  for (size_t reverseIdx = dims.size(); reverseIdx > 0; --reverseIdx) {
    size_t dim = reverseIdx - 1;
    coord[dim] = remaining % dims[dim];
    remaining /= dims[dim];
  }
  return coord;
}

} // anonymous namespace

static bool hasCompactRowMajorStrides(llvm::ArrayRef<int64_t> dims,
                                      llvm::ArrayRef<int64_t> strides) {
  const size_t rank = dims.size();
  if (strides.size() != rank) {
    return false;
  }

  int64_t expectedStride = 1;
  for (size_t reverseIdx = rank; reverseIdx > 0; --reverseIdx) {
    const size_t dim = reverseIdx - 1;
    if (strides[dim] != expectedStride) {
      return false;
    }
    expectedStride *= dims[dim];
  }
  return true;
}

template <typename T>
void fillStridedReferenceTensor(SimplifiedTensor &tensor, T value) {
  const auto &dims = tensor.getDims();
  const auto &strides = tensor.getDesc().strides;
  const int64_t totalElements = tensor.getTotalElements();
  auto *output = static_cast<T *>(tensor.hostPtr());

  if (hasCompactRowMajorStrides(dims, strides)) {
    std::fill_n(output, totalElements, value);
    return;
  }

  for (int64_t linearIndex = 0; linearIndex < totalElements; ++linearIndex) {
    tensor.hostElement<T>(linearIndex) = value;
  }
}

Status ConstantNode::validate() const {
  TIR_RETURN_IF_ERROR(validateTensors({{"Output", outputTensor}}));

  if (!isSupportedConstantDataType(outputTensor->getDataType())) {
    return unsupportedDataType(outputTensor->getDataType(), "constant");
  }

  return Status::Ok();
}

Status ConstantNode::computeValidated() {
  const DataType dtype = outputTensor->getDataType();
  if (!isSupportedConstantDataType(dtype)) {
    return unsupportedDataType(dtype, "constant");
  }

  return dispatchDataType(dtype, [&](auto typeTag) {
    using T = typename decltype(typeTag)::Type;
    if constexpr (std::is_integral_v<T>) {
      StatusOr<T> intValue = detail::convertFiniteConstantToInteger<T>(
          scalarValue, getEnumName(dtype));
      if (!intValue.ok()) {
        return intValue.status();
      }
      fillStridedReferenceTensor(*outputTensor, *intValue);
      return Status::Ok();
    } else {
      fillStridedReferenceTensor(
          *outputTensor, NumericConverter<T, double>::convert(scalarValue));
      return Status::Ok();
    }
  });
}

Status SplatNode::validate() const {
  TIR_RETURN_IF_ERROR(
      validateTensors({{"Input", inputTensor}, {"Output", outputTensor}}));

  if (inputTensor->getTotalElements() != 1) {
    return Status::InvalidArgument(
        "Splat input tensor must contain exactly one element");
  }
  TIR_RETURN_IF_ERROR(
      validateSameType({{"Input", inputTensor}, {"Output", outputTensor}}));
  if (getDataTypeSize(inputTensor->getDataType()) == 0) {
    return unsupportedDataType(inputTensor->getDataType(), "splat");
  }

  return Status::Ok();
}

Status SplatNode::computeValidated() {
  const size_t elementSize = getDataTypeSize(outputTensor->getDataType());
  const int64_t totalElements = outputTensor->getTotalElements();
  const auto *inputPtr = static_cast<const char *>(inputTensor->hostPtr());
  auto *outputPtr = static_cast<char *>(outputTensor->hostPtr());
  const char *source = inputPtr + inputTensor->getStorageIndex(0) * elementSize;

  for (int64_t linearIndex = 0; linearIndex < totalElements; ++linearIndex) {
    const int64_t destOffset = outputTensor->getStorageIndex(linearIndex);
    std::memcpy(outputPtr + destOffset * elementSize, source, elementSize);
  }

  return Status::Ok();
}

// ============================================================================
// IotaNode Implementation
// ============================================================================

namespace {

template <typename T>
static void fillIotaReferenceTensor(SimplifiedTensor &output,
                                    int64_t iotaDimension) {
  const auto &outputDims = output.getDims();
  const int64_t totalElements = output.getTotalElements();
  using Int32ToOutput = NumericConverter<T, int32_t>;

  for (int64_t linearIndex = 0; linearIndex < totalElements; ++linearIndex) {
    std::vector<int64_t> outputCoord = unravelIndex(linearIndex, outputDims);
    output.hostElement<T>(linearIndex) = Int32ToOutput::convert(
        static_cast<int32_t>(outputCoord[iotaDimension]));
  }
}

} // anonymous namespace

Status IotaNode::validate() const {
  TIR_RETURN_IF_ERROR(validateTensors({{"Output", outputTensor}}));

  const auto rank = static_cast<int64_t>(outputTensor->getDims().size());
  if (iotaDimension < 0 || iotaDimension >= rank) {
    return Status::InvalidArgument(
        "Iota dimension " + std::to_string(iotaDimension) +
        " is out of bounds for tensor rank " + std::to_string(rank));
  }

  if (!isSupportedIotaDataType(outputTensor->getDataType())) {
    return unsupportedDataType(outputTensor->getDataType(), "iota");
  }
  return Status::Ok();
}

Status IotaNode::computeValidated() {
  const DataType dtype = outputTensor->getDataType();
  if (!isSupportedIotaDataType(dtype)) {
    return unsupportedDataType(dtype, "iota");
  }

  return dispatchDataType(dtype, [&](auto typeTag) {
    using T = typename decltype(typeTag)::Type;
    fillIotaReferenceTensor<T>(*outputTensor, iotaDimension);
    return Status::Ok();
  });
}

Status ConcatenateNode::validate() const {
  if (inputs.empty()) {
    return Status::InvalidArgument(
        "Concatenate operation requires at least one input tensor");
  }

  const size_t numInputs = inputs.size();
  NamedTensors tensors{{"Output", output}};
  for (size_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    tensors.emplace_back("Input " + std::to_string(inputIdx), inputs[inputIdx]);
  }
  TIR_RETURN_IF_ERROR(validateTensors(tensors));

  const auto &firstDims = inputs.front()->getDims();
  const size_t rank = firstDims.size();
  if (concatDim < 0 || concatDim >= static_cast<int64_t>(rank)) {
    return Status::InvalidArgument(
        "Concatenation dimension " + std::to_string(concatDim) +
        " is out of bounds for tensor rank " + std::to_string(rank));
  }

  const DataType dtype = inputs.front()->getDataType();
  const auto &outputDims = output->getDims();
  if (output->getDataType() != dtype) {
    return Status::InvalidArgument(
        "Output tensor data type must match input tensor data type");
  }
  if (outputDims.size() != rank) {
    return Status::InvalidArgument(
        "Output tensor rank must match input tensor rank");
  }

  if (getDataTypeSize(dtype) == 0) {
    return unsupportedDataType(dtype, "concatenate");
  }

  int64_t totalConcatSize = 0;
  for (size_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    const std::string role = "Input " + std::to_string(inputIdx);
    const auto &inputDims = inputs[inputIdx]->getDims();
    if (inputDims.size() != rank) {
      return Status::InvalidArgument(
          role + " tensor rank must match first input rank");
    }
    if (inputs[inputIdx]->getDataType() != dtype) {
      return Status::InvalidArgument(
          role + " tensor data type must match first input tensor data type");
    }

    for (size_t dim = 0; dim < rank; ++dim) {
      if (static_cast<int64_t>(dim) == concatDim) {
        if (inputDims[dim] >
            std::numeric_limits<int64_t>::max() - totalConcatSize) {
          return Status::InvalidArgument(
              "Concatenate dimension size overflows 64-bit integer range");
        }
        totalConcatSize += inputDims[dim];
      } else if (inputDims[dim] != firstDims[dim]) {
        return Status::InvalidArgument(role + " non-concatenate dimension " +
                                       std::to_string(dim) +
                                       " must match first input tensor");
      }
    }
  }

  for (size_t dim = 0; dim < rank; ++dim) {
    if (static_cast<int64_t>(dim) == concatDim) {
      if (outputDims[dim] != totalConcatSize) {
        return Status::InvalidArgument(
            "Output concatenate dimension must equal sum of input dimensions");
      }
    } else if (outputDims[dim] != firstDims[dim]) {
      return Status::InvalidArgument(
          "Output non-concatenate dimensions must match input tensors");
    }
  }

  return Status::Ok();
}

Status ConcatenateNode::computeValidated() {
  const size_t elementSize = getDataTypeSize(output->getDataType());
  auto *outputPtr = static_cast<char *>(output->hostPtr());

  int64_t concatOffset = 0;
  for (const auto &input : inputs) {
    const auto &inputDims = input->getDims();
    const auto *inputPtr = static_cast<const char *>(input->hostPtr());

    for (int64_t linearIndex = 0; linearIndex < input->getTotalElements();
         ++linearIndex) {
      std::vector<int64_t> inputCoord = unravelIndex(linearIndex, inputDims);
      std::vector<int64_t> outputCoord = inputCoord;
      outputCoord[concatDim] += concatOffset;

      int64_t sourceOffset = input->getStorageIndex(inputCoord);
      int64_t destOffset = output->getStorageIndex(outputCoord);
      std::memcpy(outputPtr + destOffset * elementSize,
                  inputPtr + sourceOffset * elementSize, elementSize);
    }

    concatOffset += inputDims[concatDim];
  }

  return Status::Ok();
}

Status SliceNode::validate() const {
  TIR_RETURN_IF_ERROR(validateTensors({{"Input", input}, {"Output", output}}));

  TIR_RETURN_IF_ERROR(validateSameType({{"Input", input}, {"Output", output}}));
  if (getDataTypeSize(input->getDataType()) == 0) {
    return unsupportedDataType(input->getDataType(), "slice");
  }

  const auto &inputDims = input->getDims();
  const auto &outputDims = output->getDims();
  const size_t rank = inputDims.size();
  if (outputDims.size() != rank || starts.size() != rank ||
      limits.size() != rank || sliceStrides.size() != rank) {
    return Status::InvalidArgument(
        "Slice input, output, starts, limits, and strides must have the same "
        "rank");
  }

  for (size_t dim = 0; dim < rank; ++dim) {
    if (starts[dim] < 0 || limits[dim] < 0) {
      return Status::InvalidArgument(
          "Slice starts and limits must be non-negative");
    }
    if (sliceStrides[dim] <= 0) {
      return Status::InvalidArgument("Slice strides must be positive");
    }
    if (starts[dim] > limits[dim]) {
      return Status::InvalidArgument("Slice start must not exceed limit");
    }
    if (limits[dim] > inputDims[dim]) {
      return Status::InvalidArgument(
          "Slice limit exceeds input tensor dimension");
    }

    int64_t expectedExtent = 0;
    if (limits[dim] > starts[dim]) {
      expectedExtent = 1 + (limits[dim] - 1 - starts[dim]) / sliceStrides[dim];
    }
    if (outputDims[dim] != expectedExtent) {
      return Status::InvalidArgument(
          "Slice output dimension does not match starts/limits/strides");
    }
  }

  return Status::Ok();
}

Status SliceNode::computeValidated() {
  const size_t elementSize = getDataTypeSize(output->getDataType());
  const auto &outputDims = output->getDims();
  const auto *inputPtr = static_cast<const char *>(input->hostPtr());
  auto *outputPtr = static_cast<char *>(output->hostPtr());

  for (int64_t linearIndex = 0; linearIndex < output->getTotalElements();
       ++linearIndex) {
    std::vector<int64_t> outputCoord = unravelIndex(linearIndex, outputDims);
    std::vector<int64_t> inputCoord(outputCoord.size(), 0);
    for (size_t dim = 0; dim < outputCoord.size(); ++dim) {
      inputCoord[dim] = starts[dim] + outputCoord[dim] * sliceStrides[dim];
    }

    int64_t sourceOffset = input->getStorageIndex(inputCoord);
    int64_t destOffset = output->getStorageIndex(outputCoord);
    std::memcpy(outputPtr + destOffset * elementSize,
                inputPtr + sourceOffset * elementSize, elementSize);
  }

  return Status::Ok();
}

Status ReshapeNode::validate() const {
  TIR_RETURN_IF_ERROR(validateTensors({{"Input", input}, {"Output", output}}));

  TIR_RETURN_IF_ERROR(validateSameType({{"Input", input}, {"Output", output}}));

  if (getDataTypeSize(input->getDataType()) == 0) {
    return unsupportedDataType(input->getDataType(), "reshape");
  }

  if (input->getTotalElements() != output->getTotalElements()) {
    return Status::InvalidArgument(
        "Input and output tensors must have the same number of elements for "
        "reshape");
  }

  return Status::Ok();
}

Status ReshapeNode::computeValidated() {
  const size_t elementSize = getDataTypeSize(output->getDataType());
  const auto *inputPtr = static_cast<const char *>(input->hostPtr());
  auto *outputPtr = static_cast<char *>(output->hostPtr());

  for (int64_t linearIndex = 0; linearIndex < output->getTotalElements();
       ++linearIndex) {
    const int64_t sourceOffset = input->getStorageIndex(linearIndex);
    const int64_t destOffset = output->getStorageIndex(linearIndex);
    std::memcpy(outputPtr + destOffset * elementSize,
                inputPtr + sourceOffset * elementSize, elementSize);
  }

  return Status::Ok();
}

// ============================================================================
// BroadcastNode Implementation
// ============================================================================

Status BroadcastNode::validate() const {
  TIR_RETURN_IF_ERROR(validateTensors({{"Input", input}, {"Output", output}}));

  TIR_RETURN_IF_ERROR(validateSameType({{"Input", input}, {"Output", output}}));
  if (getDataTypeSize(input->getDataType()) == 0) {
    return unsupportedDataType(input->getDataType(), "broadcast");
  }

  const auto &inputDims = input->getDims();
  const auto &outputDims = output->getDims();
  if (inputDims.size() != outputDims.size()) {
    return Status::InvalidArgument(
        "Broadcast input and output tensors must have the same rank");
  }

  bool sawBroadcastDim = false;
  for (size_t dim = 0; dim < inputDims.size(); ++dim) {
    if (inputDims[dim] == outputDims[dim]) {
      continue;
    }
    if (inputDims[dim] != 1 || outputDims[dim] <= 1) {
      return Status::InvalidArgument(
          "Broadcast dimensions must expand from input size 1");
    }
    sawBroadcastDim = true;
  }

  if (!sawBroadcastDim) {
    return Status::InvalidArgument(
        "Broadcast operation must expand at least one dimension");
  }

  return Status::Ok();
}

Status BroadcastNode::computeValidated() {
  const size_t elementSize = getDataTypeSize(output->getDataType());
  const auto &inputDims = input->getDims();
  const auto &outputDims = output->getDims();
  const auto *inputPtr = static_cast<const char *>(input->hostPtr());
  auto *outputPtr = static_cast<char *>(output->hostPtr());

  for (int64_t linearIndex = 0; linearIndex < output->getTotalElements();
       ++linearIndex) {
    std::vector<int64_t> outputCoord = unravelIndex(linearIndex, outputDims);
    std::vector<int64_t> inputCoord(outputCoord.size(), 0);
    for (size_t dim = 0; dim < outputCoord.size(); ++dim) {
      inputCoord[dim] = inputDims[dim] == 1 ? 0 : outputCoord[dim];
    }

    int64_t sourceOffset = input->getStorageIndex(inputCoord);
    int64_t destOffset = output->getStorageIndex(outputCoord);
    std::memcpy(outputPtr + destOffset * elementSize,
                inputPtr + sourceOffset * elementSize, elementSize);
  }

  return Status::Ok();
}

// ============================================================================
// TransposeNode Implementation
// ============================================================================

Status TransposeNode::validate() const {
  TIR_RETURN_IF_ERROR(validateTensors({{"Input", input}, {"Output", output}}));

  TIR_RETURN_IF_ERROR(validateSameType({{"Input", input}, {"Output", output}}));

  if (getDataTypeSize(input->getDataType()) == 0) {
    return unsupportedDataType(input->getDataType(), "transpose");
  }

  const auto &inputDims = input->getDims();
  const auto &outputDims = output->getDims();
  const size_t rank = inputDims.size();
  if (outputDims.size() != rank || permutation.size() != rank) {
    return Status::InvalidArgument(
        "Transpose input, output, and permutation must have the same rank");
  }

  std::vector<bool> used(rank, false);
  for (size_t outputDim = 0; outputDim < rank; ++outputDim) {
    const int64_t inputDim = permutation[outputDim];
    if (inputDim < 0 || inputDim >= static_cast<int64_t>(rank)) {
      return Status::InvalidArgument(
          "Transpose permutation dimension is out of bounds");
    }
    if (used[inputDim]) {
      return Status::InvalidArgument(
          "Transpose permutation contains a duplicate dimension");
    }
    used[inputDim] = true;

    if (outputDims[outputDim] != inputDims[inputDim]) {
      return Status::InvalidArgument(
          "Transpose output shape does not match permuted input shape");
    }
  }

  return Status::Ok();
}

Status TransposeNode::computeValidated() {
  const size_t elementSize = getDataTypeSize(output->getDataType());
  const auto &outputDims = output->getDims();
  const auto *inputPtr = static_cast<const char *>(input->hostPtr());
  auto *outputPtr = static_cast<char *>(output->hostPtr());

  for (int64_t linearIndex = 0; linearIndex < output->getTotalElements();
       ++linearIndex) {
    std::vector<int64_t> outputCoord = unravelIndex(linearIndex, outputDims);
    std::vector<int64_t> inputCoord(outputCoord.size(), 0);
    for (size_t outputDim = 0; outputDim < outputCoord.size(); ++outputDim) {
      inputCoord[permutation[outputDim]] = outputCoord[outputDim];
    }

    int64_t sourceOffset = input->getStorageIndex(inputCoord);
    int64_t destOffset = output->getStorageIndex(outputCoord);
    std::memcpy(outputPtr + destOffset * elementSize,
                inputPtr + sourceOffset * elementSize, elementSize);
  }

  return Status::Ok();
}

// ============================================================================
// ReduceNode Implementation
// ============================================================================

namespace {

template <typename ComputeT>
static ComputeT getReductionIdentity(ReductionMode mode) {
  switch (mode) {
  case ReductionMode::add:
  case ReductionMode::amax:
  case ReductionMode::avg:
  case ReductionMode::norm1:
  case ReductionMode::norm2:
    return static_cast<ComputeT>(0);
  case ReductionMode::mul:
  case ReductionMode::mul_no_zeros:
    return static_cast<ComputeT>(1);
  case ReductionMode::min:
    return std::numeric_limits<ComputeT>::max();
  case ReductionMode::max:
    return std::numeric_limits<ComputeT>::lowest();
  case ReductionMode::customize:
    return ComputeT{};
  }
  return ComputeT{};
}

template <typename ComputeT>
static ComputeT absoluteReductionValue(ComputeT value) {
  if constexpr (std::is_floating_point_v<ComputeT> ||
                std::is_signed_v<ComputeT>) {
    return std::abs(value);
  }
  return value;
}

template <typename ComputeT>
static void accumulateReductionValue(ReductionMode mode, ComputeT value,
                                     ComputeT &accumulator) {
  switch (mode) {
  case ReductionMode::add:
  case ReductionMode::avg:
    accumulator += value;
    return;
  case ReductionMode::mul:
    accumulator *= value;
    return;
  case ReductionMode::min:
    accumulator = std::min(accumulator, value);
    return;
  case ReductionMode::max:
    accumulator = std::max(accumulator, value);
    return;
  case ReductionMode::amax:
    accumulator =
        std::max(accumulator, absoluteReductionValue<ComputeT>(value));
    return;
  case ReductionMode::norm1:
    accumulator += absoluteReductionValue<ComputeT>(value);
    return;
  case ReductionMode::norm2:
    accumulator += value * value;
    return;
  case ReductionMode::mul_no_zeros:
    accumulator *=
        value == static_cast<ComputeT>(0) ? static_cast<ComputeT>(1) : value;
    return;
  case ReductionMode::customize:
    return;
  }
}

template <typename ComputeT>
static ComputeT finalizeReductionValue(ReductionMode mode, ComputeT accumulator,
                                       int64_t reductionSize) {
  switch (mode) {
  case ReductionMode::avg:
    return accumulator / static_cast<ComputeT>(reductionSize);
  case ReductionMode::norm2:
    return std::sqrt(accumulator);
  default:
    return accumulator;
  }
}

template <typename T>
static void
reduceKernel(const SimplifiedTensor &input, SimplifiedTensor &output,
             llvm::ArrayRef<int64_t> dimensions, ReductionMode mode) {
  using ComputeT = ReferenceComputeTypeT<T>;

  const auto inputDims = input.getDims();
  const auto outputDims = output.getDims();

  int64_t reductionSize = 1;
  for (int64_t dimension : dimensions) {
    reductionSize *= inputDims[dimension];
  }

  // Reduce each output element over its input slice; a scalar accumulator
  // avoids any intermediate buffer.
  for (int64_t outputIndex = 0; outputIndex < output.getTotalElements();
       ++outputIndex) {

    std::vector<int64_t> coord = unravelIndex(outputIndex, outputDims);

    ComputeT accumulator = getReductionIdentity<ComputeT>(mode);
    for (int64_t reducedIndex = 0; reducedIndex < reductionSize;
         ++reducedIndex) {
      // Spread the flat sweep index across the reduced axes.
      int64_t remaining = reducedIndex;
      for (size_t reverseIdx = dimensions.size(); reverseIdx > 0;
           --reverseIdx) {
        const int64_t dimension = dimensions[reverseIdx - 1];
        coord[dimension] = remaining % inputDims[dimension];
        remaining /= inputDims[dimension];
      }
      const ComputeT value =
          NumericConverter<ComputeT, T>::convert(input.hostElement<T>(coord));
      accumulateReductionValue(mode, value, accumulator);
    }

    output.hostElement<T>(outputIndex) = NumericConverter<T, ComputeT>::convert(
        finalizeReductionValue(mode, accumulator, reductionSize));
  }
}

} // anonymous namespace

Status ReduceNode::validate() const {
  TIR_RETURN_IF_ERROR(validateTensors({{"Input", input}, {"Output", output}}));
  TIR_RETURN_IF_ERROR(validateSameType({{"Input", input}, {"Output", output}}));

  if (mode == ReductionMode::customize) {
    return Status::NotSupported(
        "ReduceNode does not support customize mode; use reduce_ud");
  }

  const DataType dtype = input->getDataType();
  if (dtype == DataType::BOOL ||
      (!isFloatDataType(dtype) && !isIntegerDataType(dtype))) {
    return unsupportedDataType(dtype, "reduce");
  }
  if (mode == ReductionMode::norm2 && !isFloatDataType(dtype)) {
    return Status::NotSupported(
        "norm2 reduction requires a floating-point data type");
  }

  const auto inputDims = input->getDims();
  const auto outputDims = output->getDims();
  const size_t rank = inputDims.size();
  if (outputDims.size() != rank) {
    return Status::InvalidArgument(
        "Reduce input and output tensors must have the same rank");
  }
  if (dimensions.empty()) {
    return Status::InvalidArgument(
        "Reduce requires at least one reduction dimension");
  }

  std::vector<bool> isReducedDimension(rank, false);
  for (int64_t dimension : dimensions) {
    if (dimension < 0 || dimension >= static_cast<int64_t>(rank)) {
      return Status::InvalidArgument(
          "Reduce dimension " + std::to_string(dimension) +
          " is out of bounds for rank " + std::to_string(rank));
    }
    if (isReducedDimension[dimension]) {
      return Status::InvalidArgument(
          "Reduce dimension " + std::to_string(dimension) + " is duplicated");
    }
    isReducedDimension[dimension] = true;
  }

  for (size_t dimension = 0; dimension < rank; ++dimension) {
    const int64_t expected =
        isReducedDimension[dimension] ? 1 : inputDims[dimension];
    if (outputDims[dimension] != expected) {
      return Status::InvalidArgument(
          "Reduce output shape must retain reduced dimensions with extent one");
    }
  }

  return Status::Ok();
}

Status ReduceNode::computeValidated() {
  const DataType dtype = input->getDataType();
  if (dtype == DataType::BOOL ||
      (!isFloatDataType(dtype) && !isIntegerDataType(dtype))) {
    return unsupportedDataType(dtype, "reduce");
  }

  return dispatchDataType(dtype, [&](auto typeTag) {
    using T = typename decltype(typeTag)::Type;
    if constexpr (!std::is_same_v<T, bool>) {
      reduceKernel<T>(*input, *output, dimensions, mode);
      return Status::Ok();
    }
    return unsupportedDataType(dtype, "reduce");
  });
}

// ============================================================================
// ConvertNode Implementation
// ============================================================================

template <typename InputT, typename OutputT>
static void convertKernel(const SimplifiedTensor &input,
                          SimplifiedTensor &output) {
  for (int64_t linearIndex = 0; linearIndex < output.getTotalElements();
       ++linearIndex) {
    output.hostElement<OutputT>(linearIndex) =
        NumericConverter<OutputT, InputT>::convert(
            input.hostElement<InputT>(linearIndex));
  }
}

bool ConvertNode::isSupportedTypeCombination(DataType inputType,
                                             DataType outputType) {
  if (inputType == DataType::BOOL) {
    return outputType == DataType::BOOL ||
           isSupportedConvertNumericDataType(outputType);
  }
  return isSupportedConvertNumericDataType(inputType) &&
         isSupportedConvertNumericDataType(outputType);
}

Status ConvertNode::validate() const {
  TIR_RETURN_IF_ERROR(validateTensors({{"Input", input}, {"Output", output}}));

  TIR_RETURN_IF_ERROR(
      validateSameShape({{"Input", input}, {"Output", output}}));

  DataType inputType = input->getDataType();
  DataType outputType = output->getDataType();
  if (!isSupportedTypeCombination(inputType, outputType)) {
    return Status::InvalidArgument(
        "Unsupported input/output type combination for convert: " +
        std::string(getEnumName(inputType)) + " -> " +
        std::string(getEnumName(outputType)));
  }

  return Status::Ok();
}

Status ConvertNode::computeValidated() {
  const DataType inputType = input->getDataType();
  const DataType outputType = output->getDataType();
  if (!isSupportedTypeCombination(inputType, outputType)) {
    return Status::InvalidArgument(
        "Unsupported input/output type combination for convert");
  }

  return dispatchDataType(inputType, [&](auto inputTypeTag) {
    using InputT = typename decltype(inputTypeTag)::Type;
    return dispatchDataType(outputType, [&](auto outputTypeTag) {
      using OutputT = typename decltype(outputTypeTag)::Type;
      constexpr bool isBoolInput = std::is_same_v<InputT, bool>;
      constexpr bool isBoolOutput = std::is_same_v<OutputT, bool>;
      constexpr bool inputIsNumeric = isSupportedConvertNumericType<InputT>;
      constexpr bool outputIsNumeric = isSupportedConvertNumericType<OutputT>;
      if constexpr ((isBoolInput && (isBoolOutput || outputIsNumeric)) ||
                    (inputIsNumeric && outputIsNumeric)) {
        convertKernel<InputT, OutputT>(*input, *output);
        return Status::Ok();
      }
      return Status::InvalidArgument(
          "Unsupported input/output type combination for convert");
    });
  });
}

// ============================================================================
// UnaryPointwiseNode Implementation
// ============================================================================

const char *getEnumName(UnaryPointwiseMode mode) {
  switch (mode) {
  case UnaryPointwiseMode::ABS:
    return "ABS";
  case UnaryPointwiseMode::CEIL:
    return "CEIL";
  case UnaryPointwiseMode::COS:
    return "COS";
  case UnaryPointwiseMode::ERF:
    return "ERF";
  case UnaryPointwiseMode::EXP:
    return "EXP";
  case UnaryPointwiseMode::FLOOR:
    return "FLOOR";
  case UnaryPointwiseMode::GELU_FWD:
    return "GELU_FWD";
  case UnaryPointwiseMode::LOG:
    return "LOG";
  case UnaryPointwiseMode::NEG:
    return "NEG";
  case UnaryPointwiseMode::RELU_FWD:
    return "RELU_FWD";
  case UnaryPointwiseMode::SIN:
    return "SIN";
  case UnaryPointwiseMode::SQRT:
    return "SQRT";
  case UnaryPointwiseMode::RSQRT:
    return "RSQRT";
  case UnaryPointwiseMode::TAN:
    return "TAN";
  case UnaryPointwiseMode::TANH_FWD:
    return "TANH_FWD";
  case UnaryPointwiseMode::RECIPROCAL:
    return "RECIPROCAL";
  case UnaryPointwiseMode::SIGMOID:
    return "SIGMOID";
  case UnaryPointwiseMode::GELU_APPROX_TANH:
    return "GELU_APPROX_TANH";
  case UnaryPointwiseMode::LOGICAL_NOT:
    return "LOGICAL_NOT";
  default:
    return "UNKNOWN";
  }
}

const char *getEnumName(ParametricUnaryPointwiseMode mode) {
  switch (mode) {
  case ParametricUnaryPointwiseMode::SOFTPLUS_FWD:
    return "SOFTPLUS_FWD";
  case ParametricUnaryPointwiseMode::SWISH_FWD:
    return "SWISH_FWD";
  case ParametricUnaryPointwiseMode::ELU_FWD:
    return "ELU_FWD";
  default:
    return "UNKNOWN";
  }
}

namespace {

template <typename T>
static T evaluateGeluApproxTanh(T value) {
  constexpr double kSqrtTwoOverPi = 0.7978845608028654;
  constexpr double kGeluTanhCoeff = 0.044715;
  const T xCubed = value * value * value;
  const T tanhInput = static_cast<T>(kSqrtTwoOverPi) *
                      (value + static_cast<T>(kGeluTanhCoeff) * xCubed);
  return static_cast<T>(0.5) * value *
         (static_cast<T>(1) + std::tanh(tanhInput));
}

template <typename T>
static T evaluateGeluApproxTanhBwd(T input, T gradient) {
  constexpr double kSqrtTwoOverPi = 0.7978845608028654;
  constexpr double kGeluTanhCoeff = 0.044715;
  const T xSquared = input * input;
  const T xCubed = xSquared * input;
  const T tanhInput = static_cast<T>(kSqrtTwoOverPi) *
                      (input + static_cast<T>(kGeluTanhCoeff) * xCubed);
  const T tanhValue = std::tanh(tanhInput);
  const T tanhDerivative = static_cast<T>(1) - tanhValue * tanhValue;
  const T threeTimesCoeff = static_cast<T>(3) * static_cast<T>(kGeluTanhCoeff);
  const T tanhInputDerivative =
      static_cast<T>(kSqrtTwoOverPi) *
      (static_cast<T>(1) + threeTimesCoeff * xSquared);
  const T onePlusTanh = static_cast<T>(1) + tanhValue;
  const T tanhDerivativeProduct = tanhDerivative * tanhInputDerivative;
  const T inputDerivativeProduct = input * tanhDerivativeProduct;
  const T geluDerivative =
      static_cast<T>(0.5) * (onePlusTanh + inputDerivativeProduct);
  return gradient * geluDerivative;
}

static bool isSupportedUnaryPointwiseMode(DataType dtype,
                                          UnaryPointwiseMode mode) {
  switch (dtype) {
  case DataType::FLOAT32:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::DOUBLE:
    switch (mode) {
    case UnaryPointwiseMode::ABS:
    case UnaryPointwiseMode::CEIL:
    case UnaryPointwiseMode::COS:
    case UnaryPointwiseMode::ERF:
    case UnaryPointwiseMode::EXP:
    case UnaryPointwiseMode::FLOOR:
    case UnaryPointwiseMode::GELU_FWD:
    case UnaryPointwiseMode::LOG:
    case UnaryPointwiseMode::NEG:
    case UnaryPointwiseMode::RELU_FWD:
    case UnaryPointwiseMode::SIN:
    case UnaryPointwiseMode::SQRT:
    case UnaryPointwiseMode::RSQRT:
    case UnaryPointwiseMode::TAN:
    case UnaryPointwiseMode::TANH_FWD:
    case UnaryPointwiseMode::RECIPROCAL:
    case UnaryPointwiseMode::SIGMOID:
    case UnaryPointwiseMode::GELU_APPROX_TANH:
      return true;
    case UnaryPointwiseMode::LOGICAL_NOT:
      return false;
    }
    return false;
  case DataType::INT32:
  case DataType::INT8:
    return mode == UnaryPointwiseMode::ABS;
  case DataType::BOOL:
    return mode == UnaryPointwiseMode::LOGICAL_NOT;
  default:
    return false;
  }
}

template <typename T>
static T evaluateUnaryPointwise(UnaryPointwiseMode mode, T value) {
  if constexpr (std::is_same_v<T, bool>) {
    switch (mode) {
    case UnaryPointwiseMode::LOGICAL_NOT:
      return !value;
    default:
      return false;
    }
  } else if constexpr (std::is_integral_v<T>) {
    switch (mode) {
    case UnaryPointwiseMode::ABS:
      return std::abs(value);
    default:
      return T{};
    }
  } else {
    switch (mode) {
    case UnaryPointwiseMode::ABS:
      return std::abs(value);
    case UnaryPointwiseMode::CEIL:
      return std::ceil(value);
    case UnaryPointwiseMode::COS:
      return std::cos(value);
    case UnaryPointwiseMode::ERF:
      return std::erf(value);
    case UnaryPointwiseMode::EXP:
      return std::exp(value);
    case UnaryPointwiseMode::FLOOR:
      return std::floor(value);
    case UnaryPointwiseMode::GELU_FWD:
      return static_cast<T>(0.5) * value *
             (static_cast<T>(1) +
              std::erf(value / std::sqrt(static_cast<T>(2))));
    case UnaryPointwiseMode::LOG:
      return std::log(value);
    case UnaryPointwiseMode::NEG:
      return -value;
    case UnaryPointwiseMode::RELU_FWD:
      return std::max(value, static_cast<T>(0));
    case UnaryPointwiseMode::SIN:
      return std::sin(value);
    case UnaryPointwiseMode::SQRT:
      return std::sqrt(value);
    case UnaryPointwiseMode::RSQRT:
      return static_cast<T>(1) / std::sqrt(value);
    case UnaryPointwiseMode::TAN:
      return std::tan(value);
    case UnaryPointwiseMode::TANH_FWD:
      return std::tanh(value);
    case UnaryPointwiseMode::RECIPROCAL:
      return static_cast<T>(1) / value;
    case UnaryPointwiseMode::SIGMOID:
      return static_cast<T>(1) /
             (static_cast<T>(1) + std::exp(static_cast<T>(-value)));
    case UnaryPointwiseMode::GELU_APPROX_TANH:
      return evaluateGeluApproxTanh(value);
    case UnaryPointwiseMode::LOGICAL_NOT:
      return T{};
    }
  }
  return T{};
}

template <typename T>
static void unaryPointwiseKernel(const SimplifiedTensor &input,
                                 SimplifiedTensor &output,
                                 UnaryPointwiseMode mode) {
  using ComputeT = ReferenceComputeTypeT<T>;
  for (int64_t i = 0, totalElements = output.getTotalElements();
       i < totalElements; ++i) {
    const ComputeT value =
        NumericConverter<ComputeT, T>::convert(input.hostElement<T>(i));
    const ComputeT result = evaluateUnaryPointwise(mode, value);
    output.hostElement<T>(i) = NumericConverter<T, ComputeT>::convert(result);
  }
}

using UnaryPointwiseKernelFunc = void (*)(const SimplifiedTensor &,
                                          SimplifiedTensor &,
                                          UnaryPointwiseMode);

static constexpr DataTypeKernelEntry<UnaryPointwiseKernelFunc>
    kUnaryPointwiseKernels[] = {
        {DataType::FLOAT32, &unaryPointwiseKernel<float>},
        {DataType::FLOAT16, &unaryPointwiseKernel<half>},
        {DataType::BFLOAT16, &unaryPointwiseKernel<nv_bfloat16>},
        {DataType::DOUBLE, &unaryPointwiseKernel<double>},
        {DataType::INT32, &unaryPointwiseKernel<int32_t>},
        {DataType::INT8, &unaryPointwiseKernel<int8_t>},
        {DataType::BOOL, &unaryPointwiseKernel<bool>},
};

static bool
isSupportedParametricUnaryPointwiseMode(DataType dtype,
                                        ParametricUnaryPointwiseMode mode) {
  switch (dtype) {
  case DataType::FLOAT32:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::DOUBLE:
    switch (mode) {
    case ParametricUnaryPointwiseMode::SOFTPLUS_FWD:
    case ParametricUnaryPointwiseMode::SWISH_FWD:
    case ParametricUnaryPointwiseMode::ELU_FWD:
      return true;
    }
    return false;
  default:
    return false;
  }
}

template <typename T>
static T evaluateParametricUnaryPointwise(ParametricUnaryPointwiseMode mode,
                                          T value, T beta) {
  switch (mode) {
  case ParametricUnaryPointwiseMode::SOFTPLUS_FWD: {
    const T betaValue = beta * value;
    return (std::max(betaValue, static_cast<T>(0)) +
            std::log1p(std::exp(-std::abs(betaValue)))) /
           beta;
  }
  case ParametricUnaryPointwiseMode::SWISH_FWD:
    return value / (static_cast<T>(1) + std::exp(-beta * value));
  case ParametricUnaryPointwiseMode::ELU_FWD:
    return value > static_cast<T>(0) ? value : beta * std::expm1(value);
  }
  return T{};
}

template <typename T>
static void parametricUnaryPointwiseKernel(const SimplifiedTensor &input,
                                           SimplifiedTensor &output,
                                           ParametricUnaryPointwiseMode mode,
                                           double beta) {
  using ComputeT = ReferenceComputeTypeT<T>;
  const ComputeT computeBeta = static_cast<ComputeT>(beta);
  for (int64_t i = 0, totalElements = output.getTotalElements();
       i < totalElements; ++i) {
    const ComputeT value =
        NumericConverter<ComputeT, T>::convert(input.hostElement<T>(i));
    const ComputeT result =
        evaluateParametricUnaryPointwise(mode, value, computeBeta);
    output.hostElement<T>(i) = NumericConverter<T, ComputeT>::convert(result);
  }
}

using ParametricUnaryPointwiseKernelFunc =
    void (*)(const SimplifiedTensor &, SimplifiedTensor &,
             ParametricUnaryPointwiseMode, double);

static constexpr DataTypeKernelEntry<ParametricUnaryPointwiseKernelFunc>
    kParametricUnaryPointwiseKernels[] = {
        {DataType::FLOAT32, &parametricUnaryPointwiseKernel<float>},
        {DataType::FLOAT16, &parametricUnaryPointwiseKernel<half>},
        {DataType::BFLOAT16, &parametricUnaryPointwiseKernel<nv_bfloat16>},
        {DataType::DOUBLE, &parametricUnaryPointwiseKernel<double>},
};

static Status validatePointwiseTensors(const NamedTensors &tensors) {
  assert(!tensors.empty() && "pointwise validation requires tensors");
  TIR_RETURN_IF_ERROR(validateTensors(tensors));
  TIR_RETURN_IF_ERROR(validateSameType(tensors));
  TIR_RETURN_IF_ERROR(validateSameShape(tensors));
  return Status::Ok();
}

} // anonymous namespace

Status UnaryPointwiseNode::validate() const {
  TIR_RETURN_IF_ERROR(
      validatePointwiseTensors({{"Input", input}, {"Output", output}}));

  DataType dtype = input->getDataType();
  if (!isSupportedUnaryPointwiseMode(dtype, mode) ||
      !lookupDataTypeKernel(dtype, kUnaryPointwiseKernels)) {
    return Status::InvalidArgument(
        "Unsupported data type and operation combination. DataType: " +
        std::string(getEnumName(dtype)) +
        ", Mode: " + std::string(getEnumName(mode)));
  }
  return Status::Ok();
}

Status UnaryPointwiseNode::computeValidated() {
  UnaryPointwiseKernelFunc kernel =
      lookupDataTypeKernel(input->getDataType(), kUnaryPointwiseKernels);
  if (!kernel) {
    return Status::InvalidArgument(
        "Unsupported data type and operation combination");
  }

  kernel(*input, *output, mode);
  return Status::Ok();
}

Status ParametricUnaryPointwiseNode::validate() const {
  TIR_RETURN_IF_ERROR(
      validatePointwiseTensors({{"Input", input}, {"Output", output}}));

  DataType dtype = input->getDataType();
  if (!isSupportedParametricUnaryPointwiseMode(dtype, mode) ||
      !lookupDataTypeKernel(dtype, kParametricUnaryPointwiseKernels)) {
    return Status::InvalidArgument(
        "Unsupported data type and operation combination. DataType: " +
        std::string(getEnumName(dtype)) +
        ", Mode: " + std::string(getEnumName(mode)));
  }

  if (!std::isfinite(beta)) {
    return Status::InvalidArgument(std::string(getEnumName(mode)) +
                                   " beta must be finite");
  }

  if (mode == ParametricUnaryPointwiseMode::SOFTPLUS_FWD && beta == 0.0) {
    return Status::InvalidArgument("SOFTPLUS_FWD beta must be non-zero");
  }

  return Status::Ok();
}

Status ParametricUnaryPointwiseNode::computeValidated() {
  ParametricUnaryPointwiseKernelFunc kernel = lookupDataTypeKernel(
      input->getDataType(), kParametricUnaryPointwiseKernels);
  if (!kernel) {
    return Status::InvalidArgument(
        "Unsupported data type and operation combination");
  }

  kernel(*input, *output, mode, beta);
  return Status::Ok();
}

// ============================================================================
// CompareNode Implementation
// ============================================================================

namespace {

static bool isIntegerComparator(Comparator comparator) {
  switch (comparator) {
  case Comparator::eq:
  case Comparator::neq:
  case Comparator::gt:
  case Comparator::ge:
  case Comparator::lt:
  case Comparator::le:
    return true;
  default:
    return false;
  }
}

static bool isFloatComparator(Comparator comparator) {
  switch (comparator) {
  case Comparator::ueq:
  case Comparator::une:
  case Comparator::ugt:
  case Comparator::uge:
  case Comparator::ult:
  case Comparator::ule:
  case Comparator::oeq:
  case Comparator::one:
  case Comparator::ogt:
  case Comparator::oge:
  case Comparator::olt:
  case Comparator::ole:
    return true;
  default:
    return false;
  }
}

template <typename T>
static bool evaluateComparator(T lhsRaw, T rhsRaw, Comparator comparator) {
  // All supported integer types are exactly representable as double, so this
  // conversion preserves their comparison semantics.
  const double lhs = NumericConverter<double, T>::convert(lhsRaw);
  const double rhs = NumericConverter<double, T>::convert(rhsRaw);
  const bool unordered = std::isnan(lhs) || std::isnan(rhs);

  switch (comparator) {
  case Comparator::eq:
  case Comparator::oeq:
    return !unordered && lhs == rhs;
  case Comparator::neq:
  case Comparator::one:
    return !unordered && lhs != rhs;
  case Comparator::gt:
  case Comparator::ogt:
    return !unordered && lhs > rhs;
  case Comparator::ge:
  case Comparator::oge:
    return !unordered && lhs >= rhs;
  case Comparator::lt:
  case Comparator::olt:
    return !unordered && lhs < rhs;
  case Comparator::le:
  case Comparator::ole:
    return !unordered && lhs <= rhs;
  case Comparator::ueq:
    return unordered || lhs == rhs;
  case Comparator::une:
    return unordered || lhs != rhs;
  case Comparator::ugt:
    return unordered || lhs > rhs;
  case Comparator::uge:
    return unordered || lhs >= rhs;
  case Comparator::ult:
    return unordered || lhs < rhs;
  case Comparator::ule:
    return unordered || lhs <= rhs;
  default:
    return false;
  }
}

template <typename T>
static void compareTensor(const SimplifiedTensor &lhs,
                          const SimplifiedTensor &rhs, SimplifiedTensor &output,
                          Comparator comparator) {
  for (int64_t idx = 0, totalElements = output.getTotalElements();
       idx < totalElements; ++idx) {
    output.hostElement<bool>(idx) = evaluateComparator(
        lhs.hostElement<T>(idx), rhs.hostElement<T>(idx), comparator);
  }
}

} // anonymous namespace

Status CompareNode::validate() const {
  const NamedTensors tensors{
      {"lhs", lhsTensor}, {"rhs", rhsTensor}, {"output", outputTensor}};
  TIR_RETURN_IF_ERROR(validateTensors(tensors));
  TIR_RETURN_IF_ERROR(
      validateSameType({{"lhs", lhsTensor}, {"rhs", rhsTensor}}));

  const DataType inputType = lhsTensor->getDataType();
  if (outputTensor->getDataType() != DataType::BOOL) {
    return Status::InvalidArgument(
        "Compare output tensor must have BOOL data type");
  }

  if (isIntegerDataType(inputType)) {
    if (!isIntegerComparator(comparator)) {
      return Status::InvalidArgument(
          "Integer compare requires integer comparator, got " +
          mlir::nv_tensor_ir::stringifyComparator(comparator).str());
    }
  } else if (isFloatDataType(inputType)) {
    if (!isFloatComparator(comparator)) {
      return Status::InvalidArgument(
          "Floating-point compare requires floating-point comparator, got " +
          mlir::nv_tensor_ir::stringifyComparator(comparator).str());
    }
  } else {
    return Status::InvalidArgument("Unsupported data type for compare: " +
                                   std::string(getEnumName(inputType)));
  }

  TIR_RETURN_IF_ERROR(validateSameShape(tensors));
  return Status::Ok();
}

Status CompareNode::computeValidated() {
  const DataType dtype = lhsTensor->getDataType();
  if (!isIntegerDataType(dtype) && !isFloatDataType(dtype)) {
    return unsupportedDataType(dtype, "compare");
  }

  return dispatchDataType(dtype, [&](auto typeTag) {
    using T = typename decltype(typeTag)::Type;
    compareTensor<T>(*lhsTensor, *rhsTensor, *outputTensor, comparator);
    return Status::Ok();
  });
}

// ============================================================================
// BinaryPointwiseNode Implementation
// ============================================================================

const char *getEnumName(BinaryPointwiseMode mode) {
  switch (mode) {
  case BinaryPointwiseMode::ADD:
    return "ADD";
  case BinaryPointwiseMode::SUB:
    return "SUB";
  case BinaryPointwiseMode::MUL:
    return "MUL";
  case BinaryPointwiseMode::DIV:
    return "DIV";
  case BinaryPointwiseMode::MAX:
    return "MAX";
  case BinaryPointwiseMode::MIN:
    return "MIN";
  case BinaryPointwiseMode::POW:
    return "POW";
  case BinaryPointwiseMode::ATAN2:
    return "ATAN2";
  case BinaryPointwiseMode::MOD:
    return "MOD";
  case BinaryPointwiseMode::REM:
    return "REM";
  case BinaryPointwiseMode::ADD_SQUARE:
    return "ADD_SQUARE";
  case BinaryPointwiseMode::RELU_BWD:
    return "RELU_BWD";
  case BinaryPointwiseMode::GELU_BWD:
    return "GELU_BWD";
  case BinaryPointwiseMode::SIGMOID_BWD:
    return "SIGMOID_BWD";
  case BinaryPointwiseMode::TANH_BWD:
    return "TANH_BWD";
  case BinaryPointwiseMode::GELU_APPROX_TANH_BWD:
    return "GELU_APPROX_TANH_BWD";
  case BinaryPointwiseMode::LOGICAL_AND:
    return "LOGICAL_AND";
  case BinaryPointwiseMode::LOGICAL_OR:
    return "LOGICAL_OR";
  default:
    return "UNKNOWN";
  }
}

const char *getEnumName(ParametricBinaryPointwiseMode mode) {
  switch (mode) {
  case ParametricBinaryPointwiseMode::SOFTPLUS_BWD:
    return "SOFTPLUS_BWD";
  case ParametricBinaryPointwiseMode::SWISH_BWD:
    return "SWISH_BWD";
  case ParametricBinaryPointwiseMode::ELU_BWD:
    return "ELU_BWD";
  default:
    return "UNKNOWN";
  }
}

namespace {

// ============================================================================
// Binary Pointwise Kernels
// ============================================================================

/// The exact GELU derivative is Phi(x) + x * phi(x), where Phi and phi are
/// the standard normal CDF and density. These constants encode the 1/2,
/// 1/sqrt(2), and 1/sqrt(2*pi) factors in those functions.
constexpr double kGeluHalf = 0.5;
constexpr double kGeluInverseSqrtTwo = 0.70710678118654752440;
constexpr double kGeluInverseSqrtTwoPi = 0.39894228040143267794;

template <typename T>
static T evaluateExactGeluBackward(T value, T upstreamGradient) {
  const T half = static_cast<T>(kGeluHalf);
  const T cdf = half * (static_cast<T>(1) +
                        std::erf(value * static_cast<T>(kGeluInverseSqrtTwo)));
  const T densityContribution = value * std::exp(-half * value * value) *
                                static_cast<T>(kGeluInverseSqrtTwoPi);
  return upstreamGradient * (cdf + densityContribution);
}

template <typename T>
static T applyModulo(T a, T b) {
  // Python/PyTorch/JAX style modulo: a % b = a - floor(a/b) * b.
  // This differs from C++ std::fmod, which truncates toward zero.
  const T quotient = static_cast<T>(
      std::floor(static_cast<double>(a) / static_cast<double>(b)));
  return a - quotient * b;
}

template <typename T>
static T applyRemainder(T a, T b) {
  // Truncated remainder: a - trunc(a / b) * b. Result has the sign of the
  // dividend. Sister of applyModulo (floored).
  const T quotient = static_cast<T>(
      std::trunc(static_cast<double>(a) / static_cast<double>(b)));
  return a - quotient * b;
}

static bool isSupportedBinaryPointwiseMode(DataType dtype,
                                           BinaryPointwiseMode mode) {
  switch (dtype) {
  case DataType::FLOAT32:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::DOUBLE:
    switch (mode) {
    case BinaryPointwiseMode::ADD:
    case BinaryPointwiseMode::SUB:
    case BinaryPointwiseMode::MUL:
    case BinaryPointwiseMode::DIV:
    case BinaryPointwiseMode::MAX:
    case BinaryPointwiseMode::MIN:
    case BinaryPointwiseMode::POW:
    case BinaryPointwiseMode::ATAN2:
    case BinaryPointwiseMode::MOD:
    case BinaryPointwiseMode::REM:
    case BinaryPointwiseMode::ADD_SQUARE:
    case BinaryPointwiseMode::RELU_BWD:
    case BinaryPointwiseMode::GELU_BWD:
    case BinaryPointwiseMode::SIGMOID_BWD:
    case BinaryPointwiseMode::TANH_BWD:
    case BinaryPointwiseMode::GELU_APPROX_TANH_BWD:
      return true;
    case BinaryPointwiseMode::LOGICAL_AND:
    case BinaryPointwiseMode::LOGICAL_OR:
      return false;
    }
    return false;
  case DataType::INT32:
  case DataType::UINT32:
    switch (mode) {
    case BinaryPointwiseMode::ADD:
    case BinaryPointwiseMode::SUB:
    case BinaryPointwiseMode::MUL:
    case BinaryPointwiseMode::DIV:
    case BinaryPointwiseMode::MAX:
    case BinaryPointwiseMode::MIN:
    case BinaryPointwiseMode::MOD:
    case BinaryPointwiseMode::REM:
    case BinaryPointwiseMode::ADD_SQUARE:
      return true;
    case BinaryPointwiseMode::POW:
    case BinaryPointwiseMode::ATAN2:
    case BinaryPointwiseMode::RELU_BWD:
    case BinaryPointwiseMode::GELU_BWD:
    case BinaryPointwiseMode::SIGMOID_BWD:
    case BinaryPointwiseMode::TANH_BWD:
    case BinaryPointwiseMode::GELU_APPROX_TANH_BWD:
    case BinaryPointwiseMode::LOGICAL_AND:
    case BinaryPointwiseMode::LOGICAL_OR:
      return false;
    }
    return false;
  case DataType::BOOL:
    switch (mode) {
    case BinaryPointwiseMode::LOGICAL_AND:
    case BinaryPointwiseMode::LOGICAL_OR:
      return true;
    case BinaryPointwiseMode::ADD:
    case BinaryPointwiseMode::SUB:
    case BinaryPointwiseMode::MUL:
    case BinaryPointwiseMode::DIV:
    case BinaryPointwiseMode::MAX:
    case BinaryPointwiseMode::MIN:
    case BinaryPointwiseMode::POW:
    case BinaryPointwiseMode::ATAN2:
    case BinaryPointwiseMode::MOD:
    case BinaryPointwiseMode::REM:
    case BinaryPointwiseMode::ADD_SQUARE:
    case BinaryPointwiseMode::RELU_BWD:
    case BinaryPointwiseMode::GELU_BWD:
    case BinaryPointwiseMode::SIGMOID_BWD:
    case BinaryPointwiseMode::TANH_BWD:
    case BinaryPointwiseMode::GELU_APPROX_TANH_BWD:
      return false;
    }
    return false;
  default:
    return false;
  }
}

template <typename T>
static T evaluateBinaryPointwise(BinaryPointwiseMode mode, T a, T b) {
  if constexpr (std::is_same_v<T, bool>) {
    switch (mode) {
    case BinaryPointwiseMode::LOGICAL_AND:
      return a && b;
    case BinaryPointwiseMode::LOGICAL_OR:
      return a || b;
    default:
      return false;
    }
  } else {
    switch (mode) {
    case BinaryPointwiseMode::ADD:
      return a + b;
    case BinaryPointwiseMode::SUB:
      return a - b;
    case BinaryPointwiseMode::MUL:
      return a * b;
    case BinaryPointwiseMode::DIV:
      return a / b;
    case BinaryPointwiseMode::MAX:
      return a > b ? a : b;
    case BinaryPointwiseMode::MIN:
      return a < b ? a : b;
    case BinaryPointwiseMode::POW:
      if constexpr (std::is_integral_v<T>) {
        return T{};
      } else {
        return std::pow(a, b);
      }
    case BinaryPointwiseMode::ATAN2:
      if constexpr (std::is_integral_v<T>) {
        return T{};
      } else {
        return std::atan2(a, b);
      }
    case BinaryPointwiseMode::MOD:
      return applyModulo(a, b);
    case BinaryPointwiseMode::REM:
      return applyRemainder(a, b);
    case BinaryPointwiseMode::ADD_SQUARE:
      return a + b * b;
    case BinaryPointwiseMode::RELU_BWD:
      return a > static_cast<T>(0) ? b : static_cast<T>(0);
    case BinaryPointwiseMode::GELU_BWD:
      if constexpr (std::is_integral_v<T>) {
        return T{};
      } else {
        return evaluateExactGeluBackward(a, b);
      }
    case BinaryPointwiseMode::SIGMOID_BWD:
      if constexpr (std::is_integral_v<T>) {
        return T{};
      } else {
        const T sigmoid = static_cast<T>(1) /
                          (static_cast<T>(1) + std::exp(static_cast<T>(-a)));
        const T derivative = sigmoid * (static_cast<T>(1) - sigmoid);
        return b * derivative;
      }
    case BinaryPointwiseMode::TANH_BWD:
      if constexpr (std::is_integral_v<T>) {
        return T{};
      } else {
        const T tanhValue = std::tanh(a);
        const T derivative = static_cast<T>(1) - tanhValue * tanhValue;
        return b * derivative;
      }
    case BinaryPointwiseMode::GELU_APPROX_TANH_BWD:
      if constexpr (std::is_integral_v<T>) {
        return T{};
      } else {
        return evaluateGeluApproxTanhBwd(a, b);
      }
    case BinaryPointwiseMode::LOGICAL_AND:
      return a && b;
    case BinaryPointwiseMode::LOGICAL_OR:
      return a || b;
    }
    return T{};
  }
}

} // anonymous namespace

// Generic binary pointwise kernel
template <typename T>
static void
binaryPointwiseKernel(const SimplifiedTensor &a, const SimplifiedTensor &b,
                      SimplifiedTensor &c, BinaryPointwiseMode mode) {
  using ComputeT = ReferenceComputeTypeT<T>;
  for (int64_t i = 0, totalElements = c.getTotalElements(); i < totalElements;
       ++i) {
    const ComputeT lhs =
        NumericConverter<ComputeT, T>::convert(a.hostElement<T>(i));
    const ComputeT rhs =
        NumericConverter<ComputeT, T>::convert(b.hostElement<T>(i));
    const ComputeT result = evaluateBinaryPointwise(mode, lhs, rhs);
    c.hostElement<T>(i) = NumericConverter<T, ComputeT>::convert(result);
  }
}

template <typename T>
static Status validateIntegerDivRemInputs(BinaryPointwiseMode mode,
                                          const SimplifiedTensor &a,
                                          const SimplifiedTensor &b,
                                          const SimplifiedTensor &c) {
  if (mode != BinaryPointwiseMode::DIV && mode != BinaryPointwiseMode::MOD &&
      mode != BinaryPointwiseMode::REM) {
    return Status::Ok();
  }

  const char *typeName = getEnumName(a.getDataType());
  const std::string prefix = std::string(typeName) + " " + getEnumName(mode) +
                             " binary pointwise operation ";

  for (int64_t i = 0, totalElements = c.getTotalElements(); i < totalElements;
       ++i) {
    const T lhs = a.hostElement<T>(i);
    const T rhs = b.hostElement<T>(i);
    if (rhs == 0) {
      return Status::InvalidArgument(prefix + "has zero RHS element");
    }
    if constexpr (std::is_signed_v<T>) {
      if (lhs == std::numeric_limits<T>::min() && rhs == -1) {
        return Status::InvalidArgument(prefix + "overflows for " + typeName +
                                       "_MIN / -1");
      }
    }
  }

  return Status::Ok();
}

// Type alias for binary pointwise kernel function pointer
using BinaryPointwiseKernelFunc = void (*)(const SimplifiedTensor &,
                                           const SimplifiedTensor &,
                                           SimplifiedTensor &,
                                           BinaryPointwiseMode);

namespace {

static constexpr DataTypeKernelEntry<BinaryPointwiseKernelFunc>
    kBinaryPointwiseKernels[] = {
        {DataType::FLOAT32, &binaryPointwiseKernel<float>},
        {DataType::FLOAT16, &binaryPointwiseKernel<half>},
        {DataType::BFLOAT16, &binaryPointwiseKernel<nv_bfloat16>},
        {DataType::DOUBLE, &binaryPointwiseKernel<double>},
        {DataType::INT32, &binaryPointwiseKernel<int32_t>},
        {DataType::UINT32, &binaryPointwiseKernel<uint32_t>},
        {DataType::BOOL, &binaryPointwiseKernel<bool>},
};

} // anonymous namespace

Status BinaryPointwiseNode::validate() const {
  const NamedTensors tensors{{"A", a}, {"B", b}, {"C", c}};
  TIR_RETURN_IF_ERROR(validatePointwiseTensors(tensors));
  DataType dtype = a->getDataType();
  if (!isSupportedBinaryPointwiseMode(dtype, mode) ||
      !lookupDataTypeKernel(dtype, kBinaryPointwiseKernels)) {
    return Status::InvalidArgument(
        "Unsupported data type and operation combination. DataType: " +
        std::string(getEnumName(dtype)) +
        ", Mode: " + std::string(getEnumName(mode)));
  }

  return Status::Ok();
}

Status BinaryPointwiseNode::computeValidated() {
  DataType dtype = a->getDataType();
  BinaryPointwiseKernelFunc kernelFunc =
      lookupDataTypeKernel(dtype, kBinaryPointwiseKernels);
  if (!kernelFunc) {
    return Status::InvalidArgument(
        "Unsupported data type and operation combination");
  }

  if (dtype == DataType::INT32) {
    TIR_RETURN_IF_ERROR(validateIntegerDivRemInputs<int32_t>(mode, *a, *b, *c));
  } else if (dtype == DataType::UINT32) {
    TIR_RETURN_IF_ERROR(
        validateIntegerDivRemInputs<uint32_t>(mode, *a, *b, *c));
  }

  kernelFunc(*a, *b, *c, mode);

  return Status::Ok();
}

// ============================================================================
// ParametricBinaryPointwiseNode Implementation
// ============================================================================

namespace {

static bool
isSupportedParametricBinaryPointwiseMode(DataType dtype,
                                         ParametricBinaryPointwiseMode mode) {
  switch (dtype) {
  case DataType::FLOAT32:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::DOUBLE:
    switch (mode) {
    case ParametricBinaryPointwiseMode::SOFTPLUS_BWD:
    case ParametricBinaryPointwiseMode::SWISH_BWD:
    case ParametricBinaryPointwiseMode::ELU_BWD:
      return true;
    }
    return false;
  default:
    return false;
  }
}

template <typename T>
static T evaluateParametricBinaryPointwise(ParametricBinaryPointwiseMode mode,
                                           T input, T gradient, T beta) {
  switch (mode) {
  case ParametricBinaryPointwiseMode::SOFTPLUS_BWD: {
    const T sigmoid =
        static_cast<T>(1) / (static_cast<T>(1) + std::exp(-beta * input));
    return gradient * sigmoid;
  }
  case ParametricBinaryPointwiseMode::SWISH_BWD: {
    const T betaInput = beta * input;
    const T sigmoid =
        static_cast<T>(1) / (static_cast<T>(1) + std::exp(-betaInput));
    const T derivative =
        sigmoid + betaInput * sigmoid * (static_cast<T>(1) - sigmoid);
    return gradient * derivative;
  }
  case ParametricBinaryPointwiseMode::ELU_BWD: {
    const T derivative =
        input > static_cast<T>(0) ? static_cast<T>(1) : beta * std::exp(input);
    return gradient * derivative;
  }
  }
  return T{};
}

template <typename T>
static void parametricBinaryPointwiseKernel(const SimplifiedTensor &input,
                                            const SimplifiedTensor &gradient,
                                            SimplifiedTensor &output,
                                            ParametricBinaryPointwiseMode mode,
                                            double beta) {
  using ComputeT = ReferenceComputeTypeT<T>;
  const ComputeT computeBeta = static_cast<ComputeT>(beta);
  for (int64_t i = 0, totalElements = output.getTotalElements();
       i < totalElements; ++i) {
    const ComputeT inputValue =
        NumericConverter<ComputeT, T>::convert(input.hostElement<T>(i));
    const ComputeT gradientValue =
        NumericConverter<ComputeT, T>::convert(gradient.hostElement<T>(i));
    const ComputeT result = evaluateParametricBinaryPointwise(
        mode, inputValue, gradientValue, computeBeta);
    output.hostElement<T>(i) = NumericConverter<T, ComputeT>::convert(result);
  }
}

using ParametricBinaryPointwiseKernelFunc =
    void (*)(const SimplifiedTensor &, const SimplifiedTensor &,
             SimplifiedTensor &, ParametricBinaryPointwiseMode, double);

static constexpr DataTypeKernelEntry<ParametricBinaryPointwiseKernelFunc>
    kParametricBinaryPointwiseKernels[] = {
        {DataType::FLOAT32, &parametricBinaryPointwiseKernel<float>},
        {DataType::FLOAT16, &parametricBinaryPointwiseKernel<half>},
        {DataType::BFLOAT16, &parametricBinaryPointwiseKernel<nv_bfloat16>},
        {DataType::DOUBLE, &parametricBinaryPointwiseKernel<double>},
};

} // anonymous namespace

Status ParametricBinaryPointwiseNode::validate() const {
  const NamedTensors tensors{
      {"Input", input}, {"Gradient", gradient}, {"Output", output}};
  TIR_RETURN_IF_ERROR(validatePointwiseTensors(tensors));

  const DataType dtype = input->getDataType();
  if (!isSupportedParametricBinaryPointwiseMode(dtype, mode) ||
      !lookupDataTypeKernel(dtype, kParametricBinaryPointwiseKernels)) {
    return Status::InvalidArgument(
        "Unsupported data type and operation combination. DataType: " +
        std::string(getEnumName(dtype)) +
        ", Mode: " + std::string(getEnumName(mode)));
  }

  if (!std::isfinite(beta)) {
    return Status::InvalidArgument(std::string(getEnumName(mode)) +
                                   " beta must be finite");
  }

  return Status::Ok();
}

Status ParametricBinaryPointwiseNode::computeValidated() {
  ParametricBinaryPointwiseKernelFunc kernel = lookupDataTypeKernel(
      input->getDataType(), kParametricBinaryPointwiseKernels);
  if (!kernel) {
    return Status::InvalidArgument(
        "Unsupported data type and operation combination");
  }

  kernel(*input, *gradient, *output, mode, beta);
  return Status::Ok();
}

// ============================================================================
// BinarySelectNode Implementation
// ============================================================================

static void binarySelectKernel(const SimplifiedTensor &selector,
                               const SimplifiedTensor &A,
                               const SimplifiedTensor &B, SimplifiedTensor &C,
                               size_t elementSize) {
  const char *APtr = static_cast<const char *>(A.hostPtr());
  const char *BPtr = static_cast<const char *>(B.hostPtr());
  char *CPtr = static_cast<char *>(C.hostPtr());

  for (int64_t i = 0, totalElements = C.getTotalElements(); i < totalElements;
       ++i) {
    const bool selectA = selector.hostElement<bool>(i);
    const SimplifiedTensor &sourceTensor = selectA ? A : B;
    const char *sourcePtr = selectA ? APtr : BPtr;
    const int64_t sourceIdx = sourceTensor.getStorageIndex(i);
    const int64_t cIdx = C.getStorageIndex(i);
    std::memcpy(CPtr + cIdx * elementSize, sourcePtr + sourceIdx * elementSize,
                elementSize);
  }
}

Status BinarySelectNode::validate() const {
  const NamedTensors tensors{
      {"Selector", selector_}, {"A", A_}, {"B", B_}, {"C", C_}};
  TIR_RETURN_IF_ERROR(validateTensors(tensors));

  if (selector_->getDataType() != DataType::BOOL) {
    return Status::InvalidArgument(
        "Binary select selector tensor must have BOOL data type");
  }

  TIR_RETURN_IF_ERROR(validateSameType({{"A", A_}, {"B", B_}, {"C", C_}}));

  TIR_RETURN_IF_ERROR(validateSameShape(tensors));

  return Status::Ok();
}

Status BinarySelectNode::computeValidated() {
  size_t elementSize = getDataTypeSize(A_->getDataType());
  if (elementSize == 0) {
    return unsupportedDataType(A_->getDataType(), "binary select");
  }

  binarySelectKernel(*selector_, *A_, *B_, *C_, elementSize);
  return Status::Ok();
}

} // namespace mlir::nv_tensor_ir::reference
