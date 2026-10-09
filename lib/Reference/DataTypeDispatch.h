// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_REFERENCE_DATATYPEDISPATCH_H
#define TENSOR_IR_REFERENCE_DATATYPEDISPATCH_H

#include "tensor_ir/Reference/SimplifiedTensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <utility>

namespace mlir::nv_tensor_ir::reference::detail {

/// Tag type that carries the canonical C++ element type for a DataType.
template <typename T>
struct DataTypeTag {
  using Type = T;
};

/// Invoke \p callback with a tag for the canonical C++ type of \p dtype.
///
/// Callers retain responsibility for validating which data types their
/// operation supports. The callback must return a Status.
template <typename Callback>
Status dispatchDataType(DataType dtype, Callback &&callback) {
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
  case DataType::F4E2M1FN:
    break;
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
  return Status::InvalidArgument("Unsupported data type " +
                                 std::string(getEnumName(dtype)) +
                                 " for data type dispatch");
}

} // namespace mlir::nv_tensor_ir::reference::detail

#endif // TENSOR_IR_REFERENCE_DATATYPEDISPATCH_H
