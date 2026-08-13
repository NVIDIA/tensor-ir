// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_SUPPORT_STATUS_H
#define TENSOR_IR_SUPPORT_STATUS_H

#include "tensor_ir/Support/Macros.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace mlir::nv_tensor_ir {

//===----------------------------------------------------------------------===//
// StatusCode
//===----------------------------------------------------------------------===//

enum class StatusCode : int32_t {
  kSuccess = 0,
  kInvalidArgument,
  kCudaError,
  kCompilationError,
  kNotSupported,
  kInsufficientWorkspace,
  kNotFound,
  kConstraintNotSatisfied, // checkSupport failed
  kNotInitialized,
  kNotImplemented,
};

//===----------------------------------------------------------------------===//
// Status
//===----------------------------------------------------------------------===//

class Status {
  StatusCode code_;
  std::string message_;

public:
  Status() : code_(StatusCode::kSuccess) {}
  explicit Status(StatusCode code) : code_(code) {}
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  bool ok() const { return code_ == StatusCode::kSuccess; }
  StatusCode code() const { return code_; }
  const std::string &message() const { return message_; }

  static Status Ok() { return Status(); }
  static Status InvalidArgument(std::string msg = "") {
    return Status(StatusCode::kInvalidArgument, std::move(msg));
  }
  static Status CudaError(std::string msg = "") {
    return Status(StatusCode::kCudaError, std::move(msg));
  }
  static Status CompilationError(std::string msg = "") {
    return Status(StatusCode::kCompilationError, std::move(msg));
  }
  static Status NotSupported(std::string msg = "") {
    return Status(StatusCode::kNotSupported, std::move(msg));
  }
  static Status NotFound(std::string msg = "") {
    return Status(StatusCode::kNotFound, std::move(msg));
  }
  static Status ConstraintNotSatisfied(std::string msg = "") {
    return Status(StatusCode::kConstraintNotSatisfied, std::move(msg));
  }
  static Status NotInitialized(std::string msg = "") {
    return Status(StatusCode::kNotInitialized, std::move(msg));
  }
  static Status NotImplemented(std::string msg = "") {
    return Status(StatusCode::kNotImplemented, std::move(msg));
  }
};

/// Evaluate `expr` once and return its non-OK status from the current function.
#define TIR_RETURN_IF_ERROR(expr)                                              \
  do {                                                                         \
    auto statusReturnIfError = (expr);                                         \
    if (!statusReturnIfError.ok())                                             \
      return statusReturnIfError;                                              \
  } while (false)

/// Evaluate a StatusOr expression once, return its error, or move its value
/// into `lhs`. `lhs` may declare a variable, for example:
///   TIR_ASSIGN_OR_RETURN(auto value, makeValue());
#define TIR_ASSIGN_OR_RETURN(lhs, ...)                                         \
  TIR_ASSIGN_OR_RETURN_IMPL(TIR_CONCAT(statusAssignOrReturn, __COUNTER__),     \
                            lhs, __VA_ARGS__)

#define TIR_ASSIGN_OR_RETURN_IMPL(statusOrValue, lhs, ...)                     \
  auto statusOrValue = (__VA_ARGS__);                                          \
  if (!statusOrValue.ok())                                                     \
    return statusOrValue.status();                                             \
  lhs = std::move(*statusOrValue)

inline bool ok(const Status &s) { return s.ok(); }

//===----------------------------------------------------------------------===//
// StatusOr<T>
//===----------------------------------------------------------------------===//

template <typename T>
class StatusOr {
  std::optional<T> value_;
  Status status_;

public:
  StatusOr(T value) : value_(std::move(value)), status_(Status::Ok()) {}
  StatusOr(Status status) : status_(std::move(status)) {}
  StatusOr(StatusCode code) : status_(code) {}

  bool ok() const { return status_.ok(); }
  const Status &status() const { return status_; }

  T &value() & { return *value_; }
  const T &value() const & { return *value_; }
  T &&value() && { return std::move(*value_); }

  T &operator*() & { return *value_; }
  const T &operator*() const & { return *value_; }
  T *operator->() { return &(*value_); }
  const T *operator->() const { return &(*value_); }
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_SUPPORT_STATUS_H
