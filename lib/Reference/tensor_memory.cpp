// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Reference/tensor_memory.h"

#include "tensor_ir/Reference/simplified_tensor.h"
#include "tensor_ir/Support/CudaApi.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <utility>

namespace mlir::nv_tensor_ir::reference {

TensorMemory::TensorMemory(size_t sizeInBytes)
    : hostMemory(std::calloc(1, sizeInBytes)) {
  if (!hostMemory) {
    llvm::errs() << "TensorMemory: Failed to allocate host memory: "
                 << sizeInBytes << " bytes\n";
  }
}

TensorMemory::~TensorMemory() { std::free(hostMemory); }

static void freeDeviceMemory(void *ptr, const char *context) {
  Status status = ::mlir::nv_tensor_ir::cuda::runtime::free(ptr);
  if (!status.ok()) {
    llvm::errs() << "DeviceBuffer: failed to free device memory " << context
                 << ": " << status.message() << "\n";
  }
}

StatusOr<DeviceBuffer> DeviceBuffer::create(size_t sizeInBytes) {
  if (sizeInBytes == 0) {
    return Status::InvalidArgument(
        "DeviceBuffer: cannot allocate zero bytes of device memory");
  }

  StatusOr<void *> ptrOrErr =
      ::mlir::nv_tensor_ir::cuda::runtime::allocate(sizeInBytes);
  if (!ptrOrErr.ok()) {
    return ptrOrErr.status();
  }
  void *ptr = *ptrOrErr;

  // Zero-initialize so callers observe deterministic contents. On failure the
  // just-allocated buffer is freed so we never leak device memory.
  Status zeroStatus =
      ::mlir::nv_tensor_ir::cuda::runtime::memset(ptr, 0, sizeInBytes);
  if (!zeroStatus.ok()) {
    freeDeviceMemory(ptr, "after cudaMemset failure");
    return zeroStatus;
  }

  return DeviceBuffer(ptr, sizeInBytes);
}

DeviceBuffer::~DeviceBuffer() {
  if (ptr) {
    freeDeviceMemory(ptr, "in destructor");
  }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : ptr(other.ptr), sizeInBytes(other.sizeInBytes) {
  other.ptr = nullptr;
  other.sizeInBytes = 0;
}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
  if (this != &other) {
    if (ptr) {
      freeDeviceMemory(ptr, "before move assignment");
    }
    ptr = other.ptr;
    sizeInBytes = other.sizeInBytes;
    other.ptr = nullptr;
    other.sizeInBytes = 0;
  }
  return *this;
}

Status DeviceBuffer::copyFrom(const void *host) {
  if (!ptr || !host) {
    return Status::InvalidArgument(
        "DeviceBuffer::copyFrom: cannot copy from or to a null pointer");
  }
  return ::mlir::nv_tensor_ir::cuda::runtime::copyHostToDevice(ptr, host,
                                                               sizeInBytes);
}

Status DeviceBuffer::copyTo(void *host) const {
  if (!ptr || !host) {
    return Status::InvalidArgument(
        "DeviceBuffer::copyTo: cannot copy from or to a null pointer");
  }
  return ::mlir::nv_tensor_ir::cuda::runtime::copyDeviceToHost(host, ptr,
                                                               sizeInBytes);
}

static tensor_ir::rt::Any makeTensorArg(const SimplifiedTensor &tensor,
                                        void *devicePtr) {
  tensor_ir::rt::TensorView arg = {
      devicePtr, static_cast<int32_t>(tensor.getDims().size()),
      const_cast<int64_t *>(tensor.getDims().data()),
      const_cast<int64_t *>(tensor.getDesc().strides.data())};
  return tensor_ir::rt::Any(arg);
}

static void printDeviceData(const SimplifiedTensor &tensor,
                            const DeviceBuffer &deviceBuffer);

StatusOr<DeviceRun>
DeviceRun::create(llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> inputs,
                  llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> outputs) {
  DeviceRun run;
  run.inputBuffers.reserve(inputs.size());
  run.outputBuffers.reserve(outputs.size());
  run.inputs.reserve(inputs.size());
  run.outputs.reserve(outputs.size());
  run.argsStorage.reserve(inputs.size() + outputs.size());

  for (const std::shared_ptr<SimplifiedTensor> &input : inputs) {
    if (!input || !input->hostPtr()) {
      return Status::InvalidArgument(
          "DeviceRun: input tensor must have host memory");
    }
    std::optional<size_t> sizeInBytes = input->getSizeInBytes();
    if (!sizeInBytes) {
      return Status::InvalidArgument(
          "DeviceRun: input tensor storage size overflowed");
    }
    StatusOr<DeviceBuffer> bufferOrErr = DeviceBuffer::create(*sizeInBytes);
    if (!bufferOrErr.ok()) {
      return bufferOrErr.status();
    }
    Status copyStatus = bufferOrErr->copyFrom(input->hostPtr());
    if (!copyStatus.ok()) {
      return copyStatus;
    }
    void *devicePtr = bufferOrErr->get();
    run.inputBuffers.push_back(std::move(*bufferOrErr));
    run.inputs.push_back(input);
    run.argsStorage.push_back(makeTensorArg(*input, devicePtr));
  }

  for (const std::shared_ptr<SimplifiedTensor> &output : outputs) {
    if (!output) {
      return Status::InvalidArgument("DeviceRun: output tensor must be valid");
    }
    std::optional<size_t> sizeInBytes = output->getSizeInBytes();
    if (!sizeInBytes) {
      return Status::InvalidArgument(
          "DeviceRun: output tensor storage size overflowed");
    }
    StatusOr<DeviceBuffer> bufferOrErr = DeviceBuffer::create(*sizeInBytes);
    if (!bufferOrErr.ok()) {
      return bufferOrErr.status();
    }
    void *devicePtr = bufferOrErr->get();
    run.outputBuffers.push_back(std::move(*bufferOrErr));
    run.outputs.push_back(output);
    run.argsStorage.push_back(makeTensorArg(*output, devicePtr));
  }

  return run;
}

Status
DeviceRun::copyOutputsTo(std::vector<std::vector<std::byte>> &results) const {
  results.clear();
  results.reserve(outputBuffers.size());
  for (const DeviceBuffer &output : outputBuffers) {
    std::vector<std::byte> result(output.size());
    Status copyStatus = output.copyTo(result.data());
    if (!copyStatus.ok()) {
      return copyStatus;
    }
    results.push_back(std::move(result));
  }
  return Status::Ok();
}

Status DeviceRun::printInputs() const {
  if (inputs.size() != inputBuffers.size()) {
    return Status::InvalidArgument(
        "DeviceRun: input tensor count does not match device buffers");
  }
  for (size_t i = 0, e = inputs.size(); i != e; ++i) {
    printDeviceData(*inputs[i], inputBuffers[i]);
  }
  return Status::Ok();
}

Status DeviceRun::printOutputs() const {
  if (outputs.size() != outputBuffers.size()) {
    return Status::InvalidArgument(
        "DeviceRun: output tensor count does not match device buffers");
  }
  for (size_t i = 0, e = outputs.size(); i != e; ++i) {
    printDeviceData(*outputs[i], outputBuffers[i]);
  }
  return Status::Ok();
}

static void printDeviceData(const SimplifiedTensor &tensor,
                            const DeviceBuffer &deviceBuffer) {
  const void *devPtr = deviceBuffer.get();
  const TensorDesc &desc = tensor.getDesc();
  std::optional<size_t> sizeInBytes = tensor.getSizeInBytes();

  auto printError = [&](const char *errorMsg) {
    llvm::errs() << "Device Data: SimplifiedTensor(name: " << tensor.getName()
                 << ", desc: ";
    desc.print();
    llvm::errs() << "devicePtr: " << devPtr << ", sizeInBytes: ";
    if (sizeInBytes) {
      llvm::errs() << *sizeInBytes;
    } else {
      llvm::errs() << "overflow";
    }
    llvm::errs() << ", allocated: " << (tensor.isAllocated() ? "Yes" : "No")
                 << ")"
                 << "\n";
    llvm::errs() << "  " << errorMsg << "\n";
  };

  if (!devPtr || !tensor.isAllocated()) {
    printError("[No data - not allocated or null pointer]");
    return;
  }

  if (!sizeInBytes) {
    printError("[Error: tensor storage size overflowed]");
    return;
  }

  if (desc.totalElements == 0) {
    printError("[No data - zero elements]");
    return;
  }

  // copyTo() transfers deviceBuffer.size() bytes; guard against a mismatch so
  // we never overflow the host staging buffer sized to this tensor.
  if (deviceBuffer.size() != *sizeInBytes) {
    printError("[Error: device buffer size does not match tensor size]");
    return;
  }

  std::vector<std::byte> tempHostData(*sizeInBytes);
  Status copyStatus = deviceBuffer.copyTo(tempHostData.data());
  if (!copyStatus.ok()) {
    std::string errorMsg = "[Error: ";
    errorMsg += copyStatus.message();
    errorMsg += "]";
    printError(errorMsg.c_str());
    return;
  }

  tensor.printData(tempHostData.data(), "Device Data", "devicePtr", devPtr);
}

} // namespace mlir::nv_tensor_ir::reference
