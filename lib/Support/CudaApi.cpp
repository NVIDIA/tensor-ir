// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file is TensorIR's only raw CUDA API boundary. Driver entry points are
// always resolved from the installed driver. CUDA Runtime entry points either
// use the same dynamic dispatch or bind to cudart_static at build time. Public
// callers see only the Status-based functions declared in CudaApi.h.

#include "tensor_ir/Support/CudaApi.h"

#include "cuda_runtime_api.h"
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mlir::nv_tensor_ir::cuda {
namespace {

#if defined(_WIN32)
using DynamicLibraryHandle = HMODULE;
#else
using DynamicLibraryHandle = void *;
#endif

std::string getDynamicLibraryError() {
#if defined(_WIN32)
  return "error code " + std::to_string(GetLastError());
#else
  const char *error = dlerror();
  return error ? error : "unknown dynamic loader error";
#endif
}

DynamicLibraryHandle openDynamicLibrary(std::string_view name) {
#if defined(_WIN32)
  return LoadLibraryA(std::string(name).c_str());
#else
  dlerror();
  return dlopen(std::string(name).c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
}

void *getDynamicLibrarySymbol(DynamicLibraryHandle handle, const char *name) {
#if defined(_WIN32)
  return reinterpret_cast<void *>(GetProcAddress(handle, name));
#else
  dlerror();
  return dlsym(handle, name);
#endif
}

StatusOr<DynamicLibraryHandle>
openFirstDynamicLibrary(std::initializer_list<std::string> names,
                        const char *description) {
  std::ostringstream errors;
  bool first = true;
  for (const std::string &name : names) {
    if (DynamicLibraryHandle handle = openDynamicLibrary(name)) {
      return handle;
    }
    if (!first) {
      errors << "; ";
    }
    first = false;
    errors << name << ": " << getDynamicLibraryError();
  }
  return Status::NotFound(std::string("Failed to load ") + description + ": " +
                          errors.str());
}

template <typename Function>
Status resolveSymbol(DynamicLibraryHandle handle, Function &function,
                     const char *name, const char *description) {
  function = reinterpret_cast<Function>(getDynamicLibrarySymbol(handle, name));
  if (!function) {
    return Status::NotInitialized(std::string("Failed to resolve ") + name +
                                  " from " + description + ": " +
                                  getDynamicLibraryError());
  }
  return Status::Ok();
}

#define TIR_STRINGIFY_IMPL(value) #value
#define TIR_STRINGIFY(value) TIR_STRINGIFY_IMPL(value)

#define TIR_DRIVER_FUNCTIONS(APPLY)                                            \
  APPLY(getErrorName, cuGetErrorName);                                         \
  APPLY(getErrorString, cuGetErrorString);                                     \
  APPLY(getDriverVersion, cuDriverGetVersion);                                 \
  APPLY(libraryLoadData, cuLibraryLoadData);                                   \
  APPLY(libraryGetKernel, cuLibraryGetKernel);                                 \
  APPLY(libraryUnload, cuLibraryUnload);                                       \
  APPLY(launchKernelEx, cuLaunchKernelEx)

#define TIR_RUNTIME_FUNCTIONS(APPLY)                                           \
  APPLY(getErrorString, cudaGetErrorString);                                   \
  APPLY(getLastError, cudaGetLastError);                                       \
  APPLY(freeMemory, cudaFree);                                                 \
  APPLY(mallocMemory, cudaMalloc);                                             \
  APPLY(memsetMemory, cudaMemset);                                             \
  APPLY(copyMemory, cudaMemcpy);                                               \
  APPLY(deviceSynchronize, cudaDeviceSynchronize);                             \
  APPLY(eventCreate, cudaEventCreate);                                         \
  APPLY(eventDestroy, cudaEventDestroy);                                       \
  APPLY(eventRecord, cudaEventRecord);                                         \
  APPLY(eventSynchronize, cudaEventSynchronize);                               \
  APPLY(eventElapsedTime, cudaEventElapsedTime);                               \
  APPLY(getDeviceCount, cudaGetDeviceCount);                                   \
  APPLY(getDevice, cudaGetDevice);                                             \
  APPLY(getDeviceProperties, cudaGetDeviceProperties)

struct DriverDispatch {
  DriverDispatch() {
#if defined(_WIN32)
    StatusOr<DynamicLibraryHandle> library =
        openFirstDynamicLibrary({"nvcuda.dll"}, "CUDA driver library");
#else
    StatusOr<DynamicLibraryHandle> library =
        openFirstDynamicLibrary({"libcuda.so.1"}, "CUDA driver library");
#endif
    if (!library.ok()) {
      status = library.status();
      return;
    }
    handle = *library;

#define TIR_RESOLVE_DRIVER(member, symbol)                                     \
  do {                                                                         \
    status = resolveSymbol(handle, member, TIR_STRINGIFY(symbol),              \
                           "CUDA driver library");                             \
    if (!status.ok())                                                          \
      return;                                                                  \
  } while (false)

    TIR_DRIVER_FUNCTIONS(TIR_RESOLVE_DRIVER);
#undef TIR_RESOLVE_DRIVER
  }

  DynamicLibraryHandle handle = nullptr;
  Status status = Status::Ok();
#define TIR_DECLARE_FUNCTION(member, symbol)                                   \
  decltype(&::symbol) member /* NOLINT(bugprone-macro-parentheses) */ = nullptr
  TIR_DRIVER_FUNCTIONS(TIR_DECLARE_FUNCTION);
#undef TIR_DECLARE_FUNCTION
};

struct RuntimeDispatch {
  RuntimeDispatch() {
#if defined(TENSOR_IR_STATIC_LINK_CUDART)
#define TIR_LINK_RUNTIME(member, symbol) member = &::symbol
    TIR_RUNTIME_FUNCTIONS(TIR_LINK_RUNTIME);
#undef TIR_LINK_RUNTIME
#else
    constexpr int runtimeMajorVersion = CUDART_VERSION / 1000;
#if defined(_WIN32)
    std::string majorName =
        "cudart64_" + std::to_string(runtimeMajorVersion) + ".dll";
    std::string fullName =
        "cudart64_" + std::to_string(CUDART_VERSION / 100) + ".dll";
    StatusOr<DynamicLibraryHandle> library =
        openFirstDynamicLibrary({majorName, fullName}, "CUDA Runtime library");
#else
    std::string versionedName =
        "libcudart.so." + std::to_string(runtimeMajorVersion);
    StatusOr<DynamicLibraryHandle> library = openFirstDynamicLibrary(
        {versionedName, "libcudart.so"}, "CUDA Runtime library");
#endif
    if (!library.ok()) {
      status = library.status();
      return;
    }
    handle = *library;

#define TIR_RESOLVE_RUNTIME(member, symbol)                                    \
  do {                                                                         \
    status = resolveSymbol(handle, member, TIR_STRINGIFY(symbol),              \
                           "CUDA Runtime library");                            \
    if (!status.ok())                                                          \
      return;                                                                  \
  } while (false)

    TIR_RUNTIME_FUNCTIONS(TIR_RESOLVE_RUNTIME);
#undef TIR_RESOLVE_RUNTIME
#endif
  }

  DynamicLibraryHandle handle = nullptr;
  Status status = Status::Ok();
#define TIR_DECLARE_FUNCTION(member, symbol)                                   \
  decltype(&::symbol) member /* NOLINT(bugprone-macro-parentheses) */ = nullptr
  TIR_RUNTIME_FUNCTIONS(TIR_DECLARE_FUNCTION);
#undef TIR_DECLARE_FUNCTION
};

#undef TIR_DRIVER_FUNCTIONS
#undef TIR_RUNTIME_FUNCTIONS

DriverDispatch &getDriverDispatch() {
  static DriverDispatch dispatch;
  return dispatch;
}

RuntimeDispatch &getRuntimeDispatch() {
  static RuntimeDispatch dispatch;
  return dispatch;
}

std::string formatDriverError(DriverDispatch &dispatch, CUresult result) {
  const char *name = nullptr;
  const char *description = nullptr;
  dispatch.getErrorName(result, &name);
  dispatch.getErrorString(result, &description);

  std::string message =
      name ? name
           : "CUDA driver error " + std::to_string(static_cast<int>(result));
  if (description) {
    message += " (";
    message += description;
    message += ")";
  }
  return message;
}

Status getDriverCallStatus(DriverDispatch &dispatch, CUresult result,
                           const char *operation) {
  if (result == CUDA_SUCCESS) {
    return Status::Ok();
  }
  return Status::CudaError(std::string(operation) +
                           " failed: " + formatDriverError(dispatch, result));
}

Status getRuntimeCallStatus(RuntimeDispatch &dispatch, cudaError_t result,
                            const char *operation) {
  if (result == cudaSuccess) {
    return Status::Ok();
  }
  return Status::CudaError(std::string(operation) +
                           " failed: " + dispatch.getErrorString(result));
}

template <typename Callable>
Status callDriver(const char *operation, Callable &&call) {
  DriverDispatch &dispatch = getDriverDispatch();
  if (!dispatch.status.ok()) {
    return dispatch.status;
  }
  return getDriverCallStatus(dispatch, std::forward<Callable>(call)(dispatch),
                             operation);
}

template <typename T, typename Callable>
StatusOr<T> callDriverWithOutput(const char *operation, Callable &&call) {
  T output{};
  Status status = callDriver(operation, [&](DriverDispatch &dispatch) {
    return std::forward<Callable>(call)(dispatch, output);
  });
  if (!status.ok()) {
    return status;
  }
  return output;
}

template <typename Callable>
Status callRuntime(const char *operation, Callable &&call) {
  RuntimeDispatch &dispatch = getRuntimeDispatch();
  if (!dispatch.status.ok()) {
    return dispatch.status;
  }
  return getRuntimeCallStatus(dispatch, std::forward<Callable>(call)(dispatch),
                              operation);
}

template <typename T, typename Callable>
StatusOr<T> callRuntimeWithOutput(const char *operation, Callable &&call) {
  T output{};
  Status status = callRuntime(operation, [&](RuntimeDispatch &dispatch) {
    return std::forward<Callable>(call)(dispatch, output);
  });
  if (!status.ok()) {
    return status;
  }
  return output;
}

} // namespace

namespace driver {

StatusOr<int> getVersion() {
  return callDriverWithOutput<int>("cuDriverGetVersion",
                                   [](DriverDispatch &dispatch, int &version) {
                                     return dispatch.getDriverVersion(&version);
                                   });
}

StatusOr<CUlibrary> loadLibrary(const void *deviceCode) {
  return callDriverWithOutput<CUlibrary>(
      "cuLibraryLoadData",
      [deviceCode](DriverDispatch &dispatch, CUlibrary &library) {
        return dispatch.libraryLoadData(&library, deviceCode,
                                        /*jitOptions=*/nullptr,
                                        /*jitOptionsValues=*/nullptr,
                                        /*numJitOptions=*/0,
                                        /*libraryOptions=*/nullptr,
                                        /*libraryOptionValues=*/nullptr,
                                        /*numLibraryOptions=*/0);
      });
}

StatusOr<CUkernel> getKernel(CUlibrary library, const char *name) {
  return callDriverWithOutput<CUkernel>(
      "cuLibraryGetKernel",
      [library, name](DriverDispatch &dispatch, CUkernel &kernel) {
        return dispatch.libraryGetKernel(&kernel, library, name);
      });
}

Status unloadLibrary(CUlibrary library) {
  return callDriver("cuLibraryUnload", [library](DriverDispatch &dispatch) {
    return dispatch.libraryUnload(library);
  });
}

Status launchKernel(const CUlaunchConfig &config, CUkernel kernel,
                    void **kernelParams) {
  return callDriver("cuLaunchKernelEx", [&](DriverDispatch &dispatch) {
    return dispatch.launchKernelEx(&config,
                                   reinterpret_cast<CUfunction>(kernel),
                                   kernelParams, /*extra=*/nullptr);
  });
}

} // namespace driver

namespace runtime {

Status initialize() { return free(nullptr); }

StatusOr<int> getDeviceCount() {
  RuntimeDispatch &dispatch = getRuntimeDispatch();
  if (!dispatch.status.ok()) {
    return dispatch.status;
  }
  int count = 0;
  cudaError_t result = dispatch.getDeviceCount(&count);
  if (result != cudaSuccess) {
    dispatch.getLastError();
    return getRuntimeCallStatus(dispatch, result, "cudaGetDeviceCount");
  }
  return count;
}

StatusOr<int> getDevice() {
  return callRuntimeWithOutput<int>("cudaGetDevice",
                                    [](RuntimeDispatch &dispatch, int &device) {
                                      return dispatch.getDevice(&device);
                                    });
}

StatusOr<DeviceProperties> getDeviceProperties(int device) {
  cudaDeviceProp properties{};
  Status status =
      callRuntime("cudaGetDeviceProperties", [&](RuntimeDispatch &dispatch) {
        return dispatch.getDeviceProperties(&properties, device);
      });
  if (!status.ok()) {
    return status;
  }
  DeviceProperties result;
  result.name = properties.name;
  result.major = properties.major;
  result.minor = properties.minor;
  result.multiprocessorCount = properties.multiProcessorCount;
  result.sharedMemoryPerMultiprocessor = properties.sharedMemPerMultiprocessor;
  result.l2CacheSize = properties.l2CacheSize;
  return result;
}

StatusOr<void *> allocate(std::size_t sizeInBytes) {
  return callRuntimeWithOutput<void *>(
      "cudaMalloc", [sizeInBytes](RuntimeDispatch &dispatch, void *&ptr) {
        return dispatch.mallocMemory(&ptr, sizeInBytes);
      });
}

Status free(void *ptr) {
  return callRuntime("cudaFree", [ptr](RuntimeDispatch &dispatch) {
    return dispatch.freeMemory(ptr);
  });
}

Status memset(void *ptr, int value, std::size_t sizeInBytes) {
  return callRuntime("cudaMemset", [&](RuntimeDispatch &dispatch) {
    return dispatch.memsetMemory(ptr, value, sizeInBytes);
  });
}

Status copyHostToDevice(void *dst, const void *src, std::size_t sizeInBytes) {
  return callRuntime("cudaMemcpy host to device",
                     [&](RuntimeDispatch &dispatch) {
                       return dispatch.copyMemory(dst, src, sizeInBytes,
                                                  cudaMemcpyHostToDevice);
                     });
}

Status copyDeviceToHost(void *dst, const void *src, std::size_t sizeInBytes) {
  return callRuntime("cudaMemcpy device to host",
                     [&](RuntimeDispatch &dispatch) {
                       return dispatch.copyMemory(dst, src, sizeInBytes,
                                                  cudaMemcpyDeviceToHost);
                     });
}

Status synchronizeDevice() {
  return callRuntime("cudaDeviceSynchronize", [](RuntimeDispatch &dispatch) {
    return dispatch.deviceSynchronize();
  });
}

StatusOr<Event> createEvent() {
  StatusOr<cudaEvent_t> eventOrErr = callRuntimeWithOutput<cudaEvent_t>(
      "cudaEventCreate", [](RuntimeDispatch &dispatch, cudaEvent_t &event) {
        return dispatch.eventCreate(&event);
      });
  if (!eventOrErr.ok()) {
    return eventOrErr.status();
  }
  return Event{*eventOrErr};
}

Status destroyEvent(Event event) {
  return callRuntime("cudaEventDestroy", [event](RuntimeDispatch &dispatch) {
    return dispatch.eventDestroy(
        reinterpret_cast<cudaEvent_t>(event.getNativeHandle()));
  });
}

Status recordEvent(Event event) {
  return callRuntime("cudaEventRecord", [event](RuntimeDispatch &dispatch) {
    return dispatch.eventRecord(
        reinterpret_cast<cudaEvent_t>(event.getNativeHandle()),
        /*stream=*/nullptr);
  });
}

Status synchronizeEvent(Event event) {
  return callRuntime(
      "cudaEventSynchronize", [event](RuntimeDispatch &dispatch) {
        return dispatch.eventSynchronize(
            reinterpret_cast<cudaEvent_t>(event.getNativeHandle()));
      });
}

StatusOr<float> getElapsedTime(Event start, Event end) {
  return callRuntimeWithOutput<float>(
      "cudaEventElapsedTime",
      [start, end](RuntimeDispatch &dispatch, float &elapsed) {
        return dispatch.eventElapsedTime(
            &elapsed, reinterpret_cast<cudaEvent_t>(start.getNativeHandle()),
            reinterpret_cast<cudaEvent_t>(end.getNativeHandle()));
      });
}

} // namespace runtime
} // namespace mlir::nv_tensor_ir::cuda
