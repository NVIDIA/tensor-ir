// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Artifact/TensorIRArtifactCodec.h"
#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"
#include "tensor_ir/Registration/Registration.h"

#include "mlir/Parser/Parser.h"

#include "llvm/Object/ObjectFile.h"
#include "llvm/TargetParser/Host.h"

#include "NativeHostModule.h"
#include "gtest/gtest.h"

using namespace tensor_ir::rt;
using namespace mlir::nv_tensor_ir;

TEST(NativeHostModuleTest, ExecuteSerializedObjectWithRecordingCallback) {
  mlir::DialectRegistry registry;
  registerDialects(registry);
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module @different_module_name {
      nv_tensor_ir.graph @add(
        %a: tensor<?xf32> {nv_tensor_ir.stride = "(1)"},
        %b: tensor<?xf32> {nv_tensor_ir.stride = "(1)"}
      ) -> (tensor<?xf32> {nv_tensor_ir.stride = "(1)"}) {
        %c = add %a, %b : tensor<?xf32>
        results %c : tensor<?xf32>
      }
    })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  CompilerInvocation options;
  options.codegen.hostTarget = HostTarget::Native;
  options.codegen.hostArch = HostArch::Native;
  options.codegen.tileSize = {4};
  options.deviceTarget.kernelTarget = "sm_100";
  backend::cuda_tile::CudaTileCompiler compiler;
  auto artifact = compiler.compile(*module, options);
  ASSERT_TRUE(artifact.ok()) << artifact.status().message();
  auto encoded = serializeArtifact(**artifact);
  ASSERT_TRUE(encoded.ok());
  artifact = deserializeArtifact((*encoded)->getMemBufferRef());
  ASSERT_TRUE(artifact.ok());
  ASSERT_TRUE((*artifact)->getHostModule());
  HostModule host = *(*artifact)->getHostModule();
  std::string entrySymbol = (*artifact)->getEntrypointName();
  EXPECT_EQ(entrySymbol, "tensor_ir_host_entry_add_dispatch_v1");
  auto loaded = NativeHostModule::load(host, entrySymbol);
  ASSERT_TRUE(loaded.ok()) << loaded.status().message();

  int64_t size = 9, stride = 1;
  int storage[3];
  Any args[] = {TensorView{&storage[0], 1, &size, &stride},
                TensorView{&storage[1], 1, &size, &stride},
                TensorView{&storage[2], 1, &size, &stride}};
  auto packed = packNativeHostArgs((*artifact)->getLaunchMetadata(), {args, 3});
  ASSERT_TRUE(packed.ok()) << packed.status().message();
  ASSERT_EQ(packed->size(), 9u);
  EXPECT_EQ((*packed)[1], size);
  EXPECT_EQ((*packed)[2], stride);
  struct Recording {
    int calls = 0;
    int32_t result = 0;
    int *storage;
  } recording{0, 0, storage};
  TensorIRHostRuntimeV1 runtime{
      sizeof(TensorIRHostRuntimeV1), 1, &recording,
      [](void *data, uint32_t ordinal, const TensorIRLaunchConfigV1 *config,
         void *const *params) -> int32_t {
        auto &record = *static_cast<Recording *>(data);
        ++record.calls;
        EXPECT_EQ(ordinal, 0u);
        EXPECT_EQ(config->grid[0], 3u);
        EXPECT_EQ(config->grid[1], 1u);
        EXPECT_EQ(config->block[0], 1u);
        EXPECT_EQ(config->dynamicSharedMemoryBytes, 0u);
        for (int i = 0; i < 3; ++i) {
          EXPECT_EQ(*static_cast<void **>(params[2 * i]), &record.storage[i]);
          EXPECT_EQ(*static_cast<int32_t *>(params[2 * i + 1]), 9);
        }
        return record.result;
      }};
  EXPECT_EQ((*loaded)->invoke(runtime, *packed), 0);
  EXPECT_EQ(recording.calls, 1);
  recording.result = 17;
  EXPECT_EQ((*loaded)->invoke(runtime, *packed), -3);
  EXPECT_EQ(recording.calls, 2);
  // Identical exported names must coexist in separate dylibs. Removing one
  // module must leave the other's mappings and symbols intact.
  auto second = NativeHostModule::load(host, entrySymbol);
  ASSERT_TRUE(second.ok()) << second.status().message();
  loaded.value().reset();
  EXPECT_EQ((*second)->invoke(runtime, *packed), -3);
  second.value().reset();
  auto incompatibleTriple = host;
  llvm::Triple otherOS(host.triple);
  otherOS.setOS(otherOS.isOSWindows() ? llvm::Triple::Linux
                                      : llvm::Triple::Win32);
  otherOS.setObjectFormat(llvm::Triple::ELF);
  incompatibleTriple.triple = otherOS.str();
  EXPECT_FALSE(NativeHostModule::load(incompatibleTriple, entrySymbol).ok());
  incompatibleTriple.triple.clear();
  EXPECT_FALSE(NativeHostModule::load(incompatibleTriple, entrySymbol).ok());
  EXPECT_TRUE(NativeHostModule::load(host, entrySymbol).ok());
  // ELF e_machine identifies the target architecture; an unknown machine
  // must be rejected before the object can be executed.
  if (host.bytes[0] == '\x7f' && host.bytes[1] == 'E') {
    auto incompatible = host;
    ASSERT_GT(incompatible.bytes.size(), 20u);
    incompatible.bytes[18] = 0;
    incompatible.bytes[19] = 0;
    EXPECT_FALSE(NativeHostModule::load(incompatible, entrySymbol).ok());
    options.codegen.hostArch = HostArch::X86_64;
    options.codegen.hostOS = HostOS::Windows;
    auto windowsArtifact = compiler.compile(*module, options);
    ASSERT_TRUE(windowsArtifact.ok()) << windowsArtifact.status().message();
    if (!llvm::Triple(llvm::sys::getProcessTriple()).isOSWindows()) {
      EXPECT_FALSE(
          NativeHostModule::load(*(*windowsArtifact)->getHostModule(),
                                 (*windowsArtifact)->getEntrypointName())
              .ok());
    }
    const auto &bytes = (*windowsArtifact)->getHostModule()->bytes;
    auto object =
        llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(
            llvm::StringRef(bytes.data(), bytes.size()), "windows"));
    ASSERT_TRUE(static_cast<bool>(object))
        << llvm::toString(object.takeError());
    EXPECT_TRUE((*object)->isELF());
    EXPECT_EQ((*object)->getArch(), llvm::Triple::x86_64);
    for (auto symbol : (*object)->symbols()) {
      auto flags = symbol.getFlags();
      ASSERT_TRUE(static_cast<bool>(flags))
          << llvm::toString(flags.takeError());
      EXPECT_FALSE(*flags & llvm::object::SymbolRef::SF_Undefined);
      if (*flags & llvm::object::SymbolRef::SF_Global) {
        auto name = symbol.getName();
        ASSERT_TRUE(static_cast<bool>(name))
            << llvm::toString(name.takeError());
        EXPECT_EQ(*name, entrySymbol);
      }
    }
    for (auto section : (*object)->sections()) {
      auto name = section.getName();
      ASSERT_TRUE(static_cast<bool>(name)) << llvm::toString(name.takeError());
      EXPECT_NE(*name, ".eh_frame");
    }
  }
  entrySymbol = "missing_entry";
  EXPECT_FALSE(NativeHostModule::load(host, entrySymbol).ok());
  host.bytes = {'b', 'a', 'd'};
  EXPECT_FALSE(NativeHostModule::load(host, entrySymbol).ok());
  host.abiVersion = 2;
  EXPECT_FALSE(NativeHostModule::load(host, entrySymbol).ok());
}

TEST(NativeHostModuleTest, ValidatesAndPacksTensorLayouts) {
  KernelMetadataBuilder builder;
  builder.args.push_back(
      TensorArgDesc::create(ElementType::F32, {2, 3}).value());
  builder.numInputs = 1;
  builder.tileSizes = {1, 1};
  auto metadata = KernelMetadata::create(std::move(builder));
  ASSERT_TRUE(metadata.ok());
  int storage;
  int64_t shape[] = {2, 3};
  Any value(TensorView{&storage, 2, shape, nullptr});
  auto packed = packNativeHostArgs(*metadata, {&value, 1});
  ASSERT_TRUE(packed.ok());
  EXPECT_EQ(*packed, (llvm::SmallVector<int64_t>{
                         reinterpret_cast<intptr_t>(&storage), 2, 3, 3, 1}));
  Any pointer(&storage);
  EXPECT_EQ(*packNativeHostArgs(*metadata, {&pointer, 1}), *packed);
  EXPECT_FALSE(packNativeHostArgs(*metadata, {&value, 0}).ok());
  shape[0] = 4;
  EXPECT_FALSE(packNativeHostArgs(*metadata, {&value, 1}).ok());
  shape[0] = 2;
  value.tensorView.ndim = 1;
  EXPECT_FALSE(packNativeHostArgs(*metadata, {&value, 1}).ok());
  value.tensorView.ndim = 2;
  int64_t strides[] = {1, 2};
  value.tensorView.stride = strides;
  EXPECT_FALSE(packNativeHostArgs(*metadata, {&value, 1}).ok());
  int64_t si64Value = 1;
  value = Any::makeScalarAny(ElementType::SI64, &si64Value);
  EXPECT_FALSE(packNativeHostArgs(*metadata, {&value, 1}).ok());
}

TEST(NativeHostModuleTest, PacksScalarArguments) {
  KernelMetadataBuilder builder;
  builder.args.push_back(TensorArgDesc::create(ElementType::F32, {2}).value());
  builder.args.push_back(ScalarArgDesc::create(ElementType::SI32).value());
  builder.numInputs = 2;
  builder.tileSizes = {1};
  auto metadata = KernelMetadata::create(std::move(builder));
  ASSERT_TRUE(metadata.ok());

  int storage;
  int64_t shape[] = {2};
  int32_t si32Value = 42;
  Any args[] = {Any(TensorView{&storage, 1, shape, nullptr}),
                Any::makeScalarAny(ElementType::SI32, &si32Value)};
  auto packed = packNativeHostArgs(*metadata, {args, 2});
  ASSERT_TRUE(packed.ok());
  EXPECT_EQ(*packed, (llvm::SmallVector<int64_t>{
                         reinterpret_cast<intptr_t>(&storage), 2, 1, 42}));

  // Wrong element type for the declared scalar slot is rejected.
  int64_t si64Value = 42;
  args[1] = Any::makeScalarAny(ElementType::SI64, &si64Value);
  EXPECT_FALSE(packNativeHostArgs(*metadata, {args, 2}).ok());

  // A tensor-tagged Any in a scalar slot is rejected, not silently packed.
  args[1] = Any(TensorView{&storage, 1, shape, nullptr});
  EXPECT_FALSE(packNativeHostArgs(*metadata, {args, 2}).ok());
}

TEST(NativeHostModuleTest, AcceptsEquivalentSingletonStrides) {
  for (bool explicitStrides : {false, true}) {
    KernelMetadataBuilder builder;
    int64_t compiledStrides[] = {1, 1};
    builder.args.push_back(
        TensorArgDesc::create(
            ElementType::F32, {1, 3},
            explicitStrides
                ? std::optional<llvm::ArrayRef<int64_t>>(compiledStrides)
                : std::nullopt)
            .value());
    builder.numInputs = 1;
    builder.tileSizes = {1, 1};
    auto metadata = KernelMetadata::create(std::move(builder));
    ASSERT_TRUE(metadata.ok());
    int storage;
    int64_t shape[] = {1, 3};
    int64_t strides[] = {7, 1};
    Any value(TensorView{&storage, 2, shape, strides});
    auto packed = packNativeHostArgs(*metadata, {&value, 1});
    ASSERT_TRUE(packed.ok());
    EXPECT_EQ(*packed, (llvm::SmallVector<int64_t>{
                           reinterpret_cast<intptr_t>(&storage), 1, 3, 7, 1}));
    strides[1] = 2;
    EXPECT_FALSE(packNativeHostArgs(*metadata, {&value, 1}).ok());
    strides[1] = 1;
    strides[0] = -1;
    EXPECT_FALSE(packNativeHostArgs(*metadata, {&value, 1}).ok());
  }
}
