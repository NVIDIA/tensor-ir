// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Options/OptionsEnums.h"
#include "tensor_ir/Transform/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

namespace mlir::nv_tensor_ir {
#define GEN_PASS_DEF_HOSTTOOBJECTPASS
#include "tensor_ir/Transform/Passes.h.inc"
} // namespace mlir::nv_tensor_ir

using namespace mlir;
using namespace mlir::nv_tensor_ir;

/// Returns the LLVM triple for the given host architecture and operating system.
static llvm::Triple getHostTriple(HostArch hostArch, HostOS hostOS) {
  llvm::Triple triple;
  if (hostArch == HostArch::Native) {
    triple = llvm::Triple(llvm::sys::getProcessTriple());
  } else {
    triple.setArch(hostArch == HostArch::X86_64 ? llvm::Triple::x86_64
                                                : llvm::Triple::aarch64);
    triple.setVendor(llvm::Triple::UnknownVendor);
    triple.setOS(hostOS == HostOS::Windows ? llvm::Triple::Win32
                                           : llvm::Triple::Linux);
    llvm::Triple process(llvm::sys::getProcessTriple());
    triple.setEnvironment(
        triple.getOS() == process.getOS()
            ? process.getEnvironment()
            : (triple.isOSWindows() ? llvm::Triple::MSVC : llvm::Triple::GNU));
  }
  // Match TileIR host functions: ELF avoids COFF relocation issues on Windows.
  triple.setObjectFormat(llvm::Triple::ELF);
  return triple;
}

/// Creates a native target machine for the given triple.
static FailureOr<std::unique_ptr<llvm::TargetMachine>>
createNativeTargetMachine(ModuleOp module, const llvm::Triple &triple) {
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    return module.emitError("native host target: ") << error;
  }
  llvm::TargetOptions options;
  options.UseInitArray = true;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      triple, "generic", "", options, llvm::Reloc::PIC_, llvm::CodeModel::Small,
      llvm::CodeGenOptLevel::Aggressive, false));
  if (!machine) {
    return module.emitError("cannot create native host TargetMachine");
  }
  return machine;
}

/// Filters out 'nv_tensor_ir' dialect attributes from the argument attributes
/// of an LLVM function argument.
static SmallVector<NamedAttribute> filterArgAttrs(DictionaryAttr attrs) {
  SmallVector<NamedAttribute> result;
  for (auto attr : attrs) {
    if (!attr.getName().strref().starts_with(
            TensorIRDialect::getDialectNamespace())) {
      result.push_back(attr);
    }
  }
  return result;
}

/// Assemble a detached MLIR ModuleOp containing just the LLVM operations, which
/// are basically just functions but in the future could contain globals as
/// well. This is only required because our ModuleOp may contain unrelated
/// nested modules (e.g. 'cuda_tile.module'). If 'dropHostIR' is true, we move
/// the ops over. Otherwise, we clone them. We also filter out any
/// no-longer-relevant attributes that may cause the LLVMIR translation to emit
/// warnings.
OwningOpRef<ModuleOp>
assembleDetachedLLVMModule(IRRewriter &b, ModuleOp module, bool dropHostIR,
                           const llvm::TargetMachine &machine) {
  ModuleOp translationModule = ModuleOp::create(module.getLoc());
  b.setInsertionPointToStart(translationModule.getBody());
  Block *body = translationModule.getBody();

  // Assemble a detached MLIR ModuleOp containing just the LLVM functions.
  // If 'dropHostIR' is true, we move the functions over. Otherwise, we clone them.
  // We also filter out any no-longer-relevant attributes that may cause the
  // LLVMIR translation to emit warnings.
  for (Operation &op : llvm::make_early_inc_range(module.getOps())) {
    if (!isa<LLVM::LLVMDialect>(op.getDialect())) {
      continue;
    }
    if (auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(op)) {
      auto argAttrs = llvmFunc.getAllArgAttrs();
      if (argAttrs) {
        auto argAttributes = llvm::map_to_vector(
            argAttrs, [&](Attribute attrdict) -> DictionaryAttr {
              return b.getDictionaryAttr(
                  filterArgAttrs(cast<DictionaryAttr>(attrdict)));
            });
        llvmFunc.setAllArgAttrs(argAttributes);
      }
    }
    b.setInsertionPointToEnd(body);
    if (dropHostIR) {
      b.moveOpBefore(&op, b.getInsertionBlock(), b.getInsertionPoint());
    } else {
      b.clone(op);
    }
  }

  auto entries = module->getAttrOfType<ArrayAttr>(
      TensorIRDialect::getNativeEntrypointsAttrName());
  for (auto function : translationModule.getOps<LLVM::LLVMFuncOp>()) {
    if (function.getBody().empty()) {
      continue;
    }
    if (!entries ||
        !llvm::is_contained(entries, b.getStringAttr(function.getName()))) {
      function.setLinkage(LLVM::Linkage::Internal);
    }
    function.setNoUnwind(true);
    function.setFramePointerAttr(LLVM::FramePointerKindAttr::get(
        b.getContext(), LLVM::framePointerKind::FramePointerKind::All));
  }

  // Populate the target triple and data layout attributes.
  translationModule->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                             b.getStringAttr(machine.getTargetTriple().str()));
  translationModule->setAttr(
      LLVM::LLVMDialect::getDataLayoutAttrName(),
      b.getStringAttr(machine.createDataLayout().getStringRepresentation()));

  return translationModule;
}

/// Run the LLVM code generation pass to emit the LLVM IR as a native ELF object
/// file.
FailureOr<std::unique_ptr<llvm::SmallVectorMemoryBuffer>>
runLLVMCodegenToObject(Location loc, llvm::Module &module,
                       llvm::TargetMachine &machine) {
  SmallVector<char, 0> objectBytes;
  llvm::raw_svector_ostream stream(objectBytes);
  llvm::legacy::PassManager codegen;
  if (machine.addPassesToEmitFile(codegen, stream, nullptr,
                                  llvm::CodeGenFileType::ObjectFile)) {
    return emitError(loc)
           << "native host target cannot emit relocatable objects";
  }
  codegen.run(module);
  auto buffer = std::make_unique<llvm::SmallVectorMemoryBuffer>(
      std::move(objectBytes), module.getModuleIdentifier(),
      /*RequiresNullTerminator=*/false);
  auto object =
      llvm::object::ObjectFile::createObjectFile(buffer->getMemBufferRef());
  if (!object) {
    return emitError(loc) << "invalid native host object: "
                          << llvm::toString(object.takeError());
  }
  // The host ABI calls through the runtime table; no external imports are allowed.
  for (auto symbol : (*object)->symbols()) {
    auto flags = symbol.getFlags();
    if (!flags) {
      return emitError(loc) << llvm::toString(flags.takeError());
    }
    if (*flags & llvm::object::SymbolRef::SF_Undefined) {
      auto name = symbol.getName();
      if (!name) {
        return emitError(loc) << llvm::toString(name.takeError());
      }
      return emitError(loc) << "unsupported native host import: " << *name;
    }
  }
  return buffer;
}

/// Construct a DenseI8ResourceElementsAttr from a unique_ptr<MemoryBuffer>.
/// The AsmBlobResource takes ownership of the MemoryBuffer; it's lifetime
/// will outlive any references from the top-level ModuleOp.
static DenseI8ResourceElementsAttr constructDenseI8ResourceElementsAttr(
    Location loc, std::unique_ptr<llvm::MemoryBuffer> objBuffer,
    const llvm::Triple &triple) {
  Type i8Type = IntegerType::get(loc.getContext(), 8);

  std::string name =
      llvm::formatv("object_{0}_{1}_{2}", triple.getArchName(),
                    triple.getOSName(), objBuffer->getBufferIdentifier());
  ArrayRef<char> data = objBuffer->getBuffer();
  return DenseI8ResourceElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(data.size())}, i8Type), name,
      AsmResourceBlob(
          data,
          [objBuffer = std::move(objBuffer)](void *, size_t, size_t) {
            (void)objBuffer;
          },
          /*dataIsMutable=*/false));
}

namespace {

struct HostToObjectPass
    : nv_tensor_ir::impl::HostToObjectPassBase<HostToObjectPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::Triple triple = getHostTriple(hostArch, hostOS);

    FailureOr<std::unique_ptr<llvm::TargetMachine>> machine =
        createNativeTargetMachine(module, triple);
    if (failed(machine)) {
      return signalPassFailure();
    }

    // Assemble a detached MLIR ModuleOp containing just the LLVM operations.
    IRRewriter b(module);
    OwningOpRef<ModuleOp> translationModule =
        assembleDetachedLLVMModule(b, module, dropHostIR, **machine);

    // Translate the detached MLIR ModuleOp to LLVM IR and verify it.
    llvm::LLVMContext context;
    auto ir = translateModuleToLLVMIR(*translationModule, context);
    if (!ir) {
      module.emitError("native host LLVM IR translation failed");
      return signalPassFailure();
    }
    std::string verifierOutput;
    llvm::raw_string_ostream os(verifierOutput);
    if (llvm::verifyModule(*ir, &os)) {
      os.flush();
      emitError(module.getLoc())
          << "LLVM module verification failed: " << verifierOutput;
      return signalPassFailure();
    }

    FailureOr<std::unique_ptr<llvm::SmallVectorMemoryBuffer>> objBuffer =
        runLLVMCodegenToObject(module.getLoc(), *ir, **machine);
    if (failed(objBuffer)) {
      return signalPassFailure();
    }
    DenseI8ResourceElementsAttr resourceAttr =
        constructDenseI8ResourceElementsAttr(module.getLoc(),
                                             std::move(*objBuffer), triple);

    module->setAttr(TensorIRDialect::getNativeObjectAttrName(), resourceAttr);
    module->setAttr(TensorIRDialect::getNativeTripleAttrName(),
                    b.getStringAttr((*machine)->getTargetTriple().str()));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);

    static std::once_flag initFlag;
    std::call_once(initFlag, [] {
#ifdef TENSOR_IR_HOST_TARGET_X86
      LLVMInitializeX86TargetInfo();
      LLVMInitializeX86Target();
      LLVMInitializeX86TargetMC();
      LLVMInitializeX86AsmPrinter();
#endif
#ifdef TENSOR_IR_HOST_TARGET_AArch64
      LLVMInitializeAArch64TargetInfo();
      LLVMInitializeAArch64Target();
      LLVMInitializeAArch64TargetMC();
      LLVMInitializeAArch64AsmPrinter();
#endif
    });
  }
};
} // namespace
