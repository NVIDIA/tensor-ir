// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Constraint.h"
#include "mlir/TableGen/EnumInfo.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <cassert>
#include <optional>
#include <string>
#include <utility>
#include <vector>
using namespace llvm;
using mlir::tblgen::Constraint;
using mlir::tblgen::EnumCase;
using mlir::tblgen::EnumInfo;
using mlir::tblgen::escapeString;
using mlir::tblgen::FmtContext;
using mlir::tblgen::tgfmt;
static constexpr StringLiteral kCppNamespace = "::mlir::nv_tensor_ir";
static std::string makeIdentifier(StringRef str) {
  if (!str.empty() && isDigit(static_cast<unsigned char>(str.front()))) {
    return (Twine("_") + str).str();
  }
  return str.str();
}
static bool isValidFlagName(StringRef s) {
  if (s.empty() || !isAlpha(s[0])) {
    return false;
  }
  return all_of(s, [](char c) { return isAlnum(c) || c == '-'; });
}
static StringRef getCodeTemplate(const Record *record, StringRef field) {
  StringRef text = record->getValueAsString(field);
  text.consume_front("\n");
  return text.rtrim(" \t");
}
static bool isNumericCppType(StringRef cppType) {
  return cppType == "unsigned" || cppType == "int" || cppType == "int32_t" ||
         cppType == "int64_t" || cppType == "uint64_t" || cppType == "size_t" ||
         cppType == "float" || cppType == "double";
}
static std::optional<StringRef> capiScalarType(StringRef cppType) {
  return StringSwitch<std::optional<StringRef>>(cppType)
      .Cases({"bool", "unsigned"}, cppType)
      .Cases({"int", "int32_t"}, "int32_t")
      .Case("int64_t", "int64_t")
      .Cases({"uint64_t", "size_t"}, "uint64_t")
      .Cases({"float", "double"}, cppType)
      .Case("std::string", "MlirStringRef")
      .Default(std::nullopt);
}
static void diagnose(bool &anyError, ArrayRef<SMLoc> loc,
                     const Twine &message) {
  PrintError(loc, message);
  anyError = true;
}
namespace {
struct OptionInfo {
  enum class Kind { Scalar, Enum, List };
  explicit OptionInfo(const Record *record)
      : rec(record), varName(record->getValueAsString("varName")),
        argName(record->getValueAsString("argName")),
        cppType(record->getValueAsString("cppType")),
        defaultValue(record->getValueAsString("defaultValue")),
        helpText(record->getValueAsString("helpText")),
        emitTypedCAPI(record->getValueAsBit("emitTypedCAPI")) {
    if (rec->isSubClassOf("EnumOption")) {
      kind = Kind::Enum;
      enumInfo.emplace(rec->getValueAsDef("enumInfo"));
      capiType = rec->getValueAsString("capiType");
    } else if (rec->isSubClassOf("ListOption")) {
      kind = Kind::List;
      elementType = rec->getValueAsString("elementType");
      delimiter = rec->getValueAsString("delimiter");
    }
  }
  const Record *rec;
  Kind kind = Kind::Scalar;
  StringRef varName;
  StringRef argName;
  StringRef cppType;
  StringRef defaultValue;
  StringRef helpText;
  bool emitTypedCAPI = true;
  std::optional<EnumInfo> enumInfo;
  StringRef capiType;
  StringRef elementType;
  StringRef delimiter;
  std::string fullDefaultExpr() const {
    return kind == Kind::Enum
               ? (Twine(cppType) + "::" + makeIdentifier(defaultValue)).str()
               : defaultValue.str();
  }
  std::string delimiterCharLiteral() const {
    return delimiter.empty() ? "" : (Twine("'") + delimiter + "'").str();
  }
};
struct ComponentInfo {
  explicit ComponentInfo(const Record *record)
      : rec(record), name(record->getValueAsString("componentName")),
        clCategory(record->getValueAsString("clCategory").str()),
        invocationMember(record->getValueAsString("invocationMember")),
        verifyFunction(record->getValueAsString("verifyFunction")) {
    if (clCategory.empty()) {
      clCategory = (name + " Options").str();
    }
  }
  const Record *rec;
  StringRef name;
  std::string clCategory;
  StringRef invocationMember;
  std::vector<OptionInfo> options;
  StringRef verifyFunction;
  std::string structName() const { return (name + "Options").str(); }
  std::string qualStructName() const {
    return (Twine(kCppNamespace) + "::" + structName()).str();
  }
  std::string declStructName() const {
    return (Twine(kCppNamespace.drop_front(2)) + "::" + structName()).str();
  }
  std::string capiEnumType(const OptionInfo &opt) const {
    if (!opt.capiType.empty()) {
      return opt.capiType.str();
    }
    return (Twine("MlirTensorIR") + name + opt.cppType).str();
  }
  std::string clStructName() const { return (name + "CLOptions").str(); }
  std::string clStaticName() const { return ("cl" + name + "Options").str(); }
  std::string clRegisterFnName() const {
    return ("register" + name + "CLOptions").str();
  }
  std::string clCategoryFnName() const {
    return ("get" + name + "CLOptionCategory").str();
  }
  std::string clGetterFnName() const {
    return ("get" + name + "OptionsFromCL").str();
  }
  std::string parserAliasName() const { return (name + "Opts").str(); }
};
} // namespace
static LogicalResult validateListDelimiter(const OptionInfo &opt) {
  ArrayRef<SMLoc> loc = opt.rec->getLoc();
  StringRef delim = opt.delimiter;
  if (!isNumericCppType(opt.elementType)) {
    PrintError(loc, "list option '" + opt.argName +
                        "' requires a numeric element type; elements are '" +
                        opt.elementType + "'");
    return failure();
  }
  if (delim == "," || delim == "x") {
    return success();
  }
  PrintError(loc, "delimiter '" + delim + "' of list option '" + opt.argName +
                      "' must be ',' or numeric-list delimiter 'x'");
  return failure();
}
static LogicalResult validateOption(const OptionInfo &opt) {
  bool anyError = false;
  ArrayRef<SMLoc> loc = opt.rec->getLoc();
  if (!isValidFlagName(opt.argName)) {
    diagnose(anyError, loc,
             "argName '" + opt.argName + "' must match [a-zA-Z][a-zA-Z0-9-]*");
  }
  if (opt.kind == OptionInfo::Kind::List) {
    anyError |= failed(validateListDelimiter(opt));
  }
  if (opt.emitTypedCAPI && opt.cppType == "std::string" &&
      !opt.rec->isSubClassOf("StringOption")) {
    diagnose(anyError, loc,
             "option '" + opt.argName +
                 "' uses generic std::string C API conversions; use "
                 "StringOption instead");
  }
  if (opt.emitTypedCAPI && opt.kind == OptionInfo::Kind::Scalar &&
      !capiScalarType(opt.cppType)) {
    diagnose(anyError, loc,
             "option '" + opt.argName + "' has C++ type '" + opt.cppType +
                 "' with no C mapping; set emitTypedCAPI = 0 to expose it via "
                 "SetFromString only");
  }
  return failure(anyError);
}
static FailureOr<std::vector<ComponentInfo>>
buildAndValidateAll(const RecordKeeper &records) {
  std::vector<ComponentInfo> comps;
  bool anyError = false;
  StringMap<StringRef> componentNames;
  StringMap<StringRef> globalFlags;
  for (const Record *rec :
       records.getAllDerivedDefinitions("ComponentOptions")) {
    ComponentInfo comp(rec);
    auto [componentIt, componentInserted] =
        componentNames.try_emplace(comp.name, rec->getName());
    if (!componentInserted) {
      diagnose(anyError, rec->getLoc(),
               "component name '" + comp.name + "' is used by both '" +
                   componentIt->second + "' and '" + rec->getName() + "'");
    }
    StringSet<> optionNames;
    for (const Record *optRec : rec->getValueAsListOfDefs("options")) {
      OptionInfo &opt = comp.options.emplace_back(optRec);
      anyError |= failed(validateOption(opt));
      if (!optionNames.insert(opt.varName).second) {
        diagnose(anyError, opt.rec->getLoc(),
                 "option field '" + opt.varName +
                     "' occurs more than once in component '" + comp.name +
                     "'");
      }
    }
    for (const OptionInfo &opt : comp.options) {
      auto [it, inserted] = globalFlags.try_emplace(opt.argName, comp.name);
      if (!inserted) {
        diagnose(anyError, opt.rec->getLoc(),
                 "CLI flag '--" + opt.argName +
                     "' declared by both component '" + it->second + "' and '" +
                     comp.name + "'");
      }
    }
    comps.push_back(std::move(comp));
  }
  if (anyError) {
    return failure();
  }
  return comps;
}
static FailureOr<SmallVector<const ComponentInfo *>>
buildInvocationComponents(const RecordKeeper &records,
                          ArrayRef<ComponentInfo> comps) {
  ArrayRef<const Record *> configs =
      records.getAllDerivedDefinitions("CompilerInvocationOptions");
  if (configs.size() != 1) {
    PrintError("expected exactly one CompilerInvocationOptions definition");
    return failure();
  }
  bool anyError = false;
  StringSet<> listedComponents;
  StringMap<StringRef> invocationMembers;
  StringMap<std::string> capiAccessors;
  StringMap<const Record *> capiEnums;
  SmallVector<const ComponentInfo *> invocationComps;
  for (const Record *componentRec :
       configs.front()->getValueAsListOfDefs("components")) {
    auto it = llvm::find_if(comps, [&](const ComponentInfo &comp) {
      return comp.rec == componentRec;
    });
    assert(it != comps.end() && "TableGen type guarantees a component record");
    if (!listedComponents.insert(it->name).second) {
      diagnose(anyError, componentRec->getLoc(),
               "component '" + it->name +
                   "' occurs more than once in CompilerInvocationOptions");
      continue;
    }
    if (it->invocationMember.empty()) {
      diagnose(anyError, componentRec->getLoc(),
               "component '" + it->name +
                   "' is listed in CompilerInvocationOptions but has no "
                   "invocationMember");
      continue;
    }
    auto [memberIt, memberInserted] =
        invocationMembers.try_emplace(it->invocationMember, it->name);
    if (!memberInserted) {
      diagnose(anyError, componentRec->getLoc(),
               "invocation member '" + it->invocationMember +
                   "' is used by both components '" + memberIt->second +
                   "' and '" + it->name + "'");
    }
    invocationComps.push_back(&*it);
    for (const OptionInfo &opt : it->options) {
      if (!opt.emitTypedCAPI) {
        continue;
      }
      std::string accessor =
          convertToCamelFromSnakeCase(opt.varName, /*capitalizeFirst=*/true);
      std::string member =
          (Twine(it->invocationMember) + "." + opt.varName).str();
      auto [accessorIt, inserted] = capiAccessors.try_emplace(accessor, member);
      if (!inserted) {
        diagnose(anyError, opt.rec->getLoc(),
                 "typed C API accessor '" + accessor +
                     "' is generated by both '" + accessorIt->second +
                     "' and '" + member + "'");
      }
      if (opt.kind == OptionInfo::Kind::Enum) {
        std::string type = it->capiEnumType(opt);
        const Record *enumDef = opt.rec->getValueAsDef("enumInfo");
        auto [enumIt, enumInserted] = capiEnums.try_emplace(type, enumDef);
        if (!enumInserted && enumIt->second != enumDef) {
          diagnose(anyError, opt.rec->getLoc(),
                   "C API enum type '" + type + "' is generated from both '" +
                       enumIt->second->getName() + "' and '" +
                       enumDef->getName() + "'");
        }
      }
    }
  }
  for (const ComponentInfo &comp : comps) {
    if (!comp.invocationMember.empty() &&
        !listedComponents.contains(comp.name)) {
      diagnose(anyError, comp.rec->getLoc(),
               "component '" + comp.name +
                   "' has an invocationMember but is not listed in "
                   "CompilerInvocationOptions");
    }
  }
  if (anyError) {
    return failure();
  }
  return invocationComps;
}
namespace {
class ComponentEmitter {
public:
  ComponentEmitter(const ComponentInfo &comp, raw_ostream &os)
      : comp(comp), os(os) {}
  void emitCLBindings();
  void emitOptionsStruct();
  void emitParserDefs();
  void emitParseOption(const OptionInfo &opt);
  void emitValidateImpl();
  void emitCLOption(const OptionInfo &opt);
  void emitCLAccessorDefs();
  void emitCLAssignment(const OptionInfo &opt);

private:
  const ComponentInfo &comp;
  raw_ostream &os;
};
} // namespace
void ComponentEmitter::emitOptionsStruct() {
  os << formatv("struct {0} {{\n", comp.structName());
  for (const OptionInfo &opt : comp.options) {
    if (!opt.helpText.empty()) {
      os << formatv("  /// {0}\n", opt.helpText);
    }
    os << formatv("  {0} {1} = {2};\n", opt.cppType, opt.varName,
                  opt.fullDefaultExpr());
  }
  os << "\n  [[nodiscard]] std::optional<std::string> validate() const;\n";
  os << "};\n";
}
void ComponentEmitter::emitParseOption(const OptionInfo &opt) {
  os << formatv("  if (key == \"{0}\") {{\n    consumed = true;\n",
                opt.argName);
  if (opt.kind == OptionInfo::Kind::Enum) {
    const EnumInfo &enumInfo = *opt.enumInfo;
    os << formatv("    if (auto parsed = {0}::{1}(value)) {\n"
                  "      opts.{2} = *parsed;\n"
                  "      return llvm::success();\n"
                  "    }\n"
                  "    return llvm::failure();\n",
                  enumInfo.getCppNamespace(),
                  enumInfo.getStringToSymbolFnName(), opt.varName);
  } else {
    StringRef format = getCodeTemplate(opt.rec, "parse");
    FmtContext ctx;
    ctx.addSubst("varName", opt.varName)
        .addSubst("elemType", opt.elementType)
        .addSubst("delim", escapeString(opt.delimiter))
        .addSubst("delimChar", opt.delimiterCharLiteral());
    os << tgfmt(format, &ctx);
  }
  os << "  }\n";
}
void ComponentEmitter::emitParserDefs() {
  os << formatv(R"cpp(using {0} = {1};
static llvm::LogicalResult parse{0}Option(
    {0} &opts, llvm::StringRef key, llvm::StringRef value,
    bool &consumed) {{
)cpp",
                comp.parserAliasName(), comp.qualStructName());
  for (const OptionInfo &opt : comp.options) {
    emitParseOption(opt);
  }
  os << "  return llvm::success();\n}\n\n";
}
void ComponentEmitter::emitValidateImpl() {
  os << formatv("std::optional<std::string>\n{0}::validate() const {{\n",
                comp.declStructName());
  for (const OptionInfo &opt : comp.options) {
    FmtContext ctx;
    ctx.withSelf(opt.varName);
    for (const Record *traitRec : opt.rec->getValueAsListOfDefs("traits")) {
      Constraint trait(traitRec);
      std::string condition = trait.getConditionTemplate();
      os << formatv(R"(  if (!{0}) {
    if constexpr (std::is_same_v<decltype({2}), bool>)
      return std::string("{3}");
    std::string error = "{1}: {3} (got ";
    llvm::raw_string_ostream(error) << {2} << ')';
    return error;
  }
)",
                    tgfmt(condition, &ctx), opt.argName, opt.varName,
                    escapeString(trait.getSummary()));
    }
  }
  if (!comp.verifyFunction.empty()) {
    os << "  if (auto status = " << comp.verifyFunction
       << "(*this); !status.ok()) {\n";
    os << "    return status.message();\n";
    os << "  }\n";
  }
  os << "  return std::nullopt;\n}\n";
}
void ComponentEmitter::emitCLBindings() {
  os << "namespace {\n" << formatv("struct {0} {{\n", comp.clStructName());
  os << formatv("  llvm::cl::OptionCategory category{{\"{0}\"};\n\n",
                escapeString(comp.clCategory));
  for (const OptionInfo &opt : comp.options) {
    emitCLOption(opt);
  }
  os << "};\n} // namespace\n";
  os << "\n";
  os << formatv("static llvm::ManagedStatic<{0}> {1};\n\n", comp.clStructName(),
                comp.clStaticName());
  emitCLAccessorDefs();
}
void ComponentEmitter::emitCLOption(const OptionInfo &opt) {
  if (opt.kind == OptionInfo::Kind::Enum) {
    os << formatv("  llvm::cl::opt<{0}::{1}> {2}{{\"{3}\", "
                  "llvm::cl::desc(\"{4}\"),\n"
                  "      llvm::cl::init({0}::{1}::{5}),\n"
                  "      llvm::cl::values(\n",
                  kCppNamespace, opt.cppType, opt.varName, opt.argName,
                  escapeString(opt.helpText), makeIdentifier(opt.defaultValue));
    interleave(
        opt.enumInfo->getAllCases(), os,
        [&](const EnumCase &c) {
          os << formatv("          clEnumValN({0}::{1}::{2}, \"{3}\", "
                        "\"{4}\")",
                        kCppNamespace, opt.cppType,
                        makeIdentifier(c.getSymbol()), c.getStr(),
                        escapeString(c.getDef().getValueAsString("help")));
        },
        ",\n");
    os << "),\n      llvm::cl::cat(category)};\n";
    return;
  }
  StringRef format = getCodeTemplate(opt.rec, "clDeclaration");
  FmtContext ctx;
  ctx.addSubst("cppType", opt.cppType)
      .addSubst("varName", opt.varName)
      .addSubst("argName", opt.argName)
      .addSubst("helpText", escapeString(opt.helpText))
      .addSubst("defaultValue", opt.defaultValue)
      .addSubst("elemType", opt.elementType);
  os << tgfmt(format, &ctx);
}
void ComponentEmitter::emitCLAccessorDefs() {
  os << formatv("void mlir::nv_tensor_ir::{0}() {{ *{1}; }\n\n",
                comp.clRegisterFnName(), comp.clStaticName());
  os << formatv("llvm::cl::OptionCategory &mlir::nv_tensor_ir::{0}() {{\n"
                "  return {1}->category;\n"
                "}\n\n",
                comp.clCategoryFnName(), comp.clStaticName());
  os << formatv("void mlir::nv_tensor_ir::apply{0}OptionsFromCL({1} &opts) "
                "{{\n",
                comp.name, comp.qualStructName());
  for (const OptionInfo &opt : comp.options) {
    emitCLAssignment(opt);
  }
  os << "}\n\n";
  os << formatv("{0} mlir::nv_tensor_ir::{1}() {{\n", comp.qualStructName(),
                comp.clGetterFnName());
  os << formatv("  {0} opts;\n", comp.qualStructName());
  os << formatv("  apply{0}OptionsFromCL(opts);\n", comp.name);
  os << "  return opts;\n}\n";
}
void ComponentEmitter::emitCLAssignment(const OptionInfo &opt) {
  os << formatv("  if ({0}->{1}.getNumOccurrences())\n", comp.clStaticName(),
                opt.varName);
  StringRef format = getCodeTemplate(opt.rec, "clAssignment");
  FmtContext ctx;
  ctx.addSubst("varName", opt.varName)
      .addSubst("clStatic", comp.clStaticName())
      .addSubst("delimChar", opt.delimiterCharLiteral())
      .addSubst("argName", opt.argName);
  os << tgfmt(format, &ctx);
}
static void emitPrintOption(raw_ostream &os, const OptionInfo &opt,
                            StringRef member, bool first) {
  os << formatv("    os << \"{0}{1}=\";\n", first ? "" : " ", opt.argName);
  if (opt.kind == OptionInfo::Kind::Enum) {
    const EnumInfo &enumInfo = *opt.enumInfo;
    os << formatv("    os << {0}::{1}({2});\n", enumInfo.getCppNamespace(),
                  enumInfo.getSymbolToStringFnName(), member);
    return;
  }
  StringRef format = getCodeTemplate(opt.rec, "print");
  FmtContext ctx;
  ctx.addSubst("member", member)
      .addSubst("elemType", opt.elementType)
      .addSubst("delim", escapeString(opt.delimiter))
      .addSubst("delimChar", opt.delimiterCharLiteral());
  os << tgfmt(format, &ctx);
}
static void emitInvocationComponents(ArrayRef<const ComponentInfo *> comps,
                                     raw_ostream &os) {
  os << "struct CompilerInvocationComponents {\n";
  for (const ComponentInfo *comp : comps) {
    os << formatv("  {0} {1};\n", comp->structName(), comp->invocationMember);
  }
  os << "\n  [[nodiscard]] llvm::LogicalResult "
        "parseFromString(llvm::StringRef str);\n"
        "  void print(llvm::raw_ostream &os) const {\n"
        "    // Keep the schema order stable: this is also the canonical hash "
        "input.\n";
  bool first = true;
  for (const ComponentInfo *comp : comps) {
    for (const OptionInfo &opt : comp->options) {
      emitPrintOption(
          os, opt,
          (Twine("this->") + comp->invocationMember + "." + opt.varName).str(),
          first);
      first = false;
    }
  }
  os << "  }\n"
        "  [[nodiscard]] std::string toString() const {\n"
        "    std::string s;\n"
        "    llvm::raw_string_ostream os(s);\n"
        "    print(os);\n"
        "    return s;\n"
        "  }\n"
        "  [[nodiscard]] std::optional<std::string> "
        "validateComponents() const {\n";
  for (const ComponentInfo *comp : comps) {
    os << formatv("    if (auto error = {0}.validate())\n"
                  "      return error;\n",
                  comp->invocationMember);
  }
  os << "    return std::nullopt;\n  }\n"
        "  friend bool operator==(const CompilerInvocationComponents &a,\n"
        "                         const CompilerInvocationComponents &b) {\n"
        "    return true";
  for (const ComponentInfo *comp : comps) {
    for (const OptionInfo &opt : comp->options) {
      os << formatv(" && a.{0}.{1} == b.{0}.{1}", comp->invocationMember,
                    opt.varName);
    }
  }
  os << ";\n  }\n"
        "  friend bool operator!=(const CompilerInvocationComponents &a,\n"
        "                         const CompilerInvocationComponents &b) {\n"
        "    return !(a == b);\n  }\n};\n";
}
static void emitInvocationParserDef(ArrayRef<const ComponentInfo *> comps,
                                    raw_ostream &os) {
  os << R"cpp(llvm::LogicalResult
mlir::nv_tensor_ir::CompilerInvocationComponents::parseFromString(
    llvm::StringRef str) {
  CompilerInvocationComponents parsed = *this;
  if (llvm::failed(optd::forEachOption(str, [&](auto key, auto value) {
        bool consumed = false;
)cpp";
  for (const ComponentInfo *comp : comps) {
    os << formatv("        if (llvm::failed(parse{0}OptsOption(\n"
                  "                parsed.{1}, key, value, consumed)))\n"
                  "          return llvm::failure();\n",
                  comp->name, comp->invocationMember);
  }
  os << R"cpp(        return llvm::failure(!consumed);
      })))
    return llvm::failure();
  *this = std::move(parsed);
  return llvm::success();
}
)cpp";
}
static constexpr StringRef kInvocationHandle = "MlirTensorIRCompilerInvocation";
static constexpr StringRef kInvocationFn = "mlirTensorIRCompilerInvocation";
struct CAPIOption {
  const ComponentInfo &comp;
  const OptionInfo &opt;
  std::string type;
  std::string name;
  std::string member;
};
template <typename Fn>
static void forEachCAPIOption(ArrayRef<const ComponentInfo *> comps, Fn &&fn) {
  for (const ComponentInfo *comp : comps) {
    for (const OptionInfo &opt : comp->options) {
      if (!opt.emitTypedCAPI) {
        continue;
      }
      fn(CAPIOption{*comp, opt,
                    opt.kind == OptionInfo::Kind::Enum
                        ? comp->capiEnumType(opt)
                        : capiScalarType(opt.cppType)->str(),
                    convertToCamelFromSnakeCase(opt.varName,
                                                /*capitalizeFirst=*/true),
                    (Twine(comp->invocationMember) + "." + opt.varName).str()});
    }
  }
}
static void emitInvocationCAPIDecls(ArrayRef<const ComponentInfo *> comps,
                                    raw_ostream &os) {
  os << "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n";
  StringSet<> emittedEnumTypes;
  forEachCAPIOption(comps, [&](const CAPIOption &capi) {
    if (capi.opt.kind != OptionInfo::Kind::Enum ||
        !emittedEnumTypes.insert(capi.type).second) {
      return;
    }
    os << "typedef enum " << capi.type << " {\n";
    for (const EnumCase &c : capi.opt.enumInfo->getAllCases()) {
      os << "  " << capi.type << makeIdentifier(c.getSymbol()) << " = "
         << c.getValue() << ",\n";
    }
    os << "} " << capi.type << ";\n\n";
  });
  os << "typedef struct " << kInvocationHandle << " { void *ptr; } "
     << kInvocationHandle << ";\n\n"
     << "MLIR_CAPI_EXPORTED " << kInvocationHandle << " " << kInvocationFn
     << "Create(void);\n"
     << "MLIR_CAPI_EXPORTED void " << kInvocationFn << "Destroy("
     << kInvocationHandle << " invocation);\n"
     << "MLIR_CAPI_EXPORTED MlirLogicalResult " << kInvocationFn
     << "SetFromString(" << kInvocationHandle
     << " invocation, MlirStringRef str);\n"
     << "MLIR_CAPI_EXPORTED void " << kInvocationFn << "Print("
     << kInvocationHandle
     << " invocation, MlirStringCallback callback, void *userData);\n\n";
  forEachCAPIOption(comps, [&](const CAPIOption &capi) {
    os << "MLIR_CAPI_EXPORTED void " << kInvocationFn << "Set" << capi.name
       << "(" << kInvocationHandle << " invocation, " << capi.type
       << " value);\nMLIR_CAPI_EXPORTED " << capi.type << " " << kInvocationFn
       << "Get" << capi.name << "(" << kInvocationHandle << " invocation);\n";
  });
  os << "\n#ifdef __cplusplus\n} // extern \"C\"\n#endif\n";
}
static void emitCAPIAccessorImpl(const CAPIOption &capi, raw_ostream &os) {
  const OptionInfo &opt = capi.opt;
  if (opt.kind == OptionInfo::Kind::Enum) {
    for (const EnumCase &c : opt.enumInfo->getAllCases()) {
      os << "static_assert(static_cast<int>(" << kCppNamespace
         << "::" << opt.cppType << "::" << makeIdentifier(c.getSymbol())
         << ") == " << c.getValue() << ");\n";
    }
  }
  std::string cppType = opt.kind == OptionInfo::Kind::Enum
                            ? (Twine(kCppNamespace) + "::" + opt.cppType).str()
                            : opt.cppType.str();
  std::string castType = opt.kind == OptionInfo::Kind::Enum ? "int" : capi.type;
  FmtContext ctx;
  ctx.addSubst("cppType", cppType)
      .addSubst("capiType", capi.type)
      .addSubst("member", capi.member)
      .addSubst("castType", castType);
  auto format = [&](StringRef field) {
    StringRef text = getCodeTemplate(opt.rec, field);
    return tgfmt(text, &ctx).str();
  };
  os << formatv(R"(void {0}Set{1}({2} invocation, {3} value) {{
  if (auto *opts = unwrap(invocation))
    opts->{4} = {5};
}

{3} {0}Get{1}({2} invocation) {{
  auto *opts = unwrap(invocation);
{6}
}

)",
                kInvocationFn, capi.name, kInvocationHandle, capi.type,
                capi.member, format("capiSet"), format("capiGet"));
}
static void emitInvocationCAPIImpl(ArrayRef<const ComponentInfo *> comps,
                                   raw_ostream &os) {
  os << "#include \"mlir/CAPI/Support.h\"\n\n"
        "using TensorIRInvocation = "
        "::mlir::nv_tensor_ir::CompilerInvocation;\n"
        "DEFINE_C_API_PTR_METHODS("
     << kInvocationHandle << ", TensorIRInvocation)\n\n"
     << kInvocationHandle << " " << kInvocationFn << "Create(void) {\n"
     << "  return wrap(new TensorIRInvocation());\n"
     << "}\n\n"
     << "void " << kInvocationFn << "Destroy(" << kInvocationHandle
     << " invocation) { delete unwrap(invocation); }\n\n"
     << "MlirLogicalResult " << kInvocationFn << "SetFromString("
     << kInvocationHandle
     << " invocation, MlirStringRef str) {\n"
        "  auto *opts = unwrap(invocation);\n"
        "  if (!opts) return mlirLogicalResultFailure();\n"
        "  return wrap(opts->parseFromString(unwrap(str)));\n"
        "}\n\n"
     << "void " << kInvocationFn << "Print(" << kInvocationHandle
     << " invocation, MlirStringCallback callback, void *userData) {\n"
        "  auto *opts = unwrap(invocation);\n"
        "  if (!opts || !callback) return;\n"
        "  std::string value = opts->toString();\n"
        "  callback(wrap(value), userData);\n"
        "}\n\n";
  forEachCAPIOption(
      comps, [&](const CAPIOption &capi) { emitCAPIAccessorImpl(capi, os); });
}
static bool emitComponentOptions(const RecordKeeper &records, raw_ostream &os) {
  FailureOr<std::vector<ComponentInfo>> comps = buildAndValidateAll(records);
  if (failed(comps)) {
    return true;
  }
  FailureOr<SmallVector<const ComponentInfo *>> invocationComps =
      buildInvocationComponents(records, *comps);
  if (failed(invocationComps)) {
    return true;
  }

  emitSourceFileHeader("TensorIR Component Options", os, records);
  {
    IfDefEmitter guard(os, "GEN_OPTIONS_DECLS", /*LateUndef=*/true);
    NamespaceEmitter ns(os, kCppNamespace);
    for (const ComponentInfo &comp : *comps) {
      ComponentEmitter emitter(comp, os);
      emitter.emitOptionsStruct();
      os << "\n";
      os << formatv("void {0}();\nllvm::cl::OptionCategory &{1}();\n"
                    "void apply{2}OptionsFromCL({3} &opts);\n{3} {4}();\n",
                    comp.clRegisterFnName(), comp.clCategoryFnName(), comp.name,
                    comp.structName(), comp.clGetterFnName());
    }
    emitInvocationComponents(*invocationComps, os);
  }
  {
    IfDefEmitter guard(os, "GEN_OPTIONS_PARSER", /*LateUndef=*/true);
    os << "namespace optd = ::mlir::nv_tensor_ir::opt_detail;\n\n";
    for (const ComponentInfo &comp : *comps) {
      ComponentEmitter emitter(comp, os);
      emitter.emitParserDefs();
      emitter.emitValidateImpl();
    }
    emitInvocationParserDef(*invocationComps, os);
  }
  {
    IfDefEmitter guard(os, "GEN_OPTIONS_CL_BINDINGS", /*LateUndef=*/true);
    os << "namespace optd = ::mlir::nv_tensor_ir::opt_detail;\n\n";
    for (const ComponentInfo &comp : *comps) {
      ComponentEmitter emitter(comp, os);
      emitter.emitCLBindings();
    }
  }
  {
    IfDefEmitter guard(os, "GEN_COMPILERINVOCATION_CAPI_DECLS",
                       /*LateUndef=*/true);
    emitInvocationCAPIDecls(*invocationComps, os);
  }
  {
    IfDefEmitter guard(os, "GEN_COMPILERINVOCATION_CAPI_IMPL",
                       /*LateUndef=*/true);
    emitInvocationCAPIImpl(*invocationComps, os);
  }
  return false;
}

static mlir::GenRegistration genComponentOptions(
    "gen-component-options",
    "Generate component option structs, parsers and llvm::cl bindings",
    emitComponentOptions);
