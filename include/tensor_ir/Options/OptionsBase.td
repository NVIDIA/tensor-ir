// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TableGen schema for component options and their generated API adapters.

include "mlir/IR/Constraints.td"
include "mlir/IR/EnumAttr.td"

#ifndef TENSORIR_OPTIONS_OPTIONS_BASE
#define TENSORIR_OPTIONS_OPTIONS_BASE

/// Built-in option validation traits.
class OptionTrait<Pred predicate, string summary>
    : Constraint<predicate, summary>;

class PositiveTrait
    : OptionTrait<CPred<"$_self > 0">, "must be > 0">;

class PowerOfTwoTrait
    : OptionTrait<CPred<"$_self > 0 && ($_self & ($_self - 1)) == 0">,
                  "must be a positive power of two">;

class RangeTrait<int minVal, int maxVal>
    : OptionTrait<CPred<"$_self >= " # minVal # " && $_self <= " # maxVal>,
                  "must be in [" # minVal # ", " # maxVal # "]">;

/// A scalar option and its generated surface metadata.
class Option<string cppName, string flag, string type, string default,
             string helpStr, bit _emitTypedCAPI = 1,
             list<OptionTrait> optionTraits = []> {
  string varName = cppName;
  string argName = flag;
  string cppType = type;
  string defaultValue = default;
  string helpText = helpStr;
  /// Whether the C API has a typed accessor rather than only SetFromString.
  bit emitTypedCAPI = _emitTypedCAPI;
  list<OptionTrait> traits = optionTraits;

  /// The `code`/`string` fields below are C++ templates rendered by
  /// ComponentOptionsGen.cpp via `mlir::tblgen::tgfmt`, using named
  /// placeholders (`$name`, MLIR's usual convention):
  ///   $varName      - the option's C++ struct field name
  ///   $argName      - the CLI flag name (without leading "--")
  ///   $cppType      - the option's C++ value type
  ///   $helpText     - the CLI help string
  ///   $defaultValue - the default value expression
  ///   $elemType     - (list options) the element C++ type
  ///   $delim        - (list options) the raw separator, as written in .td
  ///   $delimChar    - (list options) the separator as a C++ char literal
  ///   $member       - (print/C API only) fully-qualified member access expr,
  ///                   distinct from $varName because it may be prefixed by
  ///                   a component/receiver (e.g. "opts->frontend.tileShape")
  ///   $clStatic     - the component's ManagedStatic<...CLOptions> pointer
  ///   $capiType     - the C API type for this option (e.g. MlirStringRef)
  ///   $castType     - intermediate cast type bridging C++ <-> C API types
  code parse = [{
    return optd::parseValue(value, opts.$varName);
  }];
  code print = [{
  ::mlir::nv_tensor_ir::opt_detail::printScalar(os, $member);
  }];
  code capiSet = [{static_cast<$cppType>(static_cast<$castType>(value))}];
  code capiGet = [{
  return opts ? static_cast<$capiType>(static_cast<$castType>(opts->$member))
              : $capiType{};
  }];

  string clAssignment = "    opts.$varName = $clStatic->$varName;\n";
  code clDeclaration = [{
  llvm::cl::opt<$cppType> $varName{"$argName", llvm::cl::desc("$helpText"),
      llvm::cl::init($defaultValue), llvm::cl::cat(category)};
  }];
}

class FlagOption<string cppName, string flag, string helpStr>
    : Option<cppName, flag, "bool", "false", helpStr>;

class StringOption<string cppName, string flag, string helpStr,
                   string default = "\"\"", bit emitTypedCAPI = 1>
    : Option<cppName, flag, "std::string", default, helpStr, emitTypedCAPI> {
  let capiSet = [{
    std::string(value.data, value.length)
  }];
  let capiGet = [{
  if (!opts)
    return MlirStringRef{nullptr, 0};
  const std::string &value = opts->$member;
  return MlirStringRef{value.data(), value.size()};
  }];
}

/// An enum option whose default is an unqualified case symbol.
class EnumOption<string cppName, string flag, EnumInfo enumDef,
                 string default, string helpStr>
    : Option<cppName, flag, enumDef.className, default, helpStr> {
  EnumInfo enumInfo = enumDef;
}

/// A numeric vector parsed using `,` or `x` and exposed through strings.
class ListOption<string cppName, string flag, string elemType, string helpStr,
                 string delim = ",">
    : Option<cppName, flag, "std::vector<" # elemType # ">", "{}", helpStr> {
  string elementType = elemType;
  string delimiter = delim;
  let emitTypedCAPI = 0;

  let parse = [{
    return optd::parseElements(
        opts.$varName, value, llvm::StringRef("$delim"),
        [](llvm::StringRef text, $elemType &element) {
          return optd::parseValue(text, element);
        });
  }];
  let print = [{
  ::mlir::nv_tensor_ir::opt_detail::printElements(os, $member, $delimChar,
                      [](llvm::raw_ostream &stream, const $elemType &value) {
                        ::mlir::nv_tensor_ir::opt_detail::printScalar(stream, value);
                      });
  }];

  let clAssignment = !if(!eq(delim, ","),
                         "    opts.$varName.assign($clStatic->$varName.begin(), "
                         # "$clStatic->$varName.end());\n",
                         "    optd::parseListOrFatal(opts.$varName, $clStatic->$varName, "
                         # "$delimChar, \"$argName\");\n");
  let clDeclaration = !if(!eq(delim, ","), [{
  llvm::cl::list<$elemType> $varName{"$argName", llvm::cl::desc("$helpText"),
      llvm::cl::CommaSeparated, llvm::cl::cat(category)};
  }], [{
  llvm::cl::list<std::string> $varName{"$argName", llvm::cl::desc("$helpText"),
      llvm::cl::value_desc("v1}] # delim # [{v2}] # delim # [{..."), llvm::cl::cat(category)};
  }]);
}

/// A static numeric list conventionally spelled `128x128x64`.
class TileShapeOption<string cppName, string flag, string elemType,
                      string helpStr>
    : ListOption<cppName, flag, elemType, helpStr, "x">;

/// An int64 shape accepting decimal extents or `?` separated by `,` or `x`.
class ShapeOption<string cppName, string flag, string helpStr,
                  string delim = ",">
    : ListOption<cppName, flag, "int64_t", helpStr, delim> {
  let parse = [{
    return optd::parseElements(opts.$varName, value,
        optd::kShapeSeparators, optd::parseShapeExtent);
  }];
  let print = [{
  ::mlir::nv_tensor_ir::opt_detail::printShape(os, $member, $delimChar);
  }];

  let clAssignment =
      "    optd::parseShapeOrFatal(opts.$varName, "#"$clStatic->$varName."
                                                    "getValue(), "
                                                    "\"$argName\");\n";
  // See ListOption::clDeclaration for why `delim` is spliced via `#`
  // instead of a `$delim` placeholder here.
  let clDeclaration =
      "  llvm::cl::opt<std::string> $varName{\"$argName\", "
      # "llvm::cl::desc(\"$helpText\"),\n"
      # "      llvm::cl::value_desc(\"d1" # delim # "d2" # delim
      # "... ('?' = dynamic)\"), "
      # "llvm::cl::cat(category)};\n";
}

/// A component whose record generates one canonical struct and its adapters.
class ComponentOptions<string name> {
  string componentName = name;
  string clCategory = "";
  // Empty for tool-local components omitted from CompilerInvocation.
  string invocationMember = "";
  list<Option> options = [];
  /// Optional qualified cross-field validator returning an error string.
  string verifyFunction = "";
}

/// Components stored in CompilerInvocation, each exactly once, in stable order.
class CompilerInvocationOptions<list<ComponentOptions> comps> {
  list<ComponentOptions> components = comps;
}

#endif // TENSORIR_OPTIONS_OPTIONS_BASE
