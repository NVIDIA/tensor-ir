// RUN: tensor_ir-tblgen %s --gen-component-options \
// RUN:   -I %S/../../../include %mlir_include_dirs \
// RUN:   | FileCheck %s --check-prefix=CORE

include "tensor_ir/Options/OptionsBase.td"

class MultipleOf<int divisor>
    : OptionTrait<CPred<"($_self % " # divisor # ") == 0">,
                  "must be a multiple of " # divisor>;
class TestEnumCase<string sym, int val, string str = sym>
    : I32EnumCase<sym, val, str> {
  string help = "";
}
def ModeEnum : I32Enum<"Mode", "Compilation mode",
                       [TestEnumCase<"Auto", 0, "auto">,
                        TestEnumCase<"Fast", 1, "fast">,
                        TestEnumCase<"2D", 2, "2d">]> {
  let cppNamespace = "::mlir::nv_tensor_ir";
  let underlyingType = "int";
}

def Mode : EnumOption<"mode", "mode", ModeEnum, "Auto", "Compilation mode"> {
  let capiType = "MlirTensorIRCustomMode";
}
def FallbackMode
    : EnumOption<"fallback_mode", "fallback-mode", ModeEnum, "2D",
                 "Fallback compilation mode">;
def FrontendOptions : ComponentOptions<"Frontend"> {
  let clCategory = "Frontend options";
  let invocationMember = "frontend";
  let verifyFunction = "::verifyFrontend";
  let options = [
    Option<"optLevel", "opt-level", "unsigned", "2", "Optimization level",
           /*emitTypedCAPI=*/1, [RangeTrait<0, 3>]>,
    StringOption<"inputDialect", "input-dialect", "Input dialect",
                 "\"tensor\"">,
    Mode,
    FallbackMode,
    TileShapeOption<"tileShape", "tile-shape", "int32_t", "Static tile shape">,
    ShapeOption<"problemShape", "problem-shape", "Problem shape", "x">,
    Option<"parserBudget", "parser-budget", "unsigned", "16",
           "Parser recovery budget", /*emitTypedCAPI=*/0,
           [PositiveTrait<>, PowerOfTwoTrait<>, MultipleOf<4>]>
  ];
}
def BackendOptions : ComponentOptions<"Backend"> {
  let invocationMember = "backend";
  let options = [
    Option<"enabled", "backend-enabled", "bool", "false", "Enable backend">
  ];
}
def CompilerInvocation
    : CompilerInvocationOptions<[FrontendOptions, BackendOptions]>;

// CORE:      TableGen'erated file
// CORE:      TensorIR Component Options

// CORE:      #ifdef GEN_OPTIONS_DECLS
// CORE:      struct BackendOptions {
// CORE:        bool enabled = false;
// CORE:      struct FrontendOptions {
// CORE:        unsigned optLevel = 2;
// CORE:        std::string inputDialect = "tensor";
// CORE:        Mode mode = Mode::Auto;
// CORE:        Mode fallback_mode = Mode::_2D;
// CORE:        std::vector<int32_t> tileShape = {};
// CORE:        std::vector<int64_t> problemShape = {};
// CORE:        unsigned parserBudget = 16;
// CORE:      struct CompilerInvocationComponents {
// CORE:        FrontendOptions frontend;
// CORE-NEXT:   BackendOptions backend;
// CORE:        llvm::LogicalResult parseFromString(llvm::StringRef str);
// CORE:        void print(llvm::raw_ostream &os) const {
// CORE:        std::string toString() const {
// CORE:        std::optional<std::string> validateComponents() const
// CORE-NOT:    FrontendOptions::parseFromString
// CORE-NOT:    FrontendOptions::absorbOption
// CORE-NOT:    FrontendOptions::asTuple
// CORE:      #undef GEN_OPTIONS_DECLS
// CORE-NEXT: #endif // GEN_OPTIONS_DECLS

// CORE:      #ifdef GEN_OPTIONS_PARSER
// CORE:      namespace optd = ::mlir::nv_tensor_ir::opt_detail;
// CORE:      static llvm::LogicalResult parseBackendOptsOption(
// CORE-NOT:  namespace optd = ::mlir::nv_tensor_ir::opt_detail;
// CORE:      static llvm::LogicalResult parseFrontendOptsOption(
// CORE:        if (key == "opt-level")
// CORE-NEXT:     consumed = true;
// CORE-NEXT:     return optd::parseValue(value, opts.optLevel);
// CORE:        if (key == "mode")
// CORE:        symbolizeMode(value)
// CORE:      std::optional<std::string>
// CORE:      FrontendOptions::validate() const
// CORE:        if (auto status = ::verifyFrontend(*this); !status.ok())
// CORE-NEXT:     return status.message();
// CORE:      CompilerInvocationComponents::parseFromString(
// CORE:        CompilerInvocationComponents parsed = *this;
// CORE:        parseFrontendOptsOption(
// CORE:        return llvm::failure(!consumed);
// CORE:      #undef GEN_OPTIONS_PARSER
// CORE-NEXT: #endif // GEN_OPTIONS_PARSER

// CORE:      #ifdef GEN_OPTIONS_CL_BINDINGS
// CORE:      namespace optd = ::mlir::nv_tensor_ir::opt_detail;
// CORE:      struct BackendCLOptions {
// CORE-NOT:  namespace optd = ::mlir::nv_tensor_ir::opt_detail;
// CORE:      struct FrontendCLOptions {
// CORE:      llvm::cl::OptionCategory category{"Frontend options"};
// CORE:      llvm::cl::opt<unsigned> optLevel{"opt-level"
// CORE:      llvm::cl::list<std::string> tileShape{"tile-shape"
// CORE:      llvm::cl::opt<std::string> problemShape{"problem-shape"
// CORE:      void mlir::nv_tensor_ir::applyFrontendOptionsFromCL(
// CORE:      optd::parseListOrFatal(opts.tileShape, clFrontendOptions->tileShape, 'x',
// CORE:      optd::parseShapeOrFatal(opts.problemShape, clFrontendOptions->problemShape.getValue(),
// CORE:      FrontendOptions mlir::nv_tensor_ir::getFrontendOptionsFromCL()
// CORE:      applyFrontendOptionsFromCL(opts);
// CORE:      #undef GEN_OPTIONS_CL_BINDINGS
// CORE-NEXT: #endif // GEN_OPTIONS_CL_BINDINGS

// CORE:      #ifdef GEN_COMPILERINVOCATION_CAPI_DECLS
// CORE:      typedef enum MlirTensorIRCustomMode {
// CORE-NEXT: MlirTensorIRCustomModeAuto = 0,
// CORE-NEXT: MlirTensorIRCustomModeFast = 1,
// CORE-NEXT: MlirTensorIRCustomMode_2D = 2,
// CORE-NOT:  typedef enum MlirTensorIRCustomMode
// CORE:      typedef struct MlirTensorIRCompilerInvocation { void *ptr; } MlirTensorIRCompilerInvocation;
// CORE:      mlirTensorIRCompilerInvocationCreate(void);
// CORE:      mlirTensorIRCompilerInvocationSetFromString(
// CORE:      mlirTensorIRCompilerInvocationSetOptLevel(
// CORE:      mlirTensorIRCompilerInvocationSetInputDialect(
// CORE:      mlirTensorIRCompilerInvocationSetMode(
// CORE:      mlirTensorIRCompilerInvocationSetFallbackMode(
// CORE:      mlirTensorIRCompilerInvocationSetEnabled(
// CORE:      #undef GEN_COMPILERINVOCATION_CAPI_DECLS
// CORE-NEXT: #endif // GEN_COMPILERINVOCATION_CAPI_DECLS

// CORE:      #ifdef GEN_COMPILERINVOCATION_CAPI_IMPL
// CORE:      #include "mlir/CAPI/Support.h"
// CORE:      DEFINE_C_API_PTR_METHODS(MlirTensorIRCompilerInvocation, TensorIRInvocation)
// CORE:      return wrap(new TensorIRInvocation());
// CORE:      delete unwrap(invocation);
// CORE:      return wrap(opts->parseFromString(unwrap(str)));
// CORE:      callback(wrap(value), userData);
// CORE:      opts->frontend.optLevel
// CORE:      #undef GEN_COMPILERINVOCATION_CAPI_IMPL
// CORE-NEXT: #endif // GEN_COMPILERINVOCATION_CAPI_IMPL
