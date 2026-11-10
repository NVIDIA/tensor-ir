// RUN: tensor_ir-tblgen %s --gen-pipeline-options \
// RUN:   -I %S/../../../include %mlir_include_dirs | FileCheck %s
// RUN: not tensor_ir-tblgen %s --gen-pipeline-options -DUNKNOWN \
// RUN:   -I %S/../../../include %mlir_include_dirs 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not tensor_ir-tblgen %s --gen-pipeline-options -DDUPLICATE \
// RUN:   -I %S/../../../include %mlir_include_dirs 2>&1 | FileCheck %s --check-prefix=DUPLICATE

include "tensor_ir/Options/Options.td"

def Extra : ComponentOptions<"Extra"> {
  let invocationMember = "extra";
  let options = [StringOption<"name", "name", "Name with \"quotes\"", "\"default\"">,
                 ShapeOption<"shape", "shape", "Dynamic shape">,
                 ListOption<"extraTiles", "extra-tiles", "int32_t", "Extra tiles">];
}
def ExtraPipelineOptions : PipelineOptions<[
    PipelineOption<Extra, "name">, PipelineOption<Extra, "shape">,
    PipelineOption<Extra, "extraTiles">, PipelineOption<CodegenOptions, "ctaCount", "8">,
    PipelineOption<CodegenOptions, "tileSize", "{128, 128, 64}">
]>;
#ifdef UNKNOWN
def InvalidPipeline : PipelineOptions<[PipelineOption<CodegenOptions, "missing">]>;
#endif
#ifdef DUPLICATE
def InvalidPipeline : PipelineOptions<[
    PipelineOption<CodegenOptions, "ctaCount">, PipelineOption<CodegenOptions, "ctaCount">
]>;
#endif

// CHECK-LABEL: struct ExtraPipelineOptions : public ::mlir::PassPipelineOptions<ExtraPipelineOptions>
// CHECK: llvm::cl::desc("Name with \"quotes\"")
// CHECK: llvm::cl::init("default")
// CHECK: llvm::cl::init(8)
// CHECK: options.extra.name = name.getValue()
// CHECK: optd::parseShapeExtent
// CHECK: llvm::StringRef value = extraTiles.getValue()
// CHECK: options.codegen.ctaCount = ctaCount.getValue()
// CHECK: options.codegen.tileSize = {128, 128, 64}
// CHECK: if (tileSize.getNumOccurrences())
// CHECK: value.contains('x') && value.contains(',')
// CHECK: optd::kShapeSeparators
// CHECK: invalid value for 'tile-size'
// CHECK-LABEL: struct LayoutPropagationPipelineOptions
// CHECK: PassOptions::Option<int64_t> reductionTileSize
// CHECK: llvm::cl::init(128)
// CHECK: PersistenceMode::None
// CHECK: clEnumValN({{.*}}PersistenceMode::Static, "static"
// UNKNOWN: error: {{.*}}missing
// DUPLICATE: error: {{.*}}ctaCount
