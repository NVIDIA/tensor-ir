set(TENSOR_IR_TEST_ANALYSIS_SOURCES
  TestControlFlowInterface.cpp
  TestLayoutPropagation.cpp
  TestTileAnalyzer.cpp
  TestTilingEnumerator.cpp
  TestTilingEvaluator.cpp
)


add_tensor_ir_test_library(NVTensorIRTestAnalysis
  ${TENSOR_IR_TEST_ANALYSIS_SOURCES}

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRPass
  NVTensorIR::Analysis
  NVTensorIR::TilingAnalysis
  NVTensorIR::Utils
)
