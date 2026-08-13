set(TENSOR_IR_TEST_ANALYSIS_SOURCES
  TestTileAnalyzer.cpp
)


add_tensor_ir_test_library(NVTensorIRTestAnalysis
  ${TENSOR_IR_TEST_ANALYSIS_SOURCES}

  LINK_LIBS PUBLIC
  MLIRPass
  NVTensorIR::Analysis
)
