
add_tensor_ir_ut(TensorIRPolyExprsTests
  SRCS PolyExprsTests.cpp
  LINK_LIBS
  NVTensorIR::PolyExprs
)

add_tensor_ir_ut(TensorIRSymbolicLayoutTests
  SRCS SymbolicLayoutTests.cpp
  LINK_LIBS
  NVTensorIR::Analysis
)
