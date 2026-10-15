// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="target-bytecode-version=invalid" -verify-diagnostics

// expected-error @below {{target-bytecode-version: bytecode-version: unsupported CUDA Tile bytecode version (got 'invalid')}}
module {}
