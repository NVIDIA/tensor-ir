// RUN: tensor_ir-opt -tile-analyzer \
// RUN:   -verify-diagnostics=only-expected %s
// RUN: tensor_ir-opt "-tile-selection=tile-size=8" \
// RUN:   -verify-diagnostics=only-expected %s

// Guard the result-view rank before fixed-dimension validation indexes the
// selected tile. Normally normalization produces consistent metadata; this
// checks the verifier boundary for manually authored or stale attributes.
// expected-error @+1 {{result_views rank 2 does not match iteration space rank 1}}
nv_tensor_ir.graph @invalid_result_view_rank(
    %arg0: tensor<8xf32>) -> (tensor<8xf32>) {
  %out = abs %arg0 : tensor<8xf32>
  results %out attributes {
    iteration_space = #nv_tensor_ir.tensor_source<0, 0, "(8):(1)">,
    result_layouts = [#nv_tensor_ir.tensor_source<0, 0, "(8,2):(1,0)">],
    result_views = [#nv_tensor_ir.tensor_source<1, 0, "(8,2):(1,0)">]
  } : tensor<8xf32>
}
