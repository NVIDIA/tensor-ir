// RUN: tensor_ir-opt -convert-tensor-to-cuda-tile="codegen-strategy=affine_map" -split-input-file %s | FileCheck %s

// ============================================================================
// Test 1: Mixed-rank tensors trigger getMaxRank().
// The graph has a 3D used input and a 2D unused input with manually-set
// iteration space maps. hasUniformTensors() passes because only the 3D ops
// are walked (the 2D argument is unused). hasMixedRanks() detects the
// different argument ranks and calls getMaxRank() to determine conservative
// tile sizes. Tile sizes are (2,2,2) for maxRank=3.
// ============================================================================

// CHECK-LABEL: entry @mixed_rank_getmaxrank
// CHECK-SAME:  (%[[A:.+]]: tile<ptr<f32>>, %[[B:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
// CHECK:       %[[VA:.+]] = make_tensor_view %[[A]]{{.*}} shape = [64, 128, 32]
// CHECK:       %[[VB:.+]] = make_tensor_view %[[B]]{{.*}} shape = [64, 128]
// CHECK:       %[[VOUT:.+]] = make_tensor_view %[[OUT]]{{.*}} shape = [64, 128, 32]
// CHECK:       %[[RELU:.+]] = maxf {{.*}} propagate_nan
// CHECK:       store_view_tko weak %[[RELU]]
// CHECK:       return
module {
  nv_tensor_ir.graph @mixed_rank_getmaxrank(
    %a: tensor<64x128x32xf32> {
        nv_tensor_ir.stride = "(4096,32,1)",
        nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    },
    %b: tensor<64x128xf32> {
        nv_tensor_ir.stride = "(1,64)",
        nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1)>
    }
  ) -> (tensor<64x128x32xf32> {nv_tensor_ir.stride = "(4096,32,1)"}) {
    %relu = relu_fwd %a {nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>} : tensor<64x128x32xf32>
    results %relu : tensor<64x128x32xf32>
  }
}

// -----

// ============================================================================
// Test 2: Mixed-rank with 2D input and 3D input, 3D output.
// Only the 3D input is used, 2D input is unused. getMaxRank() returns 3.
// Tile sizes are conservative (2,2,2) because of mixed ranks.
// ============================================================================

// CHECK-LABEL: entry @mixed_rank_unused_2d
// CHECK-SAME:  (%[[A:.+]]: tile<ptr<f32>>, %[[B:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
// CHECK:       %[[VA:.+]] = make_tensor_view %[[A]]{{.*}} shape = [32, 64, 16]
// CHECK:       %[[VB:.+]] = make_tensor_view %[[B]]{{.*}} shape = [32, 64]
// CHECK:       %[[VOUT:.+]] = make_tensor_view %[[OUT]]{{.*}} shape = [32, 64, 16]
// CHECK:       %[[RELU:.+]] = maxf {{.*}} propagate_nan
// CHECK:       store_view_tko weak %[[RELU]]
// CHECK:       return
module {
  nv_tensor_ir.graph @mixed_rank_unused_2d(
    %a: tensor<32x64x16xf32> {
        nv_tensor_ir.stride = "(1024,16,1)",
        nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    },
    %b: tensor<32x64xf32> {
        nv_tensor_ir.stride = "(1,32)",
        nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1)>
    }
  ) -> (tensor<32x64x16xf32> {nv_tensor_ir.stride = "(1024,16,1)"}) {
    %relu = relu_fwd %a {nv_tensor_ir.iter_space_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>} : tensor<32x64x16xf32>
    results %relu : tensor<32x64x16xf32>
  }
}
