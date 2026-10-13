builtin.module attributes {prov.weights_file = "/home/grammy/blast/model2MLIR/workloads/mlp_basic_triton/mlp_basic_triton.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x16xf32>, %1: tensor<32xf32>, %2: tensor<32x32xf32>, %3: tensor<32xf32>, %4: tensor<8x32xf32>, %5: tensor<8xf32>, %6: tensor<4x16xf32>) -> tensor<4x8xf32> {
    %7 = func.call @triton_exp__fused_mlp2(%6, %0, %1, %2, %3) {prov.kernel = "triton_exp::fused_mlp2", prov.kernel_backend = "triton", prov.kernel_operands = ["x", "w1", "b1", "w2", "b2"], prov.kernel_args = {}, prov.region_id = "triton_exp__fused_mlp2_0", prov._pattern_hint = "kernel", prov.op = "kernel", prov.family = "kernel", prov.aten = "triton_exp.fused_mlp2.default", prov.orig_dtype = "float32", prov.dispatch_id = "triton_exp__fused_mlp2_0"} : (tensor<4x16xf32>, tensor<32x16xf32>, tensor<32xf32>, tensor<32x32xf32>, tensor<32xf32>) -> tensor<4x32xf32>
    %8 = tensor.empty() : tensor<32x8xf32>
    %9 = linalg.transpose ins(%4:tensor<8x32xf32>) outs(%8:tensor<32x8xf32>) permutation = [1, 0]
    %10 = tensor.empty() : tensor<4x8xf32>
    %11 = arith.constant {prov.module = "fc3"} 0.000000e+00 : f32
    %12 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "fc3"} ins(%11 : f32) outs(%10 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %13 = linalg.matmul {prov.region_id = "matmul_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc3", prov.fqn = "fc3", prov.transposed_b = "true"} ins(%7, %9 : tensor<4x32xf32>, tensor<32x8xf32>) outs(%12 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %14 = tensor.empty() : tensor<4x8xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%13, %5 : tensor<4x8xf32>, tensor<8xf32>) outs(%14 : tensor<4x8xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc3", prov.fqn = "fc3"} {
    ^bb0(%16: f32, %17: f32, %18: f32):
      %19 = arith.addf %16, %17 : f32
      linalg.yield %19 : f32
    } -> tensor<4x8xf32>
    func.return %15 : tensor<4x8xf32>
  }
  func.func private @triton_exp__fused_mlp2(tensor<4x16xf32>, tensor<32x16xf32>, tensor<32xf32>, tensor<32x32xf32>, tensor<32xf32>) -> tensor<4x32xf32> attributes {prov.kernel = "triton_exp::fused_mlp2", prov.kernel_backend = "triton", prov.triton_jit = ["kernels._mlp2_kernel"], prov.kernel_source = "/home/grammy/blast/model2MLIR/workloads/mlp_basic_triton/kernels.py"}
}
