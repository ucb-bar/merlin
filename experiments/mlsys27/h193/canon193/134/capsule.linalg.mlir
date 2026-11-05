"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>}> ({
  ^bb0(%0: tensor<2x3x4xf32>):
    %1 = "tensor.empty"() {prov._pattern_hint = "permute", prov.op = "permute", prov.family = "layout", prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering", prov.aten = "aten.permute.default", prov.orig_dtype = "float32"} : () -> tensor<4x2x3xf32>
    %2 = "linalg.transpose"(%0, %1) <{permutation = array<i64: 2, 0, 1>}> ({
    ^bb1(%3: f32, %4: f32):
      "linalg.yield"(%3) {prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering"} : (f32) -> ()
    }) {prov.region_id = "permute_0", prov._pattern_hint = "permute", prov.op = "permute", prov.family = "layout", prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering", prov.aten = "aten.permute.default", prov.orig_dtype = "float32", prov.dispatch_id = "permute_0"} : (tensor<2x3x4xf32>, tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
    "func.return"(%2) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
}) {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22float32\22, \22requires_grad\22: false, \22shape\22: [2, 3, 4], \22storage_offset\22: 0, \22stride\22: [12, 4, 1]}], \22results\22: [{\22alias_inputs\22: [0], \22alias_results\22: [], \22dtype\22: \22float32\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [4, 2, 3], \22storage_offset\22: 0, \22stride\22: [1, 12, 4]}], \22schema_version\22: 1}", prov.weights_file = "/scratch/loren/backend-verify/out-c1/captures/case-d7f32667aacef3e14b1cd1d65c372182ab6fe8316c9ae9ef26441044ef760900/capture-work/weights.safetensors", prov.level = "linalg-on-tensors"} : () -> ()