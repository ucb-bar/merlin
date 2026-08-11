module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %Q = merlin_iface.tensor {name = "Q", role = "input"} : tensor<32x32xf4E2M1FN>
  %K = merlin_iface.tensor {name = "K", role = "input"} : tensor<32x32xf4E2M1FN>
  %V = merlin_iface.tensor {name = "V", role = "input"} : tensor<32x32xf4E2M1FN>
  %S = merlin_iface.attention_qk %Q, %K {name = "S", block_scale = "e8m0", output_dtype = "bf16"} : (tensor<32x32xf4E2M1FN>, tensor<32x32xf4E2M1FN>) -> tensor<32x32xbf16>
  %P = merlin_iface.softmax %S {name = "P", axis = 1 : i64, scale = 0.17677669529663687 : f32} : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %Y0 = merlin_iface.attention_pv %P, %V {name = "Y0", block_scale = "e8m0", output_dtype = "bf16"} : (tensor<32x32xbf16>, tensor<32x32xf4E2M1FN>) -> tensor<32x32xbf16>
}
