module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %Q = merlin_iface.tensor {name = "Q", role = "input"} : tensor<16x32xf4E2M1FN>
  %K = merlin_iface.tensor {name = "K", role = "input"} : tensor<32x32xf4E2M1FN>
  %V = merlin_iface.tensor {name = "V", role = "input"} : tensor<32x16xf4E2M1FN>
  %Q_scale = merlin_iface.tensor {name = "Q_scale", role = "scale", scale_of = "Q", block = 32 : i64} : tensor<1x16xi8>
  %K_scale = merlin_iface.tensor {name = "K_scale", role = "scale", scale_of = "K", block = 32 : i64} : tensor<1x32xi8>
  %V_scale = merlin_iface.tensor {name = "V_scale", role = "scale", scale_of = "V", block = 32 : i64} : tensor<1x16xi8>
  %P_scale = merlin_iface.tensor {name = "P_scale", role = "scale", scale_of = "P", block = 32 : i64} : tensor<1x16xi8>
  %S = merlin_iface.attention_qk %Q, %K {name = "S", block_scale = "e8m0", output_dtype = "bf16"} : (tensor<16x32xf4E2M1FN>, tensor<32x32xf4E2M1FN>) -> tensor<16x32xbf16>
  %P = merlin_iface.softmax %S {name = "P", axis = 1 : i64, scale = 0.17677669529663687 : f32} : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
  %Y0 = merlin_iface.attention_pv %P, %V {name = "Y0", block_scale = "e8m0", output_dtype = "bf16"} : (tensor<16x32xbf16>, tensor<32x16xf4E2M1FN>) -> tensor<16x16xbf16>
}
