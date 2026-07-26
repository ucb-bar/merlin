module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %Q = merlin_iface.tensor {name = "Q", role = "input"} : tensor<32x32xf8E4M3FN>
  %K = merlin_iface.tensor {name = "K", role = "input"} : tensor<32x32xf8E4M3FN>
  %Y0 = merlin_iface.attention_qk %Q, %K {name = "Y0", output_dtype = "bf16"} : (tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) -> tensor<32x32xbf16>
}
