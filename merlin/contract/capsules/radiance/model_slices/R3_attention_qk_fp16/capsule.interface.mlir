module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %Q = merlin_iface.tensor {name = "Q", role = "input"} : tensor<16x32xf16>
  %K = merlin_iface.tensor {name = "K", role = "input"} : tensor<16x32xf16>
  %Y0 = merlin_iface.attention_qk %Q, %K {name = "Y0", output_dtype = "f32"} : (tensor<16x32xf16>, tensor<16x32xf16>) -> tensor<16x16xf32>
}
