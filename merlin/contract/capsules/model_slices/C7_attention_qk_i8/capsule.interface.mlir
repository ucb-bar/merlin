module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Q = merlin_iface.tensor {name = "Q", role = "input"} : tensor<16x32xi8>
  %K = merlin_iface.tensor {name = "K", role = "input"} : tensor<16x32xi8>
  %Y0 = merlin_iface.attention_qk %Q, %K {name = "Y0", output_dtype = "i32"} : (tensor<16x32xi8>, tensor<16x32xi8>) -> tensor<16x16xi32>
}
