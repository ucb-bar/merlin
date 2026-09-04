module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<32x31xf8E4M3FN>
  %B = merlin_iface.tensor {name = "B", role = "bias"} : tensor<31xf8E4M3FN>
  %Y0 = merlin_iface.bias_add %X, %B {name = "Y0", output_dtype = "fp8_e4m3"} : (tensor<32x31xf8E4M3FN>, tensor<31xf8E4M3FN>) -> tensor<32x31xf8E4M3FN>
}
