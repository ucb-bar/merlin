module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<32x32xbf16>
  %B = merlin_iface.tensor {name = "B", role = "bias"} : tensor<32xbf16>
  %Y0 = merlin_iface.bias_add %X, %B {name = "Y0", output_dtype = "bf16"} : (tensor<32x32xbf16>, tensor<32xbf16>) -> tensor<32x32xbf16>
}
